"""
Candidate generation (blocking), v2.

Idea: turn every record's name+address into a TF-IDF vector of words, then find
similar S2/S3 records *in the same country* in two directions:

  forward: for each S1 record, its top-k S2/S3 records            (as in v1)
  reverse: for each S2/S3 record, its top-rev_k S1 records        (new in v2)

Why reverse: every S2/S3 belongs to at most one S1. On the full-size data, a
true match can be pushed out of an S1's top 20 by crowds of similar records,
but it is almost always in the top 3 S1s *from the S2/S3 record's side*.

Why word pairs ("peacock_road", "bailey_ameren"): very common words are dropped
with max_df for speed, and sometimes a pair shared ONLY common words. A pair of
two common words next to each other is usually rare, so it survives max_df.

Returns integer positions (not string IDs) to keep memory small.
"""
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _topk_per_row(M, k):
    """Rows/cols/scores/ranks of the k largest entries in every row of sparse matrix M."""
    M = M.tocoo()
    if M.nnz == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32), e
    order = np.lexsort((-M.data, M.row))          # by row, then score descending
    r, c, d = M.row[order], M.col[order], M.data[order]
    starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    rank = np.arange(len(r)) - np.repeat(starts, np.diff(np.r_[starts, len(r)]))
    keep = rank < k
    return r[keep], c[keep], d[keep], rank[keep]


def _with_pairs(block_text, name, addr):
    """block_text plus neighbouring-word pairs from the name and from the address."""
    out = []
    for bt, n, a in zip(block_text, name, addr):
        extra = []
        for s in (n, a):
            w = s.split()
            extra += [x + "_" + y for x, y in zip(w, w[1:])]
        out.append(bt + " " + " ".join(extra) if extra else bt)
    return out


def block(s1: pd.DataFrame, index: pd.DataFrame, k: int = 20, rev_k: int = 3,
          max_df: int = 2000, chunk: int = 2000, verbose: bool = True) -> pd.DataFrame:
    """
    s1, index: DataFrames with 'country', 'block_text', 'name_n', 'addr_n' and a
    default 0..n-1 index. index = S2 and S3 records together.
    Returns DataFrame(s1_pos, cand_pos, score, rank) with positions into s1 / index.
    rank = position of the candidate in its S1's list, by score (0 = best).
    """
    out = []
    for country in s1["country"].unique():          # open set: works for France too
        a_pos = np.flatnonzero((s1["country"] == country).to_numpy())
        b_pos = np.flatnonzero((index["country"] == country).to_numpy())
        if len(a_pos) == 0 or len(b_pos) == 0:
            continue
        t0 = time.time()
        b_text = _with_pairs(index["block_text"].to_numpy()[b_pos],
                             index["name_n"].to_numpy()[b_pos], index["addr_n"].to_numpy()[b_pos])
        try:
            vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, dtype=np.float32,
                                  sublinear_tf=True, max_df=min(max_df, len(b_pos)))
            B = vec.fit_transform(b_text)
        except ValueError:                            # every word too common: keep them all
            vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, dtype=np.float32,
                                  sublinear_tf=True)
            B = vec.fit_transform(b_text)
        del b_text
        A = vec.transform(_with_pairs(s1["block_text"].to_numpy()[a_pos],
                                      s1["name_n"].to_numpy()[a_pos],
                                      s1["addr_n"].to_numpy()[a_pos]))
        parts = []
        BT = B.T.tocsr()                              # forward: S1 rows -> top-k S2/S3
        for i in range(0, A.shape[0], chunk):
            r, c, d, _ = _topk_per_row(A[i:i + chunk] @ BT, k)
            parts.append(pd.DataFrame({"s1_pos": a_pos[i + r].astype(np.int32),
                                       "cand_pos": b_pos[c].astype(np.int32),
                                       "score": d.astype(np.float32)}))
        del BT
        n_fwd = sum(len(p) for p in parts)
        if rev_k > 0:
            AT = A.T.tocsr()                          # reverse: S2/S3 rows -> top S1s
            for j in range(0, B.shape[0], chunk):
                r, c, d, _ = _topk_per_row(B[j:j + chunk] @ AT, rev_k)
                parts.append(pd.DataFrame({"s1_pos": a_pos[c].astype(np.int32),
                                           "cand_pos": b_pos[j + r].astype(np.int32),
                                           "score": d.astype(np.float32)}))
            del AT
        del A, B
        df = pd.concat(parts, ignore_index=True).drop_duplicates(["s1_pos", "cand_pos"])
        df = df.sort_values(["s1_pos", "score"], ascending=[True, False], ignore_index=True)
        df["rank"] = df.groupby("s1_pos").cumcount().astype(np.int16)
        out.append(df)
        if verbose:
            print(f"  {country}: {len(a_pos):,} S1 vs {len(b_pos):,} S2/S3, "
                  f"vocab {len(vec.vocabulary_):,}, forward {n_fwd:,} + reverse-only "
                  f"{len(df) - n_fwd:,} pairs, {time.time() - t0:.0f}s", flush=True)
    if not out:
        return pd.DataFrame(columns=["s1_pos", "cand_pos", "score", "rank"])
    return pd.concat(out, ignore_index=True)


def recall_at_k(cand: pd.DataFrame, s1: pd.DataFrame, index: pd.DataFrame,
                true_pairs: pd.DataFrame, ks=(5, 10, 20, 30)) -> pd.DataFrame:
    """
    true_pairs: DataFrame(source1_entity_id, ids) of true matches.
    Recall@k = share of true matches that appear in the candidates with rank < k.
    This is the ceiling for the final model's recall.
    """
    s1p = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    ixp = pd.Series(np.arange(len(index)), index=index["entity_id"])
    n = len(index)
    tkey = (s1p[true_pairs["source1_entity_id"]].to_numpy(np.int64) * n
            + ixp[true_pairs["ids"]].to_numpy(np.int64))
    rows = []
    for k in list(ks) + [None]:
        c = cand if k is None else cand[cand["rank"] < k]
        ckey = c["s1_pos"].to_numpy(np.int64) * n + c["cand_pos"].to_numpy(np.int64)
        rows.append({"k": "all" if k is None else k, "recall": np.isin(tkey, ckey).mean()})
    return pd.DataFrame(rows)
