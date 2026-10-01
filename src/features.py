"""
Pairwise features: for every (S1 record, candidate) pair, compute numbers that
describe how similar they are. LightGBM then learns which combinations mean "match".
"""
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz  # MIT license, fast string similarity

_NUM = re.compile(r"\d+")

# made by context_features (needs the full candidate set)
CTX_COLS = [
    "score", "rank", "score_ratio_to_top", "n_s1_competitors", "is_best_s1_for_cand",
    "cand_is_s3",
    "name_exact", "cand_n_same_name_s1", "s1_n_same_name_cands",
]
# made by string_features, in the order _pair_features returns them
STR_COLS = [
    "name_ratio", "name_token_sort", "name_token_set", "name_partial", "name_nospace",
    "name_jaccard", "name_empty_s1", "name_empty_cand", "name_len_diff",
    "addr_ratio", "addr_token_set", "addr_partial", "addr_jaccard",
    "addr_empty_s1", "addr_empty_cand", "addr_len_s1", "addr_len_cand",
    "num_jaccard", "num_shared", "num_only_s1", "num_only_cand", "first_num_equal",
]
FEATURE_COLS = CTX_COLS + STR_COLS


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return -1.0  # -1 = "can't tell" (one side empty); LightGBM handles this fine
    return len(a & b) / len(a | b)


def _pair_features(args):
    """String features for a list of (name1, addr1, name2, addr2) tuples."""
    rows = []
    for n1, a1, n2, a2 in args:
        nt1, nt2 = set(n1.split()), set(n2.split())
        at1, at2 = set(a1.split()), set(a2.split())
        nums1, nums2 = set(_NUM.findall(a1)), set(_NUM.findall(a2))
        m1, m2 = _NUM.search(a1), _NUM.search(a2)
        rows.append((
            fuzz.ratio(n1, n2), fuzz.token_sort_ratio(n1, n2), fuzz.token_set_ratio(n1, n2),
            fuzz.partial_ratio(n1, n2), fuzz.ratio(n1.replace(" ", ""), n2.replace(" ", "")),
            _jaccard(nt1, nt2), not n1, not n2, abs(len(n1) - len(n2)),
            fuzz.ratio(a1, a2), fuzz.token_set_ratio(a1, a2), fuzz.partial_ratio(a1, a2),
            _jaccard(at1, at2), not a1, not a2, len(at1), len(at2),
            _jaccard(nums1, nums2), len(nums1 & nums2), len(nums1 - nums2), len(nums2 - nums1),
            (m1.group() == m2.group()) if (m1 and m2) else -1,
        ))
    return rows


def context_features(cand: pd.DataFrame, s1: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Cheap features about how a pair ranks among the other candidates.
    Must be computed on the FULL candidate set (not in chunks)."""
    f = cand[["s1_pos", "cand_pos", "score", "rank"]].copy()
    f["score_ratio_to_top"] = f["score"] / f.groupby("s1_pos")["score"].transform("max")
    f["n_s1_competitors"] = f.groupby("cand_pos")["s1_pos"].transform("size")
    best = f.groupby("cand_pos")["score"].transform("max")
    f["is_best_s1_for_cand"] = (f["score"] >= best).astype(np.int8)
    f["cand_is_s3"] = index["entity_id"].str.startswith("S3").to_numpy()[f["cand_pos"]]

    # Same-name features. Every name gets one integer code (S1 and S2/S3 share the
    # table), so "same name?" is an integer compare instead of millions of string compares.
    # Empty names get -1 so they never count as the same name.
    names = np.concatenate([s1["name_n"].to_numpy(), index["name_n"].to_numpy()])
    codes, _ = pd.factorize(names)
    codes[names == ""] = -1
    del names
    s, c = f["s1_pos"].to_numpy(), f["cand_pos"].to_numpy()
    c1, c2 = codes[:len(s1)][s], codes[len(s1):][c]
    exact = (c1 == c2) & (c1 >= 0)
    del codes, c1, c2
    f["name_exact"] = exact.astype(np.int8)
    # per candidate: how many S1s in its candidate list have exactly its name
    f["cand_n_same_name_s1"] = np.bincount(c[exact], minlength=len(index))[c].astype(np.int32)
    # per S1: how many of its candidates have exactly its name
    f["s1_n_same_name_cands"] = np.bincount(s[exact], minlength=len(s1))[s].astype(np.int32)
    return f


def string_features(f: pd.DataFrame, s1: pd.DataFrame, index: pd.DataFrame,
                    n_jobs: int = 4, chunk: int = 50_000) -> pd.DataFrame:
    """Adds name/address/number similarity columns to f (rows = pairs)."""
    n1 = s1["name_n"].to_numpy()[f["s1_pos"]]
    a1 = s1["addr_n"].to_numpy()[f["s1_pos"]]
    n2 = index["name_n"].to_numpy()[f["cand_pos"]]
    a2 = index["addr_n"].to_numpy()[f["cand_pos"]]

    def job_iter():
        for i in range(0, len(f), chunk):
            yield list(zip(n1[i:i + chunk], a1[i:i + chunk], n2[i:i + chunk], a2[i:i + chunk]))

    if n_jobs > 1:
        # Smaller chunks + imap (instead of one big pre-built list + map) keep less
        # data in flight across the process boundary at once - full-train runs were
        # sending 200k-row chunks and hanging/crashing a worker on this Windows box.
        with Pool(n_jobs, maxtasksperchild=50) as pool:
            parts = list(pool.imap(_pair_features, job_iter()))
    else:
        parts = [_pair_features(j) for j in job_iter()]
    str_cols = STR_COLS
    vals = np.array([r for p in parts for r in p], dtype=np.float32).reshape(-1, len(str_cols))
    f = f.copy()
    for j, c in enumerate(str_cols):
        f[c] = vals[:, j]
    return f


def make_features(cand: pd.DataFrame, s1: pd.DataFrame, index: pd.DataFrame,
                  n_jobs: int = 4, chunk: int = 200_000) -> pd.DataFrame:
    """All features at once (fine for the training sample)."""
    return string_features(context_features(cand, s1, index), s1, index, n_jobs, chunk)


# ---------- post-processing and scoring ----------

def assign(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Keep pairs with prob >= threshold, and give each candidate to at most ONE S1
    (the one with the highest prob), because in the data each S2/S3 record
    belongs to only one business.
    """
    p = pairs[pairs["prob"] >= threshold]
    p = p.sort_values("prob", ascending=False).drop_duplicates("cand_pos")
    return p[["s1_pos", "cand_pos", "prob"]]


def f05_macro(pred: pd.DataFrame, truth: pd.DataFrame, s1_positions) -> float:
    """
    pred, truth: DataFrames with s1_pos, cand_pos. s1_positions: all S1 entities
    being evaluated (singletons included). Implements the official F0.5 per S1, averaged.
    """
    s1_positions = pd.Index(np.unique(s1_positions))
    pk = set(zip(pred["s1_pos"], pred["cand_pos"]))
    t = truth[truth["s1_pos"].isin(s1_positions)]
    tp_mask = [k in pk for k in zip(t["s1_pos"], t["cand_pos"])]
    tp = t[tp_mask].groupby("s1_pos").size().reindex(s1_positions, fill_value=0)
    n_true = t.groupby("s1_pos").size().reindex(s1_positions, fill_value=0)
    n_pred = pred[pred["s1_pos"].isin(s1_positions)].groupby("s1_pos").size() \
        .reindex(s1_positions, fill_value=0)
    prec = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
    rec = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
    denom = 0.25 * prec + rec
    f = np.where(denom > 0, 1.25 * prec * rec / np.where(denom > 0, denom, 1), 0.0)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)  # correct singleton
    return float(f.mean())
