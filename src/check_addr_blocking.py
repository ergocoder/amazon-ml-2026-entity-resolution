"""
Experiment (val S1s only): does an ADDRESS-ONLY TF-IDF search, added on top of the
current blocking, recover true matches that blocking misses?

Reads work/train_s1.pkl, work/train_idx.pkl (already normalized, same positions as
work/train_cand.pkl), work/train_cand.pkl and the train ground truth TSV.
Writes only output_exp/addr_val_pairs.tsv.
Run:  python code/business_entity_resolution/src/check_addr_blocking.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from blocking import _topk_per_row  # noqa: E402
from normalize import number_parts  # noqa: E402

WORK = os.path.join(HERE, "..", "work")
DATA = os.path.join(HERE, "..", "..", "..", "dataset")
OUT = os.path.join(HERE, "..", "..", "..", "output_exp")
KS = (10, 20)
MAX_DF, CHUNK = 2000, 2000


def addr_text(addrs):
    """Address + its number parts + neighbouring-word pairs (blocking.py style, address only)."""
    out = []
    for a in addrs:
        w = a.split()
        out.append(" ".join([a] + number_parts(a) + [x + "_" + y for x, y in zip(w, w[1:])]))
    return out


def main():
    """Run the address-only search for val S1s, print recall of current blocking vs the
    union at K=10/20, and save the new pairs to output_exp/addr_val_pairs.tsv."""
    t0 = time.time()
    s1 = pd.read_pickle(f"{WORK}/train_s1.pkl")[["entity_id", "country", "addr_n"]]
    idx = pd.read_pickle(f"{WORK}/train_idx.pkl")[["entity_id", "country", "addr_n"]]
    N = len(idx)
    gt = pd.read_csv(f"{DATA}/train/train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False)
    true = (gt.assign(ids=gt["matched_entity_ids"].str.split(","))
              .explode("ids").query("ids != ''"))
    s1p = pd.Series(np.arange(len(s1)), index=s1["entity_id"])
    ixp = pd.Series(np.arange(N), index=idx["entity_id"])
    ts = s1p[true["source1_entity_id"]].to_numpy(np.int64)
    tc = ixp[true["ids"]].to_numpy(np.int64)
    del gt, true, s1p, ixp
    v = ts % 20 == 0                                   # same val split as training
    ts, tc = ts[v], tc[v]
    tkey = ts * N + tc
    val_s1 = np.flatnonzero(np.arange(len(s1)) % 20 == 0)

    cand = pd.read_pickle(f"{WORK}/train_cand.pkl")
    cand = cand[cand["s1_pos"] % 20 == 0]
    ckey = np.unique(cand["s1_pos"].to_numpy(np.int64) * N + cand["cand_pos"].to_numpy(np.int64))
    del cand
    print(f"loaded: {len(val_s1):,} val S1 | {len(tkey):,} val true pairs | "
          f"{len(ckey):,} current val pairs ({time.time() - t0:.0f}s)", flush=True)

    # ---- address-only search, per country, val S1s only ----
    t1 = time.time()
    s1_addr, s1_cty = s1["addr_n"].to_numpy(), s1["country"].to_numpy()
    ix_addr, ix_cty = idx["addr_n"].to_numpy(), idx["country"].to_numpy()
    ix_empty = ix_addr == ""
    parts = []
    for country in pd.unique(s1_cty[val_s1]):          # open set of countries
        a_pos = val_s1[(s1_cty[val_s1] == country) & (s1_addr[val_s1] != "")]
        b_pos = np.flatnonzero((ix_cty == country) & ~ix_empty)
        if len(a_pos) == 0 or len(b_pos) == 0:
            continue
        tc0 = time.time()
        vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, dtype=np.float32,
                              sublinear_tf=True, max_df=min(MAX_DF, len(b_pos)))
        B = vec.fit_transform(addr_text(ix_addr[b_pos]))
        A = vec.transform(addr_text(s1_addr[a_pos]))
        BT = B.T.tocsr()
        del B
        for i in range(0, A.shape[0], CHUNK):
            r, c, d, rk = _topk_per_row(A[i:i + CHUNK] @ BT, max(KS))
            parts.append(pd.DataFrame({"s1_pos": a_pos[i + r], "cand_pos": b_pos[c],
                                       "score": d.astype(np.float32), "rank": rk.astype(np.int16)}))
        print(f"  {country}: {len(a_pos):,} val S1 vs {len(b_pos):,} S2/S3 with address, "
              f"vocab {len(vec.vocabulary_):,}, {time.time() - tc0:.0f}s", flush=True)
        del A, BT, vec
    addr = pd.concat(parts, ignore_index=True)
    akey = addr["s1_pos"].to_numpy(np.int64) * N + addr["cand_pos"].to_numpy(np.int64)
    search_s = time.time() - t1

    # ---- report ----
    missed = ~np.isin(tkey, ckey)
    emp_t = ix_empty[tc]
    n_ent = len(np.unique(ts))
    print(f"\nval S1 with >=1 true match: {n_ent:,} | val S1 with empty address: "
          f"{(s1_addr[val_s1] == '').sum():,}")
    print(f"true pairs missed by current blocking: {missed.sum():,} "
          f"(S2/S3 address empty: {(missed & emp_t).sum():,}, non-empty: {(missed & ~emp_t).sum():,})")
    rows = []
    for name, key in [("current", ckey)] + [
            (f"union@{k}", np.union1d(ckey, akey[addr["rank"].to_numpy() < k])) for k in KS]:
        found = np.isin(tkey, key)
        rec = found & missed
        rows.append({"setting": name, "val_pairs": len(key),
                     "extra_per_S1": (len(key) - len(ckey)) / len(val_s1),
                     "pair_recall": found.mean(),
                     "entity_full_recall": pd.Series(found).groupby(ts).all().mean(),
                     "recovered": rec.sum(), "rec_addr_nonempty": (rec & ~emp_t).sum(),
                     "rec_addr_empty": (rec & emp_t).sum()})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # ---- save new pairs (not in current blocking), K up to 20 ----
    new = addr[~np.isin(akey, ckey)].copy()
    new_key = new["s1_pos"].to_numpy(np.int64) * N + new["cand_pos"].to_numpy(np.int64)
    new.insert(0, "cand_entity_id", idx["entity_id"].to_numpy()[new["cand_pos"]])
    new.insert(0, "s1_entity_id", s1["entity_id"].to_numpy()[new["s1_pos"]])
    new["is_true"] = np.isin(new_key, tkey).astype(np.int8)
    os.makedirs(OUT, exist_ok=True)
    new.to_csv(f"{OUT}/addr_val_pairs.tsv", sep="\t", index=False)
    print(f"\nsaved {len(new):,} new pairs ({new['is_true'].sum():,} true) to output_exp/addr_val_pairs.tsv"
          f" (rank < 10 = the @10 set)")
    print(f"runtime: address search {search_s:.0f}s | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
