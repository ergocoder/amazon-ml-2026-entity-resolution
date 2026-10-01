"""
predict_test_v5.py - score test candidates with the v5 model (31 v4 features +
15 new ones from features_v5.py), then pick matches with the expected-F0.5
decoder (decode_f05.py). Writes both submission files.

Needs work/test_s1.pkl, work/test_idx.pkl, work/test_cand.pkl (from run_test_blocking.py).
Run from code/business_entity_resolution/:
    python src/predict_test_v5.py --gamma 1.15 --floor 0.05 --out ../../output_v5
Use the gamma/floor that won the v5 validation sweep in explore.ipynb.
Saves probabilities to work/test_scored_v5.pkl (v4's test_scored.pkl is untouched).
"""
import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import assign, string_features  # noqa: E402
from features_v5 import (FEATURE_COLS_V5, context_features_v5,  # noqa: E402
                         extra_features, record_arrays)
from decode_f05 import decode_all  # noqa: E402
from predict_test import write_lists  # noqa: E402


def main():
    """Load test pickles + v5 model, build v5 features chunk by chunk (whole S1s per
    chunk), score, decode per S1 with the F0.5 decoder, write both TSVs to --out."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--model", default="src/lgbm_v5.txt")
    ap.add_argument("--gamma", type=float, default=1.15)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--out", default="../../output_v5")
    ap.add_argument("--s1_chunk", type=int, default=200_000)
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()
    if "output_v3" in os.path.abspath(args.out).lower().split(os.sep):
        sys.exit(f"refusing to write into the v3 backup folder: {args.out}")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    s1 = pd.read_pickle(f"{args.work}/test_s1.pkl")
    idx = pd.read_pickle(f"{args.work}/test_idx.pkl")
    cand = pd.read_pickle(f"{args.work}/test_cand.pkl")
    model = lgb.Booster(model_file=args.model)
    print(f"loaded {len(s1):,} S1, {len(idx):,} S2/S3, {len(cand):,} pairs", flush=True)

    ctx = context_features_v5(cand, s1, idx)   # needs the FULL candidate set
    rec = record_arrays(s1, idx)               # per-record values, computed once
    print(f"context + record features ready ({time.time() - t0:.0f}s)", flush=True)

    scored = []
    for start in range(0, len(s1), args.s1_chunk):   # whole S1s per chunk
        part = ctx[(ctx["s1_pos"] >= start) & (ctx["s1_pos"] < start + args.s1_chunk)]
        if len(part) == 0:
            continue
        feats = string_features(part, s1, idx, n_jobs=args.n_jobs)
        feats = extra_features(feats, s1, idx, rec)
        out = part[["s1_pos", "cand_pos"]].copy()
        out["prob"] = model.predict(feats[FEATURE_COLS_V5]).astype(np.float32)
        scored.append(out)
        del feats
        print(f"  scored S1 {start:,}-{start + args.s1_chunk:,} ({time.time() - t0:.0f}s)", flush=True)
    del ctx
    scored = pd.concat(scored, ignore_index=True)
    scored.to_pickle(f"{args.work}/test_scored_v5.pkl")

    pre = assign(scored, 0.02)                 # each candidate kept only for its best S1
    del scored
    pred = decode_all(pre, "s1_pos", "cand_pos", "prob", gamma=args.gamma, p_floor=args.floor)
    s_list = [int(s) for s, ids in pred.items() for _ in ids]
    c_list = [int(c) for ids in pred.values() for c in ids]
    matches = pd.DataFrame({"s1_pos": s_list, "cand_pos": c_list})

    s1_ids = s1["entity_id"].tolist()
    cand_ids = idx["entity_id"].to_numpy()
    write_lists(f"{args.out}/candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cand, cand_ids)
    write_lists(f"{args.out}/matching_results.tsv", "matched_entity_ids", s1_ids, matches, cand_ids)
    n_empty = len(s1) - matches["s1_pos"].nunique()
    print(f"done in {time.time() - t0:.0f}s | {len(matches):,} matches | "
          f"{n_empty / len(s1):.1%} S1 with no match (v4+decoder: 5.6%)", flush=True)


if __name__ == "__main__":
    main()
