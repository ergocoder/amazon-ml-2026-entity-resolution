"""
Score the test candidates with the trained model and write both submission files.
Needs work/test_s1.pkl, work/test_idx.pkl, work/test_cand.pkl (from run_test_blocking.py).
Run from code/business_entity_resolution/:
    python src/predict_test.py --threshold 0.7 --out ../../output
(--model defaults to src/lgbm_v4.txt; output_v3/ is a backup and is never written to.)
"""
import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_COLS, assign, context_features, string_features  # noqa: E402


def write_lists(path, header2, s1_ids, pairs, cand_ids):
    """One row per S1 entity; comma-joined candidate IDs (empty if none). Tab-separated."""
    groups = {}
    for s, c in zip(pairs["s1_pos"].to_numpy(), pairs["cand_pos"].to_numpy()):
        groups.setdefault(s, []).append(cand_ids[c])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{header2}\n")
        for i, sid in enumerate(s1_ids):
            ids = list(dict.fromkeys(groups.get(i, [])))  # dedupe, keep order
            fh.write(f"{sid}\t{','.join(ids)}\n")


def main():
    """Load the test pickles and the model, score every candidate pair, pick matches
    at --threshold, and write candidate_pairs.tsv + matching_results.tsv to --out."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--model", default="src/lgbm_v4.txt")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--out", default="../../output")
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

    ctx = context_features(cand, s1, idx)   # needs the full candidate set
    scored = []
    for start in range(0, len(s1), args.s1_chunk):   # string features in pieces to save RAM
        part = ctx[(ctx["s1_pos"] >= start) & (ctx["s1_pos"] < start + args.s1_chunk)]
        if len(part) == 0:
            continue
        feats = string_features(part, s1, idx, n_jobs=args.n_jobs)
        out = part[["s1_pos", "cand_pos"]].copy()
        out["prob"] = model.predict(feats[FEATURE_COLS]).astype(np.float32)
        scored.append(out)
        print(f"  scored S1 {start:,}-{start + args.s1_chunk:,} ({time.time() - t0:.0f}s)", flush=True)
    scored = pd.concat(scored, ignore_index=True)
    scored.to_pickle(f"{args.work}/test_scored.pkl")   # keep, so thresholds can be changed later

    s1_ids = s1["entity_id"].tolist()
    cand_ids = idx["entity_id"].to_numpy()
    matches = assign(scored, args.threshold)
    write_lists(f"{args.out}/candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cand, cand_ids)
    write_lists(f"{args.out}/matching_results.tsv", "matched_entity_ids", s1_ids, matches, cand_ids)
    n_empty = len(s1) - matches["s1_pos"].nunique()
    print(f"done in {time.time() - t0:.0f}s | {len(matches):,} matches | "
          f"{n_empty / len(s1):.1%} S1 with no match", flush=True)


if __name__ == "__main__":
    main()
