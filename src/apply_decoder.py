"""
apply_decoder.py - re-decide the test matches with the expected-F0.5 decoder
(decode_f05.py), using the v4 probabilities that predict_test.py already saved
in work/test_scored.pkl. The model is NOT run again (takes minutes, not 12+).

Settings tuned on validation: gamma=1.15, floor=0.05
(val F0.5 0.9379 vs 0.9366 with the plain 0.7 threshold).

Run from code/business_entity_resolution/:
    python src/apply_decoder.py --out ../../output_v4dec
Writes candidate_pairs.tsv + matching_results.tsv to --out (never to output_v3/).
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import assign            # noqa: E402  one-S1-per-candidate step
from decode_f05 import decode_all      # noqa: E402  per-S1 expected-F0.5 choice
from predict_test import write_lists   # noqa: E402  same file writer as v4


def main():
    """Load saved test probabilities, keep each candidate only for its best S1,
    decode per S1 with the F0.5 decoder, and write both submission files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="../../output_v4dec")
    ap.add_argument("--gamma", type=float, default=1.15)
    ap.add_argument("--floor", type=float, default=0.05)
    args = ap.parse_args()
    if "output_v3" in os.path.abspath(args.out).lower().split(os.sep):
        sys.exit(f"refusing to write into the v3 backup folder: {args.out}")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    s1_ids = pd.read_pickle(f"{args.work}/test_s1.pkl")["entity_id"].tolist()
    cand_ids = pd.read_pickle(f"{args.work}/test_idx.pkl")["entity_id"].to_numpy()
    cand = pd.read_pickle(f"{args.work}/test_cand.pkl")[["s1_pos", "cand_pos"]]
    scored = pd.read_pickle(f"{args.work}/test_scored.pkl")
    print(f"loaded {len(s1_ids):,} S1, {len(scored):,} scored pairs "
          f"({time.time() - t0:.0f}s)", flush=True)

    pre = assign(scored, 0.02)          # same pre-step as the validation sweep
    del scored
    pred = decode_all(pre, "s1_pos", "cand_pos", "prob",
                      gamma=args.gamma, p_floor=args.floor)
    s_list, c_list = [], []
    for s, ids in pred.items():
        for c in ids:
            s_list.append(int(s))
            c_list.append(int(c))
    matches = pd.DataFrame({"s1_pos": s_list, "cand_pos": c_list})
    print(f"decoded ({time.time() - t0:.0f}s)", flush=True)

    write_lists(f"{args.out}/candidate_pairs.tsv", "candidate_entity_ids", s1_ids, cand, cand_ids)
    write_lists(f"{args.out}/matching_results.tsv", "matched_entity_ids", s1_ids, matches, cand_ids)
    n_empty = len(s1_ids) - matches["s1_pos"].nunique()
    print(f"done in {time.time() - t0:.0f}s | {len(matches):,} matches | "
          f"{n_empty / len(s1_ids):.1%} S1 with no match (v4 threshold run: 6.6%)", flush=True)


if __name__ == "__main__":
    main()
