"""
Normalize + block a full split (test or train), saving results to work/.
Run from code/business_entity_resolution/:
    python src/run_test_blocking.py --data ../../dataset --k 20                 (test)
    python src/run_test_blocking.py --data ../../dataset --k 20 --split train   (train)
Takes a while (possibly 1-3 hours). Safe to leave running.
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from normalize import add_normalized  # noqa: E402
from blocking import block  # noqa: E402

KEEP = ["entity_id", "country", "name_n", "addr_n", "block_text"]


def load_norm(path, n_jobs, step=1_000_000):
    """Read a TSV and normalize it in 1M-row pieces to limit memory."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    parts = [add_normalized(df.iloc[i:i + step], n_jobs=n_jobs)[KEEP]
             for i in range(0, len(df), step)]
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../dataset")
    ap.add_argument("--out", default="work")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    sp = args.split
    t1 = load_norm(f"{args.data}/{sp}/{sp}_source1.tsv", args.n_jobs)
    t23 = pd.concat([load_norm(f"{args.data}/{sp}/{sp}_source{i}.tsv", args.n_jobs)
                     for i in (2, 3)], ignore_index=True)
    t1.to_pickle(f"{args.out}/{sp}_s1.pkl")
    t23.to_pickle(f"{args.out}/{sp}_idx.pkl")
    print(f"normalized in {time.time() - t0:.0f}s", flush=True)

    cand = block(t1, t23, k=args.k)
    cand.to_pickle(f"{args.out}/{sp}_cand.pkl")
    print(f"done: {len(cand):,} pairs, total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":  # required on Windows for multiprocessing
    main()
