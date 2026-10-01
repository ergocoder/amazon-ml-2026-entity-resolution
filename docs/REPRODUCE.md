# NC4: Business Entity Resolution (TF-IDF blocking + LightGBM)

This folder regenerates both submission files, `output/candidate_pairs.tsv` and `output/matching_results.tsv`, from the challenge data. The approach is explained in `Documentation_template.md` at the root of the zip. This file only covers how to run it.

Final version: **v5**. 46 features, LightGBM, then a per-entity expected-F0.5 decoder. Validation F0.5 0.9428 (the previous version, v4, scored 0.9366 on validation and 0.9238 on the public leaderboard).

## Environment

- Python 3.11, CPU only. No GPU, no internet access and no external data are needed.
- Tested on Windows 10/11, Intel i7, 16 GB RAM. 16 GB is enough, but close other heavy programs while blocking and training run.
- Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Main libraries: pandas, numpy, scikit-learn (TF-IDF), rapidfuzz (string similarity), anyascii (transliteration), lightgbm.

## Expected folder layout

The scripts use relative paths. Run all commands from this folder (`code/business_entity_resolution/`), with the challenge `dataset/` folder two levels up:

```
<root>/
├── dataset/
│   ├── train/   train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/    test_source1.tsv, test_source2.tsv, test_source3.tsv
├── output/      (both TSVs are written here)
└── code/business_entity_resolution/
    ├── README.md
    ├── requirements.txt
    ├── work/    (created automatically, holds intermediate pickles)
    └── src/
```

If your data is somewhere else, pass its path with `--data` in steps 1 and 2 and update `DATA` in the notebook's load cell.

## Files in `src/`

| File | What it does |
|---|---|
| `normalize.py` | Cleans names and addresses: transliteration, lowercasing, legal-suffix removal, address abbreviations, number parts. Builds the text used for blocking. |
| `blocking.py` | TF-IDF candidate generation per country: top-20 candidates per Source 1 record, plus the top-3 Source 1 records for each Source 2/3 record, with word-pair tokens. |
| `run_test_blocking.py` | Runs normalization + blocking on train or test and saves the results to `work/`. |
| `features.py` | The 31 base pair features (`context_features`, `string_features`), `assign()` (each Source 2/3 record to at most one Source 1), and `f05_macro()` (the official metric). Used by v5 unchanged. |
| `features_v5.py` | The 15 v5 features on top of the 31 base ones (46 in total). |
| `decode_f05.py` | The expected-F0.5 decoder: for each Source 1 record, picks the set of candidates with the highest expected F0.5, instead of using one global threshold. |
| `predict_test_v5.py` | **Final prediction script.** Scores the test candidates with `lgbm_v5.txt`, decodes with `decode_f05.py`, and writes both output files. |
| `lgbm_v5.txt` | The trained final model (LightGBM text format). |
| `explore.ipynb` | Data exploration, error analysis, and the training cells for all versions. |
| `predict_test.py`, `lgbm_v4.txt` | Previous version (v4, threshold 0.7). `predict_test_v5.py` reuses its file writer. |
| `apply_decoder.py` | Re-decodes saved v4 probabilities with the decoder (v4 + decoder). Kept for reference. |
| `features_v3.py`, `predict_test_v3.py`, `lgbm_v3.txt` | Older version, kept for reference only. |
| `check_addr_blocking.py` | Address-only blocking experiment (not adopted). Not needed to reproduce the outputs. |

## Quick reproduction (using the included model)

The trained model `src/lgbm_v5.txt` is included, so you can skip training and regenerate the outputs in two steps.

**1. Block the test set** (normalization + candidate generation, roughly an hour on our laptop):

```bash
python src/run_test_blocking.py --data ../../dataset --k 20 --split test
```

This writes `work/test_s1.pkl`, `work/test_idx.pkl` and `work/test_cand.pkl` (about 48.2M candidate pairs).

**2. Score, decode and write the outputs:**

```bash
python src/predict_test_v5.py --out ../../output
```

It uses `src/lgbm_v5.txt` and the decoder settings `--gamma 1.15 --floor 0.05` by default. It prints progress per batch of Source 1 records and ends with a `done in ...s | ... matches | ...% S1 with no match` line. It also saves the test probabilities to `work/test_scored_v5.pkl`.

Both `candidate_pairs.tsv` and `matching_results.tsv` are now in `output/`.

## Full reproduction (retraining the model)

To retrain from scratch, run these before step 2 above.

**A. Block the training set** (about 65 minutes, 56.4M candidate pairs):

```bash
python src/run_test_blocking.py --data ../../dataset --k 20 --split train
```

This writes `work/train_s1.pkl`, `work/train_idx.pkl` and `work/train_cand.pkl`. The expected summary line is `done: 56,388,107 pairs`.

**B. Train the model in `src/explore.ipynb`.** Restart the kernel first, then run only these v5 cells, in this order. Each is identified by its first line:

| Order | Cell (first line) | What it does |
|---|---|---|
| 1 | `import numpy as np, pandas as pd, gc` | Loads the train pickles and ground truth |
| 2 | `del gt, true_tr, s1p, ixp` | Frees memory before the features step |
| 3 | `from features import string_features, assign, f05_macro` | Builds the 46 v5 features and labels (the slow cell) |
| 4 | `import lightgbm as lgb` (the cell ending in `model.save_model("lgbm_v5.txt")`) | Trains LightGBM and saves `lgbm_v5.txt` |
| 5 | `from decode_f05 import sweep` | Validation: threshold 0.7 vs the decoder, and top features by gain |

Skip all other cells (exploration, the v4 cells and error analysis).

Expected output: the features cell prints `8,452,803 pairs | 46 features (should be 46)`. Training stops early at 140 rounds with validation log-loss about 0.0287. The last cell shows F0.5 about 0.9427 at threshold 0.7 and about 0.9428 with the decoder at gamma 1.15, floor 0.05.

Training details: validation is the Source 1 entities with position % 20 == 0, so the split is by entity, not by pair. Context features are computed on the full 56.4M candidate set. String features are computed on 15% of Source 1 entities (position % 20 < 3) to keep runtime manageable. Every Source 1 entity in that sample keeps all of its candidates, which the within-entity v5 features need.

**C. Restart or close the notebook kernel** to free RAM, then run steps 1 and 2 of the quick reproduction.

## Validate the output

From the challenge's `student_resource/` folder:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

It should print `PASS`.

## Notes

- Country is treated as an open label. Blocking runs separately for every country present in the data, so France (test only) needs no special code.
- No external APIs, geocoding or web lookups are used anywhere in the pipeline.
- LightGBM is MIT licensed and the model is trained from scratch on the provided training data. No pretrained models are used.
- Blocking and prediction run in chunks to stay within 16 GB RAM. If you run out of memory, close the notebook kernel and other programs before running the scripts.
