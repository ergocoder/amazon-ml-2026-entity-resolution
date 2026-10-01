# Bhargavi's solution — TF-IDF blocking + LightGBM

## Approach

1. **Normalize** business names and addresses (`src/normalize.py`) — lowercase,
   strip legal suffixes/punctuation, transliterate non-Latin scripts
   (Indian-language names/states) to ASCII with `anyascii`.
2. **Block** (`src/blocking.py`): TF-IDF word + word-pair vectors, per country.
   Forward: top-20 S2/S3 candidates per S1. Reverse: top-3 S1 per S2/S3 record.
3. **Features** (`src/features.py`): 28 pairwise similarity features (RapidFuzz
   string metrics, TF-IDF cosine, address/zip/pin overlap, etc.) per candidate pair.
4. **Model**: LightGBM binary classifier on the pairwise features, scored to a
   match probability.
5. **Threshold**: 0.7 on the match probability, tuned on held-out validation for
   Macro F0.5. Each Source-2/3 record is assigned to at most one Source-1 entity
   (its highest-scoring match above threshold).

## Results

### v3 (current, submitted to portal)

- Blocking v2: forward top-20 S1 candidates per S1 + reverse top-3 per S2/S3
  record + word-pair tokens.
- Full-train blocking recall = 0.958.
- Model: `src/lgbm_v3.txt`.
- Validation Macro F0.5 = 0.933 at threshold 0.7 (full-train density).
- 6.5% of test Source-1 entities predicted as having no match (empty).

### v1 (superseded)

- Leaderboard F0.5 = 0.778. Its validation score (0.9617) was measured on a 10%
  train sample, which has far fewer candidates per S1 than test, so it
  overstated precision.

## Files

- `src/normalize.py` — text normalization / transliteration
- `src/blocking.py` — TF-IDF candidate generation
- `src/features.py` — pairwise similarity features
- `src/run_test_blocking.py` — runs blocking on the test set, writes candidate pairs
- `src/predict_test.py` — scores candidates with the trained model, writes both
  submission files
- `src/lgbm_v3.txt` — current trained LightGBM model (text/booster format)
- `src/lgbm_v1.txt` — earlier model, kept for reference
- `src/explore.ipynb` — data exploration notebook

## How to run

From this folder (`code/bhargavi/`), with `dataset/` and `output/` present two
levels up (repo root):

```bash
pip install -r requirements.txt

python src/run_test_blocking.py --data ../../dataset --k 20
python src/predict_test.py --model src/lgbm_v3.txt --threshold 0.7 --out ../../output
```

This writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
