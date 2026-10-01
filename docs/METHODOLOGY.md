# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** NC4  
**Team Members:** Bhargav Gajare (Team Leader), Bhargavi Shinde, Anjali Arethiya, Sanika Kadam  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We built a three-stage pipeline: TF-IDF candidate generation (blocking) on normalized name and address text, a LightGBM classifier that scores each (Source 1, candidate) pair using 46 hand-built features, and a decoder that picks, for each Source 1 entity, the set of matches with the highest expected F0.5. The final version (v5) reaches **0.9428 F0.5 on validation**. The previous version (v4, validation 0.9366) scored **0.9238 on the public leaderboard**, and so far validation gains have carried over to the leaderboard almost one to one. Most of the improvement over our first submission (0.778) came from making training and validation look like the test set in terms of how crowded the candidate lists are, from a reverse blocking pass, and from features that describe how a pair compares to the other candidates around it, not just how similar the two strings are.

Everything runs on CPU on a 16 GB laptop. No pretrained models, no external data and no web or geocoding lookups are used anywhere.

---

## 2. Methodology

### 2.1 Problem Analysis

**Size.** The data is large for a laptop. Train has 2.21M Source 1 records (US 1.32M, India 0.88M) and about 10.3M Source 2 + Source 3 records. Test has 1.73M Source 1 records, including 259k French records, and about 10M S2/S3 records. Comparing every S1 against every S2/S3 is out of the question, so blocking is necessary, and it has to be done country by country.

**Label structure.** A few facts from the ground truth shaped the whole design:

- Only 5.6% of S1 entities are singletons (no match). So an all-empty submission would score about 0.056, and the real work is in finding matches without making false ones.
- Most S1 entities have 2 to 5 matches. The mode is 3.
- Every S2/S3 record matches **at most one** S1. About 25% of S2/S3 records match nothing at all.

The "at most one S1" fact is important. It means that once a candidate is claimed by one S1, it can't belong to another, and it lets us resolve conflicts at the end instead of treating every pair independently.

**Noise we saw while reading samples.** Beyond the patterns listed in the problem statement, we noticed:

- Names and state names written in Indian scripts, for example "राम मार्केटिंग प्राइवेट लिमिटेड" for "Ram Marketing Private Limited", and Kannada state names.
- Legal suffixes moved to the front ("Pvt. EFS Print Ventures Ltd.", "LLC Moncada Learning Center").
- Accents added to plain English words even in US data ("Léarning"), so accent stripping is needed everywhere, not only for France.
- Digits substituted for letters inside words, websites used as names ("wilfordhancock.com"), and some S2/S3 names that are essentially gibberish. For those records, the address is the only usable signal.
- US states written both as full names and as codes ("Texas" vs "TX"), addresses in all caps, city placed first, leading zeros on house numbers ("004303").
- Empty addresses. These turned out to matter a lot more than their share of the data suggests (see Section 5).

**France.** France appears only in test. We couldn't validate on it directly, so the rule was to never special-case countries: blocking groups by whatever country labels exist, and normalization handles French legal forms (SARL, SAS, EURL etc.) and accents the same way it handles the others. The v5 postal-code feature also treats a 5-digit French code postal the same way as a US ZIP or an Indian PIN.

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier + one-to-one assignment + per-entity F0.5 decoding  
**Core Innovation:** Training at the same candidate density as test; context features that describe how a pair compares to the other candidates around it (competition between S1s for the same record, exact-name ambiguity, margins within an S1's list, similarity to the S1's other strong candidates); and a decoder that optimizes the expected per-entity F0.5 directly.

The pipeline has four steps:

1. **Normalize** names and addresses into clean token strings.
2. **Block**: for each S1, shortlist likely S2/S3 records with TF-IDF similarity, within the same country, searching in both directions.
3. **Score** every shortlisted pair with LightGBM on 46 features.
4. **Decide**: give each S2/S3 record to at most one S1 (the one with the highest probability), then, for each S1, choose the set of candidates with the highest expected F0.5.

The biggest lesson of the challenge came early. Our first model was trained on a 10% sample of train. Validation said 0.9617, but the leaderboard said 0.778. The sample was about 10x less crowded than the real data: each S2/S3 record had far fewer competing S1s, rare-word statistics were different, and the context features took values the test set never produces. Validation was simply measuring an easier problem. Once we rebuilt blocking on the full training set and computed all context features there, validation started tracking the leaderboard closely (val 0.79 vs LB 0.778 for the same model), and from then on we could trust validation to tell us whether a change helped.

---

## 3. Candidate Generation (Blocking)

### Normalization (`normalize.py`)

Both fields go through the same base cleanup: transliterate to ASCII with `anyascii` (this handles Devanagari, Kannada and other Indian scripts as well as accents), lowercase, turn "&" into "and", remove dots from acronyms, split hyphen/slash-joined words, drop other punctuation, fix digit-for-letter substitutions inside words, and strip leading zeros from numbers.

Then:

- **Names**: legal-form words are removed (private, pvt, limited, ltd, llc, inc, corp, company, and French forms like sarl, sas, eurl), as are website fragments like com and www. If a name is nothing but legal words, we keep it as is rather than returning an empty string.
- **Addresses**: common abbreviations are mapped to one form (rd → road, st → street, ave → avenue, nr → near, opp → opposite, and so on), US state names map to their codes, a couple of city aliases are unified (bengaluru → bangalore), and filler words that carry no identity are dropped (no, unit, flat, shop, plot, house).
- Digit runs inside compound numbers ("2702/1313" → "2702", "1313") are added as extra tokens, so municipal numbering formats still match.

### Blocking keys and search (`blocking.py`)

Each record becomes one text: normalized name + normalized address + number parts, plus **word-pair tokens** made from neighbouring words in the name and in the address (for example `peacock_road`). We vectorize this with a word-level TF-IDF and use cosine similarity, separately for each country.

Two details matter here:

- **`max_df = 2000`**: any token appearing in more than 2,000 records is dropped. Words like "road", "services" or a big city name barely help identify a business and would make the sparse matrix products far too slow. The downside is that two records sharing *only* common words become invisible to each other. The word-pair tokens fix most of that: "peacock" and "road" may both be common, but "peacock_road" usually isn't.
- **Two directions**. The forward pass keeps the **top 20** S2/S3 records for each S1. The reverse pass keeps, for each S2/S3 record, its **top 3** S1s, and adds those pairs too. The reason: in a crowded area, a true match can get pushed out of its S1's top 20 by many look-alike records, but looking from the S2/S3 side it is almost always among the best few S1s. Since each S2/S3 belongs to at most one S1, the reverse view is a natural fit.

### Numbers

| | Train (full) | Test |
|---|---|---|
| S1 records | 2,206,821 | 1,732,544 |
| S2 + S3 records | 10,320,219 | 9,969,589 |
| Candidate pairs | 56,388,107 (44.1M forward + 12.3M reverse-only) | 48,190,097 |
| Avg candidates per S1 | ~25.6 | ~27.8 |

Blocking recall on the full train set (share of true pairs that land in the candidate set) is **0.958**: 0.972 for US and 0.937 for India. India is harder mainly because of transliteration and landmark-style addresses.

Compared with all possible S1 × S2/S3 pairs on test, the candidate set keeps roughly 3 pairs in a million, a reduction ratio above 0.99999.

### How we made sure true matches weren't lost

- We measured recall on the **full** training set, not a sample. This mattered: the first version of blocking showed recall@20 of 0.954 on a 10% sample but only about 0.71 at full density. That gap is what led to the word-pair tokens and the reverse pass, which together brought full-density recall to 0.958.
- We checked that the recall loss was affordable against the metric. With recall 0.958, blocking misses cost about 0.015 F0.5 on validation (Section 5), which is less than what the model loses on missed matches and false merges combined.
- Blocking uses name and address together, so a record with a garbled name can still be found through its address, and vice versa.

`candidate_pairs.tsv` is exactly this candidate set: the pairs the model scores. There is no filtering stage between blocking and the model.

---

## 4. Matching Model

### Features (46)

The model uses 31 base features (v4) plus 15 added in v5.

**Context features (9)**, computed on the full candidate set, so they mean the same thing in training and test:

| Feature | What it captures |
|---|---|
| `score` | TF-IDF cosine similarity from blocking |
| `rank` | Position of the candidate in its S1's list (0 = best) |
| `score_ratio_to_top` | Score divided by the best score in the S1's list |
| `n_s1_competitors` | How many S1s have this candidate in their list |
| `is_best_s1_for_cand` | Whether this S1 is the candidate's highest-scoring S1 |
| `cand_is_s3` | Whether the candidate comes from Source 3 |
| `name_exact` | Normalized names are identical |
| `cand_n_same_name_s1` | How many S1s in the candidate's list have exactly its name |
| `s1_n_same_name_cands` | How many of the S1's candidates have exactly its name |

The last three were added in v4. Error analysis showed that about half of the false merges where a record got "stolen" by the wrong S1 happened because two different S1 entities had exactly the same normalized name (think of chain stores or common names like "Sharma Traders"). For those pairs the string similarity features all say "perfect match", so the model had no way to tell the two S1s apart. The same-name counts tell it when a perfect name match is ambiguous, so it relies more on the address in exactly those cases. Name comparison is done on integer codes (names factorized once across all sources), which keeps this cheap on 56M pairs.

**Name features (9)**, using RapidFuzz: `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, ratio with spaces removed (catches "abc traders" vs "abctraders"), word Jaccard, empty-name flags for each side, and length difference.

**Address features (8)**: `ratio`, `token_set_ratio`, `partial_ratio`, word Jaccard, empty-address flags for each side, and token counts for each side.

**Number features (5)**: Jaccard over the numbers in the two addresses, count of shared numbers, numbers only in S1, numbers only in the candidate, and whether the first number matches. House numbers, PIN codes and ZIP codes all flow through these. A conflicting ZIP or PIN is one of the strongest "not a match" signals the model uses.

When one side is empty, Jaccard-style features return -1 ("can't tell") rather than 0, so the model can distinguish "different" from "missing".

**v5 features (15)** (`features_v5.py`). These describe a pair relative to its neighbours: the other S1s competing for the same candidate, and the other candidates of the same S1. None of them use labels or the country value.

| Feature | What it captures |
|---|---|
| `score_margin_in_cand` | Candidate-side score margin: this S1's TF-IDF score minus the best score among the *other* S1s that have this candidate in their list. Computed on the full candidate set. |
| `name_token_set_rank_in_s1`, `name_token_set_margin_in_s1` | Within-S1 rank of the candidate by name `token_set_ratio`, and its margin over the S1's next-best (or best other) candidate |
| `name_ratio_rank_in_s1`, `name_ratio_margin_in_s1` | The same for name `ratio` |
| `addr_token_set_rank_in_s1`, `addr_token_set_margin_in_s1` | The same for address `token_set_ratio` |
| `n_cands_of_s1` | How many candidates the S1 has |
| `postal_conflict` | Both addresses have a postal code (the last 5 to 6 digit number: US ZIP, India PIN, French code postal) and the codes differ. Missing when either side has none. |
| `house_first_equal` | The first non-postal number (house or plot number) is the same on both sides. Missing when either side has none. |
| `cand_name_vocab_frac` | Share of the candidate's name words that appear anywhere in Source 1 names. Near 0 means a gibberish name, so the model should lean on the address. |
| `sib_name_sim`, `sib_addr_sim`, `sib_best_sim` | Sibling similarity: name, address and best-of-both `token_set_ratio` between this candidate and the S1's strongest *other* candidate (by TF-IDF score) |
| `sib_score` | That sibling's TF-IDF score |

The sibling features come from the observation that true matches are copies of the same business. A candidate with a gibberish name can still be recognized if its address closely resembles another strong candidate of the same S1.

### Model

**Model type:** LightGBM binary classifier (MIT license), trained from scratch on our features. No pretrained model of any size is used.

Settings: binary log-loss objective, learning rate 0.1, 63 leaves, min 50 samples per leaf, feature fraction 0.8, early stopping after 50 rounds without improvement on validation. The final v5 model stopped at 140 rounds with validation log-loss **0.0287**, down from 0.0324 for v4 (and 0.0350 for the 28-feature v3).

The two most important features by total gain are both new in v5: `score_margin_in_cand` and `sib_name_sim`, followed by `addr_token_set` and `addr_token_set_margin_in_s1`.

### Training and validation setup

The split is **by S1 entity**, never by pair, so no S1 appears in both training and validation. Validation is the S1s with position % 20 == 0 (5% of train, about 110k entities, singletons included).

String features are the slow part (RapidFuzz over tens of millions of pairs on a laptop), so we computed them for 15% of S1 entities (position % 20 < 3). That gives 8.45M labelled pairs, 13% of them true matches: about 5.7M for training and 2.8M for validation. The important point is that the context features were computed on the **full** 56M-pair candidate set before this sampling. The sample only reduces how many rows we train on. It doesn't change what each row looks like, which was exactly the problem with our first version. Each sampled S1 keeps all of its candidates, so the within-S1 v5 features are computed exactly as on test.

### Decision step

**Assignment.** Each S2/S3 record is first kept only for the S1 that gives it the highest probability. This follows directly from the "at most one S1" property of the data and removes a lot of duplicate claims on the same record.

**Threshold (v1 to v4).** Up to v4 we then kept pairs above one global probability threshold, chosen by sweeping on validation with the official macro F0.5 (our implementation reproduces the 0.714 worked example from the problem statement). For v4:

| Threshold | Val F0.5 | S1 predicted empty |
|---|---|---|
| 0.5 | 0.9340 | 5.9% |
| 0.6 | 0.9366 | 6.3% |
| **0.7** | **0.9366** | 6.7% |
| 0.8 | 0.9334 | 7.2% |
| 0.9 | 0.9230 | 8.1% |

**Expected-F0.5 decoder (final).** A single threshold treats every entity the same, but the metric is computed per S1 entity, and the best choice depends on the whole set of probabilities an entity has. For example, predicting nothing earns a full 1.0 only if the entity truly has no match. So for each S1 we choose the prediction set directly (`decode_f05.py`):

1. Take the S1's candidates (after assignment) and sort them by probability. Candidates below a floor of 0.05 are never predicted.
2. Simulate 512 possible "truths", treating each candidate as a real match with its probability.
3. For every k, compute the average F0.5 we would get by predicting the top-k candidates. Also compute the expected score of predicting nothing, which is the probability that none of the candidates is a match.
4. Predict the option with the highest expected F0.5, which may be the empty set.

Before decoding, probabilities are sharpened as p^γ with **γ = 1.15**, which makes the decoder a little more conservative. γ and the floor were tuned on validation. The decoder only re-uses the model's probabilities, so it needs no retraining.

| Model | Global threshold 0.7 | Decoder (γ 1.15, floor 0.05) |
|---|---|---|
| v4 | 0.9366 | 0.9379 |
| v5 | 0.9427 | **0.9428** |

The decoder gave v4 +0.0013. With the better-calibrated v5 model the gain is only +0.0001, but it is never worse on validation, so we kept it.

---

## 5. Results & Error Analysis

### Scores across versions

| Version | Main change | Val F0.5 | Public LB |
|---|---|---|---|
| v1 | Blocking + 28 features, trained on a 10% sample | 0.9617 (misleading) | 0.778 |
| v2 | Same model, blocking and training at full density | 0.79 | not submitted (val tracked v1's LB) |
| v3 | Word-pair tokens + reverse top-3 blocking, recall 0.71 → 0.958 | 0.9328 | 0.920 |
| v4 | + 3 same-name ambiguity features | 0.9366 | 0.9238 |
| v4 + decoder | Same model, expected-F0.5 decoder instead of threshold 0.7 | 0.9379 | not yet known |
| **v5 + decoder** | **+ 15 neighbourhood features (46 total), decoder** | **0.9428** | **not yet known** |

**F_0.5 Score (macro), best validation:** 0.9428 (v5 + decoder, γ 1.15, floor 0.05). Validation log-loss fell from 0.0324 (v4) to 0.0287 (v5).

From v2 onwards, validation ran a steady 1 to 1.3 points above the public leaderboard, and gains carried over almost one to one (v3 → v4: +0.0038 on validation, +0.0038 on the leaderboard). We think the offset is mostly France, which validation can't see, but we haven't been able to confirm that. With about 110k entities in validation, random noise in the val score is around ±0.001, so we treated any val gain of about 0.003 or more as real. The v4 → v5 gain (+0.0062 with the decoder) is well above that.

### Where the points go

On v3's validation set, we split every lost point into three buckets:

| Bucket | F0.5 lost |
|---|---|
| Missed matches the model scored below threshold (FN) | 0.035 |
| Wrong matches (FP) | 0.0175 |
| True matches blocking never found | 0.015 |

So the ceiling from blocking alone is about 0.985, and the model loses more points by being cautious than by being wrong. That's expected with F0.5 and a 0.7 threshold, and it's a sensible trade: a false merge on a singleton costs a full point, while a missed link on an entity with four matches costs much less.

We also checked whether IDs carried any hidden signal (for example, matched IDs being numerically close). They don't, so the model uses only the record contents.

### Common false positives (wrong merges)

- **Same-name ambiguity.** About half of the cases where a record was claimed by the wrong S1 involved two S1 entities with exactly the same normalized name. The v4 features target this directly and cut validation log-loss by about 7%. The v5 candidate-side score margin (`score_margin_in_cand`) goes further: it tells the model how clearly this S1 beats the other S1s competing for the same record.
- **Records with empty addresses.** S2/S3 records with no address are only 3.3% of all records, but they account for about 18% of false positives. With only a name to go on, a common name is easy to attach to the wrong business.
- Branches or related businesses on the same street with very similar names, where only a small word differs. The v5 postal-code conflict and house-number features give the model a direct signal for these.

### Common false negatives (missed matches)

- **Empty addresses again.** These records are 3.3% of the data but about 24% of missed matches and about 24% of blocking misses. The model's median probability for the missed empty-address pairs was around 0.24: it saw some evidence, but not enough to cross 0.7.
- **Heavy typos and gibberish names.** When the name is garbled, the pair has to be carried by the address, and address noise (numbering formats, landmark references, missing parts) often isn't enough to reach the threshold. The v5 sibling features and the name-vocabulary share were added for exactly this case.
- **Transliteration differences** in Indian records, where `anyascii` produces a different spelling than the Latin version in the other source. This is part of why India's blocking recall (0.937) trails the US (0.972).

---

## 6. Conclusion

A classical pipeline (careful normalization, two-way TF-IDF blocking, 46 features, LightGBM and a per-entity F0.5 decoder) reaches 0.9428 F0.5 on validation on a laptop CPU, up from 0.9366 for the version that scored 0.9238 on the leaderboard. The most useful lesson wasn't a model choice. It was making validation honest: our first model looked excellent on a sampled validation set and was 18 points worse on the leaderboard, because the sample was far less crowded than test. Once training matched test density, every later change could be judged on validation before spending a submission. The biggest late gains came from features that compare a pair with its neighbours (competing S1s, the S1's other candidates) rather than from better string similarity. The remaining gap is mostly records with empty addresses and names that are ambiguous or garbled, which point to where we'd go next.

---

## Appendix

### A. Code Artefacts

All code is in `code/business_entity_resolution/src/`:

| File | Purpose |
|---|---|
| `normalize.py` | Name and address normalization, transliteration, number parts, builds the blocking text |
| `blocking.py` | TF-IDF blocking per country: forward top-20 + reverse top-3, word-pair tokens |
| `run_test_blocking.py` | Runs normalization + blocking on train or test (`--split train` / `--split test`), saves candidate pickles to `work/` |
| `features.py` | The 31 base features: `context_features` (9) and `string_features` (22), `assign()` (one S1 per candidate), `f05_macro()` (official metric) |
| `features_v5.py` | The 15 v5 features (46 in total) |
| `decode_f05.py` | Expected-F0.5 decoder and the validation sweep for γ and the floor |
| `predict_test_v5.py` | Final prediction: scores test candidates with `lgbm_v5.txt`, decodes, writes both output files |
| `lgbm_v5.txt` | The final trained model |
| `explore.ipynb` | EDA, error analysis, and the training cells: load full-train candidates, build features, train LightGBM, validation sweep |
| `predict_test.py`, `lgbm_v4.txt` | Previous version (v4, threshold 0.7); `predict_test_v5.py` reuses its file writer |
| `apply_decoder.py` | Applies the decoder to saved v4 probabilities (v4 + decoder), kept for reference |
| `features_v3.py`, `predict_test_v3.py`, `lgbm_v3.txt` | Older version, kept for reference |
| `check_addr_blocking.py` | Address-only blocking experiment (not adopted, see below) |

End-to-end order (exact commands are in `README.md`):

1. `run_test_blocking.py --split train`, then `--split test` (blocking, about an hour each on our laptop).
2. v5 training cells in `explore.ipynb`, which save `lgbm_v5.txt`.
3. `predict_test_v5.py --out ../../output` (decoder γ 1.15, floor 0.05 by default), which writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
4. `utils/validate_submission.py` to check both files.

Libraries: pandas, NumPy, scikit-learn (TF-IDF), RapidFuzz (MIT), anyascii, LightGBM (MIT). All run locally; nothing calls the network.

### B. Additional Results

**Blocking recall by country (full train):** US 0.972, India 0.937, overall 0.958.

**Label distribution (train):** 5.6% singletons; 119k S1 with 1 match, 375k with 2, 531k with 3, 484k with 4, 322k with 5, 165k with 6, with a thin tail beyond that.

**Ideas we considered but didn't ship:**

- *Address-only secondary blocking (tested, not adopted).* A teammate's pipeline, which originally blocked on names only, gained a lot from adding an address search, so we tested the same idea on ours. We added a separate address-only TF-IDF nearest-neighbour search (top-10 and top-20 per Source 1 record) and took the union with our existing name+address candidates, on the validation split. At K=20, pair recall rose only from 0.9580 to 0.9597, while candidate pairs grew by 27%, and only about 1 in 1,200 new pairs was a true match. The gain was small because our blocking already indexes name and address together. Also, 24% of the remaining misses are Source 2/3 records with an empty address, which an address search can't reach. We kept the original blocking.
- *Special handling of empty-address features.* We looked at returning missing values instead of -1 for address features when an address is empty, but the existing empty flags and -1 values already give the model this information, so we dropped it.
- *Full use of links between S2 and S3 records.* The v5 sibling features compare each candidate with the S1's single strongest other candidate. A fuller version would compare it with all of the S1's likely matches, or cluster S2 and S3 records directly. That could help further with gibberish names, but we didn't have time to build it.
- *Multilingual text embeddings or a small cross-encoder* for transliteration and for France. Promising, but too heavy for our hardware in the time available.
