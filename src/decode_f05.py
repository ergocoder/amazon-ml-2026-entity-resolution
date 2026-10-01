"""
decode_f05.py - choose, for EACH Source-1 entity, the set of candidates that
maximises the EXPECTED per-entity F0.5 (the leaderboard metric), instead of
using one global probability threshold.

Plain-language idea:
  The model gives each candidate a probability of being a true match.
  For one S1, we imagine many possible "truths" (each candidate is a real match
  with its probability), and for k = 0, 1, 2, ... we compute the average F0.5
  we'd get by predicting the top-k candidates. We pick the best k.
  k = 0 (predict nothing) wins when "no match at all" is likely enough - which
  is exactly how singletons earn their 1.0.

Use AFTER the one-S1-per-candidate assignment step (candidates that lost to a
different S1 should already be removed / have prob 0).
No retraining needed: it only re-decides using existing probabilities.
"""
import numpy as np

BETA2 = 0.25        # beta = 0.5  ->  beta^2 = 0.25
N_SAMPLES = 512     # simulated "truths" per S1 (more = smoother, slower)
P_FLOOR = 0.02      # candidates below this prob are never worth predicting
RNG = np.random.default_rng(0)


def best_subset(p, beta2=BETA2, n_samples=N_SAMPLES, rng=RNG):
    """p: 1-D array of match probabilities for ONE S1's candidates.
    Returns positions (into p) of the candidates to predict; may be empty."""
    if len(p) == 0:
        return np.array([], dtype=int)
    order = np.argsort(-p)                       # highest probability first
    ps = p[order]
    truth = rng.random((n_samples, len(ps))) < ps  # simulated truths (S, n)
    n_true = truth.sum(1)                          # how many real matches per draw
    tp_cum = np.cumsum(truth, axis=1)              # true positives if we take top-k
    k = np.arange(1, len(ps) + 1)
    # F_beta = (1+b^2)*TP / (b^2*|truth| + |predicted|); equals 0 when TP = 0
    f = (1 + beta2) * tp_cum / (beta2 * n_true[:, None] + k[None, :])
    exp_f = f.mean(0)                              # expected F0.5 for k = 1..n
    exp_empty = float(np.prod(1.0 - ps))           # F=1 only if there are no matches
    best_k = int(np.argmax(exp_f)) + 1
    if exp_empty >= exp_f[best_k - 1]:
        return np.array([], dtype=int)
    return order[:best_k]


def decode_all(df, s1_col, cand_col, p_col, all_s1=None, gamma=1.0, p_floor=P_FLOOR):
    """df: one row per (S1, candidate) with a probability column.
    gamma: calibration knob, p -> p**gamma (gamma > 1 = more conservative).
    all_s1: every S1 id that must appear in the output (e.g. all test S1s).
    Returns dict {s1_id: [matched ids]}."""
    d = df[df[p_col] >= p_floor]
    out = {}
    for s1, g in d.groupby(s1_col, sort=False):
        p = np.clip(g[p_col].to_numpy(dtype=float) ** gamma, 1e-6, 1 - 1e-6)
        idx = best_subset(p)
        out[s1] = [str(x) for x in g[cand_col].to_numpy()[idx]]
    if all_s1 is not None:
        for s in all_s1:
            out.setdefault(s, [])                  # S1s with no candidates -> empty
    return out


def macro_f05(pred, gt):
    """pred, gt: dicts {s1_id: iterable of matched ids}. Scores every S1 in gt,
    exactly like the leaderboard (singletons: 1.0 if empty predicted, else 0)."""
    scores = []
    for s1, g in gt.items():
        g, p = set(g), set(pred.get(s1, []))
        if not g and not p:
            scores.append(1.0)
            continue
        tp = len(g & p)
        if tp == 0:
            scores.append(0.0)
            continue
        prec, rec = tp / len(p), tp / len(g)
        scores.append(1.25 * prec * rec / (0.25 * prec + rec))
    return float(np.mean(scores))


def sweep(val_df, gt, s1_col, cand_col, p_col,
          gammas=(0.7, 0.85, 1.0, 1.15, 1.3, 1.5), floors=(0.02, 0.05, 0.1)):
    """Try calibration settings on VALIDATION and print scores.
    Compare against your current threshold-0.7 score (0.9366 for v4)."""
    best = (-1, None)
    for gm in gammas:
        for fl in floors:
            pred = decode_all(val_df, s1_col, cand_col, p_col,
                              all_s1=list(gt.keys()), gamma=gm, p_floor=fl)
            s = macro_f05(pred, gt)
            print(f"gamma={gm:<5} floor={fl:<5} F0.5={s:.4f}")
            if s > best[0]:
                best = (s, (gm, fl))
    print("BEST:", best)
    return best


# ---------------------------------------------------------------------------
# Example wiring (adapt the column names to your predict_test.py / val code):
#
#   from decode_f05 import sweep, decode_all
#   # val_pairs: val candidates with columns s1, cand, prob (after assignment)
#   # gt: {s1: [ids]} built from train_ground_truth.tsv for the val S1s
#   score, (gm, fl) = sweep(val_pairs, gt, "s1", "cand", "prob")
#
#   test_pred = decode_all(test_pairs, "s1", "cand", "prob",
#                          all_s1=test_s1_ids, gamma=gm, p_floor=fl)
#   rows = [(s, ",".join(ids)) for s, ids in test_pred.items()]
#   pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"]) \
#     .to_csv("output/matching_results.tsv", sep="\t", index=False)
#   # candidate_pairs.tsv stays unchanged; decoded matches are a subset of it.
# ---------------------------------------------------------------------------
