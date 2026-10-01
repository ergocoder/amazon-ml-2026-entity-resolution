"""
features_v5.py - v5 = the 31 v4 features (features.py, unchanged) + 15 new ones.

New features, in plain words:
  * score_margin_in_cand : how much better this S1 fits the candidate than the
                           best OTHER S1 competing for it (TF-IDF score).
                           Computed on the FULL candidate set (like v4 context).
  * <sim>_rank_in_s1 / <sim>_margin_in_s1 : is this candidate the best name /
                           address match among this S1's candidates, and by how much.
  * n_cands_of_s1        : how many candidates this S1 has.
  * postal_conflict      : both addresses have a 5-6 digit postal code and they
                           differ (NaN when either is missing).
  * house_first_equal    : first non-postal number (house/plot no.) equal
                           (NaN when either is missing).
  * cand_name_vocab_frac : share of the candidate's name words that appear in
                           S1 names -> near 0 means a gibberish name.
  * sib_*                : similarity of this candidate to the S1's strongest
                           OTHER candidate. True matches are copies of the same
                           business, so they resemble each other.

Nothing here reads labels or the country label. features.py is not modified,
so v4 stays reproducible.
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from features import FEATURE_COLS, context_features

_NUM = re.compile(r"\d+")

SIM_FOR_CONTEXT = ["name_token_set", "addr_token_set", "name_ratio"]
NEW_COLS = (["score_margin_in_cand"]
            + [f"{c}_{k}_in_s1" for c in SIM_FOR_CONTEXT for k in ("rank", "margin")]
            + ["n_cands_of_s1", "postal_conflict", "house_first_equal",
               "cand_name_vocab_frac", "sib_name_sim", "sib_addr_sim",
               "sib_best_sim", "sib_score"])
FEATURE_COLS_V5 = FEATURE_COLS + NEW_COLS


def _margin_np(key, val, n):
    """For each row: val minus the best val among the OTHER rows with the same key.
    NaN if the row is alone in its group. Pure numpy (low memory, no big sorts)."""
    val = np.nan_to_num(val.astype(np.float32), nan=-1.0)
    best = np.full(n, -np.inf, np.float32)
    np.maximum.at(best, key, val)
    is_top = val >= best[key]
    n_top = np.bincount(key[is_top], minlength=n)
    sec = np.full(n, -np.inf, np.float32)
    np.maximum.at(sec, key[~is_top], val[~is_top])
    sec = np.where(n_top >= 2, best, sec)          # tie for first -> margin 0
    other = np.where(is_top, sec[key], best[key])
    return np.where(np.isfinite(other), val - other, np.nan).astype(np.float32)


def context_features_v5(cand, s1, index):
    """v4 context features + candidate-side score margin, on the FULL candidate set."""
    f = context_features(cand, s1, index)
    f["score_margin_in_cand"] = _margin_np(f["cand_pos"].to_numpy(),
                                           f["score"].to_numpy(), len(index))
    return f


def _postal_house(addr):
    """(postal code, first other number) of a normalized address, -1 when missing.
    Postal = last 5-6 digit number (US ZIP, India PIN, France code postal)."""
    nums = _NUM.findall(addr)
    postal = next((x for x in reversed(nums) if len(x) in (5, 6)), None)
    house = next((x for x in nums if x != postal), None)
    return (int(postal) if postal else -1, int(house[:15]) if house else -1)


def record_arrays(s1, index):
    """Per-record values computed ONCE (not per pair): postal code, house number,
    and name-vocabulary share. Vocabulary = words in this split's S1 names."""
    sp = np.array([_postal_house(a) for a in s1["addr_n"]], dtype=np.int64).reshape(-1, 2)
    ip = np.array([_postal_house(a) for a in index["addr_n"]], dtype=np.int64).reshape(-1, 2)
    vocab = set()
    for n in s1["name_n"]:
        vocab.update(t for t in n.split() if len(t) >= 2)

    def frac(n):
        toks = [t for t in n.split() if len(t) >= 2]
        return sum(t in vocab for t in toks) / len(toks) if toks else np.nan

    ix_vocab = np.fromiter((frac(n) for n in index["name_n"]), np.float32, len(index))
    return {"s1_postal": sp[:, 0], "s1_house": sp[:, 1],
            "ix_postal": ip[:, 0], "ix_house": ip[:, 1], "ix_vocab": ix_vocab}


def _pair_sim(a, b):
    """token_set_ratio of a[i] vs b[i] for all i, scaled 0-1; NaN if either is empty."""
    try:
        from rapidfuzz.process import cpdist          # fast, multi-threaded (rapidfuzz >= 3.6)
        r = np.asarray(cpdist(list(a), list(b), scorer=fuzz.token_set_ratio, workers=-1),
                       dtype=np.float32)
    except ImportError:
        r = np.fromiter((fuzz.token_set_ratio(x, y) for x, y in zip(a, b)),
                        np.float32, len(a))
    r = r / 100.0
    r[(a == "") | (b == "")] = np.nan
    return r


def extra_features(f, s1, index, rec):
    """Adds the per-S1 new features to f (output of string_features).
    Every S1's candidates must ALL be in f (true for the notebook sample and for
    predict_test chunks, which both split by S1)."""
    if len(f) == 0:
        for col in NEW_COLS[1:]:
            f[col] = np.array([], np.float32)
        return f
    s = f["s1_pos"].to_numpy()
    c = f["cand_pos"].to_numpy()
    n1 = len(s1)

    # --- within-S1 context on string similarities
    for col in SIM_FOR_CONTEXT:
        f[f"{col}_rank_in_s1"] = (f.groupby("s1_pos")[col]
                                  .rank(ascending=False, method="min").astype(np.float32))
        f[f"{col}_margin_in_s1"] = _margin_np(s, f[col].to_numpy(), n1)
    f["n_cands_of_s1"] = np.bincount(s, minlength=n1)[s].astype(np.float32)

    # --- conflicts (NaN = one side missing, not a mismatch)
    p1, p2 = rec["s1_postal"][s], rec["ix_postal"][c]
    f["postal_conflict"] = np.where((p1 >= 0) & (p2 >= 0), (p1 != p2).astype(np.float32), np.nan)
    h1, h2 = rec["s1_house"][s], rec["ix_house"][c]
    f["house_first_equal"] = np.where((h1 >= 0) & (h2 >= 0), (h1 == h2).astype(np.float32), np.nan)
    f["cand_name_vocab_frac"] = rec["ix_vocab"][c]

    # --- sibling: the S1's strongest OTHER candidate by TF-IDF score (no labels used)
    sc = f["score"].to_numpy(np.float32)
    order = np.lexsort((-sc, s))
    ss, cs, scs = s[order], c[order], sc[order]
    start = np.r_[True, ss[1:] != ss[:-1]]
    rank = np.arange(len(ss)) - np.maximum.accumulate(np.where(start, np.arange(len(ss)), 0))
    top_c = np.full(n1, -1, np.int64)
    sec_c = np.full(n1, -1, np.int64)
    top_sc = np.full(n1, np.nan, np.float32)
    sec_sc = np.full(n1, np.nan, np.float32)
    m0, m1 = rank == 0, rank == 1
    top_c[ss[m0]], top_sc[ss[m0]] = cs[m0], scs[m0]
    sec_c[ss[m1]], sec_sc[ss[m1]] = cs[m1], scs[m1]
    is_top = c == top_c[s]
    sib = np.where(is_top, sec_c[s], top_c[s])
    f["sib_score"] = np.where(is_top, sec_sc[s], top_sc[s]).astype(np.float32)

    names = index["name_n"].to_numpy(dtype=object)
    addrs = index["addr_n"].to_numpy(dtype=object)
    has = sib >= 0
    name_sim = np.full(len(f), np.nan, np.float32)
    addr_sim = np.full(len(f), np.nan, np.float32)
    name_sim[has] = _pair_sim(names[c[has]], names[sib[has]])
    addr_sim[has] = _pair_sim(addrs[c[has]], addrs[sib[has]])
    f["sib_name_sim"] = name_sim
    f["sib_addr_sim"] = addr_sim
    f["sib_best_sim"] = np.fmax(name_sim, addr_sim)
    return f
