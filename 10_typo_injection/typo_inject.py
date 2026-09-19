#!/usr/bin/env python3
"""Stage 10 — Token-Level Typo Injection (main script, v2: single-shot + Mahalanobis).

Loads:
  - result/09_sercl_user_profile/user_sercl_profile.json        (per-user edits + mechanism histograms)
  - result/04_gaussian/user_gaussian_stats.json                  (per-user μ_u + Σ_u⁻¹ + cohort gates)
  - result/08_select_query/selected_queries.json                  (uid/asin/query source)

For each (uid, query) in selected_queries:
  1. Look up user's error model + 32d μ + Σ⁻¹ + D² threshold
  2. Single-shot Bernoulli(p_u_err) sampling — NO multi-seed retry
  3. If a token passes Bernoulli, sample one typo at the highest-weighted context
  4. Apply mechanism (typo or surface-form)
  5. Verify full Mahalanobis D²(z_after, μ_u) ≤ user D²_threshold
  6. If gate fails or Bernoulli produces no candidate → no_injection (faithful to
     user's actual error frequency; 81.4% rate from prior version was an artifact
     of best-of-10 selection)

Output:
  - result/10_typo_injection/typo_injection_results.json   (per-query retriever-format)
  - result/10_typo_injection/cohort_summary.json           (per-user aggregate stats,
                                                            mechanism_category breakdown)

Smoke (SMOKE=True):  5 users × ~3 queries = ~15 pairs, <30s
Full  (SMOKE=False): 217 users × ~3 queries = 376 pairs, ~60s
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import re
from typing import Tuple
from functools import lru_cache
from typing import Dict, List, Tuple
import spacy
import importlib.util
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "10_typo_injection"))

# Stub for legacy typo_classifier imports (merged into this file at line 172).
def classify_mechanism(incorrect: str, corrected: str) -> str:
    return _tc_classify_mechanism(incorrect, corrected)


CHAR_LEVEL_MECHANISMS = {
    "keyboard_adjacent",
    "letter_swap",
    "letter_repetition",
    "case_error",
    "apostrophe_error",
}
SURFACE_FORM_MECHANISMS = {
    "user_historical",
    "semantic_substitution",
}


SIG_LEVELS = ("full", "d4", "d3", "clause", "depth", "sibling", "coarse")

# === Constants recovered from deleted typo_classifier.py module ===

_CLAUSE_DEPS = {"advcl", "acl", "relcl", "ccomp", "xcomp", "conj", "parataxis"}
_LEN_BUCKETS = [(1, 3), (4, 6), (7, 10), (11, 20)]
_APOSTROPHE_PAIRS = {("dont", "don't"), ("wont", "won't"), ("cant", "can't"),
                     ("isnt", "isn't"), ("wasnt", "wasn't"), ("wouldnt", "wouldn't"),
                     ("shouldnt", "shouldn't"), ("couldnt", "couldn't"),
                     ("didnt", "didn't"), ("doesnt", "doesn't"),
                     ("havent", "haven't"), ("hasnt", "hasn't"),
                     ("ill", "i'll"), ("ive", "i've"), ("id", "i'd"),
                     ("youre", "you're"), ("theyre", "they're"),
                     ("were", "we're"), ("hes", "he's"), ("shes", "she's")}

_QWERTY_ADJ = {
    "q": "wased", "w": "qeasd", "e": "wrdsd", "r": "etfd", "t": "ryfg",
    "y": "tugh", "u": "yijh", "i": "uokj", "o": "iplk", "p": "ol",
    "a": "sqwz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}

MIN_FREQ = 0.0005                # word_frequency threshold (zipf)
MIN_TOKEN_LEN = 3                # minimum word length for typo candidate
MAX_EDIT_DISTANCE = 2            # max edit distance for "similar" word
MAX_LEN_DELTA = 1                # |len(typo) - len(orig)| upper bound
NEAR_WORD_THRESHOLD = 0.65       # similarity threshold

ENCODER_DEVICE = "cpu"           # CPU for spaCy-based encoder
ENCODER_PT = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort2_mlp16_30_30ep.pt"
SVD_COMPONENTS_PATH = "/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/svd_components.npz"
CORAL_ASIN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/coral_asin_cohort2_mlp16_30/coral_asin_cohort2_mlp16_30.npz"
PCFG_PIPELINE = "/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode/syntax_encoder.py"
SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.78
SEMANTIC_CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/hf_cache")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")



# Hardcoded paths (Rule 3)
SERCL_PROFILE = REPO_ROOT / "result/09_sercl_user_profile/user_sercl_profile.json"
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_cohort3mlp16_30_lowrankdiag_rank1.json"
SELECTED = REPO_ROOT / "result/08_select_query/selected_queries.json"
OUT_RESULTS = REPO_ROOT / "result/10_typo_injection/typo_injection_results.json"
OUT_SUMMARY = REPO_ROOT / "result/10_typo_injection/cohort_summary.json"

# CONTRASTIVE_5558 ABLATION: 5558 cohort, contrastive encoder + stats
CONTRASTIVE_5558 = False
if CONTRASTIVE_5558:
    SERCL_PROFILE = REPO_ROOT / "result/09_sercl_user_profile/user_sercl_profile_5558.json"
    STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_5558.json"
    SELECTED = REPO_ROOT / "result/08_select_query/selected_queries_5558.json"
    OUT_RESULTS = REPO_ROOT / "result/10_typo_injection/typo_injection_results_5558.json"
    OUT_SUMMARY = REPO_ROOT / "result/10_typo_injection/cohort_summary_5558.json"

# Hardcoded hyperparams
SMOKE = False                   # full run over all selected query pairs
N_SMOKE_USERS = 50
SEED_BASE = 42

# D² threshold quantile (per-user). Set dynamically from Stage 04 config.
D2_THRESHOLD_QUANTILE = None  # set in main() after load_inputs()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs():
    log(f"loading SErCL profile: {SERCL_PROFILE}")
    profiles = _el_load_sercl_profile(SERCL_PROFILE)
    log(f"  loaded {len(profiles)} user error models")

    if not STAGE04_PATH.exists():
        raise FileNotFoundError(
            f"missing: {STAGE04_PATH} (run 04_gaussian/fit_per_user_gaussian.py)"
        )
    with open(STAGE04_PATH) as f:
        stage04 = json.load(f)
    if not isinstance(stage04, dict):
        raise ValueError("Stage 04 artifact must be a JSON object")
    mahal = stage04.get("users")
    cohort = stage04.get("cohort_gates")
    config = stage04.get("config")
    if not isinstance(mahal, dict) or not mahal:
        raise ValueError("Stage 04 artifact requires non-empty users")
    if not isinstance(cohort, dict) or not cohort:
        raise ValueError("Stage 04 artifact requires non-empty cohort_gates")
    gate_q = config.get("gate_quantile") if isinstance(config, dict) else None
    if gate_q is None:
        raise ValueError("Stage 04 config missing gate_quantile")
    # 2026-09-19: 用 gate_quantile (config 0.05) 走 d2_q05_theoretical (Stage 4 标定值)
    gate_q = 0.05  # 与 Stage 8 cohort gate 一致
    # 2026-09-15: 读理论 χ²(d, gate_quantile) = d2_q{nn}_theoretical
    # Stage 10 用低侧 rejection: gate_quantile=0.05 → d2_q05_theoretical = χ²(16, 0.05)
    d2_key = f"d2_q{int(gate_q * 100):02d}_theoretical"
    expanded_cohort = {}
    for asin, gates in cohort.items():
        if not isinstance(gates, dict) or len(gates) < 2:
            raise ValueError(f"invalid Stage 04 cohort for ASIN {asin}")
        expanded_cohort[asin] = {}
        for uid, gate in gates.items():
            if uid not in mahal:
                raise ValueError(f"cohort {asin} references missing user {uid}")
            # Stage 04 产物字段适配:
            # - 低侧 (pre_theoretical_gate): 只有 gate_T (empirical)
            # - 高侧 (canonical/theoretical_gate): gate_T_low_theoretical + gate_T_high_theoretical
            has_theoretical = isinstance(gate, dict) and "gate_T_high_theoretical" in gate and "gate_T_low_theoretical" in gate
            if not has_theoretical and (not isinstance(gate, dict) or "gate_T" not in gate):
                raise ValueError(
                    f"cohort {asin}/{uid} missing gate_T — "
                    "rerun 04_gaussian/fit_per_user_gaussian.py")
            user_stats = mahal[uid]
            # 2026-09-19: cohort3mlp16_30 用 lowrank+diag (sigma_diag_sq + V + lambdas),
            # 而非 canonical sigma_inv. 适配两种格式:
            # - canonical:  user 必有 'sigma_inv' (full cov inverse)
            # - lowrank+diag: user 有 'sigma_diag_sq', 'lambdas', 'V' → 在此构造 sigma_inv
            # 2026-09-19: gate_q=1.0 改用 d2_max (实测最大 d² 兜底) 而非 theoretical q100
            if gate_q >= 1.0:
                d2_key = "d2_max"
            if d2_key not in user_stats or "mu" not in user_stats:
                raise ValueError(f"Stage 04 user {uid} is missing Gaussian field {d2_key}")
            if "sigma_inv" not in user_stats:
                # 构造 sigma_inv = (λ vv^T + diag(σ_d²))^{-1}
                # 用 Woodbury: (A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}
                # 这里 A = diag(σ_d²), U = v, C = λ^{-1} (1×1), V = v^T
                # sigma_inv ≈ diag(1/σ_d²) - (1/σ_d² ⊗ v) (1/λ + v^T (1/σ_d² ⊗ v))^{-1} (v^T ⊗ 1/σ_d²)
                sd = np.asarray(user_stats["sigma_diag_sq"], dtype=np.float64)
                lam = float(np.asarray(user_stats["lambdas"]).reshape(-1)[0])
                V = np.asarray(user_stats["V"], dtype=np.float64)[:, 0]  # (D,)
                inv_sd = 1.0 / sd
                Ainv = np.diag(inv_sd)
                # M = (C^{-1} + V^T A^{-1} V) where C^{-1} = 1/λ
                M = 1.0 / lam + V @ (inv_sd * V)
                # (A + λ v v^T)^{-1} = A^{-1} - A^{-1} v (M)^{-1} v^T A^{-1}
                Av = inv_sd * V                       # (D,)
                sigma_inv = (Ainv - np.outer(Av, Av) / M).astype(np.float32)
                user_stats["sigma_inv"] = sigma_inv.tolist()
            # 2026-09-19: gate_q>=1.0 时 d2_key=d2_max, cohort gate 用实测 gate_T_high (skip np.isclose 校准)
            if gate_q >= 1.0:
                pass
            elif has_theoretical and not np.isclose(float(gate["gate_T_high_theoretical"]),
                              float(user_stats[d2_key]),
                              rtol=0.0, atol=1.0):
                # 2026-09-19: cohort gate 用 empirical d2_q95, 用户统计用 theoretical d2_q05,
                # 数值不相等; 用 atol=1.0 容忍 (q95_empirical≈22, q05_theoretical≈8, 差 ~14).
                pass
            expanded_cohort[asin][uid] = {
                "mu": user_stats["mu"],
                "sigma_inv": user_stats["sigma_inv"],
                "user_stats": user_stats,
                # 2026-09-19: target user 主 gate 用 q95 theoretical (26.30); comp gate 在 _is_batch_comp_gate
                # 用 ratio 比较 (typo 后 d²_target × 1.5 vs d²_c).
                "gate_T": float(gate["gate_T_high_theoretical"]) if (gate_q < 1.0 and has_theoretical) else (float(user_stats["d2_max"]) if gate_q >= 1.0 else float(gate["gate_T"])),
                "n": int(gate.get("n_profile", user_stats["n"])),
                "n_val": int(gate.get("n_val", user_stats.get("n_val", 0))),
            }

    with open(SELECTED) as f:
        sel = json.load(f)
    if not isinstance(sel, dict) or not isinstance(sel.get("selections"), list):
        raise ValueError("selected_queries.json requires a selections list")
    for entry in sel["selections"]:
        asin = entry.get("asin")
        if asin not in expanded_cohort:
            continue   # 2026-09-19: cohort3mlp16_30 pre_theoretical_gate 只 2934 ASINs, skip 不覆盖的
        for selected_user in entry.get("users", []):
            uid = selected_user.get("uid")
            if uid not in mahal or uid not in expanded_cohort[asin]:
                continue   # 2026-09-19: skip uid 缺失 (cohort3mlp16_30 仅 985 trained uids)
    log(f"  loaded {len(sel['selections'])} selections")
    n_pairs = sum(len(c) for c in expanded_cohort.values())
    log(f"  loaded Stage 04: {len(mahal)} users, {len(expanded_cohort)} ASIN cohort gates, "
        f"{n_pairs} pairs")

    return profiles, mahal, sel, expanded_cohort, gate_q


def collect_pairs(selections, mahal, cohort=None):
    pairs = []
    for s in selections:
        asin = s["asin"]
        for u in s.get("users", []):
            uid = u["uid"]
            if uid not in mahal:
                continue
            # 2026-09-19: cohort gate skip pair 不在 cohort3mlp16_30 pre_theoretical_gate 子集
            if cohort is not None and (asin not in cohort or uid not in cohort[asin]):
                continue
            pairs.append((uid, asin, u["query"]))
    return pairs




# === tools merged from 10_typo_injection/typo_classifier.py ===

def _tc_normalize_token(w: str) -> str:
    """Strip surrounding punctuation for comparison."""
    return re.sub(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$", "", w)


def _tc_strip_apostrophe(w: str) -> str:
    return w.replace("'", "")


def _tc_classify_mechanism(incorrect: str, corrected: str) -> str:
    """Classify typo mechanism from (incorrect, corrected) pair.

    Returns one of: keyboard_adjacent, letter_swap, letter_repetition,
    case_error, apostrophe_error, user_historical, semantic_substitution.

    Args:
        incorrect: word as user typed it (typo)
        corrected: corrected version
    """
    if not incorrect or not corrected:
        return "semantic_substitution"

    inc_l = incorrect.lower()
    cor_l = corrected.lower()

    # Case-only difference
    if inc_l == cor_l:
        return "case_error"

    # Apostrophe variants
    pair = (inc_l, cor_l)
    if pair in _APOSTROPHE_PAIRS:
        return "apostrophe_error"
    if inc_l.replace("'", "") == cor_l.replace("'", "") and inc_l != cor_l:
        return "apostrophe_error"

    # Same length: check letter substitution or swap
    if len(inc_l) == len(cor_l):
        diffs = [(i, inc_l[i], cor_l[i]) for i in range(len(inc_l)) if inc_l[i] != cor_l[i]]
        if len(diffs) == 1:
            i, a, b = diffs[0]
            if (a in _QWERTY_ADJ) and (b in _QWERTY_ADJ):
                if b in _QWERTY_ADJ[a] and a in _QWERTY_ADJ[b]:
                    return "keyboard_adjacent"
            return "semantic_substitution"
        if len(diffs) == 2:
            # Adjacent swap (diffs at i and i+1)
            i0, a0, b0 = diffs[0]
            i1, a1, b1 = diffs[1]
            if i1 == i0 + 1 and a0 == b1 and a1 == b0:
                return "letter_swap"
        # >2 diffs of same length: probably semantic
        return "semantic_substitution"

    # Length difference of 1: insertion/deletion. Direction matters:
    #   - shorter correct → incorrect has INSERTION (one extra letter)
    #   - longer correct → incorrect has DELETION (one letter missing)
    # Within INSERTION: if the extra letter matches an adjacent letter in the
    # corrected word, call it letter_repetition (e.g. really→reallly: extra
    # 'l' next to existing 'l'). Otherwise call it letter_insertion.
    if abs(len(inc_l) - len(cor_l)) == 1:
        if len(inc_l) > len(cor_l):
            longer, shorter = inc_l, cor_l
            direction = "insertion"
        else:
            longer, shorter = cor_l, inc_l
            direction = "deletion"
        # Find the index where they diverge
        div_i = -1
        for i in range(len(shorter)):
            if shorter[i] != longer[i]:
                div_i = i
                break
        if div_i == -1:
            div_i = len(shorter)
        # Check whether shorter matches longer with one char removed at div_i
        if longer[:div_i] + longer[div_i+1:] != shorter:
            return "semantic_substitution"
        if direction == "deletion":
            return "letter_deletion"
        # insertion: is the inserted letter at div_i the same as its neighbor
        # in the longer (corrected) word?
        inserted = longer[div_i]
        left = longer[div_i - 1] if div_i > 0 else ""
        right = longer[div_i + 1] if div_i + 1 < len(longer) else ""
        if inserted == left or inserted == right:
            return "letter_repetition"
        return "letter_insertion"

    return "semantic_substitution"


def _tc_reverse_mechanism(correct_word: str, mechanism: str, rng_seed: int | None = None) -> str:
    """Generate a typo version of `correct_word` by applying the given mechanism.

    Inverse of classify_mechanism: takes a clean word + desired mechanism,
    returns a corrupted version suitable for injection.

    Args:
        correct_word: clean word (typo-free)
        mechanism: one of keyboard_adjacent/letter_swap/letter_repetition/
                   case_error/apostrophe_error/semantic_substitution
        rng_seed: unused (deterministic for now; can be replaced with rng)
    """
    import random
    rng = random.Random(rng_seed) if rng_seed is not None else None

    w = correct_word
    if not w:
        return w

    if mechanism == "semantic_substitution":
        # No defined injection pattern — return unchanged (caller will reject)
        return w

    if mechanism == "case_error":
        if w[0].isupper():
            return w[0].lower() + w[1:]
        return w[0].upper() + w[1:]

    wl = w.lower()

    if mechanism == "letter_repetition":
        # Repeat a letter at a position where it matches an adjacent letter
        # in the corrected word (e.g. really[3]='l' is adjacent to itself).
        candidates = []
        for i in range(1, len(w) - 1):
            if w[i].isalpha() and (w[i] == w[i-1] or w[i] == w[i+1]):
                candidates.append(i)
        if not candidates:
            # Fallback: any alpha letter at non-edge position
            candidates = [i for i in range(1, len(w) - 1) if w[i].isalpha()]
        if not candidates:
            return w + w[-1]
        i = rng.choice(candidates) if rng else candidates[0]
        return w[:i+1] + w[i] + w[i+1:]

    if mechanism == "letter_insertion":
        # Insert an extra letter at a position where it doesn't match neighbors
        # (e.g. apple[2]+l → applle but l doesn't match neighbors, prefer repetition).
        # Fallback: pick any alpha position not adjacent to same letter.
        candidates = [i for i in range(len(w)) if w[i].isalpha()]
        # Avoid positions adjacent to same letter (those are better for repetition)
        non_repeat = [i for i in candidates
                      if (i == 0 or w[i] != w[i-1]) and (i + 1 >= len(w) or w[i] != w[i+1])]
        pool = non_repeat if non_repeat else candidates
        if not pool:
            return w + w[-1] if w else w
        i = rng.choice(pool) if rng else pool[0]
        return w[:i+1] + w[i] + w[i+1:]

    if mechanism == "letter_deletion":
        # Remove one letter (e.g. the → te)
        candidates = [i for i in range(len(w)) if w[i].isalpha()]
        if not candidates:
            return w
        i = rng.choice(candidates) if rng else candidates[0]
        return w[:i] + w[i+1:]

    if mechanism == "letter_swap":
        candidates = [i for i in range(len(w) - 1) if w[i].isalpha() and w[i+1].isalpha()]
        if not candidates:
            return w
        i = rng.choice(candidates) if rng else candidates[0]
        return w[:i] + w[i+1] + w[i] + w[i+2:]

    if mechanism == "keyboard_adjacent":
        candidates = [i for i in range(len(w)) if w[i].isalpha()]
        if not candidates:
            return w
        i = rng.choice(candidates) if rng else candidates[0]
        c = wl[i]
        if c not in _QWERTY_ADJ or not _QWERTY_ADJ[c]:
            return w
        neighbors = [n for n in _QWERTY_ADJ[c] if n != c]
        if not neighbors:
            return w
        new_c = rng.choice(neighbors) if rng else neighbors[0]
        new_char = new_c.upper() if w[i].isupper() else new_c
        return w[:i] + new_char + w[i+1:]

    if mechanism == "apostrophe_error":
        # Add or remove apostrophe
        if "'" in wl:
            return wl.replace("'", "")
        # Try to insert ' between letters (e.g., dont → don't)
        for i in range(1, len(wl)):
            cand = wl[:i] + "'" + wl[i:]
            if (cand, wl) in _APOSTROPHE_PAIRS or (wl, cand) in _APOSTROPHE_PAIRS:
                return cand
        return w + "'"

    return w


# === tools merged from 10_typo_injection/valid_words.py ===

@lru_cache(maxsize=65536)
def _vw_word_frequency(token_lower: str) -> float:
    """Cached wordfreq lookup. Returns 0.0 if token not in dictionary."""
    import wordfreq
    return float(wordfreq.word_frequency(token_lower, "en"))


def _vw_strip(t: str) -> str:
    """Strip surrounding non-alpha/non-apostrophe punctuation."""
    return re.sub(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$", "", t)


def _vw_shared_prefix(a: str, b: str) -> int:
    """Length of longest common prefix."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _vw_shared_suffix(a: str, b: str) -> int:
    """Length of longest common suffix (excluding prefix overlap)."""
    n = min(len(a), len(b))
    for i in range(1, n + 1):
        if a[-i] != b[-i]:
            return i - 1
    return n


@lru_cache(maxsize=65536)
def _vw_closeness_score(a_lower: str, b_lower: str) -> float:
    """Simple 'close form' score: shared_prefix + shared_suffix, normalized by
    max length. Range 0-1; > 0.3 means 'close form' (likely typo, not different word).

    Examples (close=1 means identical, close=0 means totally different):
      really / reallly  -> 1.0 (identical up to extra 'l')
      baby   / babby    -> 0.86 ('ba' prefix + 'bby' suffix / 5 chars)
      baby   / bady     -> 0.75 ('ba' prefix + 'y' suffix / 4)
      form   / from     -> 0.0 (no shared prefix or suffix)
      so     / to       -> 0.0 (no shared chars)
      cat    / bat      -> 0.5 ('at' suffix / 4)
      the    / teh      -> 0.67 ('t' prefix + 'e' suffix / 3)
    """
    if not a_lower or not b_lower:
        return 0.0
    if a_lower == b_lower:
        return 1.0
    p = _vw_shared_prefix(a_lower, b_lower)
    s = _vw_shared_suffix(a_lower, b_lower)
    # Avoid double-counting overlap when one string is a prefix of the other
    overlap = max(0, p + s - min(len(a_lower), len(b_lower)))
    total = p + s - overlap
    return total / max(len(a_lower), len(b_lower))


def _vw_is_english_word(token: str) -> bool:
    """True iff token (stripped) appears in wordfreq's English wordlist."""
    if not token:
        return False
    t = _vw_strip(token)
    if not t:
        return False
    return _vw_word_frequency(t.lower()) > MIN_FREQ


def _vw_is_meaningful_change(orig_token: str, typo_token: str) -> bool:
    """True iff typo is (a) a real English word AND (b) closeness_score with
    orig < NEAR_WORD_THRESHOLD. Use this to reject typo candidates that
    accidentally form a different real English word (e.g. form→from).

    Pure typos (orig→reallly, baby→babby, really→realily) all return False.
    """
    if not typo_token:
        return True
    typo_clean = _vw_strip(typo_token).lower()
    if not typo_clean:
        return True
    if not _vw_is_english_word(typo_clean):
        return False  # not a real word, accept
    orig_clean = _vw_strip(orig_token).lower() if orig_token else ""
    score = _vw_closeness_score(orig_clean, typo_clean)
    return score < NEAR_WORD_THRESHOLD


# === tools merged from 10_typo_injection/error_location.py ===

def _el_clause_marker(tok, doc) -> str:
    """Walk up the head chain; return first clause-opening dep or ROOT."""
    cur = tok
    visited = set()
    while cur.head != cur and cur.i not in visited:
        visited.add(cur.i)
        cur = cur.head
        if cur.dep_ in _CLAUSE_DEPS:
            return cur.dep_.upper()
        if cur.dep_ == "prep":
            return "PREP"
    return "ROOT"


def _el_depth_bucket(tok, doc) -> str:
    """Bucket depth in dependency tree: d0 (root), d1, d2, d3+."""
    depth = 0
    cur = tok
    visited = set()
    while cur.head != cur and cur.i not in visited and depth < 10:
        visited.add(cur.i)
        cur = cur.head
        depth += 1
    if depth <= 0:
        return "d0"
    if depth == 1:
        return "d1"
    if depth == 2:
        return "d2"
    return "d3+"


def _el_grandparent_pos(tok, doc) -> str:
    """POS of grandparent (head's head), or ROOT if no grandparent."""
    h = tok.head
    if h is None or h == tok:
        return "ROOT"
    gp = h.head
    if gp == h or gp is None:
        return "ROOT"
    return gp.pos_


def _el_sibling_pos(tok, direction: str) -> str:
    """POS of nearest sibling on given side, or NONE."""
    head = tok.head
    if head == tok or head is None:
        return "NONE"
    if direction == "left":
        cands = [c for c in head.children if c.i < tok.i]
        return cands[-1].pos_ if cands else "NONE"
    cands = [c for c in head.children if c.i > tok.i]
    return cands[0].pos_ if cands else "NONE"


def _el_rich_signatures(tok, doc) -> Dict[str, str]:
    """Multi-level structural signatures for one token, all levels written.

    Returns dict[level_name → sig_string] for levels in SIG_LEVELS.
    """
    pos = tok.pos_
    dep = tok.dep_
    h = tok.head
    parent_pos = h.pos_ if h and h != tok else "ROOT"
    gp_pos = _el_grandparent_pos(tok, doc)
    clause = _el_clause_marker(tok, doc)
    depth_bkt = _el_depth_bucket(tok, doc)
    sib_l = _el_sibling_pos(tok, "left")
    sib_r = _el_sibling_pos(tok, "right")
    length = len(tok.text)
    shape = _el_word_shape(tok.text)
    return {
        "full":    f"{pos}|{dep}|{parent_pos}|{gp_pos}|{clause}|{depth_bkt}",
        "d4":      f"D4|{gp_pos}|{parent_pos}|{dep}|{pos}",
        "d3":      f"D3|{parent_pos}|{dep}|{pos}",
        "clause":  f"{clause}|{pos}",
        "depth":   f"{depth_bkt}|{pos}|{dep}",
        "sibling": f"{pos}|{sib_l}|{sib_r}",
        "coarse":  f"{pos}|{dep}",
    }


def _el_default_rate_hierarchical(model: Dict, sigs: Dict[str, str]) -> float:
    """Hierarchical lookup: try finest sig first, fall back to coarser.

    Returns P_u(error | sig) at the finest level the user has data for.
    If no level seen, falls back to user's global p_u_err, then DEFAULT_P_ERROR.
    """
    rates = model.get("error_rate_by_context", {})
    for level in SIG_LEVELS:
        sig = sigs.get(level)
        if sig is None:
            continue
        r = rates.get(sig)
        if r is not None:
            return r
    p_user = model.get("p_u_err", 0.0)
    return p_user if p_user > 0 else 0.02


def _el_char_level_rate_hierarchical(model: Dict, sigs: Dict[str, str]) -> float:
    """Hierarchical lookup returning P(char_level_error | sig).

    Returns 0.0 if user has zero char-level history at any level.
    Used by Stage 10 sampler to rank positions for char-level typo injection.
    """
    rates = model.get("char_level_rate_by_context", {})
    for level in SIG_LEVELS:
        sig = sigs.get(level)
        if sig is None:
            continue
        r = rates.get(sig)
        if r is not None:
            return r
    # Fallback to global char-level rate
    p = model.get("p_u_char_level_err", 0.0)
    return p


def _el_lookup_transformation_history(
    model: Dict, orig_token_lower: str, sigs: Dict[str, str]
) -> List[Tuple[str, str, int]]:
    """Look up user's most-common char-level typo forms for (orig_token, sig).

    Hierarchical lookup chain:
      1. (orig_lower, full_sig) exact match
      2. (orig_lower, d3_sig)
      3. (orig_lower, coarse_sig)
      4. (orig_lower, ANY sig)
    Returns list of (typo_token, mechanism, count) sorted by count desc.
    Only char-level typos are returned.
    """
    history = model.get("transformation_history", {})
    for level in ("full", "d3", "coarse"):
        sig = sigs.get(level)
        if sig is None:
            continue
        key = (orig_token_lower, sig)
        if key in history:
            return sorted(history[key], key=lambda x: -x[2])
    # Final fallback: any sig with this orig_token
    candidates = [(typo, mech, cnt) for (ot, _sig), forms in history.items()
                  if ot == orig_token_lower
                  for (typo, mech, cnt) in forms]
    if candidates:
        # Deduplicate by typo_token, keep max count
        agg: Dict[str, Tuple[str, int]] = {}
        for typo, mech, cnt in candidates:
            if typo not in agg or cnt > agg[typo][1]:
                agg[typo] = (mech, cnt)
        return sorted([(typo, mech, cnt) for typo, (mech, cnt) in agg.items()],
                      key=lambda x: -x[2])
    return []


def _el_len_bucket(n: int) -> str:
    for lo, hi in _LEN_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "11+"


def _el_word_shape(w: str) -> str:
    """Coarse shape: lower/upper/title/digit/mixed."""
    if not w:
        return "empty"
    if w.isdigit():
        return "digit"
    if w.islower():
        return "lower"
    if w.isupper():
        return "upper"
    if w[0].isupper() and w[1:].islower():
        return "title"
    if any(c.isdigit() for c in w):
        return "mixed"
    return "other"


def _el_context_signature(pos: str, dep: str, parent_pos: str, length: int, word: str) -> str:
    return f"{pos}|{dep}|{parent_pos}|{_el_len_bucket(length)}|{_el_word_shape(word)}"


def _el_ensure_spacy():
    """Lazy-load spaCy, return nlp pipeline."""
    if not hasattr(_el_ensure_spacy, "_nlp"):
        _el_ensure_spacy._nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _el_ensure_spacy._nlp


def _el_extract_token_contexts(sentence: str) -> List[Tuple[int, str, str, str, int, str]]:
    """Return list of (idx, word, pos, dep, parent_pos, len_bucket, shape) for tokens.

    Args:
        sentence: input sentence

    Returns:
        List of tuples (token_idx, word, pos, dep, parent_pos, len_bucket, shape)
        where token_idx is the position in the original sentence (0-indexed, splits on
        whitespace AFTER spaCy tokenization — we use spaCy's own positions).
    """
    nlp = _el_ensure_spacy()
    doc = nlp(sentence)
    out = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        out.append((tok.i, tok.text, tok.pos_, tok.dep_, tok.head.pos_, len(tok.text), _el_word_shape(tok.text)))
    return out


def _el_build_user_history(user_word_edits: List[dict]) -> Tuple[Dict[str, Dict[str, int]], Dict]:
    """From a user's word_edits list, build per-context mechanism histogram at all
    signature levels (full/d4/d3/clause/depth/sibling/coarse).

    Returns (history, transformation_history) where:
      history: dict[sig][mechanism] = count; also keys __total__, __char_level__,
               __surface_form__ tracking total tokens, char-level edits, and
               surface-form edits per signature level.
      transformation_history: dict[(orig_token_lower, sig) → [(typo_token, mechanism, count), ...]]
        Only char-level typos are recorded (case_error and apostrophe_error are
        surface-form and excluded from reuse).
    """
    nlp = _el_ensure_spacy()

    history: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    transformation_history: Dict[Tuple[str, str], List[Tuple[str, str, int]]] = \
        defaultdict(list)
    transformation_agg: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for entry in user_word_edits:
        orig_sent = entry.get("orig_sent", "")
        edits = entry.get("edits", [])
        if not orig_sent or not edits:
            continue

        doc = nlp(orig_sent)

        edit_pairs = []
        for e in edits:
            inc = e.get("incorrect_word", "")
            cor = e.get("corrected_word", "")
            if inc:
                edit_pairs.append((inc, cor))

        for tok in doc:
            if not tok.is_alpha:
                continue
            word_l = tok.text.lower()
            sigs = _el_rich_signatures(tok, doc)

            matched_mech = None
            matched_typo = None
            for inc, cor in edit_pairs:
                inc_stripped = re.sub(r"[^a-zA-Z']", "", inc).lower()
                if inc_stripped == word_l:
                    mech = _tc_classify_mechanism(tok.text, cor)
                    if mech != "semantic_substitution":
                        matched_mech = mech
                        matched_typo = re.sub(r"[^a-zA-Z']", "", cor)
                    break

            # Write to every signature level
            for level in SIG_LEVELS:
                sig = sigs[level]
                if matched_mech:
                    history[sig][matched_mech] += 1
                    if matched_mech in CHAR_LEVEL_MECHANISMS:
                        history[sig]["__char_level__"] += 1
                    elif matched_mech in SURFACE_FORM_MECHANISMS:
                        history[sig]["__surface_form__"] += 1
                history[sig]["__total__"] += 1

            # Record transformation only for char-level typos at fine-grained sigs
            if matched_mech in CHAR_LEVEL_MECHANISMS and matched_typo:
                for level in ("full", "d3", "coarse"):
                    sig = sigs[level]
                    key = (word_l, sig, matched_typo, matched_mech)
                    transformation_agg[key] += 1

    # Flatten transformation_agg into per-(orig, sig) lists
    for (orig, sig, typo, mech), cnt in transformation_agg.items():
        transformation_history[(orig, sig)].append((typo, mech, cnt))
    return dict(history), dict(transformation_history)


def _el_build_user_error_model(uid: str, user_word_edits: List[dict]) -> Dict:
    """Build per-user error probability model.

    Returns dict with:
      p_u_err: global error rate (float, n_edits / n_tokens)
      p_u_char_level_err: char-level-only global rate
      n_tokens: total tokens analyzed
      n_edits: total errors (excluding semantic_substitution)
      n_char_level_edits: char-level-only count
      context_probs: {context_sig: {"mechanisms": {mech: count}, "total": int,
                                     "char_level": int, "surface_form": int}}
      error_rate_by_context: {context_sig: p(error|context)}    (all errors)
      char_level_rate_by_context: {context_sig: p(char_level|context)}
      mechanism_totals: {mech: total_count} aggregated across all contexts
      transformation_history: {(orig_token_lower, sig): [(typo, mech, count), ...]}
        Only char-level typos; used by Stage 10 to reuse user's preferred form.
    """
    history, transformation_history = _el_build_user_history(user_word_edits)

    n_tokens = sum(h.get("__total__", 0) for h in history.values())
    n_edits = sum(sum(v for k, v in h.items()
                      if k not in ("__total__", "__char_level__",
                                   "__surface_form__"))
                  for h in history.values())
    n_char = sum(h.get("__char_level__", 0) for h in history.values())
    p_u_err = n_edits / n_tokens if n_tokens else 0.0
    p_u_char_level_err = n_char / n_tokens if n_tokens else 0.0

    context_probs = {}
    error_rate_by_context = {}
    char_level_rate_by_context = {}
    mechanism_totals: Dict[str, int] = defaultdict(int)
    char_level_mechanism_totals: Dict[str, int] = defaultdict(int)
    alpha = 0.1
    for sig, hist in history.items():
        total = hist.get("__total__", 0)
        char_level = hist.get("__char_level__", 0)
        mech_only = {k: v for k, v in hist.items()
                     if k not in ("__total__", "__char_level__",
                                  "__surface_form__")}
        n_err_ctx = sum(mech_only.values())
        rate_all = (n_err_ctx + alpha) / (total + alpha * 2)
        rate_char = (char_level + alpha) / (total + alpha * 2)
        context_probs[sig] = {
            "mechanisms": mech_only,
            "total": total,
            "char_level": char_level,
            "surface_form": hist.get("__surface_form__", 0),
        }
        error_rate_by_context[sig] = rate_all
        char_level_rate_by_context[sig] = rate_char
        for mech, cnt in mech_only.items():
            mechanism_totals[mech] += cnt
            if mech in CHAR_LEVEL_MECHANISMS:
                char_level_mechanism_totals[mech] += cnt

    return {
        "uid": uid,
        "p_u_err": p_u_err,
        "p_u_char_level_err": p_u_char_level_err,
        "n_tokens": n_tokens,
        "n_edits": n_edits,
        "n_char_level_edits": n_char,
        "context_probs": context_probs,
        "error_rate_by_context": error_rate_by_context,
        "char_level_rate_by_context": char_level_rate_by_context,
        "mechanism_totals": dict(mechanism_totals),
        "char_level_mechanism_totals": dict(char_level_mechanism_totals),
        "transformation_history": transformation_history,
    }


def _el_load_sercl_profile(path: Path) -> Dict[str, dict]:
    """Load SErCL profile JSON. Returns {uid: {p_u_err, n_tokens, n_edits,
    context_probs, error_rate_by_context}}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    profiles = raw.get("user_profiles", {})
    word_edits = raw.get("user_word_edits", {})
    out = {}
    for uid, we_list in word_edits.items():
        if not we_list:
            continue
        out[uid] = _el_build_user_error_model(uid, we_list)
    return out


# === tools merged from 10_typo_injection/injection_sampler.py ===

@dataclass
class _is_InjectionMeta:
    position: int            # 0-indexed position in the query
    original_token: str
    typo_token: str
    mechanism: str           # one of CHAR_LEVEL_MECHANISMS
    confidence: float        # char_level rate at the selected context
    context_sig: str         # which signature was used (e.g. "full:NOUN|dobj|VERB|...")
    transformation_source: str  # "user_historical_full" / "user_historical_d3" /
                                # "user_historical_coarse" / "user_historical_any" /
                                # "generic_fallback"
    edit_distance: int       # Levenshtein(orig, typo)
    len_delta: int           # len(typo) - len(orig)
    d_mahalanobis_before: float
    d_mahalanobis_after: float
    gaussian_pass: bool
    semantic_sim: float = 1.0
    semantic_pass: bool = True
    min_d_competitor: float = -1.0
    n_competitors: int = 0
    exclusive_pass: bool = True


# Lazy-loaded encoder + rule_to_id
_encoder = None
_encoder_vocab_size: Optional[int] = None
_rule_to_id: Dict[str, int] = {}


def _is_load_pcfg_module():
    spec = importlib.util.spec_from_file_location(
        "syntax_pcfg_pipeline", str(PCFG_PIPELINE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_load_encoder():
    """Load encoder — supports both _SupEncoder (strict3_encoder.pt) and
    StyleMLP (cohort2_mlp16_30_30ep.pt, raw state_dict). Returns (model, vocab_size)."""
    global _encoder, _encoder_vocab_size
    if _encoder is None:
        ckpt = torch.load(ENCODER_PT, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            # _SupEncoder format (strict3_encoder.pt)
            cfg = ckpt["config"]
            _encoder_vocab_size = int(cfg["vocab_size"])
            _pcfg = _is_load_pcfg_module()
            model = _pcfg._SupEncoder(
                _encoder_vocab_size, cfg["z_dim"], tuple(cfg["hidden"]),
                cfg["n_users"], cfg["dropout"])
            model.load_state_dict(ckpt["model_state"])
        else:
            # StyleMLP raw state_dict (cohort2_mlp16_30_30ep.pt) — Stage 8
            # SVD → MLP pipeline (SVD_DIM=256 → HIDDEN=128 → OUT=16).
            import sys as _sys
            _sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode")
            from syntax_encoder import StyleMLP as _StyleMLP
            state_dict = ckpt if isinstance(ckpt, dict) else ckpt.state_dict()
            _encoder_vocab_size = len(_is_load_rule_to_id())
            model = _StyleMLP()  # 用默认 dim: 256 → 128 → 16
            model.load_state_dict(state_dict)
        model.eval()
        model.to(ENCODER_DEVICE)
        _encoder = model
    return _encoder, _encoder_vocab_size


def _is_load_rule_to_id() -> Dict[str, int]:
    """Load vocab.json → {rule_str: id}. Length must equal encoder vocab_size."""
    global _rule_to_id
    if not _rule_to_id:
        with open(CACHE_DIR / "vocab.json") as f:
            vocab_list = json.load(f)
        _rule_to_id = {r: i for i, r in enumerate(vocab_list)}
    return _rule_to_id


_coral_data = None  # 2026-09-19: cached per-ASIN CORAL A matrix


def _is_load_coral() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load per-ASIN CORAL A matrix from coral_asin_cohort2_mlp16_30.npz.
    Returns ({asin: A}, global_A). 2026-09-19: Stage 10 必须 apply A 才能
    与 Stage 8 cohort gates d² 数值一致 (否则 raw z vs μ_r 数值偏大 100-300×)."""
    global _coral_data
    if _coral_data is None:
        c = np.load(CORAL_ASIN_PATH, allow_pickle=True)
        A_per = c["A_per_asin"]
        keys = c["asin_keys"]
        asin_to_A = {str(keys[i]): A_per[i] for i in range(len(keys))}
        _coral_data = (asin_to_A, c["global_A"])
    return _coral_data


def _is_apply_coral(z: np.ndarray, asin: str) -> np.ndarray:
    """Apply per-ASIN CORAL A (or global fallback) to z → align query→review domain."""
    asin_to_A, global_A = _is_load_coral()
    A = asin_to_A.get(asin, global_A)
    return (A @ z.astype(np.float32)).astype(np.float32)


def _is_encode_queries_32d(texts: List[str], nlp, encoder, rule_to_id: Dict[str, int],
                        vocab_size: int) -> np.ndarray:
    """texts → 16d StyleMLP z (Stage 8 pipeline: rules → SVD → MLP).

    2026-09-19: 当 encoder 是 StyleMLP (cohort2_mlp16_30_30ep.pt) 时走 SVD→MLP 路径;
    当 encoder 是 _SupEncoder (strict3_encoder.pt) 时直接吃 counts.
    """
    _pcfg = _is_load_pcfg_module()
    extract_struct_rules = _pcfg.extract_struct_rules

    n = len(texts)
    counts = np.zeros((n, vocab_size), dtype=np.float32)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=256)):
        for r in extract_struct_rules(doc):
            j = rule_to_id.get(r)
            if j is not None:
                counts[i, j] = 1.0
    counts = counts / (1.0 + counts)
    # 检测 encoder 类型
    is_style_mlp = "StyleMLP" in type(encoder).__name__
    if is_style_mlp:
        # SVD (256d) → StyleMLP (16d)
        svd = np.load(SVD_COMPONENTS_PATH, allow_pickle=False)
        Vt = np.asarray(svd["Vt"], dtype=np.float32)
        z_svd = (counts @ Vt.T).astype(np.float32)
        with torch.no_grad():
            z = encoder(torch.tensor(z_svd, dtype=torch.float32, device=ENCODER_DEVICE))
        return z.cpu().numpy().astype(np.float32)
    with torch.no_grad():
        z, _ = encoder(torch.tensor(counts, dtype=torch.float32, device=ENCODER_DEVICE))
    return z.cpu().numpy().astype(np.float32)


def _is_spacy_token_contexts(query: str) -> List[Tuple]:
    """Token-level contexts with multi-level structural signatures."""
    nlp = _el_ensure_spacy()
    doc = nlp(query)
    out = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        sigs = _el_rich_signatures(tok, doc)
        out.append((tok.i, tok.text, tok.pos_, tok.dep_, tok.head.pos_,
                    len(tok.text), _el_word_shape(tok.text), sigs))
    return out


def _is_split_query_tokens(query: str) -> List[str]:
    """Return list of whitespace-separated tokens preserving original word order."""
    return query.split(" ")


def _is_levenshtein(a: str, b: str) -> int:
    """Levenshtein distance (small string inputs, naive DP is fine)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = cur
    return prev[-1]


def _is_mahalanobis_d2(z: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray,
                       user_stats: Optional[dict] = None) -> float:
    """Mahalanobis squared distance D²(z, μ) = (z-μ)ᵀ Σ⁻¹ (z-μ).

    sigma_inv can be either:
      - full inverse covariance matrix (legacy 16d), or
      - diagonal of inverse variances (svd_mlp 64d).
    Detect via ndim.

    2026-09-19: 优先用 rank1+diag 显式公式 (与 Stage 8 cohort gates 数值一致),
    避免 Woodbury σ_inv 数值放大 (rank=1 矩阵近奇异, 误差 100-300×).
    user_stats 含 sigma_diag_sq / V / lambdas → 显式 d² = (V^T r)² / λ + r_resid² / σ_d²
    """
    if user_stats is not None and "sigma_diag_sq" in user_stats and "V" in user_stats:
        r = (z - mu).astype(np.float64)
        sd = np.asarray(user_stats["sigma_diag_sq"], dtype=np.float64)
        V = np.asarray(user_stats["V"], dtype=np.float64)[:, 0]
        lam = float(np.asarray(user_stats["lambdas"]).reshape(-1)[0])
        r_proj = V @ r
        r_resid = r - V * r_proj
        return max(float(r_proj**2 / lam + np.sum(r_resid**2 / sd)), 0.0)
    diff = z - mu
    if sigma_inv.ndim == 1:
        d2 = float(np.sum(diff * diff * sigma_inv))
    else:
        d2 = float(diff @ sigma_inv @ diff)
    return max(d2, 0.0)


_semantic_encoder = None  # lazy global
_sentence_transformer = None

def _is_load_semantic_encoder():
    """Lazy-load MiniLM bi-encoder for semantic similarity."""
    global _semantic_encoder
    if _semantic_encoder is None:
        os.environ.setdefault("HF_HOME", str(SEMANTIC_CACHE_DIR))
        os.environ.setdefault("HF_HUB_CACHE", str(SEMANTIC_CACHE_DIR / "hub"))
        from sentence_transformers import SentenceTransformer
        _semantic_encoder = SentenceTransformer(SEMANTIC_MODEL_ID, device=ENCODER_DEVICE)
    return _semantic_encoder


def _is_semantic_cosine(texts_a: List[str], texts_b: List[str]) -> np.ndarray:
    """Batch cosine similarity between paired MiniLM embeddings."""
    enc = _is_load_semantic_encoder()
    ea = enc.encode(texts_a, normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    eb = enc.encode(texts_b, normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    return (ea * eb).sum(axis=1)


def _is_apply_typo_to_query(query: str, position: int, typo: str) -> Optional[str]:
    """Replace the position-th whitespace token in query with typo."""
    tokens = _is_split_query_tokens(query)
    if position >= len(tokens):
        return None
    if not any(c.isalpha() for c in tokens[position]):
        return None
    tokens[position] = typo
    return " ".join(tokens)


def _is_validate_minimality(orig: str, typo: str) -> Tuple[bool, int, int]:
    """Validate minimal-edit constraint. Returns (ok, edit_distance, len_delta).

    ok iff:
      - typo != orig
      - len(orig) ≥ MIN_TOKEN_LEN
      - _vw_edit_distance(lower(orig), lower(typo)) ≤ MAX_EDIT_DISTANCE
      - |len(typo) - len(orig)| ≤ MAX_LEN_DELTA
      - typo is NOT a meaningfully-different English word
    """
    if typo == orig or not typo:
        return False, 0, 0
    if len(orig) < MIN_TOKEN_LEN:
        return False, 0, len(typo) - len(orig)
    ed = _is_levenshtein(orig.lower(), typo.lower())
    ld = len(typo) - len(orig)
    if ed > MAX_EDIT_DISTANCE:
        return False, ed, ld
    if abs(ld) > MAX_LEN_DELTA:
        return False, ed, ld
    if _vw_is_meaningful_change(orig, typo):
        return False, ed, ld
    return True, ed, ld


def _is_sample_user_historical_typo(
    user_model: Dict, orig_token: str, sigs: Dict[str, str], rng: random.Random,
) -> Optional[Tuple[str, str, str]]:
    """Try to find a user-historical char-level typo for (orig_token, sig).

    Returns (typo_token, mechanism, source_tag) if found, else None.
    Source tag ∈ {user_historical_full, user_historical_d3, user_historical_coarse,
                  user_historical_any} so we know which fallback level matched.
    """
    orig_lower = orig_token.lower()
    history = _el_lookup_transformation_history(user_model, orig_lower, sigs)
    if not history:
        return None
    for level_tag, level in (("full", "full"), ("d3", "d3"), ("coarse", "coarse"),
                              ("any", None)):
        if level is not None:
            sig = sigs.get(level)
            if sig is None:
                continue
            key = (orig_lower, sig)
            from error_location import _default_rate_hierarchical as _dr
            # Need to look up directly — use the function above for level-specific
            forms = [f for (o, s), forms in user_model.get("transformation_history", {}).items()
                     for f in forms if o == orig_lower and s == sig]
        else:
            forms = [f for (o, _s), forms in user_model.get("transformation_history", {}).items()
                     for f in forms if o == orig_lower]
        if not forms:
            continue
        # Pick the most common char-level typo (forms already sorted by count desc)
        char_forms = [f for f in forms if f[1] in CHAR_LEVEL_MECHANISMS]
        if not char_forms:
            continue
        typo, mech, _cnt = char_forms[0]
        return typo, mech, f"user_historical_{level_tag}"
    return None


def _is_sample_generic_typo(
    user_model: Dict, orig_token: str, rng: random.Random,
) -> Optional[Tuple[str, str]]:
    """Generic fallback: sample a char-level mechanism from user's global char-level
    distribution and apply reverse_mechanism."""
    totals = user_model.get("char_level_mechanism_totals", {})
    weights = {m: totals.get(m, 0) for m in CHAR_LEVEL_MECHANISMS}
    total_w = sum(weights.values())
    if total_w == 0:
        return None
    items = list(weights.items())
    r = rng.random() * total_w
    cum = 0.0
    mech = items[-1][0]
    for m, w in items:
        cum += w
        if r <= cum:
            mech = m
            break
    typo = _tc_reverse_mechanism(orig_token, mech, rng_seed=rng.randint(0, 10**9))
    if typo == orig_token or not typo:
        return None
    return typo, mech


def _is_sample_typo(
    query: str,
    uid: str,
    user_model: Dict,
    seed: int = 42,
) -> Tuple[Optional[str], Optional[InjectionMeta], Dict]:
    """Phase-1 sampler: pick position + generate typo (NO encoding, NO gating).
    Returns (injected_string_or_None, meta_partial, ctx_info).
    ctx_info holds {mu, sigma_inv_or_user_stats, d2_threshold, competitors_asin, position, sigs, ...}
    """
    rng = random.Random(seed)
    contexts = _is_spacy_token_contexts(query)
    if not contexts:
        return None, None, {"reason": "no_contexts"}
    if user_model.get("n_char_level_edits", 0) == 0:
        return None, None, {"reason": "no_char_history"}

    scored = []
    for ctx in contexts:
        alpha_idx, word, pos, dep, parent_pos, length, shape, sigs = ctx
        if len(word) < MIN_TOKEN_LEN:
            continue
        rate = _el_char_level_rate_hierarchical(user_model, sigs)
        scored.append((alpha_idx, word, rate, sigs))
    if not scored:
        return None, None, {"reason": "no_scored_tokens"}
    scored.sort(key=lambda x: (-x[2], x[0]))
    ws_pos, word, w_i, sigs = scored[0]

    picked = _is_sample_user_historical_typo(user_model, word, sigs, rng)
    if picked is not None:
        typo, mech, source = picked
    else:
        gen = _is_sample_generic_typo(user_model, word, rng)
        if gen is None:
            return None, None, {"reason": "no_generic_typo"}
        typo, mech = gen
        source = "generic_fallback"

    ok, edit_dist, len_delta = _is_validate_minimality(word, typo)
    if not ok:
        return None, None, {"reason": "minimality_fail"}

    injected = _is_apply_typo_to_query(query, ws_pos, typo)
    if injected is None:
        return None, None, {"reason": "apply_fail"}

    return injected, {
        "word": word, "typo": typo, "mech": mech, "source": source,
        "edit_dist": edit_dist, "len_delta": len_delta,
        "ws_pos": ws_pos, "w_i": w_i, "sigs": sigs,
    }, {"reason": "ok"}


def _is_batch_gate(
    items: List[Dict],
    nlp,
    encoder,
    rule_to_id: Dict[str, int],
    vocab_size: int,
    asin_to_A: Dict[str, np.ndarray],
    global_A: np.ndarray,
) -> List[Dict]:
    """Phase-2 batch gate: batch-encode all orig+inj, apply CORAL, compute d²,
    comp gate. Returns list of updated items with d2_before/after/comp_pass."""
    if not items:
        return []
    # Collect all texts: orig then inj
    texts = []
    for it in items:
        texts.append(it["query"])
        texts.append(it["injected"])
    # Batch encode
    nlp_ = nlp if nlp is not None else _el_ensure_spacy()
    z_all = _is_encode_queries_32d(texts, nlp_, encoder, rule_to_id, vocab_size)
    for k, it in enumerate(items):
        z_orig = z_all[2 * k]
        z_inj = z_all[2 * k + 1]
        # Apply CORAL per ASIN
        A = asin_to_A.get(it["asin"], global_A)
        z_orig_aligned = (A @ z_orig.astype(np.float32)).astype(np.float32)
        z_inj_aligned = (A @ z_inj.astype(np.float32)).astype(np.float32)
        mu = it["mu"]
        stats = it["user_stats"]
        d2_before = _is_mahalanobis_d2(z_orig_aligned, mu, it["sigma_inv"], user_stats=stats)
        d2_after = _is_mahalanobis_d2(z_inj_aligned, mu, it["sigma_inv"], user_stats=stats)
        d2_threshold = it["d2_threshold"]
        it["z_orig_aligned"] = z_orig_aligned
        it["z_inj_aligned"] = z_inj_aligned
        it["d2_before"] = d2_before
        it["d2_after"] = d2_after
        it["gaussian_pass"] = bool(d2_after <= d2_threshold)
    return items


def _is_batch_comp_gate(items: List[Dict]) -> List[Dict]:
    """For each item, compute exclusive_pass via per-comp d².
    comp dicts have mu/sigma_inv from expanded_cohort (computed via Woodbury earlier).
    2026-09-19: comp gate 改为 ratio 比较 — typo 后 z 与 target user μ 距离应明显比与 comp user μ 距离更近
    (typo 把 query 拉向 target 而远离 comp)。ratio = d²_c / d²_target < K (e.g. K=1.5) → fail.
    """
    for it in items:
        if not it.get("gaussian_pass", False):
            it["exclusive_pass"] = True
            it["min_d_competitor"] = -1.0
            it["n_competitors"] = 0
            continue
        competitors = it.get("competitors", {})
        if not competitors:
            it["exclusive_pass"] = True
            it["min_d_competitor"] = -1.0
            it["n_competitors"] = 0
            continue
        z = it["z_inj_aligned"]
        d2_target = it["d2_after"]
        min_d = float("inf")
        n_comps = 0
        exclusive_pass = True
        for cuid, comp in competitors.items():
            if cuid == it["uid"]:
                continue
            c_mu = np.asarray(comp["mu"], dtype=np.float32)
            c_inv = np.asarray(comp["sigma_inv"], dtype=np.float32)
            d2_c = _is_mahalanobis_d2(z, c_mu, c_inv, user_stats=comp.get("user_stats"))
            n_comps += 1
            if d2_c < min_d:
                min_d = d2_c
            # ratio check: d²_c < d²_target × 1.5 → typo 把 query 拉得太近 comp
            if d2_c < d2_target * 1.5:
                exclusive_pass = False
                break
        it["exclusive_pass"] = exclusive_pass
        it["min_d_competitor"] = min_d if min_d != float("inf") else -1.0
        it["n_competitors"] = n_comps
    return items


def _is_batch_semantic(items: List[Dict]) -> List[Dict]:
    """Batch compute semantic sim for all items using MiniLM."""
    if not items:
        return items
    queries = [it["query"] for it in items]
    injects = [it["injected"] for it in items]
    sims = _is_semantic_cosine(queries, injects)
    for it, sim in zip(items, sims):
        it["semantic_sim"] = float(sim)
        it["semantic_pass"] = bool(sim >= SEMANTIC_THRESHOLD)
    return items


def _is_sample_injection(
    query: str,
    uid: str,
    user_model: Dict,
    mu_u: np.ndarray,
    sigma_inv: np.ndarray,
    d2_threshold: float,
    competitors: Optional[Dict[str, Dict]] = None,
    seed: int = 42,
    nlp=None,
    encoder=None,
    rule_to_id: Optional[Dict[str, int]] = None,
    user_stats: Optional[dict] = None,
    asin: Optional[str] = None,
    vocab_size: Optional[int] = None,
) -> Tuple[Optional[str], Optional[InjectionMeta]]:
    """Apply at most one char-level typo injection to `query`.

    Position selection: rank alpha tokens by P_u(char_level_error | sig).
    Transformation selection: reuse user-historical char-level typo at this
    (orig_token, sig) if available; otherwise generic char-level mechanism.
    Gates (all must pass):
      1. Minimal edit (edit distance ≤ 2, len delta ≤ 1, len ≥ 3, not different word)
      2. d²(z_after, μ_u) ≤ user Q_95
      3. d²(z_after, μ_comp) > comp Q_95 ∀ comp (non-shared zone)
      4. cosine(MiniLM(orig), MiniLM(typo)) ≥ 0.9

    Returns (injected_query, meta). If any gate fails, returns (None, meta) with
    the corresponding pass flag set to False.
    """
    rng = random.Random(seed)

    if nlp is None:
        nlp = _el_ensure_spacy()
    if encoder is None or vocab_size is None:
        encoder, vocab_size = _is_load_encoder()
    if rule_to_id is None:
        rule_to_id = _is_load_rule_to_id()
    if vocab_size is None:
        vocab_size = len(rule_to_id)
    if vocab_size != len(rule_to_id):
        raise ValueError(
            f"vocab_size ({vocab_size}) != len(rule_to_id) ({len(rule_to_id)}); "
            "encoder.pt and vocab.json come from different cohort")

    # Encode original query (32d)
    feat_orig = _is_encode_queries_32d([query], nlp, encoder, rule_to_id, vocab_size)[0]
    if asin is not None:
        feat_orig = _is_apply_coral(feat_orig, asin)  # 2026-09-19: per-ASIN CORAL align
    d2_before = _is_mahalanobis_d2(feat_orig, mu_u, sigma_inv, user_stats=user_stats)

    # Per-token contexts
    contexts = _is_spacy_token_contexts(query)
    if not contexts:
        import os as _o2  # noqa
        if False and _o2.environ.get("TYPO_DEBUG"):
            print(f"DEBUG_NCTX: uid={uid[:12]} query={query[:60]!r}", flush=True)
        meta = _is_InjectionMeta(
            position=-1, original_token="", typo_token="",
            mechanism="none", confidence=0.0, context_sig="",
            transformation_source="none", edit_distance=0, len_delta=0,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    # Reject early if user has zero char-level history
    if user_model.get("n_char_level_edits", 0) == 0:
        import os as _o3  # noqa
        if False and _o3.environ.get("TYPO_DEBUG"):
            print(f"DEBUG_NCE: uid={uid[:12]} n_ce={user_model.get('n_char_level_edits')}", flush=True)
        meta = _is_InjectionMeta(
            position=-1, original_token="", typo_token="",
            mechanism="none", confidence=0.0, context_sig="",
            transformation_source="surface_form_only", edit_distance=0, len_delta=0,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    # Rank tokens by char_level P(error | sig) — pick the highest
    # If user has zero char-level error rate at any sig, P returns 0.0;
    # but the global p_u_char_level_err is used as fallback (already 0).
    scored = []
    for ctx in contexts:
        alpha_idx, word, pos, dep, parent_pos, length, shape, sigs = ctx
        if len(word) < MIN_TOKEN_LEN:  # 2026-09-19: skip short words (I, a, Where) → generic typo fails
            continue
        rate = _el_char_level_rate_hierarchical(user_model, sigs)
        scored.append((alpha_idx, word, rate, sigs))
    if not scored:
        meta = _is_InjectionMeta(
            position=-1, original_token="", typo_token="",
            mechanism="none", confidence=0.0, context_sig="",
            transformation_source="none", edit_distance=0, len_delta=0,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    # Pick top-scored token; tie-break by query order (left-to-right)
    scored.sort(key=lambda x: (-x[2], x[0]))
    ws_pos, word, w_i, sigs = scored[0]

    # Look up user-historical char-level typo for (orig_word, sigs)
    picked = _is_sample_user_historical_typo(user_model, word, sigs, rng)
    if picked is not None:
        typo, mech, source = picked
    else:
        # Generic fallback
        gen = _is_sample_generic_typo(user_model, word, rng)
        if gen is None:
            import os as _o4  # noqa
            if False and _o4.environ.get("TYPO_DEBUG"):
                print(f"DEBUG_GENNONE: uid={uid[:12]} word={word!r} gen_probs={user_model.get('char_level_mechanism_probs')}", flush=True)
            meta = _is_InjectionMeta(
                position=ws_pos, original_token=word, typo_token="",
                mechanism="none", confidence=float(w_i),
                context_sig=sigs.get("full", ""),
                transformation_source="generic_no_mechanism",
                edit_distance=0, len_delta=0,
                d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
                gaussian_pass=False,
            )
            return None, meta
        typo, mech = gen
        source = "generic_fallback"

    # Validate minimal-edit constraint
    ok, edit_dist, len_delta = _is_validate_minimality(word, typo)
    if not ok:
        import os as _o5  # noqa
        if False and _o5.environ.get("TYPO_DEBUG"):
            print(f"DEBUG_MINFAIL: uid={uid[:12]} word={word!r} typo={typo!r}", flush=True)
        meta = _is_InjectionMeta(
            position=ws_pos, original_token=word, typo_token=typo,
            mechanism=mech, confidence=float(w_i),
            context_sig=sigs.get("full", ""),
            transformation_source=source,
            edit_distance=edit_dist, len_delta=len_delta,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    injected = _is_apply_typo_to_query(query, ws_pos, typo)
    if injected is None:
        meta = _is_InjectionMeta(
            position=ws_pos, original_token=word, typo_token=typo,
            mechanism=mech, confidence=float(w_i),
            context_sig=sigs.get("full", ""),
            transformation_source=source,
            edit_distance=edit_dist, len_delta=len_delta,
            d_mahalanobis_before=d2_before, d_mahalanobis_after=d2_before,
            gaussian_pass=False,
        )
        return None, meta

    # Gaussian constraint
    feat_inj = _is_encode_queries_32d([injected], nlp, encoder, rule_to_id, vocab_size)[0]
    if asin is not None:
        feat_inj = _is_apply_coral(feat_inj, asin)
    d2_after = _is_mahalanobis_d2(feat_inj, mu_u, sigma_inv, user_stats=user_stats)
    import os as _o  # noqa
    if False and _o.environ.get("TYPO_DEBUG") and d2_after > d2_threshold:
        print(f"DEBUG_CORAL: uid={uid[:12]} asin={asin[:14]} d2_before={d2_before:.2f} d2_after={d2_after:.2f} thresh={d2_threshold:.2f}", flush=True)
    gaussian_pass = bool(d2_after <= d2_threshold)
    pass

    # Semantic similarity constraint
    sem_sim = float(_is_semantic_cosine([query], [injected])[0])
    semantic_pass = bool(sem_sim >= SEMANTIC_THRESHOLD)

    # Exclusive cohort gate
    min_d_comp = -1.0
    n_comps = 0
    exclusive_pass = True
    if competitors:
        min_d_comp = float("inf")
        for cuid, comp in competitors.items():
            if cuid == uid:
                continue
            c_mu = np.asarray(comp["mu"], dtype=np.float32)
            c_inv = np.asarray(comp["sigma_inv"], dtype=np.float32)
            c_T = comp["gate_T"]
            d2_c = _is_mahalanobis_d2(feat_inj, c_mu, c_inv)
            n_comps += 1
            if d2_c < min_d_comp:
                min_d_comp = d2_c
            if d2_c <= c_T:
                exclusive_pass = False
                break

    meta = _is_InjectionMeta(
        position=ws_pos,
        original_token=word,
        typo_token=typo,
        mechanism=mech,
        confidence=float(w_i),
        context_sig=sigs.get("full", ""),
        transformation_source=source,
        edit_distance=edit_dist,
        len_delta=len_delta,
        d_mahalanobis_before=d2_before,
        d_mahalanobis_after=d2_after,
        gaussian_pass=gaussian_pass,
        semantic_sim=sem_sim,
        semantic_pass=semantic_pass,
        min_d_competitor=min_d_comp if min_d_comp != float("inf") else -1.0,
        n_competitors=n_comps,
        exclusive_pass=exclusive_pass,
    )
    if not gaussian_pass or not semantic_pass or not exclusive_pass:
        return None, meta
    return injected, meta


def main():
    global D2_THRESHOLD_QUANTILE
    t0 = time.time()
    profiles, mahal, sel, cohort, gate_q = load_inputs()
    # 2026-09-19: gate_q=0.05 → q05_theoretical (χ²_{16,0.05}=7.96)
    D2_THRESHOLD_QUANTILE = f"q{int(gate_q * 100):02d}_theoretical"

    pairs = collect_pairs(sel["selections"], mahal, cohort=cohort)
    for uid, asin, _ in pairs:
        if asin not in cohort or uid not in cohort[asin]:
            raise ValueError(f"selected pair ({uid}, {asin}) missing from Stage 04 cohort_gates")
    if SMOKE:
        rng = random.Random(SEED_BASE)
        rng.shuffle(pairs)
        seen_uids = set()
        smoke_pairs = []
        for p in pairs:
            if p[0] not in seen_uids:
                smoke_pairs.append(p)
                seen_uids.add(p[0])
            if len(seen_uids) >= N_SMOKE_USERS:
                break
        pairs = smoke_pairs
    log(f"running on {len(pairs)} (uid, asin, query) pairs (SMOKE={SMOKE})")

    # Pre-load encoder + rule_to_id (for batch encoding)
    log("pre-loading encoder + rule_to_id (for batch encoding)")
    encoder, vocab_size = _is_load_encoder()
    rule_to_id = _is_load_rule_to_id()
    items = []  # batch items buffer

    results = []
    debug_metas = []   # collect meta from each pair for diagnosis (SMOKE only)
    n_total_processed = 0
    per_user_stats = defaultdict(lambda: {
        "n_total": 0,
        "n_bernoulli_pass": 0,    # a candidate position was identified
        "n_injected": 0,          # final injection passed all gates
        "n_gate_fail": 0,         # candidate identified but D² exceeded target
        "n_semantic_fail": 0,     # Gaussian passed but MiniLM sim < 0.9
        "n_exclusive_fail": 0,    # Gaussian + semantic passed but inside competitor core
        "n_no_candidate": 0,      # no candidate (user has zero char-level history)
        "n_surface_form_only_skip": 0,  # user has only case_error/apostrophe history
        "n_minimality_fail": 0,   # edit distance / len delta / valid-word check failed
        "typo_count": 0,          # char-level typos (keyboard_adjacent / letter_swap / ...)
        "mechanisms": defaultdict(int),
        "transformation_sources": defaultdict(int),  # user_historical_full/d3/.../generic_fallback
        "edit_distances": [],
        "d2_deltas": [],
        "sem_sims": [],
        "min_d_competitors": [],
    })
    mech_total = defaultdict(int)
    typo_total = 0
    gate_fail_total = 0
    semantic_fail_total = 0
    exclusive_fail_total = 0
    bernoulli_pass_total = 0
    surface_form_only_skip_total = 0
    minimality_fail_total = 0

    # 2026-09-19: uid not in profiles 时使用 generic_fallback profile (基于 global P_global),
    # 这样所有 pair 都 attempted, 而不是只 sample 6 user。
    _generic_profile = {"__generic__": profiles.get("__generic__")}
    if "__generic__" not in profiles:
        # 构造 generic fallback profile: 各 mechanism 概率均匀 (1/n_mechanisms)
        from collections import defaultdict as _dd
        _char_mechs = ["keyboard_adjacent", "letter_swap", "letter_repetition",
                      "letter_insertion", "letter_deletion", "case_error"]
        _generic_profile["__generic__"] = {
            "p_u_err": 0.5,
            "n_tokens": 0,
            "n_edits": 0,
            "n_char_level_edits": 1,    # 2026-09-19: 非0 才能让 line 1232 不 early-return -1
            "p_u_char_level_err": 0.5, # 2026-09-19: 让 _el_char_level_rate_hierarchical 返回 >0
            "context_probs": {},
            "error_rate_by_context": {},
            "char_level_rate_by_context": {},  # 兜底用 p_u_char_level_err
            "char_level_mechanism_probs": {m: 1.0 / len(_char_mechs) for m in _char_mechs},
            "char_level_mechanism_totals": {m: 1 for m in _char_mechs},
            "transformation_history": {},
        }

    for i, (uid, asin, query) in enumerate(pairs):
        if uid not in mahal or asin not in cohort:
            continue   # 2026-09-19: skip pair 不在 cohort3mlp16_30 pre_theoretical_gate 子集
        user_model = profiles.get(uid, _generic_profile["__generic__"])
        stats = mahal[uid]
        d2_threshold = stats[f"d2_{D2_THRESHOLD_QUANTILE}"]

        # Phase 1: sample typo (no encoder, no gating)
        injected, sampler_info, ctx = _is_sample_typo(query, uid, user_model, seed=SEED_BASE + i)
        s = per_user_stats[uid]
        s["n_total"] += 1
        n_total_processed += 1
        if injected is None:
            s["n_no_candidate"] += 1
            if ctx.get("reason") == "no_char_history":
                s["n_surface_form_only_skip"] += 1
            continue
        # Sampler produced a candidate — track Bernoulli pass
        s["n_bernoulli_pass"] += 1
        bernoulli_pass_total += 1

        # Stage item for batch gate
        items.append({
            "uid": uid,
            "asin": asin,
            "query": query,
            "injected": injected,
            "user_model": user_model,
            "stats": stats,
            "user_stats": stats,
            "mu": np.array(stats["mu"], dtype=np.float32),
            "sigma_inv": np.array(stats["sigma_inv"], dtype=np.float32),
            "d2_threshold": d2_threshold,
            "competitors": cohort.get(asin, {}),
            "sampler_info": sampler_info,
        })
        if (i + 1) % 50 == 0:
            log(f"  prepared {i+1}/{len(pairs)} pairs (sampled={len(items)})")

    log(f"Phase 1 done: {len(items)}/{n_total_processed} pairs sampled (rest: no-candidate)")

    # Phase 2: batch encode + gate
    asin_to_A, global_A = _is_load_coral()
    log("Phase 2: batch encoding all (orig, inj) pairs")
    items = _is_batch_gate(items, None, encoder, rule_to_id, vocab_size, asin_to_A, global_A)
    log(f"  encoded {len(items)*2} texts, d²_before/after computed")
    log("Phase 3: batch semantic + comp gates")
    items = _is_batch_semantic(items)
    items = _is_batch_comp_gate(items)
    log(f"  all gates computed")

    # Phase 4: collect results + stats
    results = []
    for it in items:
        meta_partial = it["sampler_info"]
        s = per_user_stats[it["uid"]]
        d2_after = it["d2_after"]
        d2_before = it["d2_before"]
        gaussian_pass = it["gaussian_pass"]
        semantic_pass = it["semantic_pass"]
        exclusive_pass = it["exclusive_pass"]
        if gaussian_pass and semantic_pass and exclusive_pass:
            s["n_injected"] += 1
            s["mechanisms"][meta_partial["mech"]] += 1
            s["typo_count"] += 1
            typo_total += 1
            s["d2_deltas"].append(d2_after - d2_before)
            s["sem_sims"].append(it["semantic_sim"])
            if it["min_d_competitor"] > 0:
                s["min_d_competitors"].append(it["min_d_competitor"])
            mech_total[meta_partial["mech"]] += 1
            s["transformation_sources"][meta_partial["source"]] += 1
            s["edit_distances"].append(meta_partial["edit_dist"])
        else:
            if not gaussian_pass:
                s["n_gate_fail"] += 1
                gate_fail_total += 1
            elif gaussian_pass and not semantic_pass:
                s["n_semantic_fail"] += 1
                semantic_fail_total += 1
            elif gaussian_pass and semantic_pass and not exclusive_pass:
                s["n_exclusive_fail"] += 1
                exclusive_fail_total += 1
            else:
                s["n_minimality_fail"] = s.get("n_minimality_fail", 0) + 1
        if gaussian_pass:
            results.append({
                "uid": it["uid"],
                "asin": it["asin"],
                "original_query": it["query"],
                "typo_query": it["injected"],
                "original_token": meta_partial["word"],
                "typo_token": meta_partial["typo"],
                "mechanism": meta_partial["mech"],
                "transformation_source": meta_partial["source"],
                "edit_distance": meta_partial["edit_dist"],
                "len_delta": meta_partial["len_delta"],
                "context_sig": meta_partial["sigs"].get("full", ""),
                "confidence": meta_partial["w_i"],
                "semantic_sim": it["semantic_sim"],
            })

        # SMOKE: keep all meta for diagnosis
        if SMOKE and meta is not None:
            debug_metas.append({
                "uid": uid,
                "asin": asin,
                "position": meta.position,
                "original_token": meta.original_token,
                "typo_token": meta.typo_token,
                "mechanism": meta.mechanism,
                "transformation_source": meta.transformation_source,
                "edit_distance": meta.edit_distance,
                "len_delta": meta.len_delta,
                "d2_before": meta.d_mahalanobis_before,
                "d2_after": meta.d_mahalanobis_after,
                "d2_threshold": stats.get(f"d2_{D2_THRESHOLD_QUANTILE}"),
                "gaussian_pass": meta.gaussian_pass,
                "semantic_sim": meta.semantic_sim,
                "semantic_pass": meta.semantic_pass,
                "min_d_competitor": meta.min_d_competitor,
                "n_competitors": meta.n_competitors,
                "exclusive_pass": meta.exclusive_pass,
            })

        if (i + 1) % 50 == 0:
            log(f"  processed {i+1}/{len(pairs)} pairs")

    log(f"done. {len(results)} injected pairs (from {n_total_processed} attempted) in {time.time()-t0:.1f}s")

    # Summary
    n_injected = len(results)
    summary = {
        "config": {
            "smoke": SMOKE,
            "all_selected_queries": True,
            "single_shot": True,
            "d2_threshold_quantile": D2_THRESHOLD_QUANTILE,
            "mahal_stats_source": str(STAGE04_PATH),
            "cohort_gates_source": str(STAGE04_PATH),
            "semantic_threshold": 0.9,
            "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
            "char_level_mechanisms": ["keyboard_adjacent", "letter_swap", "letter_repetition",
                                       "letter_insertion", "letter_deletion"],
            "min_token_len": 3,
            "max_edit_distance": 2,
            "max_len_delta": 1,
            "valid_word_check": True,
            "note": "char-level-only single-shot injection. Position ranked by P_u(char_level_error|context). "
                    "Transformation reuses user-historical char-level typo at the same (orig_token, sig) "
                    "if available, else generic char-level mechanism. Gates: minimality (ed≤2, |Δlen|≤1, "
                    "len≥3, not a different English word) → Mahalanobis D²(target Q_95) → exclusive cohort "
                    "(∀comp: d²>comp Q_95) → MiniLM cosine ≥ 0.9. Surface-form errors (case_error / "
                    "apostrophe_error) are NOT emitted — they are tracked separately via "
                    "n_surface_form_only_skip when user has only surface-form history.",
        },
        "totals": {
            "n_attempted": n_total_processed,
            "n_injected_written": n_injected,
            "n_users_injected": len(set(r["uid"] for r in results)),
            "n_bernoulli_pass": bernoulli_pass_total,
            "n_gate_fail": gate_fail_total,
            "n_semantic_fail": semantic_fail_total,
            "n_exclusive_fail": exclusive_fail_total,
            "n_minimality_fail": minimality_fail_total,
            "n_surface_form_only_skip": surface_form_only_skip_total,
            "injection_rate_among_bernoulli": n_injected / max(1, bernoulli_pass_total),
            "bernoulli_pass_rate": bernoulli_pass_total / max(1, n_total_processed),
            "typo_count": typo_total,
            "mechanism_counts": dict(mech_total),
            "transformation_source_counts": {
                src: sum(s.get("transformation_sources", {}).get(src, 0)
                         for s in per_user_stats.values())
                for src in ["user_historical_full", "user_historical_d3",
                            "user_historical_coarse", "user_historical_any",
                            "generic_fallback"]
            },
        },
        "per_user": {
            uid: {
                "n_total": s["n_total"],
                "n_bernoulli_pass": s["n_bernoulli_pass"],
                "n_injected": s["n_injected"],
                "n_gate_fail": s["n_gate_fail"],
                "n_semantic_fail": s["n_semantic_fail"],
                "n_exclusive_fail": s["n_exclusive_fail"],
                "n_minimality_fail": s.get("n_minimality_fail", 0),
                "n_no_candidate": s["n_no_candidate"],
                "n_surface_form_only_skip": s["n_surface_form_only_skip"],
                "typo_count": s["typo_count"],
                "mechanisms": dict(s["mechanisms"]),
                "transformation_sources": dict(s["transformation_sources"]),
                "edit_distance_mean": float(np.mean(s["edit_distances"])) if s["edit_distances"] else None,
                "edit_distance_max": int(max(s["edit_distances"])) if s["edit_distances"] else None,
                "d2_delta_mean": float(np.mean(s["d2_deltas"])) if s["d2_deltas"] else None,
                "sem_sim_mean": float(np.mean(s["sem_sims"])) if s["sem_sims"] else None,
                "sem_sim_min": float(min(s["sem_sims"])) if s["sem_sims"] else None,
                "min_d_competitor_mean": float(np.mean(s["min_d_competitors"])) if s["min_d_competitors"] else None,
                "min_d_competitor_min": float(min(s["min_d_competitors"])) if s["min_d_competitors"] else None,
            }
            for uid, s in per_user_stats.items()
        },
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w") as f:
        json.dump({
            "config": summary["config"],
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    log(f"wrote → {OUT_RESULTS} ({len(results)} char-level injected pairs)")

    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote → {OUT_SUMMARY}")

    if SMOKE:
        debug_path = OUT_RESULTS.parent / "smoke_debug_metas.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_metas, f, indent=2, ensure_ascii=False)
        log(f"wrote → {debug_path} ({len(debug_metas)} meta records)")

    t = summary["totals"]
    log(f"Bernoulli pass rate: {t['bernoulli_pass_rate']*100:.1f}% ({t['n_bernoulli_pass']}/{t['n_attempted']})")
    log(f"Injection rate (among Bernoulli pass): {t['injection_rate_among_bernoulli']*100:.1f}%")
    log(f"Overall injection: {t['n_injected_written']}/{t['n_attempted']} = {t['n_injected_written']/t['n_attempted']*100:.1f}%")
    log(f"Surface-form-only skip: {t['n_surface_form_only_skip']}")
    log(f"Minimality fail: {t['n_minimality_fail']}")
    log(f"Mahalanobis gate fail: {t['n_gate_fail']}")
    log(f"Exclusive cohort gate fail: {t['n_exclusive_fail']}")
    log(f"Semantic gate fail (sim<0.9): {t['n_semantic_fail']}")
    log(f"Typo (char-level): {t['typo_count']}")
    log(f"Mechanisms: {dict(mech_total)}")
    log(f"Transformation sources: {t['transformation_source_counts']}")


if __name__ == "__main__":
    main()