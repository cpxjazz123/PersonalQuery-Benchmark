#!/usr/bin/env python3
"""E21: simplified user-vs-user syntactic style discrimination.

Question: pooled over all of a user's reviews, can we distinguish user U
from user V by syntactic style? (E20 v2 attempted the harder
cross-product LOPO question; this is the simpler aggregate question.)

Design:
  1. Pool all reviews per user (>=200 words, >=3 products, >=10 sents).
  2. Per-sentence syntactic primitives from e20_lopo_v2.per_sentence_features.
  3. Stability selection (3-criterion, not E20 v2's single corr):
     (a) |corr(feat, sent_count)| < 0.1   across users
     (b) cross-sent-count ICC > 0.6       over 5/10/20/40 sentence samples
     (c) resample stability ICC > 0.7     over 5 resamples per sent count
  4. self vs cross distance:
     self  = ||vec_U_seed_a - vec_U_seed_b||   (a != b, same sent count)
     cross = ||vec_U - vec_V||                 (U != V)
     AUC sanity-checked against synthetic case where self < cross.
  5. 999 user-label permutations.

Runs on genrec_env python (spacy 3.8.14). Hard-coded config.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from e20_lopo_v2 import per_sentence_features  # reuse
from e20_lopo_v2 import (  # reuse constants
    RATE_FEATS, DIST_FEATS, ALL_FEATS, OPENER_POSES, SENT_TYPES,
)

REVIEWS = REPO_ROOT / "data" / "Baby_Products_2023.jsonl.gz"
META = REPO_ROOT / "data" / "meta_Baby_Products_2023.jsonl.gz"
OUT = REPO_ROOT / "result" / "e21_simple_discrimination_v2.json"

SEED = 42
N_USERS = 300
MIN_ASINS = 3
MIN_WORDS = 200
MIN_SENTS = 50             # v2: need enough sents to sample 5/10/20/40 without overlap
# SAMPLE BY COMPLETE SENTENCE COUNT (not token count): syntactic primitive
# is the full sentence, not the partial token window.
SAMPLE_SENT_COUNTS = (5, 10, 20, 40)
N_RESAMPLE = 5
SENT_BIN = 20              # canonical sent count for self/cross comparison
# stability thresholds (per #21 Task 2)
STAB_CORR_THRESH = 0.1
STAB_CROSS_BIN_ICC = 0.6
STAB_RESAMPLE_ICC = 0.7
# permutation
N_PERM = 999


def user_features_from_sents(sent_feats: list[dict]) -> np.ndarray | None:
    """Aggregate per-sentence features to a fixed user vector. Same as
    e20_lopo_v2.user_feature_vector but standalone (avoid cross-import)."""
    if not sent_feats:
        return None
    total_tok = sum(s["n_tok"] for s in sent_feats)
    n_sent = len(sent_feats)
    if total_tok == 0 or n_sent == 0:
        return None
    f = {}
    f["clause_rate"] = sum(s["n_clause"] for s in sent_feats) / total_tok
    f["acl_rate"] = sum(s["acl"] for s in sent_feats) / total_tok
    f["advcl_rate"] = sum(s["advcl"] for s in sent_feats) / total_tok
    f["ccomp_rate"] = sum(s["ccomp"] for s in sent_feats) / total_tok
    f["xcomp_rate"] = sum(s["xcomp"] for s in sent_feats) / total_tok
    f["relcl_rate"] = sum(s["relcl"] for s in sent_feats) / total_tok
    f["modifier_density"] = sum(s["n_mod"] for s in sent_feats) / total_tok
    f["coordination_density"] = sum(s["n_coord"] for s in sent_feats) / total_tok
    f["mean_dep_distance"] = float(np.mean([s["mean_dist"] for s in sent_feats]))
    f["depth_variance"] = float(np.mean([s["depth_var"] for s in sent_feats]))
    f["median_dep_depth"] = float(np.median([s["max_depth"] for s in sent_feats]))
    f["passive_rate"] = sum(1 for s in sent_feats if s["has_passive"]) / n_sent
    f["interrogative_rate"] = sum(1 for s in sent_feats if s["is_interrog"]) / n_sent
    f["conditional_rate"] = sum(1 for s in sent_feats if s["has_cond"]) / n_sent
    opener_counts = Counter(s["opener"] for s in sent_feats)
    for p in OPENER_POSES:
        f[f"opener_{p}"] = opener_counts.get(p, 0) / n_sent
    stype_counts = Counter(s["stype"] for s in sent_feats)
    for t in SENT_TYPES:
        f[f"senttype_{t}"] = stype_counts.get(t, 0) / n_sent
    return np.asarray([f[name] for name in ALL_FEATS], dtype=np.float32)


def sample_by_sents(sent_feats: list[dict], n_sents: int, rng: np.random.Generator) -> list[dict]:
    """Random sample of n_sents complete sentences (no truncation).

    v2 of e21: syntactic primitives are complete sentences, so the sampling
    unit is the sentence count, not the token count. This avoids the v1 bug
    where cum>=target_tok stop condition produced 1-50 sentence samples
    depending on sentence length distribution.
    """
    if not sent_feats or n_sents <= 0:
        return []
    n = min(n_sents, len(sent_feats))
    order = rng.permutation(len(sent_feats))
    return [sent_feats[i] for i in order[:n]]


def icc_one_way(values: np.ndarray) -> float:
    """One-way ICC (ICC1) for a 1D array of measurements per subject.

    Single rater per subject: ICC = (MS_B - MS_W) / (MS_B + (k-1) MS_W)
    where k is avg measurements per subject. Here we use ICC(1,1) which
    equals (var_between - var_within) / (var_between + (k-1) var_within).

    values: shape (n_subjects, k) or ragged -> requires equal k.
    """
    if values.ndim != 2 or values.shape[1] < 2:
        return 0.0
    n, k = values.shape
    grand = values.mean()
    subj_means = values.mean(axis=1, keepdims=True)
    ms_b = float(((subj_means - grand) ** 2).sum() * k / (n - 1))
    ms_w = float(((values - subj_means) ** 2).sum() / (n * (k - 1)))
    if ms_b + (k - 1) * ms_w <= 0:
        return 0.0
    return (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # pass 1: word counts
    wc: dict[str, int] = defaultdict(int)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            t = d.get("text") or ""
            if u:
                wc[u] += len(t.split())
    print(f"users scanned: {len(wc)} (t={time.time() - t0:.1f}s)", flush=True)

    cands = [u for u, w in wc.items() if w >= MIN_WORDS]
    rng.shuffle(cands)
    cands = cands[:N_USERS * 4]
    poolset = set(cands)
    print(f"word>={MIN_WORDS} candidates: {len(cands)}", flush=True)

    # pass 2: reviews
    reviews: dict[str, list[tuple]] = defaultdict(list)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u not in poolset:
                continue
            t = (d.get("text") or "").strip()
            if t:
                reviews[u].append((d.get("parent_asin") or d.get("asin"), t[:2000]))
    for u in list(reviews):
        if len({r[0] for r in reviews[u]}) < MIN_ASINS:
            del reviews[u]
    print(f"users with >= {MIN_ASINS} products: {len(reviews)}", flush=True)

    # spaCy parse + per-sentence features per user (pooled)
    from e20_lopo_v2 import load_spacy_model
    nlp = load_spacy_model()
    user_sents: dict[str, list[dict]] = {}
    for u, revs in reviews.items():
        sfs = []
        for _, t in revs:
            doc = nlp(t)
            for sent in doc.sents:
                sf = per_sentence_features(sent)
                if sf is not None:
                    sfs.append(sf)
        if len(sfs) >= MIN_SENTS:
            user_sents[u] = sfs
    print(f"users with >= {MIN_SENTS} sents: {len(user_sents)} "
          f"(t={time.time() - t0:.1f}s)", flush=True)

    # sub-sample
    user_list = sorted(user_sents)
    rng.shuffle(user_list)
    if len(user_list) > N_USERS:
        keep = set(user_list[:N_USERS])
        user_sents = {u: user_sents[u] for u in keep}
    n_users = len(user_sents)
    print(f"using {n_users} users", flush=True)

    # ============================================================
    # Task 2: 3-criterion stability filter
    # ============================================================
    # v2: sample by complete sentence count (5, 10, 20, 40)
    # Resulting shape: (n_users, len(SAMPLE_SENT_COUNTS), N_RESAMPLE, n_feats)
    n_feats = len(ALL_FEATS)
    samples = np.zeros((n_users, len(SAMPLE_SENT_COUNTS), N_RESAMPLE, n_feats), dtype=np.float32)
    sent_counts_per_user = np.zeros((n_users, len(SAMPLE_SENT_COUNTS)), dtype=np.float32)
    for ui, u in enumerate(sorted(user_sents)):
        for ti, target_n in enumerate(SAMPLE_SENT_COUNTS):
            for ri in range(N_RESAMPLE):
                s_rng = np.random.default_rng(SEED + ui * 1000 + ti * 100 + ri)
                samps = sample_by_sents(user_sents[u], target_n, s_rng)
                v = user_features_from_sents(samps)
                if v is not None:
                    samples[ui, ti, ri] = v
                    sent_counts_per_user[ui, ti] = len(samps)
    print(f"sample grid: ({n_users}, {len(SAMPLE_SENT_COUNTS)}, {N_RESAMPLE}, {n_feats})", flush=True)

    # criterion (a): per-feature corr(feat, sent_count) across users, averaged
    # over (ti, ri). Use canonical bin (ti=index of SENT_BIN in SAMPLE_SENT_COUNTS)
    canonical_ti = SAMPLE_SENT_COUNTS.index(SENT_BIN)
    feat_corr: dict[str, float] = {}
    for fi, fname in enumerate(ALL_FEATS):
        corrs = []
        for ti in range(len(SAMPLE_SENT_COUNTS)):
            for ri in range(N_RESAMPLE):
                fvals = samples[:, ti, ri, fi]
                ns = sent_counts_per_user[:, ti]
                if np.std(fvals) == 0 or np.std(ns) == 0:
                    continue
                r = float(np.corrcoef(fvals, ns)[0, 1])
                if np.isfinite(r):
                    corrs.append(r)
        feat_corr[fname] = float(np.mean(corrs)) if corrs else 0.0
    pass_a = {f for f, c in feat_corr.items() if abs(c) < STAB_CORR_THRESH}
    print(f"criterion (a) |corr|<{STAB_CORR_THRESH}: {len(pass_a)}/{n_feats}", flush=True)

    # criterion (b): cross-sent-count ICC. For each user, take the mean across
    # resamples at each sent count (k=len(SAMPLE_SENT_COUNTS) measurements per user).
    feat_cross_icc: dict[str, float] = {}
    for fi, fname in enumerate(ALL_FEATS):
        # use mean across resamples: shape (n_users, len(SAMPLE_SENT_COUNTS))
        m = samples[:, :, :, fi].mean(axis=2)  # (n_users, n_bins)
        # drop zero-variance columns
        if np.std(m) == 0 or m.shape[1] < 2:
            feat_cross_icc[fname] = 0.0
            continue
        feat_cross_icc[fname] = icc_one_way(m)
    pass_b = {f for f, v in feat_cross_icc.items() if v > STAB_CROSS_BIN_ICC}
    print(f"criterion (b) cross-bin ICC>{STAB_CROSS_BIN_ICC}: {len(pass_b)}/{n_feats}", flush=True)

    # criterion (c): resample stability. For each sent count, ICC across the
    # N_RESAMPLE resamples (k=N_RESAMPLE measurements per user). Average over
    # sent counts.
    feat_resample_icc: dict[str, float] = {}
    for fi, fname in enumerate(ALL_FEATS):
        iccs = []
        for ti in range(len(SAMPLE_SENT_COUNTS)):
            m = samples[:, ti, :, fi]  # (n_users, N_RESAMPLE)
            if np.std(m) == 0 or m.shape[1] < 2:
                continue
            iccs.append(icc_one_way(m))
        feat_resample_icc[fname] = float(np.mean(iccs)) if iccs else 0.0
    pass_c = {f for f, v in feat_resample_icc.items() if v > STAB_RESAMPLE_ICC}
    print(f"criterion (c) resample ICC>{STAB_RESAMPLE_ICC}: {len(pass_c)}/{n_feats}", flush=True)

    stable_feats = sorted(pass_a & pass_b & pass_c)
    feat_dim = len(stable_feats)
    feat_idx = {f: ALL_FEATS.index(f) for f in stable_feats}
    print(f"intersection of a/b/c: {feat_dim} stable features", flush=True)
    if feat_dim == 0:
        print("NO STABLE FEATURES; cannot proceed with self/cross comparison", flush=True)

    # extract per-user canonical-bin vector (mean over N_RESAMPLE)
    user_vec = np.zeros((n_users, feat_dim), dtype=np.float32)
    sorted_users = sorted(user_sents)
    for ui, u in enumerate(sorted_users):
        vs = samples[ui, canonical_ti, :, :][:, [feat_idx[f] for f in stable_feats]]
        user_vec[ui] = vs.mean(axis=0)

    # unit-norm for cosine-like L2
    norms = np.linalg.norm(user_vec, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    user_vec_norm = user_vec / norms

    # ============================================================
    # Task 3: self vs cross distance
    # ============================================================
    rng_sc = np.random.default_rng(SEED + 1)
    diffs_self = []
    diffs_cross = []
    for ui in range(n_users):
        # self: two resamples (different ri) at canonical bin
        ri_pool = list(range(N_RESAMPLE))
        if len(ri_pool) < 2:
            continue
        a, b = rng_sc.choice(ri_pool, size=2, replace=False)
        va = samples[ui, canonical_ti, a, :][ [feat_idx[f] for f in stable_feats] ]
        vb = samples[ui, canonical_ti, b, :][ [feat_idx[f] for f in stable_feats] ]
        va_n = va / max(np.linalg.norm(va), 1e-12)
        vb_n = vb / max(np.linalg.norm(vb), 1e-12)
        d_self = float(np.linalg.norm(va_n - vb_n))
        diffs_self.append(d_self)
        # cross: K random other users
        others = [v for v in range(n_users) if v != ui]
        rng_sc.shuffle(others)
        for v in others[:3]:
            va_cross = samples[ui, canonical_ti, 0, :][ [feat_idx[f] for f in stable_feats] ]
            vb_cross = samples[v, canonical_ti, 0, :][ [feat_idx[f] for f in stable_feats] ]
            va_cn = va_cross / max(np.linalg.norm(va_cross), 1e-12)
            vb_cn = vb_cross / max(np.linalg.norm(vb_cross), 1e-12)
            d_cross = float(np.linalg.norm(va_cn - vb_cn))
            diffs_cross.append(d_cross)
    diffs_self = np.asarray(diffs_self)
    diffs_cross = np.asarray(diffs_cross)
    obs_delta = float(diffs_self.mean() - diffs_cross.mean())
    print(f"self/cross: n_self={len(diffs_self)} d_self={diffs_self.mean():.4f} "
          f"d_cross={diffs_cross.mean():.4f} delta={obs_delta:.4f}", flush=True)

    # AUC sanity check: with d_self < d_cross expected, AUC(label=1 if d_self < d_cross)
    # should be > 0.5. Convention: for each (d_self, d_cross) pair, label=1 if
    # d_self <= d_cross (i.e., self is "more similar"). AUC = P(d_self < d_cross).
    auc_label = 0.0
    n_auc = 0
    for ds in diffs_self:
        auc_label += float((diffs_cross > ds).sum())
        n_auc += len(diffs_cross)
    auc_label = auc_label / max(n_auc, 1)
    # alternate convention: "negative distance" as score (smaller distance => larger score)
    auc_neg = 0.0
    n_auc2 = 0
    for ds in diffs_self:
        auc_neg += float((diffs_cross < ds).sum())  # d_cross < d_self (worse)
        n_auc2 += len(diffs_cross)
    auc_neg = auc_neg / max(n_auc2, 1)
    print(f"AUC (P(d_cross>d_self) i.e. self is closer): {auc_label:.4f}", flush=True)
    print(f"AUC (P(d_cross<d_self) i.e. self is farther): {auc_neg:.4f}", flush=True)

    # ============================================================
    # Task 4: 999 user-label permutations
    # ============================================================
    # shuffle user-pair labels (which pairs are "self")
    # Construct null distribution of d_self - d_cross by randomly pairing
    # users.
    rngp = np.random.default_rng(SEED + 2)
    perm_deltas = []
    for _ in range(N_PERM):
        # permute: take K users, randomly assign to "self" or "cross" partner
        # simpler: take len(diffs_self) pairs of (user_a, user_b) random, compute d_ab
        null_d = []
        for _ in range(len(diffs_self)):
            a, b = rngp.choice(n_users, size=2, replace=False)
            va = samples[a, canonical_ti, 0, :][ [feat_idx[f] for f in stable_feats] ]
            vb = samples[b, canonical_ti, 0, :][ [feat_idx[f] for f in stable_feats] ]
            va_n = va / max(np.linalg.norm(va), 1e-12)
            vb_n = vb / max(np.linalg.norm(vb), 1e-12)
            null_d.append(float(np.linalg.norm(va_n - vb_n)))
        perm_deltas.append(np.mean(null_d))
    perm_deltas = np.asarray(perm_deltas)
    # observed: mean cross distance at this permuted set
    null_cross = float(perm_deltas.mean())
    # delta under null = mean(d_self_perm) - mean(d_cross_perm), but we only
    # have permuted "self" distances. The proper null delta assumes d_self
    # and d_cross are exchangeable. Use |d_perm - mean(d_cross)| vs
    # |d_self - mean(d_cross)|.
    abs_perm = np.abs(perm_deltas - diffs_cross.mean())
    abs_obs = abs(obs_delta)
    p_two = (float((abs_perm >= abs_obs).sum()) + 1) / (len(perm_deltas) + 1)
    # one-sided lower tail: H1 is d_self < d_cross (delta more negative)
    perm_delta_vec = perm_deltas - diffs_cross.mean()
    obs_delta_signed = obs_delta
    p_one_lower = (float((perm_delta_vec <= obs_delta_signed).sum()) + 1) / (len(perm_delta_vec) + 1)

    # ============================================================
    # Output
    # ============================================================
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "question": "pooled over all of a user's reviews, can we distinguish "
                    "user U from user V by syntactic style? (the simpler version "
                    "of E20 v2, which tested the harder cross-product LOPO question)",
        "design": "Task1 pool user reviews; Task2 3-criterion stability filter "
                 "(|corr|<0.1 + cross-bin ICC>0.6 + resample ICC>0.7); Task3 "
                 "self=two resamples, cross=other user; Task4 999 user-label "
                 "permutations; AUC sanity-checked with two conventions",
        "stability": {
            "criterion_a_corr": {f: round(c, 4) for f, c in feat_corr.items()},
            "criterion_b_cross_bin_icc": {f: round(v, 4) for f, v in feat_cross_icc.items()},
            "criterion_c_resample_icc": {f: round(v, 4) for f, v in feat_resample_icc.items()},
            "pass_a": sorted(pass_a),
            "pass_b": sorted(pass_b),
            "pass_c": sorted(pass_c),
            "stable": stable_feats,
            "n_stable": feat_dim,
        },
        "n_users": n_users, "n_self_pairs": len(diffs_self), "n_cross_pairs": len(diffs_cross),
        "n_perm": N_PERM, "canonical_sent_bin": SENT_BIN,
        "self_vs_cross": {
            "d_self_mean": round(float(diffs_self.mean()), 4),
            "d_cross_mean": round(float(diffs_cross.mean()), 4),
            "delta": round(obs_delta, 4),
            "auc_p_cross_gt_self": round(float(auc_label), 4),
            "auc_p_cross_lt_self": round(float(auc_neg), 4),
            "auc_label_correct": "auc_p_cross_gt_self" if auc_label > 0.5 else "INVERTED",
        },
        "permutation": {
            "null_dist_mean": round(float(perm_deltas.mean()), 4),
            "obs_delta": round(obs_delta, 4),
            "p_one_lower": round(float(p_one_lower), 4),
            "p_two_sided": round(float(p_two), 4),
        },
        "runtime_sec": round(time.time() - t0, 1),
    }, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} (t={time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
