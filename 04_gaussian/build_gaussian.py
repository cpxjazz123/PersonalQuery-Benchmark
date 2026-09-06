"""Stage 1: Compute per-user syntactic Gaussian in 287d spaCy space.

Reads Step 3 cache (syntax_features.npy + uid_offsets.json) directly,
computes per-user Gaussian + exclusive d_self. NO spaCy re-encoding.

Outputs:
  result/syntax_style_encoder/user_syntax_gaussian.json
  result/syntax_style_encoder/syntax_exclusive_distance.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = REPO_ROOT / "result/04_gaussian"
OUT_GAUSSIAN = OUT_DIR / "user_syntax_gaussian.json"
OUT_EXCL = OUT_DIR / "syntax_exclusive_distance.json"
ASIN_USERS_PATH = REPO_ROOT / "asin_users/asin_to_users.json"
SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract/uid_to_sentences.pkl"

# Step 3 output (read-only cache, no re-encoding)
FEAT_CACHE = REPO_ROOT / "result/03_spacy_encode/syntax_features.npy"
OFFSET_CACHE = REPO_ROOT / "result/03_spacy_encode/uid_offsets.json"
UIDLIST_CACHE = REPO_ROOT / "result/03_spacy_encode/uid_list.json"

MIN_VALID_SENTS = 100


def d_boundary_from_spheres(d_ij, sigma_i, sigma_j):
    if d_ij < 1e-8:
        return np.nan
    si = float(np.linalg.norm(sigma_i)) if hasattr(sigma_i, '__len__') else float(sigma_i)
    sj = float(np.linalg.norm(sigma_j)) if hasattr(sigma_j, '__len__') else float(sigma_j)
    return (d_ij**2 - sj**2 + si**2) / (2 * d_ij)


def compute_exclusive(mus, sigmas, asin_label: str = "?"):
    """Per-user exclusive d_self WITHIN one ASIN cohort.

    Caller MUST pass mus/sigmas for users sharing the same ASIN — already
    filtered in main() via `asin_to_users[asin] ∩ gauss_uids`. We do NOT
    enumerate across all 5000 users here; that would dilute the metric.
    """
    assert len(mus) == len(sigmas), "mus/sigmas length mismatch"
    n = len(mus)
    if n < 2:
        return np.full(n, np.nan)
    exclusive = np.full(n, np.nan)
    for i in range(n):
        min_bound = np.inf
        si = sigmas[i]
        for j in range(n):
            if i == j:
                continue
            d_ij = float(np.linalg.norm(mus[i] - mus[j]))
            bound = d_boundary_from_spheres(d_ij, si, sigmas[j])
            if bound < min_bound:
                min_bound = bound
        exclusive[i] = min_bound if np.isfinite(min_bound) else np.nan
    return exclusive


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load Step 3 cache (no re-encoding)
    print("Loading Step 3 spaCy cache...", flush=True)

    if not FEAT_CACHE.exists() or not OFFSET_CACHE.exists() or not UIDLIST_CACHE.exists():
        print(f"ERROR: Step 3 cache not found. Run 03_spacy_encode first.", flush=True)
        print(f"  Expected: {FEAT_CACHE}, {OFFSET_CACHE}, {UIDLIST_CACHE}", flush=True)
        sys.exit(1)

    features = np.load(FEAT_CACHE)  # (total_sents, 287)
    print(f"  loaded features: {features.shape}", flush=True)

    with open(OFFSET_CACHE) as f:
        uid_offsets = json.load(f)
    print(f"  loaded offsets: {len(uid_offsets)} users", flush=True)

    with open(UIDLIST_CACHE) as f:
        uid_list = json.load(f)
    print(f"  uid_list: {len(uid_list)} users", flush=True)

    # Build per-user Gaussian
    print("Building per-user syntactic Gaussian...", flush=True)
    user_gaussian = {}
    valid_user_count = 0

    for uid in uid_list:
        off = uid_offsets.get(uid)
        if off is None:
            continue
        start = off['start']
        n = off['n']
        if n < MIN_VALID_SENTS:
            continue
        X = features[start:start + n]
        valid_mask = np.any(X != 0, axis=1)
        X_valid = X[valid_mask]
        if len(X_valid) < MIN_VALID_SENTS:
            continue
        mu = np.mean(X_valid, axis=0)
        sigma = np.std(X_valid, axis=0) + 1e-8
        user_gaussian[uid] = {
            'mu': mu.tolist(),
            'sigma': sigma.tolist(),
            'n_valid': int(len(X_valid)),
        }
        valid_user_count += 1
        if valid_user_count % 1000 == 0:
            print(f"  {valid_user_count} Gaussians built...", flush=True)

    print(f"  {len(user_gaussian)} Gaussian profiles", flush=True)

    print(f"Writing → {OUT_GAUSSIAN}", flush=True)
    with open(OUT_GAUSSIAN, 'w') as f:
        json.dump(user_gaussian, f)

    # Compute exclusive d_self
    print("Computing exclusive d_self in syntax space...", flush=True)
    asin_to_users = json.load(open(ASIN_USERS_PATH))
    gauss_uids = set(user_gaussian.keys())
    syntax_exclusive = {}

    for asin, asin_uid_list in asin_to_users.items():
        cohort = [u for u in asin_uid_list if u in gauss_uids]
        if len(cohort) < 2:
            continue
        mus = np.array([user_gaussian[u]['mu'] for u in cohort])
        sigmas = np.array([user_gaussian[u]['sigma'] for u in cohort])
        exclusive = compute_exclusive(mus, sigmas)
        syntax_exclusive[asin] = {u: float(exclusive[j]) for j, u in enumerate(cohort)}

    print(f"  {len(syntax_exclusive)} ASINs with ≥2 syntax users", flush=True)

    all_excl = [d for asin_ds in syntax_exclusive.values() for d in asin_ds.values() if not np.isnan(d)]
    if all_excl:
        arr = np.array(all_excl)
        print(f"\nSyntax exclusive d_self distribution:", flush=True)
        for p in [10, 25, 50, 75, 90]:
            print(f"  {p}th: {np.percentile(arr, p):.4f}", flush=True)
        print(f"  > 0:   {(arr > 0).mean()*100:.1f}%", flush=True)
        print(f"  > 1:   {(arr > 1).mean()*100:.1f}%", flush=True)
        print(f"  > 5:   {(arr > 5).mean()*100:.1f}%", flush=True)

    print(f"Writing → {OUT_EXCL}", flush=True)
    with open(OUT_EXCL, 'w') as f:
        json.dump(syntax_exclusive, f)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
