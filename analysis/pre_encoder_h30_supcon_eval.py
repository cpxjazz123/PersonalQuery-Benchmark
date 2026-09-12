"""Phase L8.26 — Evaluate SupCon-trained 64,996-user cohort.

Same eval recipe as L8.18/L8.21:
  - per-user LOOCV (n_p=49 / n_t=1 for n≥50; for n<50 use n_t = max(1, n//10))
  - separation_med = median over users of (intra_d_med / inter_d_med)
  - intra_spread_med, inter_user_dist_med
  - lambda_min_med, cond_med, det_ratio_med

Input:
  pcfg_cache/strict_embeddings.npz (z_profile + z_test)
  pcfg_cache/supervised_embeddings.npy (z_all)
  pcfg_cache/user_n_sents.json

Output:
  result/analysis/pre_encoder_h30_supcon_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
OUT_DIR = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_user_metrics(z_profile: np.ndarray,
                     z_test: np.ndarray,
                     user_n_sents: list[int],
                     sample_size: int = 5000):
    """Compute per-user LOOCV separation/λ_min/cond/det.

    For each user u:
      - take their profile sents (subset of z_profile)
      - take their test sents (subset of z_test)
      - random sample up to sample_size users for inter-user pool
      - separation = intra_d_med / inter_d_med
    """
    rng = np.random.default_rng(SEED)
    n_users = len(user_n_sents)

    # Build per-user arrays (start/end offsets for z_profile + z_test)
    prof_off = np.zeros(n_users + 1, dtype=np.int64)
    prof_off[1:] = np.cumsum(user_n_sents)
    test_off = np.zeros(n_users + 1, dtype=np.int64)
    test_off[1:] = np.cumsum(user_n_sents)  # profile and test are equal size
    # Actually we need to know the actual split sizes from strict_embeddings.
    # Easiest: build per-user profile/test arrays by membership.
    # We have profile_idx / test_idx in npz; use them.

    # Load split indices
    z = np.load(CACHE_DIR / "strict_embeddings.npz")
    profile_idx = z["profile_idx"]
    test_idx = z["test_idx"]
    z.close()

    # Per-user mask in profile/test.
    # profile_idx and test_idx are SHUFFLED position-arrays (the SHA1 split
    # reorders them; they are NOT contiguous-by-user). So run-length
    # accumulation on profile_idx DOES NOT give per-user layout.
    # Instead: for each sent position i, the original user index is the
    # bucket it was assigned to via SHA1. We use the global sent_off
    # helper (sent_off[u]..sent_off[u+1]) to find user u for any global
    # sent idx, and re-mask.
    n_total = sum(user_n_sents)
    user_off = np.zeros(len(user_n_sents) + 1, dtype=np.int64)
    user_off[1:] = np.cumsum(user_n_sents)
    log(f"  n_total sents = {n_total:,}, user_off last = {user_off[-1]:,}")

    # profile_idx / test_idx are global sent indices into n_total axis.
    # So z_profile[i] corresponds to sent position profile_idx[i] in
    # the full vocabulary. Map each to user via searchsorted.
    profile_uid = np.searchsorted(user_off[1:], profile_idx, side="right")
    test_uid = np.searchsorted(user_off[1:], test_idx, side="right")
    log(f"profile sents: {len(profile_uid)}, test sents: {len(test_uid)}")
    log(f"profile_uid range = [{profile_uid.min()}, {profile_uid.max()}]")

    # Subsample users for inter_user pool (sample_size)
    eval_users = np.arange(n_users)
    if n_users > sample_size:
        eval_users = rng.choice(n_users, size=sample_size, replace=False)
    log(f"evaluating on {len(eval_users)} users")

    sep_list = []
    intra_list = []
    inter_list = []
    lmin_list = []
    cond_list = []
    det_list = []
    kappa_list = []
    n_eff_list = []

    for ui_idx, u in enumerate(eval_users):
        up = np.where(profile_uid == u)[0]
        ut = np.where(test_uid == u)[0]
        if len(up) < 5 or len(ut) < 1:
            continue

        zp = z_profile[up]   # (n_p, z_dim)
        zt = z_test[ut]      # (n_t, z_dim)
        all_u = np.concatenate([zp, zt], axis=0)

        # Intra distance (median pairwise within user)
        if len(all_u) > 1:
            sub = all_u[:min(len(all_u), 30)]
            d = np.linalg.norm(sub[:, None] - sub[None, :], axis=2)
            intra_d = float(np.median(d[np.triu_indices_from(d, k=1)]))
        else:
            intra_d = 0.0

        # Inter-user distance: random other users, take median dist to user centroid
        mu_u = all_u.mean(axis=0)
        other_users = rng.choice(eval_users, size=min(50, len(eval_users) - 1),
                                  replace=False)
        other_users = other_users[other_users != u]
        inter_ds = []
        for ou in other_users:
            op = np.where(profile_uid == ou)[0]
            ot = np.where(test_uid == ou)[0]
            if len(op) < 1 or len(ot) < 1:
                continue
            oo = np.concatenate([z_profile[op], z_test[ot]], axis=0)
            mu_o = oo.mean(axis=0)
            inter_ds.append(float(np.linalg.norm(mu_u - mu_o)))
        if len(inter_ds) < 5:
            continue
        inter_d = float(np.median(inter_ds))

        sep = intra_d / (inter_d + 1e-9)
        sep_list.append(sep)
        intra_list.append(intra_d)
        inter_list.append(inter_d)

        # λ_min, cond, det, κ on test sents only (held-out, n_t>=1)
        if len(zt) >= 2:
            centered = zt - zt.mean(axis=0)
            cov = (centered.T @ centered) / max(len(zt) - 1, 1)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 1e-12)
            lmin = float(eigvals.min())
            lmax = float(eigvals.max())
            cond = float(lmax / (lmin + 1e-12))
            det = float(np.prod(eigvals))
            tr = float(eigvals.sum())
            # κ = tr² / det (condition number proxy)
            kappa = float(tr * tr / max(det, 1e-12))
            lmin_list.append(lmin)
            cond_list.append(cond)
            det_list.append(det)
            kappa_list.append(kappa)
            n_eff_list.append(len(zt))

        if (ui_idx + 1) % 1000 == 0:
            log(f"  done {ui_idx+1}/{len(eval_users)} users, "
                f"sep_med so far = {np.median(sep_list):.3f}")

    def med(xs):
        return float(np.median(xs)) if xs else None

    return {
        "n_users_evaluated": len(sep_list),
        "separation_med": med(sep_list),
        "intra_spread_med": med(intra_list),
        "inter_user_dist_med": med(inter_list),
        "lambda_min_med": med(lmin_list),
        "lambda_min_pos_frac": (np.array(lmin_list) > 0).mean() if lmin_list else None,
        "cond_med": med(cond_list),
        "det_ratio_med": med([d / (n + 1e-9) for d, n in zip(det_list, n_eff_list)]),
        "kappa_med": med(kappa_list),
        "n_test_sents_med": med(n_eff_list),
    }


def main():
    t0 = time.time()
    log("loading strict_embeddings.npz")
    z = np.load(CACHE_DIR / "strict_embeddings.npz")
    z_profile = z["z_profile"]
    z_test = z["z_test"]
    log(f"  z_profile {z_profile.shape}, z_test {z_test.shape}")

    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)
    log(f"  n_users (in profile) = {len(user_n_sents)}, "
        f"sum profile sents = {sum(user_n_sents):,}")

    metrics = per_user_metrics(z_profile, z_test, user_n_sents,
                                sample_size=5000)

    log("\nMetrics summary:")
    for k, v in metrics.items():
        log(f"  {k}: {v}")

    with open(OUT_DIR / "pre_encoder_h30_supcon_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"\nwrote → {OUT_DIR / 'pre_encoder_h30_supcon_metrics.json'}")
    log(f"\nALL DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()