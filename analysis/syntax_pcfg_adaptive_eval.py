"""Stage 04 — Adaptive (variance-driven Cosine vs Maha-full) evaluator.

REPLACES 04_gaussian/run_adaptive_encoder.py (the old encoder-training variant).

Reads Stage 03 stage_strict3() output (strict3_embeddings.npz) and runs
held-out evaluation with τ-sweep over per-user variance to pick between
cosine attribution (low-variance users) and Mahalanobis-full (high-variance
users), plus a soft-mixture variant.

Inputs:
  pcfg_cache/strict3_embeddings.npz (z_profile, z_val, z_test, *_idx, uid_list)
  pcfg_cache/user_n_sents.json (uid order)
  result/02_user_review_sentence_extract/asin_to_users.json (ASIN cohort gates)

Outputs:
  result/04_gaussian/syntax_pcfg_adaptive.json (τ sweep + test eval report)

Pipeline stage chain:
  Stage 03 stage_strict3() → strict3_embeddings.npz
        ↓
  Stage 04 fit_per_user_gaussian.py → user_gaussian_stats.json (production Gaussian)
        ↓
  Analysis syntax_pcfg_adaptive_eval.py → syntax_pcfg_adaptive.json (held-out eval)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
OUT_DIR = REPO_ROOT / "result/04_gaussian"
OUT_PATH = OUT_DIR / "syntax_pcfg_adaptive.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 硬编码配置 (Rule 3)
SEED = 42
ABLATION_EPS = 1e-3
ABLATION_FULL_SHRINK = 0.1
ADAPTIVE_SOFT_SCALE = 0.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_strict3():
    """Load Stage 03 strict3_embeddings.npz."""
    npz_path = CACHE_DIR / "strict3_embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"missing: {npz_path} (run 03_spacy_encode.stage_strict3() first)")
    npz = np.load(npz_path, allow_pickle=False)
    return {
        "z_profile": np.asarray(npz["z_profile"], dtype=np.float32),
        "z_val": np.asarray(npz["z_val"], dtype=np.float32),
        "z_test": np.asarray(npz["z_test"], dtype=np.float32),
        "profile_idx": np.asarray(npz["profile_idx"], dtype=np.int64),
        "val_idx": np.asarray(npz["val_idx"], dtype=np.int64),
        "test_idx": np.asarray(npz["test_idx"], dtype=np.int64),
        "uid_list": [str(u) for u in npz["uid_list"]],
    }


def fit_per_user(z_profile, profile_idx, profile_dict, uid_list, z_dim,
                 eps=ABLATION_EPS, full_shrink=ABLATION_FULL_SHRINK):
    """Fit per-user (μ_u, σ²_diag, Σ_full) on profile subset."""
    n_users = len(uid_list)
    mu = np.zeros((n_users, z_dim), dtype=np.float64)
    var_diag = np.zeros((n_users, z_dim), dtype=np.float64)
    sigma_full = np.zeros((n_users, z_dim, z_dim), dtype=np.float64)
    n_prof = np.zeros(n_users, dtype=np.int64)
    p2l = {g: l for l, g in enumerate(profile_idx)}
    for ui, uid in enumerate(uid_list):
        idxs = profile_dict.get(uid, [])
        n_prof[ui] = len(idxs)
        if not idxs:
            continue
        z = z_profile[[p2l[i] for i in idxs]]
        mu[ui] = z.mean(axis=0)
        if len(idxs) >= 2:
            var_diag[ui] = z.var(axis=0, ddof=1)
            centered = z - mu[ui]
            cov = (centered.T @ centered) / max(len(idxs) - 1, 1)
            tr = np.trace(cov) / z_dim
            sigma_full[ui] = ((1 - full_shrink) * cov
                              + full_shrink * tr * np.eye(z_dim))
    sigma_full_reg = sigma_full + eps * np.eye(z_dim)
    try:
        inv_sigma_full = np.linalg.inv(sigma_full_reg)
    except np.linalg.LinAlgError:
        inv_sigma_full = np.linalg.pinv(sigma_full_reg)
    user_var = var_diag.mean(axis=1)
    return mu, var_diag, sigma_full, inv_sigma_full, user_var, n_prof


def compute_score_matrices(z_q_arr, mu, inv_sigma_full, z_dim, n_users,
                           device, chunk=1024, user_chunk=256):
    """Return (cos_mat, maha_mat), each (n_q, n_users)."""
    if z_q_arr.ndim == 1:
        z_q_arr = z_q_arr.reshape(1, -1)
    if z_q_arr.ndim > 2:
        z_q_arr = z_q_arr.reshape(-1, z_q_arr.shape[-1])
    n = len(z_q_arr)
    cos_mat = np.zeros((n, n_users), dtype=np.float32)
    maha_mat = np.zeros((n, n_users), dtype=np.float32)
    mu32 = mu.astype(np.float32)
    inv_S32 = inv_sigma_full.astype(np.float32)
    mu_norm32 = mu32 / (np.linalg.norm(mu32, axis=1, keepdims=True) + 1e-8)
    mu_inv_mu32 = np.einsum("ud,ude,ue->u", mu32, inv_S32, mu32)
    inv_S_dev = torch.from_numpy(inv_S32).to(device)
    mu_dev = torch.from_numpy(mu32).to(device)
    mu_inv_mu_dev = torch.from_numpy(mu_inv_mu32.copy()).to(device)
    for i in range(0, n, chunk):
        cz = z_q_arr[i:i + chunk].astype(np.float32)
        cz_n = cz / (np.linalg.norm(cz, axis=1, keepdims=True) + 1e-8)
        cos_mat[i:i + chunk] = cz_n @ mu_norm32.T
        cz_dev = torch.from_numpy(cz).to(device)
        for u_start in range(0, n_users, user_chunk):
            u_end = min(u_start + user_chunk, n_users)
            inv_S_uc = inv_S_dev[u_start:u_end]
            mu_uc = mu_dev[u_start:u_end]
            diff = cz_dev.unsqueeze(1) - mu_uc.unsqueeze(0)
            mahal = torch.einsum("cud,ude,cue->cu", diff, inv_S_uc, diff)
            mahal = mahal + mu_inv_mu_dev[u_start:u_end].unsqueeze(0)
            maha_mat[i:i + chunk, u_start:u_end] = (
                (-mahal).cpu().numpy().astype(np.float32))
    del inv_S_dev, mu_dev, mu_inv_mu_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cos_mat, maha_mat


def build_uid_for_split(uid_list, split_dict, split_idx, split_local):
    """Map local row position in split → uid index."""
    uid_for = np.zeros(len(split_idx), dtype=np.int32)
    for ui, uid in enumerate(uid_list):
        for gi in split_dict[uid]:
            if gi in split_local:
                uid_for[split_local[gi]] = ui
    return uid_for


def run():
    t0 = time.time()
    log("=== Stage 04 — Adaptive evaluator (strict3-based, no encoder training) ===")

    data = load_strict3()
    z_profile = data["z_profile"]
    z_val = data["z_val"]
    z_test = data["z_test"]
    profile_idx = data["profile_idx"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    uid_list = data["uid_list"]
    n_users = len(uid_list)
    z_dim = z_profile.shape[1]
    log(f"loaded: n_users={n_users}, z_dim={z_dim}, "
        f"profile={len(profile_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # 重建 per-user profile/val/test 字典 (从 global idx)
    # 利用 user_n_sents + cumulative offsets
    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)
    user_off = np.zeros(n_users + 1, dtype=np.int64)
    user_off[1:] = np.cumsum(user_n_sents)

    def idx_to_uid(g):
        return int(np.searchsorted(user_off[1:], g, side="right"))

    # 重建 split_dict (uid → [global idx]) for per-user profiling
    profile_dict = {uid: [] for uid in uid_list}
    val_dict = {uid: [] for uid in uid_list}
    test_dict = {uid: [] for uid in uid_list}
    for gi in profile_idx:
        uid = uid_list[idx_to_uid(int(gi))]
        profile_dict[uid].append(int(gi))
    for gi in val_idx:
        uid = uid_list[idx_to_uid(int(gi))]
        val_dict[uid].append(int(gi))
    for gi in test_idx:
        uid = uid_list[idx_to_uid(int(gi))]
        test_dict[uid].append(int(gi))

    # === Step 1: per-user (μ, σ², Σ_full) on profile ===
    mu, var_diag, sigma_full, inv_sigma_full, user_var, n_prof = fit_per_user(
        z_profile, profile_idx, profile_dict, uid_list, z_dim)
    log(f"per-user profile σ² median={np.median(user_var):.4f}, "
        f"min={user_var.min():.4f}, max={user_var.max():.4f}, "
        f"profile n/user avg={n_prof.mean():.1f}")

    # === Step 2: τ sweep on val ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  building val score matrices ...")
    t_sm = time.time()
    cos_val, maha_val = compute_score_matrices(
        z_val, mu, inv_sigma_full, z_dim, n_users, device)
    log(f"    cos_val {cos_val.shape}, maha_val {maha_val.shape} "
        f"({time.time()-t_sm:.0f}s)")

    p2l = {g: l for l, g in enumerate(profile_idx)}
    v2l = {g: l for l, g in enumerate(val_idx)}
    t2l = {g: l for l, g in enumerate(test_idx)}
    q2u_val = build_uid_for_split(uid_list, val_dict, val_idx, v2l)
    q2u_test = build_uid_for_split(uid_list, test_dict, test_idx, t2l)

    var_sorted = np.sort(user_var)
    tau_candidates = np.concatenate([
        [var_sorted[0] - 1e-6],
        np.quantile(var_sorted, np.linspace(0.05, 0.95, 19)),
        [var_sorted[-1] + 1e-6],
    ])
    tau_candidates = np.unique(np.round(tau_candidates, 6))

    val_curve = []
    chance_val = 1.0 / n_users
    for tau in tau_candidates:
        n_high_maha = int((user_var > tau).sum())
        if n_high_maha == 0:
            pred = cos_val.argmax(axis=1)
        elif n_high_maha == n_users:
            pred = maha_val.argmax(axis=1)
        else:
            use_maha = user_var > tau
            adapt = np.where(use_maha[None, :], maha_val, cos_val)
            pred = adapt.argmax(axis=1)
        c = int((pred == q2u_val).sum())
        acc = c / max(len(pred), 1)
        val_curve.append({"tau": float(tau),
                          "n_high_maha": n_high_maha,
                          "acc": acc,
                          "lift": acc / chance_val,
                          "n_correct": c, "n_total": len(q2u_val)})
    val_curve.sort(key=lambda r: -r["acc"])
    log(f"  best val τ: {val_curve[0]['tau']:.4f} "
        f"(acc={val_curve[0]['acc']*100:.2f}%, lift={val_curve[0]['lift']:.2f}x, "
        f"n_high_maha={val_curve[0]['n_high_maha']})")
    val_all_cos = next((r for r in val_curve if r["n_high_maha"] == 0), None)
    val_all_maha = next((r for r in val_curve if r["n_high_maha"] == n_users), None)
    log(f"  val all-cosine: lift={val_all_cos['lift']:.2f}x "
        f"({val_all_cos['acc']*100:.2f}%)")
    log(f"  val all-maha:   lift={val_all_maha['lift']:.2f}x "
        f"({val_all_maha['acc']*100:.2f}%)")

    tau_star = val_curve[0]["tau"]

    # === Step 3: test held-out evaluation ===
    chance = 1.0 / n_users
    log(f"\n=== TEST evaluation (held-out, encoder & τ 都未见过) ===")

    log(f"  building test score matrices ...")
    t_sm = time.time()
    cos_test, maha_test = compute_score_matrices(
        z_test, mu, inv_sigma_full, z_dim, n_users, device)
    log(f"    cos_test {cos_test.shape}, maha_test {maha_test.shape} "
        f"({time.time()-t_sm:.0f}s)")

    def test_eval(method: str):
        if method == "cosine":
            pred = cos_test.argmax(axis=1)
        elif method == "maha_full":
            pred = maha_test.argmax(axis=1)
        elif method == "adaptive":
            use_maha = user_var > tau_star
            adapt = np.where(use_maha[None, :], maha_test, cos_test)
            pred = adapt.argmax(axis=1)
        elif method == "soft_mixture":
            alpha = 1.0 / (1.0 + np.exp(
                (user_var - tau_star) / ADAPTIVE_SOFT_SCALE))
            cos_n = (cos_test + 1.0) / 2.0
            maha_n = np.exp(-np.maximum(maha_test, 0) / z_dim)
            score = alpha[None, :] * cos_n + (1.0 - alpha)[None, :] * maha_n
            pred = score.argmax(axis=1)
        else:
            raise ValueError(method)
        c = int((pred == q2u_test).sum())
        total = len(pred)
        acc = c / max(total, 1)
        return {"acc": acc, "lift": acc / chance,
                "n_correct": c, "n_total": total}

    test_results = {}
    for m in ["cosine", "maha_full", "adaptive", "soft_mixture"]:
        r = test_eval(m)
        test_results[m] = r
        log(f"  test {m:>15s}: {r['n_correct']}/{r['n_total']} "
            f"= {r['acc']*100:.2f}% lift {r['lift']:.2f}x")

    base_best = max(test_results["cosine"]["lift"],
                    test_results["maha_full"]["lift"])
    adaptive_lift = test_results["adaptive"]["lift"]
    soft_lift = test_results["soft_mixture"]["lift"]
    log(f"\n  Δ(adaptive - best of cosine/maha_full): "
        f"{adaptive_lift - base_best:+.2f}x")
    log(f"  Δ(soft_mixture - best): {soft_lift - base_best:+.2f}x")

    summary = {
        "config": {
            "z_dim": z_dim, "n_users": n_users,
            "n_profile": int(len(profile_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "split": "strict3_sha1_hash_50_20_30",
            "gauss_eps": ABLATION_EPS,
            "full_shrink_alpha": ABLATION_FULL_SHRINK,
            "soft_scale": ADAPTIVE_SOFT_SCALE,
        },
        "no_leakage": True,
        "encoder_seen_test": False,
        "tau_seen_test": False,
        "chance": chance,
        "val_curve_top10": val_curve[:10],
        "val_all_cosine": val_all_cos,
        "val_all_maha": val_all_maha,
        "tau_selected": float(tau_star),
        "test_results": test_results,
        "delta_vs_best": {
            "adaptive_minus_best_point": adaptive_lift - base_best,
            "soft_minus_best_point": soft_lift - base_best,
        },
        "user_variance": {
            "median": float(np.median(user_var)),
            "mean": float(user_var.mean()),
            "p25": float(np.percentile(user_var, 25)),
            "p75": float(np.percentile(user_var, 75)),
            "per_user": {uid_list[ui]: float(user_var[ui])
                         for ui in range(n_users)},
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote → {OUT_PATH}")
    log(f"=== Total: {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    run()