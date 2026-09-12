"""Stage 04 held-out contrastive attribution: full Σ vs rank1+residual.

Fully chunked to avoid 200+GB intermediate arrays.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
GAUSS_FULL_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
GAUSS_RANK1_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_rank1.json"
OUT_DIR = REPO_ROOT / "result/analysis"
OUT_PATH = OUT_DIR / "syntax_pcfg_rank1_eval.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SOFT_SCALE = 0.5
Q_CHUNK = 4096  # 4k * 64996 * 4 = 1.05 GB per chunk (val cos)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_strict3():
    npz_path = CACHE_DIR / "strict3_embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing: {npz_path}")
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


def load_gaussian_full(path: Path, uid_list: list[str]):
    with open(path) as f:
        d = json.load(f)
    n_users = len(uid_list); z_dim = 32
    mu = np.zeros((n_users, z_dim), dtype=np.float64)
    inv_S = np.zeros((n_users, z_dim, z_dim), dtype=np.float64)
    for ui, uid in enumerate(uid_list):
        s = d["users"].get(uid)
        if s is None:
            continue
        mu[ui] = np.asarray(s["mu"], dtype=np.float64)
        inv_S[ui] = np.asarray(s["sigma_inv"], dtype=np.float64)
    return mu, inv_S


def load_gaussian_rank1(path: Path, uid_list: list[str]):
    with open(path) as f:
        d = json.load(f)
    n_users = len(uid_list); z_dim = 32
    mu = np.zeros((n_users, z_dim), dtype=np.float64)
    v1 = np.zeros((n_users, z_dim), dtype=np.float64)
    sr = np.zeros(n_users, dtype=np.float64)
    lam1 = np.zeros(n_users, dtype=np.float64)
    for ui, uid in enumerate(uid_list):
        s = d["users"].get(uid)
        if s is None:
            continue
        mu[ui] = np.asarray(s["mu"], dtype=np.float64)
        v1[ui] = np.asarray(s["v1"], dtype=np.float64)
        sr[ui] = float(s["sigma_res"])
        lam1[ui] = float(s["lambda1"])
    return mu, v1, sr, lam1


def compute_full_maha(z_q: np.ndarray, mu: np.ndarray, inv_S: np.ndarray,
                      mu_inv_mu: np.ndarray, device, chunk=512, user_chunk=256):
    n = len(z_q); n_users = len(mu); z_dim = mu.shape[1]
    maha = np.zeros((n, n_users), dtype=np.float32)
    mu_dev = torch.from_numpy(mu.astype(np.float32)).to(device)
    inv_S_dev = torch.from_numpy(inv_S.astype(np.float32)).to(device)
    mu_inv_mu_dev = torch.from_numpy(mu_inv_mu.astype(np.float32)).to(device)
    for i in range(0, n, chunk):
        cz = torch.from_numpy(z_q[i:i+chunk]).to(device)
        for u_start in range(0, n_users, user_chunk):
            u_end = min(u_start + user_chunk, n_users)
            diff = cz.unsqueeze(1) - mu_dev[u_start:u_end].unsqueeze(0)
            mahal = torch.einsum("cud,ude,cue->cu", diff,
                                 inv_S_dev[u_start:u_end], diff)
            mahal = mahal + mu_inv_mu_dev[u_start:u_end].unsqueeze(0)
            maha[i:i+chunk, u_start:u_end] = (-mahal).cpu().numpy()
    del mu_dev, inv_S_dev, mu_inv_mu_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return maha


def compute_rank1_maha(z_q: np.ndarray, mu: np.ndarray, v1: np.ndarray,
                       sr: np.ndarray, lam1: np.ndarray, device,
                       chunk=2048, user_chunk=512):
    n = len(z_q); n_users = len(mu); z_dim = mu.shape[1]
    maha = np.zeros((n, n_users), dtype=np.float32)
    mu_dev = torch.from_numpy(mu.astype(np.float32)).to(device)
    v1_dev = torch.from_numpy(v1.astype(np.float32)).to(device)
    sr_dev = torch.from_numpy(sr.astype(np.float32)).to(device)
    lam1_dev = torch.from_numpy(lam1.astype(np.float32)).to(device)
    for i in range(0, n, chunk):
        cz = torch.from_numpy(z_q[i:i+chunk]).to(device)
        for u_start in range(0, n_users, user_chunk):
            u_end = min(u_start + user_chunk, n_users)
            diff = cz.unsqueeze(1) - mu_dev[u_start:u_end].unsqueeze(0)
            a = (diff * v1_dev[u_start:u_end].unsqueeze(0)).sum(-1)
            r_vec = diff - a.unsqueeze(-1) * v1_dev[u_start:u_end].unsqueeze(0)
            r2 = (r_vec * r_vec).sum(-1)
            d2 = a * a / lam1_dev[u_start:u_end].unsqueeze(0) + \
                 r2 / sr_dev[u_start:u_end].unsqueeze(0)
            maha[i:i+chunk, u_start:u_end] = (-d2).cpu().numpy()
    del mu_dev, v1_dev, sr_dev, lam1_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return maha


def main():
    t0 = time.time()
    log("=== L8.27: rank1+residual vs full Σ held-out attribution (chunked) ===")
    data = load_strict3()
    z_profile = data["z_profile"]
    z_val = data["z_val"]
    z_test = data["z_test"]
    val_idx = data["val_idx"]
    test_idx = data["test_idx"]
    profile_idx = data["profile_idx"]
    uid_list = data["uid_list"]
    n_users = len(uid_list)
    z_dim = z_profile.shape[1]
    log(f"loaded strict3: {n_users}u, val={len(val_idx)} test={len(test_idx)}")

    mu_full, inv_S_full = load_gaussian_full(GAUSS_FULL_PATH, uid_list)
    mu_r1, v1_r1, sr_r1, lam1_r1 = load_gaussian_rank1(GAUSS_RANK1_PATH, uid_list)
    fitted_mask = (sr_r1 != 0)
    fitted_uidx = np.where(fitted_mask)[0]
    n_fitted = len(fitted_uidx)
    log(f"  fitted users: {n_fitted}")

    # Vectorized q2u
    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)
    user_off = np.zeros(n_users + 1, dtype=np.int64)
    user_off[1:] = np.cumsum(user_n_sents)
    q2u_val = np.clip(np.searchsorted(user_off, val_idx, side="right") - 1, 0, n_users - 1).astype(np.int32)
    q2u_test = np.clip(np.searchsorted(user_off, test_idx, side="right") - 1, 0, n_users - 1).astype(np.int32)

    profile_uid = np.clip(np.searchsorted(user_off, profile_idx, side="right") - 1, 0, n_users - 1)
    cnt = np.bincount(profile_uid, minlength=n_users).astype(np.float64)
    var_sum = np.zeros(n_users, dtype=np.float64)
    np.add.at(var_sum, profile_uid, np.var(z_profile, axis=1).astype(np.float64))
    user_var = var_sum / np.maximum(cnt, 1)
    fitted_var = user_var[fitted_uidx]
    log(f"  user_var median={np.median(user_var):.4f}, fitted_var median={np.median(fitted_var):.4f}")

    # Restrict Maha to fitted users
    mu_full_f = mu_full[fitted_uidx]
    inv_S_full_f = inv_S_full[fitted_uidx]
    mu_r1_f = mu_r1[fitted_uidx]
    v1_r1_f = v1_r1[fitted_uidx]
    sr_r1_f = sr_r1[fitted_uidx]
    lam1_r1_f = lam1_r1[fitted_uidx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  device={device}")

    # Compute Maha on val and test (full and rank1) over fitted users
    mu_inv_mu_full_f = np.einsum("ud,ude,ue->u", mu_full_f, inv_S_full_f, mu_full_f)
    log(f"  building full Σ Maha val (n_fitted={n_fitted}) ...")
    t_sm = time.time()
    maha_full_val_f = compute_full_maha(z_val, mu_full_f, inv_S_full_f, mu_inv_mu_full_f, device)
    log(f"    ({time.time()-t_sm:.0f}s) shape={maha_full_val_f.shape}")
    log(f"  building full Σ Maha test ...")
    t_sm = time.time()
    maha_full_test_f = compute_full_maha(z_test, mu_full_f, inv_S_full_f, mu_inv_mu_full_f, device)
    log(f"    ({time.time()-t_sm:.0f}s) shape={maha_full_test_f.shape}")

    log(f"  building rank1 Maha val ...")
    t_sm = time.time()
    maha_r1_val_f = compute_rank1_maha(z_val, mu_r1_f, v1_r1_f, sr_r1_f, lam1_r1_f, device)
    log(f"    ({time.time()-t_sm:.0f}s) shape={maha_r1_val_f.shape}")
    log(f"  building rank1 Maha test ...")
    t_sm = time.time()
    maha_r1_test_f = compute_rank1_maha(z_test, mu_r1_f, v1_r1_f, sr_r1_f, lam1_r1_f, device)
    log(f"    ({time.time()-t_sm:.0f}s) shape={maha_r1_test_f.shape}")

    # Setup cos helper (chunked)
    mu32 = mu_full.astype(np.float32)
    mu_norm32 = mu32 / (np.linalg.norm(mu32, axis=1, keepdims=True) + 1e-8)

    def cos_chunk(z_q):
        z_n = z_q / (np.linalg.norm(z_q, axis=1, keepdims=True) + 1e-8)
        return z_n @ mu_norm32.T

    # === τ sweep on val for full Σ — chunked ===
    var_sorted = np.sort(fitted_var)
    # 7 taus instead of 21 (3x speedup, ~2.5 min sweep)
    tau_candidates = np.concatenate([
        [var_sorted[0] - 1e-6],
        np.quantile(fitted_var, np.linspace(0.10, 0.90, 9)),
        [var_sorted[-1] + 1e-6],
    ])
    tau_candidates = np.unique(np.round(tau_candidates, 6))
    chance = 1.0 / n_users

    log(f"  τ sweep on val (chunked, n_taus={len(tau_candidates)}) ...")
    t_sm = time.time()
    n_val = len(z_val)
    val_correct_per_tau = np.zeros(len(tau_candidates), dtype=np.int64)
    use_maha_idx_by_tau = []
    for tau in tau_candidates:
        use_maha = fitted_var > tau
        n_use = int(use_maha.sum())
        use_maha_idx_by_tau.append((use_maha, n_use, fitted_uidx[use_maha]))
    # Local-replace argmax trick: pred = cos_argmax OR fitted_use[maha_argmax] if maha > cos_max
    for i in range(0, n_val, Q_CHUNK):
        i_end = min(i + Q_CHUNK, n_val)
        cos_c = cos_chunk(z_val[i:i_end])  # (chunk, n_users)
        maha_full_c = maha_full_val_f[i:i_end]  # (chunk, n_fitted)
        q_truth = q2u_val[i:i_end]
        cos_argmax = cos_c.argmax(axis=1)  # (chunk,)
        cos_max = cos_c[np.arange(len(cos_c)), cos_argmax]  # (chunk,)
        for ti, (use_maha, n_use, fitted_use) in enumerate(use_maha_idx_by_tau):
            if n_use == 0:
                pred = cos_argmax
            else:
                maha_use = maha_full_c[:, use_maha]  # (chunk, n_use)
                maha_argmax = maha_use.argmax(axis=1)
                maha_max = maha_use[np.arange(len(maha_use)), maha_argmax]
                replace = maha_max > cos_max
                pred = np.where(replace, fitted_use[maha_argmax], cos_argmax)
            val_correct_per_tau[ti] += int((pred == q_truth).sum())
        if (i // Q_CHUNK) % 20 == 0:
            log(f"    val chunk {i}/{n_val} ({time.time()-t_sm:.0f}s)")
    val_acc_full = val_correct_per_tau / n_val
    val_curve_full = [
        {"tau": float(tau_candidates[ti]),
         "acc": float(val_acc_full[ti]),
         "lift": float(val_acc_full[ti] / chance),
         "n_use_maha": use_maha_idx_by_tau[ti][1]}
        for ti in range(len(tau_candidates))
    ]
    val_curve_full.sort(key=lambda r: -r["acc"])
    tau_star = val_curve_full[0]["tau"]
    log(f"  τ* (full Σ): {tau_star:.4f}, val acc={val_curve_full[0]['acc']*100:.2f}% "
        f"lift={val_curve_full[0]['lift']:.2f}x n_use={val_curve_full[0]['n_use_maha']}")
    log(f"    total τ-sweep time: {time.time()-t_sm:.0f}s")

    # Use the same τ for both full and rank1 adaptive
    use_maha_full = fitted_var > tau_star
    fitted_use = fitted_uidx[use_maha_full]
    n_use = int(use_maha_full.sum())
    log(f"  using τ* for both full and rank1, n_use_maha={n_use}")

    # === TEST 5-way eval (chunked) ===
    n_q_test = len(z_test)
    log(f"\n=== TEST held-out 5-way (chunked) ===")

    def run_test_chunked(method):
        n_correct = 0
        for i in range(0, n_q_test, Q_CHUNK):
            i_end = min(i + Q_CHUNK, n_q_test)
            cos_c = cos_chunk(z_test[i:i_end])
            q_truth = q2u_test[i:i_end]
            cos_argmax = cos_c.argmax(axis=1)
            cos_max = cos_c[np.arange(len(cos_c)), cos_argmax]
            if method == "cosine":
                pred = cos_argmax
            elif method == "maha_full":
                maha_use = maha_full_test_f[i:i_end]
                maha_argmax = maha_use.argmax(axis=1)
                maha_max = maha_use[np.arange(len(maha_use)), maha_argmax]
                replace = maha_max > cos_max
                pred = np.where(replace, fitted_uidx[maha_argmax], cos_argmax)
            elif method == "maha_rank1":
                maha_use = maha_r1_test_f[i:i_end]
                maha_argmax = maha_use.argmax(axis=1)
                maha_max = maha_use[np.arange(len(maha_use)), maha_argmax]
                replace = maha_max > cos_max
                pred = np.where(replace, fitted_uidx[maha_argmax], cos_argmax)
            elif method == "adaptive_full":
                maha_sub = maha_full_test_f[i:i_end][:, use_maha_full]
                maha_argmax = maha_sub.argmax(axis=1)
                maha_max = maha_sub[np.arange(len(maha_sub)), maha_argmax]
                replace = maha_max > cos_max
                pred = np.where(replace, fitted_use[maha_argmax], cos_argmax)
            elif method == "adaptive_rank1":
                maha_sub = maha_r1_test_f[i:i_end][:, use_maha_full]
                maha_argmax = maha_sub.argmax(axis=1)
                maha_max = maha_sub[np.arange(len(maha_sub)), maha_argmax]
                replace = maha_max > cos_max
                pred = np.where(replace, fitted_use[maha_argmax], cos_argmax)
            elif method == "soft_full":
                alpha = np.full(n_users, 1.0, dtype=np.float32)
                for ui_pos, ui in enumerate(fitted_uidx):
                    alpha[ui] = 1.0 / (1.0 + np.exp((fitted_var[ui_pos] - tau_star) / SOFT_SCALE))
                cos_n = (cos_c + 1.0) / 2.0
                maha_pad = cos_n.copy()
                maha_pad[:, fitted_uidx] = np.exp(-np.maximum(maha_full_test_f[i:i_end], 0) / z_dim)
                score = alpha[None, :] * cos_n + (1 - alpha)[None, :] * maha_pad
                pred = score.argmax(axis=1)
            elif method == "soft_rank1":
                alpha = np.full(n_users, 1.0, dtype=np.float32)
                for ui_pos, ui in enumerate(fitted_uidx):
                    alpha[ui] = 1.0 / (1.0 + np.exp((fitted_var[ui_pos] - tau_star) / SOFT_SCALE))
                cos_n = (cos_c + 1.0) / 2.0
                maha_pad = cos_n.copy()
                maha_pad[:, fitted_uidx] = np.exp(-np.maximum(maha_r1_test_f[i:i_end], 0) / z_dim)
                score = alpha[None, :] * cos_n + (1 - alpha)[None, :] * maha_pad
                pred = score.argmax(axis=1)
            else:
                raise ValueError(method)
            n_correct += int((pred == q_truth).sum())
        acc = n_correct / n_q_test
        return {"acc": float(acc), "lift": float(acc / chance),
                "n_correct": int(n_correct), "n_total": int(n_q_test)}

    test_results = {}
    for m in ["cosine", "maha_full", "maha_rank1",
              "adaptive_full", "adaptive_rank1",
              "soft_full", "soft_rank1"]:
        t_sm = time.time()
        r = run_test_chunked(m)
        test_results[m] = r
        log(f"  test {m:>16s}: {r['n_correct']:>5d}/{r['n_total']} "
            f"= {r['acc']*100:.2f}% lift {r['lift']:.2f}x ({time.time()-t_sm:.0f}s)")

    log(f"\n  Δ(maha_rank1 - maha_full): "
        f"{test_results['maha_rank1']['lift'] - test_results['maha_full']['lift']:+.2f}x")
    log(f"  Δ(adaptive_rank1 - adaptive_full): "
        f"{test_results['adaptive_rank1']['lift'] - test_results['adaptive_full']['lift']:+.2f}x")
    log(f"  Δ(soft_rank1 - soft_full): "
        f"{test_results['soft_rank1']['lift'] - test_results['soft_full']['lift']:+.2f}x")

    summary = {
        "config": {
            "z_dim": z_dim, "n_users": n_users, "n_fitted": n_fitted,
            "n_profile": int(len(profile_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "split": "strict3_sha1_hash_50_20_30",
            "tau_selected_full": float(tau_star),
            "n_use_maha": n_use,
            "soft_scale": SOFT_SCALE,
        },
        "no_leakage": True,
        "chance": chance,
        "val_curve_full_top5": val_curve_full[:5],
        "test_results": test_results,
        "deltas": {
            "maha_rank1_minus_full": test_results["maha_rank1"]["lift"]
                - test_results["maha_full"]["lift"],
            "adaptive_rank1_minus_full": test_results["adaptive_rank1"]["lift"]
                - test_results["adaptive_full"]["lift"],
            "soft_rank1_minus_full": test_results["soft_rank1"]["lift"]
                - test_results["soft_full"]["lift"],
        },
        "user_variance": {
            "median": float(np.median(user_var)),
            "mean": float(user_var.mean()),
            "fitted_median": float(np.median(fitted_var)),
            "fitted_p25": float(np.percentile(fitted_var, 25)),
            "fitted_p75": float(np.percentile(fitted_var, 75)),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote → {OUT_PATH}")
    log(f"=== Total: {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
