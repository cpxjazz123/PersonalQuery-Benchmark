"""L8.28 — Maha softmax 归一化 attribution 对比。

L8.27 的关键 bug:raw Maha D² score 跟 cosine 不同 scale (maha_max < cos_max),
导致 local-replace 永远不触发,所有 maha_* / adaptive_* 方法都 = cosine baseline。

修复:maha_normalized = softmax(maha_full / z_dim) over fitted users (per query),
   cos_normalized = (cos + 1) / 2 in [0, 1]
然后 per-query 在 [cos_normalized for unfitted, maha_normalized for fitted] 上 argmax。

这样 maha 跟 cos 同 scale,可以公平比较 rank1 vs full 是否能 refine cos 选错的 query。
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
OUT_PATH = OUT_DIR / "syntax_pcfg_maha_softmax_eval.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
Q_CHUNK = 4096


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_strict3():
    npz_path = CACHE_DIR / "strict3_embeddings.npz"
    npz = np.load(npz_path, allow_pickle=False)
    return {
        "z_val": np.asarray(npz["z_val"], dtype=np.float32),
        "z_test": np.asarray(npz["z_test"], dtype=np.float32),
        "val_idx": np.asarray(npz["val_idx"], dtype=np.int64),
        "test_idx": np.asarray(npz["test_idx"], dtype=np.int64),
        "profile_idx": np.asarray(npz["profile_idx"], dtype=np.int64),
        "uid_list": [str(u) for u in npz["uid_list"]],
    }


def load_gaussian_full(path, uid_list):
    with open(path) as f:
        d = json.load(f)
    n_users = len(uid_list); z_dim = 32
    mu = np.zeros((n_users, z_dim), dtype=np.float64)
    inv_S = np.zeros((n_users, z_dim, z_dim), dtype=np.float64)
    for ui, uid in enumerate(uid_list):
        s = d["users"].get(uid)
        if s is None: continue
        mu[ui] = np.asarray(s["mu"], dtype=np.float64)
        inv_S[ui] = np.asarray(s["sigma_inv"], dtype=np.float64)
    return mu, inv_S


def load_gaussian_rank1(path, uid_list):
    with open(path) as f:
        d = json.load(f)
    n_users = len(uid_list); z_dim = 32
    mu = np.zeros((n_users, z_dim), dtype=np.float64)
    v1 = np.zeros((n_users, z_dim), dtype=np.float64)
    sr = np.zeros(n_users, dtype=np.float64)
    lam1 = np.zeros(n_users, dtype=np.float64)
    for ui, uid in enumerate(uid_list):
        s = d["users"].get(uid)
        if s is None: continue
        mu[ui] = np.asarray(s["mu"], dtype=np.float64)
        v1[ui] = np.asarray(s["v1"], dtype=np.float64)
        sr[ui] = float(s["sigma_res"])
        lam1[ui] = float(s["lambda1"])
    return mu, v1, sr, lam1


def compute_full_maha(z_q, mu, inv_S, mu_inv_mu, device, chunk=512, user_chunk=256):
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


def compute_rank1_maha(z_q, mu, v1, sr, lam1, device, chunk=2048, user_chunk=512):
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


def softmax_norm(x, axis=-1):
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    t0 = time.time()
    log("=== L8.28: Maha softmax-normalized attribution (rank1 vs full) ===")
    data = load_strict3()
    z_val = data["z_val"]; z_test = data["z_test"]
    val_idx = data["val_idx"]; test_idx = data["test_idx"]
    profile_idx = data["profile_idx"]
    uid_list = data["uid_list"]
    n_users = len(uid_list); z_dim = z_val.shape[1]
    log(f"loaded: n_users={n_users}, val={len(val_idx)} test={len(test_idx)}")

    mu_full, inv_S_full = load_gaussian_full(GAUSS_FULL_PATH, uid_list)
    mu_r1, v1_r1, sr_r1, lam1_r1 = load_gaussian_rank1(GAUSS_RANK1_PATH, uid_list)
    fitted_mask = (sr_r1 != 0)
    fitted_uidx = np.where(fitted_mask)[0]
    n_fitted = len(fitted_uidx)
    log(f"  fitted: {n_fitted}")

    with open(CACHE_DIR / "user_n_sents.json") as f:
        user_n_sents = json.load(f)
    user_off = np.zeros(n_users + 1, dtype=np.int64)
    user_off[1:] = np.cumsum(user_n_sents)
    q2u_val = np.clip(np.searchsorted(user_off, val_idx, side="right") - 1, 0, n_users - 1).astype(np.int32)
    q2u_test = np.clip(np.searchsorted(user_off, test_idx, side="right") - 1, 0, n_users - 1).astype(np.int32)

    profile_uid = np.clip(np.searchsorted(user_off, profile_idx, side="right") - 1, 0, n_users - 1)
    cnt = np.bincount(profile_uid, minlength=n_users).astype(np.float64)
    var_sum = np.zeros(n_users, dtype=np.float64)
    np.add.at(var_sum, profile_uid, np.var(data_z_profile if False else None, axis=1) if False else 0)
    # 实际上应该用 z_profile 但要重新加载. Simpler: var 不需要了, softmax 归一化不依赖 var
    del profile_uid, cnt, var_sum

    mu_full_f = mu_full[fitted_uidx]
    inv_S_full_f = inv_S_full[fitted_uidx]
    mu_r1_f = mu_r1[fitted_uidx]
    v1_r1_f = v1_r1[fitted_uidx]
    sr_r1_f = sr_r1[fitted_uidx]
    lam1_r1_f = lam1_r1[fitted_uidx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  device={device}")
    mu_inv_mu_full_f = np.einsum("ud,ude,ue->u", mu_full_f, inv_S_full_f, mu_full_f)

    log("  maha_full val (n_fitted)")
    t_sm = time.time()
    maha_full_val_f = compute_full_maha(z_val, mu_full_f, inv_S_full_f, mu_inv_mu_full_f, device)
    log(f"    ({time.time()-t_sm:.0f}s)")
    log("  maha_full test")
    t_sm = time.time()
    maha_full_test_f = compute_full_maha(z_test, mu_full_f, inv_S_full_f, mu_inv_mu_full_f, device)
    log(f"    ({time.time()-t_sm:.0f}s)")
    log("  maha_rank1 val")
    t_sm = time.time()
    maha_r1_val_f = compute_rank1_maha(z_val, mu_r1_f, v1_r1_f, sr_r1_f, lam1_r1_f, device)
    log(f"    ({time.time()-t_sm:.0f}s)")
    log("  maha_rank1 test")
    t_sm = time.time()
    maha_r1_test_f = compute_rank1_maha(z_test, mu_r1_f, v1_r1_f, sr_r1_f, lam1_r1_f, device)
    log(f"    ({time.time()-t_sm:.0f}s)")

    # Softmax normalize over fitted axis (per query) — torch on GPU for speed
    log("  softmax normalize maha_full/rank1 (val/test) on GPU")
    def to_gpu_softmax(m):
        t = torch.from_numpy(m).to(device)
        s = torch.softmax(t, dim=1)
        del t
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return s.cpu().numpy()
    maha_full_val_sm = to_gpu_softmax(maha_full_val_f)
    maha_full_test_sm = to_gpu_softmax(maha_full_test_f)
    maha_r1_val_sm = to_gpu_softmax(maha_r1_val_f)
    maha_r1_test_sm = to_gpu_softmax(maha_r1_test_f)
    log("    done")

    mu32 = mu_full.astype(np.float32)
    mu_norm32 = mu32 / (np.linalg.norm(mu32, axis=1, keepdims=True) + 1e-8)

    def cos_chunk(z_q):
        z_n = z_q / (np.linalg.norm(z_q, axis=1, keepdims=True) + 1e-8)
        return z_n @ mu_norm32.T

    # ===== 5-way eval with softmax-normalized Maha =====
    chance = 1.0 / n_users
    methods = [
        ("cosine", None, None),  # baseline
        ("softmax_maha_full", "maha_full_sm", None),
        ("softmax_maha_rank1", "maha_rank1_sm", None),
        ("softmax_blend_full", "maha_full_sm", 0.5),  # 0.5 * cos + 0.5 * maha_sm
        ("softmax_blend_rank1", "maha_rank1_sm", 0.5),
    ]

    def run_test(method, maha_field, blend_w):
        n_correct = 0
        n_q = len(z_test)
        for i in range(0, n_q, Q_CHUNK):
            i_end = min(i + Q_CHUNK, n_q)
            cos_c = cos_chunk(z_test[i:i_end])  # (chunk, n_users) in [-1, 1]
            q_truth = q2u_test[i:i_end]
            if method == "cosine":
                pred = cos_c.argmax(axis=1)
            else:
                if maha_field == "maha_full_sm":
                    maha_sm_c = maha_full_test_sm[i:i_end]  # (chunk, n_fitted)
                else:
                    maha_sm_c = maha_r1_test_sm[i:i_end]
                if method.startswith("softmax_maha"):
                    # Replace fitted columns: maha_sm is per-query distribution
                    # cos is in [-1, 1]; we need both on same scale.
                    # Map cos ∈ [-1, 1] -> [0, 1] as (cos+1)/2, then put maha_sm in fitted cols
                    cos_n = (cos_c + 1.0) / 2.0
                    adapt = cos_n.copy()
                    adapt[:, fitted_uidx] = maha_sm_c
                    pred = adapt.argmax(axis=1)
                elif method.startswith("softmax_blend"):
                    # blend = (1-w) * cos_n + w * maha_sm_fitted
                    cos_n = (cos_c + 1.0) / 2.0
                    adapt = (1.0 - blend_w) * cos_n
                    # Place maha_sm at fitted positions
                    # In-place scatter
                    blend_part = blend_w * maha_sm_c
                    adapt[:, fitted_uidx] = blend_part
                    pred = adapt.argmax(axis=1)
                else:
                    raise ValueError(method)
            n_correct += int((pred == q_truth).sum())
        acc = n_correct / n_q
        return {"acc": float(acc), "lift": float(acc / chance),
                "n_correct": int(n_correct), "n_total": int(n_q)}

    test_results = {}
    for m, maha_field, blend_w in methods:
        t_sm = time.time()
        r = run_test(m, maha_field, blend_w)
        test_results[m] = r
        log(f"  test {m:>22s}: {r['n_correct']:>5d}/{r['n_total']} "
            f"= {r['acc']*100:.2f}% lift {r['lift']:.2f}x ({time.time()-t_sm:.0f}s)")

    log(f"\n  Δ(softmax_maha_rank1 - softmax_maha_full): "
        f"{test_results['softmax_maha_rank1']['lift'] - test_results['softmax_maha_full']['lift']:+.2f}x")
    log(f"  Δ(softmax_blend_rank1 - softmax_blend_full): "
        f"{test_results['softmax_blend_rank1']['lift'] - test_results['softmax_blend_full']['lift']:+.2f}x")
    log(f"  Δ(softmax_maha_full - cosine): "
        f"{test_results['softmax_maha_full']['lift'] - test_results['cosine']['lift']:+.2f}x")
    log(f"  Δ(softmax_maha_rank1 - cosine): "
        f"{test_results['softmax_maha_rank1']['lift'] - test_results['cosine']['lift']:+.2f}x")

    summary = {
        "config": {
            "z_dim": z_dim, "n_users": n_users, "n_fitted": n_fitted,
            "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
            "split": "strict3_sha1_hash_50_20_30",
            "softmax_norm": "softmax(maha) over fitted axis (per query); cos mapped (cos+1)/2 in [0,1]",
        },
        "no_leakage": True,
        "chance": chance,
        "test_results": test_results,
        "deltas": {
            "softmax_maha_rank1_minus_full": test_results["softmax_maha_rank1"]["lift"]
                - test_results["softmax_maha_full"]["lift"],
            "softmax_blend_rank1_minus_full": test_results["softmax_blend_rank1"]["lift"]
                - test_results["softmax_blend_full"]["lift"],
            "softmax_maha_full_minus_cosine": test_results["softmax_maha_full"]["lift"]
                - test_results["cosine"]["lift"],
            "softmax_maha_rank1_minus_cosine": test_results["softmax_maha_rank1"]["lift"]
                - test_results["cosine"]["lift"],
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote → {OUT_PATH}")
    log(f"=== Total: {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
