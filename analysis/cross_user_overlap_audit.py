#!/usr/bin/env python3
"""Stage 4 cross-user overlap audit。

用户 2026-09-13 提议:定义 O_u = 其他用户的句子里落在 R_u (95% Gaussian region) 内的比例,
O_u 太高 = 该 user region 太宽,缺乏 personalized discriminability。

实现:
  - 加载 Stage 04 raw full-Σ Gaussian (mu, sigma_inv, gate_T = d2_q95)
  - 加载 strict3 frozen embeddings (z_profile + z_val + z_test)
  - 对每个 Stage 04 user u:
      background = profile+val+test 中所有 uid != u 的句子
      O_u = (background 中 D²(z, mu_u) <= gate_T_u 的比例)
  - 画 O_u 分布 + sensitivity:τ ∈ {0.05, 0.10, 0.20, 0.30, 0.50} 各自保留多少 user

输出:
  result/05_gaussian_audit/cross_user_overlap.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats.json"
EMB_PATH = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/strict3_embeddings.npz")
OUT_PATH = REPO_ROOT / "result/05_gaussian_audit/cross_user_overlap.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def maha_d2(z: np.ndarray, mu: np.ndarray, inv: np.ndarray) -> np.ndarray:
    """z: (n, 32), mu: (32,), inv: (32, 32)."""
    diff = z - mu[None, :]
    left = diff @ inv
    return np.einsum("nd,nd->n", left, diff).astype(np.float32)


def main():
    log("=== cross-user overlap audit ===")

    # Stage 04 Gaussian
    with open(STAGE04_PATH) as f:
        s4 = json.load(f)
    s4_users = s4["users"]
    s4_uids_all = list(s4_users.keys())
    log(f"  Stage 04 users: {len(s4_uids_all)}")

    # frozen embeddings (profile+val+test) + uid index
    log("  loading frozen embeddings ...")
    emb = np.load(EMB_PATH, allow_pickle=False)
    z_prof = emb["z_profile"].astype(np.float32)
    z_val = emb["z_val"].astype(np.float32)
    z_test = emb["z_test"].astype(np.float32)
    prof_uid = emb["profile_idx"]
    val_uid = emb["val_idx"]
    test_uid = emb["test_idx"]
    log(f"    profile {z_prof.shape}  val {z_val.shape}  test {z_test.shape}")

    uid_list = list(emb["uid_list"])
    log(f"    uid_list: {len(uid_list)} users")

    user_n_sents_path = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/user_n_sents.json")
    with open(user_n_sents_path) as f:
        user_n_sents = [int(x) for x in json.load(f)]

    # Build sentence-position → uid_list index mapping using cumsum boundaries.
    # For sentence at position p, owning uid_list index is the last i where
    # cumsum[n_sents][i] <= p (i.e., searchsorted with side='right' - 1).
    cum_n_sents = np.array(user_n_sents, dtype=np.int64)
    np.cumsum(cum_n_sents, out=cum_n_sents)  # in-place: cum_n_sents[i] = total sents up to i

    # Map all profile/val/test sentence positions to uid_list indices
    def map_sent_positions(pos_arr):
        # searchsorted returns first index where cum_n_sents[i] >= p (0-indexed)
        uid_idx = np.searchsorted(cum_n_sents, pos_arr, side="right") - 1
        return np.clip(uid_idx, 0, len(uid_list) - 1)

    prof_uidlist_idx = map_sent_positions(prof_uid)
    val_uidlist_idx = map_sent_positions(val_uid)
    test_uidlist_idx = map_sent_positions(test_uid)
    log(f"    uid_list idx range: prof=[{prof_uidlist_idx.min()},{prof_uidlist_idx.max()}]  "
        f"val=[{val_uidlist_idx.min()},{val_uidlist_idx.max()}]  "
        f"test=[{test_uidlist_idx.min()},{test_uidlist_idx.max()}]")

    # Stage 4 users: filter to those present in uid_list
    uidlist_pos = {uid: i for i, uid in enumerate(uid_list)}
    s4_uids = [u for u in s4_uids_all if u in uidlist_pos]
    missing = len(s4_uids_all) - len(s4_uids)
    if missing:
        log(f"  WARNING: {missing} Stage 04 users not in strict3 uid_list — excluded from O_u audit")
    n_users = len(s4_uids)
    log(f"  Stage 04 users in uid_list (audit scope): {n_users}")

    # Repack Gaussian params for audit-scope users only
    log("  packing Gaussian params ...")
    mu_arr = np.stack([np.asarray(s4_users[u]["mu"], dtype=np.float32) for u in s4_uids], axis=0)
    inv_arr = np.stack([np.asarray(s4_users[u]["sigma_inv"], dtype=np.float32) for u in s4_uids], axis=0)
    gate_T_arr = np.asarray([float(s4_users[u]["d2_q95"]) for u in s4_uids], dtype=np.float64)
    log(f"    mu: {mu_arr.shape}  inv: {inv_arr.shape}  gate_T: {gate_T_arr.shape}")

    # uid_list index of each Stage 04 audit user
    s4_uidlist_indices = np.array([uidlist_pos[u] for u in s4_uids], dtype=np.int64)

    # Build background pool: all Stage 04 user sentences (profile+val+test)
    all_z = np.concatenate([z_prof, z_val, z_test], axis=0)
    all_uidlist_idx = np.concatenate([prof_uidlist_idx, val_uidlist_idx, test_uidlist_idx], axis=0)
    log(f"  background pool: {all_z.shape}  total sents: {all_z.shape[0]}")

    in_s4_mask = np.isin(all_uidlist_idx, s4_uidlist_indices)
    log(f"    Stage 4 user sents: {in_s4_mask.sum()}")

    bg_z = all_z[in_s4_mask]
    bg_uidlist_idx = all_uidlist_idx[in_s4_mask]

    # Map uid_list index → s4_uids position (0..n_users-1); -1 if not Stage 4
    inv_pos = np.full(len(uid_list), -1, dtype=np.int32)
    inv_pos[s4_uidlist_indices] = np.arange(n_users, dtype=np.int32)
    bg_s4_idx = inv_pos[bg_uidlist_idx]  # shape (N_bg_sents,)
    log(f"  background (Stage 4 sents only): {bg_z.shape}")

    # 对每个 user u:排除 u 自己,得到该 user 的 background
    # GPU 加速:torch on CUDA
    log("  moving data to GPU ...")
    import torch
    device = torch.device("cuda:0")
    bg_t = torch.from_numpy(bg_z).to(device)
    mu_t = torch.from_numpy(mu_arr).to(device)
    inv_t = torch.from_numpy(inv_arr).to(device)
    gate_t = torch.from_numpy(gate_T_arr).to(device, dtype=torch.float64)
    bg_s4_t = torch.from_numpy(bg_s4_idx).to(device)
    n_bg_total = bg_z.shape[0]

    # Precompute per-user self range in bg array
    sorted_order = np.argsort(bg_s4_idx, kind="stable")
    bg_s4_sorted = bg_s4_idx[sorted_order]
    user_first = np.searchsorted(bg_s4_sorted, np.arange(n_users), side="left")
    user_last = np.searchsorted(bg_s4_sorted, np.arange(n_users), side="right")
    n_self_per_user = user_last - user_first
    log(f"    self-index: n_bg={n_bg_total}, "
        f"n_self: min={n_self_per_user.min()}, max={n_self_per_user.max()}")

    log("  computing per-user O_u (GPU) ...")
    O_u = np.zeros(n_users, dtype=np.float64)
    n_bg_per_user = np.zeros(n_users, dtype=np.int64)
    n_hits_per_user = np.zeros(n_users, dtype=np.int64)

    USER_BATCH = 20  # (20, 603K, 32) × 4 tensors ≈ 616 MB on GPU
    t0 = time.time()
    for u_start in range(0, n_users, USER_BATCH):
        u_end = min(n_users, u_start + USER_BATCH)
        ub = u_end - u_start

        # (ub, n_bg, 32) on GPU
        diff_t = mu_t[u_start:u_end, None, :] - bg_t[None, :, :]
        left_t = torch.bmm(diff_t, inv_t[u_start:u_end])
        d2_t = (left_t * diff_t).sum(dim=2)                    # (ub, n_bg)
        hits_t = (d2_t <= gate_t[u_start:u_end, None]).float()  # (ub, n_bg)

        # Zero out self-sentences on GPU using scatter
        for bi in range(ub):
            ui = u_start + bi
            sf = int(user_first[ui])
            st = int(user_last[ui])
            if sf < st:
                hits_t[bi, sf:st] = 0.0

        hits_cpu = hits_t.sum(dim=1).cpu().numpy()
        n_bg_batch = n_bg_total - n_self_per_user[u_start:u_end]
        O_u[u_start:u_end] = hits_cpu / np.maximum(n_bg_batch, 1)
        n_bg_per_user[u_start:u_end] = n_bg_batch
        n_hits_per_user[u_start:u_end] = hits_cpu.astype(np.int64)

        elapsed = time.time() - t0
        pct = 100 * u_end / n_users
        rate = u_end / elapsed if elapsed > 0 else 0
        eta = (n_users - u_end) / rate if rate > 0 else 0
        log(f"    {u_end}/{n_users} ({pct:.1f}%)  "
            f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    del bg_t, mu_t, inv_t, gate_t, bg_s4_t
    del diff_t, left_t, d2_t, hits_t
    torch.cuda.empty_cache()
    log(f"  done in {time.time()-t0:.1f}s")

    # 分布统计
    log("  O_u distribution:")
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        v = float(np.quantile(O_u, q))
        log(f"    p{q*100:>4.0f}%: {v:.4f}")
    log(f"    mean: {O_u.mean():.4f}  median: {float(np.median(O_u)):.4f}  "
        f"max: {O_u.max():.4f}  min: {O_u.min():.4f}")

    # sensitivity analysis
    sensitivity = {}
    log("  sensitivity (τ → # retained / % retained):")
    for tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70]:
        kept_mask = O_u <= tau
        n_kept = int(kept_mask.sum())
        sensitivity[f"tau={tau}"] = {
            "n_kept": n_kept,
            "pct_kept": 100 * n_kept / n_users,
            "n_dropped": n_users - n_kept,
        }
        log(f"    τ={tau}: kept={n_kept} ({100*n_kept/n_users:.1f}%) "
            f"dropped={n_users-n_kept}")

    # top-5% / top-10% 砍高 overlap user
    for top_pct in [5, 10, 15, 20]:
        thr = float(np.quantile(O_u, 1 - top_pct / 100))
        kept_mask = O_u <= thr
        sensitivity[f"top{top_pct}pct_drop"] = {
            "n_kept": int(kept_mask.sum()),
            "pct_kept": 100 * int(kept_mask.sum()) / n_users,
            "O_u_threshold": thr,
        }
        log(f"    top-{top_pct}% drop (O_u > {thr:.4f}): kept={int(kept_mask.sum())}")

    # 落盘
    out = {
        "config": {
            "n_users": n_users,
            "background_pool_n_sents": int(bg_z.shape[0]),
            "background_pool_source": "strict3_embeddings profile+val+test, restricted to Stage 4 uids",
            "gate_quantile": "d2_q95 per user (Stage 04 raw full-Σ)",
            "O_u_definition": "(# of OTHER Stage 4 users' sents with D² ≤ gate_T_u) / (total other-user sents)",
        },
        "O_u_distribution": {
            "mean": float(O_u.mean()),
            "median": float(np.median(O_u)),
            "min": float(O_u.min()),
            "max": float(O_u.max()),
            "p05": float(np.quantile(O_u, 0.05)),
            "p10": float(np.quantile(O_u, 0.10)),
            "p25": float(np.quantile(O_u, 0.25)),
            "p50": float(np.quantile(O_u, 0.50)),
            "p75": float(np.quantile(O_u, 0.75)),
            "p90": float(np.quantile(O_u, 0.90)),
            "p95": float(np.quantile(O_u, 0.95)),
            "p99": float(np.quantile(O_u, 0.99)),
        },
        "sensitivity": sensitivity,
        "per_user": {
            uid: {
                "O_u": float(O_u[i]),
                "n_bg": int(n_bg_per_user[i]),
                "n_hits": int(n_hits_per_user[i]),
            }
            for i, uid in enumerate(s4_uids)
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"  wrote → {OUT_PATH}")


if __name__ == "__main__":
    main()
