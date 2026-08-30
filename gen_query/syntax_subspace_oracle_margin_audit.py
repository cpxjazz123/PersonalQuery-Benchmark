#!/usr/bin/env python3
"""Phase 6.E.1 — Oracle Margin Feasibility Audit (memory-safe).

用户 2026-08-30 23:10 提案 (在 Phase 6.D PARTIAL 后):
- 不调用 LLM
- 直接遍历 skeleton pool + real corpus 的 PCA48
- 计算 best possible d_self 和 best possible margin (M)
- 关键问题: 现有真实句法库里, 到底有没有任何结构能达到 M > 0?
- 如果 oracle 有空间 → 把 skeleton retrieval 从 nearest 改成 margin-aware

两个 pool:
1. skeleton pool (D.1): 1000 delexicalized skeletons (from train)
2. real corpus: train.jsonl 的所有 query 真实 PCA48

Per-user metrics:
- best_d_self: min_i d_self(z_i, μ_u)
- best_M: max_i (d_nearest_other(z_i) - d_self(z_i, μ_u))
- M>0 rate: |{i : M(z_i, u) > 0}| / N
- strict achievable: |{i : M(z_i, u) > 0 AND d_self(z_i, u) ≤ R}| / N

Memory-safe: OOM in initial run tried to materialize (5000, 1.2M, 48) = 2.3TB.
Use einsum trick + chunked over pool to avoid 3D broadcast.
For each chunk of pool: ||a-b||² = ||a||² + ||b||² - 2 a·b
Peak memory per chunk: (chunk_size, n_users) float32 ≤ 2.4GB.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
SKELETON_DIR = SCRATCH / "syntax_skeleton"
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"
LOG_DIR = SCRATCH / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Same config as Phase 6.D.4 pilot for direct comparison
N_USERS = 10
MU_NORM_MIN = 0.5
MU_NORM_MAX = 3.0
R_THRESHOLD = 12.0       # permissive (R_95 typically 8-15)
SEED = 42
CHUNK_SIZE = 200         # memory-safe chunk over pool dimension
DTYPE = np.float32       # half memory of float64, sufficient for distance ranking


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [oracle_audit] {msg}", flush=True)


def einsum_min_dist(pool_chunk: np.ndarray, users: np.ndarray) -> np.ndarray:
    """For each row in pool_chunk, find min L2 distance over all rows in users.
    Uses ||a-b||² = ||a||² + ||b||² - 2 a·b to avoid (n_chunk, n_users, d) broadcast.

    pool_chunk: (chunk, 48), users: (n_users, 48)
    Returns: (chunk,) — min distance over users per pool row.
    """
    a_sq = (pool_chunk * pool_chunk).sum(axis=1)[:, None]      # (chunk, 1)
    b_sq = (users * users).sum(axis=1)[None, :]                 # (1, n_users)
    dot = pool_chunk @ users.T                                  # (chunk, n_users)
    d_sq = a_sq + b_sq - 2.0 * dot                              # (chunk, n_users)
    # clamp to non-negative (numerical noise)
    np.maximum(d_sq, 0.0, out=d_sq)
    # min over users
    return np.sqrt(d_sq.min(axis=1))


def oracle_audit(mu_user: np.ndarray, pool_z: np.ndarray, users: np.ndarray,
                 pool_name: str, chunk_size: int = CHUNK_SIZE):
    """Memory-safe oracle audit.

    mu_user: (48,) target user's μ
    pool_z: (N_pool, 48) candidate PCA48 (skeleton or real)
    users: (N_users, 48) all user mus for d_nearest_other computation
    """
    n_pool = len(pool_z)
    # Per-sample trackers
    d_self = np.empty(n_pool, dtype=DTYPE)
    d_nearest_other = np.empty(n_pool, dtype=DTYPE)
    margin = np.empty(n_pool, dtype=DTYPE)
    mu_user = mu_user.astype(DTYPE)
    pool_z = pool_z.astype(DTYPE)
    users = users.astype(DTYPE)
    for i in range(0, n_pool, chunk_size):
        chunk = pool_z[i:i+chunk_size]
        ds = np.linalg.norm(chunk - mu_user[None, :], axis=1)        # (chunk,)
        dno = einsum_min_dist(chunk, users)                            # (chunk,)
        d_self[i:i+chunk_size] = ds
        d_nearest_other[i:i+chunk_size] = dno
        margin[i:i+chunk_size] = dno - ds
    return {
        "pool_name": pool_name,
        "pool_size": n_pool,
        "best_d_self": float(np.min(d_self)),
        "best_d_self_idx": int(np.argmin(d_self)),
        "best_M": float(np.max(margin)),
        "best_M_idx": int(np.argmax(margin)),
        "M_gt0_rate": float(np.mean(margin > 0)),
        "strict_rate": float(np.mean((margin > 0) & (d_self <= R_THRESHOLD))),
        "d_self_at_best_M": float(d_self[np.argmax(margin)]),
        "M_at_best_d_self": float(margin[np.argmin(d_self)]),
        "d_self_median": float(np.median(d_self)),
        "margin_median": float(np.median(margin)),
    }


def main():
    log(f"=== Phase 6.E.1 — Oracle Margin Feasibility (N={N_USERS} users, MEMORY-SAFE) ===")
    log("  不调用 LLM. 直接对 skeleton pool + real corpus 做 margin oracle.")
    log(f"  Strict criterion: M > 0 AND d_self <= R={R_THRESHOLD}")
    log(f"  Chunk size: {CHUNK_SIZE}, dtype: {DTYPE}")

    # === Load skeleton pool + embeddings ===
    skeleton_pool = []
    with open(SKELETON_DIR / "syntax_skeleton_pool_n1000.jsonl") as f:
        for line in f:
            skeleton_pool.append(json.loads(line))
    skeleton_z = np.load(SKELETON_DIR / "skeleton_48d_embeddings_n1000.npy")
    log(f"  loaded skeleton pool: {len(skeleton_pool)}, z shape={skeleton_z.shape}")

    # === Load Gaussians ===
    gauss = json.load(open(GAUSS_PATH))
    user_gauss = gauss["user_gaussians"] if "user_gaussians" in gauss else gauss.get("users", {})
    log(f"  loaded {len(user_gauss)} user Gaussians")

    # === All user mus for margin calculation (vectorized, once) ===
    all_user_mus = np.array([np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
                             for ug in user_gauss.values()
                             if np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64).shape == (48,)])
    log(f"  computed {len(all_user_mus)} user mus for margin calculation")
    # Convert to DTYPE once
    all_user_mus_f32 = all_user_mus.astype(DTYPE)

    # === Filter candidate users with μ_norm in range ===
    candidate_users = []
    for uid, ug in user_gauss.items():
        mu = np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
        if mu.shape != (48,):
            continue
        n = float(np.linalg.norm(mu))
        if MU_NORM_MIN <= n <= MU_NORM_MAX:
            candidate_users.append((uid, mu, n))
    log(f"  candidate users (μ_norm∈[{MU_NORM_MIN},{MU_NORM_MAX}]): {len(candidate_users)}")

    rng = np.random.default_rng(SEED)
    rng.shuffle(candidate_users)

    # === Load train.jsonl for attrs (need same test users as Phase 6.D.4) ===
    samples = []
    with open(SCRATCH / "pca48_dataset" / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    user_attrs = {}
    for s in samples:
        uid = s.get("user_id") or s.get("user")
        if uid is None:
            continue
        if uid not in user_attrs and s.get("c_i") and len(s["c_i"]) >= 3:
            user_attrs[uid] = s["c_i"]

    # Pick first N_USERS with attrs (SAME order as Phase 6.D.4)
    selected_users = []
    for uid, mu, mu_n in candidate_users:
        if uid not in user_attrs:
            continue
        selected_users.append({
            "user_id": uid, "mu": mu, "mu_norm": mu_n,
        })
        if len(selected_users) >= N_USERS:
            break
    log(f"  selected {len(selected_users)} test users")

    # === Real corpus PCA48 (from train.jsonl z_48 field) ===
    real_z = np.array([s["z_48"] for s in samples if "z_48" in s], dtype=np.float64)
    log(f"  real corpus PCA48: shape={real_z.shape}")
    real_sample_user_idx = []
    for s in samples:
        if "z_48" in s:
            uid = s.get("user_id") or s.get("user")
            real_sample_user_idx.append(uid)
    real_sample_user_idx = np.array(real_sample_user_idx)
    log(f"  indexed {len(real_sample_user_idx)} real samples by user")

    # === Per-user oracle audit ===
    audit_results = []

    for pi, pd in enumerate(selected_users):
        uid = pd["user_id"]
        mu = pd["mu"]
        log(f"\n=== User {pi+1}/{len(selected_users)}: {uid} (μ_norm={pd['mu_norm']:.2f}) ===")

        # === Skeleton pool oracle ===
        sk_oracle = oracle_audit(mu, skeleton_z, all_user_mus_f32, "skeleton_n1000")

        # === Real corpus oracle (excl. self) ===
        mask = real_sample_user_idx != uid
        other_real_z = real_z[mask]
        real_oracle = oracle_audit(mu, other_real_z, all_user_mus_f32, "real_corpus")

        log(f"  SKELETON pool (n={sk_oracle['pool_size']}):")
        log(f"    best_d_self = {sk_oracle['best_d_self']:.2f}  best_M = {sk_oracle['best_M']:.2f}")
        log(f"    M>0 rate = {sk_oracle['M_gt0_rate']*100:.1f}%  strict rate = {sk_oracle['strict_rate']*100:.1f}%")
        log(f"  REAL corpus (n={real_oracle['pool_size']}, excl. self):")
        log(f"    best_d_self = {real_oracle['best_d_self']:.2f}  best_M = {real_oracle['best_M']:.2f}")
        log(f"    M>0 rate = {real_oracle['M_gt0_rate']*100:.1f}%  strict rate = {real_oracle['strict_rate']*100:.1f}%")

        audit_results.append({
            "user_id": uid,
            "mu_norm": pd["mu_norm"],
            "skeleton_oracle": sk_oracle,
            "real_oracle": real_oracle,
        })

    # === Cohort summary ===
    log("\n=== COHORT ORACLE SUMMARY (N={}) ===".format(len(audit_results)))
    sk_best_ds = [r["skeleton_oracle"]["best_d_self"] for r in audit_results]
    sk_best_M = [r["skeleton_oracle"]["best_M"] for r in audit_results]
    sk_M_gt0 = [r["skeleton_oracle"]["M_gt0_rate"] for r in audit_results]
    sk_strict = [r["skeleton_oracle"]["strict_rate"] for r in audit_results]
    real_best_ds = [r["real_oracle"]["best_d_self"] for r in audit_results]
    real_best_M = [r["real_oracle"]["best_M"] for r in audit_results]
    real_M_gt0 = [r["real_oracle"]["M_gt0_rate"] for r in audit_results]
    real_strict = [r["real_oracle"]["strict_rate"] for r in audit_results]

    log(f"\n  SKELETON pool (n=1000):")
    log(f"    best_d_self:  median={np.median(sk_best_ds):.2f}  mean={np.mean(sk_best_ds):.2f}")
    log(f"    best_M:       median={np.median(sk_best_M):.2f}  mean={np.mean(sk_best_M):.2f}")
    log(f"    M>0 rate:     median={np.median(sk_M_gt0)*100:.1f}%  users with >50%: "
        f"{sum(1 for r in sk_M_gt0 if r > 0.5)}/{len(sk_M_gt0)}")
    log(f"    strict rate:  median={np.median(sk_strict)*100:.1f}%  users with >10%: "
        f"{sum(1 for r in sk_strict if r > 0.1)}/{len(sk_strict)}")

    log(f"\n  REAL corpus (excl self, ~{real_oracle['pool_size']}):")
    log(f"    best_d_self:  median={np.median(real_best_ds):.2f}  mean={np.mean(real_best_ds):.2f}")
    log(f"    best_M:       median={np.median(real_best_M):.2f}  mean={np.mean(real_best_M):.2f}")
    log(f"    M>0 rate:     median={np.median(real_M_gt0)*100:.1f}%  users with >50%: "
        f"{sum(1 for r in real_M_gt0 if r > 0.5)}/{len(real_M_gt0)}")
    log(f"    strict rate:  median={np.median(real_strict)*100:.1f}%  users with >10%: "
        f"{sum(1 for r in real_strict if r > 0.1)}/{len(real_strict)}")

    log(f"\n=== COMPARISON ===")
    log(f"  Phase 6.D.4 GENERATED d_self: median=5.88 (lower than oracle best {np.median(sk_best_ds):.2f})")
    log(f"  Phase 6.D.4 GENERATED margin: median=-4.51 (vs skeleton oracle {np.median(sk_best_M):.2f}, real oracle {np.median(real_best_M):.2f})")
    n_sk_feasible = sum(1 for r in sk_best_M if r > 0)
    n_real_feasible = sum(1 for r in real_best_M if r > 0)
    log(f"  Skeleton oracle M>0 feasible: {n_sk_feasible}/{len(sk_best_M)} users")
    log(f"  Real corpus oracle M>0 feasible: {n_real_feasible}/{len(real_best_M)} users")
    if n_sk_feasible >= len(sk_best_M) // 2:
        log(f"  ⚠️ Skeleton oracle shows M>0 IS achievable → margin-aware retrieval can lift M")
    else:
        log(f"  ⚠️ Skeleton oracle shows M>0 NOT achievable → PCA48 separability is the bottleneck")

    # === Save ===
    out_path = LOG_DIR / "phase6e_oracle_audit.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_users": N_USERS, "mu_norm_range": [MU_NORM_MIN, MU_NORM_MAX],
                "R_threshold": R_THRESHOLD, "seed": SEED,
                "chunk_size": CHUNK_SIZE, "dtype": str(DTYPE),
            },
            "cohort_summary": {
                "skeleton": {
                    "best_d_self_median": float(np.median(sk_best_ds)),
                    "best_M_median": float(np.median(sk_best_M)),
                    "M_gt0_rate_median": float(np.median(sk_M_gt0)),
                    "strict_rate_median": float(np.median(sk_strict)),
                    "users_M_gt0": int(n_sk_feasible),
                },
                "real_corpus": {
                    "best_d_self_median": float(np.median(real_best_ds)),
                    "best_M_median": float(np.median(real_best_M)),
                    "M_gt0_rate_median": float(np.median(real_M_gt0)),
                    "strict_rate_median": float(np.median(real_strict)),
                    "users_M_gt0": int(n_real_feasible),
                },
            },
            "per_user": audit_results,
        }, f, indent=2)
    log(f"\n  wrote → {out_path}")


if __name__ == "__main__":
    main()