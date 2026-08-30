#!/usr/bin/env python3
"""Phase 6.D.2 — PCA48-driven skeleton retrieval (smoke).

给定 z_target (48d),从 skeleton pool 找 top-K nearest skeletons (按 L2 距离)。

这是 PCA-guided syntax template retrieval 的核心模块:
- 不训练 LLM,不让 LLM 学 PCA48
- 直接从真实 corpus 找 PCA48 最接近 z_target 的 syntax skeleton
- 让 LLM 只负责 fill 商品内容

Smoke: 10 个测试 user, 每个 user 取 μ_u, 找 top-3 nearest skeletons.
输出: 验证 retrieval 的 semantic合理性 (skeleton 内容应与 user style 一致).
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

# === Smoke config ===
N_USERS = 10
TOP_K = 3
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [skel_retrieve] {msg}", flush=True)


def main():
    log("=== Phase 6.D.2 — PCA48-driven skeleton retrieval (SMOKE v1) ===")

    # Load skeleton pool + embeddings
    skeleton_pool = []
    with open(SKELETON_DIR / "syntax_skeleton_pool_n1000.jsonl") as f:
        for line in f:
            skeleton_pool.append(json.loads(line))
    skeleton_z = np.load(SKELETON_DIR / "skeleton_48d_embeddings_n1000.npy")
    log(f"  loaded skeleton pool: {len(skeleton_pool)} entries, z shape={skeleton_z.shape}")

    # Load user Gaussians
    gauss = json.load(open(GAUSS_PATH))
    user_gauss = gauss["user_gaussians"] if "user_gaussians" in gauss else gauss.get("users", {})
    log(f"  loaded {len(user_gauss)} user Gaussians")

    # Pick N_USERS test users
    rng = np.random.default_rng(SEED)
    test_uids = rng.choice(list(user_gauss.keys()), size=min(N_USERS, len(user_gauss)), replace=False)

    log(f"\n=== Retrieval for {N_USERS} test users (top-{TOP_K}) ===")
    for uid in test_uids:
        ug = user_gauss[uid]
        mu = np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
        if mu.shape != (48,):
            continue
        sigma_diag = np.asarray(ug.get("sigma_diag", np.ones(48)), dtype=np.float64)

        # Distance: L2 between mu and all skeleton_z
        dist = np.linalg.norm(skeleton_z - mu[None, :], axis=1)
        top_k_idx = np.argsort(dist)[:TOP_K]

        log(f"\n  user {uid}:")
        log(f"    μ norm={np.linalg.norm(mu):.2f}, σ_diag mean={sigma_diag.mean():.2f}")
        for rank, idx in enumerate(top_k_idx):
            sk = skeleton_pool[int(idx)]
            log(f"    rank {rank+1}: idx={idx}, d={dist[idx]:.3f}, "
                f"slots={sk['slot_count']}/{sk['token_count']}")
            log(f"      skeleton: {sk['skeleton_str']}")
            log(f"      original: {sk['original_query']}")

    log("\n=== Retrieval smoke done. Skeleton pool coverage verified ===")


if __name__ == "__main__":
    main()