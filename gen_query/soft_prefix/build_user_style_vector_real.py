#!/usr/bin/env python3
"""Build per-user 真实 style vector (3584-dim) from Qwen hidden residual.

背景 (用户指令 2026-08-22 修正):
  旧 generate_with_user_inject.py 用 VADES 20-dim latent user_mu + random
  projection (Linear 20 -> 3584) 注入 Qwen layer, 但 projection 未经训练,
  输出 broken ("Ajable pen, refillable ink cartridges...").

  正确路线 (用户建议,参考 ACL 2025 StyleVector / EACL 2024 Style Vectors):
    直接用 extract_residual_hidden.py 产出的 真实 LLM hidden residual
    (n_sentences, 3584, layer L), 不需要 projector。
    注入: h_l' = h_l + α * Δh_u
    其中 Δh_u 是该用户在 layer L 的真实风格方向 (3584-dim)。

Outputs:
  - user_style_vectors.jsonl: per-user 真实 mean residual (4 layers × 3584-dim)
    路径: /home/wlia0047/hj82_scratch2/wenyu/user_style_steering/user_style_vectors_real.jsonl
  - pca_model.npz: PCA(20) fit on global residual, 路径同 dir
    提供 C 路线 (PCA-Gaussian):z_u @ U.T + mean → 3584-dim bias

注入模式 (generate_with_user_inject.py 用):
  - direct_mean: Δh_u = user 在 layer L 的 mean residual (确定性, B 路线)
  - pca_gaussian: Δh_u = (sample z_u ~ N(mu_u, Sigma_u) on PCA space) inverse → 3584-dim (C 路线)

依赖:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/residual_hidden.npz
    (extract_residual_hidden.py 产出: sentences + residual_layer_16/20/24/26)
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/sentences_for_rewrite.jsonl
    (extract_sentences_spacy.py 产出: user_id + sentence_text join key)

硬编码路径 (Rule 3), 不接受 CLI 参数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/user_style_steering")
SCRATCH.mkdir(parents=True, exist_ok=True)

# === 输入 ===
RESIDUAL_NPZ = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/residual_hidden.npz")
SENTENCES_JSONL = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/sentences_for_rewrite.jsonl")

# === 输出 ===
USER_STYLE_VECTORS_JSONL = SCRATCH / "user_style_vectors_real.jsonl"
PCA_MODEL_NPZ = SCRATCH / "pca_model.npz"

# === 配置 ===
LAYERS = [16, 20, 24, 26]  # 与 extract_residual_hidden.py 默认一致
PCA_DIM = 20  # 与 VADES_LATENT_DIM 对齐, 业务侧可对比


def main() -> int:
    print(f"[main] loading residual: {RESIDUAL_NPZ}", flush=True)
    if not RESIDUAL_NPZ.exists():
        raise FileNotFoundError(f"{RESIDUAL_NPZ} 不存在, 先跑 gaussian/extract_residual_hidden.py")
    npz = np.load(RESIDUAL_NPZ, allow_pickle=True)
    sentences_np = npz["sentences"]
    layers_np = np.asarray(npz["layers"]).tolist()
    hidden_dim = int(npz["hidden_dim"][0])
    print(f"[main] residual sentences={len(sentences_np)}, layers={layers_np}, hidden_dim={hidden_dim}")
    # 校验 LAYERS 在 npz 里都有
    for L in LAYERS:
        k = f"residual_layer_{L}"
        if k not in npz.files:
            raise ValueError(f"{k} 不在 npz 里 (layers={layers_np})")

    # === 建 sentence_text -> user_id 索引 ===
    print(f"[main] loading sentence→user mapping: {SENTENCES_JSONL}", flush=True)
    sent_to_user: dict[str, str] = {}
    with open(SENTENCES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sent_to_user[row["sentence_text"]] = row["user_id"]
    print(f"[main] {len(sent_to_user)} (sentence_text, user_id) pairs")

    # === per-user mean residual (4 layers × 3584-dim) ===
    # 容许 user_id 缺失的句 (断章): 没 join 上的句子会 raise
    per_user: dict[str, dict[int, np.ndarray]] = {}
    n_skipped = 0
    for i, s in enumerate(sentences_np):
        s_str = str(s)
        uid = sent_to_user.get(s_str)
        if uid is None:
            n_skipped += 1
            continue
        if uid not in per_user:
            per_user[uid] = {}
        for L in LAYERS:
            arr = npz[f"residual_layer_{L}"]  # (N, 3584)
            if L not in per_user[uid]:
                per_user[uid][L] = []
            per_user[uid][L].append(arr[i])

    if n_skipped > 0:
        print(f"[main] warning: {n_skipped}/{len(sentences_np)} sentences 没 join 上 user_id, 跳过")

    user_means: dict[str, dict[int, np.ndarray]] = {}
    for uid, layer_dict in per_user.items():
        user_means[uid] = {}
        for L, vecs in layer_dict.items():
            stack = np.stack(vecs, axis=0)  # (n_sents, 3584)
            mean = stack.mean(axis=0)  # (3584,)
            user_means[uid][L] = mean
            print(f"  user={uid} layer={L}: n_sents={len(vecs)}, "
                  f"mean norm={float(np.linalg.norm(mean)):.2f}, "
                  f"std norm={float(np.linalg.norm(stack.std(axis=0))):.2f}")

    # === PCA fit on 全局 residual (layer 16 当代表, 与 VADES 训练对齐) ===
    # 用 layer 16 的全 residual 矩阵 fit PCA (20 dim)
    L_PCA = LAYERS[0]  # 16
    full_resid = npz[f"residual_layer_{L_PCA}"]  # (N, 3584)
    print(f"[main] fitting PCA({PCA_DIM}) on layer {L_PCA} 全 residual: shape={full_resid.shape}", flush=True)
    pca_mean = full_resid.mean(axis=0)  # (3584,)
    centered = full_resid - pca_mean[None, :]
    # SVD: centered = U S Vt, 取前 PCA_DIM 个 right singular vectors
    # 用 truncated SVD 省内存
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    z_full = pca.fit_transform(centered)  # (N, 20)
    print(f"[main] PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"[main] PCA components: {pca.components_.shape}, mean norm={float(np.linalg.norm(pca_mean)):.2f}")

    # === per-user Gaussian on PCA space (C 路线) ===
    user_gauss: dict[str, dict] = {}
    for uid in per_user:
        # 拿该 user 的 layer 16 residual indices
        sents = [str(s) for s in sentences_np]
        user_idx = [i for i, s in enumerate(sents) if sent_to_user.get(s) == uid]
        if len(user_idx) < 2:
            print(f"  [warn] user {uid} 只 {len(user_idx)} 句, PCA-Gaussian 协方差用 identity")
        z_u = z_full[user_idx]  # (n_sents, 20)
        mu_u = z_u.mean(axis=0)  # (20,)
        if len(user_idx) >= 2:
            cov_u = np.cov(z_u, rowvar=False) + np.eye(PCA_DIM) * 1e-4  # (20,20) 加 jitter 防止奇异
        else:
            cov_u = np.eye(PCA_DIM)
        user_gauss[uid] = {
            "mu": mu_u,
            "cov": cov_u,
            "n_sents": len(user_idx),
        }
        print(f"  user={uid}: z_mu norm={float(np.linalg.norm(mu_u)):.4f}, "
              f"z_cov trace={float(np.trace(cov_u)):.4f}, n={len(user_idx)}")

    # === 写出 user_style_vectors.jsonl (B 路线用, direct_mean) ===
    with open(USER_STYLE_VECTORS_JSONL, "w", encoding="utf-8") as f:
        for uid, layer_dict in user_means.items():
            row = {"user_id": uid}
            for L, mean in layer_dict.items():
                row[f"residual_mean_layer_{L}"] = mean.astype(np.float32).tolist()
                row[f"residual_mean_norm_layer_{L}"] = float(np.linalg.norm(mean))
            f.write(json.dumps(row) + "\n")
    print(f"[main] ✓ wrote {USER_STYLE_VECTORS_JSONL}")

    # === 写出 PCA model + per-user Gaussian (C 路线用) ===
    np.savez_compressed(
        PCA_MODEL_NPZ,
        pca_components=pca.components_.astype(np.float32),  # (20, 3584)
        pca_mean=pca_mean.astype(np.float32),  # (3584,)
        pca_explained_var=pca.explained_variance_.astype(np.float32),
        layers=np.asarray(LAYERS, dtype=np.int32),
        user_ids=np.asarray(list(user_gauss.keys()), dtype=object),
        user_mu=np.stack([user_gauss[u]["mu"] for u in user_gauss]).astype(np.float32),  # (n_users, 20)
        user_cov=np.stack([user_gauss[u]["cov"] for u in user_gauss]).astype(np.float32),  # (n_users, 20, 20)
    )
    print(f"[main] ✓ wrote {PCA_MODEL_NPZ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
