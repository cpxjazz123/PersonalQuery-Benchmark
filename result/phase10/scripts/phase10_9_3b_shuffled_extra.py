#!/usr/bin/env python3
"""Phase 10.9.3b: 警惕性 shuffled extra diagnostics.

用户警示:"rank=1 和 flip acc=1.0 过于完美,需要警惕偏好对构造或同一 Mahalanobis
评分器带来的循环验证"。在做 Primary Generation 之前先排查:

  D1: decoder(prefix(shuffled z)) 重建偏向 原 z 还是 shuffled z?
       → 若重建偏向 shuffled z, 说明 decoder 只 decode z 内容, 但 model 还可能用 shuffled 信息
  D2: shuffled profile 跨用户 rank 仍为 1?
       → 若仍 = 1, 说明 model 用 norm/全局编码而非 20d 句法语义 (FAIL)
  D3: sign-flip z (整个 z 取负) → rank 应 ≈ 随机 (FAIL 若仍 rank=1)
  D4: matched-norm random vector (norm 与 z_u 相同, 但内容无关) → rank 应 ≈ 随机
  D5: 只保留 mu (前 20 维) → rank 应 ≈ 1 (若有用户区分)
  D6: 只保留 log_sigma (后 20 维) → rank 应 ≈ 1 (若有用户区分)

判定:
  D2 = 1 (failure) → model 没读 20d 句法, 用 norm
  D3 ≈ 随机, D4 ≈ 随机, D5 ≈ 1, D6 ≈ 1 → model 真读 20d 句法, D1/D2 看似失败无害
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PROJ_CKPT = OUT_DIR / "phase10_9_2_projector.pt"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
META_OUT = OUT_DIR / "phase10_9_3b_shuffled_extra.json"
LOG_OUT = OUT_DIR / "phase10_9_3b_shuffled_extra.log"

USER_DIM = 40
HIDDEN_DIM = 128
NUM_TOKENS = 8
DEVICE = "cuda:0"
SEED = 42


def log(m):
    print(f"[phase10-9.3b] {m}", flush=True)


def main():
    log("=" * 70)
    log("Phase 10.9.3b: 警惕性 Shuffled Extra Diagnostics")
    log("=" * 70)

    # === Load projector + decoder ===
    log("[1] Loading projector + decoder ...")
    from query.soft_prefix.projector import SoftPrefixProjector
    ckpt = torch.load(PROJ_CKPT, map_location=DEVICE)
    projector = SoftPrefixProjector(
        user_dim=ckpt["user_dim"], hidden_dim=ckpt["hidden_dim"],
        num_tokens=ckpt["num_tokens"], model_dim=ckpt["model_dim"],
        dtype=torch.float32, gate_init=ckpt["gate_init"],
    ).to(DEVICE)
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()

    # Decoder
    import torch.nn as nn
    class PrefixDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(NUM_TOKENS * ckpt["model_dim"], USER_DIM * 2),
                nn.GELU(),
                nn.Linear(USER_DIM * 2, USER_DIM),
            )
        def forward(self, x):
            return self.fc(x)
    decoder = PrefixDecoder().to(DEVICE)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()
    log("  loaded")

    # === Load user profiles ===
    log("[2] Loading user profiles (for cross-user rank) ...")
    user_id_to_idx = {}
    user_mu_arr = []
    user_logvar_arr = []
    with USER_PROFILE_FILE.open() as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            user_id_to_idx[row["user_id"]] = idx
            user_mu_arr.append(row["user_mu"])
            user_logvar_arr.append(row["user_logvar"])
    user_mu = np.array(user_mu_arr, dtype=np.float32)
    user_logvar = np.array(user_logvar_arr, dtype=np.float32)
    log(f"  {len(user_id_to_idx)} users")

    # === Pick 30 users for diagnostic ===
    rng = np.random.default_rng(SEED)
    sample_users = rng.choice(len(user_id_to_idx), size=30, replace=False)
    sample_idx_set = set(sample_users.tolist())
    z_u_sample = torch.tensor(
        np.concatenate([user_mu[sample_users], user_logvar[sample_users]], axis=1),
        dtype=torch.float32, device=DEVICE,
    )  # [30, 40]
    target_user_ids = [list(user_id_to_idx.keys())[i] for i in sample_users]
    log(f"  sampled {len(z_u_sample)} users for diagnostic")

    # === D1: decoder(prefix(shuffled z)) ===
    log("[3] === D1: decoder 重建偏好 (原 z vs shuffled z) ===")
    # 构造 shuffled z (permute 维度, fixed seed per user)
    z_shuffled = torch.zeros_like(z_u_sample)
    for i in range(len(z_u_sample)):
        perm = rng.permutation(USER_DIM)
        z_shuffled[i] = z_u_sample[i, perm]

    with torch.no_grad():
        prefix_orig = projector(z_u_sample).float()
        prefix_shuf = projector(z_shuffled).float()
        # decoder 重建
        z_hat_orig = decoder(prefix_orig.reshape(len(z_u_sample), -1))
        z_hat_shuf = decoder(prefix_shuf.reshape(len(z_u_sample), -1))
    # MSE
    mse_orig_to_orig = ((z_hat_orig - z_u_sample) ** 2).mean().item()
    mse_orig_to_shuf = ((z_hat_orig - z_shuffled) ** 2).mean().item()
    mse_shuf_to_orig = ((z_hat_shuf - z_u_sample) ** 2).mean().item()
    mse_shuf_to_shuf = ((z_hat_shuf - z_shuffled) ** 2).mean().item()
    log(f"  decoder(orig z) → orig:    MSE = {mse_orig_to_orig:.4f}")
    log(f"  decoder(orig z) → shuffled: MSE = {mse_orig_to_shuf:.4f}")
    log(f"  decoder(shuf z) → orig:    MSE = {mse_shuf_to_orig:.4f}")
    log(f"  decoder(shuf z) → shuffled: MSE = {mse_shuf_to_shuf:.4f}")
    log(f"  → 若 decoder(shuf z) 重建偏向 shuffled z, 说明 model 仍保留原 z 信息")

    # === D2: shuffled profile 跨用户 rank ===
    log("[4] === D2: shuffled profile 跨用户 rank ===")
    # 所有 2918 user 的 z + prefix
    all_z = torch.tensor(
        np.concatenate([user_mu, user_logvar], axis=1), dtype=torch.float32, device=DEVICE
    )
    with torch.no_grad():
        all_prefixes = projector(all_z).float()  # [2918, K, H]
        all_prefixes_flat = all_prefixes.reshape(2918, -1)

    def cross_user_rank(prefix_per_user):
        """算 per-user (under perturbed z) 的 cos 距离, rank 在 2918 user 中."""
        ranks = []
        for i in range(len(z_u_sample)):
            target_user = sample_users[i]
            target_p = prefix_per_user[i:i+1]
            t_norm = target_p.norm(dim=1, keepdim=True)
            a_norm = all_prefixes_flat.norm(dim=1, keepdim=True)
            cos_sim = (target_p @ all_prefixes_flat.T) / (t_norm @ a_norm.T + 1e-9)
            cos_sim = cos_sim.squeeze(0)
            cos_dist = 1 - cos_sim
            rank = (cos_dist < cos_dist[target_user]).sum().item() + 1
            ranks.append(rank)
        return ranks

    rank_orig = cross_user_rank(prefix_orig.reshape(len(z_u_sample), -1))
    rank_shuf = cross_user_rank(prefix_shuf.reshape(len(z_u_sample), -1))
    log(f"  rank (orig): mean = {np.mean(rank_orig):.2f}, rank1 = {sum(r==1 for r in rank_orig)/len(rank_orig):.4f}")
    log(f"  rank (shuffled): mean = {np.mean(rank_shuf):.2f}, rank1 = {sum(r==1 for r in rank_shuf)/len(rank_shuf):.4f}")
    log(f"  → 若 shuffled rank1 ≈ orig rank1 (近 1.0), model 用了 norm/全局编码")

    # === D3: sign-flip z ===
    log("[5] === D3: sign-flip z (-z) ===")
    z_signflip = -z_u_sample
    with torch.no_grad():
        prefix_signflip = projector(z_signflip).float()
    rank_signflip = cross_user_rank(prefix_signflip.reshape(len(z_u_sample), -1))
    log(f"  rank (sign-flip): mean = {np.mean(rank_signflip):.2f}, rank1 = {sum(r==1 for r in rank_signflip)/len(rank_signflip):.4f}")
    log(f"  → 若 rank1 ≈ 1.0, model 只用 norm/sign 不敏感")

    # === D4: matched-norm random vector ===
    log("[6] === D4: matched-norm random vector ===")
    norms = z_u_sample.norm(dim=1, keepdim=True)
    z_random_matched = torch.randn_like(z_u_sample)
    z_random_matched = z_random_matched / z_random_matched.norm(dim=1, keepdim=True) * norms
    with torch.no_grad():
        prefix_rand = projector(z_random_matched).float()
    rank_rand = cross_user_rank(prefix_rand.reshape(len(z_u_sample), -1))
    log(f"  rank (matched-norm random): mean = {np.mean(rank_rand):.2f}, rank1 = {sum(r==1 for r in rank_rand)/len(rank_rand):.4f}")
    log(f"  → 若 rank1 ≈ 1.0, model 只用 norm, 不读内容")

    # === D5: 只用 mu (前 20 维) ===
    log("[7] === D5: 只用 mu (前 20 维) ===")
    z_mu_only = torch.zeros_like(z_u_sample)
    z_mu_only[:, :20] = z_u_sample[:, :20]
    z_mu_only[:, 20:] = z_u_sample.mean(dim=1, keepdim=True).expand_as(z_u_sample[:, 20:])  # 用 mean 填
    with torch.no_grad():
        prefix_mu = projector(z_mu_only).float()
    rank_mu = cross_user_rank(prefix_mu.reshape(len(z_u_sample), -1))
    log(f"  rank (mu only): mean = {np.mean(rank_mu):.2f}, rank1 = {sum(r==1 for r in rank_mu)/len(rank_mu):.4f}")

    # === D6: 只用 log_sigma (后 20 维) ===
    log("[8] === D6: 只用 log_sigma (后 20 维) ===")
    z_sigma_only = torch.zeros_like(z_u_sample)
    z_sigma_only[:, :20] = z_u_sample.mean(dim=1, keepdim=True).expand_as(z_u_sample[:, :20])
    z_sigma_only[:, 20:] = z_u_sample[:, 20:]
    with torch.no_grad():
        prefix_sig = projector(z_sigma_only).float()
    rank_sig = cross_user_rank(prefix_sig.reshape(len(z_u_sample), -1))
    log(f"  rank (sigma only): mean = {np.mean(rank_sig):.2f}, rank1 = {sum(r==1 for r in rank_sig)/len(rank_sig):.4f}")

    # === 总结 ===
    log("=" * 70)
    log("Summary:")
    log(f"  D1 decoder(shuffled) 重建偏向 shuffled: mse_shuf_to_shuf vs mse_shuf_to_orig")
    log(f"     mse_shuf_to_shuf = {mse_shuf_to_shuf:.4f}, mse_shuf_to_orig = {mse_shuf_to_orig:.4f}")
    log(f"     偏低 → decoder 真把 shuffled 信息解码成 shuffled")
    log(f"  D2 shuffled rank1 = {sum(r==1 for r in rank_shuf)/len(rank_shuf):.4f}")
    log(f"     (FAIL if > 0.5 — 用 norm/全局编码)")
    log(f"  D3 sign-flip rank1 = {sum(r==1 for r in rank_signflip)/len(rank_signflip):.4f}")
    log(f"     (FAIL if > 0.5 — 只用 norm)")
    log(f"  D4 random matched-norm rank1 = {sum(r==1 for r in rank_rand)/len(rank_rand):.4f}")
    log(f"     (FAIL if > 0.5 — 只用 norm)")
    log(f"  D5 mu only rank1 = {sum(r==1 for r in rank_mu)/len(rank_mu):.4f}")
    log(f"  D6 sigma only rank1 = {sum(r==1 for r in rank_sig)/len(rank_sig):.4f}")

    # === Meta ===
    meta = {
        "n_users": len(z_u_sample),
        "D1_decoder": {
            "mse_orig_to_orig": mse_orig_to_orig,
            "mse_orig_to_shuf": mse_orig_to_shuf,
            "mse_shuf_to_orig": mse_shuf_to_orig,
            "mse_shuf_to_shuf": mse_shuf_to_shuf,
        },
        "D2_shuffled_rank": {
            "mean": float(np.mean(rank_shuf)),
            "rank1_rate": float(sum(r==1 for r in rank_shuf)/len(rank_shuf)),
        },
        "D2_orig_rank": {
            "mean": float(np.mean(rank_orig)),
            "rank1_rate": float(sum(r==1 for r in rank_orig)/len(rank_orig)),
        },
        "D3_signflip_rank": {
            "mean": float(np.mean(rank_signflip)),
            "rank1_rate": float(sum(r==1 for r in rank_signflip)/len(rank_signflip)),
        },
        "D4_random_matched_norm_rank": {
            "mean": float(np.mean(rank_rand)),
            "rank1_rate": float(sum(r==1 for r in rank_rand)/len(rank_rand)),
        },
        "D5_mu_only_rank": {
            "mean": float(np.mean(rank_mu)),
            "rank1_rate": float(sum(r==1 for r in rank_mu)/len(rank_mu)),
        },
        "D6_sigma_only_rank": {
            "mean": float(np.mean(rank_sig)),
            "rank1_rate": float(sum(r==1 for r in rank_sig)/len(rank_sig)),
        },
        "verdict": "PASS" if (
            sum(r==1 for r in rank_shuf)/len(rank_shuf) < 0.5 and
            sum(r==1 for r in rank_signflip)/len(rank_signflip) < 0.5 and
            sum(r==1 for r in rank_rand)/len(rank_rand) < 0.5
        ) else "FAIL",
    }
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {META_OUT}")
    log("=" * 70)


if __name__ == "__main__":
    main()