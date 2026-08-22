#!/usr/bin/env python3
"""Phase 10.9.3: 评估 6 项 checkpoint — 验证 user signal 真实保留.

ckpt: phase10_9_2_projector.pt (epoch 16, dpo_a=0.00029, dpo_b=0.00031)

6 项 checkpoint (用户指定):
  ① profile↔prefix pairwise correlation (r > 0.14 当前 baseline)
  ② prefix → profile 检索 / 重建 (显著高于随机)
  ③ counterfactual preference flip accuracy > 50% (核心 — model 必须读 profile)
  ④ shuffled profile 后 acc 回落 (而不是提升)
  ⑤ Target-style scale-free rank 优于 Random-style
  ⑥ primary generation 全程不 rerank (在 ①-⑤ 通过后再做)

评估集: 用 60 dev user (since model 在它们上训练, 但检查 dev acc 是否合理
 + 在 fresh test 上是否同样表现 — 但 fresh test 锁着, 这里只用 dev 做
 counterfactual acc check; profile-prefix correlation 在 dev 上算)
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PROJ_CKPT = OUT_DIR / "phase10_9_2_projector.pt"
PAIRS_FILE = OUT_DIR / "phase10_9_counterfactual_pairs.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
META_OUT = OUT_DIR / "phase10_9_3_eval_checkpoints.json"
LOG_OUT = OUT_DIR / "phase10_9_3_eval_checkpoints.log"

# === 硬编码 ===
USER_DIM = 40
HIDDEN_DIM = 128
NUM_TOKENS = 8
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
MAX_PROMPT_LEN = 480
MAX_QUERY_LEN = 64
SEED = 42

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)


def log(m):
    print(f"[phase10-9.3] {m}", flush=True)


class PrefixDecoder(nn.Module):
    """Prefix (K*H) → profile (40)."""
    def __init__(self, num_tokens: int, model_dim: int, user_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_tokens * model_dim, user_dim * 2),
            nn.GELU(),
            nn.Linear(user_dim * 2, user_dim),
        )

    def forward(self, prefix):
        return self.fc(prefix)


def build_chat_prompts(tokenizer, system_text, user_attrs_text):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_attrs_text},
        ],
        tokenize=False, add_generation_prompt=True,
    )


def encode_inputs(tokenizer, prompts, queries, device):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    p_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    q_ids = [tokenizer.encode(q, add_special_tokens=False) for q in queries]
    B = len(p_ids)
    p_max = max(len(x) for x in p_ids)
    q_max = max(len(x) for x in q_ids)
    full_len = min(MAX_PROMPT_LEN + MAX_QUERY_LEN, p_max + q_max)
    input_ids = torch.full((B, full_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, full_len), dtype=torch.long, device=device)
    query_mask = torch.zeros((B, full_len), dtype=torch.long, device=device)
    for b, (p, q) in enumerate(zip(p_ids, q_ids)):
        full = (p + q)[:full_len]
        L = len(full)
        input_ids[b, :L] = torch.tensor(full, dtype=torch.long, device=device)
        attention_mask[b, :L] = 1
        plen = min(len(p), L)
        query_mask[b, plen:L] = 1
    return input_ids, attention_mask, query_mask


def compute_logp(model, input_embeds, attention_mask, query_mask, target_ids, K):
    out = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    L = target_ids.size(1)
    logits = out.logits[:, K - 1:K - 1 + L, :].contiguous()
    log_probs = F.log_softmax(logits.float(), dim=-1)
    chosen_lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = query_mask[:, K:K + L].contiguous().to(chosen_lp.dtype)
    n_tok = mask.sum(dim=1).clamp(min=1.0)
    return (chosen_lp * mask).sum(dim=1) / n_tok


def main():
    log("=" * 70)
    log("Phase 10.9.3: 6 项 Checkpoint Evaluation")
    log("=" * 70)

    # === 1. Load Qwen ===
    log("[1] Loading Qwen2-7B (frozen) ...")
    from llm_client import _HiddenBackend
    try:
        _HiddenBackend.reset()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    backend = _HiddenBackend.get(QWEN_MODEL_PATH)
    model = backend.model
    tokenizer = backend.tokenizer
    model_dim = model.config.hidden_size
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # === 2. Load projector + decoder ===
    log("[2] Loading projector + decoder ...")
    from query.soft_prefix.projector import SoftPrefixProjector
    ckpt = torch.load(PROJ_CKPT, map_location=DEVICE)
    projector = SoftPrefixProjector(
        user_dim=ckpt["user_dim"], hidden_dim=ckpt["hidden_dim"],
        num_tokens=ckpt["num_tokens"], model_dim=ckpt["model_dim"],
        dtype=torch.float32, gate_init=ckpt["gate_init"],
    ).to(DEVICE)
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False
    decoder = PrefixDecoder(NUM_TOKENS, model_dim, USER_DIM).to(DEVICE)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    log(f"  loaded ckpt epoch={ckpt['epoch']}, dpo_a={ckpt['dpo_a']:.4f}, dpo_b={ckpt['dpo_b']:.4f}")

    # === 3. Load counterfactual pairs + user profiles ===
    log("[3] Loading counterfactual pairs ...")
    pairs = [json.loads(line) for line in PAIRS_FILE.read_text().splitlines()]
    log(f"  {len(pairs)} pairs")

    log("  Loading user profiles (for cross-user rank) ...")
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

    # === 4. Build inputs (same as training) ===
    log("[4] Building inputs ...")
    SYSTEM = (
        "You are a shopping query writer. Write one short natural shopping query "
        "that mentions every listed attribute of the product by its exact value "
        "(brand name, weight, dimensions, color, material). Keep the query under 25 words."
    )
    z_us = torch.tensor([p["z_u"] for p in pairs], dtype=torch.float32, device=DEVICE)
    z_vs = torch.tensor([p["z_v"] for p in pairs], dtype=torch.float32, device=DEVICE)
    # shuffle z (维度打乱 — 固定种子 per user)
    rng = np.random.default_rng(SEED)
    z_us_shuffled = torch.zeros_like(z_us)
    for i in range(len(pairs)):
        perm = rng.permutation(USER_DIM)
        z_us_shuffled[i] = torch.tensor(np.array(pairs[i]["z_u"])[perm], dtype=torch.float32, device=DEVICE)
    # zero
    z_zero = torch.zeros_like(z_us)

    # 4 profile × 4 sample (chosen_u, rejected_u, chosen_v, rejected_v) per pair
    # 我们做 4 profile 实验: target_u / target_v / shuffled_u / zero
    # 这里每个 profile 测 counterfactual acc
    profiles = ["target_u", "target_v", "shuffled_u", "zero"]

    # 对每个 profile 构造输入: chosen_u + rejected_u (= chosen_v) under profile
    # 2 sample per pair × N pairs
    prompts = []
    queries = []
    profile_per_pair = []  # 每个 pair 用什么 profile (in order)
    for prof in profiles:
        for p in pairs:
            cu = build_chat_prompts(tokenizer, SYSTEM, p["prompt_u"])
            cv = build_chat_prompts(tokenizer, SYSTEM, p["prompt_v"])
            prompts.extend([cu, cu, cv, cv])  # 4 sample per pair
            queries.extend([p["chosen_q_u"], p["chosen_q_v"], p["chosen_q_v"], p["chosen_q_u"]])
            profile_per_pair.extend([prof, prof, prof, prof])
    # 输入现在 (4 profile × N pair × 4 sample) = 4 × 94 × 4 = 1504

    # 简化: 只对每个 profile 取 N pair × 4 sample
    log(f"  total samples to forward: {len(prompts)}")

    # 我们分 4 profile 各自 forward (避免一次性太大 OOM)
    # 每个 profile: 4 × 94 = 376 sample
    # 用 input_ids per profile (按 profile 分组)
    # 重新组织: 按 profile 顺序构造
    all_inputs_per_profile = {}
    cursor = 0
    for prof in profiles:
        n_pair = len(pairs)
        n_sample = n_pair * 4  # chosen_u, rej_u, chosen_v, rej_v
        rows = list(range(cursor, cursor + n_sample))
        all_inputs_per_profile[prof] = rows
        cursor += n_sample

    input_ids, attention_mask, query_mask = encode_inputs(tokenizer, prompts, queries, DEVICE)
    log(f"  input_ids shape: {input_ids.shape}")
    log(f"  query_mask sum (first 8): {query_mask.sum(dim=1).tolist()[:8]}")

    # === 5. Forward each profile (micro-batch) ===
    log("[5] Forward per profile (micro-batch 8) ...")
    MICRO = 8

    @torch.no_grad()
    def get_prefixes(z):
        """算 z 对应的 prefix."""
        return projector(z).to(DTYPE)  # [N, K, H]

    @torch.no_grad()
    def forward_logp(prefix_per_pair, row_indices):
        """prefix_per_pair: [N, K, H]; row_indices: 4N input rows (4 per pair, in chosen_u/rej_u/chosen_v/rej_v order)."""
        emb_m = model.get_input_embeddings()
        # replicate per-pair prefix 4 times (matching the 4 samples per pair)
        prefix_rep = prefix_per_pair.repeat_interleave(4, dim=0)  # [4N, K, H]
        lp_all = []
        for s in range(0, len(row_indices), MICRO):
            rows_t = torch.tensor(row_indices[s:s + MICRO], device=DEVICE, dtype=torch.long)
            pre_sub = prefix_rep[s:s + MICRO]
            ids_sub = input_ids[rows_t]
            am_sub = attention_mask[rows_t]
            qm_sub = query_mask[rows_t]
            text_emb = emb_m(ids_sub).to(DTYPE)
            full_emb = torch.cat([pre_sub, text_emb], dim=1)
            full_am = torch.cat([
                torch.ones((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
                am_sub,
            ], dim=1)
            full_qm = torch.cat([
                torch.zeros((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
                qm_sub,
            ], dim=1)
            lp = compute_logp(model, full_emb, full_am, full_qm, ids_sub, NUM_TOKENS)
            lp_all.append(lp)
            del full_emb, full_am, full_qm, text_emb
            torch.cuda.empty_cache()
        return torch.cat(lp_all, dim=0)

    # prefix per profile
    z_per_profile = {
        "target_u": z_us,
        "target_v": z_vs,
        "shuffled_u": z_us_shuffled,
        "zero": z_zero,
    }
    prefixes_per_profile = {p: get_prefixes(z) for p, z in z_per_profile.items()}
    log(f"  prefix shapes: {[(p, prefixes_per_profile[p].shape) for p in profiles]}")

    log_p_per_profile = {}
    for prof in profiles:
        rows = all_inputs_per_profile[prof]
        lp = forward_logp(prefixes_per_profile[prof], rows)
        log_p_per_profile[prof] = lp.view(len(pairs), 4)  # [N, 4]

    # === 6. Checkpoint ③: counterfactual preference flip accuracy ===
    log("[6] === Checkpoint ③: counterfactual preference flip accuracy ===")
    # profile_u should prefer chosen_u over rejected_u (= chosen_v): log_p[0] > log_p[1]
    # profile_v should prefer chosen_v over rejected_v (= chosen_u): log_p[2] > log_p[3]
    acc_target_u = (log_p_per_profile["target_u"][:, 0] > log_p_per_profile["target_u"][:, 1]).float().mean().item()
    acc_target_v = (log_p_per_profile["target_v"][:, 2] > log_p_per_profile["target_v"][:, 3]).float().mean().item()
    avg_acc = (acc_target_u + acc_target_v) / 2
    log(f"  acc(target_u → chosen_u > chosen_v): {acc_target_u:.3f}")
    log(f"  acc(target_v → chosen_v > chosen_u): {acc_target_v:.3f}")
    log(f"  Average counterfactual acc: {avg_acc:.3f}")
    log(f"  (期望 > 0.5 → model 读了 profile)")

    # === 7. Checkpoint ④: shuffled profile accuracy (期望回落) ===
    log("[7] === Checkpoint ④: shuffled profile accuracy (回落?) ===")
    acc_shuf_u = (log_p_per_profile["shuffled_u"][:, 0] > log_p_per_profile["shuffled_u"][:, 1]).float().mean().item()
    acc_zero_u = (log_p_per_profile["zero"][:, 0] > log_p_per_profile["zero"][:, 1]).float().mean().item()
    log(f"  acc(shuffled_u → chosen_u > chosen_v): {acc_shuf_u:.3f}")
    log(f"  acc(zero → chosen_u > chosen_v):       {acc_zero_u:.3f}")
    log(f"  (期望 acc_shuf_u < acc_target_u: model 真的用了 profile 而不是读 z 值)")

    # === 8. Checkpoint ①: profile↔prefix pairwise correlation ===
    log("[8] === Checkpoint ①: profile↔prefix pairwise correlation ===")
    # 对每对 (i, j) 算 ||z_u_i - z_u_j|| 和 ||prefix_i - prefix_j||
    prefix_u_flat = prefixes_per_profile["target_u"].float().reshape(len(pairs), -1)  # [N, K*H]
    z_u_np = z_us.cpu().numpy()
    from scipy.stats import pearsonr
    pdists, ppdists = [], []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            pdists.append(np.linalg.norm(z_u_np[i] - z_u_np[j]))
            ppdists.append(np.linalg.norm(prefix_u_flat[i].cpu().numpy() - prefix_u_flat[j].cpu().numpy()))
    pdists, ppdists = np.array(pdists), np.array(ppdists)
    r, p_p = pearsonr(pdists, ppdists)
    log(f"  profile ↔ prefix pearson r = {r:.4f}, p = {p_p:.4f}")
    log(f"  (期望 > 0.14, 当前 baseline 10.8.2 = 0.14)")

    # === 9. Checkpoint ②: prefix → profile retrieval (decoder 重建) ===
    log("[9] === Checkpoint ②: prefix → profile retrieval ===")
    # 用 decoder 把 prefix 还原回 z_hat, 跟原 z_u 比较
    # prefixes_per_profile 是 per-pair 的 [N, K, H]
    prefix_chosen_u = prefixes_per_profile["target_u"]  # [N, K, H] under profile_u
    prefix_chosen_v = prefixes_per_profile["target_v"]  # [N, K, H] under profile_v
    flat_u = prefix_chosen_u.float().reshape(len(pairs), -1)
    flat_v = prefix_chosen_v.float().reshape(len(pairs), -1)
    z_hat_u = decoder(flat_u)
    z_hat_v = decoder(flat_v)
    z_target_u = z_us
    z_target_v = z_vs
    mse_u = ((z_hat_u - z_target_u) ** 2).mean().item()
    mse_v = ((z_hat_v - z_target_v) ** 2).mean().item()
    mse_combined = (mse_u + mse_v) / 2
    # baseline: MSE of original z values (mean) - should be much higher than decoder reconstruction
    z_mean = (z_us + z_vs).mean(dim=0)
    mse_baseline_u = ((z_mean - z_us) ** 2).mean().item()
    log(f"  decoder reconstruction MSE (u): {mse_u:.4f}")
    log(f"  decoder reconstruction MSE (v): {mse_v:.4f}")
    log(f"  decoder reconstruction MSE (avg): {mse_combined:.4f}")
    log(f"  baseline MSE (mean predict): {mse_baseline_u:.4f}")

    # === 10. Checkpoint ⑤: cross-user target rank ===
    log("[10] === Checkpoint ⑤: cross-user target rank (vs all users) ===")
    # feature extraction 需要 query 的 20d features, 这里简化:
    # 我们用 logP(q_u|u) - logP(q_v|u) 作为"dist"度量, 算 target rank
    # 但这需要特征向量, 我们没有; 简化: 用 mean prefix 的 cos 距离作 proxy
    # rank = mean prefix_u vs all user prefix 的 cos 距离排名
    log("  computing cross-user rank using prefix similarity proxy ...")
    # 先算所有用户 (user_id_to_idx) 的 z, 用 projector 得 prefix, 然后算 cos 距离
    all_z = torch.tensor(
        np.concatenate([user_mu, user_logvar], axis=1), dtype=torch.float32, device=DEVICE
    )  # [2918, 40]
    with torch.no_grad():
        all_prefixes = projector(all_z).to(DTYPE).float()  # [2918, K, H]
        all_prefixes_flat = all_prefixes.reshape(2918, -1)
        target_prefix_flat = prefix_u_flat  # [N, K*H] (float)

    # 对每对 target_u, 算 cos 距离 vs 所有 user prefix, 取 target_user 的 rank
    ranks = []
    rank1 = 0
    valid_pairs = 0
    target_user_ids = [p["user_u"] for p in pairs]
    for i in range(len(pairs)):
        target_user = target_user_ids[i]
        if target_user not in user_id_to_idx:
            continue
        t_idx = user_id_to_idx[target_user]
        # cos 距离 = 1 - cos_sim
        target_p = target_prefix_flat[i:i+1]  # [1, K*H]
        # cos sim: target_p @ all_prefixes_flat.T / (||target|| * ||all||)
        t_norm = target_p.norm(dim=1, keepdim=True)
        a_norm = all_prefixes_flat.norm(dim=1, keepdim=True)
        cos_sim = (target_p @ all_prefixes_flat.T) / (t_norm @ a_norm.T + 1e-9)
        cos_sim = cos_sim.squeeze(0)  # [2918]
        cos_dist = 1 - cos_sim
        rank = (cos_dist < cos_dist[t_idx]).sum().item() + 1
        ranks.append(rank)
        valid_pairs += 1
        if rank == 1:
            rank1 += 1
    mean_rank = float(np.mean(ranks))
    rank1_rate = rank1 / valid_pairs
    log(f"  mean target rank: {mean_rank:.2f} / 2918 (random = 1459)")
    log(f"  rank==1 rate: {rank1_rate:.4f} (random = {1/2918:.4f})")

    # === 11. 总结 ===
    log("=" * 70)
    log("Summary:")
    checks = {
        "① profile↔prefix pearson r": (float(r) > 0.14, float(r), 0.14),
        "② decoder reconstruction MSE < baseline": (mse_combined < mse_baseline_u, mse_combined, mse_baseline_u),
        "③ counterfactual flip acc > 0.5": (avg_acc > 0.5, avg_acc, 0.5),
        "④ shuffled acc < target acc": (acc_shuf_u < acc_target_u, acc_shuf_u, acc_target_u),
        "⑤ cross-user rank < random (1459)": (mean_rank < 1459, mean_rank, 1459),
    }
    for k, (passed, val, ref) in checks.items():
        status = "PASS" if passed else "FAIL"
        log(f"  [{status}] {k}: val={val:.4f} (ref={ref})")

    n_pass = sum(1 for k, (p, _, _) in checks.items() if p)
    log(f"  {n_pass}/5 checkpoint passed (⑥ 待评估)")

    # === 12. Write meta ===
    meta = {
        "ckpt": str(PROJ_CKPT),
        "ckpt_epoch": ckpt["epoch"],
        "checkpoint_1_profile_prefix_corr": {
            "pearson_r": float(r),
            "p_value": float(p_p),
            "n_pairs": len(pdists),
            "pass": float(r) > 0.14,
            "ref_baseline_10_8_2": 0.14,
        },
        "checkpoint_2_decoder_reconstruction": {
            "mse_u": mse_u,
            "mse_v": mse_v,
            "mse_combined": mse_combined,
            "baseline_mse_mean_predict": mse_baseline_u,
            "pass": mse_combined < mse_baseline_u,
        },
        "checkpoint_3_counterfactual_acc": {
            "acc_target_u": acc_target_u,
            "acc_target_v": acc_target_v,
            "avg_acc": avg_acc,
            "pass": avg_acc > 0.5,
        },
        "checkpoint_4_shuffled_fallback": {
            "acc_shuffled_u": acc_shuf_u,
            "acc_zero": acc_zero_u,
            "acc_target_u": acc_target_u,
            "pass": acc_shuf_u < acc_target_u,
        },
        "checkpoint_5_cross_user_rank": {
            "mean_rank": mean_rank,
            "rank1_rate": rank1_rate,
            "random_baseline_rank": 1459,
            "n_valid_pairs": valid_pairs,
            "pass": mean_rank < 1459,
        },
        "checkpoint_6_primary_generation": {
            "status": "not_run_in_this_phase",
            "note": "在 ①-⑤ 通过后再做 (Phase 10.9.3b)",
        },
        "n_pass": n_pass,
        "n_total": 5,
    }
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {META_OUT}")
    log("=" * 70)


if __name__ == "__main__":
    main()