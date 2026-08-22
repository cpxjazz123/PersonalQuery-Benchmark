#!/usr/bin/env python3
"""Phase 10.8.2: 用户条件反事实测试 — 判定 projector 是否真用 user profile.

核心问题: 训练集 chosen_acc=0.938 证明 DPO 链路能学偏好,但 cross-user rank≈random
证明生成 query 没有真靠近 target_user。本任务直接判定 projector 究竟有没有读 user profile。

测试 4 种 profile:
  - target    : 真 target user 的 z_u = [mu_u, logvar_u]
  - random    : 另一个 random user 的 z_v (z_v ≠ z_u)
  - zero      : 全 0 z
  - shuffled  : target_user 的 z 但维度打乱 (相同值, 错位置 → 值相同但 profile 距离≠0)

对每对 (chosen, rejected, profile):
  M(u) = logP(q+|u) - logP(q-|u)

判断准则:
  真正用了 user profile 的模型应满足 M(target) > M(random/shuffled/zero)

同时记录 5 项辅助诊断:
  ① prefix 方差/有效秩 (不同 user 生成的 prefix 是不是其实一样 → projector collapse)
  ② profile 距离 vs prefix 距离相关性 (profile 距离远 → prefix 距离是否也远?)
  ③ 不同 prefix → Qwen logits/hidden 差异 (prefix 是否被模型吸收)
  ④ M(target) - M(random) 是否显著 > 0 (projector 是否区分了 user)
  ⑤ M(target) ≈ M(shuffled) → DPO 只学 chosen 特征, 完全忽略 user
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "preference_pairs_60.jsonl"
PROJ_CKPT = OUT_DIR / "phase10_6_3_projector.pt"
DEV_USER_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"
TEST_USER_FILE = OUT_DIR / "phase10_test_60_user_ids.json"
META_OUT = OUT_DIR / "phase10_8_2_profile_causal.json"
LOG_OUT = OUT_DIR / "phase10_8_2_profile_causal.log"

# === 硬编码 ===
USER_DIM = 40
HIDDEN_DIM = 128
NUM_TOKENS = 8
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
MAX_PROMPT_LEN = 480
MAX_QUERY_LEN = 64
N_PAIRS_TEST = 30       # 取前 30 对做反事实
N_RANDOM_USERS_PER_PAIR = 1  # 每对选 1 个 random user (避免组合爆炸)
SEED = 42

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)


def log(m):
    print(f"[phase10-8.2] {m}", flush=True)


# ============================================================================
# 加载
# ============================================================================
def load_dev_pairs(n_pairs: int) -> List[dict]:
    records = []
    with PAIRS_FILE.open() as f:
        for line in f:
            records.append(json.loads(line))
    # 按 maha_margin desc, 取 top n_pairs (高 margin 更稳)
    records.sort(key=lambda r: r["maha_rejected"] - r["maha_chosen"], reverse=True)
    selected = records[:n_pairs]
    log(f"  loaded {len(records)} pairs, selected top {len(selected)} by margin")
    return selected


def build_chat_prompts(tokenizer, system_text: str, user_attrs_text: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_attrs_text},
        ],
        tokenize=False, add_generation_prompt=True,
    )


def encode_inputs(tokenizer, prompt_chats: List[str], queries: List[str], device: str):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_ids_list = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_chats]
    query_ids_list = [tokenizer.encode(q, add_special_tokens=False) for q in queries]
    B = len(prompt_ids_list)
    p_max = max(len(x) for x in prompt_ids_list)
    q_max = max(len(x) for x in query_ids_list)
    full_len = min(MAX_PROMPT_LEN + MAX_QUERY_LEN, p_max + q_max)
    input_ids = torch.full((B, full_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, full_len), dtype=torch.long, device=device)
    query_mask = torch.zeros((B, full_len), dtype=torch.long, device=device)
    p_lens = []
    for b, (p_ids, q_ids) in enumerate(zip(prompt_ids_list, query_ids_list)):
        full_ids = (p_ids + q_ids)[:full_len]
        L = len(full_ids)
        input_ids[b, :L] = torch.tensor(full_ids, dtype=torch.long, device=device)
        attention_mask[b, :L] = 1
        p_len = min(len(p_ids), L)
        p_lens.append(p_len)
        query_mask[b, p_len:L] = 1
    return input_ids, attention_mask, query_mask, p_lens


def compute_logp_for_prefix(
    model, input_embeds, attention_mask, query_mask, target_ids, num_prefix_tokens: int
) -> torch.Tensor:
    """返回每个样本 query 部分的 average log-prob."""
    out = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    K = num_prefix_tokens
    L = target_ids.size(1)
    logits = out.logits[:, K - 1:K - 1 + L, :].contiguous()
    log_probs = F.log_softmax(logits.float(), dim=-1)
    chosen_lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    text_qm = query_mask[:, K:K + L].contiguous()
    mask = text_qm.to(chosen_lp.dtype)
    n_tokens = mask.sum(dim=1).clamp(min=1.0)
    return (chosen_lp * mask).sum(dim=1) / n_tokens


def effective_rank(x: np.ndarray, eps: float = 1e-12) -> float:
    """有效秩: exp(entropy of normalized singular values)."""
    s = np.linalg.svd(x, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > eps]
    ent = -np.sum(p * np.log(p))
    return float(np.exp(ent))


# ============================================================================
# Main
# ============================================================================
def main():
    log("=" * 70)
    log("Phase 10.8.2: 用户条件反事实测试")
    log("=" * 70)

    # === 1. 加载 dev/test 用户 (确保不偷看 test) ===
    dev_ids = set(json.loads(DEV_USER_FILE.read_text())["user_ids"])
    test_ids = set(json.loads(TEST_USER_FILE.read_text())["user_ids"])
    assert dev_ids & test_ids == set(), "FAIL: dev ∩ test ≠ ∅"
    log(f"  dev users = {len(dev_ids)}, test users = {len(test_ids)}, ∩ = 0 ✓")

    # === 2. 加载 dev preference pairs ===
    log("[2] Loading dev preference pairs ...")
    pairs = load_dev_pairs(N_PAIRS_TEST)
    dev_pair_users = [p["user_id"] for p in pairs]
    log(f"  test {N_PAIRS_TEST} pairs, user_ids[0]={dev_pair_users[0][:12]}...")

    # === 3. 加载 Qwen (frozen) ===
    log("[3] Loading Qwen2-7B (frozen) ...")
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
    log(f"  model_dim={model_dim}, dtype={backend.dtype}")

    # === 4. 加载 projector ===
    log("[4] Loading projector ...")
    from query.soft_prefix.projector import SoftPrefixProjector
    ckpt = torch.load(PROJ_CKPT, map_location=DEVICE)
    projector = SoftPrefixProjector(
        user_dim=ckpt.get("user_dim", USER_DIM),
        hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
        num_tokens=ckpt.get("num_tokens", NUM_TOKENS),
        model_dim=ckpt["model_dim"],
        dtype=torch.float32,
        gate_init=1e-3,
    ).to(DEVICE)
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()
    for p in projector.parameters():
        p.requires_grad = False
    log(f"  loaded ckpt, num_tokens={projector.num_tokens}, model_dim={projector.model_dim}")

    # === 5. 准备输入 ===
    log("[5] Preparing inputs (chosen + rejected per pair) ...")
    SYSTEM = (
        "You are a shopping query writer. Write one short natural shopping query "
        "that mentions every listed attribute of the product by its exact value "
        "(brand name, weight, dimensions, color, material). Keep the query under 25 words."
    )

    # 对每个 pair 构造 4 种 profile 的 z_u (40d)
    # profile = target (z_u), random (z_v), zero (zeros), shuffled (z_u 维度打乱)
    # 每 profile × 1 chosen + 1 rejected = 8 forward per pair
    rng = np.random.default_rng(SEED)
    profile_names = ["target", "random", "zero", "shuffled"]
    z_per_pair_profile: List[Dict[str, np.ndarray]] = []
    prompts = []
    queries = []  # 0=chosen, 1=rejected
    is_chosen = []
    for p in pairs:
        z_u = np.array(p["mu_u"] + p["log_sigma_u"], dtype=np.float32)
        # random: 从 test_ids 中抽一个(确保不是自己)
        random_candidates = [u for u in test_ids if u != p["user_id"]]
        random_user_id = rng.choice(random_candidates)
        # 这里 random_user 的 z_u 我们没存;但能从所有 user_profiles 中拿
        z_v = None  # 后面填
        # shuffled: z_u 维度打乱 (固定种子, 同一 user 同 shuffle)
        rng2 = np.random.default_rng(hash(p["user_id"]) & 0xFFFFFFFF)
        perm = rng2.permutation(USER_DIM)
        z_shuf = z_u[perm].copy()

        z_per_pair_profile.append({
            "target": z_u,
            "random": z_v,   # 占位
            "zero": np.zeros(USER_DIM, dtype=np.float32),
            "shuffled": z_shuf,
            "user_id": p["user_id"],
            "random_user_id": random_user_id,
        })

        # prompt + queries
        chat = build_chat_prompts(tokenizer, SYSTEM, p["prompt_target_style"])
        # chosen + rejected 配对 (2 个 sample)
        for _ in range(4):  # 4 profiles
            prompts.append(chat)
            prompts.append(chat)
            queries.append(p["chosen_query"])
            queries.append(p["rejected_query"])
            is_chosen.append(1.0)
            is_chosen.append(0.0)

    # 加载 random_user 的 z_v (inline, 避免 import 路径问题)
    VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
    TAG = "vades_prototype_3000u_v6_raw"
    profile_path = VADES_DIR / f"{TAG}_user_profiles.jsonl"
    user_id_to_idx = {}
    user_mu_arr = []
    user_logvar_arr = []
    with profile_path.open() as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            user_id_to_idx[row["user_id"]] = idx
            user_mu_arr.append(row["user_mu"])
            user_logvar_arr.append(row["user_logvar"])
    user_mu = np.array(user_mu_arr, dtype=np.float32)
    user_logvar = np.array(user_logvar_arr, dtype=np.float32)
    for entry in z_per_pair_profile:
        ridx = user_id_to_idx[entry["random_user_id"]]
        entry["random"] = np.concatenate([user_mu[ridx], user_logvar[ridx]]).astype(np.float32)

    log(f"  prepared {len(z_per_pair_profile)} pairs × 4 profiles = {len(z_per_pair_profile) * 4 * 2} samples")

    # === 6. 编码 inputs ===
    log("[6] Encoding inputs ...")
    input_ids, attention_mask, query_mask, p_lens = encode_inputs(
        tokenizer, prompts, queries, DEVICE
    )
    log(f"  input_ids shape: {input_ids.shape}")
    log(f"  query_mask sum (first 4): {query_mask.sum(dim=1).tolist()[:4]}")

    # === 7. 对每个 profile 算 prefix + logP ===
    log("[7] Computing prefix + logP per profile ...")
    results_per_pair_profile: List[Dict[str, dict]] = []

    # 收集 prefix 用于诊断 ①
    all_prefixes_target = []  # 用于算方差/秩
    all_prefixes_random = []
    all_prefixes_zero = []
    all_prefixes_shuffled = []

    # 收集 hidden state (last layer last token) 用于诊断 ③
    hidden_per_profile = {p: [] for p in profile_names}

    # profile → prefix tensor (batched)
    # 8 samples per pair × N_PAIRS = 240 samples, 一次 forward 即可
    B_total = len(prompts)  # N_PAIRS * 4 profiles * 2 (chosen/rejected)
    # 对每 profile 一次性 forward (按 profile 切分)
    log_p_per_pair_profile: Dict[str, List[Tuple[float, float]]] = {
        p: [] for p in profile_names
    }

    @torch.no_grad()
    def forward_one_profile(profile: str):
        # 每个 profile 在 input_ids 中占 2*N_PAIRS 行
        # z_all shape: [N_PAIRS*2, 40] (each pair 的 chosen/rejected 共享同一 z)
        z_all = torch.tensor(
            np.stack([entry[profile] for entry in z_per_pair_profile] * 2, axis=0),
            dtype=torch.float32, device=DEVICE,
        )  # [N_PAIRS*2, 40]
        prefix = projector(z_all).to(DTYPE)  # [N_PAIRS*2, K, H]

        # 收集 prefix 用于诊断 ①
        if profile == "target":
            all_prefixes_target.append(prefix.float().cpu().numpy())
        elif profile == "random":
            all_prefixes_random.append(prefix.float().cpu().numpy())
        elif profile == "zero":
            all_prefixes_zero.append(prefix.float().cpu().numpy())
        elif profile == "shuffled":
            all_prefixes_shuffled.append(prefix.float().cpu().numpy())

        # forward: prefix + text_emb (只取本 profile 对应的 input_ids 子集)
        prof_idx = profile_names.index(profile)
        # rows for this profile: [prof_idx * 2*N_PAIRS : (prof_idx+1) * 2*N_PAIRS]
        rows_start = prof_idx * (2 * N_PAIRS_TEST)
        rows_end = rows_start + (2 * N_PAIRS_TEST)
        ids_sub = input_ids[rows_start:rows_end]
        am_sub = attention_mask[rows_start:rows_end]
        emb_m = model.get_input_embeddings()
        text_emb = emb_m(ids_sub).to(DTYPE)
        full_emb = torch.cat([prefix, text_emb], dim=1)
        full_am = torch.cat([
            torch.ones((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
            am_sub,
        ], dim=1)
        full_qm = torch.cat([
            torch.zeros((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
            query_mask[rows_start:rows_end],
        ], dim=1)

        avg_lp = compute_logp_for_prefix(
            model, full_emb, full_am, full_qm, ids_sub, NUM_TOKENS
        )
        # reshape to (N_PAIRS, 2) — chosen / rejected
        lp_paired = avg_lp.view(-1, 2).cpu().numpy()  # [N_PAIRS, 2]
        # logP chosen - logP rejected
        M_u = lp_paired[:, 0] - lp_paired[:, 1]
        return M_u.tolist(), lp_paired.tolist(), full_emb

    log("  --- profile=target ---")
    M_target, lp_target, _ = forward_one_profile("target")
    log(f"    mean M(target) = {np.mean(M_target):.4f}, std = {np.std(M_target):.4f}")
    log("  --- profile=random ---")
    M_random, lp_random, _ = forward_one_profile("random")
    log(f"    mean M(random) = {np.mean(M_random):.4f}, std = {np.std(M_random):.4f}")
    log("  --- profile=zero ---")
    M_zero, lp_zero, _ = forward_one_profile("zero")
    log(f"    mean M(zero)   = {np.mean(M_zero):.4f}, std = {np.std(M_zero):.4f}")
    log("  --- profile=shuffled ---")
    M_shuffled, lp_shuffled, _ = forward_one_profile("shuffled")
    log(f"    mean M(shuffled)={np.mean(M_shuffled):.4f}, std = {np.std(M_shuffled):.4f}")

    # === 8. 诊断 ④: M(target) vs M(other) ===
    log("[8] === 诊断 ④: M(target) vs M(other) ===")
    M_target_arr = np.array(M_target)
    M_random_arr = np.array(M_random)
    M_zero_arr = np.array(M_zero)
    M_shuffled_arr = np.array(M_shuffled)

    diff_target_random = M_target_arr - M_random_arr
    diff_target_zero = M_target_arr - M_zero_arr
    diff_target_shuffled = M_target_arr - M_shuffled_arr

    log(f"  M(target) - M(random)   : mean = {diff_target_random.mean():.4f}, paired p ≈ sign test")
    # 配对 sign test
    pos_vs_random = (diff_target_random > 0).sum()
    log(f"    sign(target > random): {pos_vs_random}/{len(diff_target_random)} = {pos_vs_random/len(diff_target_random):.4f}")
    log(f"  M(target) - M(zero)     : mean = {diff_target_zero.mean():.4f}")
    pos_vs_zero = (diff_target_zero > 0).sum()
    log(f"    sign(target > zero): {pos_vs_zero}/{len(diff_target_zero)} = {pos_vs_zero/len(diff_target_zero):.4f}")
    log(f"  M(target) - M(shuffled) : mean = {diff_target_shuffled.mean():.4f}")
    pos_vs_shuf = (diff_target_shuffled > 0).sum()
    log(f"    sign(target > shuffled): {pos_vs_shuf}/{len(diff_target_shuffled)} = {pos_vs_shuf/len(diff_target_shuffled):.4f}")

    # === 9. 诊断 ①: prefix 方差 / 有效秩 ===
    log("[9] === 诊断 ①: prefix 方差 / 有效秩 ===")
    # prefix shape: [N_PAIRS*2, K, H]
    # 把 chosen/rejected 区分开: 取 index 0,2,4,... (chosen) 和 1,3,5,... (rejected)
    # 实际上 pair 是 (chosen, rejected) 配对,所以 reshaped 为 [N_PAIRS, 2, K, H]
    def prefix_stats(prefix_arr, name):
        # prefix_arr: [N_PAIRS*2, K, H]
        chosen = prefix_arr[0::2]  # [N_PAIRS, K, H]
        rejected = prefix_arr[1::2]  # [N_PAIRS, K, H]
        # 把所有 chosen prefix flatten 为 [N_PAIRS, K*H]
        all_flat = prefix_arr.reshape(prefix_arr.shape[0], -1)
        var_per_token = np.var(all_flat, axis=0).mean()
        std_per_token = np.std(all_flat, axis=0).mean()
        # effective rank (SVD on [N_PAIRS*2, K*H])
        if all_flat.shape[0] >= 2:
            erank = effective_rank(all_flat)
        else:
            erank = 0.0
        # 配对差异 (chosen - rejected)
        diff = chosen - rejected
        diff_norm = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1).mean()
        log(f"  {name}: var_per_token={var_per_token:.6f}, std={std_per_token:.6f}, "
            f"effective_rank={erank:.2f}, mean_||chosen-rejected||={diff_norm:.4f}")
        return {
            "var_per_token": float(var_per_token),
            "std_per_token": float(std_per_token),
            "effective_rank": float(erank),
            "mean_chosen_rejected_diff": float(diff_norm),
        }

    p_stats = {
        "target":   prefix_stats(all_prefixes_target[0], "target"),
        "random":   prefix_stats(all_prefixes_random[0], "random"),
        "zero":     prefix_stats(all_prefixes_zero[0], "zero"),
        "shuffled": prefix_stats(all_prefixes_shuffled[0], "shuffled"),
    }

    # === 10. 诊断 ②: profile 距离 vs prefix 距离相关性 ===
    log("[10] === 诊断 ②: profile 距离 vs prefix 距离相关性 ===")
    # 对每对 (i, j) 计算 ||profile_i - profile_j|| 和 ||prefix_i - prefix_j||
    z_target = np.array([e["target"] for e in z_per_pair_profile])  # [N, 40]
    prefix_target_flat = all_prefixes_target[0][0::2].reshape(N_PAIRS_TEST, -1)  # [N, K*H]

    from scipy.stats import pearsonr
    profile_dists = []
    prefix_dists = []
    for i in range(N_PAIRS_TEST):
        for j in range(i + 1, N_PAIRS_TEST):
            profile_dists.append(np.linalg.norm(z_target[i] - z_target[j]))
            prefix_dists.append(np.linalg.norm(prefix_target_flat[i] - prefix_target_flat[j]))
    profile_dists = np.array(profile_dists)
    prefix_dists = np.array(prefix_dists)
    corr, p_corr = pearsonr(profile_dists, prefix_dists)
    log(f"  profile-distance vs prefix-distance: pearson r = {corr:.4f}, p = {p_corr:.4f}")
    log(f"    (理想: r > 0.5 强相关 → prefix 真编码 profile; r ≈ 0 → prefix 忽略 profile)")

    # === 11. 诊断 ③: 不同 profile 下 Qwen logits / hidden 差异 ===
    log("[11] === 诊断 ③: 不同 prefix → Qwen logits/hidden 差异 ===")
    # 取一个 prompt (pair 0 的 chosen_query), 用 4 profile 各自 forward, 比较 logits
    @torch.no_grad()
    def get_logits_for_first_chosen(profile: str):
        z_test = torch.tensor(z_per_pair_profile[0][profile], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        prefix = projector(z_test).to(DTYPE)  # [1, K, H]
        # first chosen 的 input
        ids = input_ids[0:1]
        am = attention_mask[0:1]
        emb_m = model.get_input_embeddings()
        text_emb = emb_m(ids).to(DTYPE)
        full_emb = torch.cat([prefix, text_emb], dim=1)
        full_am = torch.cat([
            torch.ones((1, NUM_TOKENS), dtype=torch.long, device=DEVICE),
            am,
        ], dim=1)
        out = model(inputs_embeds=full_emb, attention_mask=full_am, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, NUM_TOKENS:, :].float().cpu().numpy()  # [1, L, H]
        # logits at last text position
        logits = out.logits[:, -1, :].float().cpu().numpy()  # [1, V]
        return last_hidden, logits

    h_target, l_target = get_logits_for_first_chosen("target")
    h_random, l_random = get_logits_for_first_chosen("random")
    h_zero, l_zero = get_logits_for_first_chosen("zero")
    h_shuffled, l_shuffled = get_logits_for_first_chosen("shuffled")

    logit_diffs = {
        "target_vs_random":   float(np.linalg.norm(l_target - l_random)),
        "target_vs_zero":     float(np.linalg.norm(l_target - l_zero)),
        "target_vs_shuffled": float(np.linalg.norm(l_target - l_shuffled)),
        "random_vs_zero":     float(np.linalg.norm(l_random - l_zero)),
    }
    hidden_diffs = {
        "target_vs_random":   float(np.linalg.norm(h_target - h_random)),
        "target_vs_zero":     float(np.linalg.norm(h_target - h_zero)),
        "target_vs_shuffled": float(np.linalg.norm(h_target - h_shuffled)),
        "random_vs_zero":     float(np.linalg.norm(h_random - h_zero)),
    }
    log(f"  logit diffs: {logit_diffs}")
    log(f"  hidden diffs: {hidden_diffs}")
    log(f"    (若 diffs ≈ 0 → prefix 被忽略; 若 target_vs_random > target_vs_zero → prefix 区分了 user)")

    # === 12. 诊断分流 ===
    log("[12] === 诊断分流 (根因判定) ===")
    # 5 个判定
    # A. prefix collapse: 不同 profile 下 prefix 方差极小, 有效秩低
    target_var = p_stats["target"]["var_per_token"]
    random_var = p_stats["random"]["var_per_token"]
    zero_var = p_stats["zero"]["var_per_token"]
    target_erank = p_stats["target"]["effective_rank"]

    prefix_collapsed = (
        target_var < 1e-3 and random_var < 1e-3 and
        abs(target_var - random_var) < 1e-4
    )
    # B. prefix 不同但 Qwen 忽略: prefix 有方差但 logit_diff 几乎为 0
    prefix_ignored = (
        target_var > 1e-3 and
        logit_diffs["target_vs_random"] < 1e-3
    )
    # C. DPO 目标错位: target prefix 跟 zero/shuffled 在 M 上没区别
    dpo_target_wrong = (
        abs(diff_target_zero.mean()) < 0.05 and
        abs(diff_target_shuffled.mean()) < 0.05
    )
    # D. profile 距离 ↔ prefix 距离 相关性弱
    profile_prefix_decoupled = corr < 0.3

    diagnosis = {
        "prefix_collapsed": bool(prefix_collapsed),
        "prefix_ignored_by_qwen": bool(prefix_ignored),
        "dpo_target_wrong_alignment": bool(dpo_target_wrong),
        "profile_prefix_decoupled": bool(profile_prefix_decoupled),
        "profile_prefix_pearson_r": float(corr),
        "profile_prefix_pearson_p": float(p_corr),
    }
    for k, val in diagnosis.items():
        log(f"  {k} = {val}")

    # 选主要根因 (按优先级)
    if prefix_collapsed:
        root_cause = "prefix_collapsed"
        fix = "normalization/初始化/输出方差约束"
    elif prefix_ignored:
        root_cause = "prefix_ignored_by_qwen"
        fix = "调整注入层 (input→mid), 或增加 prefix tokens (8→16/32)"
    elif dpo_target_wrong:
        root_cause = "dpo_target_wrong_alignment"
        fix = "构造用户条件反事实偏好对"
    elif profile_prefix_decoupled:
        root_cause = "profile_prefix_decoupled"
        fix = "profile→prefix 通路无训练信号;增加正则或换 DPO 目标"
    else:
        root_cause = "unknown"
        fix = "需更细诊断 (按 user 拆开看)"

    log(f"  → 推测根因: {root_cause}")
    log(f"  → 修复方向: {fix}")

    # === 13. 写 meta ===
    meta = {
        "n_pairs": N_PAIRS_TEST,
        "user_dim": USER_DIM,
        "num_tokens": NUM_TOKENS,
        "diagnostic_1_prefix_stats": p_stats,
        "diagnostic_2_profile_prefix_correlation": {
            "pearson_r": float(corr),
            "p_value": float(p_corr),
            "n_pairs": len(profile_dists),
        },
        "diagnostic_3_qwen_logit_hidden_diff": {
            "logit_diff": logit_diffs,
            "hidden_diff": hidden_diffs,
        },
        "diagnostic_4_M_per_profile": {
            "target": {"mean": float(np.mean(M_target_arr)), "std": float(np.std(M_target_arr))},
            "random": {"mean": float(np.mean(M_random_arr)), "std": float(np.std(M_random_arr))},
            "zero":   {"mean": float(np.mean(M_zero_arr)),   "std": float(np.std(M_zero_arr))},
            "shuffled": {"mean": float(np.mean(M_shuffled_arr)), "std": float(np.std(M_shuffled_arr))},
        },
        "diagnostic_4_paired_diffs": {
            "target_minus_random": {
                "mean": float(diff_target_random.mean()),
                "std": float(diff_target_random.std()),
                "sign_positive_count": int(pos_vs_random),
                "sign_positive_ratio": float(pos_vs_random / len(diff_target_random)),
            },
            "target_minus_zero": {
                "mean": float(diff_target_zero.mean()),
                "std": float(diff_target_zero.std()),
                "sign_positive_count": int(pos_vs_zero),
                "sign_positive_ratio": float(pos_vs_zero / len(diff_target_zero)),
            },
            "target_minus_shuffled": {
                "mean": float(diff_target_shuffled.mean()),
                "std": float(diff_target_shuffled.std()),
                "sign_positive_count": int(pos_vs_shuf),
                "sign_positive_ratio": float(pos_vs_shuf / len(diff_target_shuffled)),
            },
        },
        "diagnostic_5_M_target_approx_M_shuffled": bool(
            abs(diff_target_shuffled.mean()) < 0.05
        ),
        "diagnosis": diagnosis,
        "root_cause_guess": root_cause,
        "fix_direction": fix,
    }
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"已写 {META_OUT}")
    log("=" * 70)


if __name__ == "__main__":
    main()