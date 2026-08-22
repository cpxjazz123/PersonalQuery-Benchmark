#!/usr/bin/env python3
"""Phase 10.9.2: Profile-Preserving Counterfactual DPO + Cycle + Geometry + Variance.

训练目标 (用户指定):
  L = L_counterfactual_DPO + λ1·L_cycle + λ2·L_geometry + λ3·L_var

  - L_counterfactual_DPO: 同时满足 M(profile_u→q_u>q_v) > 0 AND M(profile_v→q_v>q_u) > 0
  - L_cycle: ||Decoder(Prefix(z_u)) - z_u||² (重建约束)
  - L_geometry: |cos(z_u,z_v) - cos(prefix_u,prefix_v)| (几何保留)
  - L_var: max(0, τ - std(prefix)) (避免 collapse)

设计参数 (用户指定):
  - K=8 tokens (不变)
  - hidden=128 (不变)
  - gate_init 改温和非零 (0.1), 仍 learnable
  - 60 dev user (data frozen)
  - β=0.5 (同 10.6)
"""
from __future__ import annotations

import json
import math
import os
import random
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
PAIRS_FILE = OUT_DIR / "phase10_9_counterfactual_pairs.jsonl"
PROJ_OUT_CKPT = OUT_DIR / "phase10_9_2_projector.pt"
LOG_OUT = OUT_DIR / "phase10_9_2_train.log"
META_OUT = OUT_DIR / "phase10_9_2_train_meta.json"

# === 硬编码 ===
USER_DIM = 40
HIDDEN_DIM = 128
NUM_TOKENS = 8
DTYPE = torch.bfloat16
DEVICE = "cuda:0"

# DPO + 约束
BETA = 0.5
LAMBDA_CYCLE = 1.0
LAMBDA_GEOMETRY = 1.0
LAMBDA_VAR = 0.5
VAR_THRESHOLD = 0.02  # prefix token embedding std 阈值
LEARNING_RATE = 1e-3
EPOCHS = 60
MICRO_BATCH = 2  # pair 数 / micro step
GRAD_ACCUM = 4  # micro step 数 / optimizer step
SEED = 42
GATE_INIT = 0.1  # 温和非零, 替代 1e-3

MAX_PROMPT_LEN = 480
MAX_QUERY_LEN = 64

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)


def log(m):
    print(f"[phase10-9.2] {m}", flush=True)


class PrefixDecoder(nn.Module):
    """Prefix (K*H) → profile (user_dim=40) 重建."""
    def __init__(self, num_tokens: int, model_dim: int, user_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_tokens * model_dim, user_dim * 2),  # K*H → 80
            nn.GELU(),
            nn.Linear(user_dim * 2, user_dim),  # 80 → 40
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        # prefix: [B, K*H]
        return self.fc(prefix)


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
    for b, (p_ids, q_ids) in enumerate(zip(prompt_ids_list, query_ids_list)):
        full_ids = (p_ids + q_ids)[:full_len]
        L = len(full_ids)
        input_ids[b, :L] = torch.tensor(full_ids, dtype=torch.long, device=device)
        attention_mask[b, :L] = 1
        p_len = min(len(p_ids), L)
        query_mask[b, p_len:L] = 1
    return input_ids, attention_mask, query_mask


def compute_logp(model, input_embeds, attention_mask, query_mask, target_ids, num_prefix_tokens: int):
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


def main():
    log("=" * 70)
    log("Phase 10.9.2: Profile-Preserving Counterfactual DPO + Cycle + Geometry + Variance")
    log("=" * 70)
    log(f"  K={NUM_TOKENS}, hidden={HIDDEN_DIM}, gate_init={GATE_INIT}")
    log(f"  β={BETA}, λ1={LAMBDA_CYCLE}, λ2={LAMBDA_GEOMETRY}, λ3={LAMBDA_VAR}, τ={VAR_THRESHOLD}")
    log(f"  LR={LEARNING_RATE}, EPOCHS={EPOCHS}")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # === 1. 加载 counterfactual pairs ===
    log("[1] Loading counterfactual pairs ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  {len(pairs)} pairs ({len(pairs)*2} DPO samples)")

    # === 2. 加载 Qwen ===
    log("[2] Loading Qwen2-7B (frozen) ...")
    from llm_client import _HiddenBackend
    try:
        _HiddenBackend.reset()
    except Exception:
        pass
    import gc
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
    log(f"  GPU mem after Qwen load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # === 3. 加载 projector (init for both train + ref) ===
    log("[3] Initializing projector + ref + decoder ...")
    from query.soft_prefix.projector import SoftPrefixProjector
    torch.manual_seed(SEED)
    projector = SoftPrefixProjector(
        user_dim=USER_DIM, hidden_dim=HIDDEN_DIM, num_tokens=NUM_TOKENS,
        model_dim=model_dim, dtype=torch.float32, gate_init=GATE_INIT,
    ).to(DEVICE)
    ref_projector = SoftPrefixProjector(
        user_dim=USER_DIM, hidden_dim=HIDDEN_DIM, num_tokens=NUM_TOKENS,
        model_dim=model_dim, dtype=torch.float32, gate_init=GATE_INIT,
    ).to(DEVICE)
    ref_projector.load_state_dict(projector.state_dict())
    for p in ref_projector.parameters():
        p.requires_grad = False
    ref_projector.eval()
    decoder = PrefixDecoder(NUM_TOKENS, model_dim, USER_DIM).to(DEVICE)
    log(f"  projector params: {projector.num_parameters(trainable_only=True)}")
    log(f"  decoder params: {sum(p.numel() for p in decoder.parameters())}")

    # 验证 init 一致
    with torch.no_grad():
        z_test = torch.randn(2, USER_DIM, device=DEVICE)
        p1 = projector(z_test); p2 = ref_projector(z_test)
        assert torch.allclose(p1, p2, atol=1e-6), "init 不一致!"
    log(f"  ✓ train/ref projector init 一致")

    optim = torch.optim.AdamW(
        list(projector.parameters()) + list(decoder.parameters()),
        lr=LEARNING_RATE, weight_decay=0.0,
    )

    # === 4. 准备 batch 数据 (固定, 全 epoch 用同一批) ===
    log("[4] Preparing batch data ...")
    SYSTEM = (
        "You are a shopping query writer. Write one short natural shopping query "
        "that mentions every listed attribute of the product by its exact value "
        "(brand name, weight, dimensions, color, material). Keep the query under 25 words."
    )
    # 每个 pair → 4 samples: (profile_u, chosen_u), (profile_u, rejected_u),
    #                          (profile_v, chosen_v), (profile_v, rejected_v)
    z_us = torch.tensor([p["z_u"] for p in pairs], dtype=torch.float32, device=DEVICE)
    z_vs = torch.tensor([p["z_v"] for p in pairs], dtype=torch.float32, device=DEVICE)
    prompts = []
    queries = []
    for p in pairs:
        cu = build_chat_prompts(tokenizer, SYSTEM, p["prompt_u"])
        cv = build_chat_prompts(tokenizer, SYSTEM, p["prompt_v"])
        # profile_u: chosen_u, rejected_u (= chosen_v)
        prompts.extend([cu, cu, cv, cv])
        queries.extend([p["chosen_q_u"], p["chosen_q_v"], p["chosen_q_v"], p["chosen_q_u"]])
    input_ids, attention_mask, query_mask = encode_inputs(tokenizer, prompts, queries, DEVICE)
    log(f"  batch: {len(pairs)} pairs × 4 samples = {len(pairs)*4} forward samples")
    log(f"  input_ids shape: {input_ids.shape}, query_mask sum (first 4): {query_mask.sum(dim=1).tolist()[:4]}")

    # === 5. 训练循环 ===
    log("[5] Training ...")
    history = []
    best_dpo = float("inf")
    n_pairs = len(pairs)
    n_micro = (n_pairs + MICRO_BATCH - 1) // MICRO_BATCH

    @torch.no_grad()
    def eval_dpo():
        """评估当前 DPO: dpo_A + dpo_B (应都 > 0). 用 micro-batch 避免 OOM."""
        projector.eval(); decoder.eval()
        # 构造 z 序列: per pair 4 samples (chosen_u, rejected_u, chosen_v, rejected_v)
        z_eval = []
        for i in range(n_pairs):
            z_eval.extend([z_us[i], z_us[i], z_vs[i], z_vs[i]])
        z_eval = torch.stack(z_eval)  # [4*N, 40]
        prefix_train = projector(z_eval).to(DTYPE)  # [4*N, K, H]
        prefix_ref = ref_projector(z_eval).to(DTYPE)
        # cycle loss 用 chosen 对应的 prefix
        prefix_chosen_flat = prefix_train.view(n_pairs, 2, 2, NUM_TOKENS, model_dim)[
            :, :, 0, :, :
        ].reshape(2 * n_pairs, NUM_TOKENS * model_dim).float()
        z_chosen_target = torch.stack([z_us, z_vs], dim=1).reshape(-1, USER_DIM)
        z_hat = decoder(prefix_chosen_flat)
        cycle = ((z_hat - z_chosen_target) ** 2).mean().item()
        prefix_std = prefix_train.float().std(dim=0).mean().item()

        emb_m = model.get_input_embeddings()

        # === micro-batch forward ===
        EVAL_MICRO = 8  # 2 pairs per micro (8 samples)

        def lp_with(prefix):
            n_total = prefix.size(0)
            lp_all = []
            for s in range(0, n_total, EVAL_MICRO):
                e = min(s + EVAL_MICRO, n_total)
                pre_sub = prefix[s:e]
                ids_sub = input_ids[s:e]
                am_sub = attention_mask[s:e]
                qm_sub = query_mask[s:e]
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

        lp_train = lp_with(prefix_train).view(n_pairs, 2, 2)
        lp_ref = lp_with(prefix_ref).view(n_pairs, 2, 2)
        dpo_A = -(F.logsigmoid(BETA * (
            (lp_train[:, 0, 0] - lp_train[:, 0, 1]) - (lp_ref[:, 0, 0] - lp_ref[:, 0, 1])
        ))).mean().item()
        dpo_B = -(F.logsigmoid(BETA * (
            (lp_train[:, 1, 0] - lp_train[:, 1, 1]) - (lp_ref[:, 1, 0] - lp_ref[:, 1, 1])
        ))).mean().item()
        acc_u = (lp_train[:, 0, 0] > lp_train[:, 0, 1]).float().mean().item()
        acc_v = (lp_train[:, 1, 0] > lp_train[:, 1, 1]).float().mean().item()
        return dpo_A, dpo_B, acc_u, acc_v, cycle, prefix_std

    pre_a, pre_b, pre_acc_u, pre_acc_v, pre_cycle, pre_std = eval_dpo()
    log(f"  PRE: dpo_A={pre_a:.4f}, dpo_B={pre_b:.4f}, "
        f"acc_u={pre_acc_u:.3f}, acc_v={pre_acc_v:.3f}, cycle={pre_cycle:.4f}, prefix_std={pre_std:.6f}")

    for epoch in range(EPOCHS):
        projector.train(); decoder.train()
        losses = {"dpo": [], "cycle": [], "geom": [], "var": []}
        grad_norms = []
        perm = np.random.permutation(n_pairs)
        for mi in range(n_micro):
            s, e = mi * MICRO_BATCH, min((mi + 1) * MICRO_BATCH, n_pairs)
            idx = perm[s:e]
            idx_t = torch.tensor(idx, device=DEVICE)
            zu_sub = z_us[idx_t]
            zv_sub = z_vs[idx_t]

            # 取对应的 4*micro samples
            rows = []
            for i in idx:
                base = i * 4
                rows.extend([base + 0, base + 1, base + 2, base + 3])
            rows_t = torch.tensor(rows, device=DEVICE, dtype=torch.long)
            ids_sub = input_ids[rows_t]
            am_sub = attention_mask[rows_t]
            qm_sub = query_mask[rows_t]
            # z 顺序: [zu, zu, zv, zv] per pair
            z_sub = torch.stack([
                zu_sub, zu_sub, zv_sub, zv_sub,
            ], dim=1).reshape(-1, USER_DIM)  # [4*micro, 40]

            # === Train forward ===
            prefix_train = projector(z_sub).to(DTYPE)
            emb_m = model.get_input_embeddings()
            text_emb = emb_m(ids_sub).to(DTYPE)
            full_emb = torch.cat([prefix_train, text_emb], dim=1)
            full_am = torch.cat([
                torch.ones((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
                am_sub,
            ], dim=1)
            full_qm = torch.cat([
                torch.zeros((full_emb.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
                qm_sub,
            ], dim=1)
            lp_train = compute_logp(model, full_emb, full_am, full_qm, ids_sub, NUM_TOKENS)
            lp_train_p = lp_train.view(-1, 2, 2)  # [micro, sample, chosen/rej]

            # === Ref forward (no grad) ===
            with torch.no_grad():
                prefix_ref = ref_projector(z_sub).to(DTYPE)
                full_emb_r = torch.cat([prefix_ref, text_emb], dim=1)
                full_am_r = torch.cat([
                    torch.ones((full_emb_r.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
                    am_sub,
                ], dim=1)
                full_qm_r = torch.cat([
                    torch.zeros((full_emb_r.size(0), NUM_TOKENS), dtype=torch.long, device=DEVICE),
                    qm_sub,
                ], dim=1)
                lp_ref = compute_logp(model, full_emb_r, full_am_r, full_qm_r, ids_sub, NUM_TOKENS)
                lp_ref_p = lp_ref.view(-1, 2, 2)

            # === DPO loss (counterfactual) ===
            diff_A = (lp_train_p[:, 0, 0] - lp_train_p[:, 0, 1]) - (lp_ref_p[:, 0, 0] - lp_ref_p[:, 0, 1])
            diff_B = (lp_train_p[:, 1, 0] - lp_train_p[:, 1, 1]) - (lp_ref_p[:, 1, 0] - lp_ref_p[:, 1, 1])
            dpo_loss = -(F.logsigmoid(BETA * diff_A).mean() + F.logsigmoid(BETA * diff_B).mean())

            # === Cycle loss (用 chosen 对应的 prefix) ===
            # chosen samples: idx 0 (chosen_u under profile_u) + idx 2 (chosen_v under profile_v)
            prefix_chosen = prefix_train.view(-1, 2, 2, NUM_TOKENS, model_dim)[
                :, :, 0, :, :
            ].reshape(-1, NUM_TOKENS * model_dim).float()  # [2*micro, K*H]
            z_chosen_target = torch.stack([zu_sub, zv_sub], dim=1).reshape(-1, USER_DIM)  # [2*micro, 40]
            z_hat = decoder(prefix_chosen)
            cycle_loss = ((z_hat - z_chosen_target) ** 2).mean()

            # === Geometry loss (batch 内 (u, v) 对的 cos 距离) ===
            prefix_u_flat = prefix_train.view(-1, 2, 2, NUM_TOKENS, model_dim)[
                :, 0, 0, :, :
            ].reshape(-1, NUM_TOKENS * model_dim).float()  # [micro, K*H]
            prefix_v_flat = prefix_train.view(-1, 2, 2, NUM_TOKENS, model_dim)[
                :, 1, 0, :, :
            ].reshape(-1, NUM_TOKENS * model_dim).float()  # [micro, K*H]
            cos_z = F.cosine_similarity(zu_sub, zv_sub, dim=-1)  # [micro]
            cos_p = F.cosine_similarity(prefix_u_flat, prefix_v_flat, dim=-1)  # [micro]
            geom_loss = (cos_z - cos_p).abs().mean()

            # === Variance loss ===
            prefix_u_std = prefix_u_flat.std(dim=0).mean()
            prefix_v_std = prefix_v_flat.std(dim=0).mean()
            var_loss = F.relu(VAR_THRESHOLD - prefix_u_std) + F.relu(VAR_THRESHOLD - prefix_v_std)

            total = dpo_loss + LAMBDA_CYCLE * cycle_loss + LAMBDA_GEOMETRY * geom_loss + LAMBDA_VAR * var_loss
            (total / GRAD_ACCUM).backward()
            losses["dpo"].append(dpo_loss.item())
            losses["cycle"].append(cycle_loss.item())
            losses["geom"].append(geom_loss.item())
            losses["var"].append(var_loss.item())
            g = sum(p.grad.norm().item() ** 2 for p in projector.parameters() if p.grad is not None) ** 0.5
            grad_norms.append(g)

            if (mi + 1) % GRAD_ACCUM == 0 or mi == n_micro - 1:
                torch.nn.utils.clip_grad_norm_(
                    list(projector.parameters()) + list(decoder.parameters()), 1.0,
                )
                optim.step()
                optim.zero_grad()

            del full_emb, full_am, full_qm, full_emb_r, full_am_r, full_qm_r
            torch.cuda.empty_cache()

        # === eval after epoch ===
        dpo_a, dpo_b, acc_u, acc_v, cyc, std = eval_dpo()
        dpo_total = dpo_a + dpo_b
        if dpo_total < best_dpo:
            best_dpo = dpo_total
            # 保存 ckpt
            torch.save({
                "state_dict": projector.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "user_dim": USER_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_tokens": NUM_TOKENS,
                "model_dim": model_dim,
                "gate_init": GATE_INIT,
                "epoch": epoch + 1,
                "dpo_a": dpo_a,
                "dpo_b": dpo_b,
            }, PROJ_OUT_CKPT)
        if (epoch + 1) % 5 == 0 or epoch < 2:
            log(f"  [epoch {epoch+1:3d}/{EPOCHS}] dpo_A={dpo_a:.4f} dpo_B={dpo_b:.4f} "
                f"acc_u={acc_u:.3f} acc_v={acc_v:.3f} "
                f"cycle={cyc:.4f} prefix_std={std:.5f}")

        history.append({
            "epoch": epoch + 1,
            "dpo_A": dpo_a, "dpo_B": dpo_b,
            "acc_u": acc_u, "acc_v": acc_v,
            "cycle": cyc, "prefix_std": std,
            "loss_dpo": float(np.mean(losses["dpo"])),
            "loss_cycle": float(np.mean(losses["cycle"])),
            "loss_geom": float(np.mean(losses["geom"])),
            "loss_var": float(np.mean(losses["var"])),
            "grad_norm": float(np.mean(grad_norms)),
        })

    # === 6. Final ===
    log("=" * 70)
    log("Final (best ckpt saved):")
    log(f"  PRE: acc_u={pre_acc_u:.3f}, acc_v={pre_acc_v:.3f}, cycle={pre_cycle:.4f}")
    final = eval_dpo()
    log(f"  POST: dpo_A={final[0]:.4f}, dpo_B={final[1]:.4f}, "
        f"acc_u={final[2]:.3f}, acc_v={final[3]:.3f}, cycle={final[4]:.4f}, prefix_std={final[5]:.5f}")
    log(f"  counterfactual accuracy (avg u+v): {(final[2] + final[3]) / 2:.3f}")
    log(f"  ckpt → {PROJ_OUT_CKPT}")

    meta = {
        "n_pairs": len(pairs),
        "hyperparams": {
            "beta": BETA, "lambda_cycle": LAMBDA_CYCLE, "lambda_geom": LAMBDA_GEOMETRY,
            "lambda_var": LAMBDA_VAR, "var_threshold": VAR_THRESHOLD,
            "lr": LEARNING_RATE, "epochs": EPOCHS, "gate_init": GATE_INIT,
            "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
        },
        "pre": {
            "dpo_A": pre_a, "dpo_B": pre_b,
            "acc_u": pre_acc_u, "acc_v": pre_acc_v,
            "cycle": pre_cycle, "prefix_std": pre_std,
        },
        "post": {
            "dpo_A": final[0], "dpo_B": final[1],
            "acc_u": final[2], "acc_v": final[3],
            "cycle": final[4], "prefix_std": final[5],
            "avg_acc": (final[2] + final[3]) / 2,
        },
        "history": history,
    }
    META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {META_OUT}")
    log("=" * 70)


if __name__ == "__main__":
    main()