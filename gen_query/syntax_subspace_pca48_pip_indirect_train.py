#!/usr/bin/env python3
"""Phase 6.C.1 — PIP-Indirect-inspired PCA48 Alignment training.

用户指令 2026-08-30: 在 Phase 6.A (+0.251 ρ conditioning) 基础上加
L_align + L_margin 强迫 prefix 表达 PCA48 而不只是参与 LM loss。

参考 PIP (ACL Findings 2023) PIP-Indirect 变体:
- Parse Encoding Loss 强迫 prefix-modified representation 保留 syntax encoding
- + wrong-condition margin 防止 shortcut

改造到 PCA48:
  z_t → PCA48Projector → Prefix → Qwen → r(z_t) [mean pool hidden states]
                                                    ↓
                                              H (small head)
                                                    ↓
                                              ẑ ∈ R^48

Loss = L_LM + λ_cos · L_cos + λ_dist · L_dist + λ_margin · L_margin
where:
  L_cos = 1 - cos(ẑ_correct, z_correct)
  L_dist = ||ẑ_correct - z_correct||²
  L_margin = max(0, m + ||ẑ_correct - z_correct||² - ||ẑ_wrong - z_correct||²)

Training: 在 Phase 6.A init ckpt (projector_final.pt) 基础上 finetune,
不破坏已有 +0.251 ρ conditioning。

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
DATA_DIR = SCRATCH / "pca48_dataset"
CKPT_DIR = SCRATCH / "pca48_pip_indirect_ckpt"
LOG_DIR = SCRATCH / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# === Train config (SMOKE per Rule 18) ===
PHASE6A_CKPT = SCRATCH / "pca48_ckpt" / "projector_final.pt"
N_EPOCHS = 1
BATCH_SIZE = 2
LR = 2e-5
MAX_STEPS = 200         # smoke (Rule 18): ultra-smoke 10 OK → 200 steps (vs Phase 6.B.4 750)
LAMBDA_COS = 0.5
LAMBDA_DIST = 0.1
LAMBDA_MARGIN = 0.5
MARGIN = 1.0
SEED = 42
LOG_EVERY = 25


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pip_train] {msg}", flush=True)


class PCA48Reconstructor(nn.Module):
    """H: Qwen hidden state (3584) → ẑ (48)."""

    def __init__(self, hidden_size: int = 3584, out_dim: int = 48):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


def mean_pool_target_hidden(hidden_states: tuple, prefix_len: int,
                            prompt_len: int, target_attn: torch.Tensor) -> torch.Tensor:
    """从所有 layer 的 hidden states 中抽取 target token 区域并 mean pool.

    Args:
        hidden_states: tuple of (num_layers+1, B, T, H), 由 output_hidden_states=True 返回
        prefix_len: 8 (固定)
        prompt_len: prompt token 长度(per batch)
        target_attn: (B, T_target) attention mask, 1 for real tokens, 0 for pad

    Returns:
        (B, H) mean-pooled hidden state across all layers and target tokens
    """
    # 取所有 transformer layer 的 output (跳过 embedding layer = hidden_states[0])
    # hidden_states[i]: (B, T, H) for i in [1, num_layers]
    num_layers = len(hidden_states) - 1
    layer_hs = torch.stack(hidden_states[1:], dim=0)  # (num_layers, B, T, H)

    # 提取 target region: [prefix_len + prompt_len : end]
    # 但 prompt_len 是 per-batch 的(因 padding), 需要从 target_attn 推断
    # target_attn 是 (B, T_target), 把它 pad 到 full length
    B = layer_hs.size(1)
    T_full = layer_hs.size(2)
    # T_target = T_full - prefix_len - max_prompt_len
    # 简化:假设所有 sample prompt_len 一致(因 padding 到相同长度)
    # 取 T_full - target_attn.size(1) 作为 prefix_len + prompt_len
    pre_target_len = T_full - target_attn.size(1)

    target_hs = layer_hs[:, :, pre_target_len:, :]  # (num_layers, B, T_target, H)

    # Build target mask: (1, B, T_target, 1) broadcast
    target_mask = target_attn.to(layer_hs.dtype).unsqueeze(0).unsqueeze(-1)  # (1, B, T_target, 1)
    target_mask = target_mask.expand(num_layers, B, target_attn.size(1), 1)

    # Apply mask
    masked = target_hs * target_mask
    denom = target_mask.sum(dim=2, keepdim=True).clamp(min=1e-6)  # (num_layers, B, 1, 1)
    mean_per_layer = masked.sum(dim=2) / denom.squeeze(-1)  # (num_layers, B, H)

    # Mean across layers
    r = mean_per_layer.mean(dim=0)  # (B, H)
    return r


def main():
    log("=== Phase 6.C.1 — PIP-Indirect-inspired PCA48 Alignment (SMOKE) ===")

    torch.manual_seed(SEED)

    # Load dataset
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    log(f"  loaded {len(samples)} training pairs")

    # Load model
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix

    log("  loading Qwen2-7B + LoRA + projector (from Phase 6.A ckpt) ...")
    model = Qwen2WithPrefix()
    model.load_projector(str(PHASE6A_CKPT))
    log(f"  loaded Phase 6.A ckpt: {PHASE6A_CKPT}")

    # Add H head
    H_head = PCA48Reconstructor(hidden_size=model.hidden_size, out_dim=48).to(
        "cuda", dtype=model.dtype
    )
    n_trainable_H = sum(p.numel() for p in H_head.parameters())
    log(f"  H head: {n_trainable_H:,} params (Linear(3584→256) + Linear(256→48))")

    # Optimizer
    params = list(model.trainable_parameters()) + list(H_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR)

    # Pad tokenizer
    model.tokenizer.padding_side = "right"

    # Loss tracking
    losses = {"total": [], "lm": [], "cos": [], "dist": [], "margin": []}
    n_skipped = 0
    t_start = time.time()

    log(f"  training: {MAX_STEPS} steps × {BATCH_SIZE} pairs × 2 forwards (correct + wrong)")
    log(f"  loss = L_LM + {LAMBDA_COS}·L_cos + {LAMBDA_DIST}·L_dist + {LAMBDA_MARGIN}·L_margin (m={MARGIN})")

    rng = np.random.default_rng(SEED)
    idx_pool = rng.integers(0, len(samples), size=MAX_STEPS * BATCH_SIZE * 2)

    pool_ptr = 0
    for step in range(MAX_STEPS):
        # Sample BATCH_SIZE pairs
        batch_indices = idx_pool[pool_ptr: pool_ptr + BATCH_SIZE]
        pool_ptr += BATCH_SIZE
        if pool_ptr + BATCH_SIZE > len(idx_pool):
            pool_ptr = 0

        batch = [samples[int(i)] for i in batch_indices]

        z_correct = torch.tensor(
            np.stack([np.asarray(s["z_48"], dtype=np.float32) for s in batch]),
            dtype=torch.float32, device="cuda",
        )  # (B, 48)

        # z_wrong = cyclic shift in batch
        z_wrong = torch.roll(z_correct, shifts=1, dims=0)
        # Add tiny noise to make wrong ≠ correct even for duplicates
        z_wrong = z_wrong + torch.randn_like(z_wrong) * 0.01

        attrs_list = [s["c_i"] for s in batch]
        target_texts = [s["s_i"] for s in batch]

        # ---- Forward pass 1: z_correct (L_LM + L_align) ----
        z_d = z_correct.to(model.dtype)
        prefix_embeds = model.projector(z_d)
        prompt_enc = model.tokenizer(
            [model._build_prompt([a])[0] for a in attrs_list],
            return_tensors="pt", padding=True, add_special_tokens=False,
        ).to("cuda")
        target_enc = model.tokenizer(
            target_texts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to("cuda")
        prompt_ids = prompt_enc.input_ids
        prompt_attn = prompt_enc.attention_mask
        target_ids = target_enc.input_ids
        target_attn = target_enc.attention_mask

        embed_layer = model.model.get_input_embeddings()
        prompt_embeds = embed_layer(prompt_ids).to(model.dtype)
        target_embeds = embed_layer(target_ids).to(model.dtype)

        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)
        B = z_correct.size(0)
        prefix_attn = torch.ones((B, model.prefix_len), dtype=torch.long, device="cuda")
        attention_mask = torch.cat([prefix_attn, prompt_attn, target_attn], dim=1)

        # Labels: -100 for prefix & prompt, target_ids (with pad masked)
        prefix_labels = torch.full((B, model.prefix_len), -100, dtype=torch.long, device="cuda")
        prompt_labels = torch.full_like(prompt_ids, -100)
        target_labels = target_ids.clone()
        target_labels[target_attn == 0] = -100
        labels = torch.cat([prefix_labels, prompt_labels, target_labels], dim=1)

        out = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
        )
        loss_lm = out.loss
        # Extract r_correct: mean pool target hidden states across all layers
        r_correct = mean_pool_target_hidden(
            out.hidden_states, model.prefix_len, prompt_ids.size(1), target_attn
        )
        z_hat_correct = H_head(r_correct.to(model.dtype))  # (B, 48)

        # ---- Forward pass 2: z_wrong (only need hidden states for L_margin) ----
        z_dw = z_wrong.to(model.dtype)
        prefix_embeds_w = model.projector(z_dw)
        inputs_embeds_w = torch.cat([prefix_embeds_w, prompt_embeds, target_embeds], dim=1)
        attention_mask_w = torch.cat([prefix_attn, prompt_attn, target_attn], dim=1)
        labels_w = torch.cat([prefix_labels, prompt_labels, target_labels], dim=1)

        out_w = model.model(
            inputs_embeds=inputs_embeds_w,
            attention_mask=attention_mask_w,
            labels=labels_w,
            output_hidden_states=True,
            use_cache=False,
        )
        # NOTE: don't use out_w.loss (we already have loss_lm from correct z)
        r_wrong = mean_pool_target_hidden(
            out_w.hidden_states, model.prefix_len, prompt_ids.size(1), target_attn
        )
        z_hat_wrong = H_head(r_wrong.to(model.dtype))

        # ---- Compute losses ----
        # Convert z to model dtype
        z_target = z_correct.to(model.dtype)
        z_target_w = z_correct.to(model.dtype)  # compare against z_correct, not z_wrong

        d_correct = ((z_hat_correct - z_target) ** 2).sum(dim=-1)  # (B,)
        d_wrong = ((z_hat_wrong - z_target_w) ** 2).sum(dim=-1)    # (B,)
        loss_dist = d_correct.mean()
        loss_cos = (1 - F.cosine_similarity(z_hat_correct, z_target, dim=-1)).mean()
        loss_margin = torch.clamp(MARGIN + d_correct - d_wrong, min=0).mean()

        loss_total = (
            loss_lm
            + LAMBDA_COS * loss_cos
            + LAMBDA_DIST * loss_dist
            + LAMBDA_MARGIN * loss_margin
        )

        # Backward
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        # Track
        losses["total"].append(float(loss_total.item()))
        losses["lm"].append(float(loss_lm.item()))
        losses["cos"].append(float(loss_cos.item()))
        losses["dist"].append(float(loss_dist.item()))
        losses["margin"].append(float(loss_margin.item()))

        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            avg = {k: np.mean(v[-LOG_EVERY:]) for k, v in losses.items()}
            log(f"  step {step+1}/{MAX_STEPS} elapsed={elapsed:.0f}s  "
                f"loss={avg['total']:.3f} (LM={avg['lm']:.3f} cos={avg['cos']:.3f} "
                f"dist={avg['dist']:.3f} margin={avg['margin']:.3f})")

    # Save final ckpt
    final_path = CKPT_DIR / "pip_indirect_final.pt"
    state = {
        "projector": model.projector.state_dict(),
        "h_head": H_head.state_dict(),
        "lora_state_dict": {
            k: v for k, v in model.model.state_dict().items() if "lora_" in k
        },
    }
    torch.save(state, final_path)
    log(f"  saved final ckpt → {final_path}")

    # Save losses
    with open(LOG_DIR / "phase6c1_pip_losses.json", "w") as f:
        json.dump(losses, f, indent=2)
    log(f"  saved losses → {LOG_DIR / 'phase6c1_pip_losses.json'}")

    # Summary
    log(f"\n=== TRAIN SUMMARY ===")
    log(f"  total steps: {MAX_STEPS}  total time: {time.time() - t_start:.0f}s")
    log(f"  final loss avg (last 25):")
    for k, v in losses.items():
        log(f"    {k}: {np.mean(v[-LOG_EVERY:]):.4f}")


if __name__ == "__main__":
    main()
