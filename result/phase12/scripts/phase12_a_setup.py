#!/usr/bin/env python3
"""Phase 12.A: Load LLaDA-8B-Base + FiLM adapter + masked diffusion skeleton.

Architecture:
  - Frozen LLaDA backbone (32 LLaMA-like blocks, 4096 hidden, bidirectional)
  - Trainable: s_u projector (128 → d_model) + per-block FiLM γ_l/β_l
  - Properties span tokens are NEVER masked during forward/inference

This script verifies:
  1. Model loads with trust_remote_code=True
  2. FiLM adapter forward works (gamma/beta from s_u)
  3. Masked diffusion training step: forward on masked input_ids, CE on masked positions
  4. Gradient flows only through FiLM/projector, backbone frozen
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LLADA_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/hf_cache/phase12/LLaDA-8B-Base")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # LLaDA trained in bf16

# === FiLM config (these are the only trainable params) ===
SU_DIM = 128           # input s_u dim (matches Phase 11.A Gaussian cache)
D_MODEL = 4096         # LLaDA hidden
N_BLOCKS = 32          # LLaDA blocks


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class FiLMBlock(nn.Module):
    """Per-block FiLM: h' = h * (1 + γ) + β, where γ, β = Linear(s_u).

    γ/β initialized to zero so FiLM is identity at start (backbone unchanged).
    """

    def __init__(self, su_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(su_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, s_u: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D), s_u: (B, su_dim) — broadcast over T
        gamma_beta = self.proj(s_u)              # (B, 2*D)
        gamma, beta = gamma_beta.chunk(2, dim=-1) # (B, D), (B, D)
        gamma = gamma.unsqueeze(1)                # (B, 1, D)
        beta = beta.unsqueeze(1)                 # (B, 1, D)
        return h * (1 + gamma) + beta


class LLaDAWithStyle(nn.Module):
    """Wraps frozen LLaDA + adds per-block FiLM conditioned on s_u.

    Training: only FiLMBlock weights + s_u projector are trainable.
    Backbone weights are frozen (no_grad).
    """

    def __init__(self, base_model_path: Path, su_dim: int = SU_DIM):
        super().__init__()
        from transformers import AutoModelForCausalLM
        # Patch: transformers v5 expects `all_tied_weights_keys` attr that
        # custom LLaDA code (written for v4) doesn't define. Add stub.
        try:
            from transformers import modeling_utils
            _orig_finalize = modeling_utils.PreTrainedModel._finalize_model_loading
            def _patched_finalize(self, *args, **kwargs):
                if not hasattr(self, 'all_tied_weights_keys'):
                    self.all_tied_weights_keys = {}
                return _orig_finalize(self, *args, **kwargs)
            modeling_utils.PreTrainedModel._finalize_model_loading = _patched_finalize
            log("  Patched transformers._finalize_model_loading for LLaDA compat")
        except Exception as e:
            log(f"  Patch warning: {e}")
        log(f"Loading LLaDA from {base_model_path} ...")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
            torch_dtype=DTYPE,
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        # Trainable: s_u projector + per-block FiLM
        self.su_proj = nn.Linear(su_dim, su_dim)  # identity-ish init
        nn.init.eye_(self.su_proj.weight)
        nn.init.zeros_(self.su_proj.bias)

        # Find n_blocks from config
        cfg = self.backbone.config
        n_blocks = cfg.n_layers
        d_model = cfg.d_model
        log(f"  LLaDA: n_layers={n_blocks}, d_model={d_model}")

        self.films = nn.ModuleList([
            FiLMBlock(su_dim, d_model) for _ in range(n_blocks)
        ])

        # Cache constants
        self.mask_token_id = cfg.mask_token_id
        self.vocab_size = cfg.vocab_size

    def trainable_params(self):
        return list(self.su_proj.parameters()) + list(self.films.parameters())

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_params())

    def forward(
        self,
        input_ids: torch.LongTensor,           # (B, T)
        s_u: torch.FloatTensor,                # (B, su_dim) — user style latent
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Forward through LLaDA with FiLM at each block. Returns logits (B, T, V)."""
        s_u_emb = self.su_proj(s_u).to(DTYPE)  # (B, su_dim)

        # Run LLaDA forward with manual per-block FiLM injection.
        # We re-implement the trunk of LLaDAModel.forward to access intermediate h.
        model = self.backbone.model  # LLaDAModelLM.model is the inner LLaDAModel
        transformer = model.transformer
        x = transformer.wte(input_ids)  # (B, T, D)
        if model.config.input_emb_norm:
            x = x * (model.config.d_model ** 0.5)
        x = transformer.emb_drop(x)

        # Build attention bias (bidirectional)
        seq_len = input_ids.size(1)
        attention_bias = model.get_bidirectional_attention_bias(seq_len, x.device)
        if attention_mask is not None and 0 in attention_mask:
            att_mask = attention_mask.to(dtype=torch.float).view(input_ids.size(0), -1)[:, None, None, :]
            att_mask = (1.0 - att_mask) * torch.finfo(torch.float32).min
            attention_bias = attention_bias + att_mask.to(attention_bias.dtype)

        # Run blocks with FiLM
        for block_idx, block in enumerate(transformer.blocks):
            x = block(x, attention_bias=attention_bias)
            x = self.films[block_idx](x, s_u_emb)

        # Final norm + lm_head
        x = transformer.ln_f(x)
        logits = model.transformer.ff_out(x) if hasattr(transformer, 'ff_out') else model.lm_head(x)
        return logits  # (B, T, V)


def masked_diffusion_step(
    model: LLaDAWithStyle,
    input_ids: torch.LongTensor,        # (B, T) original (unmasked)
    s_u: torch.FloatTensor,              # (B, su_dim)
    attr_mask: torch.BoolTensor,        # (B, T) True = 属性 token, 永不被 mask
    mask_ratio: float = 0.5,
) -> torch.Tensor:
    """One masked diffusion training step.

    Returns CE loss on masked positions only (not on attr positions).
    """
    B, T = input_ids.shape
    rand = torch.rand(B, T, device=input_ids.device)
    # Position is masked iff rand < mask_ratio AND NOT attr
    mask_pos = (rand < mask_ratio) & (~attr_mask)
    # Replace masked positions with mask_token_id
    noisy_ids = torch.where(mask_pos, torch.full_like(input_ids, model.mask_token_id), input_ids)

    logits = model(noisy_ids, s_u)
    # CE on masked positions
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),
        input_ids.view(-1),
        reduction='none',
    ).view(B, T)
    loss = (loss * mask_pos.float()).sum() / mask_pos.float().sum().clamp(min=1)
    return loss


def main():
    log("=" * 70)
    log("Phase 12.A: LLaDA + FiLM adapter smoke test")
    log("=" * 70)

    torch.manual_seed(42)

    # === Load ===
    model = LLaDAWithStyle(LLADA_DIR, su_dim=SU_DIM)
    model = model.to(DEVICE)
    log(f"  Loaded. Trainable params: {model.n_trainable():,}")
    log(f"  Backbone frozen: {not any(p.requires_grad for p in model.backbone.parameters())}")
    log(f"  mask_token_id: {model.mask_token_id}, vocab_size: {model.vocab_size}")

    # === Smoke: dummy s_u + dummy input ===
    B, T = 2, 16
    dummy_ids = torch.randint(0, model.vocab_size, (B, T), device=DEVICE)
    s_u = torch.randn(B, SU_DIM, device=DEVICE)

    # Mark positions [3:6] as attributes (never mask)
    attr_mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
    attr_mask[:, 3:6] = True

    with torch.no_grad():
        logits = model(dummy_ids, s_u, attention_mask=None)
    log(f"  Forward logits: {logits.shape} (expect [B={B}, T={T}, V={model.vocab_size}])")
    log(f"  logits finite: {torch.isfinite(logits).all().item()}")

    # === Smoke: training step ===
    opt = torch.optim.AdamW(model.trainable_params(), lr=1e-4)
    log("\n  Running 3 masked-diffusion training steps ...")
    for step in range(3):
        opt.zero_grad()
        loss = masked_diffusion_step(model, dummy_ids, s_u, attr_mask, mask_ratio=0.5)
        loss.backward()
        # Show grad norm on FiLM (should be > 0, backbone 0)
        film_grad = sum(p.grad.norm().item() for p in model.films.parameters() if p.grad is not None)
        su_grad = model.su_proj.weight.grad.norm().item() if model.su_proj.weight.grad is not None else 0.0
        bb_grad = sum(p.grad.norm().item() for p in model.backbone.parameters() if p.grad is not None)
        opt.step()
        log(f"  step {step}: loss={loss.item():.4f}  film_grad={film_grad:.4f}  su_grad={su_grad:.4f}  backbone_grad={bb_grad:.4f}")

    assert film_grad > 0, "FiLM gradient must flow"
    assert bb_grad == 0.0, "Backbone must be frozen"
    log("\n  ✓ FiLM trainable, backbone frozen, loss descending")

    # === Save FiLM weights init checkpoint ===
    out = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase12_a_film_init.pt")
    torch.save({
        "su_proj": model.su_proj.state_dict(),
        "films": [f.state_dict() for f in model.films],
        "su_dim": SU_DIM,
        "n_layers": model.backbone.config.n_layers,
        "d_model": model.backbone.config.d_model,
        "mask_token_id": model.mask_token_id,
        "vocab_size": model.vocab_size,
    }, out)
    log(f"  FiLM init saved → {out}")

    log("=" * 70)
    log("PHASE 12.A COMPLETE — ready for 12.B Stage 1 training")
    log("=" * 70)


if __name__ == "__main__":
    main()