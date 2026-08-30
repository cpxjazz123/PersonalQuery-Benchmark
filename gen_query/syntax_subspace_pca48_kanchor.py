#!/usr/bin/env python3
"""Phase 6.C.2 — PCA48 K-anchor controller (用户 2026-08-30 设计).

最小改动版: 替换 Phase 6.A 的 PCA48Projector (MLP→8 prefix tokens) 为
K-anchor controller (32 anchor softmax 加权 → 1 control embedding).

架构:
  PCA48 z (B, 48)
    ↓  KMeans K=32 anchors (frozen, from train PCA48)
    ↓  weights = softmax(-||z - a_k||² / τ)
    ↓  control_emb = sum_k w_k · E_k (E ∈ R^{32×3584}, learnable)
  inputs_embeds[:, 0, :] = control_emb  (替换 BOS 位置 embedding)
    ↓
  Qwen2-7B (frozen) + LoRA r=16 on q/k/v/o

训练 loss: L = L_LM (Phase 6.A v1 方案, 不加 align / DPO / CC / mirror)

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
QWEN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# === K-anchor config (用户锁定) ===
PCA_DIM = 48
K_ANCHORS = 32                # KMeans K=32
MODEL_DIM = 3584              # Qwen2-7B hidden_size
TEMPERATURE = 1.0             # softmax 温度 (用户 v1 默认)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

ANCHORS_PATH_DEFAULT = "/home/wlia0047/hj82_scratch2/wenyu/pca48_kanchor/anchors.npy"

PROMPT_TEMPLATE = (
    "Product attributes: {attrs}. "
    "Write a first-person search query mentioning these attributes: "
)


class PCA48KAnchorController(nn.Module):
    """PCA48 z → 1 control embedding (B, hidden_size).

    Frozen K-Means anchors + learnable per-anchor embeddings.
    """

    def __init__(
        self,
        anchors: np.ndarray,
        hidden_size: int = MODEL_DIM,
        temperature: float = TEMPERATURE,
    ):
        super().__init__()
        assert anchors.shape[1] == PCA_DIM, f"anchors must be (K, {PCA_DIM}), got {anchors.shape}"
        K = anchors.shape[0]
        self.K = K
        self.hidden_size = hidden_size
        self.temperature = temperature

        # Frozen anchors
        self.register_buffer(
            "anchors",
            torch.tensor(anchors, dtype=torch.float32),
        )

        # Learnable per-anchor embeddings (small init scale 0.02)
        self.anchor_embeddings = nn.Parameter(
            torch.randn(K, hidden_size) * 0.02
        )

        # Diagnostic
        n_param = self.anchor_embeddings.numel()
        print(f"[PCA48KAnchorController] K={K}, hidden={hidden_size}, τ={temperature}, "
              f"learnable params={n_param:,}")

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, 48) → control_emb (B, hidden_size), weights (B, K)."""
        # d_k = ||z - a_k||²
        dist = ((z[:, None, :] - self.anchors[None, :, :]) ** 2).sum(dim=-1)  # (B, K)
        weights = torch.softmax(-dist / self.temperature, dim=-1)             # (B, K)
        control_emb = weights @ self.anchor_embeddings                        # (B, hidden_size)
        return control_emb, weights


class Qwen2WithKAnchor(nn.Module):
    """Qwen2-7B frozen + LoRA (r=16, q/k/v/o) + K-anchor controller (trainable).

    Forward 时 z 通过 KAnchor controller 得到 1 个 control_emb, 替换
    inputs_embeds 的第 0 位 (BOS 位置)。
    """

    def __init__(
        self,
        anchors_path: str = ANCHORS_PATH_DEFAULT,
        model_path: str = QWEN_PATH,
        lora_r: int = LORA_R,
        lora_alpha: int = LORA_ALPHA,
        temperature: float = TEMPERATURE,
        dtype: str = "bfloat16",
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        base_model.config.use_cache = False
        for p in base_model.parameters():
            p.requires_grad = False

        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_cfg)
        self.dtype = torch_dtype
        self.hidden_size = base_model.config.hidden_size
        self.control_len = 1  # 1 control_emb (替换 BOS 位置)

        # Load anchors
        if not Path(anchors_path).exists():
            raise FileNotFoundError(
                f"anchors not found at {anchors_path}. "
                f"Run K-Means first (see syntax_subspace_pca48_kanchor_data.py)."
            )
        anchors_np = np.load(anchors_path)
        print(f"[Qwen2WithKAnchor] loaded anchors from {anchors_path}: shape={anchors_np.shape}")

        # K-anchor controller
        self.controller = PCA48KAnchorController(
            anchors=anchors_np,
            hidden_size=self.hidden_size,
            temperature=temperature,
        ).to("cuda", dtype=torch_dtype)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[Qwen2WithKAnchor] trainable params: {n_trainable:,} / {n_total:,} "
              f"({100 * n_trainable / n_total:.4f}%)")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _build_prompt(self, attrs_list: list[list[str]]) -> list[str]:
        return [PROMPT_TEMPLATE.format(attrs=", ".join(a)) for a in attrs_list]

    def forward(
        self,
        z: torch.Tensor,
        attrs_list: list[list[str]],
        target_texts: list[str],
        return_components: bool = False,
    ):
        """Compute L_LM = -log p(target | control_emb, prompt)."""
        device = next(self.model.parameters()).device
        B = z.size(0)

        # 1. Control embedding from PCA48 (1 control_emb per sample)
        z_d = z.to(device=device, dtype=self.dtype)
        control_emb, weights = self.controller(z_d)  # (B, H), (B, K)

        # 2. Build prompts (right-padding for training)
        orig_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            prompt_texts = self._build_prompt(attrs_list)
            prompt_enc = self.tokenizer(
                prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            target_enc = self.tokenizer(
                target_texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
        finally:
            self.tokenizer.padding_side = orig_padding

        prompt_ids = prompt_enc.input_ids
        target_ids = target_enc.input_ids
        prompt_attn = prompt_enc.attention_mask
        target_attn = target_enc.attention_mask

        # 3. Get text embeddings
        embed_layer = self.model.get_input_embeddings()
        prompt_embeds = embed_layer(prompt_ids).to(self.dtype)  # (B, T_p, H)
        target_embeds = embed_layer(target_ids).to(self.dtype)  # (B, T_t, H)

        # 4. Concat: [prompt; target] (no prefix here)
        text_embeds = torch.cat([prompt_embeds, target_embeds], dim=1)  # (B, T_p+T_t, H)
        text_attn = torch.cat([prompt_attn, target_attn], dim=1)

        # 5. Prepend control_emb to text_embeds
        # inputs_embeds[:, 0, :] = control_emb  (替换 BOS-like 位置)
        ctrl_emb_unsq = control_emb.unsqueeze(1)  # (B, 1, H)
        ctrl_attn = torch.ones((B, 1), dtype=torch.long, device=device)
        inputs_embeds = torch.cat([ctrl_emb_unsq, text_embeds], dim=1)  # (B, 1+T, H)
        attention_mask = torch.cat([ctrl_attn, text_attn], dim=1)

        # 6. Labels: -100 for control + prompt, target_ids (with pad masked)
        ctrl_labels = torch.full((B, 1), -100, dtype=torch.long, device=device)
        prompt_labels = torch.full_like(prompt_ids, -100)
        target_labels = target_ids.clone()
        target_labels[target_attn == 0] = -100
        labels = torch.cat([ctrl_labels, prompt_labels, target_labels], dim=1)

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        if return_components:
            return out.loss, weights
        return out.loss

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        attrs_list: list[list[str]],
        max_new_tokens: int = 60,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> list[str]:
        """Generate queries given PCA48 z and attrs."""
        device = next(self.model.parameters()).device
        B = z.size(0)
        z_d = z.to(device=device, dtype=self.dtype)
        control_emb, _ = self.controller(z_d)  # (B, H)

        orig_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            prompt_texts = self._build_prompt(attrs_list)
            prompt_enc = self.tokenizer(
                prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
        finally:
            self.tokenizer.padding_side = orig_padding

        prompt_ids = prompt_enc.input_ids
        prompt_attn = prompt_enc.attention_mask
        embed_layer = self.model.get_input_embeddings()
        prompt_embeds = embed_layer(prompt_ids).to(self.dtype)

        ctrl_emb_unsq = control_emb.unsqueeze(1)
        ctrl_attn = torch.ones((B, 1), dtype=torch.long, device=device)
        inputs_embeds = torch.cat([ctrl_emb_unsq, prompt_embeds], dim=1)
        attention_mask = torch.cat([ctrl_attn, prompt_attn], dim=1)

        out = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return [
            self.tokenizer.decode(t, skip_special_tokens=True).strip()
            for t in out
        ]

    def save_controller(self, path: str) -> None:
        """Save K-anchor controller + LoRA state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "anchor_embeddings": self.controller.anchor_embeddings.detach().cpu(),
            "anchors": self.controller.anchors.detach().cpu(),  # for sanity / restore
            "lora_state_dict": {
                k: v for k, v in self.model.state_dict().items() if "lora_" in k
            },
        }
        torch.save(state, path)

    def load_controller(self, path: str) -> None:
        state = torch.load(path, map_location="cuda")
        self.controller.anchor_embeddings.data = state["anchor_embeddings"].to(
            device=self.controller.anchor_embeddings.device,
            dtype=self.controller.anchor_embeddings.dtype,
        )
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state["lora_state_dict"])


__all__ = [
    "PCA48KAnchorController", "Qwen2WithKAnchor",
    "PROMPT_TEMPLATE", "PCA_DIM", "K_ANCHORS", "MODEL_DIM", "TEMPERATURE",
    "ANCHORS_PATH_DEFAULT",
]
