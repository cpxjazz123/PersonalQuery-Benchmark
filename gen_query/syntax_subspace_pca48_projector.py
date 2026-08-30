#!/usr/bin/env python3
"""Phase 6.A.2 — PCA48 Syntax Controller (projector + Qwen2-7B + LoRA).

用户指令 2026-08-30: Phase 5 NO-GO 后转 Phase 6 — 把 PCA48 直接送进模型内部。
架构:
  PCA48 z (B, 48)
    ↓  PCA48Projector: Linear(48→256) → ReLU → Linear(256→prefix_len × 3584)
  prefix_embeds (B, prefix_len=8, model_dim=3584)
    ↓
  inputs_embeds = [prefix_embeds; embed(prompt); embed(target)]
    ↓
  Qwen2-7B (frozen) + LoRA r=16 on q/k/v/o (trainable)

训练 loss:
  L_LM = -log p(target | prefix, prompt)
  (Phase 6.A pilot 暂不加 L_control / DPO)

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
QWEN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

# === PCA48 Syntax Controller config (locked) ===
PCA_DIM = 48
PROJECTOR_HIDDEN = 256
PREFIX_LEN = 8
MODEL_DIM = 3584  # Qwen2-7B hidden_size
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

PROMPT_TEMPLATE = (
    "Product attributes: {attrs}. "
    "Write a first-person search query mentioning these attributes: "
)


class PCA48Projector(nn.Module):
    """Map PCA48 z (B, 48) → prefix embeddings (B, prefix_len, model_dim).

    两层 MLP, 用户设计: 48 → 256 → prefix_len × 3584.
    """

    def __init__(
        self,
        pca_dim: int = PCA_DIM,
        hidden: int = PROJECTOR_HIDDEN,
        prefix_len: int = PREFIX_LEN,
        model_dim: int = MODEL_DIM,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(pca_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, prefix_len * model_dim),
        )
        self.prefix_len = prefix_len
        self.model_dim = model_dim
        # 零初始化最后一层, 让 prefix 从"零扰动"开始, 训练早期不会污染 LM 输出
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, pca_dim) → (B, prefix_len, model_dim)."""
        h = self.proj(z)
        return h.view(-1, self.prefix_len, self.model_dim)


class Qwen2WithPrefix(nn.Module):
    """Qwen2-7B frozen + LoRA (r=16, q/k/v/o) + PCA48 projector (trainable).

    Forward 时 z 通过 projector 得到 prefix_embeds, prefix_embeds 拼在
    prompt token embeddings 前面一起送进 Qwen2。
    """

    def __init__(
        self,
        model_path: str = QWEN_PATH,
        lora_r: int = LORA_R,
        lora_alpha: int = LORA_ALPHA,
        prefix_len: int = PREFIX_LEN,
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
        # left-padding 是 generation 的标准, training 临时切到 right
        self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        base_model.config.use_cache = False
        # 冻结 backbone
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
        self.prefix_len = prefix_len
        self.hidden_size = base_model.config.hidden_size

        # projector (trainable, 单独管理的 nn.Module, 不通过 peft)
        self.projector = PCA48Projector(
            pca_dim=PCA_DIM,
            hidden=PROJECTOR_HIDDEN,
            prefix_len=prefix_len,
            model_dim=base_model.config.hidden_size,
        ).to("cuda", dtype=torch_dtype)

        # 打印 trainable params
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[Qwen2WithPrefix] trainable params: {n_trainable:,} / {n_total:,} "
              f"({100 * n_trainable / n_total:.4f}%)")

    def trainable_parameters(self):
        """返回 trainable 参数 (projector + LoRA), 用于 optimizer."""
        return [p for p in self.parameters() if p.requires_grad]

    def _build_prompt(self, attrs_list: list[list[str]]) -> list[str]:
        return [PROMPT_TEMPLATE.format(attrs=", ".join(a)) for a in attrs_list]

    def forward(
        self,
        z: torch.Tensor,
        attrs_list: list[list[str]],
        target_texts: list[str],
    ) -> torch.Tensor:
        """Compute L_LM = -log p(target | prefix, prompt)."""
        device = next(self.model.parameters()).device
        B = z.size(0)

        # 1. Prefix embeddings from PCA48
        z_d = z.to(device=device, dtype=self.dtype)
        prefix_embeds = self.projector(z_d)  # (B, prefix_len, model_dim)

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

        # 4. Concat: [prefix; prompt; target]
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)
        # 5. Attention mask: prefix 全 1, prompt_attn, target_attn
        prefix_attn = torch.ones((B, self.prefix_len), dtype=torch.long, device=device)
        attention_mask = torch.cat([prefix_attn, prompt_attn, target_attn], dim=1)

        # 6. Labels: -100 for prefix, -100 for prompt, target_ids (with pad masked)
        prefix_labels = torch.full((B, self.prefix_len), -100, dtype=torch.long, device=device)
        prompt_labels = torch.full_like(prompt_ids, -100)
        target_labels = target_ids.clone()
        target_labels[target_attn == 0] = -100  # mask pad
        labels = torch.cat([prefix_labels, prompt_labels, target_labels], dim=1)

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
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
        prefix_embeds = self.projector(z_d)  # (B, prefix_len, model_dim)

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

        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        prefix_attn = torch.ones((B, self.prefix_len), dtype=torch.long, device=device)
        attention_mask = torch.cat([prefix_attn, prompt_attn], dim=1)

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
        # transformers generate(inputs_embeds=...) 返回仅新生成的 token
        # (与 input_ids 路径不同 — input_ids 路径会返回完整序列)
        # 所以直接 decode 整个 out 即可
        return [
            self.tokenizer.decode(t, skip_special_tokens=True).strip()
            for t in out
        ]

    def save_projector(self, path: str) -> None:
        """只存 projector + LoRA adapter (Qwen 本身不存, 推理时从 path 加载)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "projector": self.projector.state_dict(),
            "lora_state_dict": {
                k: v for k, v in self.model.state_dict().items() if "lora_" in k
            },
        }
        torch.save(state, path)

    def load_projector(self, path: str) -> None:
        state = torch.load(path, map_location="cuda")
        self.projector.load_state_dict(state["projector"])
        # LoRA state_dict 需要对齐 keys, peft 接口:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state["lora_state_dict"])


__all__ = ["PCA48Projector", "Qwen2WithPrefix", "PROMPT_TEMPLATE",
           "PCA_DIM", "PROJECTOR_HIDDEN", "PREFIX_LEN", "MODEL_DIM"]