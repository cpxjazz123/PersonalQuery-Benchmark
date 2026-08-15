"""E14-P — Copy-aware (pointer) attribute copying for free-form query
generation.

Design (#14 P-series): the model generates a NATURAL query (no <A*> template
constraint) while price/brand/model strings are COPIED verbatim from the
product-attribute input via a pointer/copy distribution.

    P(token) = p_copy * P_copy(token) + (1 - p_copy) * P_generate(token)

- P_copy: attention over the input attribute spans (pointer network). Any
  token in the input that belongs to an attribute value receives copy
  probability; spans are copied whole (span-level, e.g. "30.77" stays one
  unit).
- P_generate: the base LM distribution over the full vocabulary.
- p_copy: learned gate from the current hidden state.
- The user style condition (soft prefix / vector) enters ONLY through
  P_generate (P7): attribute strings are unaffected by the user condition.

Training loss = CE on the mixed distribution + lambda * copy pointer loss on
target positions that belong to an attribute value.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def build_attr_prompt_lines(attrs_used: Dict[str, str]) -> str:
    """Structured attribute input; each attribute on its own line so spans are
    easy to locate. Value text is verbatim."""
    lines = ["Product attributes:"]
    for key in sorted(attrs_used, key=lambda k: int(k[1:])):
        lines.append(f"{key}: {attrs_used[key]}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)


def find_attr_value_spans(prompt_str: str, attrs_used: Dict[str, str]):
    """Character spans [start, end) of each attribute value inside prompt_str.

    Longest value first so that a value contained in a longer one is not
    shadowed (e.g. "Baby" inside "Baby Trend").
    """
    spans: Dict[str, List[Tuple[int, int]]] = {}
    used: List[Tuple[int, int]] = []
    items = sorted(
        ((k, str(v)) for k, v in attrs_used.items() if str(v)),
        key=lambda kv: -len(kv[1]),
    )
    for key, value in items:
        matches = [m.span() for m in re.finditer(re.escape(value), prompt_str)]
        for sp in matches:
            if any(not (sp[1] <= o[0] or sp[0] >= o[1]) for o in used):
                continue
            used.append(sp)
            spans.setdefault(key, []).append(sp)
            break
    return spans


def attr_token_spans(prompt_str: str, attrs_used: Dict[str, str], tokenizer) -> Dict[str, List[List[int]]]:
    """Token spans of each attribute value inside the encoded prompt.

    Uses offset mapping of the fast tokenizer to convert character spans to
    token index ranges (in the FULL prompt token sequence, including specials).
    """
    enc = tokenizer(prompt_str, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    char_spans = find_attr_value_spans(prompt_str, attrs_used)
    out: Dict[str, List[List[int]]] = {}
    for key, spans in char_spans.items():
        key_spans = []
        for (cs, ce) in spans:
            tok_span = []
            for i, (os_, oe) in enumerate(offsets):
                if os_ < ce and oe > cs:
                    tok_span.append(i)
            if tok_span:
                key_spans.append([tok_span[0], tok_span[-1]])
        if key_spans:
            out[key] = key_spans
    return out


class CopyAwareHead(nn.Module):
    """Pointer-copy head over the input attribute tokens.

    forward(hidden, src_hidden, src_mask, attr_token_ids, attr_span_mask):
      copy_logits[t] = (W_c h_t) . src_hidden  over source positions
      P_copy = scatter-add of copy_logits to vocabulary (by source token id)
      gate p_copy = sigmoid(MLP(h_t))
    """

    def __init__(self, hidden_dim: int, vocab_size: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.vocab_size = vocab_size
        self.to(dtype)

    def forward(
        self,
        hidden: torch.Tensor,            # [B, T, D]  generation positions
        src_hidden: torch.Tensor,        # [B, S, D]  input token hidden states
        src_attr_mask: torch.Tensor,     # [B, S]    1 = token belongs to an attribute value span
        src_ids: torch.Tensor,           # [B, S]    input token ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (p_copy [B, T, 1], copy_vocab_logits [B, T, V])."""
        q = self.proj_q(hidden)                          # [B, T, D]
        attn = torch.bmm(q, src_hidden.transpose(1, 2))  # [B, T, S]
        attn = attn.masked_fill(src_attr_mask.unsqueeze(1) == 0, float("-inf"))
        # safe softmax: all-masked rows become uniform (no NaN)
        safe = attn.clone()
        row_inf = (~torch.isfinite(safe)).all(dim=-1, keepdim=True)
        safe = torch.where(row_inf, torch.zeros_like(safe), safe)
        attn = torch.softmax(safe, dim=-1)               # [B, T, S]
        # scatter copy mass to vocabulary positions — fully vectorized:
        # [B*T, S] attn scattered into [B*T, V] by source token id (single
        # kernel; no per-(b,t) Python loop).
        B, T, S = attn.shape
        BT = B * T
        flat_attn = attn.reshape(BT, S)
        flat_src = src_ids.unsqueeze(1).expand(B, T, S).reshape(BT, S).long()
        copy_flat = torch.zeros(BT, self.vocab_size, device=hidden.device, dtype=hidden.dtype)
        copy_flat.scatter_add_(1, flat_src, flat_attn)
        copy_logits = copy_flat.reshape(B, T, self.vocab_size)
        p_copy = torch.sigmoid(self.gate(hidden))        # [B, T, 1]
        return p_copy, copy_logits

    def pointer_logits(self, hidden: torch.Tensor, src_hidden: torch.Tensor,
                       src_attr_mask: torch.Tensor) -> torch.Tensor:
        """Raw pointer attention over source positions [B, T, S] (for the
        pointer loss: supervise the span position of a copied target token)."""
        q = self.proj_q(hidden)
        attn = torch.bmm(q, src_hidden.transpose(1, 2))
        attn = attn.masked_fill(src_attr_mask.unsqueeze(1) == 0, float("-inf"))
        return attn


def mixed_logits(gen_logits: torch.Tensor, p_copy: torch.Tensor,
                 copy_logits: torch.Tensor) -> torch.Tensor:
    """P = p_copy * P_copy + (1 - p_copy) * P_generate (log-space mixing).

    gen_logits: [B, T, V]; copy_logits: [B, T, V] (probabilities); p_copy [B,T,1].
    Returns mixed logits (safe: log(p) with p clamped).
    """
    log_copy = torch.log(copy_logits.clamp_min(1e-9))
    log_gen = gen_logits.log_softmax(dim=-1)
    log_mix = torch.logsumexp(
        torch.stack([log_copy + torch.log(p_copy.clamp_min(1e-9)),
                     log_gen + torch.log((1 - p_copy).clamp_min(1e-9))], dim=-1),
        dim=-1,
    )
    return log_mix
