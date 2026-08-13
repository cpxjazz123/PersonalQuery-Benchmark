"""Soft-prefix projector for E11 user style conditioning.

P_u = reshape(LayerNorm(MLP(z_u)))  ->  K virtual tokens of width d_model.

Init is small-std (std=0.02) so the projected prefix starts close to the
LayerNorm output of a near-zero input (i.e. close to the model's default
behavior), matching the "gate zero-init keeps initial behavior" spirit of
prefix-injection work.
"""
from __future__ import annotations

import torch
from torch import nn


class SoftPrefixProjector(nn.Module):
    """Projects a user style vector z_u (user_dim,) into K soft-prefix tokens.

    Architecture: Linear(user_dim -> hidden_dim) -> GELU ->
                  Linear(hidden_dim -> K * d_model) -> LayerNorm(d_model)
    """

    def __init__(
        self,
        user_dim: int,
        hidden_dim: int,
        num_tokens: int,
        model_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        self.mlp = nn.Sequential(
            nn.Linear(user_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * model_dim),
        )
        self.layernorm = nn.LayerNorm(model_dim)
        for name, param in self.named_parameters():
            if param.ndim >= 2 and "layernorm" not in name:
                nn.init.normal_(param, std=0.02)
            elif param.ndim == 1 and "layernorm" not in name:
                nn.init.zeros_(param)
        self.to(dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, user_dim] -> prefix embeddings [B, num_tokens, model_dim]."""
        out = self.mlp(z)  # [B, K * d_model]
        out = out.view(-1, self.num_tokens, self.model_dim)
        return self.layernorm(out)

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )
