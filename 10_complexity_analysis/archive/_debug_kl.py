#!/usr/bin/env python3
"""Debug: print intermediate values for the heavy-tailed test case."""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as M  # noqa: E402

torch.manual_seed(0)

K = 256
mu_q = torch.tensor([[0.5]])
logvar_q = torch.tensor([[0.0]])
df_q = torch.tensor([[5.0]])
mu_p = torch.tensor([[0.0]])
logvar_p = torch.tensor([[0.7]])
df_p = torch.tensor([[20.0]])

# Sample z from St_q
V = torch._standard_gamma(df_q.unsqueeze(0) / 2.0) * (2.0 / df_q.unsqueeze(0))
print(f"V shape: {V.shape}, V[:3]: {V.flatten()[:3].tolist()}")
print(f"V stats: mean={V.mean().item():.4f}, std={V.std().item():.4f}, min={V.min().item():.4f}, max={V.max().item():.4f}")

eps = torch.randn(K, *mu_q.shape)
print(f"eps stats: mean={eps.mean().item():.4f}, std={eps.std().item():.4f}")

z = mu_q.unsqueeze(0) + torch.exp(0.5 * logvar_q).unsqueeze(0) * eps / torch.sqrt(V)
print(f"z shape: {z.shape}, mean={z.mean().item():.4f}, std={z.std().item():.4f}")
print(f"Expected z mean=0.5, std=sqrt(5/3)={np.sqrt(5/3):.4f}")

# Check z distribution
print(f"z[:5].flatten(): {z[:5].flatten().tolist()}")

# log q and log p
log_q = M._student_t_log_pdf_per_dim(z, mu_q.unsqueeze(0), logvar_q.unsqueeze(0), df_q.unsqueeze(0).unsqueeze(-1))
log_p = M._student_t_log_pdf_per_dim(
    z.unsqueeze(2),
    mu_p.unsqueeze(0).unsqueeze(0),
    logvar_p.unsqueeze(0).unsqueeze(0),
    df_p.unsqueeze(0).unsqueeze(0).unsqueeze(-1),
)
print(f"log_q shape: {log_q.shape}, log_p shape: {log_p.shape}")

# Per-sample KL
log_q_sum = log_q.sum(dim=-1)  # [K, B]
log_p_sum = log_p.sum(dim=-1)  # [K, B, U]
print(f"log_q_sum[:5]: {log_q_sum[:5].flatten().tolist()}")
print(f"log_p_sum[:5]: {log_p_sum[:5].flatten().tolist()}")

kl_per_sample = (log_q_sum.unsqueeze(2) - log_p_sum)  # [K, B, U]
print(f"kl_per_sample[:5].flatten(): {kl_per_sample[:5].flatten().tolist()}")
print(f"kl_per_sample stats: mean={kl_per_sample.mean().item():.4f}, std={kl_per_sample.std().item():.4f}")
print(f"kl_per_sample min={kl_per_sample.min().item():.4f}, max={kl_per_sample.max().item():.4f}")

# Negative samples?
n_negative = (kl_per_sample < 0).sum().item()
print(f"Negative samples: {n_negative}/{kl_per_sample.numel()}")
