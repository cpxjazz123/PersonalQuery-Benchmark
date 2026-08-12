#!/usr/bin/env python3
"""Compute ground truth KL via scipy MC and compare with our closed-form."""
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as M  # noqa: E402

from scipy.stats import t as student_t


def kl_mc(mu_p, sigma_p, df_p, mu_q, sigma_q, df_q, n=200000, seed=0):
    """MC estimate of KL(St_q || St_p) by sampling from St_q (matches our code's direction)."""
    rng = np.random.default_rng(seed)
    x = student_t.rvs(df=df_q, loc=mu_q, scale=sigma_q, size=n, random_state=rng)
    log_p_p = student_t.logpdf(x, df=df_p, loc=mu_p, scale=sigma_p)
    log_p_q = student_t.logpdf(x, df=df_q, loc=mu_q, scale=sigma_q)
    return float(np.mean(log_p_q - log_p_p))


# Test cases
tests = [
    # (mu_p, sig_p, df_p, mu_q, sig_q, df_q, label)
    (0.5, 1.0, 5.0, 0.0, np.exp(0.7), 20.0, "q=heavy-tailed, p=near-Gauss"),
    (0.0, 1.0, 5.0, 0.0, 1.0, 5.0, "same params (KL=0)"),
    (0.5, 1.0, 5.0, 0.0, 1.0, 5.0, "only μ differs"),
    (1.0, 1.0, 50.0, 0.0, 1.0, 50.0, "ν=50 (near-Gauss)"),
]

print("=" * 70)
print(f"{'label':<35} {'MC ground':<12} {'our code':<12} {'diff':<10}")
print("=" * 70)
for mu_p, sig_p, df_p, mu_q, sig_q, df_q, label in tests:
    kl_truth = kl_mc(mu_p, sig_p, df_p, mu_q, sig_q, df_q)
    # Convert to torch + our code
    mu_pt = torch.tensor([[mu_q]])  # NOTE: in our code, arg 1 is q, arg 2 is p
    logvar_pt = torch.tensor([[np.log(sig_q**2)]])
    df_pt = torch.tensor([[df_q]])
    mu_p_pt = torch.tensor([[mu_p]])
    logvar_p_pt = torch.tensor([[np.log(sig_p**2)]])
    df_p_pt = torch.tensor([[df_p]])
    # Use K=1024 for the validation test to reduce MC variance.
    kl_ours = float(
        M.student_t_kl_mc(mu_pt, logvar_pt, df_pt, mu_p_pt, logvar_p_pt, df_p_pt, K=1024)
        .mean().item()
    )
    print(f"{label:<35} {kl_truth:<12.4f} {kl_ours:<12.4f} {abs(kl_truth - kl_ours):<10.4f}")
