#!/usr/bin/env python3
"""Smoke test for the diagonal_gmm VADES user prior.

Constructs a tiny fake encoder + UserDistributionTableGMM, runs a forward
pass, computes user_match_loss + user_prior_kl_loss, and verifies backward
populates all parameter gradients with non-zero norms. CPU only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/fs04/ar57/wenyu")
sys.path.insert(0, str(REPO_ROOT / "PersoanlQuery" / "10_complexity_analysis" / "common"))

# Force the env vars to take effect *before* importing the train module.
os.environ.setdefault("VADES_COVARIANCE_MODE", "diagonal_gmm")
os.environ.setdefault("VADES_GMM_K", "2")

import importlib
import train_vades_lite_sentence_latent_threshold as tm
importlib.reload(tm)


def main() -> None:
    print(f"COVARIANCE_MODE = {tm.COVARIANCE_MODE}")
    print(f"GMM_COMPONENTS = {tm.GMM_COMPONENTS}")
    if tm.COVARIANCE_MODE != "diagonal_gmm":
        raise RuntimeError("env var did not propagate to tm.COVARIANCE_MODE")
    if tm.GMM_COMPONENTS != 2:
        raise RuntimeError("env var did not propagate to tm.GMM_COMPONENTS")

    B, U, D, K = 8, 16, 20, 2
    device = "cpu"

    encoder = tm.SentenceEncoder(input_dim=D, hidden_dim=32, latent_dim=D).to(device)
    user_table = tm.UserDistributionTableGMM(num_users=U, latent_dim=D, num_components=K).to(device)
    print(f"UserDistributionTableGMM params: {sum(1 for _ in user_table.parameters())} (expect 3)")

    x = torch.randn(B, D, device=device)
    mu, logvar, recon = encoder(x)
    print(f"encoder mu: {mu.shape}, logvar: {logvar.shape}, recon: {recon.shape}")

    log_p_xu = tm.gmm_log_likelihood(mu, logvar, user_table.user_mu, user_table.user_logvar, user_table.mix_logits)
    print(f"log_p_xu: {log_p_xu.shape} (expect [{B}, {U}])")
    if tuple(log_p_xu.shape) != (B, U):
        raise RuntimeError(f"shape mismatch: got {log_p_xu.shape}, expected ({B}, {U})")

    target = torch.randint(0, U, (B,))
    user_match_loss = F.cross_entropy(log_p_xu, target)
    print(f"user_match_loss: {user_match_loss.item():.4f}")

    batch_user_idx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    bu_mu, bu_logvar, bu_mix = user_table(batch_user_idx)
    print(f"batch_user_mu: {bu_mu.shape}, mix: {bu_mix.shape}")
    if tuple(bu_mu.shape) != (B, K, D):
        raise RuntimeError(f"shape mismatch: got {bu_mu.shape}, expected ({B}, {K}, {D})")

    per_comp_kl = tm.standard_normal_kl(bu_mu, bu_logvar)
    mix_probs = F.softmax(bu_mix, dim=-1)
    user_prior_kl_loss = (mix_probs * per_comp_kl).sum(dim=-1).mean()
    print(f"user_prior_kl_loss: {user_prior_kl_loss.item():.4f}")

    total = user_match_loss + 0.02 * user_prior_kl_loss
    total.backward()

    grad_norms = []
    for name, p in encoder.named_parameters():
        if p.grad is not None:
            grad_norms.append((f"encoder.{name}", float(p.grad.norm())))
    for name, p in user_table.named_parameters():
        if p.grad is not None:
            grad_norms.append((f"user_table.{name}", float(p.grad.norm())))

    print(f"\n所有参数是否都有 grad (norm>0): {all(g > 0 for _, g in grad_norms)}")
    print(f"grad norm 范围: min={min(g for _, g in grad_norms):.4f} max={max(g for _, g in grad_norms):.4f}")
    if not all(g > 0 for _, g in grad_norms):
        raise RuntimeError("some parameters have zero gradient")
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
