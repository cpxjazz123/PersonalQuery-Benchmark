#!/usr/bin/env python3
"""Smoke test for 3 new non-Gaussian user prior modes."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/fs04/ar57/wenyu")
sys.path.insert(0, str(REPO / "PersoanlQuery" / "10_complexity_analysis" / "common"))
import train_vades_lite_sentence_latent_threshold as M  # noqa: E402

NUM_USERS = 8
LATENT_DIM = 20
BATCH = 32
GMM_K = 2


def test_mode(mode: str, label: str, device: torch.device) -> None:
    print(f"\n=== {label} ===")
    M.COVARIANCE_MODE = mode
    M.GMM_COMPONENTS = GMM_K

    if mode == "diagonal_student_t":
        user_table = M.UserDistributionTableStudentT(NUM_USERS, LATENT_DIM).to(device)
    elif mode == "diagonal_laplace":
        user_table = M.UserDistributionTableLaplace(NUM_USERS, LATENT_DIM).to(device)
    elif mode == "diagonal_student_t_gmm":
        user_table = M.UserDistributionTableStudentTGMM(NUM_USERS, LATENT_DIM, GMM_K).to(device)
    else:
        raise ValueError(f"unknown mode: {mode}")
    encoder = M.SentenceEncoder(input_dim=LATENT_DIM, hidden_dim=64, latent_dim=LATENT_DIM).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(user_table.parameters()), lr=1e-3)

    x = torch.randn(BATCH, LATENT_DIM, device=device)
    user_idx = torch.randint(0, NUM_USERS, (BATCH,), device=device)
    all_idx = torch.arange(NUM_USERS, device=device)

    for step in range(3):
        sent_mu, sent_dispersion, reconstruction = encoder(x)
        if mode == "diagonal_student_t":
            log_p = M.student_t_log_likelihood(sent_mu, sent_dispersion, *user_table(user_idx))
            loss_match = F.cross_entropy(log_p, user_idx)
            mu, log_scale, df = user_table(all_idx)
            nu = df
            nu_factor = nu / (nu - 2.0).clamp_min(1e-3)
            log_nu_factor = torch.log(nu_factor.clamp_min(1.0)).unsqueeze(-1)  # [8, 1]
            equiv_logvar = 2.0 * log_scale + log_nu_factor
            loss_prior = M.standard_normal_kl(mu, equiv_logvar).mean()
        elif mode == "diagonal_laplace":
            log_p = M.laplace_log_likelihood(sent_mu, sent_dispersion, *user_table(user_idx))
            loss_match = F.cross_entropy(log_p, user_idx)
            mu, log_b = user_table(all_idx)
            equiv_logvar = 2.0 * log_b + float(np.log(2.0))
            loss_prior = M.standard_normal_kl(mu, equiv_logvar).mean()
        elif mode == "diagonal_student_t_gmm":
            log_p = M.student_t_gmm_log_likelihood(sent_mu, sent_dispersion, *user_table(user_idx))
            loss_match = F.cross_entropy(log_p, user_idx)
            mu_k, log_scale_k, df_k, mix = user_table(all_idx)
            nu = df_k
            nu_factor = nu / (nu - 2.0).clamp_min(1e-3)
            log_nu_factor = torch.log(nu_factor.clamp_min(1.0)).unsqueeze(-1)  # [8, 2, 1]
            equiv_logvar_k = 2.0 * log_scale_k + log_nu_factor
            per_comp_kl = M.standard_normal_kl(mu_k, equiv_logvar_k)
            mix_probs = F.softmax(mix, dim=-1)
            loss_prior = (mix_probs * per_comp_kl).sum(dim=-1).mean()
        loss_recon = F.mse_loss(reconstruction, x)
        loss_align = F.mse_loss(sent_mu, x)
        loss = loss_match + 0.8 * loss_recon + 0.02 * loss_prior + loss_align
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), f"step {step}: loss is not finite: {loss.item()}"
        for name, p in user_table.named_parameters():
            assert p.grad is not None, f"step {step}: {name} grad is None"
            assert torch.isfinite(p.grad).all(), f"step {step}: {name} grad not finite"
        print(f"  step {step}: loss={loss.item():.4f} match={loss_match.item():.4f} prior={loss_prior.item():.4f} recon={loss_recon.item():.4f} align={loss_align.item():.4f}")
    print(f"  ✓ {label} 完整 forward + backward 跑通")


if __name__ == "__main__":
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke test] using device: {device}")
    test_mode("diagonal_student_t", "Student-t (single)", device)
    test_mode("diagonal_laplace", "Laplace (single)", device)
    test_mode("diagonal_student_t_gmm", "Student-t GMM (K=2)", device)
    print("\n[ALL PASS] 3 个新模式 forward + backward 全部正常")
