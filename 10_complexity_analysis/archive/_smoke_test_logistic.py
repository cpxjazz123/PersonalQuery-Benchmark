#!/usr/bin/env python3
"""Smoke test: verify diagonal_logistic forward + backward on dummy data."""
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as M  # noqa: E402

torch.manual_seed(0)

# Dummy dims
B, U, D = 32, 8, 20

# Fake encoder outputs
mu_q = torch.randn(B, D, requires_grad=True)
logvar_q = torch.randn(B, D, requires_grad=True)

# UserDistributionTableLogistic
user_table = M.UserDistributionTableLogistic(num_users=U, latent_dim=D)
# Inject some non-trivial params so gradient is non-zero
with torch.no_grad():
    user_table.user_mu.add_(torch.randn_like(user_table.user_mu) * 0.1)
    user_table.user_log_s.add_(torch.randn_like(user_table.user_log_s) * 0.1)

# 1) Forward: log p(μ_q | u) -> [B, U]
user_idx = torch.arange(U)
u_mu, u_ls = user_table(user_idx)
log_p = M.logistic_log_likelihood(mu_q, logvar_q, u_mu, u_ls)
print(f"log_p shape: {log_p.shape}, expected: [{B}, {U}]")
assert log_p.shape == (B, U), f"shape mismatch: {log_p.shape}"

# 2) user_match_loss = CE(log_p, target_user_idx) and backward
target = torch.randint(0, U, (B,))
loss = F.cross_entropy(log_p, target)
print(f"user_match_loss: {loss.item():.4f}")
loss.backward()

assert mu_q.grad is not None and torch.isfinite(mu_q.grad).all(), "mu_q grad invalid"
# logvar_q is in the signature for branch uniformity but doesn't participate in
# the point-estimate log p — its grad is expected to be None (matches student_t,
# laplace, etc. — only GMM with `var_q/var_p` would propagate through it).
assert user_table.user_mu.grad is not None and torch.isfinite(user_table.user_mu.grad).all(), "user_mu grad invalid"
assert user_table.user_log_s.grad is not None and torch.isfinite(user_table.user_log_s.grad).all(), "user_log_s grad invalid"
print("✓ forward + backward for diagonal_logistic all pass")

# 3) Test user_prior_kl_loss branch
mu_all = user_table.user_mu
log_s_all = user_table.user_log_s
equiv_logvar = 2.0 * log_s_all + torch.log(torch.tensor(torch.pi**2 / 3.0))
kl = M.standard_normal_kl(mu_all, equiv_logvar).mean()
print(f"user_prior_kl_loss: {kl.item():.4f}")
kl.backward()  # should already be backwarded via cross_entropy but check no NaN

# 4) Compare with a known answer
# At mu=0, log_s=0 (s=1), x=0:
#   log p(0|0,1) = -log(1) - 0 - 2*log(1+exp(0)) = -2*log(2) = -2*0.6931 = -1.3863
# (we already know p(0) = 1/(4*1) = 0.25, log p = log(0.25) = -1.3863)
u_mu_test = torch.zeros(1, D)
u_ls_test = torch.zeros(1, D)
mu_q_test = torch.zeros(1, D)
log_p_test = M.logistic_log_likelihood(mu_q_test, None, u_mu_test, u_ls_test)
expected = -D * torch.log(torch.tensor(4.0))  # = -20 * 1.3863 = -27.726
print(f"log p(0|0,1) per dim 0 case: actual={log_p_test.item():.4f}, expected={expected.item():.4f}")
assert abs(log_p_test.item() - expected.item()) < 1e-3, "log p(0|0,1) mismatch"
print("✓ analytical check pass")
print("=" * 60)
print("ALL OK")
