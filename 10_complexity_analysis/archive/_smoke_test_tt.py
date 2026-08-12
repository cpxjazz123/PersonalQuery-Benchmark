#!/usr/bin/env python3
"""Smoke test: t-t strict VIB (Student-t encoder + Student-t user prior + one-MC-sample log p)."""
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as M  # noqa: E402

torch.manual_seed(0)

# === 1. reparameterize_student_t ===
mu = torch.zeros(100, 4)
logvar = torch.zeros(100, 4)
df = torch.full((100, 4), 26.0)
z = M.reparameterize_student_t(mu, logvar, df)
print(f"reparam z shape: {z.shape}")
assert z.shape == (100, 4)

# Sample variance should be close to df/(df-2) (Student-t variance)
sample_var = z.var(dim=0).mean()
expected_var = 26.0 / (26.0 - 2.0)  # = 1.0833
print(f"sample_var={sample_var.item():.4f}, expected={expected_var:.4f}")
assert abs(sample_var.item() - expected_var) < 0.2, f"reparam variance mismatch: {sample_var.item()} vs {expected_var}"
print("✓ reparameterize_student_t pass")

# === 2. student_t_log_p_matrix — one-MC-sample log p_t(z | u) ===
# Higher p when z is closer to user's mean → diagonal should be highest.
B, U, D = 4, 3, 2
z = torch.tensor([
    [0.0, 0.0],   # close to user 0 (mu=[0,0])
    [1.0, 1.0],   # close to user 1 (mu=[1,1])
    [-1.0, -1.0], # close to user 2 (mu=[-1,-1])
    [0.5, 0.5],   # ambiguous
], dtype=torch.float)
user_mu = torch.tensor([
    [0.0, 0.0],
    [1.0, 1.0],
    [-1.0, -1.0],
], dtype=torch.float)
user_log_scale = torch.zeros(U, D)
user_df = torch.full((U,), 10.0)
log_p = M.student_t_log_p_matrix(z, user_mu, user_log_scale, user_df)
print(f"log_p shape: {log_p.shape}")
print(f"log_p:\n{log_p}")
# Diagonal should be the argmax for samples 0,1,2 (and 3 is ambiguous).
assert log_p[0, 0] > log_p[0, 1] and log_p[0, 0] > log_p[0, 2], "row 0 should max at col 0"
assert log_p[1, 1] > log_p[1, 0] and log_p[1, 1] > log_p[1, 2], "row 1 should max at col 1"
assert log_p[2, 2] > log_p[2, 0] and log_p[2, 2] > log_p[2, 1], "row 2 should max at col 2"
print("✓ student_t_log_p_matrix pass")

# === 3. SentenceEncoderStudentT forward + backward ===
enc = M.SentenceEncoderStudentT(input_dim=10, hidden_dim=16, latent_dim=4)
x = torch.randn(32, 10)
mu_e, logvar_e, df_e, recon = enc(x)
print(f"encoder outputs: mu={mu_e.shape}, logvar={logvar_e.shape}, df={df_e.shape}, recon={recon.shape}")
assert mu_e.shape == (32, 4)
assert df_e.shape == (32, 4)
assert (df_e >= 2.0).all() and (df_e <= 50.0).all(), f"df out of range: {df_e}"
print(f"df range: [{df_e.min().item():.2f}, {df_e.max().item():.2f}]")
loss = F.mse_loss(recon, x)
loss.backward()
for name, p in enc.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(), f"invalid grad for {name}"
print("✓ SentenceEncoderStudentT forward + backward pass")

# === 4. End-to-end: t-t strict VIB — sample z, score log p_t, cross_entropy ===
U, D, B = 5, 4, 8
user_table = M.UserDistributionTableStudentT(num_users=U, latent_dim=D)
enc2 = M.SentenceEncoderStudentT(input_dim=10, hidden_dim=16, latent_dim=D)
x = torch.randn(B, 10)
batch_user_idx = torch.randint(0, U, (B,))
mu_q, logvar_q, df_q, _ = enc2(x)
z = M.reparameterize_student_t(mu_q, logvar_q, df_q)
user_mu_t, user_log_scale_t, user_df_t = user_table(batch_user_idx)
user_mu_all = user_table.user_mu
user_log_scale_all = user_table.user_log_scale
user_df_all = user_table.df_all()
log_p_matrix = M.student_t_log_p_matrix(
    z, user_mu_all, user_log_scale_all, user_df_all
)
print(f"log_p_matrix shape: {log_p_matrix.shape}")
assert log_p_matrix.shape == (B, U)
loss = F.cross_entropy(log_p_matrix, batch_user_idx)
print(f"user_match_loss: {loss.item():.4f}")
loss.backward()
assert torch.isfinite(enc2.mu_head.weight.grad).all()
assert torch.isfinite(user_table.user_mu.grad).all()
print("✓ end-to-end t-t strict VIB pass")

print("=" * 60)
print("ALL OK")
