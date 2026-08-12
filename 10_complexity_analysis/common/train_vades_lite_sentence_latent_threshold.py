#!/usr/bin/env python3
"""Train a sentence-level review-only VADES model with explicit user distributions and range-aware query ranking."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


SCRIPT_DIR = Path(__file__).resolve().parent

from cluster_strict5550_query_gmm_and_attach_retrieval import run_query_gmm_pipeline  # noqa: E402
from extract_clause_features_single_query import extract_clause_features_from_doc, load_spacy_model  # noqa: E402


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = os.environ.get("PQ_CATEGORY", "Baby_Products")
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
REVIEW_SOURCE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
)
CANDIDATE_QUERY_FILE = INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
RAW_CANDIDATE_QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / "query_by_expression_style_no_depth_check_10.json"

SEED = 42
DEVICE: torch.device | None = None
TRAIN_SENTENCES_PER_USER = 10
MAX_HOLDOUT_SENTENCES_PER_USER = 5
TOTAL_SENTENCES_PER_USER = TRAIN_SENTENCES_PER_USER + MAX_HOLDOUT_SENTENCES_PER_USER
MIN_SENTENCES_PER_USER = TRAIN_SENTENCES_PER_USER
OUTPUT_TAG = os.environ.get("VADES_OUTPUT_TAG", "vades_lite_sentence_user_distribution_train10_holdout10")
LATENT_DIM = int(os.environ.get("VADES_LATENT_DIM", "20"))
HIDDEN_DIM = int(os.environ.get("VADES_HIDDEN_DIM", "64"))
EPOCHS = int(os.environ.get("VADES_EPOCHS", "100"))
BATCH_SIZE = int(os.environ.get("VADES_BATCH_SIZE", "1024"))
USER_CHUNK_SIZE_TRAIN = int(os.environ.get("VADES_USER_CHUNK_SIZE_TRAIN", "256"))
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = int(os.environ.get("VADES_EARLY_STOP", "5"))
WEIGHT_DECAY = 1e-5
USER_MATCH_WEIGHT = 1.0
STYLE_RECON_WEIGHT = 0.8
SENT_KL_WEIGHT = 0.05
USER_PRIOR_KL_WEIGHT = 0.02
LATENT_ALIGN_WEIGHT = float(os.environ.get("VADES_LATENT_ALIGN_WEIGHT", "1.0"))
ABS_THRESHOLD_QUANTILE = float(os.environ.get("VADES_ABS_THRESHOLD_QUANTILE", "0.95"))
MAX_USERS_OVERRIDE = os.environ.get("VADES_MAX_USERS")
SKIP_POST_CLUSTERING = os.environ.get("VADES_SKIP_POST_CLUSTERING", "0") == "1"
COVARIANCE_MODE = os.environ.get("VADES_COVARIANCE_MODE", "diagonal_gmm")
VALID_COVARIANCE_MODES = {
    "diagonal", "full", "diagonal_gmm",
    "diagonal_student_t", "diagonal_laplace", "diagonal_student_t_gmm",
    "diagonal_logistic",
}
if COVARIANCE_MODE not in VALID_COVARIANCE_MODES:
    raise ValueError(
        f"VADES_COVARIANCE_MODE 必须是 {sorted(VALID_COVARIANCE_MODES)} 之一，得到 {COVARIANCE_MODE}"
    )
REGENERATION_MAX_ROUNDS = 10
CANDIDATES_PER_ROUND = 10
ENCODER_DIST = os.environ.get("VADES_ENCODER_DIST", "gaussian")
VALID_ENCODER_DISTS = {"gaussian", "student_t"}
if ENCODER_DIST not in VALID_ENCODER_DISTS:
    raise ValueError(
        f"VADES_ENCODER_DIST 必须是 {sorted(VALID_ENCODER_DISTS)} 之一，得到 {ENCODER_DIST}"
    )
# Encoder=student_t requires closed-form Student-t KL, which only exists when
# the user prior is also Student-t. Force COVARIANCE_MODE=diagonal_student_t.
if ENCODER_DIST == "student_t" and COVARIANCE_MODE != "diagonal_student_t":
    raise ValueError(
        f"VADES_ENCODER_DIST=student_t 强制要求 VADES_COVARIANCE_MODE=diagonal_student_t，得到 {COVARIANCE_MODE}"
    )
GMM_COMPONENTS = int(os.environ.get("VADES_GMM_K", "2"))
if COVARIANCE_MODE == "diagonal_student_t_gmm" and GMM_COMPONENTS < 2:
    raise ValueError(f"VADES_GMM_K 对 diagonal_student_t_gmm 必须 >= 2，得到 {GMM_COMPONENTS}")
if COVARIANCE_MODE == "diagonal_gmm" and GMM_COMPONENTS < 2:
    raise ValueError(f"VADES_GMM_K 必须 >= 2，得到 {GMM_COMPONENTS}")

SUMMARY_FILE = INPUT_DIR / f"{OUTPUT_TAG}_summary.json"
DETAIL_FILE = INPUT_DIR / f"{OUTPUT_TAG}_epoch_details.jsonl"
USER_PROFILE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_sentences.jsonl"
EXCLUDED_USER_FILE = INPUT_DIR / f"{OUTPUT_TAG}_excluded_users.jsonl"
SENTENCE_EXTRACT_CACHE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_extracted_sentences.jsonl"
SELECTED_RECORD_FILE = INPUT_DIR / f"{OUTPUT_TAG}_selected_query_records.jsonl"
REJECTED_RECORD_FILE = INPUT_DIR / f"{OUTPUT_TAG}_rejected_query_records.jsonl"
QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / f"query_by_expression_style_{OUTPUT_TAG}.json"


from common_utils import log  # 统一 log 函数


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def summarize_array(values: np.ndarray) -> dict:
    if len(values) == 0:
        raise ValueError("无法汇总空数组")
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def diagonal_gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """Compute KL(q||p) for diagonal Gaussians.
    
    Supports two input shapes:
    - Unvectorized: mu_q [D], mu_p [K,D] -> output [K]
    - Vectorized: mu_q [B,D], mu_p [B,K,D] or [K,D] -> output [B,K]
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    
    # Expand mu_p/logvar_p to match mu_q if needed for vectorized case
    if mu_q.dim() == 3 and mu_p.dim() == 2:
        mu_p = mu_p.unsqueeze(0).expand(mu_q.size(0), -1, -1)
        logvar_p = logvar_p.unsqueeze(0).expand(mu_q.size(0), -1, -1)
    
    result = 0.5 * (
        logvar_p
        - logvar_q
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    )
    # Handle both 2D (unvectorized) and 3D (vectorized) cases
    return result.sum(dim=-1)


def standard_normal_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).sum(dim=-1)


def multivariate_standard_normal_kl(mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, L L^T) || N(0, I)) for full covariance.

    L: [*, D, D] lower-triangular Cholesky factor (diag strictly > 0).
    """
    diag = torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-6)
    log_det = 2.0 * torch.log(diag).sum(dim=-1)
    trace_term = (L * L).sum(dim=(-2, -1))
    quad = (mu * mu).sum(dim=-1)
    D = mu.size(-1)
    return 0.5 * (trace_term + quad - D - log_det)


def reparameterize_student_t(mu: torch.Tensor, logvar: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    """Reparameterize z ~ St(μ, σ², ν) via Gamma + Normal.

        V ~ Gamma(ν/2, ν/2)  # scale random source, mean = 1
        eps ~ N(0, I)        # direction random source
        z = μ + σ · eps / sqrt(V)

    Derivation: standard t is Y = Z / sqrt(V/ν) with V ~ χ²_ν. Since
    χ²_ν = Gamma(ν/2, rate=ν/2) (mean=1), and Z/sqrt(V/ν) = Z·sqrt(ν/V)
    is equivalent to Z / sqrt(V) when V already has mean 1, the formula
    z = μ + σ·ε/sqrt(V) with V ~ Gamma(ν/2, ν/2) gives z ~ St(μ, σ², ν).

    E[Var(z|V)] = σ² · E[1/V] = σ² · (ν/2)/(ν/2-1) = σ²·ν/(ν-2) ✓

    Inputs:
      mu, logvar:  [..., latent_dim]
      df:          broadcastable to mu (e.g. [..., latent_dim] or [..., 1])

    Returns: z of the same shape as mu, with z ~ St(μ, σ², df).
    """
    # torch._standard_gamma(α) returns Gamma(α, rate=1) with mean=α. To get
    # Gamma(ν/2, rate=ν/2) with mean=1, scale by (2/ν).
    V = torch._standard_gamma(df / 2.0) * (2.0 / df)
    eps = torch.randn_like(mu)
    z = mu + torch.exp(0.5 * logvar) * eps / torch.sqrt(V)
    return z


def student_t_log_p_matrix(
    z: torch.Tensor,
    user_mu: torch.Tensor, user_log_scale: torch.Tensor, user_df: torch.Tensor,
) -> torch.Tensor:
    """log p_t(z | u) for a per-user diagonal Student-t user prior (z sampled from q).

    One-sample Monte-Carlo estimator of E_q[log p_t(z|u)] when z is a single
    reparameterized sample from the encoder's St_q. Summed over latent dim D.

    Inputs:
      z:               [B, D]   sample drawn from the encoder's St_q
      user_mu:         [U, D]
      user_log_scale:  [U, D]
      user_df:         [U]

    Output: [B, U] — log p_t(z | u), summed over latent dim D.
    """
    z_b = z.unsqueeze(1)                              # [B, 1, D]
    user_mu_b = user_mu.unsqueeze(0)                  # [1, U, D]
    user_log_scale_b = user_log_scale.unsqueeze(0)
    user_df_b = user_df.view(1, -1, 1)                # [1, U, 1]
    log_p_per_dim = _student_t_log_pdf_per_dim(
        z_b, user_mu_b, user_log_scale_b, user_df_b
    )                                                  # [B, U, D]
    return log_p_per_dim.sum(dim=-1)                  # [B, U]


def multivariate_gaussian_kl(
    mu_q: torch.Tensor, L_q: torch.Tensor,
    mu_p: torch.Tensor, L_p: torch.Tensor,
) -> torch.Tensor:
    """KL(N(mu_q, L_q L_q^T) || N(mu_p, L_p L_p^T)) for full covariance.

    L_q, L_p: [*, D, D] lower-triangular Cholesky factors.
    Returns [*] scalar KL per row.

    Uses einsum-based formulation to avoid the O(B*U*D^2) intermediate of
    `solve_triangular` broadcasting. We compute Σ_p⁻¹ = L_p⁻ᵀ L_p⁻¹ once and
    then express trace + Mahalanobis as contractions.
    """
    diff = mu_q - mu_p  # [*] leading, then D

    # Σ_p⁻¹ via bmm (shape-preserving): L_p is lower-triangular so L_p⁻ᵀ is upper-triangular.
    # We compute it by solving L_p @ X = I -> X = L_p⁻¹, then Σ_p⁻¹ = X^T X.
    D = mu_q.size(-1)
    eye = torch.eye(D, dtype=L_p.dtype, device=L_p.device)
    eye_broadcast = eye.expand(*L_p.shape[:-2], D, D)
    Lp_inv = torch.linalg.solve_triangular(L_p, eye_broadcast, upper=False)  # [*, D, D]
    Sigma_p_inv = Lp_inv.transpose(-1, -2) @ Lp_inv  # [*, D, D]

    # Mahalanobis: diff^T Σ_p⁻¹ diff
    quad = torch.einsum("...i,...ij,...j->...", diff, Sigma_p_inv, diff)

    # tr(Σ_p⁻¹ Σ_q) = einsum('...ij,...ji->...', Sigma_p_inv, Sigma_q)
    Sigma_q = L_q @ L_q.transpose(-1, -2)
    trace_term = torch.einsum("...ij,...ji->...", Sigma_p_inv, Sigma_q)

    log_det_q = 2.0 * torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1).clamp_min(1e-6)).sum(dim=-1)
    log_det_p = 2.0 * torch.log(torch.diagonal(L_p, dim1=-2, dim2=-1).clamp_min(1e-6)).sum(dim=-1)
    return 0.5 * (trace_term + quad - D + log_det_q - log_det_p)


def gmm_log_likelihood(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    user_mu: torch.Tensor, user_logvar: torch.Tensor, mix_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute log p(x|u) for a K-component diagonal-GMM user prior.

    Approximates E_q[log p(z|u)] via the Jensen-style logsumexp surrogate
    (standard VAE-GMM choice):

        E_q[log p(z|u)] ≈ logsumexp_k( log π_{u,k} + E_q[log N_k(z)] )
                        = logsumexp_k( log π_{u,k} - KL(q || N_{u,k}) ) + const_i

    Inputs:
      mu_q, logvar_q:    [B, D]   (or [*, D])
      user_mu:           [U, K, D]
      user_logvar:       [U, K, D]
      mix_logits:        [U, K]

    Output: [B, U] (or [*leading, U])  — log p(x|u) up to an i-only constant
                                       that cancels in cross-entropy.
    """
    if mu_q.dim() == 2:
        mu_q_b = mu_q.unsqueeze(1).unsqueeze(1)         # [B, 1, 1, D]
        logvar_q_b = logvar_q.unsqueeze(1).unsqueeze(1)
    else:
        mu_q_b = mu_q
        logvar_q_b = logvar_q
    # Explicit broadcast to [B, U, K, D] instead of relying on
    # diagonal_gaussian_kl's manual expansion (which only handles the [3,2] case).
    user_mu_b = user_mu.unsqueeze(0)                    # [1, U, K, D]
    user_logvar_b = user_logvar.unsqueeze(0)
    var_q = torch.exp(logvar_q_b)
    var_p = torch.exp(user_logvar_b)
    diff = mu_q_b - user_mu_b
    kl_per_dim = 0.5 * (user_logvar_b - logvar_q_b + (var_q + diff ** 2) / var_p - 1.0)
    kl_per_k = kl_per_dim.sum(dim=-1)                   # [B, U, K]
    log_mix = F.log_softmax(mix_logits, dim=-1)         # [U, K]
    log_p_xu = torch.logsumexp(log_mix.unsqueeze(0) - kl_per_k, dim=-1)  # [B, U]
    return log_p_xu


def _student_t_log_pdf_per_dim(
    x: torch.Tensor, mu: torch.Tensor, log_scale: torch.Tensor, df: torch.Tensor
) -> torch.Tensor:
    """Per-dim log p_t(x; mu, sigma, df) for a univariate Student-t.

    All inputs broadcast to the same shape. Returns log p (sum over dims needed
    by the caller).

    log p_t(x) = log Γ((ν+1)/2) - log Γ(ν/2)
                 - 0.5 · log(ν·π·σ²)
                 - ((ν+1)/2) · log(1 + ((x-μ)/σ)² / ν)
    """
    # Use softplus(df) to ensure df > 0; we map to (2, 50) via sigmoid outside.
    sigma2 = torch.exp(2.0 * log_scale)
    z2 = (x - mu) ** 2 / sigma2
    nu = df
    log_norm = (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * (torch.log(nu) + torch.log(torch.tensor(np.pi, device=x.device, dtype=x.dtype)) + 2.0 * log_scale)
    )
    log_kernel = -((nu + 1.0) / 2.0) * torch.log1p(z2 / nu)
    return log_norm + log_kernel


def student_t_log_likelihood(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    user_mu: torch.Tensor, user_log_scale: torch.Tensor, user_df: torch.Tensor,
) -> torch.Tensor:
    """log p(μ_q | u) for a per-user diagonal Student-t user prior (point estimate).

    Inputs:
      mu_q, logvar_q:   [B, D]   encoder mean / log-variance (logvar unused but
                                  kept in signature for branch uniformity)
      user_mu:          [U, D]
      user_log_scale:   [U, D]
      user_df:          [U]

    Output: [B, U] — log p(μ_q | u) for a diagonal Student-t.
    """
    if mu_q.dim() == 2:
        mu_q_b = mu_q.unsqueeze(1)            # [B, 1, D]
    else:
        mu_q_b = mu_q
    user_mu_b = user_mu.unsqueeze(0)          # [1, U, D]
    user_log_scale_b = user_log_scale.unsqueeze(0)
    user_df_b = user_df.unsqueeze(0).unsqueeze(-1)  # [1, U, 1]
    log_p_per_dim = _student_t_log_pdf_per_dim(
        mu_q_b, user_mu_b, user_log_scale_b, user_df_b
    )                                          # [B, U, D]
    return log_p_per_dim.sum(dim=-1)           # [B, U]


def laplace_log_likelihood(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    user_mu: torch.Tensor, user_log_b: torch.Tensor,
) -> torch.Tensor:
    """log p(μ_q | u) for a per-user diagonal Laplace user prior (point estimate).

    Inputs:
      mu_q, logvar_q:   [B, D]   (logvar unused)
      user_mu:          [U, D]
      user_log_b:       [U, D]   b = exp(user_log_b), so log b = user_log_b

    Output: [B, U] — log p(μ_q | u) for a diagonal Laplace.
    """
    if mu_q.dim() == 2:
        mu_q_b = mu_q.unsqueeze(1)
    else:
        mu_q_b = mu_q
    user_mu_b = user_mu.unsqueeze(0)
    user_log_b_b = user_log_b.unsqueeze(0)
    abs_diff = torch.abs(mu_q_b - user_mu_b)
    log_p_per_dim = -abs_diff * torch.exp(-user_log_b_b) - user_log_b_b - np.log(2.0)
    return log_p_per_dim.sum(dim=-1)


def logistic_log_likelihood(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    user_mu: torch.Tensor, user_log_s: torch.Tensor,
) -> torch.Tensor:
    """log p(μ_q | u) for a per-user diagonal Logistic user prior (point estimate).

    Logistic(μ, s) has pdf
        p(x) = exp(-(x-μ)/s) / (s · (1 + exp(-(x-μ)/s))²)
    so log p(x) = -log s - z - 2·log(1 + exp(-z))
    where z = (x-μ)/s. We use F.softplus(-z) for numerical stability.

    Tail decays as exp(-|x|/s) (sub-exponential), in between Gaussian (exp(-x²))
    and Laplace (also exp(-|x|/b) but with sharper peak at 0).

    Inputs:
      mu_q, logvar_q:   [B, D]   (logvar unused)
      user_mu:          [U, D]
      user_log_s:       [U, D]   s = exp(user_log_s), so log s = user_log_s

    Output: [B, U] — log p(μ_q | u) for a diagonal Logistic.
    """
    if mu_q.dim() == 2:
        mu_q_b = mu_q.unsqueeze(1)
    else:
        mu_q_b = mu_q
    user_mu_b = user_mu.unsqueeze(0)
    user_log_s_b = user_log_s.unsqueeze(0)
    z = (mu_q_b - user_mu_b) * torch.exp(-user_log_s_b)
    log_p_per_dim = -user_log_s_b - z - 2.0 * F.softplus(-z)
    return log_p_per_dim.sum(dim=-1)


def student_t_gmm_log_likelihood(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    user_mu: torch.Tensor, user_log_scale: torch.Tensor, user_df: torch.Tensor,
    mix_logits: torch.Tensor,
) -> torch.Tensor:
    """log p(μ_q | u) for a per-user K-component diagonal Student-t mixture.

    Uses point estimate x = μ_q (same approximation style as gmm_log_likelihood):

        log p(x | u) = logsumexp_k( log π_{u,k} + log p_t_k(x | u) )

    Inputs:
      mu_q, logvar_q:   [B, D]
      user_mu:          [U, K, D]
      user_log_scale:   [U, K, D]
      user_df:          [U, K]
      mix_logits:       [U, K]

    Output: [B, U]
    """
    if mu_q.dim() == 2:
        mu_q_b = mu_q.unsqueeze(1).unsqueeze(1)         # [B, 1, 1, D]
    else:
        mu_q_b = mu_q
    user_mu_b = user_mu.unsqueeze(0)                    # [1, U, K, D]
    user_log_scale_b = user_log_scale.unsqueeze(0)
    user_df_b = user_df.unsqueeze(0).unsqueeze(-1)      # [1, U, K, 1]
    log_p_per_dim = _student_t_log_pdf_per_dim(
        mu_q_b, user_mu_b, user_log_scale_b, user_df_b
    )                                                   # [B, U, K, D]
    log_p_per_k = log_p_per_dim.sum(dim=-1)              # [B, U, K]
    log_mix = F.log_softmax(mix_logits, dim=-1)          # [U, K]
    return torch.logsumexp(log_mix.unsqueeze(0) + log_p_per_k, dim=-1)  # [B, U]


class SentenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        reconstruction = self.decoder(mu)
        return mu, logvar, reconstruction


class SentenceEncoderStudentT(nn.Module):
    """SentenceEncoder with diagonal Student-t posterior (μ, σ, ν per-dim).

    Outputs: (mu, logvar, df, reconstruction) where df ∈ (2, 50) via sigmoid.
    Decoder consumes sampled z (reparameterized via Gamma + Normal) so the
    gradient flows through the stochastic layer — standard VAE reparameterization
    extended to Student-t.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.df_raw_head = nn.Linear(hidden_dim, latent_dim)   # per-dim df via sigmoid
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        df = 2.0 + 48.0 * torch.sigmoid(self.df_raw_head(hidden))  # df ∈ (2, 50)
        z = reparameterize_student_t(mu, logvar, df)
        reconstruction = self.decoder(z)
        return mu, logvar, df, reconstruction


class UserDistributionTable(nn.Module):
    def __init__(self, num_users: int, latent_dim: int):
        super().__init__()
        self.user_mu = nn.Embedding(num_users, latent_dim)
        self.user_logvar = nn.Embedding(num_users, latent_dim)
        nn.init.zeros_(self.user_mu.weight)
        nn.init.zeros_(self.user_logvar.weight)

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_mu(user_index), self.user_logvar(user_index)


class UserDistributionTableGMM(nn.Module):
    """Per-user K-component diagonal-Gaussian mixture.

    Stores (mu, logvar, mix_logits) for K components per user. The mixture
    weights are obtained via softmax over mix_logits along the K axis.

    init: component 0 is all-zero (matches legacy UserDistributionTable init),
          component k>0 has its mu perturbed by N(0, 0.01^2) to break symmetry;
          all logvars are zero and mix_logits are zero (uniform π).
    """

    def __init__(self, num_users: int, latent_dim: int, num_components: int):
        super().__init__()
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.user_mu = nn.Parameter(torch.zeros(num_users, num_components, latent_dim))
        self.user_logvar = nn.Parameter(torch.zeros(num_users, num_components, latent_dim))
        self.mix_logits = nn.Parameter(torch.zeros(num_users, num_components))
        if num_components >= 2:
            with torch.no_grad():
                self.user_mu[:, 1:, :].add_(torch.randn_like(self.user_mu[:, 1:, :]) * 0.01)

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.user_mu[user_index]
        logvar = self.user_logvar[user_index]
        mix_logits = self.mix_logits[user_index]
        return mu, logvar, mix_logits


class UserDistributionTableStudentT(nn.Module):
    """Per-user diagonal Student-t distribution.

    Parameters: user_mu [U, D], user_log_scale [U, D] (σ = exp(log_scale)),
    user_df_raw [U] mapped to df ∈ (2, 50) via sigmoid:
        df = 2 + 48 * sigmoid(user_df_raw)

    The (2, 50) range keeps ν > 2 (finite variance) and bounded above to avoid
    numerical overflow in log(1 + z²/ν).

    init: mu=0, log_scale=0 (σ=1), df_raw=0 (df=26 — close to Gaussian for
    warm start), to match the legacy single-Gaussian init.
    """

    def __init__(self, num_users: int, latent_dim: int):
        super().__init__()
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.user_mu = nn.Parameter(torch.zeros(num_users, latent_dim))
        self.user_log_scale = nn.Parameter(torch.zeros(num_users, latent_dim))
        self.user_df_raw = nn.Parameter(torch.zeros(num_users))

    def _df(self) -> torch.Tensor:
        return 2.0 + 48.0 * torch.sigmoid(self.user_df_raw)

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_mu[user_index], self.user_log_scale[user_index], self._df()[user_index]

    def df_all(self) -> torch.Tensor:
        return self._df()


class UserDistributionTableLaplace(nn.Module):
    """Per-user diagonal Laplace distribution.

    Parameters: user_mu [U, D], user_log_b [U, D] (b = exp(log_b)).
    Laplace has sharper peak than Gaussian (sub-Gaussian exponential tail).

    init: mu=0, log_b=0 (b=1), matching the legacy single-Gaussian init.
    """

    def __init__(self, num_users: int, latent_dim: int):
        super().__init__()
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.user_mu = nn.Parameter(torch.zeros(num_users, latent_dim))
        self.user_log_b = nn.Parameter(torch.zeros(num_users, latent_dim))

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_mu[user_index], self.user_log_b[user_index]


class UserDistributionTableLogistic(nn.Module):
    """Per-user diagonal Logistic distribution.

    Parameters: user_mu [U, D], user_log_s [U, D] (s = exp(log_s)).
    Logistic tail decays as exp(-|x|/s), in between Gaussian (exp(-x²)) and
    Laplace (also exp(-|x|/b) but with sharper peak at 0).

    init: mu=0, log_s=0 (s=1), matching the legacy single-Gaussian init.
    """

    def __init__(self, num_users: int, latent_dim: int):
        super().__init__()
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.user_mu = nn.Parameter(torch.zeros(num_users, latent_dim))
        self.user_log_s = nn.Parameter(torch.zeros(num_users, latent_dim))

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_mu[user_index], self.user_log_s[user_index]


class UserDistributionTableStudentTGMM(nn.Module):
    """Per-user K-component diagonal Student-t mixture.

    Parameters: user_mu [U, K, D], user_log_scale [U, K, D], user_df_raw [U, K]
    (sigmoid → df ∈ (2, 50)), mix_logits [U, K].
    """

    def __init__(self, num_users: int, latent_dim: int, num_components: int):
        super().__init__()
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.user_mu = nn.Parameter(torch.zeros(num_users, num_components, latent_dim))
        self.user_log_scale = nn.Parameter(torch.zeros(num_users, num_components, latent_dim))
        self.user_df_raw = nn.Parameter(torch.zeros(num_users, num_components))
        self.mix_logits = nn.Parameter(torch.zeros(num_users, num_components))
        if num_components >= 2:
            with torch.no_grad():
                self.user_mu[:, 1:, :].add_(torch.randn_like(self.user_mu[:, 1:, :]) * 0.01)

    def _df(self) -> torch.Tensor:
        return 2.0 + 48.0 * torch.sigmoid(self.user_df_raw)

    def forward(
        self, user_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.user_mu[user_index]
        log_scale = self.user_log_scale[user_index]
        df = self._df()[user_index]
        mix_logits = self.mix_logits[user_index]
        return mu, log_scale, df, mix_logits

    def df_all(self) -> torch.Tensor:
        return self._df()


class SentenceEncoderFull(nn.Module):
    """SentenceEncoder with full-covariance latent (Cholesky L)."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        n_tril = latent_dim * (latent_dim + 1) // 2
        self.tril_head = nn.Linear(hidden_dim, n_tril)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _build_L(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.tril_head(hidden)
        latent_dim = self.latent_dim
        L = torch.zeros(*raw.shape[:-1], latent_dim, latent_dim, device=raw.device, dtype=raw.dtype)
        tril_indices = torch.tril_indices(latent_dim, latent_dim, device=raw.device)
        L[..., tril_indices[0], tril_indices[1]] = raw
        diag_idx = torch.arange(latent_dim, device=raw.device)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + 1e-4
        return L

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(x)
        mu = self.mu_head(hidden)
        L = self._build_L(hidden)
        reconstruction = self.decoder(mu)
        return mu, L, reconstruction


class UserDistributionTableFull(nn.Module):
    """UserDistributionTable with full-covariance latent (per-user Cholesky L)."""

    def __init__(self, num_users: int, latent_dim: int):
        super().__init__()
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.user_mu = nn.Embedding(num_users, latent_dim)
        self.user_L_raw = nn.Parameter(torch.empty(num_users, latent_dim, latent_dim))
        nn.init.zeros_(self.user_mu.weight)
        nn.init.zeros_(self.user_L_raw)
        with torch.no_grad():
            eye = torch.eye(latent_dim)
            for i in range(num_users):
                self.user_L_raw[i].copy_(eye)

    def get_L(self) -> torch.Tensor:
        latent_dim = self.latent_dim
        L = self.user_L_raw.clone()
        diag_idx = torch.arange(latent_dim, device=L.device)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + 1e-4
        return L

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.user_mu(user_index)
        L_all = self.get_L()
        L = L_all[user_index]
        return mu, L


def normalize_text(text: str) -> str:
    return re.sub(r"\\s+", " ", text.strip())


def extract_sentences_from_review_text(text: str) -> list[str]:
    nlp = load_spacy_model()
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        normalized = normalize_text(sent.text)
        if normalized:
            sentences.append(normalized)
    return sentences


def load_filtered_user_reviews() -> list[dict]:
    if not REVIEW_SOURCE_FILE.exists():
        raise FileNotFoundError(f"缺少用户评论文件: {REVIEW_SOURCE_FILE}")
    payload = load_json(REVIEW_SOURCE_FILE)
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"{REVIEW_SOURCE_FILE} 必须是非空列表")
        return payload
    if isinstance(payload, dict):
        users = payload.get("users")
        if not isinstance(users, list) or not users:
            raise ValueError(f"{REVIEW_SOURCE_FILE} 的 users 字段必须是非空列表")
        return users
    raise ValueError(f"{REVIEW_SOURCE_FILE} 必须是列表或包含 users 的字典")


def filter_users_with_duplicate_reviews() -> None:
    """过滤掉有重复 target_reviews 的用户，避免句子提取时产生重复句子导致 holdout 不足"""
    global REVIEW_SOURCE_FILE

    filtered_file = INPUT_DIR / "stage1_filtered_users_reviews_dedup.json"
    if filtered_file.exists():
        log(f"已存在去重后的评论文件，跳过过滤: {filtered_file}")
        REVIEW_SOURCE_FILE = filtered_file
        return

    log(f"开始过滤重复 target_reviews 用户")
    payload = load_json(REVIEW_SOURCE_FILE)

    if isinstance(payload, list):
        original_users = payload
    elif isinstance(payload, dict):
        original_users = payload.get("users", [])
    else:
        raise ValueError(f"{REVIEW_SOURCE_FILE} 格式错误")

    filtered_users = []
    duplicate_users = []

    for user in original_users:
        user_id = user.get("user_id")
        reviews = user.get("reviews") or user.get("results", [])
        has_duplicate = False

        for review in reviews:
            target_reviews = review.get("target_reviews", [])
            if len(target_reviews) > 1 and len(set(target_reviews)) == 1:
                has_duplicate = True
                break

        if has_duplicate:
            duplicate_users.append(user_id)
        else:
            filtered_users.append(user)

    log(f"过滤完成: 原始用户 {len(original_users)}, 去重后用户 {len(filtered_users)}, 移除重复用户 {len(duplicate_users)}")

    # 保存去重后的数据
    if isinstance(payload, dict):
        filtered_payload = {"users": filtered_users}
    else:
        filtered_payload = filtered_users

    if isinstance(filtered_payload, dict):
        filtered_file.write_text(json.dumps(filtered_payload, ensure_ascii=False), encoding="utf-8")
    else:
        with filtered_file.open("w", encoding="utf-8") as f:
            json.dump(filtered_payload, f, ensure_ascii=False)

    REVIEW_SOURCE_FILE = filtered_file
    log(f"已保存去重后的评论文件: {REVIEW_SOURCE_FILE}")


def extract_first_twenty_sentences_for_users(user_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    log(
        f"开始为用户抽取最多 {TOTAL_SENTENCES_PER_USER} 个句子"
        f"（前 {TRAIN_SENTENCES_PER_USER} 个为训练，最多 {MAX_HOLDOUT_SENTENCES_PER_USER} 个为 holdout）"
    )
    kept_rows: list[dict] = []
    excluded_rows: list[dict] = []
    total_users = len(user_rows)
    for user_offset, user_row in enumerate(user_rows, start=1):
        user_id = user_row.get("user_id")
        reviews = user_row.get("reviews")
        if reviews is None:
            reviews = user_row.get("results")
        if user_id is None or not isinstance(reviews, list):
            raise ValueError(f"用户记录缺少 user_id 或 reviews/results: index={user_offset}")
        collected: list[dict] = []
        seen_sentence_texts: set[str] = set()  # 用于去重
        for review_index, review_row in enumerate(reviews):
            review_texts = review_row.get("target_reviews")
            if review_texts is not None:
                if not isinstance(review_texts, list):
                    raise ValueError(f"target_reviews 必须是列表: user_id={user_id}, review_index={review_index}")
                candidate_texts = [text for text in review_texts if text]
            else:
                single_review_text = review_row.get("text")
                candidate_texts = [single_review_text] if single_review_text else []
            if not candidate_texts:
                continue
            for source_text in candidate_texts:
                for sentence_index, sentence_text in enumerate(extract_sentences_from_review_text(source_text)):
                    # 去重：如果已经见过相同句子文本，则跳过
                    if sentence_text in seen_sentence_texts:
                        continue
                    seen_sentence_texts.add(sentence_text)
                    collected.append(
                        {
                            "user_id": user_id,
                            "review_index": review_index,
                            "sentence_index": sentence_index,
                            "sentence_text": sentence_text,
                            "word_count": len(sentence_text.split()),
                        }
                    )
                    if len(collected) == TOTAL_SENTENCES_PER_USER:
                        break
                if len(collected) == TOTAL_SENTENCES_PER_USER:
                    break
            if len(collected) == TOTAL_SENTENCES_PER_USER:
                break

        if len(collected) < MIN_SENTENCES_PER_USER:
            log(
                f"用户 {user_offset}/{total_users}: {user_id}, 句子数不足 {MIN_SENTENCES_PER_USER}, "
                f"实际={len(collected)}，从本方法中过滤"
            )
            excluded_rows.append(
                {
                    "user_id": user_id,
                    "available_sentence_count": int(len(collected)),
                    "required_sentence_count": MIN_SENTENCES_PER_USER,
                }
            )
            continue

        log(
            f"已完成用户 {user_offset}/{total_users}: {user_id}, "
            f"句子数={len(collected)}（holdout={len(collected) - TRAIN_SENTENCES_PER_USER}）"
        )
        kept_rows.extend(collected)
    return kept_rows, excluded_rows


def build_sentence_feature_rows(sentence_rows: list[dict]) -> tuple[list[dict], list[str]]:
    log("开始提取评论句法特征")
    nlp = load_spacy_model()
    feature_names: list[str] | None = None
    enriched_rows: list[dict] = []
    total = len(sentence_rows)
    last_user_id: str | None = None
    processed_users: int = 0

    # 使用 nlp.pipe() 批量处理，比逐句处理快 2-3 倍
    batch_size = 500
    texts = [row["sentence_text"] for row in sentence_rows]

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_texts = texts[batch_start:batch_end]
        batch_docs = list(nlp.pipe(batch_texts, batch_size=batch_size))

        for row_idx_offset, (row, doc) in enumerate(zip(sentence_rows[batch_start:batch_end], batch_docs)):
            row_idx = batch_start + row_idx_offset + 1
            user_id = row.get("user_id")
            if user_id != last_user_id:
                last_user_id = user_id
                processed_users += 1
                if processed_users % 500 == 0 or processed_users <= 10:
                    log(f"提取评论句法特征进度: 用户 {processed_users}, 句子 {row_idx}/{total}")
            extracted = extract_clause_features_from_doc(doc, row["sentence_text"])
            if feature_names is None:
                feature_names = list(extracted.keys())
            elif list(extracted.keys()) != feature_names:
                raise ValueError(f"评论句法特征字段顺序不一致: user_id={row['user_id']}")
            enriched = dict(row)
            enriched["features"] = extracted
            enriched_rows.append(enriched)
    if feature_names is None:
        raise ValueError("没有可用评论句法特征")
    return enriched_rows, feature_names


def load_candidate_query_rows() -> list[dict]:
    if CANDIDATE_QUERY_FILE.exists():
        log(f"开始读取候选 query 特征文件: {CANDIDATE_QUERY_FILE}")
        rows = []
        with CANDIDATE_QUERY_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"{CANDIDATE_QUERY_FILE} 为空")
        return rows

    if not RAW_CANDIDATE_QUERY_FILE.exists():
        raise FileNotFoundError(f"缺少原始 10 候选 query 文件: {RAW_CANDIDATE_QUERY_FILE}")
    log(f"未发现候选特征文件，开始从原始 10 候选 query 生成: {RAW_CANDIDATE_QUERY_FILE}")
    raw_rows = load_json(RAW_CANDIDATE_QUERY_FILE)
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{RAW_CANDIDATE_QUERY_FILE} 必须是非空列表")
    rows = build_candidate_feature_rows_from_raw_query_file(raw_rows)
    write_jsonl(CANDIDATE_QUERY_FILE, rows)
    log(f"候选特征文件已写入: {CANDIDATE_QUERY_FILE}")
    return rows


def build_candidate_feature_rows_from_raw_query_file(raw_rows: list[dict]) -> list[dict]:
    nlp = load_spacy_model()
    total_users = len(raw_rows)
    rows: list[dict] = []
    for user_offset, row in enumerate(raw_rows, start=1):
        user_id = row.get("user_id")
        asin = row.get("asin")
        candidates = row.get("expression_style_queries")
        if user_id is None or asin is None or not isinstance(candidates, list):
            raise ValueError(f"原始 10 候选 query 记录缺少 user_id / asin / expression_style_queries: index={user_offset}")
        for candidate_index, candidate in enumerate(candidates, start=1):
            query_text = candidate.get("query")
            if not query_text:
                raise ValueError(f"候选 query 缺少 query 文本: user_id={user_id}, candidate_index={candidate_index}")
            doc = nlp(query_text)
            extracted = extract_clause_features_from_doc(doc, query_text)
            rows.append(
                {
                    "user_id": user_id,
                    "asin": asin,
                    "candidate_index": candidate_index,
                    "query": query_text,
                    "word_count": int(candidate.get("word_count", len(query_text.split()))),
                    "target_depth": candidate.get("target_depth"),
                    "user_avg_depth": candidate.get("user_avg_depth"),
                    "attrs_used": candidate.get("attrs_used"),
                    "features": extracted,
                }
            )
        log(f"已完成用户 {user_offset}/{total_users}: {user_id}, 10 候选已处理")
    return rows


def build_training_dataset(sentence_rows: list[dict], feature_names: list[str]) -> tuple[list[str], dict]:
    feature_matrix = np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in sentence_rows],
        dtype=np.float64,
    )
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in sentence_rows:
        grouped_rows[row["user_id"]].append(row)
    user_ids = sorted(grouped_rows.keys())
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}

    user_indices: list[int] = []
    train_mask: list[bool] = []
    holdout_mask: list[bool] = []
    for row in sentence_rows:
        user_rows = grouped_rows[row["user_id"]]
        per_user_offset = user_rows.index(row)
        user_indices.append(user_to_index[row["user_id"]])
        train_mask.append(per_user_offset < TRAIN_SENTENCES_PER_USER)
        holdout_mask.append(per_user_offset >= TRAIN_SENTENCES_PER_USER)

    dataset = {
        "scaler": scaler,
        "feature_names": feature_names,
        "sentence_rows": sentence_rows,
        "scaled_features": scaled_features,
        "feature_matrix": feature_matrix,
        "user_indices": np.asarray(user_indices, dtype=np.int64),
        "train_mask": np.asarray(train_mask, dtype=bool),
        "holdout_mask": np.asarray(holdout_mask, dtype=bool),
        "user_to_index": user_to_index,
        "grouped_rows": grouped_rows,
    }
    return user_ids, dataset


def train_vades_user_distribution_model(user_ids: list[str], dataset: dict, feature_names: list[str]) -> tuple[SentenceEncoder, nn.Module, dict]:
    input_dim = len(feature_names)
    if COVARIANCE_MODE == "full":
        encoder = SentenceEncoderFull(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        user_table = UserDistributionTableFull(num_users=len(user_ids), latent_dim=LATENT_DIM).to(DEVICE)
    elif COVARIANCE_MODE == "diagonal_gmm":
        if ENCODER_DIST == "student_t":
            encoder = SentenceEncoderStudentT(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        else:
            encoder = SentenceEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        user_table = UserDistributionTableGMM(num_users=len(user_ids), latent_dim=LATENT_DIM, num_components=GMM_COMPONENTS).to(DEVICE)
    elif COVARIANCE_MODE == "diagonal_student_t":
        if ENCODER_DIST == "student_t":
            encoder = SentenceEncoderStudentT(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        else:
            encoder = SentenceEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        user_table = UserDistributionTableStudentT(num_users=len(user_ids), latent_dim=LATENT_DIM).to(DEVICE)
    elif COVARIANCE_MODE == "diagonal_laplace":
        if ENCODER_DIST == "student_t":
            encoder = SentenceEncoderStudentT(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        else:
            encoder = SentenceEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        user_table = UserDistributionTableLaplace(num_users=len(user_ids), latent_dim=LATENT_DIM).to(DEVICE)
    elif COVARIANCE_MODE == "diagonal_logistic":
        if ENCODER_DIST == "student_t":
            encoder = SentenceEncoderStudentT(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        else:
            encoder = SentenceEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        user_table = UserDistributionTableLogistic(num_users=len(user_ids), latent_dim=LATENT_DIM).to(DEVICE)
    elif COVARIANCE_MODE == "diagonal_student_t_gmm":
        if ENCODER_DIST == "student_t":
            encoder = SentenceEncoderStudentT(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        else:
            encoder = SentenceEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        user_table = UserDistributionTableStudentTGMM(
            num_users=len(user_ids), latent_dim=LATENT_DIM, num_components=GMM_COMPONENTS
        ).to(DEVICE)
    else:
        if ENCODER_DIST == "student_t":
            encoder = SentenceEncoderStudentT(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        else:
            encoder = SentenceEncoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
        user_table = UserDistributionTable(num_users=len(user_ids), latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(user_table.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler() if DEVICE.type == "cuda" else None

    features_tensor = torch.tensor(dataset["scaled_features"], dtype=torch.float32, device=DEVICE)
    user_index_tensor = torch.tensor(dataset["user_indices"], dtype=torch.long, device=DEVICE)
    train_indices = np.flatnonzero(dataset["train_mask"])
    train_indices_tensor = torch.tensor(train_indices, dtype=torch.long, device=DEVICE)

    epoch_rows: list[dict] = []
    encoder.train()
    user_table.train()

    # Compile model for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            encoder = torch.compile(encoder)
            user_table = torch.compile(user_table)
            log("模型已编译 (torch.compile)")
        except Exception as e:
            log(f"torch.compile 失败: {e}")

    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0

    pbar = tqdm(total=EPOCHS, desc="训练 VADES 模型", unit="epoch")
    for epoch in range(1, EPOCHS + 1):
        shuffled_indices = train_indices.copy()
        np.random.shuffle(shuffled_indices)
        batch_metrics = []
        pbar.set_postfix({"loss": "-", "epoch": epoch})
        for start in tqdm(range(0, len(shuffled_indices), BATCH_SIZE), desc=f"Epoch {epoch}/{EPOCHS}", unit="batch", leave=False):
            batch_indices = shuffled_indices[start : start + BATCH_SIZE]
            batch_idx_tensor = torch.tensor(batch_indices, dtype=torch.long, device=DEVICE)
            batch_features = features_tensor[batch_idx_tensor]
            batch_user_indices = user_index_tensor[batch_idx_tensor]

            if ENCODER_DIST == "student_t" and COVARIANCE_MODE != "full":
                # Encoder is Student-t → forward returns (mu, logvar, df, reconstruction).
                sent_mu, sent_logvar, sent_df, reconstruction = encoder(batch_features)
            else:
                sent_mu, sent_dispersion, reconstruction = encoder(batch_features)
                if COVARIANCE_MODE == "full":
                    sent_logvar = 2.0 * torch.log(torch.diagonal(sent_dispersion, dim1=-2, dim2=-1).clamp_min(1e-6))
                else:
                    sent_logvar = sent_dispersion
            if COVARIANCE_MODE == "diagonal_gmm":
                # gmm_log_likelihood returns log p(x|u) ≈ logsumexp_k(log π - KL(q||N_k))
                # which is what we need for cross-entropy. NO outer negation (single-Gaussian's
                # -KL trick is already absorbed by the per-component -KL inside logsumexp).
                # Access params directly (works for both raw and torch.compile-wrapped module).
                user_mu_all = user_table.user_mu
                user_logvar_all = user_table.user_logvar
                mix_logits_all = user_table.mix_logits
                kl_matrix_tensor = gmm_log_likelihood(
                    sent_mu, sent_logvar,
                    user_mu_all, user_logvar_all, mix_logits_all,
                )
            elif COVARIANCE_MODE == "diagonal_student_t":
                if ENCODER_DIST == "student_t":
                    # Strict t-t VIB: sample z from encoder's St_q, then score
                    # log p_t(z | u) under the per-user Student-t prior. This is a
                    # one-sample MC estimator of E_q[log p_t(z|u)]; the gradient
                    # through z is unbiased (reparam trick). kl_matrix_tensor
                    # is a log p matrix, NOT a KL — use cross_entropy directly
                    # (no negation). The digamma-based closed form was tried but
                    # was wrong (negative KL on test cases).
                    z = reparameterize_student_t(sent_mu, sent_logvar, sent_df)  # [B, D]
                    user_mu_all = user_table.user_mu
                    user_log_scale_all = user_table.user_log_scale
                    user_df_all = user_table.df_all()
                    kl_matrix_tensor = student_t_log_p_matrix(
                        z, user_mu_all, user_log_scale_all, user_df_all
                    )                                   # → [B, U]
                else:
                    # Gaussian encoder → point-estimate log p_t(μ_q | u); cross_entropy directly.
                    user_mu_all = user_table.user_mu
                    user_log_scale_all = user_table.user_log_scale
                    user_df_all = user_table.df_all()
                    kl_matrix_tensor = student_t_log_likelihood(
                        sent_mu, sent_logvar,
                        user_mu_all, user_log_scale_all, user_df_all,
                    )
            elif COVARIANCE_MODE == "diagonal_laplace":
                user_mu_all = user_table.user_mu
                user_log_b_all = user_table.user_log_b
                kl_matrix_tensor = laplace_log_likelihood(
                    sent_mu, sent_logvar,
                    user_mu_all, user_log_b_all,
                )
            elif COVARIANCE_MODE == "diagonal_logistic":
                user_mu_all = user_table.user_mu
                user_log_s_all = user_table.user_log_s
                kl_matrix_tensor = logistic_log_likelihood(
                    sent_mu, sent_logvar,
                    user_mu_all, user_log_s_all,
                )
            elif COVARIANCE_MODE == "diagonal_student_t_gmm":
                # Same Jensen-style logsumexp structure as gmm_log_likelihood,
                # but per-component log p_t(μ_q) instead of -KL.
                user_mu_all = user_table.user_mu
                user_log_scale_all = user_table.user_log_scale
                user_df_all = user_table.df_all()
                mix_logits_all = user_table.mix_logits
                kl_matrix_tensor = student_t_gmm_log_likelihood(
                    sent_mu, sent_logvar,
                    user_mu_all, user_log_scale_all, user_df_all, mix_logits_all,
                )
            elif COVARIANCE_MODE == "full":
                user_mu_all = user_table.user_mu.weight
                user_dispersion_all = user_table.get_L()
                # einsum-based KL avoids the B*U*D^2 broadcast of L_p.
                num_users = user_mu_all.size(0)
                diff = sent_mu.unsqueeze(1) - user_mu_all.unsqueeze(0)  # [B, U, D]
                Sigma_p_inv = torch.cholesky_inverse(user_dispersion_all)  # [U, D, D]
                v = torch.einsum('uij,buj->bui', Sigma_p_inv, diff)  # [B, U, D]
                quad = (v * v).sum(dim=-1)  # [B, U]
                Sigma_q = sent_dispersion @ sent_dispersion.transpose(-1, -2)  # [B, D, D]
                trace_term = torch.einsum('uij,bji->bu', Sigma_p_inv, Sigma_q)  # [B, U]
                log_det_q = 2.0 * torch.log(
                    torch.diagonal(sent_dispersion, dim1=-2, dim2=-1).clamp_min(1e-6)
                ).sum(dim=-1).unsqueeze(1)  # [B, 1]
                log_det_p = 2.0 * torch.log(
                    torch.diagonal(user_dispersion_all, dim1=-2, dim2=-1).clamp_min(1e-6)
                ).sum(dim=-1).unsqueeze(0)  # [1, U]
                D = sent_mu.size(-1)
                kl_matrix_tensor = 0.5 * (trace_term + quad - D + log_det_q + log_det_p)
            else:
                user_mu_all = user_table.user_mu.weight
                user_logvar_all = user_table.user_logvar.weight
                kl_matrix_tensor = diagonal_gaussian_kl(
                    sent_mu.unsqueeze(1),      # [batch_size, 1, latent_dim]
                    sent_logvar.unsqueeze(1),  # [batch_size, 1, latent_dim]
                    user_mu_all.unsqueeze(0), # [1, num_users, latent_dim]
                    user_logvar_all.unsqueeze(0)  # [1, num_users, latent_dim]
                )
            # kl_matrix_tensor convention:
            #   - log-likelihood (higher = better) for: non-Gaussian prior + Gaussian encoder,
            #     or strict t-t VIB (Student-t prior + Student-t encoder, one-MC-sample log p).
            #   - KL (lower = better) for: Gaussian-Gaussian, or Gaussian encoder + full cov.
            if (COVARIANCE_MODE in {"diagonal_gmm", "diagonal_student_t", "diagonal_laplace", "diagonal_student_t_gmm", "diagonal_logistic"}
                    and ENCODER_DIST == "gaussian"):
                user_match_loss = F.cross_entropy(kl_matrix_tensor, batch_user_indices)
            elif (COVARIANCE_MODE == "diagonal_student_t" and ENCODER_DIST == "student_t"):
                # t-t strict VIB: kl_matrix_tensor is one-MC-sample log p_t(z|u).
                user_match_loss = F.cross_entropy(kl_matrix_tensor, batch_user_indices)
            else:
                # Gaussian-Gaussian or full-cov paths: kl_matrix is KL; use -KL as logit.
                user_match_loss = F.cross_entropy(-kl_matrix_tensor, batch_user_indices)

            style_recon_loss = F.mse_loss(reconstruction, batch_features)
            # VADES feature loss (α≈0.9 mode): 强制 latent dim i 对齐到 features dim i
            latent_align_loss = F.mse_loss(sent_mu, batch_features)
            if COVARIANCE_MODE == "full":
                sent_kl_loss = multivariate_standard_normal_kl(sent_mu, sent_dispersion).mean()
            elif ENCODER_DIST == "student_t":
                # Strict t-t VIB: KL(St_q || N(0, I)) via Gaussian equivalent
                # proxy. St(μ, σ², ν) has variance σ²·ν/(ν-2); for ν>2 use this.
                # sent_df is per-dim [B, D], so log_nu_factor is also [B, D] —
                # no unsqueeze needed (unlike user_prior_kl_loss where user_df
                # is per-user [B] and we do unsqueeze(-1)).
                nu = sent_df
                nu_factor = nu / (nu - 2.0).clamp_min(1e-3)
                log_nu_factor = torch.log(nu_factor.clamp_min(1.0))  # [B, D]
                equiv_logvar = sent_logvar + log_nu_factor  # [B, D]
                sent_kl_loss = standard_normal_kl(sent_mu, equiv_logvar).mean()
            else:
                sent_kl_loss = standard_normal_kl(sent_mu, sent_logvar).mean()
            if COVARIANCE_MODE == "diagonal_gmm":
                batch_user_mu_k, batch_user_logvar_k, batch_mix_logits = user_table(batch_user_indices)
                # Σ_k π_{u,k} · KL(N_{u,k} || N(0,I)) — an upper bound on KL(GMM || N(0,I))
                # that ignores cross-component terms; weak regularizer, closed-form.
                per_component_kl = standard_normal_kl(batch_user_mu_k, batch_user_logvar_k)  # [B, K]
                mix_probs = F.softmax(batch_mix_logits, dim=-1)  # [B, K]
                user_prior_kl_loss = (mix_probs * per_component_kl).sum(dim=-1).mean()
            elif COVARIANCE_MODE == "diagonal_student_t":
                # Weak regularizer: KL of the equivalent Gaussian (μ, σ²) against N(0, I).
                # t(μ, σ, ν) has Var = σ²·ν/(ν-2); for ν>2 use the variance; otherwise σ² is
                # an OK proxy. The point of this loss is gentle shrinkage — exact form is
                # not critical (weight is 0.02).
                batch_user_mu, batch_user_log_scale, batch_user_df = user_table(batch_user_indices)
                # Convert to log-variance of equivalent Gaussian: log(σ² · ν/(ν-2))
                # df is [B]; log_scale is [B, D] — unsqueeze df to broadcast cleanly.
                nu = batch_user_df
                nu_factor = nu / (nu - 2.0).clamp_min(1e-3)
                log_nu_factor = torch.log(nu_factor.clamp_min(1.0)).unsqueeze(-1)  # [B, 1]
                equiv_logvar = 2.0 * batch_user_log_scale + log_nu_factor  # [B, D]
                user_prior_kl_loss = standard_normal_kl(batch_user_mu, equiv_logvar).mean()
            elif COVARIANCE_MODE == "diagonal_laplace":
                # Laplace L(μ, b) has Var = 2b²; use equivalent Gaussian for the regularizer.
                batch_user_mu, batch_user_log_b = user_table(batch_user_indices)
                equiv_logvar = 2.0 * batch_user_log_b + np.log(2.0)
                user_prior_kl_loss = standard_normal_kl(batch_user_mu, equiv_logvar).mean()
            elif COVARIANCE_MODE == "diagonal_logistic":
                # Logistic L(μ, s) has Var = π²·s²/3; use equivalent Gaussian for the regularizer.
                batch_user_mu, batch_user_log_s = user_table(batch_user_indices)
                equiv_logvar = 2.0 * batch_user_log_s + np.log(np.pi**2 / 3.0)
                user_prior_kl_loss = standard_normal_kl(batch_user_mu, equiv_logvar).mean()
            elif COVARIANCE_MODE == "diagonal_student_t_gmm":
                # Per-component equivalent-Gaussian KL, mix-weighted.
                batch_user_mu_k, batch_user_log_scale_k, batch_user_df_k, batch_mix_logits = user_table(batch_user_indices)
                nu = batch_user_df_k
                nu_factor = nu / (nu - 2.0).clamp_min(1e-3)
                # df is [B, K]; log_scale is [B, K, D] — unsqueeze to broadcast.
                log_nu_factor = torch.log(nu_factor.clamp_min(1.0)).unsqueeze(-1)  # [B, K, 1]
                equiv_logvar_k = 2.0 * batch_user_log_scale_k + log_nu_factor  # [B, K, D]
                per_component_kl = standard_normal_kl(batch_user_mu_k, equiv_logvar_k)  # [B, K]
                mix_probs = F.softmax(batch_mix_logits, dim=-1)  # [B, K]
                user_prior_kl_loss = (mix_probs * per_component_kl).sum(dim=-1).mean()
            else:
                batch_user_mu, batch_user_dispersion = user_table(batch_user_indices)
                if COVARIANCE_MODE == "full":
                    user_prior_kl_loss = multivariate_standard_normal_kl(batch_user_mu, batch_user_dispersion).mean()
                else:
                    user_prior_kl_loss = standard_normal_kl(batch_user_mu, batch_user_dispersion).mean()

            total_loss = (
                USER_MATCH_WEIGHT * user_match_loss
                + STYLE_RECON_WEIGHT * style_recon_loss
                + SENT_KL_WEIGHT * sent_kl_loss
                + USER_PRIOR_KL_WEIGHT * user_prior_kl_loss
                + LATENT_ALIGN_WEIGHT * latent_align_loss
            )

            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            batch_metrics.append(
                {
                    "total_loss": float(total_loss.item()),
                    "user_match_loss": float(user_match_loss.item()),
                    "style_recon_loss": float(style_recon_loss.item()),
                    "sent_kl_loss": float(sent_kl_loss.item()),
                    "user_prior_kl_loss": float(user_prior_kl_loss.item()),
                    "latent_align_loss": float(latent_align_loss.item()),
                }
            )

        epoch_summary = {
            "epoch": epoch,
            "batch_count": int(len(batch_metrics)),
            "total_loss_mean": float(np.mean([row["total_loss"] for row in batch_metrics])),
            "user_match_loss_mean": float(np.mean([row["user_match_loss"] for row in batch_metrics])),
            "style_recon_loss_mean": float(np.mean([row["style_recon_loss"] for row in batch_metrics])),
            "sent_kl_loss_mean": float(np.mean([row["sent_kl_loss"] for row in batch_metrics])),
            "user_prior_kl_loss_mean": float(np.mean([row["user_prior_kl_loss"] for row in batch_metrics])),
            "latent_align_loss_mean": float(np.mean([row["latent_align_loss"] for row in batch_metrics])),
        }
        epoch_rows.append(epoch_summary)
        pbar.set_postfix({
            "epoch": epoch,
            "loss": f'{epoch_summary["total_loss_mean"]:.4f}'
        })
        pbar.update(1)
        # Print epoch details
        from datetime import datetime; ts = datetime.now().strftime('%H:%M:%S'); print(f"[{ts}] [Epoch {epoch}/{EPOCHS}] loss={epoch_summary['total_loss_mean']:.4f} | "
              f"user_match={epoch_summary['user_match_loss_mean']:.4f} | "
              f"style_recon={epoch_summary['style_recon_loss_mean']:.4f} | "
              f"sent_kl={epoch_summary['sent_kl_loss_mean']:.4f} | "
              f"user_prior_kl={epoch_summary['user_prior_kl_loss_mean']:.4f} | "
              f"latent_align={epoch_summary['latent_align_loss_mean']:.4f}", flush=True)
        
        # Early stopping check
        current_loss = epoch_summary['total_loss_mean']
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                log(f"早停触发！连续 {EARLY_STOP_PATIENCE} 个 epoch 损失未下降")
                log(f"最佳 loss: {best_loss:.4f}")
                break
    pbar.close()
    # 保存模型
    MODEL_SAVE_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    encoder_path = MODEL_SAVE_DIR / "vades_encoder.pt"
    user_table_path = MODEL_SAVE_DIR / "vades_user_table.pt"
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": HIDDEN_DIM,
        "latent_dim": LATENT_DIM,
        "covariance_mode": COVARIANCE_MODE,
        "encoder_dist": ENCODER_DIST,
    }, encoder_path)
    torch.save({
        "model_state_dict": user_table.state_dict(),
        "num_users": len(user_ids),
        "latent_dim": LATENT_DIM,
        "covariance_mode": COVARIANCE_MODE,
        **({"gmm_components": GMM_COMPONENTS} if COVARIANCE_MODE in {"diagonal_gmm", "diagonal_student_t_gmm"} else {}),
    }, user_table_path)
    log(f"模型已保存: {encoder_path}, {user_table_path}")

    return encoder, user_table, {"epochs": epoch_rows}


def infer_user_sentence_distributions(
    encoder: SentenceEncoder,
    user_table,
    user_ids: list[str],
    dataset: dict,
) -> tuple[list[dict], list[dict]]:
    feature_names = dataset["feature_names"]
    sentence_rows = dataset["sentence_rows"]
    feature_tensor = torch.tensor(dataset["scaled_features"], dtype=torch.float32, device=DEVICE)
    encoder.eval()
    user_table.eval()
    with torch.no_grad():
        encoder_out = encoder(feature_tensor)
        if len(encoder_out) == 4:
            sent_mu, sent_dispersion, _, _ = encoder_out  # StudentT encoder: (mu, logvar, df, recon)
        else:
            sent_mu, sent_dispersion, _ = encoder_out
        if COVARIANCE_MODE == "diagonal_gmm":
            user_mu_weight = user_table.user_mu.detach().cpu().numpy()
            user_logvar_weight = user_table.user_logvar.detach().cpu().numpy()
            mix_logits_weight = user_table.mix_logits.detach().cpu().numpy()
            user_mu_weight_tensor = user_table.user_mu.detach()
            user_logvar_weight_tensor = user_table.user_logvar.detach()
            mix_logits_weight_tensor = user_table.mix_logits.detach()
        elif COVARIANCE_MODE == "diagonal_student_t":
            user_mu_weight = user_table.user_mu.detach().cpu().numpy()
            user_log_scale_weight = user_table.user_log_scale.detach().cpu().numpy()
            user_df_weight = user_table.df_all().detach().cpu().numpy()
            user_mu_weight_tensor = user_table.user_mu.detach()
            user_log_scale_weight_tensor = user_table.user_log_scale.detach()
            user_df_weight_tensor = user_table.df_all().detach()
        elif COVARIANCE_MODE == "diagonal_laplace":
            user_mu_weight = user_table.user_mu.detach().cpu().numpy()
            user_log_b_weight = user_table.user_log_b.detach().cpu().numpy()
            user_mu_weight_tensor = user_table.user_mu.detach()
            user_log_b_weight_tensor = user_table.user_log_b.detach()
        elif COVARIANCE_MODE == "diagonal_logistic":
            user_mu_weight = user_table.user_mu.detach().cpu().numpy()
            user_log_s_weight = user_table.user_log_s.detach().cpu().numpy()
            user_mu_weight_tensor = user_table.user_mu.detach()
            user_log_s_weight_tensor = user_table.user_log_s.detach()
        elif COVARIANCE_MODE == "diagonal_student_t_gmm":
            user_mu_weight = user_table.user_mu.detach().cpu().numpy()
            user_log_scale_weight = user_table.user_log_scale.detach().cpu().numpy()
            user_df_weight = user_table.df_all().detach().cpu().numpy()
            mix_logits_weight = user_table.mix_logits.detach().cpu().numpy()
            user_mu_weight_tensor = user_table.user_mu.detach()
            user_log_scale_weight_tensor = user_table.user_log_scale.detach()
            user_df_weight_tensor = user_table.df_all().detach()
            mix_logits_weight_tensor = user_table.mix_logits.detach()
        elif COVARIANCE_MODE == "full":
            user_mu_weight = user_table.user_mu.weight.detach().cpu().numpy()
            sent_logvar_t = 2.0 * torch.log(torch.diagonal(sent_dispersion, dim1=-2, dim2=-1).clamp_min(1e-6))
            user_dispersion_all = user_table.get_L()  # [num_users, D, D]
            user_diag_logvar_t = 2.0 * torch.log(torch.diagonal(user_dispersion_all, dim1=-2, dim2=-1).clamp_min(1e-6))
        else:
            user_mu_weight = user_table.user_mu.weight.detach().cpu().numpy()
            sent_logvar_t = sent_dispersion
            user_logvar_weight = user_table.user_logvar.weight.detach().cpu().numpy()
            user_logvar_weight_tensor = user_table.user_logvar.weight.detach()

    sentence_output_rows: list[dict] = []
    user_scores: dict[str, list[float]] = defaultdict(list)
    user_profile_rows: list[dict] = []
    if COVARIANCE_MODE == "diagonal_gmm":
        user_mu_weight_tensor = user_table.user_mu.detach()
        user_logvar_weight_tensor = user_table.user_logvar.detach()
        mix_logits_weight_tensor = user_table.mix_logits.detach()
    elif COVARIANCE_MODE in {"diagonal_student_t", "diagonal_laplace", "diagonal_student_t_gmm", "diagonal_logistic"}:
        user_mu_weight_tensor = user_table.user_mu.detach()
    else:
        user_mu_weight_tensor = user_table.user_mu.weight.detach()
    if COVARIANCE_MODE == "full":
        user_dispersion_t = user_dispersion_all.detach()

    for idx, row in enumerate(sentence_rows):
        user_id = row["user_id"]
        user_index = dataset["user_to_index"][user_id]
        sentence_output_rows.append(
            {
                "user_id": user_id,
                "review_index": row["review_index"],
                "sentence_index": row["sentence_index"],
                "sentence_text": row["sentence_text"],
                "word_count": row["word_count"],
                "features": row["features"],
            }
        )
        if COVARIANCE_MODE == "diagonal_gmm":
            # log-likelihood up to i-only const; for storage as a "score", we negate so
            # lower = better fit (matches the existing KL convention in the JSONL).
            log_p_xu = gmm_log_likelihood(
                sent_mu[idx].unsqueeze(0),
                sent_dispersion[idx].unsqueeze(0),
                user_mu_weight_tensor[user_index].unsqueeze(0),
                user_logvar_weight_tensor[user_index].unsqueeze(0),
                mix_logits_weight_tensor[user_index].unsqueeze(0),
            ).item()
            score = -log_p_xu
        elif COVARIANCE_MODE == "diagonal_student_t":
            log_p_xu = student_t_log_likelihood(
                sent_mu[idx].unsqueeze(0),
                sent_dispersion[idx].unsqueeze(0),
                user_mu_weight_tensor[user_index].unsqueeze(0),
                user_log_scale_weight_tensor[user_index].unsqueeze(0),
                user_df_weight_tensor[user_index].unsqueeze(0),
            ).item()
            score = -log_p_xu
        elif COVARIANCE_MODE == "diagonal_laplace":
            log_p_xu = laplace_log_likelihood(
                sent_mu[idx].unsqueeze(0),
                sent_dispersion[idx].unsqueeze(0),
                user_mu_weight_tensor[user_index].unsqueeze(0),
                user_log_b_weight_tensor[user_index].unsqueeze(0),
            ).item()
            score = -log_p_xu
        elif COVARIANCE_MODE == "diagonal_logistic":
            log_p_xu = logistic_log_likelihood(
                sent_mu[idx].unsqueeze(0),
                sent_dispersion[idx].unsqueeze(0),
                user_mu_weight_tensor[user_index].unsqueeze(0),
                user_log_s_weight_tensor[user_index].unsqueeze(0),
            ).item()
            score = -log_p_xu
        elif COVARIANCE_MODE == "diagonal_student_t_gmm":
            log_p_xu = student_t_gmm_log_likelihood(
                sent_mu[idx].unsqueeze(0),
                sent_dispersion[idx].unsqueeze(0),
                user_mu_weight_tensor[user_index].unsqueeze(0),
                user_log_scale_weight_tensor[user_index].unsqueeze(0),
                user_df_weight_tensor[user_index].unsqueeze(0),
                mix_logits_weight_tensor[user_index].unsqueeze(0),
            ).item()
            score = -log_p_xu
        elif COVARIANCE_MODE == "full":
            score = multivariate_gaussian_kl(
                sent_mu[idx].unsqueeze(0),
                sent_dispersion[idx].unsqueeze(0),
                user_mu_weight_tensor[user_index].unsqueeze(0),
                user_dispersion_t[user_index].unsqueeze(0),
            ).item()
        else:
            score = diagonal_gaussian_kl(
                sent_mu[idx].unsqueeze(0),
                sent_logvar_t[idx].unsqueeze(0),
                user_mu_weight_tensor[user_index].unsqueeze(0),
                user_logvar_weight_tensor[user_index].unsqueeze(0),
            ).item()
        user_scores[user_id].append(float(score))

    for user_id in user_ids:
        user_index = dataset["user_to_index"][user_id]
        score_array = np.asarray(user_scores[user_id], dtype=np.float64)
        if COVARIANCE_MODE == "diagonal_gmm":
            profile = {
                "user_id": user_id,
                "component_mus": user_mu_weight[user_index].tolist(),
                "component_logvars": user_logvar_weight[user_index].tolist(),
                "component_vars": np.exp(user_logvar_weight[user_index]).tolist(),
                "mix_logits": mix_logits_weight[user_index].tolist(),
                "kl_score_summary": summarize_array(score_array),
            }
        elif COVARIANCE_MODE == "diagonal_student_t":
            user_logvar_diag = (2.0 * user_log_scale_weight[user_index]).tolist()
            user_var_diag = np.exp(2.0 * user_log_scale_weight[user_index]).tolist()
            profile = {
                "user_id": user_id,
                "user_mu": user_mu_weight[user_index].tolist(),
                "user_log_scale": user_log_scale_weight[user_index].tolist(),
                "user_df": float(user_df_weight[user_index]),
                "user_logvar": user_logvar_diag,
                "user_var": user_var_diag,
                "kl_score_summary": summarize_array(score_array),
            }
        elif COVARIANCE_MODE == "diagonal_laplace":
            user_logvar_diag = (2.0 * user_log_b_weight[user_index] + np.log(2.0)).tolist()
            user_var_diag = (2.0 * np.exp(2.0 * user_log_b_weight[user_index])).tolist()
            profile = {
                "user_id": user_id,
                "user_mu": user_mu_weight[user_index].tolist(),
                "user_log_b": user_log_b_weight[user_index].tolist(),
                "user_logvar": user_logvar_diag,
                "user_var": user_var_diag,
                "kl_score_summary": summarize_array(score_array),
            }
        elif COVARIANCE_MODE == "diagonal_logistic":
            user_logvar_diag = (2.0 * user_log_s_weight[user_index] + np.log(np.pi**2 / 3.0)).tolist()
            user_var_diag = ((np.pi**2 / 3.0) * np.exp(2.0 * user_log_s_weight[user_index])).tolist()
            profile = {
                "user_id": user_id,
                "user_mu": user_mu_weight[user_index].tolist(),
                "user_log_s": user_log_s_weight[user_index].tolist(),
                "user_logvar": user_logvar_diag,
                "user_var": user_var_diag,
                "kl_score_summary": summarize_array(score_array),
            }
        elif COVARIANCE_MODE == "diagonal_student_t_gmm":
            user_logvar_diag = (2.0 * user_log_scale_weight[user_index]).tolist()
            user_var_diag = np.exp(2.0 * user_log_scale_weight[user_index]).tolist()
            profile = {
                "user_id": user_id,
                "component_mus": user_mu_weight[user_index].tolist(),
                "component_log_scales": user_log_scale_weight[user_index].tolist(),
                "component_dfs": user_df_weight[user_index].tolist(),
                "component_logvars": user_logvar_diag,
                "component_vars": user_var_diag,
                "mix_logits": mix_logits_weight[user_index].tolist(),
                "kl_score_summary": summarize_array(score_array),
            }
        elif COVARIANCE_MODE == "full":
            diag_logvar = user_diag_logvar_t[user_index].detach().cpu().numpy()
            user_logvar_diag = diag_logvar.tolist()
            user_var_diag = np.exp(diag_logvar).tolist()
            profile = {
                "user_id": user_id,
                "user_mu": user_mu_weight[user_index].tolist(),
                "user_logvar": user_logvar_diag,
                "user_var": user_var_diag,
                "kl_score_summary": summarize_array(score_array),
            }
        else:
            user_logvar_diag = user_logvar_weight[user_index].tolist()
            user_var_diag = np.exp(user_logvar_weight[user_index]).tolist()
            profile = {
                "user_id": user_id,
                "user_mu": user_mu_weight[user_index].tolist(),
                "user_logvar": user_logvar_diag,
                "user_var": user_var_diag,
                "kl_score_summary": summarize_array(score_array),
            }
        user_profile_rows.append(profile)
    return sentence_output_rows, user_profile_rows


def calibrate_absolute_threshold_with_unseen_holdout(
    encoder: SentenceEncoder,
    user_table,
    user_ids: list[str],
    dataset: dict,
) -> dict[str, float]:
    feature_tensor = torch.tensor(dataset["scaled_features"], dtype=torch.float32, device=DEVICE)
    encoder.eval()
    user_table.eval()
    thresholds: dict[str, float] = {}
    holdout_indices = np.flatnonzero(dataset["holdout_mask"])
    train_indices = np.flatnonzero(dataset["train_mask"])
    with torch.no_grad():
        encoder_out = encoder(feature_tensor)
        if len(encoder_out) == 4:
            sent_mu, sent_dispersion, _, _ = encoder_out  # StudentT encoder
        else:
            sent_mu, sent_dispersion, _ = encoder_out
        if COVARIANCE_MODE == "diagonal_gmm":
            user_mu_weight = user_table.user_mu
            user_logvar_weight = user_table.user_logvar
            mix_logits_weight = user_table.mix_logits
        elif COVARIANCE_MODE == "diagonal_student_t":
            user_mu_weight = user_table.user_mu
            user_log_scale_weight = user_table.user_log_scale
            user_df_weight = user_table.df_all()
        elif COVARIANCE_MODE == "diagonal_laplace":
            user_mu_weight = user_table.user_mu
            user_log_b_weight = user_table.user_log_b
        elif COVARIANCE_MODE == "diagonal_logistic":
            user_mu_weight = user_table.user_mu
            user_log_s_weight = user_table.user_log_s
        elif COVARIANCE_MODE == "diagonal_student_t_gmm":
            user_mu_weight = user_table.user_mu
            user_log_scale_weight = user_table.user_log_scale
            user_df_weight = user_table.df_all()
            mix_logits_weight = user_table.mix_logits
        elif COVARIANCE_MODE == "full":
            user_mu_weight = user_table.user_mu.weight
            sent_logvar_t = 2.0 * torch.log(torch.diagonal(sent_dispersion, dim1=-2, dim2=-1).clamp_min(1e-6))
            user_dispersion_t = user_table.get_L()
        else:
            user_mu_weight = user_table.user_mu.weight
            user_logvar_weight = user_table.user_logvar.weight
            sent_logvar_t = sent_dispersion
        grouped_holdout_scores: dict[str, list[float]] = defaultdict(list)
        for idx in holdout_indices:
            user_id = dataset["sentence_rows"][idx]["user_id"]
            user_index = dataset["user_to_index"][user_id]
            if COVARIANCE_MODE == "diagonal_gmm":
                log_p_xu = gmm_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_logvar_weight[user_index].unsqueeze(0),
                    mix_logits_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "diagonal_student_t":
                log_p_xu = student_t_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_log_scale_weight[user_index].unsqueeze(0),
                    user_df_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "diagonal_laplace":
                log_p_xu = laplace_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_log_b_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "diagonal_logistic":
                log_p_xu = logistic_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_log_s_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "diagonal_student_t_gmm":
                log_p_xu = student_t_gmm_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_log_scale_weight[user_index].unsqueeze(0),
                    user_df_weight[user_index].unsqueeze(0),
                    mix_logits_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "full":
                score = multivariate_gaussian_kl(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_dispersion_t[user_index].unsqueeze(0),
                ).item()
            else:
                score = diagonal_gaussian_kl(
                    sent_mu[idx].unsqueeze(0),
                    sent_logvar_t[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_logvar_weight[user_index].unsqueeze(0),
                ).item()
            grouped_holdout_scores[user_id].append(float(score))

        grouped_train_scores: dict[str, list[float]] = defaultdict(list)
        for idx in train_indices:
            user_id = dataset["sentence_rows"][idx]["user_id"]
            user_index = dataset["user_to_index"][user_id]
            if COVARIANCE_MODE == "diagonal_gmm":
                log_p_xu = gmm_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_logvar_weight[user_index].unsqueeze(0),
                    mix_logits_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "diagonal_student_t":
                log_p_xu = student_t_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_log_scale_weight[user_index].unsqueeze(0),
                    user_df_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "diagonal_laplace":
                log_p_xu = laplace_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_log_b_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "diagonal_logistic":
                log_p_xu = logistic_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_log_s_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "diagonal_student_t_gmm":
                log_p_xu = student_t_gmm_log_likelihood(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_log_scale_weight[user_index].unsqueeze(0),
                    user_df_weight[user_index].unsqueeze(0),
                    mix_logits_weight[user_index].unsqueeze(0),
                ).item()
                score = -log_p_xu
            elif COVARIANCE_MODE == "full":
                score = multivariate_gaussian_kl(
                    sent_mu[idx].unsqueeze(0),
                    sent_dispersion[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_dispersion_t[user_index].unsqueeze(0),
                ).item()
            else:
                score = diagonal_gaussian_kl(
                    sent_mu[idx].unsqueeze(0),
                    sent_logvar_t[idx].unsqueeze(0),
                    user_mu_weight[user_index].unsqueeze(0),
                    user_logvar_weight[user_index].unsqueeze(0),
                ).item()
            grouped_train_scores[user_id].append(float(score))

    for user_id in user_ids:
        holdout_values = grouped_holdout_scores.get(user_id)
        if holdout_values is None or len(holdout_values) == 0:
            raise ValueError(f"用户 {user_id} 缺少 holdout 句子得分")
        thresholds[user_id] = float(np.quantile(np.asarray(holdout_values, dtype=np.float64), ABS_THRESHOLD_QUANTILE))
    return thresholds


def rank_and_select_queries(
    encoder: SentenceEncoder,
    user_table,
    dataset: dict,
    candidate_rows: list[dict],
    user_ids: list[str],
    user_profile_rows: list[dict],
    abs_thresholds: dict[str, float],
) -> tuple[list[dict], list[dict], list[dict]]:
    feature_names = dataset["feature_names"]
    feature_scaler: StandardScaler = dataset["scaler"]

    grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        grouped_candidates[row["user_id"]].append(row)

    user_profile_by_id = {row["user_id"]: row for row in user_profile_rows}
    selected_rows: list[dict] = []
    rejected_rows: list[dict] = []
    query_output_rows: list[dict] = []

    encoder.eval()
    user_table.eval()
    with torch.no_grad():
        for user_id in user_ids:
            candidates = grouped_candidates.get(user_id)
            if not candidates:
                raise ValueError(f"用户 {user_id} 没有候选 query")
            user_profile = user_profile_by_id.get(user_id)
            if user_profile is None:
                raise ValueError(f"用户 {user_id} 缺少 user profile")
            user_index = dataset["user_to_index"][user_id]
            if COVARIANCE_MODE == "diagonal_gmm":
                user_mu, user_logvar, user_mix_logits = user_table(torch.tensor([user_index], dtype=torch.long, device=DEVICE))
            elif COVARIANCE_MODE == "diagonal_student_t":
                user_mu, user_log_scale, user_df = user_table(torch.tensor([user_index], dtype=torch.long, device=DEVICE))
            elif COVARIANCE_MODE == "diagonal_laplace":
                user_mu, user_log_b = user_table(torch.tensor([user_index], dtype=torch.long, device=DEVICE))
            elif COVARIANCE_MODE == "diagonal_logistic":
                user_mu, user_log_s = user_table(torch.tensor([user_index], dtype=torch.long, device=DEVICE))
            elif COVARIANCE_MODE == "diagonal_student_t_gmm":
                user_mu, user_log_scale, user_df, user_mix_logits = user_table(torch.tensor([user_index], dtype=torch.long, device=DEVICE))
            else:
                user_mu, user_dispersion = user_table(torch.tensor([user_index], dtype=torch.long, device=DEVICE))
                if COVARIANCE_MODE == "full":
                    user_logvar_t = 2.0 * torch.log(torch.diagonal(user_dispersion, dim1=-2, dim2=-1).clamp_min(1e-6))
                else:
                    user_logvar_t = user_dispersion

            candidate_records: list[dict] = []
            for candidate in candidates:
                feature_vector = np.asarray(
                    [float(candidate["features"][name]) for name in feature_names],
                    dtype=np.float64,
                ).reshape(1, -1)
                scaled_vector = feature_scaler.transform(feature_vector)
                feature_tensor = torch.tensor(scaled_vector, dtype=torch.float32, device=DEVICE)
                encoder_out = encoder(feature_tensor)
                if len(encoder_out) == 4:
                    query_mu, query_dispersion, _, _ = encoder_out  # StudentT encoder
                else:
                    query_mu, query_dispersion, _ = encoder_out
                if COVARIANCE_MODE == "diagonal_gmm":
                    log_p_xu = gmm_log_likelihood(
                        query_mu, query_dispersion,
                        user_mu, user_logvar, user_mix_logits,
                    ).item()
                    range_score = -log_p_xu  # lower = better fit, matches existing convention
                    query_logvar_t = query_dispersion  # store as proxy; readers should use score, not this
                elif COVARIANCE_MODE == "diagonal_student_t":
                    log_p_xu = student_t_log_likelihood(
                        query_mu, query_dispersion,
                        user_mu, user_log_scale, user_df,
                    ).item()
                    range_score = -log_p_xu
                    query_logvar_t = query_dispersion
                elif COVARIANCE_MODE == "diagonal_laplace":
                    log_p_xu = laplace_log_likelihood(
                        query_mu, query_dispersion,
                        user_mu, user_log_b,
                    ).item()
                    range_score = -log_p_xu
                    query_logvar_t = query_dispersion
                elif COVARIANCE_MODE == "diagonal_logistic":
                    log_p_xu = logistic_log_likelihood(
                        query_mu, query_dispersion,
                        user_mu, user_log_s,
                    ).item()
                    range_score = -log_p_xu
                    query_logvar_t = query_dispersion
                elif COVARIANCE_MODE == "diagonal_student_t_gmm":
                    log_p_xu = student_t_gmm_log_likelihood(
                        query_mu, query_dispersion,
                        user_mu, user_log_scale, user_df, user_mix_logits,
                    ).item()
                    range_score = -log_p_xu
                    query_logvar_t = query_dispersion
                elif COVARIANCE_MODE == "full":
                    query_logvar_t = 2.0 * torch.log(torch.diagonal(query_dispersion, dim1=-2, dim2=-1).clamp_min(1e-6))
                    range_score = multivariate_gaussian_kl(query_mu, query_dispersion, user_mu, user_dispersion).item()
                else:
                    query_logvar_t = query_dispersion
                    range_score = diagonal_gaussian_kl(query_mu, query_logvar_t, user_mu, user_logvar_t).item()
                abs_threshold = abs_thresholds[user_id]
                passes_abs_threshold = range_score <= abs_threshold
                candidate_records.append(
                    {
                        **candidate,
                        "query_mu": query_mu.squeeze(0).detach().cpu().numpy().tolist(),
                        "query_logvar": query_logvar_t.squeeze(0).detach().cpu().numpy().tolist(),
                        "range_score": float(range_score),
                        "passes_abs_threshold": bool(passes_abs_threshold),
                    }
                )

            passed = [row for row in candidate_records if row["passes_abs_threshold"]]
            if passed:
                best = min(passed, key=lambda row: row["range_score"])
                selected_rows.append(best)
                query_output_rows.append(
                    {
                        "user_id": best["user_id"],
                        "asin": best["asin"],
                        "expression_style_query": {
                            "query": best["query"],
                            "word_count": int(best["word_count"]),
                            "target_depth": best["target_depth"],
                            "actual_depth": None,
                            "user_avg_depth": best["user_avg_depth"],
                            "attrs_used": best["attrs_used"],
                            "accepted_candidate_index": int(best["candidate_index"]),
                            "candidate_count": int(len(candidate_records)),
                        },
                    }
                )
            else:
                best = min(candidate_records, key=lambda row: row["range_score"])
                rejected_rows.append(best)
    # Regeneration tracking: paper §2.2 requires 10 rounds × 10 candidates max.
    # In current implementation, Stage 04 generates 10 candidates at once (no true regeneration).
    # Track regeneration_needed flag for future Stage 04 regeneration implementation.
    regeneration_info: dict[str, dict] = {}
    for user_id in user_ids:
        candidates = grouped_candidates.get(user_id, [])
        candidate_records_for_user = [
            r for r in candidate_records if r["user_id"] == user_id
        ]
        passed_user = [r for r in candidate_records_for_user if r["passes_abs_threshold"]]
        regeneration_info[user_id] = {
            "regeneration_needed": len(passed_user) == 0,
            "rounds_needed": 1 if passed_user else REGENERATION_MAX_ROUNDS,
            "passes_threshold": len(passed_user) > 0,
            "candidate_count": len(candidates),
        }
    return selected_rows, rejected_rows, query_output_rows, regeneration_info


def build_summary(
    user_ids: list[str],
    sentence_rows: list[dict],
    excluded_rows: list[dict],
    feature_names: list[str],
    training_info: dict,
    user_profile_rows: list[dict],
    selected_rows: list[dict],
    rejected_rows: list[dict],
    query_output_rows: list[dict],
    regeneration_info: dict[str, dict] | None = None,
) -> dict:
    kl_thresholds = [row["kl_score_summary"]["q75"] for row in user_profile_rows]
    selected_scores = np.asarray([row["range_score"] for row in selected_rows], dtype=np.float64) if selected_rows else np.asarray([0.0])
    rejected_scores = np.asarray([row["range_score"] for row in rejected_rows], dtype=np.float64) if rejected_rows else np.asarray([0.0])
    return {
        "category": CATEGORY,
        "output_tag": OUTPUT_TAG,
        "device": str(DEVICE),
        "feature_names": feature_names,
        "user_count_total": int(len(user_ids) + len(excluded_rows)),
        "user_count_trained": int(len(user_ids)),
        "user_count_excluded": int(len(excluded_rows)),
        "sentence_count_total": int(len(sentence_rows)),
        "train_sentences_per_user": TRAIN_SENTENCES_PER_USER,
        "holdout_sentences_per_user": MAX_HOLDOUT_SENTENCES_PER_USER,
        "candidate_count_total": int(len(selected_rows) + len(rejected_rows)),
        "selected_query_count": int(len(selected_rows)),
        "rejected_query_count": int(len(rejected_rows)),
        "selected_user_count": int(len(query_output_rows)),
        "training": {
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "user_match_weight": USER_MATCH_WEIGHT,
            "style_recon_weight": STYLE_RECON_WEIGHT,
            "sent_kl_weight": SENT_KL_WEIGHT,
            "user_prior_kl_weight": USER_PRIOR_KL_WEIGHT,
            "latent_align_weight": LATENT_ALIGN_WEIGHT,
            "epoch_summaries": training_info["epochs"],
        },
        "holdout_abs_threshold_quantile": ABS_THRESHOLD_QUANTILE,
        "user_kl_q75_summary": summarize_array(np.asarray(kl_thresholds, dtype=np.float64)),
        "selected_range_score_summary": summarize_array(selected_scores),
        "rejected_range_score_summary": summarize_array(rejected_scores),
        "summary_file": str(SUMMARY_FILE),
        "detail_file": str(DETAIL_FILE),
        "user_profile_file": str(USER_PROFILE_FILE),
        "sentence_file": str(SENTENCE_FILE),
        "excluded_user_file": str(EXCLUDED_USER_FILE),
        "selected_record_file": str(SELECTED_RECORD_FILE),
        "rejected_record_file": str(REJECTED_RECORD_FILE),
        "query_file": str(QUERY_FILE),
        "regeneration": {
            "regeneration_max_rounds": REGENERATION_MAX_ROUNDS,
            "candidates_per_round": CANDIDATES_PER_ROUND,
            "note": "Stage 04 generates 10 candidates at once (no true LLM-based regeneration). "
                    "regeneration_needed=True means all 10 candidates failed 95th percentile filtering. "
                    "To fully match paper §2.2, Stage 04 must implement LLM-based regeneration loop.",
            "regeneration_info": regeneration_info or {},
            "users_needing_regeneration": int(
                sum(1 for v in (regeneration_info or {}).values() if v.get("regeneration_needed", False))
            ),
        },
    }


def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUERY_FILE.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    global DEVICE
    DEVICE = infer_device()
    set_random_seed(SEED)
    ensure_directories()
    log(f"运行设备: {DEVICE}")
    log(f"训练批大小: {BATCH_SIZE}")
    log("开始读取候选 query")
    candidate_rows = load_candidate_query_rows()
    candidate_user_ids = {row["user_id"] for row in candidate_rows}
    user_rows = load_filtered_user_reviews()
    user_rows = [row for row in user_rows if row.get("user_id") in candidate_user_ids]
    if MAX_USERS_OVERRIDE is not None:
        max_users = int(MAX_USERS_OVERRIDE)
        if max_users <= 0:
            raise ValueError("VADES_MAX_USERS 必须是正整数")
        user_rows = user_rows[:max_users]
        log(f"启用用户数限制: {len(user_rows)}")

    # 尝试从缓存加载已提取的句子
    sentence_rows: list[dict] = []
    excluded_rows: list[dict] = []
    if SENTENCE_EXTRACT_CACHE_FILE.exists():
        log(f"从缓存加载已提取的句子: {SENTENCE_EXTRACT_CACHE_FILE}")
        cache_data = load_json(SENTENCE_EXTRACT_CACHE_FILE)
        if isinstance(cache_data, dict) and "kept_rows" in cache_data and "excluded_rows" in cache_data:
            sentence_rows = cache_data["kept_rows"]
            excluded_rows = cache_data["excluded_rows"]
            log(f"已从缓存加载 {len(sentence_rows)} 条句子，{len(excluded_rows)} 个排除用户")
        else:
            log(f"缓存格式错误，将重新提取句子")
    else:
        sentence_rows, excluded_rows = extract_first_twenty_sentences_for_users(user_rows)
        # 保存到缓存
        log(f"保存句子提取结果到缓存: {SENTENCE_EXTRACT_CACHE_FILE}")
        SENTENCE_EXTRACT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SENTENCE_EXTRACT_CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump({"kept_rows": sentence_rows, "excluded_rows": excluded_rows}, f, ensure_ascii=False)

    enriched_sentence_rows, feature_names = build_sentence_feature_rows(sentence_rows)
    user_ids, dataset = build_training_dataset(enriched_sentence_rows, feature_names)
    encoder, user_table, training_info = train_vades_user_distribution_model(user_ids, dataset, feature_names)
    sentence_output_rows, user_profile_rows = infer_user_sentence_distributions(encoder, user_table, user_ids, dataset)
    abs_thresholds = calibrate_absolute_threshold_with_unseen_holdout(encoder, user_table, user_ids, dataset)
    selected_rows, rejected_rows, query_output_rows, regeneration_info = rank_and_select_queries(
        encoder=encoder,
        user_table=user_table,
        dataset=dataset,
        candidate_rows=candidate_rows,
        user_ids=user_ids,
        user_profile_rows=user_profile_rows,
        abs_thresholds=abs_thresholds,
    )

    write_jsonl(SENTENCE_FILE, sentence_output_rows)
    write_jsonl(EXCLUDED_USER_FILE, excluded_rows)
    write_jsonl(USER_PROFILE_FILE, user_profile_rows)
    write_jsonl(SELECTED_RECORD_FILE, selected_rows)
    write_jsonl(REJECTED_RECORD_FILE, rejected_rows)
    QUERY_FILE.write_text(json.dumps(query_output_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = build_summary(
        user_ids=user_ids,
        sentence_rows=sentence_output_rows,
        excluded_rows=excluded_rows,
        feature_names=feature_names,
        training_info=training_info,
        user_profile_rows=user_profile_rows,
        selected_rows=selected_rows,
        rejected_rows=rejected_rows,
        query_output_rows=query_output_rows,
        regeneration_info=regeneration_info,
    )
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(DETAIL_FILE, training_info["epochs"])
    if SKIP_POST_CLUSTERING:
        log("按配置跳过训练后的 query GMM 聚类与 retrieval attach")
    else:
        run_query_gmm_pipeline(
            category=CATEGORY,
            query_file=QUERY_FILE,
            write_back_to_query_file=False,
            attach_retrieval=True,
        )
    log(f"已写入 summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()