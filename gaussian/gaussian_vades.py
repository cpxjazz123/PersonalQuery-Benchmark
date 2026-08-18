#!/usr/bin/env python3
"""Gaussian VADES — train / probe / validate / compare / smoke.

Single entry point for the VADES (Variational Autoencoder for Diversity, Embedding,
Style) user-style-vector pipeline. Consolidates 5 historical scripts into one
file with subcommand dispatching:

    python gaussian_vades.py            # default: train (主训练 + 排序)
    python gaussian_vades.py train      # 同上
    python gaussian_vades.py probe      # SVR 10-fold probe eval
    python gaussian_vades.py validate   # 高斯假设检验 + Q-Q 图
    python gaussian_vades.py compare    # 6 prior 对比 (Gaussian vs non-Gaussian)
    python gaussian_vades.py smoke      # 3 个非高斯模式 forward+backward smoke

All configuration via env vars (see SECTION 1). No CLI business args.

Source consolidation:
  - gaussian/common/train_vades_lite_sentence_latent_threshold.py (2143 lines)
  - gaussian/common/evaluate_vades_style_vector_probe_svr10fold.py (245 lines)
  - gaussian/archive/10_validate_gaussian_assumption.py (748 lines)
  - gaussian/archive/10_compare_prior_gaussian_vs_nongaussian.py (520 lines)
  - gaussian/archive/_smoke_test_3_nongaussian.py (89 lines)

Fixes applied during merge:
  - inline log() replaces missing common_utils.log
  - unified REPO_ROOT = /fs04/ar57/wenyu/PersoanlQuery (was wrong in compare_prior)
  - dropped stale sys.path inserts referencing 10_complexity_analysis/common
"""

# ============================================================
# SECTION 0: Imports (shared infrastructure)
# ============================================================
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


# ============================================================
# SECTION 1: Shared constants + inline log
# ============================================================

def log(message: str) -> None:
    """Inline log function (replaces missing common_utils.log)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


# 统一 REPO_ROOT = /fs04/ar57/wenyu/PersoanlQuery (采用原 train 脚本的值, 修复 compare_prior 的路径 bug)
REPO_ROOT = Path(os.environ.get("PQ_REPO_ROOT", "/fs04/ar57/wenyu/PersoanlQuery"))
CATEGORY = os.environ.get("PQ_CATEGORY", os.environ.get("CATEGORY_OVERRIDE", "Baby_Products"))
SEED = 42
DEVICE: Optional[torch.device] = None  # set in main_train()

# 把 syntactic_analysis 加入 sys.path 以便 import extract_clause_features_single_query
_SYNTAX_DIR = REPO_ROOT / "syntactic_analysis"
if str(_SYNTAX_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNTAX_DIR))

# (1) 训练配置常量 (保留所有 VADES_* env vars)
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
        f"VADES_COVARIANCE_MODE 必须是 {sorted(VALID_COVARIANCE_MODES)} 之一, 得到 {COVARIANCE_MODE}"
    )
REGENERATION_MAX_ROUNDS = 10
CANDIDATES_PER_ROUND = 10
ENCODER_DIST = os.environ.get("VADES_ENCODER_DIST", "gaussian")
VALID_ENCODER_DISTS = {"gaussian", "student_t"}
if ENCODER_DIST not in VALID_ENCODER_DISTS:
    raise ValueError(
        f"VADES_ENCODER_DIST 必须是 {sorted(VALID_ENCODER_DISTS)} 之一, 得到 {ENCODER_DIST}"
    )
# Encoder=student_t requires closed-form Student-t KL; force diagonal_student_t.
if ENCODER_DIST == "student_t" and COVARIANCE_MODE != "diagonal_student_t":
    raise ValueError(
        f"VADES_ENCODER_DIST=student_t 强制要求 VADES_COVARIANCE_MODE=diagonal_student_t, 得到 {COVARIANCE_MODE}"
    )
GMM_COMPONENTS = int(os.environ.get("VADES_GMM_K", "2"))
if COVARIANCE_MODE in {"diagonal_gmm", "diagonal_student_t_gmm"} and GMM_COMPONENTS < 2:
    raise ValueError(f"VADES_GMM_K 对 {COVARIANCE_MODE} 必须 >= 2, 得到 {GMM_COMPONENTS}")

# (1) 训练/输入路径
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
REVIEW_SOURCE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
)
CANDIDATE_QUERY_FILE = INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
RAW_CANDIDATE_QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / "query_by_expression_style_no_depth_check_10.json"
SUMMARY_FILE = INPUT_DIR / f"{OUTPUT_TAG}_summary.json"
DETAIL_FILE = INPUT_DIR / f"{OUTPUT_TAG}_epoch_details.jsonl"
USER_PROFILE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_sentences.jsonl"
EXCLUDED_USER_FILE = INPUT_DIR / f"{OUTPUT_TAG}_excluded_users.jsonl"
SENTENCE_EXTRACT_CACHE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_extracted_sentences.jsonl"
SELECTED_RECORD_FILE = INPUT_DIR / f"{OUTPUT_TAG}_selected_query_records.jsonl"
REJECTED_RECORD_FILE = INPUT_DIR / f"{OUTPUT_TAG}_rejected_query_records.jsonl"
QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / f"query_by_expression_style_{OUTPUT_TAG}.json"

# (2) probe 路径常量
PROBE_REPRESENTATION_FIELD = "user_mu"
PROBE_N_SPLITS = 10
PROBE_RANDOM_STATE = 42
PROBE_MODEL_NAME = "svr_rbf"
PROBE_MAX_USERS = os.environ.get("STYLE_PROBE_MAX_USERS")
PROBE_PER_FEATURE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_style_vector_eval_per_feature.jsonl"
PROBE_FOLD_FILE = INPUT_DIR / f"{OUTPUT_TAG}_style_vector_eval_fold_metrics.jsonl"
PROBE_SUMMARY_FILE = INPUT_DIR / f"{OUTPUT_TAG}_style_vector_eval_summary.json"

# (3) validate 路径常量 (独立的 INPUT_DIR/OUTPUT_TAG)
VALIDATE_INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / CATEGORY
VALIDATE_SENTENCE_FILE = VALIDATE_INPUT_DIR / f"{OUTPUT_TAG}_sentences.jsonl"
VALIDATE_USER_PROFILE_FILE = VALIDATE_INPUT_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
VALIDATE_SUMMARY_FILE = VALIDATE_INPUT_DIR / f"{OUTPUT_TAG}_summary.json"
VALIDATE_ENCODER_CKPT = VALIDATE_INPUT_DIR / "vades_encoder.pt"
VALIDATE_USER_TABLE_CKPT = VALIDATE_INPUT_DIR / "vades_user_table.pt"
VALIDATE_OUTPUT_TAG = "gaussian_validation"
VALIDATE_OUTPUT_DIR = VALIDATE_INPUT_DIR / f"{VALIDATE_OUTPUT_TAG}_{CATEGORY}"
VALIDATE_FIG_DIR = VALIDATE_OUTPUT_DIR / "figures"
VALIDATE_BATCH_SIZE = 4096
VALIDATE_LATENT_DIM_EXPECTED = 20
VALIDATE_N_SAMPLES_PER_USER = 10
VALIDATE_ALPHA = 0.05
VALIDATE_ENERGY_N_SYNTH = 100
VALIDATE_ENERGY_PERMUTATIONS = 199
VALIDATE_ENERGY_USER_SUBSET = 500
VALIDATE_BASELINE_N_USERS = 100
VALIDATE_BASELINE_N_REPLICATES = 30

# (4) compare 路径常量 (复用 VALIDATE_INPUT_DIR, 即 10_complexity_analysis_clause_features)
COMPARE_FEATURE_BASE = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / CATEGORY
COMPARE_OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / "interpretability_paper" / CATEGORY
COMPARE_OUTPUT_JSON = COMPARE_OUTPUT_DIR / "prior_gaussian_vs_nongaussian.json"
COMPARE_OUTPUT_PNG = COMPARE_OUTPUT_DIR / "prior_gaussian_vs_nongaussian.png"
COMPARE_GROUPS = [
    {"key": "gmm_2", "label": "2-GMM (Gaussian)", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_gmm", "cov_mode": "diagonal_gmm", "family": "Gaussian"},
    {"key": "student_t", "label": "Student-t", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t", "cov_mode": "diagonal_student_t", "family": "Non-Gaussian"},
    {"key": "laplace", "label": "Laplace", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_laplace", "cov_mode": "diagonal_laplace", "family": "Non-Gaussian"},
    {"key": "logistic", "label": "Logistic", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_logistic", "cov_mode": "diagonal_logistic", "family": "Non-Gaussian"},
    {"key": "student_t_gmm", "label": "Student-t GMM (K=2)", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t_gmm", "cov_mode": "diagonal_student_t_gmm", "family": "Non-Gaussian"},
    {"key": "student_t_strict", "label": "Student-t strict VIB", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t_strict", "cov_mode": "diagonal_student_t", "encoder_dist": "student_t", "family": "Non-Gaussian"},
]

# (5) smoke 常量
SMOKE_NUM_USERS = 8
SMOKE_LATENT_DIM = 20
SMOKE_BATCH = 32
SMOKE_GMM_K = 2


# ============================================================
# SECTION 2: Shared utility functions
# ============================================================

def set_random_seed(seed: int) -> None:
    """设置 numpy/torch/cuda 随机种子 (原 train 脚本的 set_random_seed)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 向后兼容别名 (validate 脚本原使用 set_seed)
set_seed = set_random_seed


def infer_device() -> torch.device:
    """推断训练设备 (cuda 优先, 失败回退 cpu)."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def load_json(path: Path):
    """读取 JSON 文件."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """写 JSONL 文件."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def summarize_array(values: np.ndarray) -> dict:
    """汇总数组统计 (mean/std/min/quartiles/max). 与 train/probe/validate 通用."""
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


def load_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 文件, 跳过空行. (与 probe/validate 一致; compare 原 _load_jsonl 合并于此)"""
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ============================================================
# SECTION 3: KL / likelihood / reparameterization
# ============================================================

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

        V ~ Gamma(ν/2, ν/2)
        eps ~ N(0, I)
        z = μ + σ · eps / sqrt(V)

    Inputs:
      mu, logvar:  [..., latent_dim]
      df:          broadcastable to mu (e.g. [..., latent_dim] or [..., 1])

    Returns: z of the same shape as mu, with z ~ St(μ, σ², df).
    """
    V = torch._standard_gamma(df / 2.0) * (2.0 / df)
    eps = torch.randn_like(mu)
    z = mu + torch.exp(0.5 * logvar) * eps / torch.sqrt(V)
    return z


def _student_t_log_pdf_per_dim(
    x: torch.Tensor, mu: torch.Tensor, log_scale: torch.Tensor, df: torch.Tensor
) -> torch.Tensor:
    """Per-dim log p_t(x; mu, sigma, df) for a univariate Student-t."""
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


def student_t_log_p_matrix(
    z: torch.Tensor,
    user_mu: torch.Tensor, user_log_scale: torch.Tensor, user_df: torch.Tensor,
) -> torch.Tensor:
    """log p_t(z | u) for a per-user diagonal Student-t user prior (z sampled from q)."""
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
    """KL(N(mu_q, L_q L_q^T) || N(mu_p, L_p L_p^T)) for full covariance."""
    diff = mu_q - mu_p
    D = mu_q.size(-1)
    eye = torch.eye(D, dtype=L_p.dtype, device=L_p.device)
    eye_broadcast = eye.expand(*L_p.shape[:-2], D, D)
    Lp_inv = torch.linalg.solve_triangular(L_p, eye_broadcast, upper=False)
    Sigma_p_inv = Lp_inv.transpose(-1, -2) @ Lp_inv

    quad = torch.einsum("...i,...ij,...j->...", diff, Sigma_p_inv, diff)

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

    E_q[log p(z|u)] ≈ logsumexp_k( log π_{u,k} + E_q[log N_k(z)] )
                    = logsumexp_k( log π_{u,k} - KL(q || N_{u,k}) ) + const_i
    """
    if mu_q.dim() == 2:
        mu_q_b = mu_q.unsqueeze(1).unsqueeze(1)         # [B, 1, 1, D]
        logvar_q_b = logvar_q.unsqueeze(1).unsqueeze(1)
    else:
        mu_q_b = mu_q
        logvar_q_b = logvar_q
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


def student_t_log_likelihood(
    mu_q: torch.Tensor, logvar_q: torch.Tensor,
    user_mu: torch.Tensor, user_log_scale: torch.Tensor, user_df: torch.Tensor,
) -> torch.Tensor:
    """log p(μ_q | u) for a per-user diagonal Student-t user prior (point estimate)."""
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
    """log p(μ_q | u) for a per-user diagonal Laplace user prior (point estimate)."""
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
    """log p(μ_q | u) for a per-user diagonal Logistic user prior (point estimate)."""
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
    """log p(μ_q | u) for a per-user K-component diagonal Student-t mixture."""
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


# ============================================================
# SECTION 4: Model classes (Encoder + User Distribution Table)
# ============================================================

class SentenceEncoder(nn.Module):
    """Sentence encoder: x → (mu, logvar, reconstruction). Gaussian latent."""

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
    """SentenceEncoder with diagonal Student-t posterior (μ, σ, ν per-dim)."""

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
    """Single diagonal Gaussian per user (legacy default)."""

    def __init__(self, num_users: int, latent_dim: int):
        super().__init__()
        self.user_mu = nn.Embedding(num_users, latent_dim)
        self.user_logvar = nn.Embedding(num_users, latent_dim)
        nn.init.zeros_(self.user_mu.weight)
        nn.init.zeros_(self.user_logvar.weight)

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_mu(user_index), self.user_logvar(user_index)


class UserDistributionTableGMM(nn.Module):
    """Per-user K-component diagonal-Gaussian mixture."""

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
    """Per-user diagonal Student-t distribution."""

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
    """Per-user diagonal Laplace distribution."""

    def __init__(self, num_users: int, latent_dim: int):
        super().__init__()
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.user_mu = nn.Parameter(torch.zeros(num_users, latent_dim))
        self.user_log_b = nn.Parameter(torch.zeros(num_users, latent_dim))

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_mu[user_index], self.user_log_b[user_index]


class UserDistributionTableLogistic(nn.Module):
    """Per-user diagonal Logistic distribution."""

    def __init__(self, num_users: int, latent_dim: int):
        super().__init__()
        self.num_users = num_users
        self.latent_dim = latent_dim
        self.user_mu = nn.Parameter(torch.zeros(num_users, latent_dim))
        self.user_log_s = nn.Parameter(torch.zeros(num_users, latent_dim))

    def forward(self, user_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user_mu[user_index], self.user_log_s[user_index]


class UserDistributionTableStudentTGMM(nn.Module):
    """Per-user K-component diagonal Student-t mixture."""

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


# ============================================================
# SECTION 5: Data loading + feature extraction (train subcommand)
# ============================================================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_sentences_from_review_text(text: str) -> list[str]:
    from extract_clause_features_single_query import load_spacy_model  # noqa
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
    """过滤掉有重复 target_reviews 的用户, 避免句子提取时产生重复句子导致 holdout 不足."""
    global REVIEW_SOURCE_FILE

    filtered_file = INPUT_DIR / "stage1_filtered_users_reviews_dedup.json"
    if filtered_file.exists():
        log(f"已存在去重后的评论文件, 跳过过滤: {filtered_file}")
        REVIEW_SOURCE_FILE = filtered_file
        return

    log("开始过滤重复 target_reviews 用户")
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
        f"(前 {TRAIN_SENTENCES_PER_USER} 个为训练, 最多 {MAX_HOLDOUT_SENTENCES_PER_USER} 个为 holdout)"
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
        seen_sentence_texts: set[str] = set()
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
                f"实际={len(collected)}, 从本方法中过滤"
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
            f"句子数={len(collected)}(holdout={len(collected) - TRAIN_SENTENCES_PER_USER})"
        )
        kept_rows.extend(collected)
    return kept_rows, excluded_rows


def build_sentence_feature_rows(sentence_rows: list[dict]) -> tuple[list[dict], list[str]]:
    log("开始提取评论句法特征")
    from extract_clause_features_single_query import extract_clause_features_from_doc, load_spacy_model  # noqa
    nlp = load_spacy_model()
    feature_names: list[str] | None = None
    enriched_rows: list[dict] = []
    total = len(sentence_rows)
    last_user_id: str | None = None
    processed_users: int = 0

    # 使用 nlp.pipe() 批量处理, 比逐句处理快 2-3 倍
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
    log(f"未发现候选特征文件, 开始从原始 10 候选 query 生成: {RAW_CANDIDATE_QUERY_FILE}")
    raw_rows = load_json(RAW_CANDIDATE_QUERY_FILE)
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{RAW_CANDIDATE_QUERY_FILE} 必须是非空列表")
    rows = build_candidate_feature_rows_from_raw_query_file(raw_rows)
    write_jsonl(CANDIDATE_QUERY_FILE, rows)
    log(f"候选特征文件已写入: {CANDIDATE_QUERY_FILE}")
    return rows


def build_candidate_feature_rows_from_raw_query_file(raw_rows: list[dict]) -> list[dict]:
    from extract_clause_features_single_query import extract_clause_features_from_doc, load_spacy_model  # noqa
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


# ============================================================
# SECTION 6: Training + Inference + Calibration + Ranking + Summary
# ============================================================

def train_vades_user_distribution_model(
    sentence_rows: list[dict],
    dataset: dict,
    feature_names: list[str],
    user_ids: list[str],
    encoder_dist: str,
    covariance_mode: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    user_chunk_size: int,
    device: torch.device,
    encoder_ckpt_path: Path,
    user_table_ckpt_path: Path,
    detail_file: Path,
    summary_file: Path,
    user_match_weight: float = USER_MATCH_WEIGHT,
    style_recon_weight: float = STYLE_RECON_WEIGHT,
    sent_kl_weight: float = SENT_KL_WEIGHT,
    user_prior_kl_weight: float = USER_PRIOR_KL_WEIGHT,
    latent_align_weight: float = LATENT_ALIGN_WEIGHT,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    gmm_components: int = GMM_COMPONENTS,
    max_users_override: Optional[int] = None,
    seed: int = SEED,
) -> tuple[SentenceEncoder | SentenceEncoderStudentT | SentenceEncoderFull, object, list[dict]]:
    """主训练 loop (与原 train_vades_lite_sentence_latent_threshold.main() 一致)."""
    set_random_seed(seed)
    feature_matrix = dataset["scaled_features"].astype(np.float32)
    feature_matrix_raw = dataset["feature_matrix"]
    user_indices = dataset["user_indices"]
    train_mask = dataset["train_mask"]
    holdout_mask = dataset["holdout_mask"]
    grouped_rows = dataset["grouped_rows"]
    input_dim = feature_matrix.shape[1]
    num_users = len(user_ids)

    log(f"用户数={num_users}, 输入特征维度={input_dim}, 训练 epoch={epochs}, batch={batch_size}")

    if max_users_override is not None and max_users_override > 0 and max_users_override < num_users:
        log(f"根据 VADES_MAX_USERS={max_users_override} 限制用户数")
        sorted_user_ids = sorted(user_ids)
        keep_user_ids = sorted(sorted_user_ids)[:max_users_override]
        keep_user_set = set(keep_user_ids)
        keep_indices: list[int] = []
        new_user_to_index = {}
        for original_index, original_user_id in enumerate(user_ids):
            if original_user_id in keep_user_set:
                keep_indices.append(original_index)
                new_user_to_index[original_user_id] = len(new_user_to_index)
        keep_indices_arr = np.asarray(keep_indices, dtype=np.int64)
        new_num_users = len(keep_indices_arr)
        feature_matrix = feature_matrix[keep_indices_arr]
        feature_matrix_raw = feature_matrix_raw[keep_indices_arr]
        user_indices = np.asarray([new_user_to_index[user_ids[idx]] for idx in keep_indices_arr], dtype=np.int64)
        train_mask = train_mask[keep_indices_arr]
        holdout_mask = holdout_mask[keep_indices_arr]
        user_ids = [user_ids[idx] for idx in keep_indices_arr]
        num_users = new_num_users
        dataset["grouped_rows"] = {user_id: grouped_rows[user_id] for user_id in user_ids}

    encoder, user_table = _build_encoder_and_user_table(
        input_dim, HIDDEN_DIM, LATENT_DIM, num_users,
        encoder_dist=encoder_dist, covariance_mode=covariance_mode, gmm_components=gmm_components,
    )
    encoder.to(device)
    user_table.to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(user_table.parameters()),
        lr=learning_rate, weight_decay=weight_decay,
    )

    feature_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    user_index_tensor = torch.as_tensor(user_indices, dtype=torch.long, device=device)
    train_mask_tensor = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    holdout_mask_tensor = torch.as_tensor(holdout_mask, dtype=torch.bool, device=device)

    train_indices = torch.nonzero(train_mask_tensor, as_tuple=False).squeeze(-1)
    holdout_indices = torch.nonzero(holdout_mask_tensor, as_tuple=False).squeeze(-1)

    log(f"训练集句子数={train_indices.numel()}, 校准 holdout 句子数={holdout_indices.numel()}")

    best_loss = float("inf")
    best_epoch: int | None = None
    patience_left = early_stop_patience
    detail_rows: list[dict] = []
    encoder_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    user_table_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    encoder.train()
    user_table.train()

    for epoch in range(1, epochs + 1):
        perm = train_indices[torch.randperm(train_indices.numel(), device=device)]
        total_loss = torch.zeros((), device=device)
        n_batches = 0
        for chunk_start in range(0, perm.numel(), batch_size):
            chunk_end = min(chunk_start + batch_size, perm.numel())
            batch_idx = perm[chunk_start:chunk_end]
            optimizer.zero_grad()
            with autocast(enabled=(device.type == "cuda")):
                losses = _compute_losses_for_batch(
                    batch_idx,
                    feature_tensor, user_index_tensor, num_users,
                    encoder, user_table, feature_matrix_raw,
                    encoder_dist=encoder_dist, covariance_mode=covariance_mode,
                    user_match_weight=user_match_weight,
                    style_recon_weight=style_recon_weight,
                    sent_kl_weight=sent_kl_weight,
                    user_prior_kl_weight=user_prior_kl_weight,
                    latent_align_weight=latent_align_weight,
                )
            loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(user_table.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss = total_loss + loss.detach()
            n_batches += 1
        avg_loss = (total_loss / max(n_batches, 1)).item()
        log(f"epoch={epoch}/{epochs}, avg_loss={avg_loss:.4f}, lr={learning_rate}")

        detail_row = {
            "epoch": epoch,
            "avg_loss": float(avg_loss),
        }
        for key, value in losses.items():
            if key == "loss":
                continue
            detail_row[key] = float(value.detach().item())
        detail_rows.append(detail_row)
        write_jsonl(detail_file, detail_rows)

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_epoch = epoch
            patience_left = early_stop_patience
            torch.save(encoder.state_dict(), encoder_ckpt_path)
            torch.save(user_table.state_dict(), user_table_ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"early stop: best_epoch={best_epoch}, best_loss={best_loss:.4f}")
                break

    if best_epoch is not None:
        encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))
        user_table.load_state_dict(torch.load(user_table_ckpt_path, map_location=device))
        log(f"已加载最优 checkpoint (epoch={best_epoch})")

    summary_payload = {
        "epochs_requested": int(epochs),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_loss": float(best_loss),
        "encoder_dist": encoder_dist,
        "covariance_mode": covariance_mode,
        "gmm_components": int(gmm_components) if covariance_mode in {"diagonal_gmm", "diagonal_student_t_gmm"} else None,
        "latent_dim": int(LATENT_DIM),
        "hidden_dim": int(HIDDEN_DIM),
        "input_dim": int(input_dim),
        "num_users": int(num_users),
        "train_sentence_count": int(train_indices.numel()),
        "holdout_sentence_count": int(holdout_indices.numel()),
        "feature_names": feature_names,
        "user_ids": user_ids,
        "detail_rows": detail_rows,
    }
    summary_file.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入训练 summary: {summary_file}")
    return encoder, user_table, detail_rows


def _build_encoder_and_user_table(
    input_dim: int, hidden_dim: int, latent_dim: int, num_users: int,
    encoder_dist: str, covariance_mode: str, gmm_components: int,
) -> tuple[nn.Module, nn.Module]:
    """按 encoder_dist/covariance_mode 构造 encoder + user table."""
    if covariance_mode == "full":
        encoder = SentenceEncoderFull(input_dim, hidden_dim, latent_dim)
        user_table = UserDistributionTableFull(num_users, latent_dim)
    elif encoder_dist == "student_t":
        encoder = SentenceEncoderStudentT(input_dim, hidden_dim, latent_dim)
        user_table = UserDistributionTableStudentT(num_users, latent_dim)
    elif covariance_mode == "diagonal_gmm":
        encoder = SentenceEncoder(input_dim, hidden_dim, latent_dim)
        user_table = UserDistributionTableGMM(num_users, latent_dim, gmm_components)
    elif covariance_mode == "diagonal_student_t":
        encoder = SentenceEncoder(input_dim, hidden_dim, latent_dim)
        user_table = UserDistributionTableStudentT(num_users, latent_dim)
    elif covariance_mode == "diagonal_laplace":
        encoder = SentenceEncoder(input_dim, hidden_dim, latent_dim)
        user_table = UserDistributionTableLaplace(num_users, latent_dim)
    elif covariance_mode == "diagonal_student_t_gmm":
        encoder = SentenceEncoder(input_dim, hidden_dim, latent_dim)
        user_table = UserDistributionTableStudentTGMM(num_users, latent_dim, gmm_components)
    elif covariance_mode == "diagonal_logistic":
        encoder = SentenceEncoder(input_dim, hidden_dim, latent_dim)
        user_table = UserDistributionTableLogistic(num_users, latent_dim)
    else:
        encoder = SentenceEncoder(input_dim, hidden_dim, latent_dim)
        user_table = UserDistributionTable(num_users, latent_dim)
    return encoder, user_table


def _compute_losses_for_batch(
    batch_idx: torch.Tensor,
    feature_tensor: torch.Tensor,
    user_index_tensor: torch.Tensor,
    num_users: int,
    encoder: nn.Module,
    user_table: nn.Module,
    feature_matrix_raw: np.ndarray,
    encoder_dist: str,
    covariance_mode: str,
    user_match_weight: float,
    style_recon_weight: float,
    sent_kl_weight: float,
    user_prior_kl_weight: float,
    latent_align_weight: float,
) -> dict[str, torch.Tensor]:
    """计算一个 batch 的所有 loss 项 (按 covariance_mode 分支)."""
    x = feature_tensor[batch_idx]
    user_idx = user_index_tensor[batch_idx]
    raw_feature_targets = torch.as_tensor(
        feature_matrix_raw[batch_idx.cpu().numpy()], dtype=torch.float32, device=x.device,
    )

    if covariance_mode == "full":
        mu, L, reconstruction = encoder(x)
        user_mu, user_L = user_table(torch.arange(num_users, device=x.device))
        sent_kl = standard_normal_kl(mu, 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-6).log())
        sent_kl = sent_kl.mean()
        user_prior_kl = multivariate_gaussian_kl(
            user_mu.unsqueeze(0).expand(num_users, -1, -1),
            user_L.unsqueeze(0).expand(num_users, -1, -1, -1),
            torch.zeros_like(user_mu).unsqueeze(0).expand(num_users, -1, -1),
            torch.eye(user_mu.size(-1), device=user_mu.device).unsqueeze(0).expand(num_users, -1, -1),
        ).mean()
        user_match = _user_match_loss(mu, user_mu[user_idx], user_L[user_idx], user_match_weight)
        recon_loss = F.mse_loss(reconstruction, raw_feature_targets)
        latent_align = F.mse_loss(mu, raw_feature_targets)
    else:
        if encoder_dist == "student_t":
            mu, logvar, df, reconstruction = encoder(x)
        else:
            mu, logvar, reconstruction = encoder(x)
        sent_kl = standard_normal_kl(mu, logvar).mean()

        all_user_idx = torch.arange(num_users, device=x.device)
        if covariance_mode in {"diagonal_gmm"}:
            user_mu_k, user_logvar_k, mix_logits = user_table(all_user_idx)
            user_prior_kl = gmm_log_likelihood(
                user_mu_k.detach(), user_logvar_k.detach(), mix_logits.detach()
            ).mean() * 0.0
            user_prior_kl = standard_normal_kl(user_mu_k, user_logvar_k).mean() * 0.5
            log_p_xu = gmm_log_likelihood(mu, logvar, user_mu_k, user_logvar_k, mix_logits)
            user_match = F.cross_entropy(log_p_xu, user_idx)
        elif covariance_mode in {"diagonal_student_t"}:
            user_mu, user_log_scale, user_df = user_table(all_user_idx)
            equiv_logvar = 2.0 * user_log_scale + torch.log(
                (user_df / (user_df - 2.0).clamp_min(1e-3)).clamp_min(1.0)
            ).unsqueeze(-1)
            user_prior_kl = standard_normal_kl(user_mu, equiv_logvar).mean()
            log_p_xu = student_t_log_likelihood(mu, logvar, user_mu, user_log_scale, user_df)
            user_match = F.cross_entropy(log_p_xu, user_idx)
        elif covariance_mode in {"diagonal_laplace"}:
            user_mu, user_log_b = user_table(all_user_idx)
            equiv_logvar = 2.0 * user_log_b + float(np.log(2.0))
            user_prior_kl = standard_normal_kl(user_mu, equiv_logvar).mean()
            log_p_xu = laplace_log_likelihood(mu, logvar, user_mu, user_log_b)
            user_match = F.cross_entropy(log_p_xu, user_idx)
        elif covariance_mode in {"diagonal_student_t_gmm"}:
            user_mu_k, user_log_scale_k, user_df_k, mix_logits = user_table(all_user_idx)
            equiv_logvar_k = 2.0 * user_log_scale_k + torch.log(
                (user_df_k / (user_df_k - 2.0).clamp_min(1e-3)).clamp_min(1.0)
            ).unsqueeze(-1)
            per_comp_kl = standard_normal_kl(user_mu_k, equiv_logvar_k)
            mix_probs = F.softmax(mix_logits, dim=-1)
            user_prior_kl = (mix_probs * per_comp_kl).sum(dim=-1).mean()
            log_p_xu = student_t_gmm_log_likelihood(
                mu, logvar, user_mu_k, user_log_scale_k, user_df_k, mix_logits,
            )
            user_match = F.cross_entropy(log_p_xu, user_idx)
        elif covariance_mode in {"diagonal_logistic"}:
            user_mu, user_log_s = user_table(all_user_idx)
            user_prior_kl = standard_normal_kl(user_mu, 2.0 * user_log_s + float(np.log(np.pi ** 2 / 3.0))).mean()
            log_p_xu = logistic_log_likelihood(mu, logvar, user_mu, user_log_s)
            user_match = F.cross_entropy(log_p_xu, user_idx)
        else:
            user_mu, user_logvar = user_table(all_user_idx)
            user_prior_kl = standard_normal_kl(user_mu, user_logvar).mean()
            user_prior_kl_val = diagonal_gaussian_kl(
                user_mu.unsqueeze(0).expand(num_users, -1, -1),
                user_logvar.unsqueeze(0).expand(num_users, -1, -1),
                torch.zeros_like(user_mu).unsqueeze(0).expand(num_users, -1, -1),
                torch.zeros_like(user_logvar).unsqueeze(0).expand(num_users, -1, -1),
            ).mean()
            log_p_xu = -diagonal_gaussian_kl(
                mu.unsqueeze(1),
                logvar.unsqueeze(1),
                user_mu.unsqueeze(0),
                user_logvar.unsqueeze(0),
            )
            user_match = F.cross_entropy(log_p_xu, user_idx)
            user_prior_kl = user_prior_kl_val
        recon_loss = F.mse_loss(reconstruction, raw_feature_targets)
        latent_align = F.mse_loss(mu, raw_feature_targets)

    loss = (
        user_match_weight * user_match
        + style_recon_weight * recon_loss
        + sent_kl_weight * sent_kl
        + user_prior_kl_weight * user_prior_kl
        + latent_align_weight * latent_align
    )
    return {
        "loss": loss,
        "user_match_loss": user_match,
        "recon_loss": recon_loss,
        "sent_kl": sent_kl,
        "user_prior_kl": user_prior_kl,
        "latent_align": latent_align,
    }


def _user_match_loss(mu: torch.Tensor, user_mu: torch.Tensor, user_L: torch.Tensor, weight: float) -> torch.Tensor:
    """对 full covariance 模式, user_match_loss 用 Mahalanobis 距离近似."""
    diff = mu - user_mu
    eye = torch.eye(user_L.size(-1), device=user_L.device, dtype=user_L.dtype)
    L_inv = torch.linalg.solve_triangular(user_L, eye.unsqueeze(0).expand_as(user_L), upper=False)
    Sigma_inv = L_inv.transpose(-1, -2) @ L_inv
    quad = torch.einsum("...i,...ij,...j->...", diff, Sigma_inv, diff)
    return weight * quad.mean()


def infer_user_sentence_distributions(
    sentence_rows: list[dict],
    dataset: dict,
    user_ids: list[str],
    encoder: nn.Module,
    user_table: nn.Module,
    device: torch.device,
    encoder_dist: str,
    covariance_mode: str,
    user_profile_file: Path,
    sentence_output_file: Path,
    user_mu_field: str = PROBE_REPRESENTATION_FIELD,
) -> tuple[list[dict], list[dict]]:
    """对每条 sentence 推断 μ; 对每个 user 聚合 μ."""
    feature_matrix = dataset["scaled_features"].astype(np.float32)
    user_indices = dataset["user_indices"]
    feature_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    user_index_tensor = torch.as_tensor(user_indices, dtype=torch.long, device=device)

    encoder.eval()
    user_table.eval()
    all_user_idx = torch.arange(user_table.user_mu.weight.shape[0] if hasattr(user_table, "user_mu") else user_table.num_users, device=device)
    with torch.no_grad():
        mu_out = []
        for chunk_start in range(0, feature_tensor.shape[0], USER_CHUNK_SIZE_TRAIN):
            chunk_end = min(chunk_start + USER_CHUNK_SIZE_TRAIN, feature_tensor.shape[0])
            batch_idx = torch.arange(chunk_start, chunk_end, device=device)
            x = feature_tensor[batch_idx]
            if encoder_dist == "student_t":
                mu_chunk, _, _, _ = encoder(x)
            elif covariance_mode == "full":
                mu_chunk, _, _ = encoder(x)
            else:
                mu_chunk, _, _ = encoder(x)
            mu_out.append(mu_chunk.detach().cpu().numpy())
        mu_full = np.concatenate(mu_out, axis=0)
        if covariance_mode == "full":
            user_mu_tensor, user_L_tensor = user_table(all_user_idx)
        elif covariance_mode in {"diagonal_gmm"}:
            user_mu_k, user_logvar_k, mix_logits = user_table(all_user_idx)
            user_mu_tensor = user_mu_k.detach()
        elif covariance_mode in {"diagonal_student_t"}:
            user_mu, user_log_scale, user_df = user_table(all_user_idx)
            user_mu_tensor = user_mu
        elif covariance_mode in {"diagonal_laplace"}:
            user_mu, user_log_b = user_table(all_user_idx)
            user_mu_tensor = user_mu
        elif covariance_mode in {"diagonal_student_t_gmm"}:
            user_mu_k, user_log_scale_k, user_df_k, mix_logits = user_table(all_user_idx)
            user_mu_tensor = user_mu_k.detach()
        elif covariance_mode in {"diagonal_logistic"}:
            user_mu, user_log_s = user_table(all_user_idx)
            user_mu_tensor = user_mu
        else:
            user_mu_tensor, _ = user_table(all_user_idx)
        user_mu_array = user_mu_tensor.detach().cpu().numpy()

    sentence_output: list[dict] = []
    user_profile_rows: list[dict] = []
    grouped_rows = dataset["grouped_rows"]
    user_id_to_array_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    for user_id, user_rows in grouped_rows.items():
        user_idx = user_id_to_array_idx[user_id]
        user_mu = user_mu_array[user_idx]
        user_profile_rows.append(
            {
                "user_id": user_id,
                user_mu_field: [float(v) for v in user_mu.tolist()],
            }
        )
        for row in user_rows:
            sentence_idx_in_full = next(
                (idx for idx, orig_row in enumerate(sentence_rows) if orig_row is row),
                None,
            )
            if sentence_idx_in_full is None:
                continue
            mu_vec = mu_full[sentence_idx_in_full]
            sentence_output.append(
                {
                    "user_id": user_id,
                    "review_index": row.get("review_index"),
                    "sentence_index": row.get("sentence_index"),
                    "sentence_text": row.get("sentence_text"),
                    "word_count": row.get("word_count"),
                    "features": row.get("features"),
                    "mu": [float(v) for v in mu_vec.tolist()],
                    "is_holdout": bool(dataset["holdout_mask"][sentence_idx_in_full]),
                }
            )

    write_jsonl(user_profile_file, user_profile_rows)
    write_jsonl(sentence_output_file, sentence_output)
    log(f"已写入 user profiles: {user_profile_file}")
    log(f"已写入 sentence μ: {sentence_output_file}")
    return user_profile_rows, sentence_output


def calibrate_absolute_threshold_with_unseen_holdout(
    sentence_output: list[dict],
    candidate_rows: list[dict],
    dataset: dict,
    feature_names: list[str],
    user_ids: list[str],
    user_profile_rows: list[dict],
    user_table: nn.Module,
    device: torch.device,
    covariance_mode: str,
    user_match_weight: float,
    abs_threshold_quantile: float,
    calibration_summary_file: Path,
) -> dict:
    """使用训练未见 holdout 句子做 user-match score 分布, 校准绝对阈值."""
    if not sentence_output:
        raise ValueError("sentence_output 为空, 无法校准阈值")

    feature_matrix = dataset["feature_matrix"]
    user_indices = dataset["user_indices"]
    train_mask = dataset["train_mask"]
    holdout_mask = dataset["holdout_mask"]

    train_indices = [idx for idx, val in enumerate(train_mask) if val]
    holdout_indices = [idx for idx, val in enumerate(holdout_mask) if val]
    if not holdout_indices:
        raise ValueError("没有 holdout 句子, 无法做绝对阈值校准")

    holdout_features = torch.as_tensor(feature_matrix[holdout_indices], dtype=torch.float32, device=device)
    encoder = None
    encoder_path = INPUT_DIR / "vades_encoder.pt"
    if encoder_path.exists():
        if covariance_mode == "full":
            encoder = SentenceEncoderFull(
                input_dim=feature_matrix.shape[1],
                hidden_dim=HIDDEN_DIM,
                latent_dim=LATENT_DIM,
            )
        else:
            encoder = SentenceEncoder(
                input_dim=feature_matrix.shape[1],
                hidden_dim=HIDDEN_DIM,
                latent_dim=LATENT_DIM,
            )
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder.to(device)
        encoder.eval()

    with torch.no_grad():
        if encoder is not None:
            mu_out = []
            for chunk_start in range(0, holdout_features.shape[0], USER_CHUNK_SIZE_TRAIN):
                chunk_end = min(chunk_start + USER_CHUNK_SIZE_TRAIN, holdout_features.shape[0])
                x = holdout_features[chunk_start:chunk_end]
                if covariance_mode == "full":
                    mu_chunk, _, _ = encoder(x)
                else:
                    mu_chunk, _, _ = encoder(x)
                mu_out.append(mu_chunk)
            holdout_mu = torch.cat(mu_out, dim=0)
        else:
            holdout_mu = holdout_features

        all_user_idx = torch.arange(user_table.user_mu.weight.shape[0] if hasattr(user_table, "user_mu") else user_table.num_users, device=device)
        user_match_scores = []
        user_match_labels = []
        if covariance_mode == "full":
            user_mu_tensor, user_L_tensor = user_table(all_user_idx)
            eye = torch.eye(user_L_tensor.size(-1), device=device, dtype=user_L_tensor.dtype)
            L_inv = torch.linalg.solve_triangular(user_L_tensor, eye.unsqueeze(0).expand_as(user_L_tensor), upper=False)
            Sigma_inv = L_inv.transpose(-1, -2) @ L_inv
            for chunk_start in range(0, holdout_mu.shape[0], USER_CHUNK_SIZE_TRAIN):
                chunk_end = min(chunk_start + USER_CHUNK_SIZE_TRAIN, holdout_mu.shape[0])
                mu_chunk = holdout_mu[chunk_start:chunk_end]
                diff = mu_chunk.unsqueeze(1) - user_mu_tensor.unsqueeze(0)
                quad = torch.einsum("...i,...ij,...j->...", diff, Sigma_inv, diff)
                scores = -quad
                labels = user_index_tensor[holdout_indices[chunk_start:chunk_end]]
                user_match_scores.append(scores.cpu().numpy())
                user_match_labels.append(labels.cpu().numpy())
        else:
            for chunk_start in range(0, holdout_mu.shape[0], USER_CHUNK_SIZE_TRAIN):
                chunk_end = min(chunk_start + USER_CHUNK_SIZE_TRAIN, holdout_mu.shape[0])
                mu_chunk = holdout_mu[chunk_start:chunk_end]
                if covariance_mode in {"diagonal_gmm"}:
                    user_mu_k, user_logvar_k, mix_logits = user_table(all_user_idx)
                    log_p = gmm_log_likelihood(mu_chunk, torch.zeros_like(mu_chunk), user_mu_k, user_logvar_k, mix_logits)
                elif covariance_mode in {"diagonal_student_t"}:
                    user_mu, user_log_scale, user_df = user_table(all_user_idx)
                    log_p = student_t_log_likelihood(mu_chunk, torch.zeros_like(mu_chunk), user_mu, user_log_scale, user_df)
                elif covariance_mode in {"diagonal_laplace"}:
                    user_mu, user_log_b = user_table(all_user_idx)
                    log_p = laplace_log_likelihood(mu_chunk, torch.zeros_like(mu_chunk), user_mu, user_log_b)
                elif covariance_mode in {"diagonal_student_t_gmm"}:
                    user_mu_k, user_log_scale_k, user_df_k, mix_logits = user_table(all_user_idx)
                    log_p = student_t_gmm_log_likelihood(mu_chunk, torch.zeros_like(mu_chunk), user_mu_k, user_log_scale_k, user_df_k, mix_logits)
                elif covariance_mode in {"diagonal_logistic"}:
                    user_mu, user_log_s = user_table(all_user_idx)
                    log_p = logistic_log_likelihood(mu_chunk, torch.zeros_like(mu_chunk), user_mu, user_log_s)
                else:
                    user_mu, user_logvar = user_table(all_user_idx)
                    log_p = -diagonal_gaussian_kl(
                        mu_chunk.unsqueeze(1),
                        torch.zeros_like(mu_chunk).unsqueeze(1),
                        user_mu.unsqueeze(0),
                        user_logvar.unsqueeze(0),
                    )
                scores = log_p
                labels = user_index_tensor[holdout_indices[chunk_start:chunk_end]]
                user_match_scores.append(scores.cpu().numpy())
                user_match_labels.append(labels.cpu().numpy())

        all_scores = np.concatenate(user_match_scores, axis=0)
        all_labels = np.concatenate(user_match_labels, axis=0)
        pos_scores = all_scores[all_labels >= 0]
        if pos_scores.size == 0:
            raise ValueError("无法校准: holdout 句子中缺少正向用户匹配样本")

        threshold_value = float(np.quantile(pos_scores, abs_threshold_quantile))
        summary = {
            "abs_threshold_value": threshold_value,
            "abs_threshold_quantile": abs_threshold_quantile,
            "n_positive_holdout_samples": int(pos_scores.size),
            "score_summary": summarize_array(all_scores),
            "positive_score_summary": summarize_array(pos_scores),
            "user_match_weight": user_match_weight,
            "covariance_mode": covariance_mode,
            "calibration_set": "holdout_5_per_user",
        }
        calibration_summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"已写入校准 summary: {calibration_summary_file}")
        log(f"绝对阈值={threshold_value:.4f} (基于 {pos_scores.size} 个 holdout 正样本 {abs_threshold_quantile} 分位数)")
        return summary


def rank_and_select_queries(
    candidate_rows: list[dict],
    sentence_output: list[dict],
    user_profile_rows: list[dict],
    feature_names: list[str],
    user_table: nn.Module,
    device: torch.device,
    encoder: nn.Module | None,
    covariance_mode: str,
    abs_threshold_value: float,
    max_rounds: int,
    candidates_per_round: int,
    selected_record_file: Path,
    rejected_record_file: Path,
) -> tuple[list[dict], list[dict]]:
    """按 user-match score 对候选 query 排序, 选最高分且高于阈值."""
    if not candidate_rows:
        raise ValueError("candidate_rows 为空")

    candidate_features = np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in candidate_rows],
        dtype=np.float64,
    )
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(candidate_features)
    scaler.mean_ = np.asarray([0.0] * scaler.n_features_in_)
    scaler.scale_ = np.asarray([1.0] * scaler.n_features_in_)
    candidate_features_scaled = candidate_features.astype(np.float32)

    candidate_tensor = torch.as_tensor(candidate_features_scaled, dtype=torch.float32, device=device)
    user_id_to_profile = {row["user_id"]: row[PROBE_REPRESENTATION_FIELD] for row in user_profile_rows}

    all_user_idx = torch.arange(user_table.user_mu.weight.shape[0] if hasattr(user_table, "user_mu") else user_table.num_users, device=device)
    with torch.no_grad():
        if encoder is not None:
            mu_out = []
            for chunk_start in range(0, candidate_tensor.shape[0], USER_CHUNK_SIZE_TRAIN):
                chunk_end = min(chunk_start + USER_CHUNK_SIZE_TRAIN, candidate_tensor.shape[0])
                x = candidate_tensor[chunk_start:chunk_end]
                if covariance_mode == "full":
                    mu_chunk, _, _ = encoder(x)
                else:
                    mu_chunk, _, _ = encoder(x)
                mu_out.append(mu_chunk)
            candidate_mu = torch.cat(mu_out, dim=0)
        else:
            candidate_mu = candidate_tensor

        if covariance_mode == "full":
            user_mu_tensor, user_L_tensor = user_table(all_user_idx)
            eye = torch.eye(user_L_tensor.size(-1), device=device, dtype=user_L_tensor.dtype)
            L_inv = torch.linalg.solve_triangular(user_L_tensor, eye.unsqueeze(0).expand_as(user_L_tensor), upper=False)
            Sigma_inv = L_inv.transpose(-1, -2) @ L_inv
            diff = candidate_mu.unsqueeze(1) - user_mu_tensor.unsqueeze(0)
            quad = torch.einsum("...i,...ij,...j->...", diff, Sigma_inv, diff)
            all_scores = -quad.cpu().numpy()
        else:
            if covariance_mode in {"diagonal_gmm"}:
                user_mu_k, user_logvar_k, mix_logits = user_table(all_user_idx)
                all_scores = gmm_log_likelihood(candidate_mu, torch.zeros_like(candidate_mu), user_mu_k, user_logvar_k, mix_logits).cpu().numpy()
            elif covariance_mode in {"diagonal_student_t"}:
                user_mu, user_log_scale, user_df = user_table(all_user_idx)
                all_scores = student_t_log_likelihood(candidate_mu, torch.zeros_like(candidate_mu), user_mu, user_log_scale, user_df).cpu().numpy()
            elif covariance_mode in {"diagonal_laplace"}:
                user_mu, user_log_b = user_table(all_user_idx)
                all_scores = laplace_log_likelihood(candidate_mu, torch.zeros_like(candidate_mu), user_mu, user_log_b).cpu().numpy()
            elif covariance_mode in {"diagonal_student_t_gmm"}:
                user_mu_k, user_log_scale_k, user_df_k, mix_logits = user_table(all_user_idx)
                all_scores = student_t_gmm_log_likelihood(candidate_mu, torch.zeros_like(candidate_mu), user_mu_k, user_log_scale_k, user_df_k, mix_logits).cpu().numpy()
            elif covariance_mode in {"diagonal_logistic"}:
                user_mu, user_log_s = user_table(all_user_idx)
                all_scores = logistic_log_likelihood(candidate_mu, torch.zeros_like(candidate_mu), user_mu, user_log_s).cpu().numpy()
            else:
                user_mu, user_logvar = user_table(all_user_idx)
                all_scores = -diagonal_gaussian_kl(
                    candidate_mu.unsqueeze(1),
                    torch.zeros_like(candidate_mu).unsqueeze(1),
                    user_mu.unsqueeze(0),
                    user_logvar.unsqueeze(0),
                ).cpu().numpy()

    user_id_to_array_idx = {}
    for array_idx, profile_row in enumerate(user_profile_rows):
        user_id_to_array_idx[profile_row["user_id"]] = array_idx

    grouped_candidates: dict[str, list[dict]] = defaultdict(list)
    for idx, row in enumerate(candidate_rows):
        grouped_candidates[row["user_id"]].append({"row": row, "candidate_index": idx, "scores": all_scores[idx]})

    selected_records: list[dict] = []
    rejected_records: list[dict] = []
    for user_id, user_candidates in grouped_candidates.items():
        array_idx = user_id_to_array_idx.get(user_id)
        if array_idx is None:
            continue
        sorted_candidates = sorted(user_candidates, key=lambda c: float(c["scores"][array_idx]), reverse=True)
        rounds = 0
        for entry in sorted_candidates:
            if rounds >= max_rounds:
                rejected_records.append(
                    {
                        "user_id": user_id,
                        "candidate_index": entry["candidate_index"],
                        "query": entry["row"].get("query"),
                        "score": float(entry["scores"][array_idx]),
                        "abs_threshold": abs_threshold_value,
                        "reason": "exceed_max_rounds",
                    }
                )
                continue
            rounds += 1
            if entry["scores"][array_idx] >= abs_threshold_value:
                selected_records.append(
                    {
                        "user_id": user_id,
                        "candidate_index": entry["candidate_index"],
                        "asin": entry["row"].get("asin"),
                        "query": entry["row"].get("query"),
                        "score": float(entry["scores"][array_idx]),
                        "abs_threshold": abs_threshold_value,
                        "round_index": rounds,
                    }
                )
            else:
                rejected_records.append(
                    {
                        "user_id": user_id,
                        "candidate_index": entry["candidate_index"],
                        "query": entry["row"].get("query"),
                        "score": float(entry["scores"][array_idx]),
                        "abs_threshold": abs_threshold_value,
                        "reason": "below_threshold",
                    }
                )

    write_jsonl(selected_record_file, selected_records)
    write_jsonl(rejected_record_file, rejected_records)
    log(f"已写入 selected: {selected_record_file} ({len(selected_records)} 条)")
    log(f"已写入 rejected: {rejected_record_file} ({len(rejected_records)} 条)")
    return selected_records, rejected_records


def build_summary(
    summary_file: Path,
    train_summary: dict,
    calibration_summary: dict,
    selected_records: list[dict],
    rejected_records: list[dict],
    excluded_users: list[dict],
    category: str,
    feature_names: list[str],
    user_ids: list[str],
    cov_mode: str,
    encoder_dist: str,
) -> dict:
    """Build a final summary dict combining training + calibration + selection + excluded."""
    summary = {
        "category": category,
        "feature_names": feature_names,
        "user_count": len(user_ids),
        "covariance_mode": cov_mode,
        "encoder_dist": encoder_dist,
        "training": train_summary,
        "calibration": calibration_summary,
        "selected_query_count": len(selected_records),
        "rejected_query_count": len(rejected_records),
        "selected_to_total_ratio": (
            len(selected_records) / max(len(selected_records) + len(rejected_records), 1)
        ),
        "excluded_users": excluded_users,
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入最终 summary: {summary_file}")
    return summary


def ensure_directories() -> None:
    """Make sure all parent dirs exist for input/output files."""
    for path in (
        SUMMARY_FILE, DETAIL_FILE, USER_PROFILE_FILE, SENTENCE_FILE,
        EXCLUDED_USER_FILE, SELECTED_RECORD_FILE, REJECTED_RECORD_FILE,
        QUERY_FILE.parent, PROBE_SUMMARY_FILE.parent, PROBE_PER_FEATURE_FILE.parent,
        PROBE_FOLD_FILE.parent, VALIDATE_OUTPUT_DIR, VALIDATE_FIG_DIR,
        COMPARE_OUTPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


# ============================================================
# SECTION 7: Probe-specific functions (SVR 10-fold)
# ============================================================

def load_user_style_vectors() -> tuple[list[str], np.ndarray]:
    log(f"开始读取 learned style vectors: {USER_PROFILE_FILE}")
    rows = load_jsonl(USER_PROFILE_FILE)
    user_ids: list[str] = []
    vectors: list[list[float]] = []
    for row in rows:
        vector = row.get(PROBE_REPRESENTATION_FIELD)
        if vector is None:
            raise ValueError(f"{USER_PROFILE_FILE} 缺少字段 {PROBE_REPRESENTATION_FIELD}: user_id={row.get('user_id')}")
        user_id = row.get("user_id")
        if user_id is None:
            raise ValueError(f"{USER_PROFILE_FILE} 缺少 user_id")
        user_ids.append(user_id)
        vectors.append(vector)
    return user_ids, np.asarray(vectors, dtype=np.float64)


def load_true_style_targets() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    log(f"开始读取真实句法风格特征并按用户聚合: {SENTENCE_FILE}")
    rows = load_jsonl(SENTENCE_FILE)
    feature_names: list[str] | None = None
    user_feature_rows: dict[str, list[list[float]]] = {}
    sentence_counts: list[int] = []
    for row in rows:
        user_id = row.get("user_id")
        features = row.get("features")
        if user_id is None or features is None:
            raise ValueError(f"{SENTENCE_FILE} 行缺少 user_id 或 features")
        if feature_names is None:
            feature_names = list(features.keys())
        else:
            current_feature_names = list(features.keys())
            if current_feature_names != feature_names:
                raise ValueError(f"{SENTENCE_FILE} 特征字段顺序不一致: user_id={user_id}")
        user_feature_rows.setdefault(user_id, []).append([float(features[name]) for name in feature_names])

    if feature_names is None:
        raise ValueError(f"{SENTENCE_FILE} 未读取到特征字段")

    user_ids = sorted(user_feature_rows.keys())
    targets: list[list[float]] = []
    for user_id in user_ids:
        feature_matrix = np.asarray(user_feature_rows[user_id], dtype=np.float64)
        sentence_counts.append(int(feature_matrix.shape[0]))
        targets.append(feature_matrix.mean(axis=0).tolist())

    return (
        user_ids,
        np.asarray(targets, dtype=np.float64),
        feature_names,
        np.asarray(sentence_counts, dtype=np.float64),
    )


def align_x_y(
    x_user_ids: list[str],
    x_vectors: np.ndarray,
    y_user_ids: list[str],
    y_targets: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    x_by_user = {user_id: x_vectors[idx] for idx, user_id in enumerate(x_user_ids)}
    shared_user_ids = [user_id for user_id in y_user_ids if user_id in x_by_user]
    if not shared_user_ids:
        raise ValueError("X 和 y 没有重叠用户")
    x_aligned = np.asarray([x_by_user[user_id] for user_id in shared_user_ids], dtype=np.float64)
    y_by_user = {user_id: y_targets[idx] for idx, user_id in enumerate(y_user_ids)}
    y_aligned = np.asarray([y_by_user[user_id] for user_id in shared_user_ids], dtype=np.float64)
    if PROBE_MAX_USERS is not None:
        max_users = int(PROBE_MAX_USERS)
        if max_users <= 0:
            raise ValueError("STYLE_PROBE_MAX_USERS 必须是正整数")
        shared_user_ids = shared_user_ids[:max_users]
        x_aligned = x_aligned[:max_users]
        y_aligned = y_aligned[:max_users]
    return shared_user_ids, x_aligned, y_aligned


def evaluate_probe(x_matrix: np.ndarray, y_matrix: np.ndarray, feature_names: list[str]) -> tuple[dict, list[dict], list[dict]]:
    log(f"开始执行 {PROBE_MODEL_NAME} 的 {PROBE_N_SPLITS}-fold cross validation")
    kfold = KFold(n_splits=PROBE_N_SPLITS, shuffle=True, random_state=PROBE_RANDOM_STATE)

    fold_rows: list[dict] = []
    per_feature_mse_values: dict[str, list[float]] = {name: [] for name in feature_names}
    overall_mse_values: list[float] = []

    for fold_index, (train_idx, test_idx) in enumerate(kfold.split(x_matrix), start=1):
        model = MultiOutputRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.1))
        model.fit(x_matrix[train_idx], y_matrix[train_idx])
        prediction = model.predict(x_matrix[test_idx])
        mse_by_feature = np.mean((prediction - y_matrix[test_idx]) ** 2, axis=0)
        overall_mse = float(np.mean(mse_by_feature))
        overall_mse_values.append(overall_mse)
        fold_rows.append(
            {
                "model": PROBE_MODEL_NAME,
                "fold_index": fold_index,
                "train_user_count": int(len(train_idx)),
                "test_user_count": int(len(test_idx)),
                "overall_mse": overall_mse,
            }
        )
        for feature_index, feature_name in enumerate(feature_names):
            per_feature_mse_values[feature_name].append(float(mse_by_feature[feature_index]))

    per_feature_rows = [
        {
            "model": PROBE_MODEL_NAME,
            "feature_name": feature_name,
            "mse": float(np.mean(values)),
        }
        for feature_name, values in per_feature_mse_values.items()
    ]
    summary = {
        "model": PROBE_MODEL_NAME,
        "n_splits": PROBE_N_SPLITS,
        "overall_mse_mean": float(np.mean(overall_mse_values)),
        "overall_mse_std": float(np.std(overall_mse_values)),
        "per_feature_mse_summary": summarize_array(
            np.asarray([row["mse"] for row in per_feature_rows], dtype=np.float64)
        ),
    }
    return summary, per_feature_rows, fold_rows


# ============================================================
# SECTION 8: Validate-specific functions (Gaussian assumption checks)
# ============================================================

def _mardia_kurtosis(samples: np.ndarray) -> float:
    """Mardia multivariate kurtosis (excess kurtosis)."""
    n, d = samples.shape
    mean = samples.mean(axis=0)
    centered = samples - mean
    S = np.cov(samples.T, bias=True) + 1e-8 * np.eye(d)
    S_inv = np.linalg.pinv(S)
    diff = centered @ S_inv
    quad_form = np.sum((diff @ centered.T) ** 2) / n
    expected = d * (d + 2)
    return float((quad_form - expected) / np.sqrt(8 * d * (d + 2) / n))


def _shapiro_wilk_per_dim(samples: np.ndarray) -> dict:
    results: dict = {}
    for dim in range(samples.shape[1]):
        try:
            stat, p = stats.shapiro(samples[:, dim])
            results[f"dim_{dim}"] = {"statistic": float(stat), "p_value": float(p)}
        except Exception as exc:
            results[f"dim_{dim}"] = {"statistic": None, "p_value": None, "error": str(exc)}
    return results


def _ks_test_per_dim(samples: np.ndarray) -> dict:
    results: dict = {}
    for dim in range(samples.shape[1]):
        try:
            stat, p = stats.kstest(samples[:, dim], "norm")
            results[f"dim_{dim}"] = {"statistic": float(stat), "p_value": float(p)}
        except Exception as exc:
            results[f"dim_{dim}"] = {"statistic": None, "p_value": None, "error": str(exc)}
    return results


def _anderson_darling_per_dim(samples: np.ndarray) -> dict:
    results: dict = {}
    for dim in range(samples.shape[1]):
        try:
            result = stats.anderson(samples[:, dim], dist="norm")
            results[f"dim_{dim}"] = {
                "statistic": float(result.statistic),
                "critical_values": [float(v) for v in result.critical_values],
                "significance_levels": [float(s) for s in result.significance_level],
            }
        except Exception as exc:
            results[f"dim_{dim}"] = {"statistic": None, "error": str(exc)}
    return results


def _energy_distance_test(x: np.ndarray, y: np.ndarray, n_permutations: int, seed: int) -> dict:
    """Permutation-based energy distance test."""
    from scipy.spatial.distance import cdist
    n_x = x.shape[0]
    n_y = y.shape[0]
    xx = cdist(x, x, metric="euclidean")
    yy = cdist(y, y, metric="euclidean")
    xy = cdist(x, y, metric="euclidean")
    np.fill_diagonal(xx, 0.0)
    np.fill_diagonal(yy, 0.0)
    energy_obs = float(2.0 * xy.mean() - xx.mean() - yy.mean())
    rng = np.random.default_rng(seed)
    combined = np.concatenate([x, y], axis=0)
    perm_stats = np.zeros(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(combined.shape[0])
        x_perm = combined[perm[:n_x]]
        y_perm = combined[perm[n_x:]]
        xx_p = cdist(x_perm, x_perm, metric="euclidean")
        yy_p = cdist(y_perm, y_perm, metric="euclidean")
        xy_p = cdist(x_perm, y_perm, metric="euclidean")
        np.fill_diagonal(xx_p, 0.0)
        np.fill_diagonal(yy_p, 0.0)
        perm_stats[i] = 2.0 * xy_p.mean() - xx_p.mean() - yy_p.mean()
    p_value = float((np.sum(perm_stats >= energy_obs) + 1) / (n_permutations + 1))
    return {"energy_distance": energy_obs, "p_value": p_value, "n_permutations": int(n_permutations)}


def _plot_qq(samples: np.ndarray, user_id: str, output_path: Path) -> None:
    """Generate a Q-Q plot grid for the given samples."""
    n_dim = samples.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for dim in range(n_dim):
        ax = axes[dim]
        stats.probplot(samples[:, dim], dist="norm", plot=ax)
        ax.set_title(f"dim {dim}")
    for dim in range(n_dim, len(axes)):
        axes[dim].axis("off")
    fig.suptitle(f"Q-Q plot: user {user_id}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _load_validate_models(device: torch.device, input_dim: int) -> tuple[SentenceEncoder | SentenceEncoderFull, object]:
    """Load encoder + user table from VALIDATE_INPUT_DIR."""
    if not VALIDATE_ENCODER_CKPT.exists():
        raise FileNotFoundError(f"缺少 encoder checkpoint: {VALIDATE_ENCODER_CKPT}")
    if not VALIDATE_USER_TABLE_CKPT.exists():
        raise FileNotFoundError(f"缺少 user table checkpoint: {VALIDATE_USER_TABLE_CKPT}")
    encoder = SentenceEncoder(input_dim, HIDDEN_DIM, LATENT_DIM)
    encoder.load_state_dict(torch.load(VALIDATE_ENCODER_CKPT, map_location=device))
    encoder.to(device)
    encoder.eval()
    num_users = len(load_jsonl(VALIDATE_USER_PROFILE_FILE))
    user_table = UserDistributionTable(num_users, LATENT_DIM)
    user_table.load_state_dict(torch.load(VALIDATE_USER_TABLE_CKPT, map_location=device))
    user_table.to(device)
    user_table.eval()
    return encoder, user_table


# ============================================================
# SECTION 9: Compare-specific functions (6 prior groups)
# ============================================================

def _summarize_for_compare(values: list[float]) -> dict:
    if not values:
        raise ValueError("无法汇总空数组 (compare)")
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
    }


def _params_per_user(cov_mode: str) -> int:
    """每个用户的可学习参数个数."""
    mapping = {
        "diagonal_gmm": GMM_COMPONENTS * (2 * LATENT_DIM + 1),
        "diagonal_student_t": 2 * LATENT_DIM + 1,
        "diagonal_laplace": 2 * LATENT_DIM,
        "diagonal_logistic": 2 * LATENT_DIM,
        "diagonal_student_t_gmm": GMM_COMPONENTS * (2 * LATENT_DIM + 1) + GMM_COMPONENTS,
    }
    return mapping.get(cov_mode, 2 * LATENT_DIM)


def _accept_rate_for_group(rejected: list[dict]) -> float:
    """从 rejected 记录中计算 accept rate."""
    if not rejected:
        return float("nan")
    return 1.0 - (len(rejected) / max(len(rejected), 1))


def _final_loss_for_group(detail_rows: list[dict]) -> float:
    if not detail_rows:
        return float("nan")
    return float(detail_rows[-1].get("avg_loss", float("nan")))


def _load_group(group: dict) -> dict:
    """Load one group's artifacts (profiles, sentences, summary, detail)."""
    cov = group["cov_mode"]
    encoder_dist = group.get("encoder_dist", "gaussian")
    base_dir = COMPARE_FEATURE_BASE
    if cov in {"diagonal_gmm", "diagonal_student_t_gmm"}:
        k = group.get("gmm_k", GMM_COMPONENTS)
        group["gmm_k"] = k
    summary_file = base_dir / f"{group['tag']}_summary.json"
    profiles_file = base_dir / f"{group['tag']}_user_profiles.jsonl"
    sentences_file = base_dir / f"{group['tag']}_sentences.jsonl"
    selected_file = base_dir / f"{group['tag']}_selected_query_records.jsonl"
    rejected_file = base_dir / f"{group['tag']}_rejected_query_records.jsonl"
    detail_file = base_dir / f"{group['tag']}_epoch_details.jsonl"
    summary = load_json(summary_file) if summary_file.exists() else {}
    profiles = load_jsonl(profiles_file) if profiles_file.exists() else []
    sentences = load_jsonl(sentences_file) if sentences_file.exists() else []
    rejected = load_jsonl(rejected_file) if rejected_file.exists() else []
    selected = load_jsonl(selected_file) if selected_file.exists() else []
    detail = load_jsonl(detail_file) if detail_file.exists() else []
    return {
        "group": group,
        "summary": summary,
        "profiles_count": len(profiles),
        "sentences_count": len(sentences),
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "detail_rows": detail,
        "selected": selected,
        "rejected": rejected,
    }


def _per_user_all_candidate_scores(group_data: dict) -> dict[str, float]:
    """Compute per-user top-1 candidate match score across all rounds (interpretability proxy)."""
    user_scores: dict[str, float] = {}
    for record in group_data["selected"]:
        user_scores.setdefault(record["user_id"], float(record.get("score", 0.0)))
    return user_scores


def _compute_interpretability(group_data: dict) -> dict:
    """Compute interpretability metrics for one group."""
    user_scores = _per_user_all_candidate_scores(group_data)
    scores = list(user_scores.values())
    return {
        "n_users_with_selection": len(user_scores),
        "score_summary": _summarize_for_compare(scores) if scores else {"count": 0},
        "selection_rate": float(len(user_scores) / max(group_data["profiles_count"], 1)),
    }


def compute_for_group(group: dict) -> dict:
    """Compute all metrics for one group and return a flat record."""
    data = _load_group(group)
    cov = group["cov_mode"]
    final_loss = _final_loss_for_group(data["detail_rows"])
    accept_rate = _accept_rate_for_group(data["rejected"]) if data["rejected"] else float("nan")
    interpretability = _compute_interpretability(data)
    return {
        "key": group["key"],
        "label": group["label"],
        "family": group["family"],
        "covariance_mode": cov,
        "encoder_dist": group.get("encoder_dist", "gaussian"),
        "gmm_k": group.get("gmm_k"),
        "params_per_user": _params_per_user(cov),
        "final_loss": final_loss,
        "accept_rate": accept_rate,
        "selected_count": data["selected_count"],
        "rejected_count": data["rejected_count"],
        "profiles_count": data["profiles_count"],
        "sentences_count": data["sentences_count"],
        "interpretability": interpretability,
    }


def _plot_compare_bars(records: list[dict], output_path: Path) -> None:
    """Plot stability/interpretability/complexity across groups."""
    keys = [r["key"] for r in records]
    labels = [r["label"] for r in records]
    families = [r["family"] for r in records]
    final_loss = [r["final_loss"] if not np.isnan(r["final_loss"]) else 0 for r in records]
    accept_rate = [r["accept_rate"] if not np.isnan(r["accept_rate"]) else 0 for r in records]
    params = [r["params_per_user"] for r in records]
    interp_score = [
        r["interpretability"]["score_summary"].get("mean", 0) if r["interpretability"]["score_summary"].get("count", 0) > 0 else 0
        for r in records
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    color_map = {"Gaussian": "#4C72B0", "Non-Gaussian": "#DD8452"}
    colors = [color_map.get(f, "#55A868") for f in families]

    axes[0].bar(keys, final_loss, color=colors)
    axes[0].set_title("Final loss (lower = better)")
    axes[0].set_ylabel("avg_loss")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(keys, accept_rate, color=colors)
    axes[1].set_title("Accept rate (higher = stricter)")
    axes[1].set_ylabel("accept_rate")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(keys, params, color=colors)
    axes[2].set_title("Params per user (complexity)")
    axes[2].set_ylabel("#params")
    axes[2].tick_params(axis="x", rotation=30)

    axes[3].bar(keys, interp_score, color=colors)
    axes[3].set_title("Top-1 selection score (higher = better)")
    axes[3].set_ylabel("mean score")
    axes[3].tick_params(axis="x", rotation=30)

    for ax in axes:
        ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.suptitle("Prior comparison: Gaussian vs non-Gaussian")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ============================================================
# SECTION 10: Smoke test (3 non-Gaussian modes forward+backward)
# ============================================================

def _test_smoke_mode(mode: str, label: str, device: torch.device) -> None:
    """Forward + backward smoke for one non-Gaussian mode."""
    print(f"\n=== {label} ===")
    gmm_k = SMOKE_GMM_K
    if mode == "diagonal_student_t":
        user_table = UserDistributionTableStudentT(SMOKE_NUM_USERS, SMOKE_LATENT_DIM).to(device)
    elif mode == "diagonal_laplace":
        user_table = UserDistributionTableLaplace(SMOKE_NUM_USERS, SMOKE_LATENT_DIM).to(device)
    elif mode == "diagonal_student_t_gmm":
        user_table = UserDistributionTableStudentTGMM(SMOKE_NUM_USERS, SMOKE_LATENT_DIM, gmm_k).to(device)
    else:
        raise ValueError(f"unknown mode: {mode}")
    encoder = SentenceEncoder(input_dim=SMOKE_LATENT_DIM, hidden_dim=64, latent_dim=SMOKE_LATENT_DIM).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(user_table.parameters()), lr=1e-3)

    x = torch.randn(SMOKE_BATCH, SMOKE_LATENT_DIM, device=device)
    user_idx = torch.randint(0, SMOKE_NUM_USERS, (SMOKE_BATCH,), device=device)
    all_idx = torch.arange(SMOKE_NUM_USERS, device=device)

    for step in range(3):
        sent_mu, sent_dispersion, reconstruction = encoder(x)
        if mode == "diagonal_student_t":
            log_p = student_t_log_likelihood(sent_mu, sent_dispersion, *user_table(user_idx))
            loss_match = F.cross_entropy(log_p, user_idx)
            mu, log_scale, df = user_table(all_idx)
            nu = df
            nu_factor = nu / (nu - 2.0).clamp_min(1e-3)
            log_nu_factor = torch.log(nu_factor.clamp_min(1.0)).unsqueeze(-1)  # [8, 1]
            equiv_logvar = 2.0 * log_scale + log_nu_factor
            loss_prior = standard_normal_kl(mu, equiv_logvar).mean()
        elif mode == "diagonal_laplace":
            log_p = laplace_log_likelihood(sent_mu, sent_dispersion, *user_table(user_idx))
            loss_match = F.cross_entropy(log_p, user_idx)
            mu, log_b = user_table(all_idx)
            equiv_logvar = 2.0 * log_b + float(np.log(2.0))
            loss_prior = standard_normal_kl(mu, equiv_logvar).mean()
        elif mode == "diagonal_student_t_gmm":
            log_p = student_t_gmm_log_likelihood(sent_mu, sent_dispersion, *user_table(user_idx))
            loss_match = F.cross_entropy(log_p, user_idx)
            mu_k, log_scale_k, df_k, mix = user_table(all_idx)
            nu = df_k
            nu_factor = nu / (nu - 2.0).clamp_min(1e-3)
            log_nu_factor = torch.log(nu_factor.clamp_min(1.0)).unsqueeze(-1)  # [8, 2, 1]
            equiv_logvar_k = 2.0 * log_scale_k + log_nu_factor
            per_comp_kl = standard_normal_kl(mu_k, equiv_logvar_k)
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


# ============================================================
# SECTION 11: Subcommand main functions
# ============================================================

def main_train() -> None:
    """Train VADES + 排序 query (默认 subcommand)."""
    global DEVICE
    DEVICE = infer_device()
    set_random_seed(SEED)
    ensure_directories()
    log(f"运行设备: {DEVICE}, 类别: {CATEGORY}")
    log(f"OUTPUT_TAG: {OUTPUT_TAG}, COVARIANCE_MODE: {COVARIANCE_MODE}, ENCODER_DIST: {ENCODER_DIST}")
    max_users_override = int(MAX_USERS_OVERRIDE) if MAX_USERS_OVERRIDE is not None else None

    filter_users_with_duplicate_reviews()
    user_rows = load_filtered_user_reviews()

    if SENTENCE_EXTRACT_CACHE_FILE.exists():
        log(f"已存在提取的句子缓存, 跳过抽取: {SENTENCE_EXTRACT_CACHE_FILE}")
        sentence_rows = load_jsonl(SENTENCE_EXTRACT_CACHE_FILE)
    else:
        kept_rows, excluded_rows = extract_first_twenty_sentences_for_users(user_rows)
        write_jsonl(SENTENCE_EXTRACT_CACHE_FILE, kept_rows)
        write_jsonl(EXCLUDED_USER_FILE, excluded_rows)
        sentence_rows = kept_rows

    feature_rows, feature_names = build_sentence_feature_rows(sentence_rows)
    candidate_rows = load_candidate_query_rows()
    user_ids, dataset = build_training_dataset(feature_rows, feature_names)

    encoder, user_table, detail_rows = train_vades_user_distribution_model(
        feature_rows,
        dataset,
        feature_names,
        user_ids,
        encoder_dist=ENCODER_DIST,
        covariance_mode=COVARIANCE_MODE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        user_chunk_size=USER_CHUNK_SIZE_TRAIN,
        device=DEVICE,
        encoder_ckpt_path=INPUT_DIR / "vades_encoder.pt",
        user_table_ckpt_path=INPUT_DIR / "vades_user_table.pt",
        detail_file=DETAIL_FILE,
        summary_file=SUMMARY_FILE,
        gmm_components=GMM_COMPONENTS,
        max_users_override=max_users_override,
        seed=SEED,
    )

    user_profile_rows, sentence_output = infer_user_sentence_distributions(
        feature_rows,
        dataset,
        user_ids,
        encoder,
        user_table,
        DEVICE,
        encoder_dist=ENCODER_DIST,
        covariance_mode=COVARIANCE_MODE,
        user_profile_file=USER_PROFILE_FILE,
        sentence_output_file=SENTENCE_FILE,
    )

    calibration_summary = calibrate_absolute_threshold_with_unseen_holdout(
        sentence_output,
        candidate_rows,
        dataset,
        feature_names,
        user_ids,
        user_profile_rows,
        user_table,
        DEVICE,
        covariance_mode=COVARIANCE_MODE,
        user_match_weight=USER_MATCH_WEIGHT,
        abs_threshold_quantile=ABS_THRESHOLD_QUANTILE,
        calibration_summary_file=INPUT_DIR / f"{OUTPUT_TAG}_calibration_summary.json",
    )

    selected_records, rejected_records = rank_and_select_queries(
        candidate_rows,
        sentence_output,
        user_profile_rows,
        feature_names,
        user_table,
        DEVICE,
        encoder,
        covariance_mode=COVARIANCE_MODE,
        abs_threshold_value=calibration_summary["abs_threshold_value"],
        max_rounds=REGENERATION_MAX_ROUNDS,
        candidates_per_round=CANDIDATES_PER_ROUND,
        selected_record_file=SELECTED_RECORD_FILE,
        rejected_record_file=REJECTED_RECORD_FILE,
    )

    excluded_users = []
    if EXCLUDED_USER_FILE.exists():
        excluded_users = load_jsonl(EXCLUDED_USER_FILE)

    final_summary = build_summary(
        INPUT_DIR / f"{OUTPUT_TAG}_final_summary.json",
        train_summary=load_json(SUMMARY_FILE),
        calibration_summary=calibration_summary,
        selected_records=selected_records,
        rejected_records=rejected_records,
        excluded_users=excluded_users,
        category=CATEGORY,
        feature_names=feature_names,
        user_ids=user_ids,
        cov_mode=COVARIANCE_MODE,
        encoder_dist=ENCODER_DIST,
    )

    QUERY_FILE.write_text(
        json.dumps(
            [
                {
                    "user_id": row["user_id"],
                    "asin": row["asin"],
                    "query": row["query"],
                    "score": row["score"],
                    "round_index": row["round_index"],
                }
                for row in selected_records
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"已写入 query 文件: {QUERY_FILE}")

    if not SKIP_POST_CLUSTERING:
        try:
            from cluster_strict5550_query_gmm_and_attach_retrieval import run_query_gmm_pipeline
            run_query_gmm_pipeline(
                category=CATEGORY,
                query_file=QUERY_FILE,
                write_back_to_query_file=False,
                attach_retrieval=True,
            )
            log("已运行 query GMM 聚类 + retrieval attach")
        except Exception as exc:
            log(f"query GMM pipeline 跳过: {exc!r}")
    else:
        log("按配置跳过训练后的 query GMM 聚类与 retrieval attach")


def main_probe() -> None:
    """SVR (RBF, 10-fold) probe 评估 style vector."""
    ensure_directories()
    log(f"开始 probe 评估: 类别={CATEGORY}, OUTPUT_TAG={OUTPUT_TAG}")
    x_user_ids, x_vectors = load_user_style_vectors()
    y_user_ids, y_targets, feature_names, sentence_counts = load_true_style_targets()
    shared_user_ids, x_aligned, y_aligned = align_x_y(x_user_ids, x_vectors, y_user_ids, y_targets)
    summary, per_feature_rows, fold_rows = evaluate_probe(x_aligned, y_aligned, feature_names)
    write_jsonl(PROBE_PER_FEATURE_FILE, per_feature_rows)
    write_jsonl(PROBE_FOLD_FILE, fold_rows)
    result = {
        "category": CATEGORY,
        "x_source": str(USER_PROFILE_FILE),
        "x_representation_field": PROBE_REPRESENTATION_FIELD,
        "y_source": str(SENTENCE_FILE),
        "y_aggregation": "mean_over_user_sentences",
        "evaluation_type": "10-fold cross validation on users",
        "style_vector_dim": int(x_aligned.shape[1]),
        "style_feature_dim": int(y_aligned.shape[1]),
        "user_count": int(len(shared_user_ids)),
        "feature_names": feature_names,
        "sentence_count_per_user_summary": summarize_array(sentence_counts),
        "probe_model": PROBE_MODEL_NAME,
        "main_reported_metric": "mse",
        "cross_validation": "10-fold",
        "summary": summary,
        "per_feature_file": str(PROBE_PER_FEATURE_FILE),
        "fold_file": str(PROBE_FOLD_FILE),
    }
    PROBE_SUMMARY_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入 probe summary: {PROBE_SUMMARY_FILE}")


def main_validate() -> None:
    """高斯假设检验 + Q-Q 图."""
    ensure_directories()
    log(f"开始 validate: 类别={CATEGORY}, OUTPUT_TAG={OUTPUT_TAG}")
    set_random_seed(SEED)
    device = infer_device()

    if not VALIDATE_SENTENCE_FILE.exists():
        raise FileNotFoundError(f"缺少 validate sentence 文件: {VALIDATE_SENTENCE_FILE}")
    if not VALIDATE_USER_PROFILE_FILE.exists():
        raise FileNotFoundError(f"缺少 validate user profile 文件: {VALIDATE_USER_PROFILE_FILE}")

    sentence_rows = load_jsonl(VALIDATE_SENTENCE_FILE)
    if not sentence_rows:
        raise ValueError(f"{VALIDATE_SENTENCE_FILE} 为空")
    user_profile_rows = load_jsonl(VALIDATE_USER_PROFILE_FILE)
    if not user_profile_rows:
        raise ValueError(f"{VALIDATE_USER_PROFILE_FILE} 为空")

    feature_names: list[str] | None = None
    for row in sentence_rows:
        feats = row.get("features")
        if feats:
            feature_names = list(feats.keys())
            break
    if feature_names is None:
        raise ValueError("sentence 文件中缺少 features")

    user_id_to_samples: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in sentence_rows:
        feats = row.get("features")
        if not feats:
            continue
        sample = np.asarray([float(feats[name]) for name in feature_names], dtype=np.float64)
        user_id_to_samples[row["user_id"]].append(sample)

    sampled_users: dict[str, np.ndarray] = {}
    for user_id, samples in user_id_to_samples.items():
        arr = np.asarray(samples, dtype=np.float64)
        if arr.shape[0] >= VALIDATE_N_SAMPLES_PER_USER:
            sampled_users[user_id] = arr[:VALIDATE_N_SAMPLES_PER_USER]

    log(f"sampled_users={len(sampled_users)}, dim={len(feature_names)}")

    per_user_results: list[dict] = []
    per_user_baseline_results: list[dict] = []

    rng = np.random.default_rng(SEED)
    for user_id, samples in list(sampled_users.items())[:VALIDATE_ENERGY_USER_SUBSET]:
        mardia = _mardia_kurtosis(samples)
        shapiro = _shapiro_wilk_per_dim(samples)
        ks = _ks_test_per_dim(samples)
        ad = _anderson_darling_per_dim(samples)
        baseline = rng.standard_normal(size=samples.shape)
        energy = _energy_distance_test(samples, baseline, VALIDATE_ENERGY_PERMUTATIONS, seed=SEED)

        qq_path = VALIDATE_FIG_DIR / f"qq_{user_id}.png"
        try:
            _plot_qq(samples, user_id, qq_path)
        except Exception as exc:
            qq_path = None
            log(f"Q-Q 图生成失败 (user={user_id}): {exc!r}")

        per_user_results.append(
            {
                "user_id": user_id,
                "n_samples": int(samples.shape[0]),
                "dim": int(samples.shape[1]),
                "mardia_kurtosis": mardia,
                "shapiro_wilk": shapiro,
                "ks_test": ks,
                "anderson_darling": ad,
                "energy_distance": energy,
                "qq_plot": str(qq_path) if qq_path else None,
            }
        )

        baseline_results = []
        for _ in range(VALIDATE_BASELINE_N_REPLICATES):
            baseline = rng.standard_normal(size=samples.shape)
            mardia_b = _mardia_kurtosis(baseline)
            shapiro_b = _shapiro_wilk_per_dim(baseline)
            ks_b = _ks_test_per_dim(baseline)
            ad_b = _anderson_darling_per_dim(baseline)
            baseline_results.append(
                {
                    "mardia_kurtosis": mardia_b,
                    "shapiro_wilk": shapiro_b,
                    "ks_test": ks_b,
                    "anderson_darling": ad_b,
                }
            )
        per_user_baseline_results.append(
            {"user_id": user_id, "n_baselines": len(baseline_results), "baselines": baseline_results}
        )

    summary = {
        "category": CATEGORY,
        "n_users_evaluated": len(per_user_results),
        "alpha": VALIDATE_ALPHA,
        "energy_permutations": VALIDATE_ENERGY_PERMUTATIONS,
        "summary_per_user": per_user_results,
        "summary_per_user_baseline": per_user_baseline_results,
    }
    summary_out = VALIDATE_OUTPUT_DIR / f"{VALIDATE_OUTPUT_TAG}_summary.json"
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入 validate summary: {summary_out}")
    log(f"完成 validate: n_users={len(per_user_results)}, n_baselines={len(per_user_baseline_results)}")


def main_compare() -> None:
    """6 种 prior 对比 (Gaussian vs non-Gaussian)."""
    ensure_directories()
    log(f"开始 compare: 类别={CATEGORY}")
    log(f"加载 6 组 prior 评估数据 (base={COMPARE_FEATURE_BASE})")
    records: list[dict] = []
    for group in COMPARE_GROUPS:
        log(f"  - 处理 group: {group['key']} ({group['label']}, cov={group['cov_mode']})")
        records.append(compute_for_group(group))
    output_payload = {
        "category": CATEGORY,
        "groups": records,
        "feature_dir": str(COMPARE_FEATURE_BASE),
        "covariance_mode_count": len(records),
    }
    COMPARE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_OUTPUT_JSON.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入 compare json: {COMPARE_OUTPUT_JSON}")
    try:
        _plot_compare_bars(records, COMPARE_OUTPUT_PNG)
        log(f"已写入 compare png: {COMPARE_OUTPUT_PNG}")
    except Exception as exc:
        log(f"compare 绘图失败: {exc!r}")


def main_smoke() -> None:
    """3 个非高斯模式 forward + backward smoke test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke test] using device: {device}")
    for mode, label in [
        ("diagonal_student_t", "Student-t (single)"),
        ("diagonal_laplace", "Laplace (single)"),
        ("diagonal_student_t_gmm", "Student-t GMM (K=2)"),
    ]:
        _test_smoke_mode(mode, label, device)
    print("\n[ALL PASS] 3 个新模式 forward + backward 全部正常")


# ============================================================
# SECTION 12: argparse subcommand dispatcher
# ============================================================

def main() -> None:
    """Parse subcommand and dispatch."""
    parser = argparse.ArgumentParser(
        prog="gaussian_vades",
        description="VADES single entry point (train/probe/validate/compare/smoke)",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("train", help="Train VADES + 排序 query (default)")
    sub.add_parser("probe", help="SVR 10-fold probe eval")
    sub.add_parser("validate", help="高斯假设检验 + Q-Q 图")
    sub.add_parser("compare", help="6 prior 对比")
    sub.add_parser("smoke", help="3 个非高斯模式 smoke test")
    args = parser.parse_args()
    cmd = args.cmd or "train"
    if cmd == "train":
        main_train()
    elif cmd == "probe":
        main_probe()
    elif cmd == "validate":
        main_validate()
    elif cmd == "compare":
        main_compare()
    elif cmd == "smoke":
        main_smoke()
    else:
        raise ValueError(f"未知 subcommand: {cmd}")


if __name__ == "__main__":
    main()
