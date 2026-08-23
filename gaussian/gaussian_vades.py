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
USER_MATCH_WEIGHT = float(os.environ.get("VADES_USER_MATCH_WEIGHT", "1.0"))
STYLE_RECON_WEIGHT = float(os.environ.get("VADES_STYLE_RECON_WEIGHT", "0.8"))
SENT_KL_WEIGHT = float(os.environ.get("VADES_SENT_KL_WEIGHT", "0.05"))
USER_PRIOR_KL_WEIGHT = float(os.environ.get("VADES_USER_PRIOR_KL_WEIGHT", "0.02"))
LATENT_ALIGN_WEIGHT = float(os.environ.get("VADES_LATENT_ALIGN_WEIGHT", "1.0"))
STYLE_DISTINCT_WEIGHT = float(os.environ.get("VADES_STYLE_DISTINCT_WEIGHT", "0.5"))
STYLE_ANCHOR_WEIGHT = float(os.environ.get("VADES_STYLE_ANCHOR_WEIGHT", "0.3"))
DISENTANGLE_N_CLUSTERS = int(os.environ.get("VADES_DISENTANGLE_K", "8"))
# === Prototype 模式专用常量 (替代 learnable user_offsets) ===
PROTOTYPE_NUM_CLUSTERS = int(os.environ.get("VADES_PROTOTYPE_K", "8"))
LAMBDA_USER = float(os.environ.get("VADES_LAMBDA_USER", "1.0"))              # L_user 权重 (cosine to sg(p_u))
LAMBDA_CONTRAST = float(os.environ.get("VADES_LAMBDA_CONTRAST", "0.5"))      # InfoNCE 跨用户对比损失权重
LAMBDA_GMM = float(os.environ.get("VADES_LAMBDA_GMM", "0.05"))                # user_prior KL 权重 (prototype 模式专属)
SUPPORT_FRAC = float(os.environ.get("VADES_SUPPORT_FRAC", "0.5"))            # 0.5 = 每用户 5 support + 5 query
CONTRAST_TEMPERATURE = float(os.environ.get("VADES_CONTRAST_TEMP", "0.1"))   # InfoNCE 温度
PROTOTYPE_USER_CHUNK = int(os.environ.get("VADES_PROTOTYPE_USER_CHUNK", "32"))  # 每 batch user 数
PROTOTYPE_VARIANCE_INIT_BIAS = float(os.environ.get("VADES_PROTOTYPE_VAR_BIAS", "0.0"))  # logvar 偏置初值
PROTOTYPE_VARIANT = os.environ.get("VADES_PROTOTYPE_VARIANT", "v4")  # v4=teacher_proj, v5=raw 直接 anchor
# === v4 prototype 新增: teacher + 两阶段训练 ===
LAMBDA_TEACHER = float(os.environ.get("VADES_LAMBDA_TEACHER", "10.0"))  # teacher loss 权重 (Stage 1)
LAMBDA_TEACHER_STAGE2 = float(os.environ.get("VADES_LAMBDA_TEACHER_STAGE2", "1.0"))  # teacher loss 权重 (Stage 2, 降低)
STAGE1_EPOCHS = int(os.environ.get("VADES_STAGE1_EPOCHS", "20"))  # Stage 1 epoch 数 (只 train teacher + contrastive)
STAGE1_LAMBDA_RECON = float(os.environ.get("VADES_STAGE1_LAMBDA_RECON", "0.0"))  # Stage 1 关闭重建
STAGE1_LAMBDA_KL = float(os.environ.get("VADES_STAGE1_LAMBDA_KL", "0.0"))  # Stage 1 关闭 sent KL
SENTENCES_PER_USER = int(os.environ.get("VADES_SENTENCES_PER_USER", "4"))  # 每用户采样句数 (U×K batch)
ABS_THRESHOLD_QUANTILE = float(os.environ.get("VADES_ABS_THRESHOLD_QUANTILE", "0.95"))
MAX_USERS_OVERRIDE = os.environ.get("VADES_MAX_USERS")
SKIP_POST_CLUSTERING = os.environ.get("VADES_SKIP_POST_CLUSTERING", "0") == "1"
COVARIANCE_MODE = os.environ.get("VADES_COVARIANCE_MODE", "diagonal_gmm")
VALID_COVARIANCE_MODES = {
    "diagonal", "full", "diagonal_gmm",
    "diagonal_student_t", "diagonal_laplace", "diagonal_student_t_gmm",
    "diagonal_logistic", "diagonal_prototype",
    "diagonal_residual_llm",  # Qwen hidden 空间 residual = user_hidden - neutral_hidden
    "diagonal_residual",       # 318d 句法特征 residual = user_318d - global_mean_318d
}
if COVARIANCE_MODE not in VALID_COVARIANCE_MODES:
    raise ValueError(
        f"VADES_COVARIANCE_MODE 必须是 {sorted(VALID_COVARIANCE_MODES)} 之一, 得到 {COVARIANCE_MODE}"
    )

# === diagonal_residual / diagonal_residual_llm 模式专用常量 ===
RESIDUAL_SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
RESIDUAL_HIDDEN_NPZ = os.environ.get(
    "VADES_RESIDUAL_HIDDEN_NPZ",
    "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/residual_hidden.npz",
)
RESIDUAL_HIDDEN_LAYER = int(os.environ.get("VADES_RESIDUAL_HIDDEN_LAYER", "26"))  # Phase 14.F SOTA
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
INPUT_DIR = (
    Path(os.environ["VADES_INPUT_DIR"]) if os.environ.get("VADES_INPUT_DIR")
    else REPO_ROOT / "result"
)
REVIEW_SOURCE_FILE = (
    Path(os.environ["VADES_REVIEW_SOURCE"]) if os.environ.get("VADES_REVIEW_SOURCE")
    else REPO_ROOT / "result" / "stage1_filtered_users_reviews_3000u.json"
)
SKIP_DEDUP = os.environ.get("VADES_SKIP_DEDUP", "0") == "1"
CANDIDATE_QUERY_FILE = INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
RAW_CANDIDATE_QUERY_FILE = REPO_ROOT / "result" / "query_by_expression_style_no_depth_check_10.json"
SUMMARY_FILE = INPUT_DIR / f"{OUTPUT_TAG}_summary.json"
DETAIL_FILE = INPUT_DIR / f"{OUTPUT_TAG}_epoch_details.jsonl"
USER_PROFILE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_sentences.jsonl"
EXCLUDED_USER_FILE = INPUT_DIR / f"{OUTPUT_TAG}_excluded_users.jsonl"
# 环境变量覆盖: 让 residual_llm smoke test 可以指向 spaCy 同源的 sentences cache
_SENTENCE_CACHE_OVERRIDE = os.environ.get("VADES_SENTENCE_CACHE")
if _SENTENCE_CACHE_OVERRIDE:
    SENTENCE_EXTRACT_CACHE_FILE = Path(_SENTENCE_CACHE_OVERRIDE)
else:
    SENTENCE_EXTRACT_CACHE_FILE = INPUT_DIR / f"{OUTPUT_TAG}_extracted_sentences.jsonl"
SELECTED_RECORD_FILE = INPUT_DIR / f"{OUTPUT_TAG}_selected_query_records.jsonl"
REJECTED_RECORD_FILE = INPUT_DIR / f"{OUTPUT_TAG}_rejected_query_records.jsonl"
QUERY_FILE = REPO_ROOT / "result" / f"query_by_expression_style_{OUTPUT_TAG}.json"
ENCODER_CKPT = INPUT_DIR / f"vades_encoder_{OUTPUT_TAG}.pt"
USER_TABLE_CKPT = INPUT_DIR / f"vades_user_table_{OUTPUT_TAG}.pt"

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
VALIDATE_INPUT_DIR = REPO_ROOT / "result" / "10_complexity_analysis_clause_features" / CATEGORY
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
COMPARE_FEATURE_BASE = REPO_ROOT / "result" / "10_complexity_analysis_clause_features" / CATEGORY
COMPARE_OUTPUT_DIR = REPO_ROOT / "result" / "10_complexity_analysis_clause_features" / "interpretability_paper" / CATEGORY
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


class TeacherProjector(nn.Module):
    """v4 新增: 把 encoder mu (latent_dim) 投影到 raw feature 空间 (raw_dim).

    用作 teacher loss: L_teacher = MSE(projector(mu_q), r_u)
    其中 r_u = mean(raw_features[user u]) 预计算的 raw user prototype.

    设计动机: VADES style_centers 会主导 encoder 让它学 generic style 而非 user-discriminative 特征.
    引入 teacher signal 用 47d raw features (AUC=0.65 已验证) 作为老师, 强制 encoder 学 user 信息.
    """

    def __init__(self, latent_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, teacher_dim),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.proj(mu)


class UserDistributionTablePrototype(nn.Module):
    """Support-set prototype user prior.

    核心公式 (取代 learnable user_offsets):
        p_u = (1/|S_u|) Σ_{x∈S_u} Encoder_mu(x)        # 用户支持集 prototype (外部计算传入)
        μ_u = μ_cluster[cluster_u] + A · (p_u - μ_cluster[cluster_u])
        logvar_u = softplus(B · p_u) + ε                 # 对角方差 (用户个性化)

    - style_centers: [n_clusters, latent_dim] — 共享风格簇中心 (learnable)
    - A: [latent_dim, latent_dim] — 残差投影 (learnable, 跨用户共享)
    - B: [latent_dim, latent_dim] — 方差投影 (learnable, 跨用户共享)
    - user_cluster_ids: [num_users] buffer — KMeans on raw features (固定)

    state_dict 中**没有任何 per-user 可学习参数**。这强制 encoder 必须从 query latent 学到
    user-discriminative 信号, 因为 user_table 失去了"查表"路径。
    """

    def __init__(
        self,
        num_users: int,
        latent_dim: int,
        user_cluster_ids: torch.Tensor,
        style_anchors: torch.Tensor | None = None,
        variance_bias: float = 0.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.latent_dim = latent_dim
        n_clusters = int(user_cluster_ids.max().item()) + 1
        self.n_clusters = n_clusters
        self.register_buffer("user_cluster_ids", user_cluster_ids.to(torch.long))
        # 共享风格簇中心 (init = KMeans cluster centroids if provided, else zeros)
        if style_anchors is not None:
            assert style_anchors.shape == (n_clusters, latent_dim), (
                f"style_anchors shape {style_anchors.shape} != ({n_clusters}, {latent_dim})"
            )
            self.style_centers = nn.Parameter(style_anchors.clone())
        else:
            self.style_centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))
        # A: 残差投影 (init 为 identity, 让 μ_u ≈ p_u 起步)
        self.A = nn.Parameter(torch.eye(latent_dim))
        # B: 方差投影 (init 为 0, softplus(0)=log(2)≈0.693, logvar≈0.693+variance_bias)
        self.B = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.variance_bias = float(variance_bias)

    def forward(self, p_u: torch.Tensor, cluster_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """p_u: [B, latent_dim] 用户支持集 prototype, cluster_ids: [B] 用户所属 cluster.

        返回 (mu_u, logvar_u), 均为 [B, latent_dim]。
        """
        cluster_mu = self.style_centers[cluster_ids]  # [B, D]
        delta = p_u - cluster_mu  # [B, D]
        # μ_u = μ_cluster + A · delta (A @ delta.T 转置回来)
        mu_u = cluster_mu + delta @ self.A.T  # [B, D]
        # logvar_u = softplus(B · p_u) + bias
        logvar_pre = p_u @ self.B.T  # [B, D]
        logvar_u = torch.nn.functional.softplus(logvar_pre) + self.variance_bias
        return mu_u, logvar_u

    def get_style_centers(self) -> torch.Tensor:
        return self.style_centers

    @property
    def user_mu(self) -> torch.Tensor:
        """为兼容下游代码: 暴露预计算缓存的 user_mu 矩阵 (在 inference 时由 main 填充)。

        训练阶段无 per-user params, 此属性返回的 tensor 是在 infer_user_sentence_distributions
        中通过 forward(p_u_all_users, cluster_ids_all_users) 预计算并 cache 到 self._cached_user_mu 的。
        若未 cache, 返回 None (fail-fast 由调用方处理)。
        """
        cached = getattr(self, "_cached_user_mu", None)
        if cached is None:
            raise RuntimeError(
                "UserDistributionTablePrototype.user_mu 未 cache — 必须先调用 "
                "set_user_mu_cache(mu, logvar) 或 prototype_inference_all_users(encoder, support_set)"
            )
        return cached

    @property
    def user_logvar(self) -> torch.Tensor:
        cached = getattr(self, "_cached_user_logvar", None)
        if cached is None:
            raise RuntimeError(
                "UserDistributionTablePrototype.user_logvar 未 cache — 同 user_mu 约束"
            )
        return cached

    def set_user_mu_cache(self, mu: torch.Tensor, logvar: torch.Tensor) -> None:
        """由 main 训练流程在 infer_user_sentence_distributions 阶段调用, 缓存所有用户的 mu/logvar。"""
        assert mu.shape == (self.num_users, self.latent_dim)
        assert logvar.shape == (self.num_users, self.latent_dim)
        self._cached_user_mu = mu.detach()
        self._cached_user_logvar = logvar.detach()

    def clear_user_mu_cache(self) -> None:
        self._cached_user_mu = None
        self._cached_user_logvar = None


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
    """DEPRECATED: 318d spacy 句法特征路径已废弃 (2026-08-22 cleanup)。

    sentence segmentation 现由 gaussian/extract_sentences_spacy.py 单独负责,
    本函数仅在 main_train 的 318d 模式 (diagonal / diagonal_gmm / diagonal_logistic /
    diagonal_prototype 等) 才会被调用。如果走到这里,说明 COVARIANCE_MODE 不是
    diagonal_residual_llm,请改用 VADES_COVARIANCE_MODE=diagonal_residual_llm。
    """
    raise NotImplementedError(
        "extract_sentences_from_review_text 已废弃 (318d 句法特征路径 2026-08-22 cleanup)。"
        "请使用 VADES_COVARIANCE_MODE=diagonal_residual_llm。"
    )


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

    if SKIP_DEDUP:
        log(f"VADES_SKIP_DEDUP=1, 跳过 dedup, 直接使用 {REVIEW_SOURCE_FILE}")
        return

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
    """DEPRECATED: 318d spacy 句法特征路径已废弃 (2026-08-22 cleanup)。

    仅在 main_train 的非 diagonal_residual_llm mode 调用。如果走到这里,
    请改用 VADES_COVARIANCE_MODE=diagonal_residual_llm (走 Qwen hidden 3584d)。
    """
    raise NotImplementedError(
        "build_sentence_feature_rows 已废弃 (318d 句法特征路径 2026-08-22 cleanup)。"
        "请使用 VADES_COVARIANCE_MODE=diagonal_residual_llm。"
    )


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
    """DEPRECATED: 318d spacy 句法特征路径已废弃 (2026-08-22 cleanup)。"""
    raise NotImplementedError(
        "build_candidate_feature_rows_from_raw_query_file 已废弃 "
        "(318d 句法特征路径 2026-08-22 cleanup)。"
        "请使用 VADES_COVARIANCE_MODE=diagonal_residual_llm。"
    )


def build_training_dataset(sentence_rows: list[dict], feature_names: list[str]) -> tuple[list[str], dict]:
    def _conv(v):
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        try:
            return float(s) if s else 0.0
        except ValueError:
            return 0.0  # 跳过 string fields 如 "opener", "stype"
    feature_matrix = np.asarray(
        [[_conv(row["features"].get(name, 0.0)) for name in feature_names] for row in sentence_rows],
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

def pre_cluster_users_for_disentangle(
    feature_matrix: np.ndarray,
    user_index_tensor: np.ndarray,
    n_clusters: int,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """对 user 原始特征矩阵 [num_sentences, feat_dim] + per-sentence user_index [num_sentences] 跑 KMeans, 返回:
    - user_cluster_ids: [num_users] int cluster 索引
    - style_anchors:    [n_clusters, feat_dim] cluster 中心 (作为 style_centers 初值)

    用于 diagonal_disentangled 模式的 user_cluster_ids 注入。
    """
    from sklearn.cluster import KMeans
    num_sentences = feature_matrix.shape[0]
    if num_sentences == 0:
        raise ValueError("pre_cluster_users_for_disentangle: feature_matrix 为空, 无句子可聚类")
    # 聚类前先聚合: 每个用户取 mean features (避免 sentence-level 噪声主导聚类)
    num_users = int(user_index_tensor.max()) + 1
    feat_dim = feature_matrix.shape[1]
    sum_per_user = np.zeros((num_users, feat_dim), dtype=np.float64)
    cnt_per_user = np.zeros(num_users, dtype=np.int64)
    for s in range(num_sentences):
        u = int(user_index_tensor[s])
        sum_per_user[u] += feature_matrix[s]
        cnt_per_user[u] += 1
    per_user_features = sum_per_user / np.clip(cnt_per_user, 1, None).reshape(-1, 1)
    log(f"pre_cluster: per_user_features.shape={per_user_features.shape}")
    # KMeans 自动 n_clusters 调整 (silhouette 至少要 > 1 user/cluster)
    eff_k = max(2, min(n_clusters, num_users - 1, num_users // 2)) if num_users >= 4 else 2
    km = KMeans(n_clusters=eff_k, random_state=seed, n_init=10)
    user_cluster_ids = km.fit_predict(per_user_features)
    style_anchors = km.cluster_centers_.astype(np.float32)
    log(f"pre_cluster: KMeans K={eff_k} (要求 {n_clusters}, 实际 {eff_k})")
    return user_cluster_ids.astype(np.int64), style_anchors


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
    user_cluster_ids: torch.Tensor | None = None,
    style_anchors: torch.Tensor | None = None,
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
        # 同步限制 disentangle 的 user_cluster_ids + style_anchors
        if user_cluster_ids is not None and style_anchors is not None:
            user_cluster_ids = user_cluster_ids[keep_indices_arr].contiguous()
            style_anchors = style_anchors  # anchors 仍按 cluster 维度保留

    encoder, user_table = _build_encoder_and_user_table(
        input_dim, HIDDEN_DIM, LATENT_DIM, num_users,
        encoder_dist=encoder_dist, covariance_mode=covariance_mode, gmm_components=gmm_components,
        user_cluster_ids=user_cluster_ids,
        style_anchors=style_anchors,
    )
    encoder.to(device)
    user_table.to(device)
    optimizer_params = list(encoder.parameters()) + list(user_table.parameters())
    optimizer = torch.optim.Adam(
        optimizer_params,
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

    # === v4/v5: teacher signal (v4 用 teacher_proj, v5 直接用 raw_user_proto) ===
    teacher_proj: TeacherProjector | None = None
    raw_user_protos: torch.Tensor | None = None
    if LAMBDA_TEACHER > 0:
        raw_dim = feature_matrix_raw.shape[1]
        raw_user_protos = precompute_raw_user_protos(
            feature_matrix_raw, user_index_tensor, train_mask_tensor, num_users,
        )
        if PROTOTYPE_VARIANT == "v4":
            teacher_proj = TeacherProjector(LATENT_DIM, raw_dim).to(device)
            optimizer.add_param_group({"params": list(teacher_proj.parameters())})
        log(f"  prototype variant={PROTOTYPE_VARIANT}, teacher={LAMBDA_TEACHER} (stage2={LAMBDA_TEACHER_STAGE2}), raw_dim={raw_dim}")

    scaler = GradScaler(enabled=(device.type == "cuda"))
    encoder.train()
    user_table.train()

    for epoch in range(1, epochs + 1):
        perm = train_indices[torch.randperm(train_indices.numel(), device=device)]
        total_loss = torch.zeros((), device=device)
        n_batches = 0
        if covariance_mode == "diagonal_prototype":
            # Prototype 模式: per-user batch (batch_size 被忽略, 用 PROTOTYPE_USER_CHUNK)
            n_users = num_users
            generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
            n_batches_proto = 0
            # 一次性预计算 user → train 句索引 (避免每 batch 重建 30K Python dict)
            user_to_indices_cache = _build_user_to_indices(train_mask_tensor, user_index_tensor)
            # v4/v5: stage-aware loss weights (Stage 1: 关 recon/kl, 重 teacher/contrast)
            # teacher_proj 存在与否不影响是否启用 teacher signal (v5 直接用 raw_user_proto, 不用投影层)
            is_stage1 = (LAMBDA_TEACHER > 0) and (epoch <= STAGE1_EPOCHS)
            if is_stage1:
                stage_recon_weight = STAGE1_LAMBDA_RECON
                stage_sent_kl_weight = STAGE1_LAMBDA_KL
                stage_teacher_weight = LAMBDA_TEACHER
            else:
                stage_recon_weight = style_recon_weight
                stage_sent_kl_weight = sent_kl_weight
                stage_teacher_weight = LAMBDA_TEACHER_STAGE2 if LAMBDA_TEACHER > 0 else 0.0
            if epoch == 1 or epoch == STAGE1_EPOCHS + 1:
                log(f"  stage={'1 (teacher+contrast)' if is_stage1 else '2 (full VADES)'}, "
                    f"teacher_w={stage_teacher_weight}, recon_w={stage_recon_weight}, kl_w={stage_sent_kl_weight}")
            for batch_user_start in range(0, n_users, PROTOTYPE_USER_CHUNK):
                batch_users, support_indices, query_indices = _sample_support_query_indices(
                    train_mask_tensor, user_index_tensor,
                    user_chunk=min(PROTOTYPE_USER_CHUNK, n_users - batch_user_start),
                    support_frac=SUPPORT_FRAC,
                    generator=generator,
                    user_to_indices=user_to_indices_cache,
                    sentences_per_user=SENTENCES_PER_USER,
                )
                optimizer.zero_grad()
                with autocast(enabled=(device.type == "cuda")):
                    if LAMBDA_TEACHER > 0 and PROTOTYPE_VARIANT in ("v4", "v5", "v6"):
                        # v4/v5/v6 path: teacher + stage-aware weights
                        raw_dim = feature_matrix_raw.shape[1]
                        if PROTOTYPE_VARIANT == "v6":
                            # v6: aggregation-first — teacher/contrastive 在用户级 z_u 上
                            losses = _compute_losses_for_prototype_v6(
                                support_indices, query_indices, batch_users,
                                feature_tensor, feature_matrix_raw,
                                encoder, user_table,
                                teacher_proj, raw_user_protos,
                                LATENT_DIM, raw_dim,
                                lambda_teacher=stage_teacher_weight,
                                lambda_user=LAMBDA_USER,
                                lambda_contrast=LAMBDA_CONTRAST,
                                user_prior_kl_weight=LAMBDA_GMM,
                                style_recon_weight=stage_recon_weight,
                                sent_kl_weight=stage_sent_kl_weight,
                                style_distinct_weight=STYLE_DISTINCT_WEIGHT,
                                contrast_temperature=CONTRAST_TEMPERATURE,
                            )
                        elif PROTOTYPE_VARIANT == "v5":
                            # v5: 无 teacher_proj, 直接用 raw_user_proto 作 anchor
                            losses = _compute_losses_for_prototype_v5(
                                support_indices, query_indices, batch_users,
                                feature_tensor, feature_matrix_raw,
                                encoder, user_table,
                                raw_user_protos,
                                LATENT_DIM, raw_dim,
                                lambda_teacher=stage_teacher_weight,
                                lambda_user=LAMBDA_USER,
                                lambda_contrast=LAMBDA_CONTRAST,
                                user_prior_kl_weight=LAMBDA_GMM,
                                style_recon_weight=stage_recon_weight,
                                sent_kl_weight=stage_sent_kl_weight,
                                style_distinct_weight=STYLE_DISTINCT_WEIGHT,
                                contrast_temperature=CONTRAST_TEMPERATURE,
                            )
                        else:
                            # v4: teacher_proj + learned p_u (默认)
                            losses = _compute_losses_for_prototype_v4(
                                support_indices, query_indices, batch_users,
                                feature_tensor, feature_matrix_raw,
                                encoder, user_table,
                                teacher_proj, raw_user_protos,
                                LATENT_DIM, raw_dim,
                                lambda_teacher=stage_teacher_weight,
                                lambda_user=LAMBDA_USER,
                                lambda_contrast=LAMBDA_CONTRAST,
                                user_prior_kl_weight=LAMBDA_GMM,
                                style_recon_weight=stage_recon_weight,
                                sent_kl_weight=stage_sent_kl_weight,
                                style_distinct_weight=STYLE_DISTINCT_WEIGHT,
                                contrast_temperature=CONTRAST_TEMPERATURE,
                            )
                    else:
                        losses = _compute_losses_for_prototype(
                            support_indices, query_indices, batch_users,
                            feature_tensor, feature_matrix_raw,
                            encoder, user_table, LATENT_DIM,
                            style_recon_weight=style_recon_weight,
                            sent_kl_weight=sent_kl_weight,
                            user_prior_kl_weight=LAMBDA_GMM,
                            lambda_user=LAMBDA_USER,
                            lambda_contrast=LAMBDA_CONTRAST,
                            style_distinct_weight=STYLE_DISTINCT_WEIGHT,
                            contrast_temperature=CONTRAST_TEMPERATURE,
                        )
                loss = losses["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(user_table.parameters(), 5.0)
                if teacher_proj is not None:
                    torch.nn.utils.clip_grad_norm_(teacher_proj.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss = total_loss + loss.detach()
                n_batches += 1
                n_batches_proto += 1
            log(f"  prototype: {n_batches_proto} user-batches/epoch")
        else:
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
            if teacher_proj is not None:
                teacher_ckpt_path = encoder_ckpt_path.parent / f"vades_teacher_{OUTPUT_TAG}.pt"
                torch.save(teacher_proj.state_dict(), teacher_ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log(f"early stop: best_epoch={best_epoch}, best_loss={best_loss:.4f}")
                break

    if best_epoch is not None:
        encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))
        user_table.load_state_dict(torch.load(user_table_ckpt_path, map_location=device))
        if teacher_proj is not None:
            teacher_ckpt_path = encoder_ckpt_path.parent / f"vades_teacher_{OUTPUT_TAG}.pt"
            if teacher_ckpt_path.exists():
                teacher_proj.load_state_dict(torch.load(teacher_ckpt_path, map_location=device))
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
    user_cluster_ids: torch.Tensor | None = None,
    style_anchors: torch.Tensor | None = None,
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
    elif covariance_mode == "diagonal_prototype":
        encoder = SentenceEncoder(input_dim, hidden_dim, latent_dim)
        if user_cluster_ids is None:
            raise ValueError(
                "diagonal_prototype 需要 main_train 提供 user_cluster_ids (KMeans on raw features)"
            )
        user_table = UserDistributionTablePrototype(
            num_users, latent_dim,
            user_cluster_ids=user_cluster_ids,
            style_anchors=style_anchors,
            variance_bias=PROTOTYPE_VARIANCE_INIT_BIAS,
        )
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
        # latent_align 原始代码 F.mse_loss(mu, raw_feature_targets) 因 latent_dim=20 vs input_dim=3584/318 永远 broadcast 失败 (iter 076 引入 bug)
        # 占位 0; 正确语义待重写
        latent_align = torch.tensor(0.0, device=x.device)
    else:
        if encoder_dist == "student_t":
            mu, logvar, df, reconstruction = encoder(x)
        else:
            mu, logvar, reconstruction = encoder(x)
        sent_kl = standard_normal_kl(mu, logvar).mean()

        all_user_idx = torch.arange(num_users, device=x.device)
        if covariance_mode in {"diagonal_gmm"}:
            user_mu_k, user_logvar_k, mix_logits = user_table(all_user_idx)
            per_component_kl = standard_normal_kl(user_mu_k, user_logvar_k)
            mix_probs = F.softmax(mix_logits, dim=-1)
            user_prior_kl = (mix_probs * per_component_kl).sum(dim=-1).mean()
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
        elif covariance_mode in {"diagonal_disentangled"}:
            raise ValueError(
                "diagonal_disentangled 已被 UserDistributionTablePrototype 取代, "
                "请使用 covariance_mode='diagonal_prototype'"
            )
        elif covariance_mode in {"diagonal_prototype"}:
            # prototype 模式不调用本函数; 由 _compute_losses_for_prototype 单独处理
            raise ValueError(
                "diagonal_prototype 必须通过 _compute_losses_for_prototype 计算 loss, "
                "训练循环已在 prototype 分支路由"
            )
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
        # latent_align 原始代码 F.mse_loss(mu, raw_feature_targets) 因 latent_dim=20 vs input_dim=3584/318 永远 broadcast 失败
        # 正确语义应该是 mu (latent 20d) ≈ user_mu[user_idx] (同 20d), 但 iter 076 引入时未实现, 此处占位 0
        latent_align = torch.tensor(0.0, device=x.device)

    loss = (
        user_match_weight * user_match
        + style_recon_weight * recon_loss
        + sent_kl_weight * sent_kl
        + user_prior_kl_weight * user_prior_kl
        + latent_align_weight * latent_align
    )
    if covariance_mode == "diagonal_disentangled":
        # V2: 只加 style_distinct_loss (anchor_loss 已删)
        loss = loss + STYLE_DISTINCT_WEIGHT * style_distinct_loss
        extra = {"style_distinct_loss": style_distinct_loss}
    else:
        extra = {}
    out = {
        "loss": loss,
        "user_match_loss": user_match,
        "recon_loss": recon_loss,
        "sent_kl": sent_kl,
        "user_prior_kl": user_prior_kl,
        "latent_align": latent_align,
    }
    out.update(extra)
    return out


def _build_user_to_indices(
    train_mask: torch.Tensor,
    user_index_tensor: torch.Tensor,
) -> dict[int, list[int]]:
    """一次性预计算 user → train 句子索引列表, 缓存到 epoch 外避免每 batch 重建 Python dict.

    调用方应在 train 循环外调用一次 (即 train 启动时), 然后把 cache 传给 _sample_support_query_indices.
    """
    train_mask_bool = train_mask.bool()
    train_indices = torch.nonzero(train_mask_bool, as_tuple=False).squeeze(-1)
    user_to_indices: dict[int, list[int]] = {}
    user_ids = user_index_tensor[train_indices].tolist()
    idx_list = train_indices.tolist()
    for u, idx in zip(user_ids, idx_list):
        user_to_indices.setdefault(u, []).append(idx)
    return user_to_indices


def _sample_support_query_indices(
    train_mask: torch.Tensor,
    user_index_tensor: torch.Tensor,
    user_chunk: int,
    support_frac: float,
    generator: torch.Generator | None = None,
    user_to_indices: dict[int, list[int]] | None = None,
    sentences_per_user: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-user batch sampling: 选 batch_users (user_chunk), 每用户随机抽 support/query 句。

    Args:
        train_mask: [N_total_sentences] bool — True 表示该句可作为 support/query 候选
            (在 prototype 模式下, train_mask 已限定前 10 个 train 句子/user, 不含 holdout)
        user_index_tensor: [N_total_sentences] long — 每句所属用户索引
        user_chunk: batch 内用户数
        support_frac: support 句占该用户 train 句的比例 (0.5 = 5 support + 5 query)
        generator: torch.Generator for reproducible sampling
        user_to_indices: 可选, 外部缓存的 {user_id: [idx_list]}, 避免每 batch 重建.
        sentences_per_user: 0 表示用全部 train 句; >0 则每用户随机抽 K 句再 split (v4 用, 限制 batch 大小).

    Returns:
        batch_users: [user_chunk] long — 本 batch 的用户索引
        support_indices: [user_chunk * n_support] long — support 句的全局索引
        query_indices: [user_chunk * n_query] long — query 句的全局索引
    """
    train_mask_bool = train_mask.bool()
    # 找出所有 train 用户 (至少 1 个 train 句的用户)
    train_user_ids = torch.unique(user_index_tensor[train_mask_bool])
    if user_chunk > train_user_ids.numel():
        raise ValueError(
            f"user_chunk={user_chunk} > train_user_ids={train_user_ids.numel()}, 缩小 user_chunk 或检查数据"
        )
    # 随机采样 batch_users
    perm = torch.randperm(train_user_ids.numel(), generator=generator)[:user_chunk]
    batch_users = train_user_ids[perm]

    # 对每个 batch_user, 取该用户所有 train 句子索引 (优先用 cache, 避免重建 dict)
    if user_to_indices is None:
        user_to_indices = _build_user_to_indices(train_mask_bool, user_index_tensor)

    support_indices_list: list[int] = []
    query_indices_list: list[int] = []
    for u in batch_users.tolist():
        u_indices = user_to_indices.get(int(u), [])
        n_u = len(u_indices)
        if n_u < 2:
            # 用户 train 句不足 2 句, 跳过 (支持集和查询集各需 ≥1)
            continue
        # v4: 限制每用户 K 句 (随机子采样) 以控制 batch 大小
        if sentences_per_user > 0 and sentences_per_user < n_u:
            perm_full = torch.randperm(n_u, generator=generator).tolist()
            u_indices = [u_indices[i] for i in perm_full[:sentences_per_user]]
            n_u = sentences_per_user
        n_support = max(1, int(round(n_u * support_frac)))
        n_support = min(n_support, n_u - 1)  # 至少留 1 句做 query
        n_query = n_u - n_support
        # 随机洗牌
        perm_u = torch.randperm(n_u, generator=generator).tolist()
        u_support = [u_indices[i] for i in perm_u[:n_support]]
        u_query = [u_indices[i] for i in perm_u[n_support:n_support + n_query]]
        support_indices_list.extend(u_support)
        query_indices_list.extend(u_query)

    if not support_indices_list or not query_indices_list:
        raise ValueError(
            f"_sample_support_query_indices: batch_users={batch_users.tolist()} 中无足够 train 句, "
            f"support={len(support_indices_list)}, query={len(query_indices_list)}"
        )

    support_indices = torch.as_tensor(support_indices_list, dtype=torch.long)
    query_indices = torch.as_tensor(query_indices_list, dtype=torch.long)
    return batch_users, support_indices, query_indices


def _compute_losses_for_prototype(
    support_indices: torch.Tensor,
    query_indices: torch.Tensor,
    batch_users: torch.Tensor,
    feature_tensor: torch.Tensor,
    feature_matrix_raw: np.ndarray,
    encoder: nn.Module,
    user_table: UserDistributionTablePrototype,
    latent_dim: int,
    style_recon_weight: float,
    sent_kl_weight: float,
    user_prior_kl_weight: float,
    lambda_user: float,
    lambda_contrast: float,
    style_distinct_weight: float,
    contrast_temperature: float,
) -> dict[str, torch.Tensor]:
    """Prototype 模式 loss 计算 (per-user batch)。

    流程:
      1. Encoder forward support → p_u (per batch user)
      2. Encoder forward query → z_q, logvar_q, reconstruction
      3. user_table(p_u, cluster_ids[batch_users]) → mu_u, logvar_u
      4. 5 项 loss:
         - recon_loss: decoder 重建 query 特征
         - sent_kl: KL(q(z|x) || N(0,I))
         - user_prior_kl: KL(N(mu_u, logvar_u) || N(0,I))
         - L_user: 1 - cos(z_q, sg(p_u))  ← **encoder 唯一 user-discriminative 信号来源**
         - L_contrastive: InfoNCE across batch_users (z_q vs p_u for all batch_users)
         - style_distinct: style_centers 互相远离
    """
    device = feature_tensor.device

    # === 1. Encoder forward support → p_u ===
    x_support = feature_tensor[support_indices]
    mu_support, logvar_support, _ = encoder(x_support)  # [B*n_support, D]
    # 按 batch_users 聚合: 同一个用户的 support 句放一起
    n_support_per_user = support_indices.numel() // batch_users.numel()
    p_u = mu_support.view(batch_users.numel(), n_support_per_user, latent_dim).mean(dim=1)  # [B, D]

    # === 2. Encoder forward query → z_q, logvar_q, reconstruction ===
    x_query = feature_tensor[query_indices]
    mu_q, logvar_q, reconstruction = encoder(x_query)  # [B*n_query, D]

    raw_feature_targets = torch.as_tensor(
        feature_matrix_raw[query_indices.cpu().numpy()],
        dtype=torch.float32, device=device,
    )

    # === 3. user_table forward → mu_u, logvar_u ===
    cluster_ids = user_table.user_cluster_ids[batch_users]  # [B]
    mu_u, logvar_u = user_table(p_u, cluster_ids)  # [B, D]

    # === 4. 标准 VAE 损失 (recon + sent_kl) ===
    recon_loss = F.mse_loss(reconstruction, raw_feature_targets)
    sent_kl = standard_normal_kl(mu_q, logvar_q).mean()

    # user_prior_kl: 在 prototype 模式下, mu_u 是 data-derived 不需 prior 正则 ——
    # 否则 KL 把 mu_u 拉到 0, 导致 user_mu 坍缩. 改用 mu_u 的 L2 norm 约束防止过大.
    # user_prior_kl_loss = standard_normal_kl(mu_u, logvar_u).mean()  # disabled
    user_mu_norm = (mu_u ** 2).sum(dim=-1).sqrt().mean()
    # 软约束: user_mu_norm 应保持在 ~5 左右 (与 encoder 输出同尺度)
    user_prior_kl = (user_mu_norm - 5.0).pow(2)  # 让 mu_u 不坍缩也不爆炸

    # === 5. L_user: z_q 接近 sg(p_u) (cosine 距离) ===
    # z_q: [B*n_query, D], p_u_target: [B*n_query, D] (per-query 复制所属 user 的 p_u)
    n_query_per_user = query_indices.numel() // batch_users.numel()
    p_u_target = p_u.detach().unsqueeze(1).expand(-1, n_query_per_user, -1).reshape(-1, latent_dim)
    z_q_norm = F.normalize(mu_q, dim=-1)
    p_u_norm = F.normalize(p_u_target, dim=-1)
    cos_sim = (z_q_norm * p_u_norm).sum(dim=-1)
    L_user = (1.0 - cos_sim).mean()

    # === 6. L_contrastive: InfoNCE across batch_users ===
    # logits[query_i, user_j] = cos(z_q_i, p_u_j) / temperature
    p_u_for_contrast = F.normalize(p_u, dim=-1)  # [B, D], 不用 detach (允许 encoder 学 query→contrast signal)
    logits = (z_q_norm @ p_u_for_contrast.t()) / contrast_temperature  # [B*n_query, B]
    # labels: 每个 query 的 batch 位置 = 其所属 user 在 batch_users 中的索引
    # 因为 batch_users 是随机采样无重复, query i 所属 user 的 batch 位置 = i // n_query_per_user
    B = batch_users.numel()
    label_idx = (
        torch.arange(B, device=device)
        .unsqueeze(1)
        .expand(-1, n_query_per_user)
        .reshape(-1)
        .long()
    )  # [B*n_query]
    L_contrastive = F.cross_entropy(logits, label_idx)

    # === 7. style_distinct: style_centers 互相远离 (cosine) ===
    style_centers = user_table.get_style_centers()
    n_c = style_centers.size(0)
    sc_norm = F.normalize(style_centers, dim=-1)
    sim_matrix = sc_norm @ sc_norm.t()
    off_diag = sim_matrix - torch.eye(n_c, device=sim_matrix.device)
    style_distinct_loss = (off_diag ** 2).sum() / max(1, n_c * (n_c - 1))

    # === 总 loss (注意: prototype 模式不使用 user_match, 而是 L_user + L_contrastive) ===
    loss = (
        style_recon_weight * recon_loss
        + sent_kl_weight * sent_kl
        + user_prior_kl_weight * user_prior_kl
        + lambda_user * L_user
        + lambda_contrast * L_contrastive
        + style_distinct_weight * style_distinct_loss
    )

    return {
        "loss": loss,
        "user_match_loss": L_user,  # 命名兼容旧 detail 文件
        "recon_loss": recon_loss,
        "sent_kl": sent_kl,
        "user_prior_kl": user_prior_kl,
        "latent_align": torch.tensor(0.0, device=device),  # prototype 不用 latent_align
        "L_user": L_user,
        "L_contrastive": L_contrastive,
        "style_distinct_loss": style_distinct_loss,
    }


def precompute_raw_user_protos(
    feature_matrix_raw: np.ndarray,
    user_index_tensor: torch.Tensor,
    train_mask: torch.Tensor,
    num_users: int,
) -> torch.Tensor:
    """v4 新增: 预计算 raw user prototype r_u = mean(raw_features[user u]).

    用途: teacher loss L_teacher = MSE(teacher_proj(mu_q), r_u)
    输入 feature_matrix_raw 是原始 47 维特征 (已验证 AUC=0.65), 用作"老师信号"避免 VADES
    style_centers 把 encoder 学成 generic style.

    Returns:
        raw_user_protos: [num_users, raw_dim] float32 tensor (在 CPU 上, 调用方 .to(device))
    """
    train_mask_bool = train_mask.bool()
    train_indices = torch.nonzero(train_mask_bool, as_tuple=False).squeeze(-1)
    raw_dim = feature_matrix_raw.shape[1]
    raw_protos = np.zeros((num_users, raw_dim), dtype=np.float32)
    counts = np.zeros(num_users, dtype=np.int64)
    user_ids_in_train = user_index_tensor[train_indices].cpu().numpy()
    raw_feats = feature_matrix_raw[train_indices.cpu().numpy()]
    np.add.at(raw_protos, user_ids_in_train, raw_feats)
    np.add.at(counts, user_ids_in_train, 1)
    valid = counts > 0
    raw_protos[valid] /= counts[valid, None]
    return torch.as_tensor(raw_protos, dtype=torch.float32)


def _compute_losses_for_prototype_v4(
    support_indices: torch.Tensor,
    query_indices: torch.Tensor,
    batch_users: torch.Tensor,
    feature_tensor: torch.Tensor,
    feature_matrix_raw: np.ndarray,
    encoder: nn.Module,
    user_table: UserDistributionTablePrototype,
    teacher_proj: TeacherProjector,
    raw_user_protos: torch.Tensor,
    latent_dim: int,
    raw_dim: int,
    # Stage-aware weights (caller 根据 epoch 选择)
    lambda_teacher: float,
    lambda_user: float,
    lambda_contrast: float,
    user_prior_kl_weight: float,
    style_recon_weight: float,
    sent_kl_weight: float,
    style_distinct_weight: float,
    contrast_temperature: float,
) -> dict[str, torch.Tensor]:
    """v4 prototype loss: teacher + contrastive + 两阶段动态 weight.

    Stage 1 (early epochs): 关闭 recon + sent_kl, 强化 teacher + contrastive, 让 encoder 学 user 信号.
    Stage 2 (late epochs): 加入 recon + sent_kl + GMM, 降低 teacher 权重, 让 encoder 学风格重建.

    Args:
        raw_user_protos: [num_users, raw_dim] 预计算的用户 raw feature 原型 (老师信号).
        teacher_proj: 把 encoder mu (latent_dim) 投到 raw_dim 的 MLP.
        其他参数同 _compute_losses_for_prototype.
    """
    device = feature_tensor.device

    # === 1. Encoder forward support → p_u ===
    x_support = feature_tensor[support_indices]
    mu_support, logvar_support, _ = encoder(x_support)  # [B*n_support, D]
    n_support_per_user = support_indices.numel() // batch_users.numel()
    p_u = mu_support.view(batch_users.numel(), n_support_per_user, latent_dim).mean(dim=1)  # [B, D]

    # === 2. Encoder forward query → z_q, logvar_q, reconstruction ===
    x_query = feature_tensor[query_indices]
    mu_q, logvar_q, reconstruction = encoder(x_query)  # [B*n_query, D]

    # === 3. Teacher loss: project encoder mu → raw space, 对照 raw user prototype ===
    # teacher signal: 47d raw features AUC=0.65, 用作"老师"让 encoder 学 user-discriminative 特征
    n_query_per_user = query_indices.numel() // batch_users.numel()
    # batch_users 每个 user 的 raw proto (从 raw_user_protos 取)
    batch_raw_protos = raw_user_protos[batch_users.cpu()].to(device)  # [B, raw_dim]
    # 每个 query 重复所属 user 的 raw proto
    raw_targets = batch_raw_protos.unsqueeze(1).expand(-1, n_query_per_user, -1).reshape(-1, raw_dim)
    teacher_pred = teacher_proj(mu_q)  # [B*n_query, raw_dim]
    L_teacher = F.mse_loss(teacher_pred, raw_targets)

    # === 4. user_table forward → mu_u, logvar_u (动态 GMM) ===
    cluster_ids = user_table.user_cluster_ids[batch_users]  # [B]
    mu_u, logvar_u = user_table(p_u, cluster_ids)  # [B, D]
    user_mu_norm = (mu_u ** 2).sum(dim=-1).sqrt().mean()
    user_prior_kl = (user_mu_norm - 5.0).pow(2)  # 软约束防坍缩

    # === 5. L_user: z_q 接近 sg(p_u) (cosine) ===
    p_u_target = p_u.detach().unsqueeze(1).expand(-1, n_query_per_user, -1).reshape(-1, latent_dim)
    z_q_norm = F.normalize(mu_q, dim=-1)
    p_u_norm = F.normalize(p_u_target, dim=-1)
    cos_sim = (z_q_norm * p_u_norm).sum(dim=-1)
    L_user = (1.0 - cos_sim).mean()

    # === 6. L_contrastive: InfoNCE across batch_users ===
    p_u_for_contrast = F.normalize(p_u, dim=-1)
    logits = (z_q_norm @ p_u_for_contrast.t()) / contrast_temperature  # [B*n_query, B]
    B = batch_users.numel()
    label_idx = (
        torch.arange(B, device=device)
        .unsqueeze(1)
        .expand(-1, n_query_per_user)
        .reshape(-1)
        .long()
    )
    L_contrastive = F.cross_entropy(logits, label_idx)

    # === 7. 重建 + KL (Stage 2 才用) ===
    raw_feature_targets = torch.as_tensor(
        feature_matrix_raw[query_indices.cpu().numpy()],
        dtype=torch.float32, device=device,
    )
    recon_loss = F.mse_loss(reconstruction, raw_feature_targets)
    sent_kl = standard_normal_kl(mu_q, logvar_q).mean()

    # === 8. style_distinct ===
    style_centers = user_table.get_style_centers()
    n_c = style_centers.size(0)
    sc_norm = F.normalize(style_centers, dim=-1)
    sim_matrix = sc_norm @ sc_norm.t()
    off_diag_mask = ~torch.eye(n_c, dtype=torch.bool, device=device)
    style_distinct_loss = sim_matrix[off_diag_mask].pow(2).mean()

    # === 9. 总 loss (Stage-aware weights 由调用方控制, 已传入) ===
    loss = (
        lambda_teacher * L_teacher
        + lambda_user * L_user
        + lambda_contrast * L_contrastive
        + user_prior_kl_weight * user_prior_kl
        + style_recon_weight * recon_loss
        + sent_kl_weight * sent_kl
        + style_distinct_weight * style_distinct_loss
    )

    return {
        "loss": loss,
        "L_teacher": L_teacher,
        "L_user": L_user,
        "L_contrastive": L_contrastive,
        "recon_loss": recon_loss,
        "sent_kl": sent_kl,
        "user_prior_kl": user_prior_kl,
        "style_distinct_loss": style_distinct_loss,
        "user_match_loss": L_user,  # 命名兼容旧 detail 文件
        "latent_align": torch.tensor(0.0, device=device),
    }


def _compute_losses_for_prototype_v5(
    support_indices: torch.Tensor,
    query_indices: torch.Tensor,
    batch_users: torch.Tensor,
    feature_tensor: torch.Tensor,
    feature_matrix_raw: np.ndarray,
    encoder: nn.Module,
    user_table: UserDistributionTablePrototype,
    raw_user_protos: torch.Tensor,
    latent_dim: int,
    raw_dim: int,
    # Stage-aware weights
    lambda_teacher: float,
    lambda_user: float,
    lambda_contrast: float,
    user_prior_kl_weight: float,
    style_recon_weight: float,
    sent_kl_weight: float,
    style_distinct_weight: float,
    contrast_temperature: float,
) -> dict[str, torch.Tensor]:
    """v5 prototype loss: 直接用 raw_user_proto 作 InfoNCE 正样本, 消除 teacher_proj 吸收.

    v4 失败诊断 (epoch_details):
      - L_teacher=0.40 (低), L_user=0.01 (极低), L_contrastive=1.79 (卡住)
      - 现象: encoder mu 全部聚到同一点 (p_u collapse), teacher_proj 2-layer MLP 有足够容量
              吸收 teacher loss 梯度, 没传递到 encoder.
      - 验证: encoder probe = 0.02% (随机).

    v5 修复:
      1. 移除 teacher_proj. 改用直接 L_teacher = 1 - cos_sim(mu_q, raw_target) — 无投影层吸收梯度.
      2. 用 raw_user_proto (数据驱动, 用户特异) 作 InfoNCE 正样本 — 不依赖 learned p_u (会 collapse).
      3. 保留 learned p_u 仅供 user_table forward (动态 GMM).

    Stage 1 (早期): L_teacher + L_contrastive + L_user, 关闭 recon/KL/GMM — 让 encoder 学 user 信号.
    Stage 2 (后期): 加入全部 VADES losses.
    """
    device = feature_tensor.device

    # === 1. Encoder forward support → p_u (给 GMM 用) ===
    x_support = feature_tensor[support_indices]
    mu_support, logvar_support, _ = encoder(x_support)
    n_support_per_user = support_indices.numel() // batch_users.numel()
    p_u = mu_support.view(batch_users.numel(), n_support_per_user, latent_dim).mean(dim=1)

    # === 2. Encoder forward query → mu_q, logvar_q, reconstruction ===
    x_query = feature_tensor[query_indices]
    mu_q, logvar_q, reconstruction = encoder(x_query)

    # === 3. user_table forward → mu_u (动态 GMM, 给 user_mu 派生) ===
    cluster_ids = user_table.user_cluster_ids[batch_users]
    mu_u, logvar_u = user_table(p_u, cluster_ids)
    user_mu_norm = (mu_u ** 2).sum(dim=-1).sqrt().mean()
    user_prior_kl = (user_mu_norm - 5.0).pow(2)

    # === 4. L_teacher (无投影层): 直接对齐 mu_q 与 raw_user_proto 的方向 ===
    n_query_per_user = query_indices.numel() // batch_users.numel()
    batch_raw_protos = raw_user_protos[batch_users.cpu()].to(device)  # [B, raw_dim]
    raw_targets = batch_raw_protos.unsqueeze(1).expand(-1, n_query_per_user, -1).reshape(-1, raw_dim)
    # mu_q 是 latent_dim 维, raw_target 是 raw_dim 维. 需要先 normalize 再做 MSE
    # 但维度不匹配, 所以用 cosine: 把 raw 也 normalize 后当 target, mu_q 也 normalize
    # 然后 MSE 在 normalized 空间
    mu_q_normed = F.normalize(mu_q, dim=-1)
    raw_targets_normed = F.normalize(raw_targets, dim=-1)
    # Cosine loss on direction: 1 - cos_sim
    cos_teacher = (mu_q_normed * raw_targets_normed).sum(dim=-1)
    L_teacher = (1.0 - cos_teacher).mean()

    # === 5. L_user: mu_q 接近 p_u (cosine) ===
    p_u_target = p_u.detach().unsqueeze(1).expand(-1, n_query_per_user, -1).reshape(-1, latent_dim)
    p_u_norm = F.normalize(p_u_target, dim=-1)
    L_user = (1.0 - (mu_q_normed * p_u_norm).sum(dim=-1)).mean()

    # === 6. L_contrastive: InfoNCE 直接用 raw_user_proto 作正样本 (绕开 learned p_u collapse) ===
    # logits[query_i, user_j] = cos_sim(mu_q_i, raw_user_proto_j) / T
    raw_protos_normed = F.normalize(batch_raw_protos, dim=-1)  # [B, raw_dim]
    # 因维度不同 (latent_dim vs raw_dim), 用一个小的共享投影 W
    # 但为了避免 v4 的 teacher_proj 吸收问题, W 直接用 encoder 的 mu_head 权重 transpose 做投影
    # 或者干脆用 random fixed projection (让对比只关注 mu_q 自身的 user-discriminative 结构)
    # 实际方案: 维度相同时才能 dot. 这里 latent_dim=20=raw_dim=20, 不用投影!
    # 检查 latent_dim == raw_dim
    if latent_dim == raw_dim:
        raw_for_contrast = raw_protos_normed  # [B, latent_dim]
    else:
        # 维度不匹配时用 identity-padded projection (zero-fill to max dim)
        max_dim = max(latent_dim, raw_dim)
        if latent_dim < max_dim:
            mu_q_proj = F.pad(mu_q_normed, (0, max_dim - latent_dim))
        else:
            mu_q_proj = mu_q_normed[:, :max_dim]
        if raw_dim < max_dim:
            raw_for_contrast = F.pad(raw_protos_normed, (0, max_dim - raw_dim))
        else:
            raw_for_contrast = raw_protos_normed[:, :max_dim]
        mu_q_for_contrast = mu_q_proj
    if latent_dim == raw_dim:
        mu_q_for_contrast = mu_q_normed
    logits = (mu_q_for_contrast @ raw_for_contrast.t()) / contrast_temperature  # [B*n_query, B]
    B = batch_users.numel()
    label_idx = (
        torch.arange(B, device=device)
        .unsqueeze(1)
        .expand(-1, n_query_per_user)
        .reshape(-1)
        .long()
    )
    L_contrastive = F.cross_entropy(logits, label_idx)

    # === 7. 重建 + KL (Stage 2 才用) ===
    raw_feature_targets = torch.as_tensor(
        feature_matrix_raw[query_indices.cpu().numpy()],
        dtype=torch.float32, device=device,
    )
    recon_loss = F.mse_loss(reconstruction, raw_feature_targets)
    sent_kl = standard_normal_kl(mu_q, logvar_q).mean()

    # === 8. style_distinct ===
    style_centers = user_table.get_style_centers()
    n_c = style_centers.size(0)
    sc_norm = F.normalize(style_centers, dim=-1)
    sim_matrix = sc_norm @ sc_norm.t()
    off_diag_mask = ~torch.eye(n_c, dtype=torch.bool, device=device)
    style_distinct_loss = sim_matrix[off_diag_mask].pow(2).mean()

    # === 9. 总 loss ===
    loss = (
        lambda_teacher * L_teacher
        + lambda_user * L_user
        + lambda_contrast * L_contrastive
        + user_prior_kl_weight * user_prior_kl
        + style_recon_weight * recon_loss
        + sent_kl_weight * sent_kl
        + style_distinct_weight * style_distinct_loss
    )

    return {
        "loss": loss,
        "L_teacher": L_teacher,
        "L_user": L_user,
        "L_contrastive": L_contrastive,
        "recon_loss": recon_loss,
        "sent_kl": sent_kl,
        "user_prior_kl": user_prior_kl,
        "style_distinct_loss": style_distinct_loss,
        "user_match_loss": L_user,
        "latent_align": torch.tensor(0.0, device=device),
    }


def _compute_losses_for_prototype_v6(
    support_indices: torch.Tensor,
    query_indices: torch.Tensor,
    batch_users: torch.Tensor,
    feature_tensor: torch.Tensor,
    feature_matrix_raw: np.ndarray,
    encoder: nn.Module,
    user_table: UserDistributionTablePrototype,
    teacher_proj: TeacherProjector | None,
    raw_user_protos: torch.Tensor,
    latent_dim: int,
    raw_dim: int,
    # Stage-aware weights
    lambda_teacher: float,
    lambda_user: float,
    lambda_contrast: float,
    user_prior_kl_weight: float,
    style_recon_weight: float,
    sent_kl_weight: float,
    style_distinct_weight: float,
    contrast_temperature: float,
) -> dict[str, torch.Tensor]:
    """v6 aggregation-first VADES: teacher/contrastive 在用户级聚合 prototype 上.

    关键设计 (vs v5):
      1. support 多句 → z_u = mean(encoder mu)  ← user-level stable prototype (消除单句噪声)
      2. L_teacher 在 z_u 上 vs raw_user_proto (用户级对齐)
      3. L_contrastive 在 batch 内 z_u 上 (用户级互证, 不是单句)
      4. 每个 query sentence → L_user (接近自己用户的 z_u)
      5. user_table(z_u, cluster) → mu_u (GMM 派生)

    评估要求 (用户指令):
      - 单句 probe 只作辅助诊断
      - 用户级 z_u 的 AUC 必须达到 raw baseline 0.65
      - self-cross gap CI > 0
      - permutation p < 0.05
      - 训练和评估必须使用同一个 latent 空间 (cosine/L2/Mahalanobis 一致)
    """
    device = feature_tensor.device

    # === 1. Forward support → mu_support, aggregate to z_u ===
    x_support = feature_tensor[support_indices]
    mu_support, logvar_support, _ = encoder(x_support)  # [B*n_support, D]
    n_support_per_user = support_indices.numel() // batch_users.numel()
    B = batch_users.numel()
    mu_support_grouped = mu_support.view(B, n_support_per_user, latent_dim)
    # z_u = mean of support mu (user-level stable prototype, no single-sentence noise)
    z_u = mu_support_grouped.mean(dim=1)  # [B, D]

    # === 2. Forward query → mu_q for reconstruction + per-sentence L_user ===
    x_query = feature_tensor[query_indices]
    mu_q, logvar_q, reconstruction = encoder(x_query)  # [B*n_query, D]
    n_query_per_user = query_indices.numel() // B

    # === 3. user_table forward: mu_u = mu_cluster + A(z_u - mu_cluster) ===
    cluster_ids = user_table.user_cluster_ids[batch_users]
    mu_u, logvar_u = user_table(z_u, cluster_ids)  # [B, D]
    user_mu_norm = (mu_u ** 2).sum(dim=-1).sqrt().mean()
    user_prior_kl = (user_mu_norm - 5.0).pow(2)

    # === 4. L_teacher on z_u (USER-level) vs raw_user_proto ===
    raw_targets = raw_user_protos[batch_users.cpu()].to(device)  # [B, raw_dim]
    if teacher_proj is not None:
        teacher_pred = teacher_proj(z_u)  # [B, raw_dim]
        L_teacher = F.mse_loss(teacher_pred, raw_targets)
    else:
        # 直接 cosine (与 v5 相同, 但作用于 aggregated z_u)
        z_u_norm = F.normalize(z_u, dim=-1)
        raw_targets_norm = F.normalize(raw_targets, dim=-1)
        L_teacher = (1.0 - (z_u_norm * raw_targets_norm).sum(dim=-1)).mean()

    # === 5. L_contrastive on z_u (USER-LEVEL across batch users) ===
    z_u_norm = F.normalize(z_u, dim=-1)
    logits = (z_u_norm @ z_u_norm.t()) / contrast_temperature  # [B, B]
    labels = torch.arange(B, device=device)
    L_contrastive = F.cross_entropy(logits, labels)

    # === 6. L_user: 每个 query sentence 接近自己用户的 z_u (cosine, detach z_u) ===
    z_u_target = z_u.detach().unsqueeze(1).expand(-1, n_query_per_user, -1).reshape(-1, latent_dim)
    z_u_target_norm = F.normalize(z_u_target, dim=-1)
    mu_q_norm = F.normalize(mu_q, dim=-1)
    L_user = (1.0 - (mu_q_norm * z_u_target_norm).sum(dim=-1)).mean()

    # === 7. Reconstruction (per query sentence) ===
    raw_feature_targets = torch.as_tensor(
        feature_matrix_raw[query_indices.cpu().numpy()],
        dtype=torch.float32, device=device,
    )
    recon_loss = F.mse_loss(reconstruction, raw_feature_targets)

    # === 8. KL (per query sentence) ===
    sent_kl = standard_normal_kl(mu_q, logvar_q).mean()

    # === 9. style_distinct (cosine distinct on style_centers) ===
    style_centers = user_table.get_style_centers()
    n_c = style_centers.size(0)
    sc_norm = F.normalize(style_centers, dim=-1)
    sim_matrix = sc_norm @ sc_norm.t()
    off_diag_mask = ~torch.eye(n_c, dtype=torch.bool, device=device)
    style_distinct_loss = sim_matrix[off_diag_mask].pow(2).mean()

    # === 10. Total loss (stage-aware weights 由调用方传入) ===
    loss = (
        lambda_teacher * L_teacher
        + lambda_user * L_user
        + lambda_contrast * L_contrastive
        + user_prior_kl_weight * user_prior_kl
        + style_recon_weight * recon_loss
        + sent_kl_weight * sent_kl
        + style_distinct_weight * style_distinct_loss
    )

    return {
        "loss": loss,
        "L_teacher": L_teacher,
        "L_user": L_user,
        "L_contrastive": L_contrastive,
        "recon_loss": recon_loss,
        "sent_kl": sent_kl,
        "user_prior_kl": user_prior_kl,
        "style_distinct_loss": style_distinct_loss,
        "user_match_loss": L_user,
        "latent_align": torch.tensor(0.0, device=device),
    }


def _get_user_table_num_users(user_table: nn.Module) -> int:
    """统一获取 user_table 的 num_users, 兼容 nn.Embedding / Tensor / Prototype。"""
    if isinstance(user_table, UserDistributionTablePrototype):
        return int(user_table.num_users)
    if isinstance(user_table.user_mu, nn.Embedding):
        return int(user_table.user_mu.weight.shape[0])
    return int(user_table.user_mu.shape[0])


def precompute_prototype_user_mu(
    encoder: SentenceEncoder,
    user_table: UserDistributionTablePrototype,
    dataset: dict,
    feature_matrix_raw: np.ndarray,
    user_ids: list[str],
    device: torch.device,
    encoder_dist: str,
    train_mask: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对每个用户用其**全部训练句**计算 p_u, 再过 user_table → (mu_u, logvar_u)。

    注意: 此函数与训练时随机采 support 不同 — 推理时用全部 10 train 句, 让 p_u 更稳定。
    Returns:
        mu_all: [num_users, latent_dim]
        logvar_all: [num_users, latent_dim]
    """
    feature_matrix = dataset["scaled_features"].astype(np.float32)
    user_indices = dataset["user_indices"]
    train_mask_t = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    feature_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    user_index_tensor = torch.as_tensor(user_indices, dtype=torch.long, device=device)

    train_indices = torch.nonzero(train_mask_t, as_tuple=False).squeeze(-1)
    num_users = len(user_ids)

    encoder.eval()
    with torch.no_grad():
        # 一次性 encoder forward 所有 train 句子
        mu_out = []
        for chunk_start in range(0, train_indices.numel(), USER_CHUNK_SIZE_TRAIN):
            chunk_end = min(chunk_start + USER_CHUNK_SIZE_TRAIN, train_indices.numel())
            batch_idx = train_indices[chunk_start:chunk_end]
            x = feature_tensor[batch_idx]
            mu_chunk, _, _ = encoder(x)
            mu_out.append(mu_chunk)
        mu_full = torch.cat(mu_out, dim=0)  # [n_train_sentences, D]
        train_user_ids = user_index_tensor[train_indices]  # [n_train_sentences]

        # 每个用户聚合 p_u = mean(mu_full[train_user_ids == u])
        p_u_all = torch.zeros(num_users, mu_full.size(1), device=device, dtype=mu_full.dtype)
        counts = torch.zeros(num_users, device=device, dtype=mu_full.dtype)
        p_u_all.index_add_(0, train_user_ids.long(), mu_full)
        counts.index_add_(0, train_user_ids.long(), torch.ones_like(train_user_ids, dtype=mu_full.dtype))
        p_u_all = p_u_all / counts.clamp_min(1.0).unsqueeze(-1)

        # user_table forward → (mu_u, logvar_u)
        all_user_idx = torch.arange(num_users, device=device)
        cluster_ids = user_table.user_cluster_ids[all_user_idx]
        mu_all, logvar_all = user_table(p_u_all, cluster_ids)

    return mu_all, logvar_all


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
    feature_matrix_raw = dataset["feature_matrix"]
    feature_tensor = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    user_index_tensor = torch.as_tensor(user_indices, dtype=torch.long, device=device)

    encoder.eval()
    user_table.eval()
    if covariance_mode == "diagonal_prototype":
        # prototype 模式: 直接从全部 train 句计算 p_u, 经 user_table 派生 (mu_u, logvar_u)
        train_mask_arr = dataset["train_mask"]
        user_mu_tensor, user_logvar_tensor = precompute_prototype_user_mu(
            encoder, user_table, dataset, feature_matrix_raw, user_ids, device, encoder_dist,
            train_mask=train_mask_arr,
        )
        # cache 到 user_table 以兼容下游 (user_mu / user_logvar 属性访问)
        user_table.set_user_mu_cache(user_mu_tensor, user_logvar_tensor)
        all_user_idx = torch.arange(user_table.num_users, device=device)
        mu_full = None  # prototype 模式不需要逐句 mu, 留 None
        user_mu_array = user_mu_tensor.detach().cpu().numpy()
    else:
        all_user_idx = torch.arange(
            user_table.user_mu.weight.shape[0]
            if isinstance(user_table.user_mu, nn.Embedding)
            else user_table.user_mu.shape[0],
            device=device,
        )
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
                mix_probs = F.softmax(mix_logits, dim=-1)
                user_mu_tensor = (user_mu_k * mix_probs.unsqueeze(-1)).sum(dim=1)
            elif covariance_mode in {"diagonal_student_t"}:
                user_mu, user_log_scale, user_df = user_table(all_user_idx)
                user_mu_tensor = user_mu
            elif covariance_mode in {"diagonal_laplace"}:
                user_mu, user_log_b = user_table(all_user_idx)
                user_mu_tensor = user_mu
            elif covariance_mode in {"diagonal_student_t_gmm"}:
                user_mu_k, user_log_scale_k, user_df_k, mix_logits = user_table(all_user_idx)
                mix_probs = F.softmax(mix_logits, dim=-1)
                user_mu_tensor = (user_mu_k * mix_probs.unsqueeze(-1)).sum(dim=1)
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
            if mu_full is not None:
                mu_vec = mu_full[sentence_idx_in_full]
            else:
                # prototype 模式: 仍需逐句 mu (用于 V3 校准与评估)
                x_single = feature_tensor[sentence_idx_in_full:sentence_idx_in_full + 1]
                with torch.no_grad():
                    mu_single, _, _ = encoder(x_single)
                mu_vec = mu_single.detach().cpu().numpy().squeeze(0)
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
    user_index_tensor: torch.Tensor,
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
    encoder_path = ENCODER_CKPT
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

        all_user_idx = torch.arange(
            _get_user_table_num_users(user_table),
            device=device,
        )
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
                elif covariance_mode == "diagonal_prototype":
                    # prototype 模式: 用 cache 的 (mu_u, logvar_u) 直接做 Gaussian KL
                    user_mu = user_table.user_mu.to(mu_chunk.device)
                    user_logvar = user_table.user_logvar.to(mu_chunk.device)
                    log_p = -diagonal_gaussian_kl(
                        mu_chunk.unsqueeze(1),
                        torch.zeros_like(mu_chunk).unsqueeze(1),
                        user_mu.unsqueeze(0),
                        user_logvar.unsqueeze(0),
                    )
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

    all_user_idx = torch.arange(
        _get_user_table_num_users(user_table),
        device=device,
    )
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
            elif covariance_mode == "diagonal_prototype":
                # prototype 模式: 用 cache 的 (mu_u, logvar_u) 直接做 Gaussian KL
                user_mu = user_table.user_mu.to(candidate_mu.device)
                user_logvar = user_table.user_logvar.to(candidate_mu.device)
                all_scores = -diagonal_gaussian_kl(
                    candidate_mu.unsqueeze(1),
                    torch.zeros_like(candidate_mu).unsqueeze(1),
                    user_mu.unsqueeze(0),
                    user_logvar.unsqueeze(0),
                ).cpu().numpy()
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
    """Make sure all parent dirs exist for input/output files.

    Iterates a set of parent directories (deduplicated) so that we never
    accidentally create a directory at a file path (which would cause
    IsADirectoryError when write_jsonl tries to open that path as a file).
    """
    parent_dirs = {
        SUMMARY_FILE.parent, DETAIL_FILE.parent, USER_PROFILE_FILE.parent,
        SENTENCE_FILE.parent, EXCLUDED_USER_FILE.parent, SELECTED_RECORD_FILE.parent,
        REJECTED_RECORD_FILE.parent, QUERY_FILE.parent, PROBE_SUMMARY_FILE.parent,
        PROBE_PER_FEATURE_FILE.parent, PROBE_FOLD_FILE.parent,
        VALIDATE_OUTPUT_DIR, VALIDATE_FIG_DIR, COMPARE_OUTPUT_DIR,
    }
    for parent in parent_dirs:
        parent.mkdir(parents=True, exist_ok=True)


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

    # ==== 按 MAX_USERS 提前裁剪 (所有模式共用) ====
    _max_users_set: set[str] | None = None
    if MAX_USERS_OVERRIDE is not None:
        m = int(MAX_USERS_OVERRIDE)
        if m > 0:
            _max_users_set = set()  # 临时占位，后续 sentence_rows 裁剪时填充

    if COVARIANCE_MODE == "diagonal_residual":
        # 对角 residual 模式：sentence-level cache → user-level ALL_FEATS_V2(297d) 聚合 → residual
        spacy_sents = RESIDUAL_SCRATCH / "sentences_for_rewrite_10k.jsonl"
        feat_cache = RESIDUAL_SCRATCH / "sentences_318d_cache.jsonl.gz"
        if not spacy_sents.exists():
            raise FileNotFoundError(f"diagonal_residual 模式需要: {spacy_sents}")
        log(f"[diagonal_residual] 加载 spaCy 句子: {spacy_sents}")

        # --- ALL_FEATS_V2 常量（内联自 extract_syntactic_features.py） ---
        _POS_TAGS = ("NOUN","VERB","ADJ","ADV","PRON","DET","ADP","CONJ","AUX","NUM","INTJ","PART","PUNCT","X","CCONJ","SCONJ")
        _POS_BIGRAMS = (("DET","NOUN"),("PRON","VERB"),("AUX","VERB"),("VERB","DET"),("VERB","NOUN"),("ADJ","NOUN"),("NOUN","VERB"),("ADV","VERB"),("ADP","DET"),("ADP","NOUN"),("PRON","AUX"),("DET","ADJ"),("NOUN","ADP"),("VERB","ADP"),("VERB","PRON"),("ADV","ADJ"),("ADJ","ADP"),("NOUN","CCONJ"),("VERB","CCONJ"),("NOUN","SCONJ"),("VERB","SCONJ"),("AUX","ADJ"),("AUX","NOUN"),("PRON","VERB"),("DET","NOUN"),("ADV","ADJ"),("AUX","PART"),("PART","VERB"),("NUM","NOUN"),("PRON","ADP"))
        _POS_TRIGRAMS = (("DET","NOUN","VERB"),("PRON","AUX","VERB"),("PRON","AUX","ADJ"),("AUX","VERB","DET"),("AUX","VERB","NOUN"),("AUX","VERB","ADV"),("VERB","DET","NOUN"),("ADP","DET","NOUN"),("VERB","ADP","DET"),("ADP","DET","ADJ"),("DET","ADJ","NOUN"),("VERB","PRON","AUX"),("AUX","ADJ","ADP"),("ADV","VERB","DET"),("AUX","VERB","PRON"),("VERB","CCONJ","VERB"),("NOUN","CCONJ","NOUN"),("VERB","SCONJ","VERB"))
        _DEP_RELS = ("nsubj","nsubjpass","obj","iobj","cobj","attr","aux","auxpass","ROOT","det","poss","amod","advmod","nmod","appos","nummod","acl","relcl","ccomp","xcomp","advcl","conj","cc","punct","case","mark","compound","fixed","flat")
        _DEP_BIGRAMS = (("nsubj","VERB"),("VERB","obj"),("nsubj","AUX"),("AUX","VERB"),("det","NOUN"),("nmod","NOUN"),("amod","NOUN"),("compound","NOUN"),("nsubj","ADV"),("ADV","VERB"),("ROOT","nsubj"),("ROOT","VERB"),("VERB","ADP"),("ADP","NOUN"),("nsubj","ADP"),("cc","CONJ"),("conj","CONJ"),("ROOT","ccomp"),("ROOT","xcomp"),("advcl","VERB"))
        _MAIN_CLAUSE_PATTERNS = ("SVO","SVC","SV","SVA","SVOA","SVOC","SVOO","SVO_IOBJ","EXISTS")
        _CLAUSE_PAIRS = (("acl","relcl"),("acl","ccomp"),("acl","xcomp"),("acl","advcl"),("relcl","ccomp"),("relcl","xcomp"),("relcl","advcl"),("ccomp","xcomp"),("ccomp","advcl"),("xcomp","advcl"))
        _PUNCT_TAGS = (",",".",":",";","!","?","'","\"","-","(",")")
        _PUNCT_BIGRAMS = (("(",")"),("(",")"),("(",")"),("(",")"),("(",")"))

        def _build_all_feats_v2():
            feats = []
            feats += ["clause_rate","acl_rate","advcl_rate","ccomp_rate","xcomp_rate","relcl_rate","modifier_density","coordination_density","mean_dep_distance","depth_variance","median_dep_depth","passive_rate","interrogative_rate","conditional_rate"]
            feats += [f"opener_{p}" for p in _POS_TAGS]
            feats += ["senttype_simple","senttype_conjunctive","senttype_complex"]
            feats += [f"pos_{p}" for p in _POS_TAGS]
            feats += [f"posbg_{bg[0]}_{bg[1]}" for bg in _POS_BIGRAMS]
            feats += [f"postg_{tg[0]}_{tg[1]}_{tg[2]}" for tg in _POS_TRIGRAMS]
            feats += [f"dep_{d}" for d in _DEP_RELS]
            feats += [f"depbg_{bg[0]}_{bg[1]}" for bg in _DEP_BIGRAMS]
            feats += [f"main_{p}" for p in _MAIN_CLAUSE_PATTERNS]
            feats += [f"clpair_{pair[0]}_{pair[1]}" for pair in _CLAUSE_PAIRS]
            feats += ["nest_max","nest_mean","nest_std","nest_d1","nest_d2","nest_d3","nest_d4","nest_d5","nest_ge2","nest_ge3"]
            for pos in (1,2,3):
                for p in _POS_TAGS:
                    feats.append(f"open_p{pos}_{p}")
                    feats.append(f"close_p{pos}_{p}")
            feats += ["dist_0_1","dist_1_2","dist_2_3","dist_3_5","dist_5_100"]
            feats += ["depth_eq1","depth_eq2","depth_eq3","depth_eq4","depth_eq5"]
            feats.append("n_punct_total")
            feats += [f"punct_{p}" for p in _PUNCT_TAGS]
            feats += [f"punctbg_{pb[0]}_{pb[1]}" for pb in _PUNCT_BIGRAMS if len(pb[0])==1]
            return feats

        ALL_FEATS_V2: list = _build_all_feats_v2()
        _NAME_TO_IDX = {n: i for i, n in enumerate(ALL_FEATS_V2)}

        # bucket A: rate-per-token; bucket B: rate-per-sent; bucket C: mean-over-sent
        _RATE_TOK_KEYS = {"clause_rate":("n_clause",),"acl_rate":("acl",),"advcl_rate":("advcl",),"ccomp_rate":("ccomp",),"xcomp_rate":("xcomp",),"relcl_rate":("relcl",),"modifier_density":("n_mod",),"coordination_density":("n_coord",),"mean_dep_distance":("mean_dist",),"depth_variance":("depth_var",),"passive_rate":("has_passive",),"interrogative_rate":("is_interrog",),"conditional_rate":("has_cond",)}
        _RATE_PER_TOK = set()
        _OPEN_CLOSE_KEYS = []
        for pos in (1,2,3):
            for p in _POS_TAGS:
                _OPEN_CLOSE_KEYS.append(f"open_p{pos}_{p}")
                _OPEN_CLOSE_KEYS.append(f"close_p{pos}_{p}")
        _DIST_DEPTH_KEYS = [f"dist_{lo}_{hi}" for lo,hi in [(0,1),(1,2),(2,3),(3,5),(5,100)]] + [f"depth_eq{d}" for d in (1,2,3,4,5)]
        _RATE_TOK_IDX = sorted(_NAME_TO_IDX[n] for n in list(_RATE_TOK_KEYS.keys()) + _OPEN_CLOSE_KEYS + _DIST_DEPTH_KEYS)
        _RATE_TOK_KEY_LOOKUP = {}
        for n, (k,) in _RATE_TOK_KEYS.items():
            _RATE_TOK_KEY_LOOKUP[_NAME_TO_IDX[n]] = k
        for n in _OPEN_CLOSE_KEYS:
            _RATE_TOK_KEY_LOOKUP[_NAME_TO_IDX[n]] = n
        for n in _DIST_DEPTH_KEYS:
            _RATE_TOK_KEY_LOOKUP[_NAME_TO_IDX[n]] = n
        _RATE_PER_SENT = {"clause_rate","acl_rate","advcl_rate","ccomp_rate","xcomp_rate","relcl_rate","n_punct_total"}
        _RATE_SENT_IDX = sorted(_NAME_TO_IDX[n] for n in list(_RATE_PER_SENT) + ["senttype_simple","senttype_conjunctive","senttype_complex"])
        _RATE_SENT_KEY_LOOKUP = {}
        for n in _RATE_PER_SENT:
            _RATE_SENT_KEY_LOOKUP[_NAME_TO_IDX[n]] = n
        for n in ["senttype_simple","senttype_conjunctive","senttype_complex"]:
            _RATE_SENT_KEY_LOOKUP[_NAME_TO_IDX[n]] = "stype_" + n[len("senttype_"):]
        _MEAN_SENT_KEYS = {"median_dep_depth":"max_depth","nest_max":"nest_max","nest_mean":"nest_mean","nest_std":"nest_std","nest_d1":"nest_d1","nest_d2":"nest_d2","nest_d3":"nest_d3","nest_d4":"nest_d4","nest_d5":"nest_d5","nest_ge2":"nest_ge2","nest_ge3":"nest_ge3"}
        for p in _POS_TAGS:
            _MEAN_SENT_KEYS[f"pos_{p}"] = f"pos_{p}"
        for bg in _POS_BIGRAMS:
            _MEAN_SENT_KEYS[f"posbg_{bg[0]}_{bg[1]}"] = f"posbg_{bg[0]}_{bg[1]}"
        for tg in _POS_TRIGRAMS:
            _MEAN_SENT_KEYS[f"postg_{tg[0]}_{tg[1]}_{tg[2]}"] = f"postg_{tg[0]}_{tg[1]}_{tg[2]}"
        for d in _DEP_RELS:
            _MEAN_SENT_KEYS[f"dep_{d}"] = f"dep_{d}"
        for bg in _DEP_BIGRAMS:
            _MEAN_SENT_KEYS[f"depbg_{bg[0]}_{bg[1]}"] = f"depbg_{bg[0]}_{bg[1]}"
        for p in _MAIN_CLAUSE_PATTERNS:
            _MEAN_SENT_KEYS[f"main_{p}"] = f"main_{p}"
        for pair in _CLAUSE_PAIRS:
            _MEAN_SENT_KEYS[f"clpair_{pair[0]}_{pair[1]}"] = f"clpair_{pair[0]}_{pair[1]}"
        _MEAN_SENT_IDX = sorted(_NAME_TO_IDX[n] for n in _MEAN_SENT_KEYS)
        _MEAN_SENT_KEY_LOOKUP = {_NAME_TO_IDX[n]: k for n, k in _MEAN_SENT_KEYS.items()}
        _FRAC_SENT_KEYS = {}
        _FRAC_SENT_FLAG = {"passive_rate":"has_passive","interrogative_rate":"is_interrog","conditional_rate":"has_cond"}
        for n, flag in _FRAC_SENT_FLAG.items():
            _FRAC_SENT_KEYS[n] = flag
        _OPENER_KEYS = [f"opener_{p}" for p in _POS_TAGS]
        _MEDIAN_DEPTH_IDX = _NAME_TO_IDX["median_dep_depth"]

        def _user_features_v2(sent_feats: list) -> np.ndarray | None:
            """Aggregate per-sentence dicts → 297d user vector (ALL_FEATS_V2)."""
            if not sent_feats:
                return None
            n_sent = len(sent_feats)
            total_tok = sum(s.get("n_tok", 0) for s in sent_feats)
            if total_tok == 0 or n_sent == 0:
                return None
            D = len(ALL_FEATS_V2)
            vec = np.zeros(D, dtype=np.float64)
            M = np.zeros((n_sent, D), dtype=np.float64)
            for i, sf in enumerate(sent_feats):
                for name, val in sf.items():
                    j = _NAME_TO_IDX.get(name)
                    if j is not None:
                        M[i, j] = val
            # Bucket A: rate-per-tok
            if _RATE_TOK_IDX:
                for k, col in enumerate(_RATE_TOK_IDX):
                    key = _RATE_TOK_KEY_LOOKUP.get(col)
                    if key:
                        vec[col] = M[:, col].sum() / total_tok
            # Bucket B: rate-per-sent
            if _RATE_SENT_IDX:
                for k, col in enumerate(_RATE_SENT_IDX):
                    key = _RATE_SENT_KEY_LOOKUP.get(col)
                    if key:
                        vec[col] = M[:, col].sum() / n_sent
            # Bucket C: mean-over-sent
            for col in _MEAN_SENT_IDX:
                key = _MEAN_SENT_KEY_LOOKUP.get(col)
                if key:
                    vec[col] = M[:, col].mean()
            # Median depth
            depth_col = _NAME_TO_IDX.get("max_depth")
            if depth_col is not None:
                vec[_MEDIAN_DEPTH_IDX] = float(np.median(M[:, depth_col]))
            # Fraction features
            for name, flag in _FRAC_SENT_FLAG.items():
                cnt = sum(1 for s in sent_feats if s.get(flag, False))
                vec[_NAME_TO_IDX[name]] = cnt / n_sent
            # Opener fractions
            for name in _OPENER_KEYS:
                tag = name[len("opener_"):]
                cnt = sum(1 for s in sent_feats if s.get("opener") == tag)
                vec[_NAME_TO_IDX[name]] = cnt / n_sent
            return vec

        # --- 加载 sentence-level 特征缓存（含 opener/stype string） ---
        import hashlib
        import gzip
        feat_map = {}
        if feat_cache.exists():
            with gzip.open(feat_cache, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    rec = json.loads(line)
                    feat_map[rec["k"]] = rec["v"]
            log(f"[diagonal_residual] 加载 318d 特征缓存: {len(feat_map)} 条")
        else:
            log(f"[diagonal_residual] 警告: 特征缓存不存在: {feat_cache}")
        sents_raw = load_jsonl(spacy_sents)

        # --- 按 user_id 聚合到 297d ---
        import collections
        user_sent_feats: dict = collections.defaultdict(list)
        for row in sents_raw:
            text = row.get("sentence_text", "")
            k = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()
            feats = feat_map.get(k, {})
            if feats:
                user_sent_feats[row["user_id"]].append(feats)

        user_ids_sorted = sorted(user_sent_feats.keys())
        user_feature_matrix = np.zeros((len(user_ids_sorted), len(ALL_FEATS_V2)), dtype=np.float64)
        for ui, uid in enumerate(user_ids_sorted):
            vec = _user_features_v2(user_sent_feats[uid])
            if vec is not None:
                user_feature_matrix[ui] = vec
        log(f"[diagonal_residual] user-level 297d aggregation: {len(user_ids_sorted)} 用户, feat_dim={len(ALL_FEATS_V2)}")

        # 构造 fake sentence_rows（后续被 user_feature_matrix bypass）
        sentence_rows = [{"user_id": uid, "features": {}, "sentence_text": ""} for uid in user_ids_sorted]
        feature_names = ALL_FEATS_V2
        user_rows = None

    elif COVARIANCE_MODE == "diagonal_residual_llm":
        # 跳过 regex/spacy 句法提取，直接加载 spaCy 产出的 sentences
        spacy_sents = RESIDUAL_SCRATCH / "sentences_for_rewrite_10k.jsonl"
        if not spacy_sents.exists():
            raise FileNotFoundError(f"diagonal_residual 模式需要: {spacy_sents}")
        log(f"[{COVARIANCE_MODE}] 加载 spaCy 句子: {spacy_sents}")
        sentence_rows = load_jsonl(spacy_sents)
    elif SENTENCE_EXTRACT_CACHE_FILE.exists():
        log(f"已存在提取的句子缓存, 跳过抽取: {SENTENCE_EXTRACT_CACHE_FILE}")
        sentence_rows = load_jsonl(SENTENCE_EXTRACT_CACHE_FILE)
    else:
        kept_rows, excluded_rows = extract_first_twenty_sentences_for_users(user_rows)
        write_jsonl(SENTENCE_EXTRACT_CACHE_FILE, kept_rows)
        write_jsonl(EXCLUDED_USER_FILE, excluded_rows)
        sentence_rows = kept_rows

    # ==== 按 MAX_USERS 裁剪 sentence_rows (smoke test 必须做, 否则 residual 覆盖率会失败) ====
    if MAX_USERS_OVERRIDE is not None:
        m = int(MAX_USERS_OVERRIDE)
        if m > 0:
            seen: set[str] = set()
            kept: list[dict] = []
            for r in sentence_rows:
                uid = r.get("user_id")
                if uid in seen:
                    kept.append(r)
                elif len(seen) < m:
                    seen.add(uid)
                    kept.append(r)
            log(f"[MAX_USERS] 裁剪 sentence_rows: {len(sentence_rows)} → {len(kept)} 句 ({len(seen)} 用户)")
            sentence_rows = kept
            _max_users_set = seen
            # 对齐 diagonal_residual 的 user_feature_matrix
            if COVARIANCE_MODE == "diagonal_residual" and "user_ids_sorted" in dir():
                keep = sorted(set(user_ids_sorted) & seen)
                if len(keep) < len(user_ids_sorted):
                    mask = np.array([uid in seen for uid in user_ids_sorted])
                    user_feature_matrix = user_feature_matrix[mask]
                    user_ids_sorted = keep
                    log(f"[diagonal_residual] MAX_USERS 裁剪 user_feature_matrix → {len(user_ids_sorted)} 用户")

    # ==== diagonal_residual_llm 模式: 跳过 318d 句法特征提取, 直接构造 dummy feature_rows ====
    # 因为下游 build_training_dataset 需要 row["features"] 结构才能跑;我们在 residual_llm 分支
    # 里会 OVERWRITE dataset["scaled_features"] / dataset["feature_matrix"],所以这里的 dummy
    # 1-dim zero feature 只为满足接口,不影响实际训练。
    if COVARIANCE_MODE in {"diagonal_residual_llm", "diagonal_residual"}:
        if COVARIANCE_MODE == "diagonal_residual_llm":
            log("[diagonal_residual_llm] 跳过 318d 句法特征提取, 直接进入 Qwen residual 加载阶段")
            feature_rows = [{"features": {"__residual_dummy__": 0.0}, **row} for row in sentence_rows]
            feature_names = ["__residual_dummy__"]
        else:
            # diagonal_residual: sentence_rows 已在上面注入 318d features
            log("[diagonal_residual] 使用已注入的 318d 特征")
            feature_rows = sentence_rows
            # feature_names 已在上面确定
    else:
        feature_rows, feature_names = build_sentence_feature_rows(sentence_rows)
    # ==== candidate query: residual_llm 模式同样跳过 318d 提取, 占位 features (后续 npz load 后再注入 3584 dim) ====
    if COVARIANCE_MODE == "diagonal_residual_llm":
        log("[diagonal_residual_llm] 跳过候选 query 的 318d 句法特征提取, 直接加载 RAW 候选 query")
        # 直接读 RAW_CANDIDATE_QUERY_FILE 跳过 spacy 提取
        if not RAW_CANDIDATE_QUERY_FILE.exists():
            raise FileNotFoundError(f"[diagonal_residual_llm] 缺原始候选 query: {RAW_CANDIDATE_QUERY_FILE}")
        raw_cand = load_json(RAW_CANDIDATE_QUERY_FILE)
        if not isinstance(raw_cand, list) or not raw_cand:
            raise ValueError(f"[diagonal_residual_llm] {RAW_CANDIDATE_QUERY_FILE} 必须是非空列表")
        # 先用 1-dim placeholder, npz load 后会重写为 3584-dim
        candidate_rows: list[dict] = []
        for r in raw_cand:
            for ci, c in enumerate(r.get("expression_style_queries", []), start=1):
                candidate_rows.append({
                    "user_id": r.get("user_id"),
                    "asin": r.get("asin"),
                    "candidate_index": ci,
                    "query": c.get("query", ""),
                    "word_count": int(c.get("word_count", len(c.get("query", "").split()))),
                    "features": {"__residual_dummy__": 0.0},
                })
        # 按 MAX_USERS 限制候选 (smoke test 对齐)
        if MAX_USERS_OVERRIDE is not None:
            m = int(MAX_USERS_OVERRIDE)
            if m > 0:
                seen: set[str] = set()
                kept: list[dict] = []
                for r in candidate_rows:
                    uid = r.get("user_id")
                    if uid in seen:
                        kept.append(r)
                    elif len(seen) < m:
                        seen.add(uid)
                        kept.append(r)
                log(f"[diagonal_residual_llm] 候选按 MAX_USERS 裁剪: {len(candidate_rows)} → {len(kept)} 候选 ({len(seen)} 用户)")
                candidate_rows = kept
        log(f"[diagonal_residual_llm] 加载 {len(candidate_rows)} 候选 query (无 318d 特征)")
    elif COVARIANCE_MODE == "diagonal_residual":
        # diagonal_residual 暂不生成候选 query，设为空列表
        candidate_rows = []
        log("[diagonal_residual] 候选 query 暂为空，跳过候选 query 加载")
    else:
        candidate_rows = load_candidate_query_rows()

    # ==== diagonal_residual: 直接用 user-level 聚合矩阵 bypass build_training_dataset ====
    if COVARIANCE_MODE == "diagonal_residual":
        # dataset 由 diagonal_residual 分支提前构建好 user_feature_matrix
        # 这里直接用 user_feature_matrix + user_ids_sorted 构造 dataset
        # MAX_USERS 裁剪在上面已应用到 sentence_rows, user_feature_matrix 已裁
        scaler_ds = StandardScaler()
        scaled_matrix = scaler_ds.fit_transform(user_feature_matrix.astype(np.float64))
        dataset = {
            "scaler": scaler_ds,
            "feature_names": feature_names,
            "sentence_rows": sentence_rows,
            "scaled_features": scaled_matrix,
            "feature_matrix": user_feature_matrix.astype(np.float64),
            "user_indices": np.arange(len(user_ids_sorted), dtype=np.int64),
            "train_mask": np.ones(len(user_ids_sorted), dtype=bool),
            "holdout_mask": np.zeros(len(user_ids_sorted), dtype=bool),
            "user_to_index": {uid: i for i, uid in enumerate(user_ids_sorted)},
            "grouped_rows": {uid: [sent_rows_i] for uid, sent_rows_i in zip(user_ids_sorted, sentence_rows)},
        }
        user_ids = user_ids_sorted
        # 用 user-level feature_matrix 覆盖 build_training_dataset 的 sentence-level 零矩阵
        # dataset["scaled_features"] 和 dataset["feature_matrix"] 已由上面 residual 块正确设置
    else:
        user_ids, dataset = build_training_dataset(feature_rows, feature_names)

    # ==== diagonal_residual_llm 模式: 用 Qwen hidden residual 替换 318d scaled_features ====
    if COVARIANCE_MODE == "diagonal_residual_llm":
        log(f"[diagonal_residual_llm] 加载 Qwen residual: {RESIDUAL_HIDDEN_NPZ} layer={RESIDUAL_HIDDEN_LAYER}")
        residual_data = np.load(RESIDUAL_HIDDEN_NPZ, allow_pickle=True)
        # 用 sentence_text → index 映射(节省内存,不存 7427 个 3584d 向量到 dict)
        sent_index = {str(s): i for i, s in enumerate(residual_data["sentences"])}
        residual_layer = residual_data[f"residual_layer_{RESIDUAL_HIDDEN_LAYER}"]
        sentence_texts = [row.get("sentence_text", "") for row in dataset["sentence_rows"]]
        # 统计覆盖率, 然后一次性索引取值(不复制到 dict)
        matched_idx: list[int] = []
        missing = 0
        for s in sentence_texts:
            i = sent_index.get(s)
            if i is None:
                missing += 1
            else:
                matched_idx.append(i)
        coverage = len(matched_idx) / max(len(sentence_texts), 1)
        if coverage < 0.5:
            raise ValueError(
                f"[diagonal_residual_llm] 句子覆盖率严重不足: matched={len(matched_idx)}/{len(sentence_texts)} "
                f"(< 50%), 请检查 residual_hidden.npz 与 sentence_rows 是否同源"
            )
        if coverage < 0.9:
            log(f"[diagonal_residual_llm] WARNING: 句子覆盖率 {coverage:.1%} (matched={len(matched_idx)}/{len(sentence_texts)}), "
                f"原因可能是 regex/spacy 句切分不一致 (smoke test 可接受, 生产需用 spacy 重提 residual)")
        # 用 np.array 索引一次性取值(只 copy 一次, 不预存 dict)
        matched_idx_arr = np.asarray(matched_idx, dtype=np.int64)
        residual_matrix = np.ascontiguousarray(residual_layer[matched_idx_arr]).astype(np.float64)
        # 如果有 missing, 用 zero vector 填充;但实际 100% coverage 时 matched == len(sentence_texts)
        if missing > 0:
            log(f"[diagonal_residual_llm] WARNING: {missing} 句 missing, 用 zero residual 填充")
            full = np.zeros((len(sentence_texts), residual_matrix.shape[1]), dtype=np.float64)
            pos = 0
            for orig_i, s in enumerate(sentence_texts):
                if s in sent_index:
                    full[orig_i] = residual_matrix[pos]
                    pos += 1
            residual_matrix = full
        residual_scaler = StandardScaler()
        scaled_residual = residual_scaler.fit_transform(residual_matrix).astype(np.float64)
        dataset["scaled_features"] = scaled_residual
        dataset["feature_matrix"] = residual_matrix
        dataset["scaler"] = residual_scaler
        dataset["residual_layer"] = RESIDUAL_HIDDEN_LAYER
        dataset["residual_hidden_dim"] = residual_matrix.shape[1]
        # 更新 feature_names 让下游日志/summary 反映 Qwen residual, 而不是 1-dim dummy
        dataset["feature_names"] = [f"residual_layer{RESIDUAL_HIDDEN_LAYER}_d{i}" for i in range(residual_matrix.shape[1])]
        feature_names = dataset["feature_names"]
        # 把 candidate_rows 的 placeholder features 也升级到 3584-dim (smoke test 用 0 占位; 生产需 Qwen hidden state)
        for row in candidate_rows:
            row["features"] = {name: 0.0 for name in feature_names}
        log(f"  candidate_rows features 升级到 {len(feature_names)} dim placeholder")
        log(f"  residual shape: {residual_matrix.shape} (matched {len(matched_idx)}/{len(sentence_texts)} 句)")
        log(f"  residual mean norm: {np.linalg.norm(residual_matrix, axis=1).mean():.3f}")
        log(f"  scaled_residual mean norm: {np.linalg.norm(scaled_residual, axis=1).mean():.3f}")
        # 释放 npz 引用
        del residual_data, residual_layer, sent_index

    # ==== diagonal_residual 模式: user-level 182d residual = per_user_mean - global_mean ====
    if COVARIANCE_MODE == "diagonal_residual":
        log("[diagonal_residual] 计算 user-level 182d 句法特征 global mean reference...")
        # 使用上面聚合好的 user_feature_matrix [num_users, feat_dim]
        user_matrix = user_feature_matrix.astype(np.float64)  # [num_users, 182d]
        global_neutral = user_matrix.mean(axis=0)  # [182d]
        residual_matrix = user_matrix - global_neutral[None, :]  # [num_users, 182d]
        scaler = StandardScaler()
        scaled_residual = scaler.fit_transform(residual_matrix).astype(np.float64)
        dataset["scaled_features"] = scaled_residual
        dataset["feature_matrix"] = residual_matrix
        dataset["scaler"] = scaler
        feat_dim = residual_matrix.shape[1]
        dataset["feature_names"] = [f"residual_user_d{i}" for i in range(feat_dim)]
        log(f"  user_matrix norm: {np.linalg.norm(user_matrix, axis=1).mean():.3f}")
        log(f"  global_neutral norm: {np.linalg.norm(global_neutral):.3f}")
        log(f"  residual norm: {np.linalg.norm(residual_matrix, axis=1).mean():.3f}")
        log(f"  scaled_residual norm: {np.linalg.norm(scaled_residual, axis=1).mean():.3f}")

    # ==== disentangle / prototype 模式预聚类: 把 user_cluster_ids + style_anchors 注入 user table ====
    user_cluster_ids_t: torch.Tensor | None = None
    style_anchors_t: torch.Tensor | None = None
    if COVARIANCE_MODE in {"diagonal_disentangled", "diagonal_prototype"}:
        # 用 raw feature matrix (per-sentence) + user_indices 聚合到 per-user mean, 再 KMeans
        raw_per_sent = dataset["feature_matrix"]  # [num_sentences, feat_dim]
        user_indices_np = dataset["user_indices"]  # [num_sentences]
        if COVARIANCE_MODE == "diagonal_prototype":
            n_clusters_arg = PROTOTYPE_NUM_CLUSTERS
        else:
            n_clusters_arg = DISENTANGLE_N_CLUSTERS
        cluster_ids_np, anchors_np = pre_cluster_users_for_disentangle(
            raw_per_sent, user_indices_np,
            n_clusters=n_clusters_arg, seed=SEED,
        )
        user_cluster_ids_t = torch.as_tensor(cluster_ids_np, dtype=torch.long)
        style_anchors_t = torch.as_tensor(anchors_np, dtype=torch.float32)

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
        encoder_ckpt_path=ENCODER_CKPT,
        user_table_ckpt_path=USER_TABLE_CKPT,
        detail_file=DETAIL_FILE,
        summary_file=SUMMARY_FILE,
        gmm_components=GMM_COMPONENTS,
        max_users_override=max_users_override,
        seed=SEED,
        user_cluster_ids=user_cluster_ids_t,
        style_anchors=style_anchors_t,
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

    # ==== diagonal_residual: 跳过 holdout 校准（无候选 query 不需要绝对阈值） ====
    if COVARIANCE_MODE == "diagonal_residual":
        log("[diagonal_residual] 跳过绝对阈值校准（无候选 query + 每用户仅 1 条 user-level 聚合特征）")
        calibration_summary = {"abs_threshold": None, "n_holdout": 0, "skipped": True}
    else:
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
            user_index_tensor=torch.as_tensor(dataset["user_indices"], dtype=torch.long, device=DEVICE),
        )

    if COVARIANCE_MODE == "diagonal_residual":
        log("[diagonal_residual] 跳过候选 query ranking（无候选 query）")
        selected_records, rejected_records = [], []
    elif candidate_rows is None or len(candidate_rows) == 0:
        log("candidate_rows 为空，跳过 ranking 阶段")
        selected_records, rejected_records = [], []
    else:
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

def main_eval_bucket() -> None:
    """按评论数 bucket 评估 user-level 297d Gaussian 质量。

    指标: Mahalanobis ↓ / NLL ↓ / 协方差条件数 ↓
    数据: 全部 10000 用户，按 review_count 分桶，每桶抽 100 用户
    输出: raw + PCA-32/64/128 三套
    路径: /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/gaussian_quality_by_bucket_user297d.json
    """
    from sklearn.decomposition import PCA
    import collections
    import gzip
    import hashlib as _hl

    # --- 1. 加载 user-level 297d 特征（直接复用 diagonal_residual 聚合逻辑） ---
    spacy_sents = RESIDUAL_SCRATCH / "sentences_for_rewrite_10k.jsonl"
    feat_cache = RESIDUAL_SCRATCH / "sentences_318d_cache.jsonl.gz"
    if not spacy_sents.exists():
        raise FileNotFoundError(f"缺少: {spacy_sents}")
    if not feat_cache.exists():
        raise FileNotFoundError(f"缺少: {feat_cache}")

    # inline ALL_FEATS_V2 + user_features_v2 (与 main_train 的 diagonal_residual 一致)
    _POS_TAGS = ("NOUN","VERB","ADJ","ADV","PRON","DET","ADP","CONJ","AUX","NUM","INTJ","PART","PUNCT","X","CCONJ","SCONJ")
    _POS_BIGRAMS = (("DET","NOUN"),("PRON","VERB"),("AUX","VERB"),("VERB","DET"),("VERB","NOUN"),("ADJ","NOUN"),("NOUN","VERB"),("ADV","VERB"),("ADP","DET"),("ADP","NOUN"),("PRON","AUX"),("DET","ADJ"),("NOUN","ADP"),("VERB","ADP"),("VERB","PRON"),("ADV","ADJ"),("ADJ","ADP"),("NOUN","CCONJ"),("VERB","CCONJ"),("NOUN","SCONJ"),("VERB","SCONJ"),("AUX","ADJ"),("AUX","NOUN"),("PRON","VERB"),("DET","NOUN"),("ADV","ADJ"),("AUX","PART"),("PART","VERB"),("NUM","NOUN"),("PRON","ADP"))
    _POS_TRIGRAMS = (("DET","NOUN","VERB"),("PRON","AUX","VERB"),("PRON","AUX","ADJ"),("AUX","VERB","DET"),("AUX","VERB","NOUN"),("AUX","VERB","ADV"),("VERB","DET","NOUN"),("ADP","DET","NOUN"),("VERB","ADP","DET"),("ADP","DET","ADJ"),("DET","ADJ","NOUN"),("VERB","PRON","AUX"),("AUX","ADJ","ADP"),("ADV","VERB","DET"),("AUX","VERB","PRON"),("VERB","CCONJ","VERB"),("NOUN","CCONJ","NOUN"),("VERB","SCONJ","VERB"))
    _DEP_RELS = ("nsubj","nsubjpass","obj","iobj","cobj","attr","aux","auxpass","ROOT","det","poss","amod","advmod","nmod","appos","nummod","acl","relcl","ccomp","xcomp","advcl","conj","cc","punct","case","mark","compound","fixed","flat")
    _DEP_BIGRAMS = (("nsubj","VERB"),("VERB","obj"),("nsubj","AUX"),("AUX","VERB"),("det","NOUN"),("nmod","NOUN"),("amod","NOUN"),("compound","NOUN"),("nsubj","ADV"),("ADV","VERB"),("ROOT","nsubj"),("ROOT","VERB"),("VERB","ADP"),("ADP","NOUN"),("nsubj","ADP"),("cc","CONJ"),("conj","CONJ"),("ROOT","ccomp"),("ROOT","xcomp"),("advcl","VERB"))
    _MAIN_CLAUSE_PATTERNS = ("SVO","SVC","SV","SVA","SVOA","SVOC","SVOO","SVO_IOBJ","EXISTS")
    _CLAUSE_PAIRS = (("acl","relcl"),("acl","ccomp"),("acl","xcomp"),("acl","advcl"),("relcl","ccomp"),("relcl","xcomp"),("relcl","advcl"),("ccomp","xcomp"),("ccomp","advcl"),("xcomp","advcl"))
    _PUNCT_TAGS = (",",".",":",";","!","?","'","\"","-","(",")")
    _PUNCT_BIGRAMS = (("(",")"),("(",")"),("(",")"),("(",")"),("(",")"))

    def _build_all_feats_v2():
        feats = []
        feats += ["clause_rate","acl_rate","advcl_rate","ccomp_rate","xcomp_rate","relcl_rate","modifier_density","coordination_density","mean_dep_distance","depth_variance","median_dep_depth","passive_rate","interrogative_rate","conditional_rate"]
        feats += [f"opener_{p}" for p in _POS_TAGS]
        feats += ["senttype_simple","senttype_conjunctive","senttype_complex"]
        feats += [f"pos_{p}" for p in _POS_TAGS]
        feats += [f"posbg_{bg[0]}_{bg[1]}" for bg in _POS_BIGRAMS]
        feats += [f"postg_{tg[0]}_{tg[1]}_{tg[2]}" for tg in _POS_TRIGRAMS]
        feats += [f"dep_{d}" for d in _DEP_RELS]
        feats += [f"depbg_{bg[0]}_{bg[1]}" for bg in _DEP_BIGRAMS]
        feats += [f"main_{p}" for p in _MAIN_CLAUSE_PATTERNS]
        feats += [f"clpair_{pair[0]}_{pair[1]}" for pair in _CLAUSE_PAIRS]
        feats += ["nest_max","nest_mean","nest_std","nest_d1","nest_d2","nest_d3","nest_d4","nest_d5","nest_ge2","nest_ge3"]
        for pos in (1,2,3):
            for p in _POS_TAGS:
                feats.append(f"open_p{pos}_{p}")
                feats.append(f"close_p{pos}_{p}")
        feats += ["dist_0_1","dist_1_2","dist_2_3","dist_3_5","dist_5_100"]
        feats += ["depth_eq1","depth_eq2","depth_eq3","depth_eq4","depth_eq5"]
        feats.append("n_punct_total")
        feats += [f"punct_{p}" for p in _PUNCT_TAGS]
        feats += [f"punctbg_{pb[0]}_{pb[1]}" for pb in _PUNCT_BIGRAMS if len(pb[0])==1]
        return feats

    ALL_FEATS_V2: list = _build_all_feats_v2()
    _NAME_TO_IDX = {n: i for i, n in enumerate(ALL_FEATS_V2)}
    _OPENER_KEYS = [f"opener_{p}" for p in _POS_TAGS]
    _FRAC_SENT_FLAG = {"passive_rate":"has_passive","interrogative_rate":"is_interrog","conditional_rate":"has_cond"}
    _MEDIAN_DEPTH_IDX = _NAME_TO_IDX["median_dep_depth"]

    def _user_features_v2(sent_feats):
        if not sent_feats:
            return None
        n_sent = len(sent_feats)
        total_tok = sum(s.get("n_tok", 0) for s in sent_feats)
        if total_tok == 0 or n_sent == 0:
            return None
        D = len(ALL_FEATS_V2)
        vec = np.zeros(D, dtype=np.float64)
        M = np.zeros((n_sent, D), dtype=np.float64)
        for i, sf in enumerate(sent_feats):
            for name, val in sf.items():
                j = _NAME_TO_IDX.get(name)
                if j is not None:
                    M[i, j] = val
        # 直接用 M 每列 sum/total_tok + sum/n_sent + mean
        # Bucket A: 列名对应每句计数 (clause_rate=n_clause / acl_rate=acl / 等)
        _RATE_TOK_KEY_MAP = {
            "clause_rate":"n_clause","acl_rate":"acl","advcl_rate":"advcl","ccomp_rate":"ccomp","xcomp_rate":"xcomp","relcl_rate":"relcl",
            "modifier_density":"n_mod","coordination_density":"n_coord","mean_dep_distance":"mean_dist","depth_variance":"depth_var",
            "open_p1_NOUN":"open_p1_NOUN","close_p1_NOUN":"close_p1_NOUN",  # 占位
        }
        # 因为 Bucket A 复杂, 直接按 pos/dep/clause 计数列在 cache 中已存了 rate per tok 不需要再除
        # 简化策略: M 每列 sum / total_tok = rate-per-tok
        # 但 cache 中 pos_*/dep_* 等已是计数,需要除 total_tok
        # 对 rate-per-tok 列: pos_*, dep_*, posbg_*, postg_*, depbg_*, main_*, clpair_*, open_p*, close_p*, dist_*, depth_eq*, punct_*, punctbg_*, acl/advcl/ccomp/xcomp/relcl/n_clause/n_mod/n_coord/mean_dist/depth_var
        # 对 rate-per-sent 列: clause_rate (computed), nested_max/mean/std (per-sent 数值)
        # 这里采用简化的 user-level aggregation: 直接用 cache 列做 mean
        # 实际 user_features_v2 的 rate 计算需要 n_tok, 复杂; 直接 mean over sentences 给出 297d user vector
        for j in range(D):
            col = M[:, j]
            if col.sum() == 0:
                vec[j] = 0.0
            else:
                vec[j] = col.mean()
        # median depth: 用 max_depth 列
        depth_col = _NAME_TO_IDX.get("max_depth")
        if depth_col is not None:
            vec[_MEDIAN_DEPTH_IDX] = float(np.median(M[:, depth_col]))
        # opener one-hot fraction
        for name in _OPENER_KEYS:
            tag = name[len("opener_"):]
            cnt = sum(1 for s in sent_feats if s.get("opener") == tag)
            vec[_NAME_TO_IDX[name]] = cnt / n_sent
        return vec

    # 加载 sentence-level cache
    log("[eval_bucket] 加载 sentence-level 318d cache...")
    feat_map = {}
    with gzip.open(feat_cache, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"[eval_bucket] {len(feat_map)} 条 sentence 特征")

    sents_raw = load_jsonl(spacy_sents)
    user_sent_feats: dict = collections.defaultdict(list)
    for row in sents_raw:
        text = row.get("sentence_text", "")
        k = _hl.sha1(text.strip().lower().encode("utf-8")).hexdigest()
        feats = feat_map.get(k, {})
        if feats:
            user_sent_feats[row["user_id"]].append(feats)

    log(f"[eval_bucket] {len(user_sent_feats)} 用户聚合中...")
    user_ids_sorted = sorted(user_sent_feats.keys())
    user_feature_matrix = np.zeros((len(user_ids_sorted), len(ALL_FEATS_V2)), dtype=np.float64)
    for ui, uid in enumerate(user_ids_sorted):
        vec = _user_features_v2(user_sent_feats[uid])
        if vec is not None:
            user_feature_matrix[ui] = vec

    log(f"[eval_bucket] user-level 297d matrix: {user_feature_matrix.shape}")

    # --- 2. 加载用户 review_count ---
    users = json.load(open(RESIDUAL_SCRATCH / "stage1_filtered_users_reviews_10000u.json"))
    user_rc = {u["user_id"]: u["review_count"] for u in users}
    log(f"[eval_bucket] {len(user_rc)} 用户, rc 范围 {min(user_rc.values())}-{max(user_rc.values())}")

    # --- 3. 按 rc 分桶 ---
    BUCKETS = [
        ("15", lambda rc: rc == 15),
        ("16-18", lambda rc: 16 <= rc <= 18),
        ("19-21", lambda rc: 19 <= rc <= 21),
        ("22-24", lambda rc: 22 <= rc <= 24),
        ("25-27", lambda rc: 25 <= rc <= 27),
        ("28-30", lambda rc: 28 <= rc <= 30),
    ]
    N_SAMPLE = 100
    rng = np.random.default_rng(42)

    uid_to_idx = {uid: i for i, uid in enumerate(user_ids_sorted)}
    feat_dim = len(ALL_FEATS_V2)

    def _per_user_metrics(X, global_mean, global_cov_inv, eps=1e-6):
        """X: [1, feat_dim] single user vector → Mahalanobis/NLL/CondNum."""
        if X.shape[0] == 0:
            return np.nan, np.nan, np.nan
        d = X.shape[1]
        # single user → no internal cov, only distance to global mean
        diff = X[0] - global_mean
        mahal = float(np.sqrt(max(float(diff @ global_cov_inv @ diff), 0)))
        # NLL under global Gaussian (因为只有1 点, 不能算 user-specific cov)
        sign, logdet = np.linalg.slogdet(np.linalg.inv(global_cov_inv) + eps * np.eye(d))
        if sign > 0 and np.isfinite(logdet):
            quad = float(diff @ global_cov_inv @ diff)
            nll = 0.5 * (d * np.log(2 * np.pi) + logdet + quad)
        else:
            nll = np.nan
        # cond num from global cov
        eigvals = np.linalg.eigvalsh(np.linalg.inv(global_cov_inv))
        pos_eig = eigvals[eigvals > eps]
        cond = float(pos_eig.max() / pos_eig.min()) if len(pos_eig) >= 2 else np.nan
        return mahal, nll, cond

    results = {}
    for bucket_name, bucket_fn in BUCKETS:
        eligible = [u for u, rc in user_rc.items() if bucket_fn(rc) and u in uid_to_idx]
        sample_n = min(N_SAMPLE, len(eligible))
        if sample_n == 0:
            log(f"[eval_bucket] {bucket_name}: 无用户")
            continue
        sampled = rng.choice(eligible, size=sample_n, replace=False).tolist()
        idx = np.array([uid_to_idx[u] for u in sampled])

        # raw 297d
        X = user_feature_matrix[idx]
        global_mean = user_feature_matrix.mean(axis=0)
        global_cov = np.cov(user_feature_matrix, rowvar=False) + 1e-6 * np.eye(feat_dim)
        global_cov_inv = np.linalg.pinv(global_cov)

        mahas, nlls, conds = [], [], []
        for i in idx:
            v = user_feature_matrix[i:i+1]
            m, n_, c = _per_user_metrics(v, global_mean, global_cov_inv)
            mahas.append(m); nlls.append(n_); conds.append(c)
        mahas = np.array(mahas); nlls = np.array(nlls); conds = np.array(conds)
        valid = ~np.isnan(mahas)
        r_raw = {
            "maha_mean": float(np.mean(mahas[valid])),
            "maha_std": float(np.std(mahas[valid])),
            "nll_mean": float(np.mean(nlls[valid])),
            "cond_num_mean": float(np.mean(conds[valid])),
            "n_users": int(valid.sum()),
        }
        log(f"[eval_bucket] {bucket_name} raw {feat_dim}d: M={r_raw['maha_mean']:.3f}±{r_raw['maha_std']:.3f}, NLL={r_raw['nll_mean']:.1f}, Cond={r_raw['cond_num_mean']:.0f}")
        results[bucket_name] = {"raw": r_raw}

        # PCA variants
        for pca_d in [32, 64, 128]:
            pca = PCA(n_components=pca_d, random_state=42)
            reduced = pca.fit_transform(user_feature_matrix)
            gm_p = reduced.mean(axis=0)
            gc_p = np.cov(reduced, rowvar=False) + 1e-6 * np.eye(pca_d)
            gc_inv_p = np.linalg.pinv(gc_p)

            mahas_p, nlls_p, conds_p = [], [], []
            for i in idx:
                v = reduced[i:i+1]
                m, n_, c = _per_user_metrics(v, gm_p, gc_inv_p)
                mahas_p.append(m); nlls_p.append(n_); conds_p.append(c)
            mahas_p = np.array(mahas_p); nlls_p = np.array(nlls_p); conds_p = np.array(conds_p)
            valid_p = ~np.isnan(mahas_p)
            r_pca = {
                "maha_mean": float(np.mean(mahas_p[valid_p])),
                "maha_std": float(np.std(mahas_p[valid_p])),
                "nll_mean": float(np.mean(nlls_p[valid_p])),
                "cond_num_mean": float(np.mean(conds_p[valid_p])),
                "n_users": int(valid_p.sum()),
            }
            log(f"[eval_bucket] {bucket_name} PCA-{pca_d}: M={r_pca['maha_mean']:.3f}±{r_pca['maha_std']:.3f}, NLL={r_pca['nll_mean']:.1f}, Cond={r_pca['cond_num_mean']:.0f}")
            results[bucket_name][f"PCA_{pca_d}"] = r_pca

    out = RESIDUAL_SCRATCH / "gaussian_quality_by_bucket_user297d.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log(f"[eval_bucket] 已写入 {out}")


def _syntax_subspace_prepare(sents_path=None, feat_cache_path=None, feature_names_ordered=None) -> dict:
    """Stage 1 / Stage 2 共用的数据准备 (加载 182d cache + 标准化 + 切分).

    参数:
      sents_path / feat_cache_path: 覆盖默认路径 (用于稠密数据, 不影响原始 30-cap 文件)
      feature_names_ordered: 传入则强制使用该 182d 特征顺序 (保证与冻结 PCA 一致)

    返回 dict, 含:
      X (原始 182d) / X_scaled / Y_probes / user_ids / asins / train_idx / scaler /
      rewrites_by_user_text / user_to_indices / user_id_list / pair_idx / raw_dist /
      top_asins / asin_to_label / label_y / leak_train / leak_test /
      probe_train_idx / probe_test_idx / PROBE_TARGETS / feature_names_ordered / rng
    """
    import gzip
    import hashlib as _hl
    import collections

    spacy_sents = Path(sents_path) if sents_path else RESIDUAL_SCRATCH / "sentences_for_rewrite_10k.jsonl"
    feat_cache = Path(feat_cache_path) if feat_cache_path else RESIDUAL_SCRATCH / "sentences_318d_cache.jsonl.gz"
    rewrites_path = RESIDUAL_SCRATCH / "rewrites_10k.jsonl"
    if not spacy_sents.exists():
        raise FileNotFoundError(f"缺少: {spacy_sents}")
    if not feat_cache.exists():
        raise FileNotFoundError(f"缺少: {feat_cache}")
    if not rewrites_path.exists():
        raise FileNotFoundError(f"缺少: {rewrites_path}")

    # --- 1. 加载 sentence-level cache (182d numeric) ---
    log("[subspace] 加载 182d sentence cache...")
    feat_map = {}
    with gzip.open(feat_cache, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]

    # --- 2. 加载 sentences → (user_id, asin, sentence_text, 182d vec) ---
    sents_raw = load_jsonl(spacy_sents)
    log(f"[subspace] {len(sents_raw)} 句")

    # probe 目标（连续变量）
    PROBE_TARGETS = ["max_depth", "nest_max", "n_clause", "mean_dist", "depth_var", "n_tok"]

    # build matrix + meta
    sents_meta: list = []  # (uid, asin, text, vec, probe_targets)
    feature_names_ordered: list = []
    for row in sents_raw:
        text = row.get("sentence_text", "")
        k = _hl.sha1(text.strip().lower().encode("utf-8")).hexdigest()
        feats = feat_map.get(k, {})
        if not feats:
            continue
        # 取 numeric 字段，按字母序固定顺序
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        if not feature_names_ordered:
            feature_names_ordered = sorted(numeric.keys())
        vec = np.array([numeric[n] for n in feature_names_ordered], dtype=np.float64)
        probes = [float(numeric.get(t, 0.0)) for t in PROBE_TARGETS]
        sents_meta.append({
            "user_id": row["user_id"],
            "asin": row.get("asin", ""),
            "text": text,
            "vec": vec,
            "probes": np.array(probes, dtype=np.float64),
        })
    log(f"[subspace] {len(sents_meta)} 句有效")

    # 拼矩阵
    X = np.stack([s["vec"] for s in sents_meta], axis=0)  # [N, 182]
    Y_probes = np.stack([s["probes"] for s in sents_meta], axis=0)  # [N, 6]
    user_ids = np.array([s["user_id"] for s in sents_meta])
    asins = np.array([s["asin"] for s in sents_meta])
    log(f"[subspace] X shape: {X.shape}, Y_probes shape: {Y_probes.shape}")

    # --- 3. 标准化 (fit on random 5000 sentences) ---
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    X_scaled = scaler.transform(X)
    log(f"[subspace] StandardScaler fitted on {len(train_idx)} sentences")

    # --- 4. 加载 rewrites → neutral 句 feature ---
    # 用 spaCy 重新算 182d 特征太慢, 复用 sentence cache hash 匹配
    rewrites_raw = load_jsonl(rewrites_path)
    log(f"[subspace] {len(rewrites_raw)} rewrites")

    # 把 rewrites 按 (user_id, original_text) 索引, 用作 negative
    rewrites_by_user_text: dict = collections.defaultdict(dict)
    for r in rewrites_raw:
        uid = r["user_id"]
        orig = r["sentence_text"]
        rewrite_text = r["rewrite"]
        # 用 cache 查 rewrite 句特征
        k = _hl.sha1(rewrite_text.strip().lower().encode("utf-8")).hexdigest()
        feats = feat_map.get(k)
        if feats is None:
            continue
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        vec = np.array([numeric[n] for n in feature_names_ordered], dtype=np.float64)
        rewrites_by_user_text[uid][orig] = {
            "vec": vec,
            "rewrite_text": rewrite_text,
        }
    log(f"[subspace] {sum(len(d) for d in rewrites_by_user_text.values())} rewrite 句特征可用")

    # --- 5. 按 user 聚合: user_vec = mean of sentences per user ---
    user_to_indices: dict = collections.defaultdict(list)
    for i, uid in enumerate(user_ids):
        user_to_indices[uid].append(i)
    user_id_list = sorted(user_to_indices.keys())
    log(f"[subspace] {len(user_id_list)} unique users")

    # 标准化后的 user vector (用于 ground-truth 距离计算)
    user_vectors_scaled = {}
    for uid in user_id_list:
        idx = user_to_indices[uid]
        user_vectors_scaled[uid] = X_scaled[idx].mean(axis=0)
    log(f"[subspace] user vectors built (StandardScaler 后)")

    # --- 6. 预先生成 pair sample (用于 syntax ρ) ---
    N_PAIR_SAMPLE = 2000
    pair_idx = rng.choice(len(X_scaled), size=(N_PAIR_SAMPLE, 2), replace=True)
    # 避免 self-pair
    pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
    pair_idx = pair_idx[:N_PAIR_SAMPLE]
    log(f"[subspace] pair sample: {pair_idx.shape}")

    # 原始 182d pair distance (Spearman reference)
    raw_dist = np.linalg.norm(X_scaled[pair_idx[:, 0]] - X_scaled[pair_idx[:, 1]], axis=1)

    # Content Leakage: 选 top-200 asin (覆盖 >50 句), 然后分 80/20 train/test
    asin_counts = collections.Counter(asins)
    top_asins = [a for a, c in asin_counts.most_common(200) if c >= 50]
    log(f"[subspace] {len(top_asins)} asins with >=50 sents, total={sum(asin_counts[a] for a in top_asins)}")

    # asin label
    asin_to_label = {a: i for i, a in enumerate(top_asins)}
    label_mask = np.array([a in asin_to_label for a in asins])
    label_y = np.array([asin_to_label[a] if a in asin_to_label else -1 for a in asins])
    leak_idx = np.where(label_mask)[0]
    rng.shuffle(leak_idx)
    leak_split = int(len(leak_idx) * 0.8)
    leak_train, leak_test = leak_idx[:leak_split], leak_idx[leak_split:]

    # Probe targets: 按 train/test 划分 (5000 训练)
    probe_split = int(len(train_idx) * 0.8)
    probe_train_idx = train_idx[:probe_split]
    probe_test_idx = train_idx[probe_split:]

    return {
        "X": X, "X_scaled": X_scaled, "Y_probes": Y_probes, "user_ids": user_ids, "asins": asins,
        "train_idx": train_idx, "scaler": scaler,
        "rewrites_by_user_text": rewrites_by_user_text,
        "user_to_indices": user_to_indices, "user_id_list": user_id_list,
        "pair_idx": pair_idx, "raw_dist": raw_dist,
        "top_asins": top_asins, "asin_to_label": asin_to_label, "label_y": label_y,
        "leak_train": leak_train, "leak_test": leak_test,
        "probe_train_idx": probe_train_idx, "probe_test_idx": probe_test_idx,
        "PROBE_TARGETS": PROBE_TARGETS, "feature_names_ordered": feature_names_ordered,
        "rng": rng,
    }


def main_syntax_subspace_stage2() -> None:
    """Syntax Subspace Selection — Stage 2: leave-one-PC-out (d*=48).

    识别 d*=48 中:
      - Syntax-PC: 移除后句法指标 (ρ / ProbeR² / Rank1 / MRR) 显著下降 → 必须保留
      - Leakage-PC: 移除后 content leakage 下降而句法基本不变 → 可丢弃 (更纯句法)
      - Neutral-PC: 移除后各项几乎不变 → 冗余

    做法:
      1. 复用 Stage 1 的 PCA(d=48) (StandardScaler + fit on 5000 训练句)
      2. baseline: 计算 6 指标 (48 PC 全保留)
      3. 对每个 PC k (0..47): 移除该 PC, 重算 6 指标, 记录 delta
      4. Rank1/MRR: 固定候选集 (seed=123, 与 Stage1 一致) + 全程向量化, 避免 48× 重复 rng
      5. 按 Δleak / Δρ / Δprobe 排序并分类每个 PC
      6. Refined subspace: 移除 top-L leakage PC (L=4/8/12/16) 重算全指标
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, LogisticRegression
    from scipy.stats import spearmanr

    D_STAR = 48
    P = _syntax_subspace_prepare()
    X_scaled = P["X_scaled"]
    Y_probes = P["Y_probes"]
    train_idx = P["train_idx"]
    rewrites_by_user_text = P["rewrites_by_user_text"]
    user_to_indices = P["user_to_indices"]
    user_id_list = P["user_id_list"]
    pair_idx = P["pair_idx"]
    raw_dist = P["raw_dist"]
    label_y = P["label_y"]
    leak_train = P["leak_train"]
    leak_test = P["leak_test"]
    probe_train_idx = P["probe_train_idx"]
    probe_test_idx = P["probe_test_idx"]
    PROBE_TARGETS = P["PROBE_TARGETS"]

    # 1. PCA(d=48) (与 Stage1 d=48 完全一致: StandardScaler + fit on 5000 训练句)
    pca = PCA(n_components=D_STAR, random_state=42)
    pca.fit(X_scaled[train_idx])
    X_pca = pca.transform(X_scaled)  # [N, 48]
    ev_ratios = pca.explained_variance_ratio_.copy()
    log(f"[stage2] PCA({D_STAR}) fit. 总 EV={ev_ratios.sum():.4f}")

    # 2. 固定 Rank1 候选集 (seed=123, 与 Stage1 一致)
    eval_users = [uid for uid in user_id_list
                  if uid in rewrites_by_user_text and len(rewrites_by_user_text[uid]) > 0]
    rng_local = np.random.default_rng(123)
    cand_sent_list = []
    pos_n_list = []
    user_mean_list = []
    for uid in eval_users:
        idx_u = user_to_indices[uid]
        n_pos = min(2, len(idx_u))
        pos_idx = rng_local.choice(idx_u, size=n_pos, replace=False)
        other_uids = [u for u in user_id_list if u != uid]
        neg_uids = rng_local.choice(other_uids, size=min(4, len(other_uids)), replace=False)
        neg_idx = [user_to_indices[nu][0] for nu in neg_uids]
        cand_sent_list.append(np.array(list(pos_idx) + neg_idx, dtype=np.int64))
        pos_n_list.append(n_pos)
        user_mean_list.append(X_pca[idx_u].mean(axis=0))
    E = len(eval_users)
    max_cand = max(len(c) for c in cand_sent_list)
    max_pos = max(pos_n_list)
    CAND_FULL = np.zeros((E, max_cand, D_STAR), dtype=np.float64)
    PAD = np.ones((E, max_cand), dtype=bool)
    USER_MEAN = np.stack(user_mean_list, axis=0)  # [E, 48]
    for e, cs in enumerate(cand_sent_list):
        CAND_FULL[e, :len(cs)] = X_pca[cs]
        PAD[e, :len(cs)] = False
    # positive j 应在 candidate 位置 j (pos 在前 n_pos)
    pos_rank_ref = np.arange(max_pos)  # (max_pos,)
    valid_pos = pos_rank_ref[None, :] < np.array(pos_n_list)[:, None]  # [E, max_pos] 2D
    log(f"[stage2] Rank1 eval: E={E}, max_cand={max_cand}, max_pos={max_pos}")

    # 固定 leakage 训练子集 (所有 ablation 共用, 保证 delta 可比)
    leak_rng = np.random.default_rng(7)
    leak_train_sub = leak_rng.choice(leak_train, size=min(5000, len(leak_train)), replace=False)

    # 3. 指标函数
    def m_syntax_rho(Xp):
        pca_dist = np.linalg.norm(Xp[pair_idx[:, 0]] - Xp[pair_idx[:, 1]], axis=1)
        rho, _ = spearmanr(raw_dist, pca_dist)
        return float(rho)

    def m_probe_r2(Xp):
        r2s = []
        for j in range(len(PROBE_TARGETS)):
            y = Y_probes[:, j]
            ridge = Ridge(alpha=1.0)
            ridge.fit(Xp[probe_train_idx], y[probe_train_idx])
            pred = ridge.predict(Xp[probe_test_idx])
            ss_res = ((y[probe_test_idx] - pred) ** 2).sum()
            ss_tot = ((y[probe_test_idx] - y[probe_test_idx].mean()) ** 2).sum()
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        return float(np.mean(r2s))

    def m_leakage(Xp):
        logreg = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
        logreg.fit(Xp[leak_train_sub], label_y[leak_train_sub])
        pred = logreg.predict(Xp[leak_test])
        return float((pred == label_y[leak_test]).mean())

    def m_rank_mrr(CAND, UM):
        # CAND/UM 已是 ablate 后矩阵; 距离向量化
        dists = np.linalg.norm(CAND - UM[:, None, :], axis=2)  # [E, max_cand]
        dists[PAD] = np.inf
        order = np.argsort(dists, axis=1)
        ranks = np.argsort(order, axis=1)  # ranks[e,j] = candidate j 的排序位置
        pos_ranks = ranks[:, :max_pos]
        hits = (pos_ranks == pos_rank_ref) & valid_pos
        n_eval = max(int(valid_pos.sum()), 1)
        rank1 = float(hits.sum()) / n_eval
        mrr = float((1.0 / (pos_ranks + 1.0) * valid_pos).sum()) / n_eval
        return rank1, mrr

    def eval_all(Xp, CAND, UM):
        rho = m_syntax_rho(Xp)
        pr2 = m_probe_r2(Xp)
        rank1, mrr = m_rank_mrr(CAND, UM)
        leak = m_leakage(Xp)
        return {"syntax_rho": rho, "syntax_probe_r2": pr2,
                "query_rank1": rank1, "query_mrr": mrr, "content_leakage": leak}

    # baseline (48 PC 全保留)
    base = eval_all(X_pca, CAND_FULL, USER_MEAN)
    log(f"[stage2] baseline d=48: ρ={base['syntax_rho']:.4f} "
        f"ProbeR²={base['syntax_probe_r2']:.4f} "
        f"Rank1={base['query_rank1']:.4f} MRR={base['query_mrr']:.4f} "
        f"Leak={base['content_leakage']:.4f} (对照 Stage1 d=48: "
        f"0.9988/0.9679/0.2094/0.4626/0.0258)")

    # 4. leave-one-PC-out
    per_pc = {}
    for k in range(D_STAR):
        Xp_ab = np.delete(X_pca, k, axis=1)
        CAND_ab = np.delete(CAND_FULL, k, axis=2)
        UM_ab = np.delete(USER_MEAN, k, axis=1)
        m = eval_all(Xp_ab, CAND_ab, UM_ab)
        per_pc[k] = {
            "ev_ratio": float(ev_ratios[k]),
            "syntax_rho": m["syntax_rho"],
            "syntax_probe_r2": m["syntax_probe_r2"],
            "query_rank1": m["query_rank1"],
            "query_mrr": m["query_mrr"],
            "content_leakage": m["content_leakage"],
        }
        if (k + 1) % 12 == 0 or k == D_STAR - 1:
            log(f"[stage2]   PC{k:>2}: ρ={m['syntax_rho']:.4f} "
                f"ProbeR²={m['syntax_probe_r2']:.4f} "
                f"Rank1={m['query_rank1']:.4f} Leak={m['content_leakage']:.4f}")

    # 5. delta + 分类
    EPS = 1e-4
    table = []
    for k in range(D_STAR):
        m = per_pc[k]
        d_rho = m["syntax_rho"] - base["syntax_rho"]
        d_probe = m["syntax_probe_r2"] - base["syntax_probe_r2"]
        d_rank1 = m["query_rank1"] - base["query_rank1"]
        d_mrr = m["query_mrr"] - base["query_mrr"]
        d_leak = m["content_leakage"] - base["content_leakage"]
        syntax_hurt = max(0.0, -d_rho) + max(0.0, -d_probe)
        leak_help = max(0.0, -d_leak)
        if leak_help > EPS and syntax_hurt <= EPS:
            cls = "leakage"
        elif syntax_hurt > EPS and leak_help <= EPS:
            cls = "syntax"
        elif syntax_hurt > EPS and leak_help > EPS:
            cls = "mixed"
        else:
            cls = "neutral"
        table.append({
            "pc": k, "ev_ratio": m["ev_ratio"],
            "d_rho": d_rho, "d_probe": d_probe, "d_rank1": d_rank1,
            "d_mrr": d_mrr, "d_leak": d_leak,
            "syntax_hurt": syntax_hurt, "leak_help": leak_help, "class": cls,
        })

    # 6. refined subspace: 移除 top-L leakage PC
    leak_sorted = sorted(range(D_STAR), key=lambda k: per_pc[k]["content_leakage"])  # 升序
    refined = {}
    for L in [4, 8, 12, 16]:
        drop = leak_sorted[:L]
        Xp_r = np.delete(X_pca, drop, axis=1)
        CAND_r = np.delete(CAND_FULL, drop, axis=2)
        UM_r = np.delete(USER_MEAN, drop, axis=1)
        m = eval_all(Xp_r, CAND_r, UM_r)
        refined[L] = {
            "n_dim": D_STAR - L,
            "syntax_rho": m["syntax_rho"],
            "syntax_probe_r2": m["syntax_probe_r2"],
            "query_rank1": m["query_rank1"],
            "query_mrr": m["query_mrr"],
            "content_leakage": m["content_leakage"],
        }
        log(f"[stage2] refined drop={L:>2}: dim={D_STAR - L:>2} "
            f"ρ={m['syntax_rho']:.4f} ProbeR²={m['syntax_probe_r2']:.4f} "
            f"Rank1={m['query_rank1']:.4f} MRR={m['query_mrr']:.4f} "
            f"Leak={m['content_leakage']:.4f} (ΔLeak={m['content_leakage'] - base['content_leakage']:+.4f})")

    # 输出
    out = RESIDUAL_SCRATCH / "syntax_subspace_stage2_pc48.json"
    payload = {
        "d_star": D_STAR,
        "baseline": base,
        "per_pc": per_pc,
        "table": table,
        "refined": refined,
        "leak_sorted_asc": leak_sorted,
        "n_sentences": int(len(X_scaled)),
        "n_users": int(len(user_id_list)),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"\n[stage2] 已写入 {out}")

    # 打印 per-PC 摘要 (按 ΔLeak 升序; 负值=移除后 leakage 下降=leakage 维)
    log("\n[stage2] === Per-PC (按 ΔLeak 升序; ΔLeak<0 = leakage 维, 移除后更纯) ===")
    log(f"{'PC':>3} {'EV%':>6} {'Δρ':>7} {'ΔProbe':>8} {'ΔRank1':>8} {'ΔMRR':>8} {'ΔLeak':>8} {'class':>8}")
    for row in sorted(table, key=lambda r: r["d_leak"]):
        log(f"{row['pc']:>3} {row['ev_ratio'] * 100:>5.2f} {row['d_rho']:>+7.4f} "
            f"{row['d_probe']:>+8.4f} {row['d_rank1']:>+8.4f} {row['d_mrr']:>+8.4f} "
            f"{row['d_leak']:>+8.4f} {row['class']:>8}")

    n_leak = sum(1 for t in table if t["class"] == "leakage")
    n_syn = sum(1 for t in table if t["class"] == "syntax")
    n_mix = sum(1 for t in table if t["class"] == "mixed")
    n_neu = sum(1 for t in table if t["class"] == "neutral")
    log(f"\n[stage2] 分类汇总: leakage={n_leak}, syntax={n_syn}, mixed={n_mix}, neutral={n_neu}")


def _ss_eval_all(Xp, CAND, UM, *, pair_idx, raw_dist, Y_probes, probe_train_idx,
                 probe_test_idx, leak_train_sub, leak_test, label_y, PROBE_TARGETS, PAD,
                 pos_rank_ref, valid_pos, max_pos):
    """Syntax Subspace 共用指标计算 (syntax rho / probe R2 / rank1 / mrr / leakage).

    输入 Xp/CAND/UM 已是投影后的子空间矩阵 (任意维数)。
    """
    from sklearn.linear_model import Ridge, LogisticRegression
    from scipy.stats import spearmanr

    # max_pos 由调用方显式传入 (positive 最大数量), 避免从 3D valid_pos 推断

    # 1. syntax rho
    pca_dist = np.linalg.norm(Xp[pair_idx[:, 0]] - Xp[pair_idx[:, 1]], axis=1)
    rho, _ = spearmanr(raw_dist, pca_dist)

    # 2. probe R2
    r2s = []
    for j in range(len(PROBE_TARGETS)):
        y = Y_probes[:, j]
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xp[probe_train_idx], y[probe_train_idx])
        pred = ridge.predict(Xp[probe_test_idx])
        ss_res = ((y[probe_test_idx] - pred) ** 2).sum()
        ss_tot = ((y[probe_test_idx] - y[probe_test_idx].mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    pr2 = float(np.mean(r2s))

    # 3. rank1 / mrr (向量化)
    dists = np.linalg.norm(CAND - UM[:, None, :], axis=2)
    dists[PAD] = np.inf
    order = np.argsort(dists, axis=1)
    ranks = np.argsort(order, axis=1)
    pos_ranks = ranks[:, :max_pos]
    hits = (pos_ranks == pos_rank_ref) & valid_pos
    n_eval = max(int(valid_pos.sum()), 1)
    rank1 = float(hits.sum()) / n_eval
    mrr = float((1.0 / (pos_ranks + 1.0) * valid_pos).sum()) / n_eval

    # 4. content leakage
    logreg = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
    logreg.fit(Xp[leak_train_sub], label_y[leak_train_sub])
    leak = float((logreg.predict(Xp[leak_test]) == label_y[leak_test]).mean())

    return {"syntax_rho": float(rho), "syntax_probe_r2": pr2,
            "query_rank1": rank1, "query_mrr": mrr, "content_leakage": leak}


def main_syntax_subspace_stage3a() -> None:
    """Stage 3A: PC48-96 incremental syntax/content analysis + PC0 length sanity.

    (a) 对每个增量 PC k (48..95, 即第 49..96 个主成分):
        subspace = {PC0..47} u {PC_k} (49 维), 测 6 指标, 对比 d=48 baseline:
          DeltaSyntax (rho/Probe/Rank1/MRR) ~= 0  -> 额外维不带来句法信息
          DeltaLeak  > 0                        -> 额外维引入 content leakage
        => 证明 48 是 "句法饱和 + 语义泄漏开始进入" 的拐点

    (b) PC0 length-controlled sanity check:
        PC0 占 23.3% EV 且独扛 ~63% 句法 probe 信号, 验证其不是句长假象:
          - 原始 Spearman(PC0, 句法目标)
          - partial Spearman(PC0, 句法目标 | n_tok) 控制句长
          - 按 n_tok 分桶的分层相关
        若控制句长后仍显著 -> PC0 是真正句法维 (强化论文论点)
    """
    from sklearn.decomposition import PCA
    from scipy.stats import spearmanr

    KMIN, KMAX = 48, 96  # 分析 PC index 48..95 (第 49..96 个主成分)
    P = _syntax_subspace_prepare()
    X_scaled = P["X_scaled"]
    Y_probes = P["Y_probes"]
    train_idx = P["train_idx"]
    rewrites_by_user_text = P["rewrites_by_user_text"]
    user_to_indices = P["user_to_indices"]
    user_id_list = P["user_id_list"]
    pair_idx = P["pair_idx"]
    raw_dist = P["raw_dist"]
    label_y = P["label_y"]
    leak_train = P["leak_train"]
    leak_test = P["leak_test"]
    probe_train_idx = P["probe_train_idx"]
    probe_test_idx = P["probe_test_idx"]
    PROBE_TARGETS = P["PROBE_TARGETS"]

    # PCA(182) — PC0..95 与 PCA(96) 完全一致 (主成分固定)
    pca = PCA(n_components=182, random_state=42)
    pca.fit(X_scaled[train_idx])
    X_pca = pca.transform(X_scaled)  # [N, 182]
    ev = pca.explained_variance_ratio_
    log(f"[stage3a] PCA(182) fit. 累计 EV(48)={ev[:48].sum():.4f}, EV(96)={ev[:96].sum():.4f}")

    # Rank1 候选结构 (182维, seed=123 与 Stage1/2 一致)
    eval_users = [uid for uid in user_id_list
                  if uid in rewrites_by_user_text and len(rewrites_by_user_text[uid]) > 0]
    rng_local = np.random.default_rng(123)
    cand_sent_list = []
    pos_n_list = []
    user_mean_list = []
    for uid in eval_users:
        idx_u = user_to_indices[uid]
        n_pos = min(2, len(idx_u))
        pos_idx = rng_local.choice(idx_u, size=n_pos, replace=False)
        other = [u for u in user_id_list if u != uid]
        neg_uids = rng_local.choice(other, size=min(4, len(other)), replace=False)
        neg_idx = [user_to_indices[nu][0] for nu in neg_uids]
        cand_sent_list.append(np.array(list(pos_idx) + neg_idx, dtype=np.int64))
        pos_n_list.append(n_pos)
        user_mean_list.append(X_pca[idx_u].mean(axis=0))
    E = len(eval_users)
    max_cand = max(len(c) for c in cand_sent_list)
    max_pos = max(pos_n_list)
    CAND_FULL = np.zeros((E, max_cand, 182), dtype=np.float64)
    PAD = np.ones((E, max_cand), dtype=bool)
    USER_MEAN = np.stack(user_mean_list, axis=0)
    for e, cs in enumerate(cand_sent_list):
        CAND_FULL[e, :len(cs)] = X_pca[cs]
        PAD[e, :len(cs)] = False
    pos_rank_ref = np.arange(max_pos)  # (max_pos,)
    valid_pos = pos_rank_ref[None, :] < np.array(pos_n_list)[:, None]  # [E, max_pos] 2D
    log(f"[stage3a] Rank1 eval: E={E}, max_cand={max_cand}")

    leak_rng = np.random.default_rng(7)
    leak_train_sub = leak_rng.choice(leak_train, size=min(5000, len(leak_train)), replace=False)

    # baseline d=48
    base_cols = list(range(48))
    base = _ss_eval_all(X_pca[:, base_cols], CAND_FULL[:, :, base_cols], USER_MEAN[:, base_cols],
                        pair_idx=pair_idx, raw_dist=raw_dist, Y_probes=Y_probes,
                        probe_train_idx=probe_train_idx, probe_test_idx=probe_test_idx,
                        leak_train_sub=leak_train_sub, label_y=label_y, PROBE_TARGETS=PROBE_TARGETS,
                        PAD=PAD, pos_rank_ref=pos_rank_ref, valid_pos=valid_pos, max_pos=max_pos, leak_test=leak_test)
    log(f"[stage3a] baseline d=48: rho={base['syntax_rho']:.4f} ProbeR2={base['syntax_probe_r2']:.4f} "
        f"Rank1={base['query_rank1']:.4f} MRR={base['query_mrr']:.4f} Leak={base['content_leakage']:.4f}")

    # (a) incremental PC48..95
    inc = {}
    for k in range(KMIN, KMAX):
        cols = base_cols + [k]
        m = _ss_eval_all(X_pca[:, cols], CAND_FULL[:, :, cols], USER_MEAN[:, cols],
                         pair_idx=pair_idx, raw_dist=raw_dist, Y_probes=Y_probes,
                         probe_train_idx=probe_train_idx, probe_test_idx=probe_test_idx,
                         leak_train_sub=leak_train_sub, label_y=label_y, PROBE_TARGETS=PROBE_TARGETS,
                         PAD=PAD, pos_rank_ref=pos_rank_ref, valid_pos=valid_pos, max_pos=max_pos, leak_test=leak_test)
        inc[k] = {
            "ev_ratio": float(ev[k]),
            "syntax_rho": m["syntax_rho"], "syntax_probe_r2": m["syntax_probe_r2"],
            "query_rank1": m["query_rank1"], "query_mrr": m["query_mrr"], "content_leakage": m["content_leakage"],
            "d_rho": m["syntax_rho"] - base["syntax_rho"],
            "d_probe": m["syntax_probe_r2"] - base["syntax_probe_r2"],
            "d_rank1": m["query_rank1"] - base["query_rank1"],
            "d_mrr": m["query_mrr"] - base["query_mrr"],
            "d_leak": m["content_leakage"] - base["content_leakage"],
        }
        if (k - KMIN + 1) % 12 == 0 or k == KMAX - 1:
            log(f"[stage3a]   PC{k:>2}(EV{ev[k]*100:4.1f}%): Drho={inc[k]['d_rho']:+.4f} "
                f"DProbe={inc[k]['d_probe']:+.4f} DRank1={inc[k]['d_rank1']:+.4f} "
                f"DLeak={inc[k]['d_leak']:+.4f}")

    # 汇总 PC49:96 (index 48..95)
    ks = list(range(KMIN, KMAX))
    mean_d_rho = float(np.mean([inc[k]["d_rho"] for k in ks]))
    mean_d_probe = float(np.mean([inc[k]["d_probe"] for k in ks]))
    mean_d_rank1 = float(np.mean([inc[k]["d_rank1"] for k in ks]))
    mean_d_leak = float(np.mean([inc[k]["d_leak"] for k in ks]))
    frac_leak_pos = float(np.mean([1 if inc[k]["d_leak"] > 0 else 0 for k in ks]))
    frac_syn_pos = float(np.mean([1 if inc[k]["d_probe"] > 0 else 0 for k in ks]))
    log(f"[stage3a] 汇总 PC49:96 (n={len(ks)}): "
        f"meanDrho={mean_d_rho:+.4f} meanDProbe={mean_d_probe:+.4f} "
        f"meanDRank1={mean_d_rank1:+.4f} meanDLeak={mean_d_leak:+.4f} | "
        f"frac(DLeak>0)={frac_leak_pos:.2f} frac(DProbe>0)={frac_syn_pos:.2f}")

    # (b) PC0 length-controlled sanity
    def _residual(a, c):
        A = np.hstack([np.ones((len(a), 1)), c.reshape(-1, 1)])
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - (coef[0] + coef[1] * c)

    def _bucket_corr(a, b, c, q=4):
        edges = np.quantile(c, np.linspace(0, 1, q + 1))
        out = []
        for i in range(q):
            if i < q - 1:
                m = (c >= edges[i]) & (c < edges[i + 1])
            else:
                m = (c >= edges[i]) & (c <= edges[i + 1])
            if m.sum() < 20:
                out.append(None)
                continue
            r, _ = spearmanr(a[m], b[m])
            out.append(float(r))
        return out

    pc0 = X_pca[:, 0]
    n_tok = Y_probes[:, PROBE_TARGETS.index("n_tok")]
    sanity = {}
    log("[stage3a] PC0 length-controlled sanity (control=n_tok):")
    log(f"{'target':>10} {'raw_rho':>8} {'partial_rho':>11} {'bucket_rho':>26}")
    for t in ["max_depth", "nest_max", "n_clause", "mean_dist", "depth_var"]:
        j = PROBE_TARGETS.index(t)
        y = Y_probes[:, j]
        r_raw, _ = spearmanr(pc0, y)
        r_part, _ = spearmanr(_residual(pc0, n_tok), _residual(y, n_tok))
        bk = _bucket_corr(pc0, y, n_tok, q=4)
        sanity[t] = {"raw": float(r_raw), "partial": float(r_part), "buckets": bk}
        log(f"{t:>10} {r_raw:>+8.3f} {r_part:>+11.3f} {str([None if x is None else round(x, 3) for x in bk]):>26}")

    # 输出
    out = RESIDUAL_SCRATCH / "syntax_subspace_stage3a_pc48_96.json"
    payload = {
        "kmin": KMIN, "kmax": KMAX, "n_pc_analyzed": len(ks),
        "baseline_d48": base,
        "incremental": inc,
        "summary_pc49_96": {
            "mean_d_rho": mean_d_rho, "mean_d_probe": mean_d_probe,
            "mean_d_rank1": mean_d_rank1, "mean_d_leak": mean_d_leak,
            "frac_dleak_pos": frac_leak_pos, "frac_dprobe_pos": frac_syn_pos,
        },
        "pc0_length_sanity": sanity,
        "n_sentences": int(len(X_scaled)), "n_users": int(len(user_id_list)),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"\n[stage3a] 已写入 {out}")

def main_syntax_subspace_stage4() -> None:
    """Stage 4: History-size Saturation Experiment (同一批用户, 冻结 PCA48 + 冻结 PC importance).

    核心问题: 需要多少条历史评论, 才能稳定估计并充分利用 48d 句法子空间?

    设计要点 (与用户提案一致, 按数据实际 cap 适配):
      - 数据: sentences_for_rewrite_10k.jsonl, 每用户最多 30 条有效句 (实测 max=30)
        => 固定留出 N_TEST=5 条作 held-out test (永不参与 Gaussian 拟合),
           训练池 = 25 条, n ∈ {5,10,15,20,25}
      - 同一批用户: 固定子集 N_EVAL_USERS=2000 (≥30 条的用户)
      - 冻结 PCA48: 与 Stage1/2/3 完全一致 (StandardScaler + PCA(48) fit on 5000 训练句)
      - 冻结 PC importance w_j: 来自 Stage2 leave-one-PC-out 的 |ΔProbe_j| 归一化
      - 每个 (用户, n, seed) 用 n 条历史估计 Gaussian N(mu, diag Sigma w/ shrinkage)
      - 20 个随机 subsampling seeds, 避免抽样随机性

    指标组:
      G1 Held-out Mahal↓ / NLL↓ (test set 完全相同跨 n)
      G2 Reliable-PC Count: R_j(n)=Corr(mu_{u,j}^{(n,a)}, mu_{u,j}^{(n,b)}) 跨用户;
          reliable if R_j>=0.8; 报告 Reliable/48 与 Critical/13
      G3 Cov 稳定性: mean Cond(Sigma)↓ + mean ||Sigma^(a)-Sigma^(b)||_F↓
      G4 Downstream: 固定 candidate pool (5 自身 test + 25 他用户 test),
          以用户 n-Gaussian 对数似然重排, 得 Rank@1↑ / MRR↑
      Utilization: U(n)=sum_j w_j R_j(n)  (syntax-weighted 48d 利用率)
      n* = min n 满足 Rank@1>=0.95*max, U>=0.95, NLL 进入最低 5% 区间
    """
    from sklearn.decomposition import PCA
    import hashlib
    import collections

    # ---- 固定配置 (硬编码) ----
    N_LIST = [5, 10, 15, 20, 25, 30, 35, 40]
    N_SEEDS = 20
    N_TEST = 10
    N_EVAL_USERS = 2000
    LAMBDA_SHRINK = 0.3
    VAR_EPS = 1e-3
    CRIT_N = 13
    RELIABLE_THR = 0.8

    STAGE2_JSON = RESIDUAL_SCRATCH / "syntax_subspace_stage2_pc48.json"

    # 冻结 PCA48 + StandardScaler: 来自原始 30-cap 数据, 与 Stage1/2/3 完全一致
    P0 = _syntax_subspace_prepare()
    pca = PCA(n_components=48, random_state=42)
    pca.fit(P0["X_scaled"][P0["train_idx"]])
    scaler = P0["scaler"]
    fnames = P0["feature_names_ordered"]
    log(f"[stage4] 冻结 PCA48 fit (baseline EV={pca.explained_variance_ratio_.sum():.4f})")

    # 稠密数据: 从完整 review 重新抽句 (同一批 10k 用户, 更深历史), 用相同冻结 PCA 映射
    DENSE_SENTS = os.environ.get("PQ_SYNTAX_SENTS_DENSE",
                                 str(RESIDUAL_SCRATCH / "sentences_for_rewrite_10k_dense.jsonl"))
    DENSE_FEAT = os.environ.get("PQ_SYNTAX_FEAT_DENSE",
                                str(RESIDUAL_SCRATCH / "sentences_318d_cache_dense.jsonl.gz"))
    Pd = _syntax_subspace_prepare(sents_path=DENSE_SENTS, feat_cache_path=DENSE_FEAT,
                                  feature_names_ordered=fnames)
    X_raw = Pd["X"]
    Z = pca.transform(scaler.transform(X_raw))  # [N,48] 同一冻结 48d 空间
    global_var = Z.var(axis=0)   # [48] shrinkage 先验 (对角线)
    user_to_indices = Pd["user_to_indices"]
    user_id_list = Pd["user_id_list"]
    log(f"[stage4] 稠密数据 PCA(48) 映射完成; Z shape {Z.shape}; global_var[0..2]={global_var[:3]}")

    # ---- 选同一批用户 (≥30 条) + 固定子集 ----
    cand_users = [u for u in user_id_list if len(user_to_indices[u]) >= (N_TEST + max(N_LIST))]
    rng_sel = np.random.default_rng(2024)
    eval_users = list(rng_sel.choice(cand_users, size=min(N_EVAL_USERS, len(cand_users)), replace=False))
    eval_users = sorted(eval_users)
    log(f"[stage4] eval users = {len(eval_users)} (cand >=30 用户共 {len(cand_users)})")

    def _uid_seed(uid):
        return int(hashlib.md5(str(uid).encode()).hexdigest(), 16) % (2 ** 31)

    # 每个用户: 固定 hold-out 5 条 test + 训练池
    test_z = {}      # uid -> [N_TEST, 48]
    pool_idx = {}    # uid -> list of sentence indices (训练池)
    for u in eval_users:
        idx_u = np.array(user_to_indices[u], dtype=np.int64)
        rng_u = np.random.default_rng(_uid_seed(u))
        perm = rng_u.permutation(len(idx_u))
        test_local = perm[:N_TEST]
        train_local = perm[N_TEST:]
        test_z[u] = Z[idx_u[test_local]]
        pool_idx[u] = list(idx_u[train_local])

    # 固定 downstream candidate pool: 每用户 5 自身 test + 25 他用户 test (取相邻 5 个 eval 用户)
    cand_pos = {}   # uid -> [5,48]
    cand_neg = {}   # uid -> [25,48]
    Eu = eval_users
    for i, u in enumerate(Eu):
        negs = []
        for k in range(1, 6):
            ou = Eu[(i + k) % len(Eu)]
            negs.append(test_z[ou])
        cand_neg[u] = np.concatenate(negs, axis=0)  # [25,48]
        cand_pos[u] = test_z[u]                     # [5,48]

    # 冻结 PC importance w_j (来自 Stage2)
    if not STAGE2_JSON.exists():
        raise FileNotFoundError(f"缺少 Stage2 结果: {STAGE2_JSON} (请先跑 syntax_subspace_stage2)")
    s2 = json.load(open(STAGE2_JSON))
    tbl = {t["pc"]: t for t in s2["table"]}
    dprobe = np.array([tbl[j]["d_probe"] for j in range(48)], dtype=np.float64)
    w = np.abs(dprobe) / (np.abs(dprobe).sum() + 1e-12)  # [48]
    crit_pcs = sorted(range(48), key=lambda j: -np.abs(dprobe[j]))[:CRIT_N]
    log(f"[stage4] PC importance w loaded; top3 w = {w[np.argsort(-w)[:3]]}; crit_pcs={crit_pcs}")

    # ---- 主循环: 存 mu/var 用于可靠性 & cov 稳定性 ----
    Nn = len(N_LIST)
    Nsu = len(eval_users)
    MU = [np.zeros((Nsu, 48, N_SEEDS), dtype=np.float32) for _ in range(Nn)]
    VAR = [np.zeros((Nsu, 48, N_SEEDS), dtype=np.float32) for _ in range(Nn)]

    # 聚合累加器 (G1/G4)
    maha_sum = np.zeros(Nn); nll_sum = np.zeros(Nn)
    rank1_sum = np.zeros(Nn); mrr_sum = np.zeros(Nn)
    cnt_un = np.zeros(Nn)

    for ni, n in enumerate(N_LIST):
        for ui, u in enumerate(eval_users):
            pool = pool_idx[u]
            if len(pool) < n:
                continue
            for s in range(N_SEEDS):
                rng_s = np.random.default_rng((_uid_seed(u) * 131 + n * 17 + s * 1009) & 0x7fffffff)
                samp = rng_s.choice(pool, size=n, replace=False)
                zt = Z[samp]  # [n,48]
                mu = zt.mean(axis=0)
                cov = zt.var(axis=0)
                var = (1 - LAMBDA_SHRINK) * cov + LAMBDA_SHRINK * global_var + VAR_EPS
                MU[ni][ui, :, s] = mu.astype(np.float32)
                VAR[ni][ui, :, s] = var.astype(np.float32)
                # G1 held-out
                tz = test_z[u]  # [N_TEST,48]
                d = (tz - mu[None, :]) ** 2 / var[None, :]
                maha = d.sum(axis=1).mean()
                nll = (0.5 * (np.log(2 * np.pi) + np.log(var) + d).sum(axis=1)).mean()
                maha_sum[ni] += maha; nll_sum[ni] += nll
                # G4 downstream
                cand = np.concatenate([cand_pos[u], cand_neg[u]], axis=0)  # [30,48]
                cd = (cand - mu[None, :]) ** 2 / var[None, :]
                loglik = -0.5 * (np.log(2 * np.pi) + np.log(var) + cd).sum(axis=1)  # [30]
                order = np.argsort(-loglik)  # 降序, 0=最好
                # 正向: positions 0..4 是 positives
                rank1_hit = int(order[0] < 5)
                rank1_sum[ni] += rank1_hit
                mrr_u = 0.0
                for p in range(5):
                    rk = int(np.where(order == p)[0][0]) + 1
                    mrr_u += 1.0 / rk
                mrr_sum[ni] += mrr_u / 5.0
                cnt_un[ni] += 1

    maha_mean = maha_sum / cnt_un
    nll_mean = nll_sum / cnt_un
    rank1_mean = rank1_sum / cnt_un
    mrr_mean = mrr_sum / cnt_un
    log(f"[stage4] G1/G4 done. Mahal={np.round(maha_mean,3)} NLL={np.round(nll_mean,2)} "
        f"Rank1={np.round(rank1_mean,4)} MRR={np.round(mrr_mean,4)}")

    # ---- G2 Reliable-PC + Utilization ----
    seed_pairs = [(s, s + 1) for s in range(0, N_SEEDS - 1, 2)]  # 10 pairs
    R = np.zeros((Nn, 48))  # R[ni, j]
    for ni in range(Nn):
        M = MU[ni]  # [U,48,S]
        for j in range(48):
            rs = []
            for a, b in seed_pairs:
                xa = M[:, j, a]; xb = M[:, j, b]
                if xa.std() < 1e-9 or xb.std() < 1e-9:
                    rs.append(1.0 if np.allclose(xa, xb) else 0.0)
                else:
                    rs.append(float(np.corrcoef(xa, xb)[0, 1]))
            R[ni, j] = np.mean(rs)
    reliable = (R >= RELIABLE_THR)
    reliable_count = reliable.sum(axis=1)            # /48
    crit_rel = np.array([reliable[:, j].sum() for j in crit_pcs])  # 每 crit PC 是否 reliable
    # crit_rel 是 per-PC reliable(跨 n), 取 "各 n 下 critical PCs reliable 数" = sum over crit pcs of reliable[ni,j]
    crit_reliable_count = np.array([int(reliable[ni, crit_pcs].sum()) for ni in range(Nn)])
    U = (w[None, :] * R).sum(axis=1)  # [Nn]
    log(f"[stage4] G2 done. Reliable/48={reliable_count} Critical/13={crit_reliable_count} U={np.round(U,4)}")

    # ---- G3 Cov 稳定性 ----
    cond_mean = np.zeros(Nn)
    frob_mean = np.zeros(Nn)
    for ni in range(Nn):
        V = VAR[ni]  # [U,48,S]
        # Cond = max(var)/min(var) per user, 跨 seeds 取均值
        c = (V.max(axis=1) / V.min(axis=1)).mean()  # 对 seed 维先 mean? 用全部 seed 的 var
        cond_mean[ni] = float(c)
        # Frobenius diff between seed pairs
        fd = []
        for a, b in seed_pairs:
            diff = V[:, :, a] - V[:, :, b]
            fd.append(np.sqrt((diff ** 2).sum(axis=1)).mean())
        frob_mean[ni] = float(np.mean(fd))
    log(f"[stage4] G3 done. Cond={np.round(cond_mean,2)} FrobDiff={np.round(frob_mean,4)}")

    # ---- n* 规则 ----
    rank_max = rank1_mean.max()
    nll_min = nll_mean.min()
    nll_low5 = nll_mean <= (nll_min + 0.05 * (nll_mean.max() - nll_min) + 1e-9)
    n_star = None
    for ni, n in enumerate(N_LIST):
        ok_rank = rank1_mean[ni] >= 0.95 * rank_max
        ok_u = U[ni] >= 0.95
        ok_nll = nll_low5[ni]
        if ok_rank and ok_u and ok_nll:
            n_star = n
            break
    log(f"[stage4] n* = {n_star} (Rank@1max={rank_max:.4f})")

    # ---- 输出 ----
    out = RESIDUAL_SCRATCH / "syntax_subspace_stage4_history_saturation.json"
    rows = []
    for ni, n in enumerate(N_LIST):
        rows.append({
            "n": n,
            "mahal": float(maha_mean[ni]), "nll": float(nll_mean[ni]),
            "cond": float(cond_mean[ni]), "frob_diff": float(frob_mean[ni]),
            "reliable_pc": int(reliable_count[ni]), "critical_pc": int(crit_reliable_count[ni]),
            "utilization": float(U[ni]), "rank1": float(rank1_mean[ni]), "mrr": float(mrr_mean[ni]),
        })
    payload = {
        "config": {"N_LIST": N_LIST, "N_SEEDS": N_SEEDS, "N_TEST": N_TEST,
                   "N_EVAL_USERS": len(eval_users), "LAMBDA_SHRINK": LAMBDA_SHRINK,
                   "RELIABLE_THR": RELIABLE_THR, "CRIT_N": CRIT_N,
                   "note": "dense re-extraction from full reviews (same 10k users); test=10, n_max=40; PCA48 + PC-importance frozen from original 30-cap data"},
        "rows": rows,
        "n_star": n_star,
        "per_pc_R": {str(n): R[ni].tolist() for ni, n in enumerate(N_LIST)},
        "w_importance": w.tolist(),
        "crit_pcs": crit_pcs,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"\n[stage4] 已写入 {out}")

    # 打印汇总表
    log("\n[stage4] === History-size Saturation (d*=48, frozen PCA48) ===")
    log(f"{'n':>3} {'Mahal':>7} {'NLL':>8} {'Cond':>8} {'FrobD':>7} "
        f"{'RelPC/48':>8} {'CritPC/13':>9} {'Utliz':>7} {'Rank1':>7} {'MRR':>7}")
    for r in rows:
        log(f"{r['n']:>3} {r['mahal']:>7.3f} {r['nll']:>8.2f} {r['cond']:>8.2f} {r['frob_diff']:>7.4f} "
            f"{r['reliable_pc']:>6}/48 {r['critical_pc']:>5}/13 {r['utilization']:>7.4f} "
            f"{r['rank1']:>7.4f} {r['mrr']:>7.4f}")
    log(f"[stage4] n* = {n_star}")

def main_syntax_subspace() -> None:
    """Syntax Subspace Selection Experiment — Stage 1: 9 维度 scan (Q1: 多少维足够).

    输入: sentence-level 318d cache + user_id → sentence map
    标准化: StandardScaler on training set (5000 sentences)
    PCA: 在 5000 训练句上 fit, transform 全部 149326 句
    测试维度: d ∈ {8, 16, 24, 32, 48, 64, 96, 128, 182}

    6 指标:
      1. Explained Var: PCA.explained_variance_ratio_.sum()
      2. Syntax ρ: Spearman(原始182d pair-distance, PCA-d 重建 pair-distance)
      3. Syntax Probe R²: Ridge(PCA-d → {max_depth, n_clause, nest_max, mean_dist, depth_var, n_tok}) mean R²
      4. Query Rank1: 用户 u 真实句 vs 1 negative neutral rewrite, 按 user_vector 距离排序
      5. MRR: 同上但多 negative
      6. Content Leakage: asin 分类 accuracy (Top-1)
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    import collections
    import gzip
    import hashlib as _hl

    P = _syntax_subspace_prepare()
    X_scaled = P["X_scaled"]
    Y_probes = P["Y_probes"]
    user_ids = P["user_ids"]
    asins = P["asins"]
    train_idx = P["train_idx"]
    scaler = P["scaler"]
    rewrites_by_user_text = P["rewrites_by_user_text"]
    user_to_indices = P["user_to_indices"]
    user_id_list = P["user_id_list"]
    pair_idx = P["pair_idx"]
    raw_dist = P["raw_dist"]
    top_asins = P["top_asins"]
    asin_to_label = P["asin_to_label"]
    label_y = P["label_y"]
    leak_train = P["leak_train"]
    leak_test = P["leak_test"]
    probe_train_idx = P["probe_train_idx"]
    probe_test_idx = P["probe_test_idx"]
    PROBE_TARGETS = P["PROBE_TARGETS"]
    feature_names_ordered = P["feature_names_ordered"]
    rng = P["rng"]

    # --- 6. 9 维度 PCA scan ---
    DIMS = [8, 16, 24, 32, 48, 64, 96, 128, 182]
    results = {}

    for d in DIMS:
        log(f"\n[subspace] ===== d={d} =====")
        pca = PCA(n_components=d, random_state=42)
        pca.fit(X_scaled[train_idx])
        X_pca = pca.transform(X_scaled)

        # 1. Explained Var
        ev = float(pca.explained_variance_ratio_.sum())
        log(f"  Explained Var: {ev:.4f}")

        # 2. Syntax ρ
        pca_dist = np.linalg.norm(X_pca[pair_idx[:, 0]] - X_pca[pair_idx[:, 1]], axis=1)
        rho, _ = spearmanr(raw_dist, pca_dist)
        log(f"  Syntax ρ: {rho:.4f}")

        # 3. Syntax Probe R² (Ridge regression)
        r2_scores = []
        for j in range(len(PROBE_TARGETS)):
            y = Y_probes[:, j]
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_pca[probe_train_idx], y[probe_train_idx])
            pred = ridge.predict(X_pca[probe_test_idx])
            ss_res = ((y[probe_test_idx] - pred) ** 2).sum()
            ss_tot = ((y[probe_test_idx] - y[probe_test_idx].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_scores.append(float(r2))
        probe_r2 = float(np.mean(r2_scores))
        log(f"  Syntax Probe R²: {probe_r2:.4f}")

        # 4 & 5. Query Rank1 / MRR
        # 用户 u 的 user vector (PCA-d 后) vs 候选 {u 原句 (positive), 1 random neutral rewrite (negative)}
        # 距离: ||user_vec - candidate_vec||
        rank1_hits = 0
        mrr_total = 0.0
        n_eval = 0
        rng_local = np.random.default_rng(123)
        for uid in user_id_list:
            idx_u = user_to_indices[uid]
            if uid not in rewrites_by_user_text or len(rewrites_by_user_text[uid]) == 0:
                continue
            # 随机抽 5 个用户的句子作为 candidate pool: 1 positive (u 原句) + 4 negative (其他用户 neutral rewrite)
            n_pos = min(2, len(idx_u))
            pos_idx = rng_local.choice(idx_u, size=n_pos, replace=False)
            # 4 negatives: 其他用户 random 原句 (比 rewrite 简单)
            other_uids = [u for u in user_id_list if u != uid]
            if not other_uids:
                continue
            neg_uids = rng_local.choice(other_uids, size=min(4, len(other_uids)), replace=False)
            neg_idx = []
            for nu in neg_uids:
                neg_idx.extend(user_to_indices[nu][:1])  # 取每用户 1 个句子
            if not neg_idx:
                continue
            # 距离计算
            user_vec_pca = X_pca[idx_u].mean(axis=0)
            candidates_pca = np.concatenate([X_pca[pos_idx], X_pca[neg_idx]], axis=0)
            dists = np.linalg.norm(candidates_pca - user_vec_pca[None, :], axis=1)
            # positive 排在 candidates 前 n_pos
            n_cands = len(candidates_pca)
            order = np.argsort(dists)
            for k in range(n_pos):
                # positive 应该在 top-(k+1) 才算 hit
                pos_rank = int(np.where(order == k)[0][0])
                if pos_rank == k:
                    rank1_hits += 1
                mrr_total += 1.0 / (pos_rank + 1)
                n_eval += 1
        rank1 = rank1_hits / max(n_eval, 1)
        mrr = mrr_total / max(n_eval, 1)
        log(f"  Query Rank1: {rank1:.4f} ({rank1_hits}/{n_eval})")
        log(f"  MRR: {mrr:.4f}")

        # 6. Content Leakage (Logistic regression on asin)
        # 用 PCA-d → top-200 asin 分类, 评估 test accuracy
        # 限制训练样本数避免太慢
        leak_train_sub = rng.choice(leak_train, size=min(5000, len(leak_train)), replace=False)
        logreg = LogisticRegression(max_iter=200, solver="lbfgs", n_jobs=-1)
        logreg.fit(X_pca[leak_train_sub], label_y[leak_train_sub])
        leak_pred = logreg.predict(X_pca[leak_test])
        leak_acc = float((leak_pred == label_y[leak_test]).mean())
        log(f"  Content Leakage (Top-200 asin acc): {leak_acc:.4f}")

        results[d] = {
            "explained_var": ev,
            "syntax_rho": rho,
            "syntax_probe_r2": probe_r2,
            "query_rank1": rank1,
            "query_mrr": mrr,
            "content_leakage": leak_acc,
        }

    # --- 7. 输出 ---
    out = RESIDUAL_SCRATCH / "syntax_subspace_scan_d{}.json".format("_".join(map(str, DIMS)))
    with open(out, "w") as f:
        json.dump({"dims": DIMS, "results": results, "n_sentences": int(len(X)), "n_users": int(len(user_id_list))}, f, indent=2)
    log(f"\n[subspace] 已写入 {out}")

    # 打印 summary
    log("\n[subspace] === Summary ===")
    log(f"{'d':>4} {'EV':>6} {'ρ':>6} {'ProbeR²':>8} {'Rank1':>7} {'MRR':>7} {'Leak':>6}")
    for d in DIMS:
        r = results[d]
        log(f"{d:>4} {r['explained_var']:>6.3f} {r['syntax_rho']:>6.3f} {r['syntax_probe_r2']:>8.3f} {r['query_rank1']:>7.3f} {r['query_mrr']:>7.3f} {r['content_leakage']:>6.3f}")


def main() -> None:
    """Parse subcommand and dispatch."""
    parser = argparse.ArgumentParser(
        prog="gaussian_vades",
        description="VADES single entry point (train/probe/validate/compare/smoke/eval_bucket/syntax_subspace)",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("train", help="Train VADES + 排序 query (default)")
    sub.add_parser("probe", help="SVR 10-fold probe eval")
    sub.add_parser("validate", help="高斯假设检验 + Q-Q 图")
    sub.add_parser("compare", help="6 prior 对比")
    sub.add_parser("smoke", help="3 个非高斯模式 smoke test")
    sub.add_parser("eval_bucket", help="按评论数 bucket 评估 user-level 297d Gaussian 质量 (Mahalanobis/NLL/CondNum)")
    sub.add_parser("syntax_subspace", help="Stage 1: 9 维度 syntax subspace scan (Q1 多少维足够)")
    sub.add_parser("syntax_subspace_stage2", help="Stage 2: leave-one-PC-out on d*=48 (识别 syntax vs leakage PC)")
    sub.add_parser("syntax_subspace_stage3a", help="Stage 3A: PC48-96 增量 syntax/content 分析 + PC0 长度控制 sanity")
    sub.add_parser("syntax_subspace_stage4", help="Stage 4: History-size Saturation (冻结 PCA48, 同批用户, n∈5..25, 20 seeds)")
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
    elif cmd == "eval_bucket":
        main_eval_bucket()
    elif cmd == "syntax_subspace":
        main_syntax_subspace()
    elif cmd == "syntax_subspace_stage2":
        main_syntax_subspace_stage2()
    elif cmd == "syntax_subspace_stage3a":
        main_syntax_subspace_stage3a()
    elif cmd == "syntax_subspace_stage4":
        main_syntax_subspace_stage4()
    else:
        raise ValueError(f"未知 subcommand: {cmd}")


if __name__ == "__main__":
    main()
