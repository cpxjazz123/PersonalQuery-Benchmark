#!/usr/bin/env python3
"""验证 VADES 模型中用户表达分布的高斯性假设。

对每个用户，使用已训练的 encoder 提取其 20 个句子的 latent mu，
检验这些 latent 是否符合 N(user_mu, diag(user_var)) 分布。

实验设计：
  1. 加载 VADES encoder 和 user_table (diagonal covariance)
  2. 编码所有句子 → 20 维 latent mu
  3. 对每个用户运行 4 组检验：
     a. 单变量 (20 维 × 用户): Shapiro-Wilk / KS / Anderson-Darling
        — 在 (z - user_mu) / sqrt(user_var) 标准化残差上做 1-sample 检验
     b. 多元 Mardia 偏度 + 峰度 (标准化残差矩阵)
     c. 模型拟合: 2-sample 能量距离 (实际 vs N(user_mu, diag(user_var)) 抽样)
  4. 与基线对比: 同样流程跑 N 组从用户自身 N(μ,σ²) 中抽样的合成样本
  5. 画 Q-Q 图、p-value 分布、能量距离分布

输出 (per category):
  result/personal_query/10_complexity_analysis_clause_features/<Cat>/gaussian_validation_<Cat>/
    ├── summary.json
    ├── per_user_results.jsonl
    ├── per_user_baseline_results.jsonl
    └── figures/
        ├── qq_representative.png
        ├── shapiro_per_dim.png
        ├── energy_distribution.png
        ├── mardia_pvalue.png
        └── rejection_rates.png
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / CATEGORY
SENTENCE_FILE = INPUT_DIR / "vades_lite_sentence_user_distribution_train10_holdout10_sentences.jsonl"
USER_PROFILE_FILE = INPUT_DIR / "vades_lite_sentence_user_distribution_train10_holdout10_user_profiles.jsonl"
SUMMARY_FILE = INPUT_DIR / "vades_lite_sentence_user_distribution_train10_holdout10_summary.json"
ENCODER_CKPT = INPUT_DIR / "vades_encoder.pt"
USER_TABLE_CKPT = INPUT_DIR / "vades_user_table.pt"

OUTPUT_TAG = "gaussian_validation"
OUTPUT_DIR = INPUT_DIR / f"{OUTPUT_TAG}_{CATEGORY}"
FIG_DIR = OUTPUT_DIR / "figures"

SEED = 42
BATCH_SIZE = 4096
LATENT_DIM_EXPECTED = 20
N_SAMPLES_PER_USER = 10  # 实际数据中各用户最少 10 句，统一用 10 句做检验
ALPHA = 0.05

# 性能控制：能量距离检验是瓶颈，做 999 次置换 (n=10, m=100) 约 0.3-0.5s/用户
ENERGY_N_SYNTH = 100
ENERGY_PERMUTATIONS = 199
ENERGY_USER_SUBSET = 500  # 实际数据跑能量距离的用户子集
BASELINE_N_USERS = 100  # 基线抽样用户数
BASELINE_N_REPLICATES = 30  # 每用户合成样本组数

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as vades  # noqa: E402


def log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_compiled_state_dict(model: torch.nn.Module, ckpt: dict) -> None:
    """Load state dict that may have _orig_mod. prefix from torch.compile."""
    sd = ckpt["model_state_dict"]
    new_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd)


def load_vades_model(device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module, int]:
    encoder_ckpt = torch.load(ENCODER_CKPT, map_location=device, weights_only=False)
    user_table_ckpt = torch.load(USER_TABLE_CKPT, map_location=device, weights_only=False)
    if encoder_ckpt["covariance_mode"] != "diagonal":
        raise ValueError(f"本实验只支持对角协方差模型，得到: {encoder_ckpt['covariance_mode']}")
    if encoder_ckpt["latent_dim"] != LATENT_DIM_EXPECTED:
        raise ValueError(f"latent_dim 与预期不符: {encoder_ckpt['latent_dim']} vs {LATENT_DIM_EXPECTED}")

    encoder = vades.SentenceEncoder(
        input_dim=encoder_ckpt["input_dim"],
        hidden_dim=encoder_ckpt["hidden_dim"],
        latent_dim=encoder_ckpt["latent_dim"],
    ).to(device)
    load_compiled_state_dict(encoder, encoder_ckpt)
    encoder.eval()

    user_table = vades.UserDistributionTable(
        num_users=user_table_ckpt["num_users"],
        latent_dim=user_table_ckpt["latent_dim"],
    ).to(device)
    load_compiled_state_dict(user_table, user_table_ckpt)
    user_table.eval()
    return encoder, user_table, encoder_ckpt["latent_dim"]


def encode_sentences(encoder, sentence_rows: list[dict], feature_names: list[str], device: torch.device) -> np.ndarray:
    """对所有句子进行 latent mu 编码，返回 (N, latent_dim) 矩阵。"""
    feature_matrix = np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in sentence_rows],
        dtype=np.float32,
    )
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std < 1e-8] = 1.0
    scaled = (feature_matrix - mean) / std
    scaled_tensor = torch.tensor(scaled, dtype=torch.float32, device=device)

    latents = np.empty((len(sentence_rows), LATENT_DIM_EXPECTED), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(sentence_rows), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(sentence_rows))
            mu, _, _ = encoder(scaled_tensor[start:end])
            latents[start:end] = mu.detach().cpu().numpy()
    return latents


def mardia_kurtosis_test(z_standardized: np.ndarray) -> dict:
    """Mardia 多元峰度检验。

    警告: 在 n=p (本实验 n=20, p=20) 渐近 p-value 不可靠 (样本协方差退化)。
    b2p 统计量本身仍可比，渐近期望 p(p+2)=440, 渐近方差 8p(p+2)/n。
    """
    n, p = z_standardized.shape
    centered = z_standardized - z_standardized.mean(axis=0)
    cov = np.cov(centered, rowvar=False, bias=False)
    cov = cov + 1e-6 * np.eye(p)
    cov_inv = np.linalg.inv(cov)
    d2 = np.sum(centered @ cov_inv * centered, axis=1)
    b2p = float(np.mean(d2**2))
    expected_mean = p * (p + 2)
    expected_var = 8.0 * p * (p + 2) / n
    z_stat = (b2p - expected_mean) / np.sqrt(expected_var)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))
    return {
        "b2p": b2p,
        "expected_mean": float(expected_mean),
        "expected_std": float(np.sqrt(expected_var)),
        "z_stat": float(z_stat),
        "p_value": p_value,
        "p_value_reliable": False,
        "note": "n=p=20, 渐近 p-value 不可靠；以 b2p 分布与基线对比为准",
    }


def per_dim_univariate_tests(z_standardized: np.ndarray) -> dict:
    """对 (n, p) 矩阵的每列做 Shapiro-Wilk / KS / AD。"""
    n, p = z_standardized.shape
    out: dict = {"shapiro": [], "ks": [], "anderson": []}
    for d in range(p):
        col = z_standardized[:, d]
        sw_stat, sw_p = stats.shapiro(col)
        out["shapiro"].append({"stat": float(sw_stat), "p_value": float(sw_p)})
        ks_stat, ks_p = stats.kstest(col, "norm", args=(0.0, 1.0))
        out["ks"].append({"stat": float(ks_stat), "p_value": float(ks_p)})
        ad_res = stats.anderson(col, dist="norm")
        out["anderson"].append({"stat": float(ad_res.statistic), "critical_5pct": float(ad_res.critical_values[2])})
    return out


def energy_distance_stat(data: np.ndarray, synth: np.ndarray) -> float:
    """计算 2-sample 能量距离 (Székely & Rizzo)。

    D(F, G) = sqrt(2 E|X-Y| - E|X-X'| - E|Y-Y'|)
    其中 X,X' iid F, Y,Y' iid G。值域 [0, +inf)，D=0 表示同分布。
    """
    n = data.shape[0]
    m = synth.shape[0]
    if n < 2 or m < 2:
        raise ValueError(f"样本数不足: n={n}, m={m}")
    # 欧氏距离（不是平方距离）
    xx = np.sqrt(np.sum((data[:, None, :] - data[None, :, :]) ** 2, axis=-1) + 1e-12)
    yy = np.sqrt(np.sum((synth[:, None, :] - synth[None, :, :]) ** 2, axis=-1) + 1e-12)
    xy = np.sqrt(np.sum((data[:, None, :] - synth[None, :, :]) ** 2, axis=-1) + 1e-12)
    xx_mean = xx[np.triu_indices(n, k=1)].mean()
    yy_mean = yy[np.triu_indices(m, k=1)].mean()
    xy_mean = xy.mean()
    d2 = 2.0 * xy_mean - xx_mean - yy_mean
    # 数值稳定：截断到非负
    d2 = max(0.0, d2)
    return float(np.sqrt(d2))


def energy_distance_pvalue(data: np.ndarray, synth: np.ndarray, n_permutations: int, rng: np.random.Generator) -> dict:
    """2-sample 能量距离检验：计算统计量 + 置换 p-value。"""
    obs = energy_distance_stat(data, synth)
    combined = np.concatenate([data, synth], axis=0)
    perm_stats = np.empty(n_permutations, dtype=np.float64)
    n = data.shape[0]
    for k in range(n_permutations):
        perm = rng.permutation(combined.shape[0])
        perm_data = combined[perm[:n]]
        perm_synth = combined[perm[n:]]
        perm_stats[k] = energy_distance_stat(perm_data, perm_synth)
    p_value = float((np.sum(perm_stats >= obs) + 1) / (n_permutations + 1))
    return {"energy_distance": obs, "p_value": p_value}


def aggregate_pvalues(records: list[dict], test_key: str, alpha: float = ALPHA) -> dict:
    """聚合每用户 shapiro/ks/mardia/energy 的 p-value 列表 (per-dim 多次检验展平)。"""
    pvalues = []
    for r in records:
        test = r.get(test_key)
        if test is None:
            continue
        if isinstance(test, list):
            for entry in test:
                if "p_value" in entry:
                    pvalues.append(entry["p_value"])
        else:
            if "p_value" in test:
                pvalues.append(test["p_value"])
    if not pvalues:
        raise ValueError(f"无 p-value 可聚合: {test_key}")
    arr = np.asarray(pvalues, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "alpha": alpha,
        "n_rejected": int(np.sum(arr < alpha)),
        "rejection_rate": float(np.mean(arr < alpha)),
        "p_mean": float(np.mean(arr)),
        "p_median": float(np.median(arr)),
        "p_std": float(np.std(arr)),
        "p_q10": float(np.quantile(arr, 0.1)),
        "p_q25": float(np.quantile(arr, 0.25)),
    }


def aggregate_mardia_b2p(records: list[dict]) -> dict:
    """聚合 Mardia b2p 统计量：实际数据 vs 渐近期望 p(p+2) 的距离。"""
    b2ps = []
    for r in records:
        m = r.get("mardia_kurtosis")
        if m is not None and "b2p" in m:
            b2ps.append(m["b2p"])
    if not b2ps:
        raise ValueError("无 b2p 可聚合")
    arr = np.asarray(b2ps, dtype=np.float64)
    expected_mean = 20 * 22
    return {
        "n": int(len(arr)),
        "b2p_mean": float(np.mean(arr)),
        "b2p_median": float(np.median(arr)),
        "b2p_std": float(np.std(arr)),
        "b2p_q05": float(np.quantile(arr, 0.05)),
        "b2p_q25": float(np.quantile(arr, 0.25)),
        "b2p_q75": float(np.quantile(arr, 0.75)),
        "b2p_q95": float(np.quantile(arr, 0.95)),
        "expected_mean": float(expected_mean),
        "mean_distance_from_expected": float(np.mean(arr - expected_mean)),
        "fraction_above_expected": float(np.mean(arr > expected_mean)),
    }


def per_dim_aggregate(records: list[dict], test_key: str, latent_dim: int, alpha: float = ALPHA) -> list[dict]:
    """按 latent dim 聚合 p-value。"""
    out = []
    for d in range(latent_dim):
        pvalues = []
        for r in records:
            test = r.get(test_key, [])
            if d < len(test) and "p_value" in test[d]:
                pvalues.append(test[d]["p_value"])
        if not pvalues:
            raise ValueError(f"dim {d} 无 p-value 数据")
        arr = np.asarray(pvalues, dtype=np.float64)
        out.append({
            "dim": d,
            "n": int(len(arr)),
            "rejection_rate_at_alpha": float(np.mean(arr < alpha)),
            "p_mean": float(np.mean(arr)),
            "p_median": float(np.median(arr)),
        })
    return out


def plot_qq_representative(
    latents_by_user: dict[str, np.ndarray],
    user_profiles: dict[str, dict],
    energy_records: list[dict],
    save_path: Path,
    n_samples: int = 9,
) -> list[dict]:
    """为代表性用户 (按 energy distance 排序) 画 Q-Q 图。"""
    sorted_users = sorted(energy_records, key=lambda r: r["energy_test"]["energy_distance"])
    n = len(sorted_users)
    low = sorted_users[: max(1, n_samples // 3)]
    mid_lo = sorted_users[n // 2 - max(1, n_samples // 6) : n // 2]
    mid_hi = sorted_users[n // 2 : n // 2 + max(1, n_samples // 6)]
    high = sorted_users[-max(1, n_samples // 3) :]
    picks = (low + mid_lo + mid_hi + high)[:n_samples]
    nrows = (len(picks) + 2) // 3
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.atleast_2d(axes).reshape(nrows, ncols)
    selected_meta = []
    for i, rec in enumerate(picks):
        r = i // ncols
        c = i % ncols
        ax = axes_flat[r, c]
        user_id = rec["user_id"]
        latents = latents_by_user[user_id]
        profile = user_profiles[user_id]
        user_mu = np.asarray(profile["user_mu"], dtype=np.float64)
        user_std = np.sqrt(np.asarray(profile["user_var"], dtype=np.float64))
        z = (latents - user_mu) / user_std
        worst_dim = -1
        worst_p = 1.0
        for d in range(z.shape[1]):
            _, p = stats.shapiro(z[:, d])
            if p < worst_p:
                worst_p = p
                worst_dim = d
        stats.probplot(z[:, worst_dim], dist="norm", plot=ax)
        ax.set_title(
            f"user {user_id[:10]}… E²={rec['energy_test']['energy_distance']:.2f}\n"
            f"worst dim {worst_dim}, Shapiro p={worst_p:.3f}",
            fontsize=9,
        )
        ax.legend(["Data", "Reference"], loc="upper left", fontsize=8)
        selected_meta.append({
            "user_id": user_id,
            "energy_distance": rec["energy_test"]["energy_distance"],
            "worst_dim_shapiro": int(worst_dim),
            "worst_dim_shapiro_p": float(worst_p),
        })
    for j in range(len(picks), nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes_flat[r, c].axis("off")
    fig.suptitle("Q-Q plots of standardized residuals (most-informative dim per user)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return selected_meta


def plot_per_dim_pvalue(per_dim_stats: list[dict], save_path: Path) -> None:
    dims = [d["dim"] for d in per_dim_stats]
    rej_rates = [d["rejection_rate_at_alpha"] for d in per_dim_stats]
    p_means = [d["p_mean"] for d in per_dim_stats]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.bar(dims, rej_rates, color="steelblue", alpha=0.8)
    ax1.axhline(ALPHA, color="red", linestyle="--", label=f"α={ALPHA}")
    ax1.set_xlabel("Latent dim")
    ax1.set_ylabel(f"Shapiro-Wilk rejection rate (α={ALPHA})")
    ax1.set_title("Per-dim rejection rate (Shapiro-Wilk)")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax2.bar(dims, p_means, color="darkorange", alpha=0.8)
    ax2.axhline(ALPHA, color="red", linestyle="--", label=f"α={ALPHA}")
    ax2.set_xlabel("Latent dim")
    ax2.set_ylabel("Mean p-value")
    ax2.set_title("Per-dim mean Shapiro-Wilk p-value")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pvalue_distribution(actual_p: list[float], baseline_p: list[float], title: str, xlabel: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(actual_p, bins=40, alpha=0.65, color="steelblue", label="Actual", density=True)
    ax.hist(baseline_p, bins=40, alpha=0.65, color="darkorange", label="Baseline N(μ,σ²)", density=True)
    ax.axvline(ALPHA, color="red", linestyle="--", label=f"α={ALPHA}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_b2p_distribution(actual_b2p: list[float], baseline_b2p: list[float], save_path: Path) -> None:
    """画 Mardia b2p 分布 + Q-Q 图，比较实际 vs 基线。

    注: n=p=20 渐近分布不可靠，此图直接比较 b2p 分布形状。
    """
    actual_arr = np.asarray(actual_b2p, dtype=np.float64)
    baseline_arr = np.asarray(baseline_b2p, dtype=np.float64)
    expected = 20 * 22
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.hist(actual_arr, bins=40, alpha=0.65, color="steelblue", label=f"Actual (n={len(actual_arr)})", density=True)
    ax1.hist(baseline_arr, bins=40, alpha=0.65, color="darkorange", label=f"Baseline (n={len(baseline_arr)})", density=True)
    ax1.axvline(expected, color="red", linestyle="--", label=f"Asymptotic E[b2p]={expected}")
    ax1.set_xlabel("Mardia kurtosis b2p")
    ax1.set_ylabel("Density")
    ax1.set_title("Mardia kurtosis b2p distribution")
    ax1.legend(fontsize=8)

    sorted_actual = np.sort(actual_arr)
    sorted_baseline = np.sort(baseline_arr)
    n = min(len(sorted_actual), len(sorted_baseline))
    q_actual = sorted_actual[np.linspace(0, len(sorted_actual) - 1, n).astype(int)]
    q_baseline = sorted_baseline[np.linspace(0, len(sorted_baseline) - 1, n).astype(int)]
    ax2.scatter(q_baseline, q_actual, alpha=0.5, s=10)
    lo = min(q_baseline.min(), q_actual.min())
    hi = max(q_baseline.max(), q_actual.max())
    ax2.plot([lo, hi], [lo, hi], "r--", label="y=x (相同分布)")
    ax2.set_xlabel("Baseline b2p quantile")
    ax2.set_ylabel("Actual b2p quantile")
    ax2.set_title("Q-Q: actual vs baseline b2p")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rejection_rates(agg_actual: dict, agg_baseline: dict, save_path: Path) -> None:
    tests = list(agg_actual.keys())
    actual_rates = [agg_actual[t]["rejection_rate"] for t in tests]
    baseline_rates = [agg_baseline[t]["rejection_rate"] for t in tests]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tests))
    width = 0.35
    ax.bar(x - width / 2, actual_rates, width, label="Actual data", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, baseline_rates, width, label="Baseline (N(0,I) synthetic)", color="darkorange", alpha=0.85)
    ax.axhline(ALPHA, color="red", linestyle="--", label=f"α={ALPHA}")
    ax.set_xticks(x)
    ax.set_xticklabels(tests, rotation=20, ha="right")
    ax.set_ylabel(f"Rejection rate at α={ALPHA}")
    ymax = max(0.5, max(actual_rates + baseline_rates) * 1.2)
    ax.set_ylim(0, ymax)
    ax.set_title("Per-test rejection rate: actual user latents vs N(0,I) baseline")
    ax.legend()
    for xi, (a, b) in enumerate(zip(actual_rates, baseline_rates)):
        ax.text(xi - width / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + width / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_user_tests(
    latents: np.ndarray,
    user_mu: np.ndarray,
    user_std: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """对单个用户运行 4 组正态性检验。"""
    z = (latents - user_mu) / user_std
    uni = per_dim_univariate_tests(z)
    mardia = mardia_kurtosis_test(z)
    synth = rng.normal(loc=user_mu, scale=user_std, size=(ENERGY_N_SYNTH, len(user_mu)))
    energy = energy_distance_pvalue(latents, synth, ENERGY_PERMUTATIONS, rng)
    return {
        "shapiro": uni["shapiro"],
        "ks": uni["ks"],
        "anderson": uni["anderson"],
        "mardia_kurtosis": mardia,
        "energy_test": energy,
    }


def main() -> None:
    log("=" * 60)
    log("VADES 用户表达分布的高斯性验证")
    log("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"运行设备: {device}")

    log("加载数据...")
    sentence_rows = load_jsonl(SENTENCE_FILE)
    user_profiles = load_jsonl(USER_PROFILE_FILE)
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    feature_names = summary["feature_names"]
    log(f"句子数: {len(sentence_rows)}, 用户数: {len(user_profiles)}, 特征维度: {len(feature_names)}")

    encoder, user_table, latent_dim = load_vades_model(device)
    log(f"VADES 模型加载完成: latent_dim={latent_dim}")

    log("编码所有句子为 latent mu...")
    latents_matrix = encode_sentences(encoder, sentence_rows, feature_names, device)
    log(f"latent 矩阵形状: {latents_matrix.shape}")

    user_to_latents: dict[str, np.ndarray] = {}
    for row, z in zip(sentence_rows, latents_matrix):
        uid = row["user_id"]
        if uid not in user_to_latents:
            user_to_latents[uid] = []
        user_to_latents[uid].append(z)
    for uid in list(user_to_latents.keys()):
        arr = np.asarray(user_to_latents[uid], dtype=np.float64)
        if arr.shape[1] != latent_dim:
            raise ValueError(f"用户 {uid} latent 维度异常: {arr.shape[1]} != {latent_dim}")
        user_to_latents[uid] = arr

    # 只保留有足够句子的用户 + 在 user_profiles 中的用户
    rng_filter = np.random.default_rng(SEED)
    profile_by_id = {p["user_id"]: p for p in user_profiles}
    eligible_users: list[str] = []
    for profile in user_profiles:
        uid = profile["user_id"]
        if uid in user_to_latents and user_to_latents[uid].shape[0] >= N_SAMPLES_PER_USER:
            eligible_users.append(uid)
    log(f"符合条件的用户 (≥{N_SAMPLES_PER_USER} 句): {len(eligible_users)} / {len(user_profiles)}")
    for uid in eligible_users:
        arr = user_to_latents[uid]
        # 固定 N_SAMPLES_PER_USER 句：取前 N 句 (按文档顺序)
        if arr.shape[0] > N_SAMPLES_PER_USER:
            user_to_latents[uid] = arr[:N_SAMPLES_PER_USER]
    user_profiles = [p for p in user_profiles if p["user_id"] in set(eligible_users)]

    log("对每用户运行单变量 + Mardia 检验...")
    main_rng = np.random.default_rng(SEED)
    per_user_results: list[dict] = []
    for profile in user_profiles:
        user_id = profile["user_id"]
        latents = user_to_latents[user_id]
        user_mu = np.asarray(profile["user_mu"], dtype=np.float64)
        user_std = np.sqrt(np.asarray(profile["user_var"], dtype=np.float64))
        z = (latents - user_mu) / user_std
        uni = per_dim_univariate_tests(z)
        mardia = mardia_kurtosis_test(z)
        per_user_results.append({
            "user_id": user_id,
            "shapiro": uni["shapiro"],
            "ks": uni["ks"],
            "anderson": uni["anderson"],
            "mardia_kurtosis": mardia,
        })
    log(f"已完成 {len(per_user_results)} 用户的单变量 + Mardia 检验")

    # 能量距离在子集上运行（置换检验最慢）
    log(f"对 {ENERGY_USER_SUBSET} 用户子集运行能量距离 (置换数 {ENERGY_PERMUTATIONS})...")
    rng_subset = np.random.default_rng(SEED + 1)
    energy_subset_user_ids = rng_subset.choice(
        [p["user_id"] for p in user_profiles], size=min(ENERGY_USER_SUBSET, len(user_profiles)), replace=False
    )
    energy_subset_set = set(energy_subset_user_ids)
    for profile in user_profiles:
        if profile["user_id"] not in energy_subset_set:
            continue
        latents = user_to_latents[profile["user_id"]]
        user_mu = np.asarray(profile["user_mu"], dtype=np.float64)
        user_std = np.sqrt(np.asarray(profile["user_var"], dtype=np.float64))
        user_rng = np.random.default_rng(SEED + hash(profile["user_id"]) % 100000)
        synth = user_rng.normal(loc=user_mu, scale=user_std, size=(ENERGY_N_SYNTH, latent_dim))
        energy = energy_distance_pvalue(latents, synth, ENERGY_PERMUTATIONS, user_rng)
        for r in per_user_results:
            if r["user_id"] == profile["user_id"]:
                r["energy_test"] = energy
                break
    energy_pvalues_actual = [r["energy_test"]["p_value"] for r in per_user_results if "energy_test" in r]
    log(f"能量距离子集完成 (n={len(energy_pvalues_actual)})")

    # 基线：从用户的 N(μ, σ²) 中抽样做同样的检验
    log(f"基线: {BASELINE_N_USERS} 用户 × {BASELINE_N_REPLICATES} 复现")
    rng_baseline = np.random.default_rng(SEED + 2)
    baseline_user_ids = rng_baseline.choice(
        [p["user_id"] for p in user_profiles], size=min(BASELINE_N_USERS, len(user_profiles)), replace=False
    )
    baseline_records: list[dict] = []
    flat_baseline: list[dict] = []
    for uid in baseline_user_ids:
        profile = profile_by_id[uid]
        user_mu = np.asarray(profile["user_mu"], dtype=np.float64)
        user_std = np.sqrt(np.asarray(profile["user_var"], dtype=np.float64))
        rep_results = []
        for rep in range(BASELINE_N_REPLICATES):
            samples = rng_baseline.normal(loc=user_mu, scale=user_std, size=(N_SAMPLES_PER_USER, latent_dim))
            z = (samples - user_mu) / user_std
            uni = per_dim_univariate_tests(z)
            mardia = mardia_kurtosis_test(z)
            # 能量距离：实际 → 同分布合成
            synth = rng_baseline.normal(loc=user_mu, scale=user_std, size=(ENERGY_N_SYNTH, latent_dim))
            energy = energy_distance_pvalue(samples, synth, ENERGY_PERMUTATIONS, rng_baseline)
            record = {
                "shapiro": uni["shapiro"],
                "ks": uni["ks"],
                "anderson": uni["anderson"],
                "mardia_kurtosis": mardia,
                "energy_test": energy,
            }
            rep_results.append(record)
            flat_baseline.append(record)
        baseline_records.append({"user_id": uid, "replicate_results": rep_results})
    log("基线完成")

    # 实际数据聚合
    log("聚合结果...")
    ad_reject_counts = []
    ad_per_dim_reject = np.zeros(latent_dim, dtype=np.int64)
    for r in per_user_results:
        ad = r["anderson"]
        n_reject = 0
        for d, entry in enumerate(ad):
            if entry["stat"] > entry["critical_5pct"]:
                n_reject += 1
                ad_per_dim_reject[d] += 1
        ad_reject_counts.append(n_reject)
    ad_reject_arr = np.asarray(ad_reject_counts, dtype=np.float64)

    agg_actual: dict = {
        "shapiro": aggregate_pvalues(per_user_results, "shapiro"),
        "ks": aggregate_pvalues(per_user_results, "ks"),
        "mardia_kurtosis": aggregate_pvalues(per_user_results, "mardia_kurtosis"),
        "energy_test": aggregate_pvalues(per_user_results, "energy_test"),
        "shapiro_per_dim": per_dim_aggregate(per_user_results, "shapiro", latent_dim),
        "anderson": {
            "n_users": int(len(per_user_results)),
            "alpha": ALPHA,
            "n_users_any_dim_rejected": int(np.sum(ad_reject_arr > 0)),
            "any_dim_rejection_rate": float(np.mean(ad_reject_arr > 0)),
            "mean_dims_rejected": float(np.mean(ad_reject_arr)),
            "median_dims_rejected": float(np.median(ad_reject_arr)),
            "per_dim_reject_count": ad_per_dim_reject.tolist(),
            "per_dim_reject_rate": (ad_per_dim_reject / max(1, len(per_user_results))).tolist(),
        },
    }
    agg_baseline: dict = {
        "shapiro": aggregate_pvalues(flat_baseline, "shapiro"),
        "ks": aggregate_pvalues(flat_baseline, "ks"),
        "mardia_kurtosis": aggregate_pvalues(flat_baseline, "mardia_kurtosis"),
        "energy_test": aggregate_pvalues(flat_baseline, "energy_test"),
        "mardia_b2p": aggregate_mardia_b2p(flat_baseline),
    }
    agg_actual["mardia_b2p"] = aggregate_mardia_b2p(per_user_results)

    log("画图...")
    qq_meta = plot_qq_representative(
        latents_by_user=user_to_latents,
        user_profiles=profile_by_id,
        energy_records=per_user_results,
        save_path=FIG_DIR / "qq_representative.png",
    )
    plot_per_dim_pvalue(agg_actual["shapiro_per_dim"], FIG_DIR / "shapiro_per_dim.png")
    plot_pvalue_distribution(
        energy_pvalues_actual,
        [r["p_value"] for fr in flat_baseline for r in [fr["energy_test"]]],
        "Energy test p-value distribution",
        "Energy test p-value",
        FIG_DIR / "energy_pvalue.png",
    )
    plot_b2p_distribution(
        [r["mardia_kurtosis"]["b2p"] for r in per_user_results],
        [fr["mardia_kurtosis"]["b2p"] for fr in flat_baseline],
        FIG_DIR / "mardia_b2p.png",
    )
    plot_pvalue_distribution(
        [r["shapiro"][0]["p_value"] for r in per_user_results],
        [fr["shapiro"][0]["p_value"] for fr in flat_baseline],
        "Shapiro-Wilk p-value distribution (dim 0)",
        "Shapiro p-value (dim 0)",
        FIG_DIR / "shapiro_pvalue.png",
    )
    plot_rejection_rates(
        {k: agg_actual[k] for k in ["shapiro", "ks", "energy_test"]},
        {k: agg_baseline[k] for k in ["shapiro", "ks", "energy_test"]},
        FIG_DIR / "rejection_rates.png",
    )

    summary_payload = {
        "category": CATEGORY,
        "n_users": len(user_profiles),
        "n_sentences_per_user": N_SAMPLES_PER_USER,
        "latent_dim": latent_dim,
        "alpha": ALPHA,
        "energy_n_synth": ENERGY_N_SYNTH,
        "energy_permutations": ENERGY_PERMUTATIONS,
        "energy_user_subset": len(energy_pvalues_actual),
        "baseline_n_users": BASELINE_N_USERS,
        "baseline_n_replicates_per_user": BASELINE_N_REPLICATES,
        "caveat_mardia": "n=p=20 使 Mardia 渐近 p-value 不可靠；以 b2p 分布 + Q-Q 图为准",
        "actual": agg_actual,
        "baseline": agg_baseline,
        "qq_representative": qq_meta,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    per_user_path = OUTPUT_DIR / "per_user_results.jsonl"
    with per_user_path.open("w", encoding="utf-8") as handle:
        for row in per_user_results:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    baseline_path = OUTPUT_DIR / "per_user_baseline_results.jsonl"
    with baseline_path.open("w", encoding="utf-8") as handle:
        for entry in baseline_records:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")

    log(f"结果已保存: {OUTPUT_DIR}")
    log("=" * 60)
    log("=== 关键结论 ===")
    log(f"用户数: {len(user_profiles)}, 能量距离子集: {len(energy_pvalues_actual)}")
    log(f"α = {ALPHA} 下的拒绝率 (实际数据 vs 基线 N(μ,σ²) 抽样):")
    log(f"  Shapiro-Wilk (per-dim, 展平): {agg_actual['shapiro']['rejection_rate']:.3f}  vs  baseline {agg_baseline['shapiro']['rejection_rate']:.3f}")
    log(f"  KS (per-dim, 展平):           {agg_actual['ks']['rejection_rate']:.3f}  vs  baseline {agg_baseline['ks']['rejection_rate']:.3f}")
    log(f"  Energy 2-sample:              {agg_actual['energy_test']['rejection_rate']:.3f}  vs  baseline {agg_baseline['energy_test']['rejection_rate']:.3f}")
    log(f"  Anderson-Darling: 至少 1 维被拒用户比例: {agg_actual['anderson']['any_dim_rejection_rate']:.3f}")
    log("  (Mardia 渐近 p 在 n=p=20 不可靠, 已省略; 见 mardia_b2p.png)")
    log("")
    log("p-value 均值 (H0 成立时理论 ≈ 0.5):")
    log(f"  shapiro:     actual={agg_actual['shapiro']['p_mean']:.3f}, baseline={agg_baseline['shapiro']['p_mean']:.3f}")
    log(f"  ks:          actual={agg_actual['ks']['p_mean']:.3f}, baseline={agg_baseline['ks']['p_mean']:.3f}")
    log(f"  energy:      actual={agg_actual['energy_test']['p_mean']:.3f}, baseline={agg_baseline['energy_test']['p_mean']:.3f}")
    log("")
    log("Mardia b2p 分布 (渐近期望 p(p+2)=440):")
    log(f"  实际: mean={agg_actual['mardia_b2p']['b2p_mean']:.2f}, median={agg_actual['mardia_b2p']['b2p_median']:.2f}, q05={agg_actual['mardia_b2p']['b2p_q05']:.2f}, q95={agg_actual['mardia_b2p']['b2p_q95']:.2f}")
    log(f"  基线: mean={agg_baseline['mardia_b2p']['b2p_mean']:.2f}, median={agg_baseline['mardia_b2p']['b2p_median']:.2f}, q05={agg_baseline['mardia_b2p']['b2p_q05']:.2f}, q95={agg_baseline['mardia_b2p']['b2p_q95']:.2f}")
    log("=" * 60)
    log("判定: 若 actual 拒绝率 ≫ baseline, actual p 均值 < baseline p 均值")
    log("      ⇒ 数据显著偏离 N(μ,σ²), 高斯假设不成立")
    log("      若 actual ≈ baseline (拒绝率都 ≈ α), ⇒ 高斯假设可接受")
    log("      Mardia b2p 实际分布与基线分布形状一致 (Q-Q 在 y=x 附近) ⇒ 高斯")
    log("=" * 60)


if __name__ == "__main__":
    main()
