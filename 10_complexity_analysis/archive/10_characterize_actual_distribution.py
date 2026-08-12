#!/usr/bin/env python3
"""刻画用户句 latent 分布的实际形状。

复现流程:
  1. 从 stage1 reviews 提取前 N 句
  2. 通过 VADES encoder 得到 latent mu
  3. 用 user_mu / user_var 标准化
  4. 计算 per-dim: 偏度 (skewness), 峰度 (kurtosis), Hartigan dip test
  5. 对每个用户拟合 N(μ,σ), Student-t(ν), 2-mixture-of-Gaussians
     用 AIC / BIC 选最优
  6. 画 per-dim 偏度/峰度图、典型 dim 的直方图+拟合曲线

输出:
  result/.../gaussian_validation_<Cat>/characterization/
    ├── characterization_summary.json
    └── figures/
        ├── per_dim_skewness.png
        ├── per_dim_kurtosis.png
        ├── modality_dip_test.png
        ├── best_fit_distribution_per_dim.png
        ├── dim0_histogram_fit_examples.png
        └── overall_evidence.txt
"""

from __future__ import annotations

import json
import os
import sys
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize
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
REVIEW_FILE = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"

VALIDATION_DIR = INPUT_DIR / "gaussian_validation_Baby_Products"
OUTPUT_DIR = VALIDATION_DIR / "characterization"
FIG_DIR = OUTPUT_DIR / "figures"

SEED = 42
N_SAMPLES_PER_USER = 10
LATENT_DIM_EXPECTED = 20
N_USERS_SUBSET = 500  # 限制用户数以加速 (5772 → 500)
GMM_N_INIT = 2  # 减少 GMM 随机重启
GMM_MAX_ITER = 100

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from train_vades_lite_sentence_latent_threshold import (  # noqa: E402
    SentenceEncoder, load_spacy_model, extract_clause_features_from_doc,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_compiled_state_dict(model: torch.nn.Module, ckpt: dict) -> None:
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(sd)


def extract_sentences_for_users(user_ids: list[str], n_per_user: int = N_SAMPLES_PER_USER) -> dict[str, list[str]]:
    """从原始 review JSON 提取每用户前 n 句 (按 VADES 训练的逻辑)。"""
    if not REVIEW_FILE.exists():
        raise FileNotFoundError(f"缺少 review 文件: {REVIEW_FILE}")
    with REVIEW_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    users = payload if isinstance(payload, list) else payload.get("users", [])
    user_id_set = set(user_ids)
    nlp = load_spacy_model()
    out: dict[str, list[str]] = {uid: [] for uid in user_ids}

    for user_row in users:
        uid = user_row.get("user_id")
        if uid not in user_id_set:
            continue
        reviews = user_row.get("reviews")
        if reviews is None:
            reviews = user_row.get("results")
        if not isinstance(reviews, list):
            continue
        seen: set[str] = set()
        for review in reviews:
            target_reviews = review.get("target_reviews")
            if target_reviews is not None:
                if not isinstance(target_reviews, list):
                    continue
                candidates = [t for t in target_reviews if t]
            else:
                t = review.get("text")
                candidates = [t] if t else []
            for src in candidates:
                doc = nlp(src)
                for sent in doc.sents:
                    text = re.sub(r"\s+", " ", sent.text.strip())
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    out[uid].append(text)
                    if len(out[uid]) >= n_per_user:
                        break
                if len(out[uid]) >= n_per_user:
                    break
            if len(out[uid]) >= n_per_user:
                break
    return out


def encode_text_to_latent(
    encoder, user_texts: dict[str, list[str]], scaler_mean: np.ndarray, scaler_std: np.ndarray
) -> dict[str, np.ndarray]:
    """对每用户文本提取 clause 特征，标准化，编码为 latent。"""
    nlp = load_spacy_model()
    feature_names = [
        "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
        "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
        "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
        "max_dependency_distance", "long_dependency_ratio", "amod_count", "advmod_count",
        "nmod_count", "compound_count", "modifier_density", "coordination_count",
        "max_branching_factor",
    ]
    out: dict[str, np.ndarray] = {}
    for uid, texts in user_texts.items():
        feats = []
        for t in texts:
            doc = nlp(t)
            extracted = extract_clause_features_from_doc(doc, t)
            feats.append([extracted[n] for n in feature_names])
        if not feats:
            continue
        F = np.asarray(feats, dtype=np.float32)
        F_scaled = (F - scaler_mean) / scaler_std
        with torch.no_grad():
            mu, _, _ = encoder(torch.tensor(F_scaled, dtype=torch.float32, device=DEVICE))
        out[uid] = mu.detach().cpu().numpy()
    return out


def fit_normal(x: np.ndarray) -> dict:
    """拟合 N(μ, σ)。AIC = 2k - 2lnL。"""
    mu, sigma = np.mean(x), np.std(x, ddof=0)
    n = len(x)
    ll = np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))
    return {"name": "Normal", "params": {"mu": float(mu), "sigma": float(sigma)},
            "loglik": float(ll), "aic": float(2 * 2 - 2 * ll), "bic": float(2 * np.log(n) - 2 * ll)}


def fit_student_t(x: np.ndarray) -> dict:
    """拟合 Student-t(ν, μ, σ)。"""
    nu, mu, sigma = stats.t.fit(x, floc=np.mean(x))
    n = len(x)
    ll = np.sum(stats.t.logpdf(x, df=nu, loc=mu, scale=sigma))
    return {"name": "Student-t", "params": {"nu": float(nu), "mu": float(mu), "sigma": float(sigma)},
            "loglik": float(ll), "aic": float(2 * 3 - 2 * ll), "bic": float(3 * np.log(n) - 2 * ll)}


def fit_2gmm(x: np.ndarray) -> dict:
    """拟合 2-component Gaussian Mixture。用 EM 初始化，固定 2 个 component。"""
    from sklearn.mixture import GaussianMixture
    x_reshape = x.reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=2, n_init=GMM_N_INIT, max_iter=GMM_MAX_ITER, random_state=SEED)
        gmm.fit(x_reshape)
        ll = gmm.score(x_reshape) * len(x_reshape)
        return {
            "name": "GMM-2",
            "params": {
                "weights": gmm.weights_.tolist(),
                "means": gmm.means_.flatten().tolist(),
                "stds": np.sqrt(gmm.covariances_.flatten()).tolist(),
            },
            "loglik": float(ll),
            "aic": float(2 * 6 - 2 * ll),  # 2 个 weight+mean+var + 1 weight 由归一化确定 = 5 个自由参数 + 1 总和约束 → AIC 用 6
            "bic": float(6 * np.log(len(x)) - 2 * ll),
        }
    except Exception as e:
        return {"name": "GMM-2", "params": {}, "loglik": float("nan"),
                "aic": float("inf"), "bic": float("inf"), "error": str(e)}


def hartigan_dip(x: np.ndarray) -> dict:
    """Hartigan's dip test (单峰性检验)。H0: 单峰。"""
    try:
        from diptest import diptest
        dip, pval = diptest(x)
        return {"stat": float(dip), "p_value": float(pval), "multimodal_at_05": bool(pval < 0.05)}
    except ImportError:
        # 用 Silverman 带宽 + KDE 找峰数作为替代
        from scipy.signal import argrelextrema
        kde = stats.gaussian_kde(x)
        xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
        ys = kde(xs)
        peaks = argrelextrema(ys, np.greater)[0]
        n_peaks = len(peaks)
        return {
            "stat": float("nan"),
            "p_value": float("nan"),
            "n_peaks_kde": int(n_peaks),
            "multimodal_at_05": n_peaks > 1,
            "note": "diptest 未安装；用 KDE 峰数估计",
        }


def main() -> None:
    log("=" * 60)
    log("用户表达分布的形状刻画")
    log("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with USER_PROFILE_FILE.open("r", encoding="utf-8") as f:
        user_profiles = [json.loads(l) for l in f if l.strip()]
    with SUMMARY_FILE.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    feature_names = summary["feature_names"]
    log(f"用户数: {len(user_profiles)}")

    # 仅保留有足够句子的用户
    if SENTENCE_FILE.exists():
        log("从 sentence 缓存加载...")
        sentence_rows = [json.loads(l) for l in SENTENCE_FILE.open()]
        user_to_texts: dict[str, list[str]] = defaultdict(list)
        for r in sentence_rows:
            user_to_texts[r["user_id"]].append(r["sentence_text"])
        user_to_texts = {uid: lst[:N_SAMPLES_PER_USER] for uid, lst in user_to_texts.items() if len(lst) >= N_SAMPLES_PER_USER}
        user_ids = [p["user_id"] for p in user_profiles if p["user_id"] in user_to_texts]
    else:
        log(f"sentence 缓存不存在, 从 review 抽取: {REVIEW_FILE}")
        user_ids = [p["user_id"] for p in user_profiles]
        user_to_texts = extract_sentences_for_users(user_ids, N_SAMPLES_PER_USER)
        user_ids = [uid for uid in user_ids if len(user_to_texts.get(uid, [])) >= N_SAMPLES_PER_USER]
    log(f"有效用户: {len(user_ids)}")

    # 计算 sentence feature scaler (用本次的全部数据)
    nlp = load_spacy_model()
    all_texts = [t for lst in user_to_texts.values() for t in lst[:N_SAMPLES_PER_USER]]
    log(f"抽取 clause 特征: {len(all_texts)} 句")
    feat_list = []
    for t in all_texts:
        doc = nlp(t)
        feat_list.append([extract_clause_features_from_doc(doc, t)[n] for n in feature_names])
    F = np.asarray(feat_list, dtype=np.float32)
    scaler_mean = F.mean(axis=0)
    scaler_std = F.std(axis=0)
    scaler_std[scaler_std < 1e-8] = 1.0

    # 加载 encoder
    encoder_ckpt = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    encoder = SentenceEncoder(
        input_dim=encoder_ckpt["input_dim"],
        hidden_dim=encoder_ckpt["hidden_dim"],
        latent_dim=encoder_ckpt["latent_dim"],
    ).to(DEVICE)
    load_compiled_state_dict(encoder, encoder_ckpt)
    encoder.eval()

    log("编码每用户 latent...")
    user_latents = encode_text_to_latent(encoder, user_to_texts, scaler_mean, scaler_std)

    # 计算标准化残差
    profile_by_id = {p["user_id"]: p for p in user_profiles}
    z_by_user: dict[str, np.ndarray] = {}
    for uid in user_ids:
        if uid not in user_latents or uid not in profile_by_id:
            continue
        z = user_latents[uid][:N_SAMPLES_PER_USER]
        mu = np.asarray(profile_by_id[uid]["user_mu"], dtype=np.float64)
        std = np.sqrt(np.asarray(profile_by_id[uid]["user_var"], dtype=np.float64))
        z_norm = (z - mu) / std
        z_by_user[uid] = z_norm

    log(f"得到 {len(z_by_user)} 个用户的标准化残差矩阵")

    # 子集化以加速拟合阶段
    rng_subset = np.random.default_rng(SEED)
    if len(z_by_user) > N_USERS_SUBSET:
        all_uids = list(z_by_user.keys())
        subset_uids = rng_subset.choice(all_uids, size=N_USERS_SUBSET, replace=False).tolist()
        z_by_user = {uid: z_by_user[uid] for uid in subset_uids}
        log(f"子集化: 保留 {len(z_by_user)} 用户以加速拟合")

    # === 计算 per-dim 偏度/峰度 ===
    all_skew = []
    all_kurt = []
    for uid, z in z_by_user.items():
        all_skew.append(stats.skew(z, axis=0, bias=False))
        all_kurt.append(stats.kurtosis(z, axis=0, fisher=True, bias=False))
    all_skew = np.asarray(all_skew)
    all_kurt = np.asarray(all_kurt)

    skew_per_dim = {
        "mean": all_skew.mean(axis=0).tolist(),
        "abs_mean": np.abs(all_skew).mean(axis=0).tolist(),
        "std": all_skew.std(axis=0).tolist(),
    }
    kurt_per_dim = {
        "mean": all_kurt.mean(axis=0).tolist(),
        "std": all_kurt.std(axis=0).tolist(),
    }

    # === 多模态 (Hartigan dip test) per-user 平均 ===
    log("运行 Hartigan dip test (多模态)...")
    dip_per_dim: list[list[dict]] = [[] for _ in range(LATENT_DIM_EXPECTED)]
    n_users_per_dim = 0
    for uid, z in z_by_user.items():
        for d in range(LATENT_DIM_EXPECTED):
            result = hartigan_dip(z[:, d])
            dip_per_dim[d].append(result)
    dip_summary_per_dim = []
    for d in range(LATENT_DIM_EXPECTED):
        tests = dip_per_dim[d]
        # 用 KDE 峰数 (diptest 未安装)
        n_peaks_list = [t.get("n_peaks_kde", 0) for t in tests if "n_peaks_kde" in t]
        if n_peaks_list:
            arr = np.asarray(n_peaks_list)
            dip_summary_per_dim.append({
                "dim": d,
                "n_users": len(tests),
                "n_peaks_mean": float(arr.mean()),
                "n_peaks_median": float(np.median(arr)),
                "fraction_multimodal": float(np.mean(arr > 1)),
                "fraction_unimodal": float(np.mean(arr == 1)),
            })
        else:
            dip_summary_per_dim.append({"dim": d, "n_users": len(tests), "note": "no dip data"})

    # === 拟合 Normal / Student-t / GMM-2 per user per dim, 选最优 (AIC) ===
    log("拟合 Normal / Student-t / GMM-2 per-dim...")
    fit_records: list[dict] = []  # 聚合 best fit 统计
    n_dim = LATENT_DIM_EXPECTED
    best_count = {"Normal": 0, "Student-t": 0, "GMM-2": 0}
    aic_improvement_t_vs_normal: list[float] = []
    aic_improvement_gmm2_vs_normal: list[float] = []

    for uid, z in z_by_user.items():
        for d in range(n_dim):
            x = z[:, d]
            fit_n = fit_normal(x)
            fit_t = fit_student_t(x)
            fit_g = fit_2gmm(x)
            fits = [fit_n, fit_t, fit_g]
            best = min(fits, key=lambda f: f["aic"])
            best_count[best["name"]] += 1
            aic_improvement_t_vs_normal.append(fit_t["aic"] - fit_n["aic"])
            aic_improvement_gmm2_vs_normal.append(fit_g["aic"] - fit_n["aic"])
            fit_records.append({
                "user_id": uid, "dim": d,
                "best_fit": best["name"],
                "aic_normal": fit_n["aic"], "aic_student_t": fit_t["aic"], "aic_gmm2": fit_g["aic"],
                "student_t_nu": fit_t["params"].get("nu"),
            })
        if (hash(uid) % 100) < 2:
            log(f"  拟合进度: 已完成 {len(fit_records)}/{len(z_by_user) * n_dim} (用户 {uid[:10]}...)")

    total_fits = len(fit_records)
    log(f"拟合完成: total={total_fits}")
    log(f"  Normal wins:   {best_count['Normal']} ({best_count['Normal']/total_fits*100:.1f}%)")
    log(f"  Student-t wins:{best_count['Student-t']} ({best_count['Student-t']/total_fits*100:.1f}%)")
    log(f"  GMM-2 wins:    {best_count['GMM-2']} ({best_count['GMM-2']/total_fits*100:.1f}%)")
    log(f"  Student-t AIC < Normal AIC 比例: {np.mean(np.array(aic_improvement_t_vs_normal) < 0)*100:.1f}%")
    log(f"  Student-t 改善的 AIC 均值: {np.mean(aic_improvement_t_vs_normal):.2f} (负值表示 t 更好)")
    log(f"  GMM-2 AIC < Normal AIC 比例: {np.mean(np.array(aic_improvement_gmm2_vs_normal) < 0)*100:.1f}%")

    # === 画图 ===
    log("画图...")

    # 1. 偏度/峰度 per dim
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    dims = np.arange(n_dim)
    ax1.bar(dims, skew_per_dim["mean"], yerr=skew_per_dim["std"], color="steelblue", alpha=0.8, capsize=3)
    ax1.axhline(0, color="red", linestyle="--", alpha=0.5, label="Skew=0 (symmetric)")
    ax1.set_xlabel("Latent dim")
    ax1.set_ylabel("Mean skewness (across users)")
    ax1.set_title(f"Per-dim skewness (mean ± std over {len(z_by_user)} users)\nN(0,1) expects ≈0")
    ax1.legend()
    ax2.bar(dims, kurt_per_dim["mean"], yerr=kurt_per_dim["std"], color="darkorange", alpha=0.8, capsize=3)
    ax2.axhline(0, color="red", linestyle="--", alpha=0.5, label="Excess kurtosis=0 (Gaussian tail)")
    ax2.set_xlabel("Latent dim")
    ax2.set_ylabel("Mean excess kurtosis")
    ax2.set_title("Per-dim excess kurtosis (Fisher; N(0,1) expects ≈0)\n+ = heavy tail, - = light tail / bimodal")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_dim_skew_kurt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. 偏度/峰度的全局分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(all_skew.flatten(), bins=40, color="steelblue", alpha=0.8, density=True)
    ax1.axvline(0, color="red", linestyle="--", label="Skew=0")
    ax1.set_xlabel("Skewness (per-user per-dim)")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribution of per-user per-dim skewness")
    ax1.legend()
    ax2.hist(all_kurt.flatten(), bins=40, color="darkorange", alpha=0.8, density=True)
    ax2.axvline(0, color="red", linestyle="--", label="Excess kurt=0")
    ax2.set_xlabel("Excess kurtosis (per-user per-dim)")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of per-user per-dim excess kurtosis")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "skew_kurt_global.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. dip test (KDE 峰数) per dim
    if dip_summary_per_dim and "n_peaks_mean" in dip_summary_per_dim[0]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        n_peaks_means = [d.get("n_peaks_mean", 1) for d in dip_summary_per_dim]
        frac_multi = [d.get("fraction_multimodal", 0) for d in dip_summary_per_dim]
        ax1.bar(dims, n_peaks_means, color="seagreen", alpha=0.8)
        ax1.axhline(1, color="red", linestyle="--", alpha=0.5, label="Unimodal")
        ax1.set_xlabel("Latent dim")
        ax1.set_ylabel("Mean # of peaks (KDE)")
        ax1.set_title(f"Average number of modes per dim (across {len(z_by_user)} users)")
        ax1.legend()
        ax2.bar(dims, frac_multi, color="indianred", alpha=0.8)
        ax2.set_xlabel("Latent dim")
        ax2.set_ylabel("Fraction of users with multimodal")
        ax2.set_title("Fraction of users showing multimodal per dim (KDE > 1 peak)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "modality_per_dim.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. 拟合胜出比例
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(best_count.keys())
    counts = [best_count[n] for n in names]
    percentages = [c / total_fits * 100 for c in counts]
    bars = ax.bar(names, percentages, color=["steelblue", "darkorange", "seagreen"], alpha=0.85)
    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{pct:.1f}%", ha="center")
    ax.set_ylabel("Fraction (AIC-best across users × dims)")
    ax.set_title(f"Best-fit distribution per (user, dim): n={total_fits} fits")
    ax.set_ylim(0, max(percentages) * 1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_fit_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. AIC 改善分布 (Student-t vs Normal, GMM-2 vs Normal)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(aic_improvement_t_vs_normal, bins=50, color="darkorange", alpha=0.8)
    ax1.axvline(0, color="red", linestyle="--", label="AIC_t = AIC_Normal")
    ax1.set_xlabel("AIC_t - AIC_Normal (负值 = t 更优)")
    ax1.set_ylabel("Count")
    ax1.set_title("Student-t vs Normal (per-user per-dim)")
    ax1.legend()
    ax2.hist(aic_improvement_gmm2_vs_normal, bins=50, color="seagreen", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", label="AIC_GMM2 = AIC_Normal")
    ax2.set_xlabel("AIC_GMM2 - AIC_Normal (负值 = GMM2 更优)")
    ax2.set_ylabel("Count")
    ax2.set_title("2-GMM vs Normal (per-user per-dim)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "aic_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. 典型 dim 直方图 + 拟合曲线
    # 选样本量足够的最具代表性的 dim (peak |skew| + peak |kurt| 各 3)
    flat_idx = np.argsort(np.abs(all_skew.mean(axis=0)))[-3:]
    flat_idx_kurt = np.argsort(np.abs(all_kurt.mean(axis=0)))[-3:]
    example_dims = sorted(set(flat_idx.tolist() + flat_idx_kurt.tolist()))[:6]
    n_ex = len(example_dims)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.flatten()
    for i, d in enumerate(example_dims):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        all_x = np.concatenate([z[:, d] for z in z_by_user.values()])
        ax.hist(all_x, bins=40, density=True, alpha=0.6, color="steelblue", label="data")
        xs = np.linspace(all_x.min() - 0.5, all_x.max() + 0.5, 200)
        # 拟合所有用户合并数据
        fit_n = fit_normal(all_x)
        fit_t = fit_student_t(all_x)
        ax.plot(xs, stats.norm.pdf(xs, fit_n["params"]["mu"], fit_n["params"]["sigma"]), "r-", label=f"N AIC={fit_n['aic']:.1f}")
        ax.plot(xs, stats.t.pdf(xs, df=fit_t["params"]["nu"], loc=fit_t["params"]["mu"], scale=fit_t["params"]["sigma"]), "g--", label=f"t(ν={fit_t['params']['nu']:.1f}) AIC={fit_t['aic']:.1f}")
        ax.set_title(f"dim {d}: skew={skew_per_dim['mean'][d]:.2f}, kurt={kurt_per_dim['mean'][d]:.2f}")
        ax.legend(fontsize=8)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Standardized residual distributions vs N(0,1) and Student-t fits (all users pooled)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dim_histogram_fits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # === 输出文字总结 ===
    n_users = len(z_by_user)
    heavy_tail_frac = float((all_kurt > 0.5).mean())
    light_or_bimodal_frac = float((all_kurt < -0.5).mean())
    symmetric_frac = float((np.abs(all_skew) < 0.5).mean())
    asymmetric_frac = float((np.abs(all_skew) >= 0.5).mean())

    summary_text = f"""
用户表达分布形状诊断 (Baby_Products, n_users={n_users}, n_per_user=10, latent_dim=20)
=================================================================================

1. 偏度 (skewness): N(0,1) 期望 0
   - |偏度| 均值: {np.abs(all_skew).mean():.3f}
   - |偏度| > 0.5 的用户-维对比例: {asymmetric_frac*100:.1f}% (明显不对称)
   - |偏度| < 0.5 的比例: {symmetric_frac*100:.1f}% (基本对称)

2. 峰度 (excess kurtosis, Fisher): N(0,1) 期望 0
   - 平均: {all_kurt.mean():.3f}
   - > 0.5 (重尾) 比例: {heavy_tail_frac*100:.1f}%
   - < -0.5 (轻尾/双峰) 比例: {light_or_bimodal_frac*100:.1f}%
   - 在 [-0.5, 0.5] (高斯区间) 比例: {float(((all_kurt >= -0.5) & (all_kurt <= 0.5)).mean())*100:.1f}%

3. 多模态 (KDE 峰数 > 1 视为多模态)
   - 至少有 1 维多模态的用户比例: {float(np.mean([any(s.get('fraction_multimodal', 0) > 0.1 for s in dip_summary_per_dim if 'fraction_multimodal' in s) for _ in [0]]))*100:.1f}%
   - dim 0-19 中多模态比例 > 30% 的 dim 数: {sum(1 for s in dip_summary_per_dim if s.get('fraction_multimodal', 0) > 0.3)}

4. 候选分布拟合 (AIC 选最优)
   - Normal 胜出: {best_count['Normal']/total_fits*100:.1f}%
   - Student-t 胜出: {best_count['Student-t']/total_fits*100:.1f}%
   - 2-GMM 胜出: {best_count['GMM-2']/total_fits*100:.1f}%
   - Student-t AIC < Normal AIC 的比例: {np.mean(np.array(aic_improvement_t_vs_normal) < 0)*100:.1f}%, 平均改善 {np.mean(aic_improvement_t_vs_normal):.2f}
   - 2-GMM AIC < Normal AIC 的比例: {np.mean(np.array(aic_improvement_gmm2_vs_normal) < 0)*100:.1f}%, 平均改善 {np.mean(aic_improvement_gmm2_vs_normal):.2f}

5. 主要结论
   - 如果 Student-t 胜出比例高 → 用户的 latent 有重尾，模型用 N 不够灵活
   - 如果 2-GMM 胜出比例高 → 用户的表达可能是双峰 (e.g. 简短句 + 长句)
   - 如果偏度显著 → 单峰对称假设被违反
"""

    with (OUTPUT_DIR / "overall_evidence.txt").open("w", encoding="utf-8") as f:
        f.write(summary_text)
    log(summary_text)

    # 写 summary
    summary_payload = {
        "category": CATEGORY,
        "n_users": n_users,
        "n_samples_per_user": N_SAMPLES_PER_USER,
        "latent_dim": n_dim,
        "skew_per_dim": skew_per_dim,
        "kurt_per_dim": kurt_per_dim,
        "dip_summary_per_dim": dip_summary_per_dim,
        "fit_best_counts": best_count,
        "fit_total": total_fits,
        "aic_improvement": {
            "student_t_vs_normal_mean": float(np.mean(aic_improvement_t_vs_normal)),
            "gmm2_vs_normal_mean": float(np.mean(aic_improvement_gmm2_vs_normal)),
            "student_t_better_rate": float(np.mean(np.array(aic_improvement_t_vs_normal) < 0)),
            "gmm2_better_rate": float(np.mean(np.array(aic_improvement_gmm2_vs_normal) < 0)),
        },
        "global_metrics": {
            "abs_skew_mean": float(np.abs(all_skew).mean()),
            "excess_kurt_mean": float(all_kurt.mean()),
            "heavy_tail_rate": heavy_tail_frac,
            "light_or_bimodal_rate": light_or_bimodal_frac,
            "asymmetric_rate": asymmetric_frac,
            "symmetric_rate": symmetric_frac,
        },
    }
    with (OUTPUT_DIR / "characterization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    log(f"结果已保存: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
