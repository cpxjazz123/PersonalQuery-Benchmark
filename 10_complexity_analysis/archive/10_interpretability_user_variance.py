#!/usr/bin/env python3
"""Per-user latent 方差分布 (VADES 风格变异性对应实验).

对每个用户:
- 从 10 个句子的 latent mu 向量 (20 维) 计算 trace(cov) = Σ_i var(mu_i)
- 得到一个 "该用户 latent 表达稳定性" 的标量

输出:
- result/.../interpretability_paper/<CAT>/user_latent_variance_hist.png
- result/.../interpretability_paper/<CAT>/user_latent_variance_ranked.png
- result/.../interpretability_paper/<CAT>/user_latent_variance_stats.json

数据:
- vades_encoder_*.pt: 训练好的 encoder
- <TAG>_sentences.jsonl: 每句 features
- 用 encoder 重跑 inference 得每句 mu，按 user_id 聚合
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORIES = ["Baby_Products", "Pet_Supplies", "Grocery_and_Gourmet_Food"]
FEATURE_BASE = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features"
OUTPUT_BASE = FEATURE_BASE / "interpretability_paper"

GMM_ONLY_TAG = "vades_lite_sentence_user_distribution_train10_holdout10_gmm"
GMM_ALIGN_TAG = "vades_lite_sentence_user_distribution_train10_holdout10_gmm_align"

sys.path.insert(0, str(REPO_ROOT / "PersoanlQuery" / "10_complexity_analysis" / "common"))
import train_vades_lite_sentence_latent_threshold as train_module  # noqa: E402


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_encoder_and_infer_per_sentence_mu(
    category: str, tag: str
) -> dict[str, np.ndarray]:
    """Load encoder, re-run inference on the saved sentences, return {user_id: [n_sent, 20]}."""
    input_dir = FEATURE_BASE / category
    cov_tag = f"_{train_module.COVARIANCE_MODE}" if train_module.COVARIANCE_MODE != "diagonal" else ""
    encoder_path = input_dir / f"vades_encoder_{tag}{cov_tag}.pt"
    user_table_path = input_dir / f"vades_user_table_{tag}{cov_tag}.pt"
    excluded_user_file = input_dir / f"{tag}_excluded_users.jsonl"
    sentence_file = input_dir / f"{tag}_sentences.jsonl"

    log(f"  Loading {encoder_path.name} + {user_table_path.name}")
    train_module.REPO_ROOT = REPO_ROOT
    train_module.CATEGORY = category
    train_module.INPUT_DIR = input_dir
    train_module.DEVICE = train_module.infer_device()

    encoder_ckpt = torch.load(encoder_path, map_location=train_module.DEVICE, weights_only=False)
    ckpt_mode = encoder_ckpt.get("covariance_mode", "diagonal")
    if ckpt_mode != train_module.COVARIANCE_MODE:
        raise ValueError(f"covariance_mode 不一致: ckpt={ckpt_mode}, env={train_module.COVARIANCE_MODE}")

    if train_module.COVARIANCE_MODE == "full":
        encoder = train_module.SentenceEncoderFull(
            input_dim=encoder_ckpt["input_dim"],
            hidden_dim=encoder_ckpt["hidden_dim"],
            latent_dim=encoder_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    else:
        encoder = train_module.SentenceEncoder(
            input_dim=encoder_ckpt["input_dim"],
            hidden_dim=encoder_ckpt["hidden_dim"],
            latent_dim=encoder_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    new_sd = {k.replace("_orig_mod.", ""): v for k, v in encoder_ckpt["model_state_dict"].items()}
    encoder.load_state_dict(new_sd)
    encoder.eval()

    user_table_ckpt = torch.load(user_table_path, map_location=train_module.DEVICE, weights_only=False)
    if train_module.COVARIANCE_MODE == "full":
        user_table = train_module.UserDistributionTableFull(
            num_users=user_table_ckpt["num_users"], latent_dim=user_table_ckpt["latent_dim"]
        ).to(train_module.DEVICE)
    elif train_module.COVARIANCE_MODE == "diagonal_gmm":
        user_table = train_module.UserDistributionTableGMM(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
            num_components=user_table_ckpt.get("gmm_components", int(train_module.os.environ.get("VADES_GMM_K", "2"))),
        ).to(train_module.DEVICE)
    else:
        user_table = train_module.UserDistributionTable(
            num_users=user_table_ckpt["num_users"], latent_dim=user_table_ckpt["latent_dim"]
        ).to(train_module.DEVICE)
    new_sd = {k.replace("_orig_mod.", ""): v for k, v in user_table_ckpt["model_state_dict"].items()}
    user_table.load_state_dict(new_sd)
    user_table.eval()

    log(f"  Building dataset from {sentence_file.name}")
    sentence_rows = _load_jsonl(sentence_file)
    if excluded_user_file.exists():
        excluded_ids = {r["user_id"] for r in _load_jsonl(excluded_user_file)}
        sentence_rows = [r for r in sentence_rows if r["user_id"] not in excluded_ids]
    feature_names = list(sentence_rows[0]["features"].keys())
    user_ids_filtered, dataset = train_module.build_training_dataset(sentence_rows, feature_names)
    log(f"  Dataset: {len(user_ids_filtered)} users, {len(sentence_rows)} sentences")

    log(f"  Running encoder inference...")
    feature_tensor = torch.tensor(dataset["scaled_features"], dtype=torch.float32, device=train_module.DEVICE)
    with torch.no_grad():
        sent_mu, sent_dispersion, _ = encoder(feature_tensor)
    sent_mu_np = sent_mu.detach().cpu().numpy()  # [N, 20]
    log(f"  Per-sentence mu shape: {sent_mu_np.shape}")

    user_to_mus: dict[str, list[np.ndarray]] = defaultdict(list)
    for idx, row in enumerate(dataset["sentence_rows"]):
        user_to_mus[row["user_id"]].append(sent_mu_np[idx])
    return {uid: np.stack(vs, axis=0) for uid, vs in user_to_mus.items()}


def _compute_per_user_variance(user_mus: dict[str, np.ndarray]) -> dict[str, float]:
    """trace(cov) = Σ_i Var(mu_i) for each user. Returned as plain float dict."""
    out: dict[str, float] = {}
    for uid, mus in user_mus.items():
        if mus.shape[0] < 2:
            out[uid] = 0.0
            continue
        out[uid] = float(np.sum(np.var(mus, axis=0, ddof=0)))
    return out


def _plot_distribution(per_user_var: dict[str, float], category: str, label: str) -> tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = OUTPUT_BASE / category
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "align" if "align" in label.lower() else "gmm"

    vals = np.asarray(list(per_user_var.values()), dtype=np.float64)
    log_p99 = float(np.quantile(vals, 0.99))
    log_p50 = float(np.quantile(vals, 0.50))
    log_p10 = float(np.quantile(vals, 0.10))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.hist(vals, bins=80, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Per-user latent trace variance (Σ_i Var(mu_i))")
    ax.set_ylabel("# users (log scale)")
    ax.set_title(
        f"Per-User Latent Variance Distribution — {category} ({label})\n"
        f"n={len(vals)} users | median={log_p50:.3f}, p10={log_p10:.3f}, p99={log_p99:.3f}\n"
        f"min={vals.min():.3f}, max={vals.max():.3f}, std={vals.std():.3f}"
    )
    ax.axvline(log_p50, color="red", linestyle="--", alpha=0.7, label=f"median = {log_p50:.3f}")
    ax.axvline(log_p99, color="orange", linestyle="--", alpha=0.7, label=f"p99 = {log_p99:.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    sorted_vals = np.sort(vals)
    ax.plot(np.arange(len(sorted_vals)), sorted_vals, color="steelblue", linewidth=1.0)
    ax.set_xlabel("Users (sorted by latent variance, ascending)")
    ax.set_ylabel("Per-user latent trace variance")
    ax.set_yscale("log")
    ax.set_title(
        f"Sorted Per-User Latent Variance — {category} ({label})\n"
        f"shows the heavy-tail users (style variability outliers)"
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png = out_dir / f"user_latent_variance_{suffix}.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {out_png}")
    return out_png, out_png


def run_for_category(category: str, tag: str, label: str) -> dict:
    log(f"\n>>> {category} ({label})")
    user_mus = _load_encoder_and_infer_per_sentence_mu(category, tag)
    log(f"  Users with mu: {len(user_mus)}")

    per_user_var = _compute_per_user_variance(user_mus)
    vals = np.asarray(list(per_user_var.values()), dtype=np.float64)

    out_dir = OUTPUT_BASE / category
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "align" if "align" in label.lower() else "gmm"
    stats = {
        "category": category,
        "tag": tag,
        "label": label,
        "n_users": int(len(per_user_var)),
        "n_users_ge2_sents": int(np.sum([mus.shape[0] >= 2 for mus in user_mus.values()])),
        "stats": {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(np.median(vals)),
            "p10": float(np.quantile(vals, 0.10)),
            "p25": float(np.quantile(vals, 0.25)),
            "p75": float(np.quantile(vals, 0.75)),
            "p90": float(np.quantile(vals, 0.90)),
            "p99": float(np.quantile(vals, 0.99)),
            "p99_median_ratio": float(np.quantile(vals, 0.99) / (np.median(vals) + 1e-9)),
        },
    }
    out_json = out_dir / f"user_latent_variance_{suffix}_stats.json"
    out_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  Stats: median={stats['stats']['median']:.3f}, p99={stats['stats']['p99']:.3f}, "
        f"p99/median={stats['stats']['p99_median_ratio']:.1f}x")

    _plot_distribution(per_user_var, category, label)

    # Also dump raw per-user values for downstream use
    raw_json = out_dir / f"user_latent_variance_{suffix}_raw.json"
    raw_json.write_text(json.dumps(per_user_var, ensure_ascii=False, indent=0), encoding="utf-8")
    log(f"  Raw: {raw_json}")
    return stats


def main() -> None:
    log("=" * 60)
    log("Per-user latent variance distribution (VADES-style author variability)")
    log("=" * 60)

    all_stats: list[dict] = []
    for cat in CATEGORIES:
        has_only = (FEATURE_BASE / cat / f"{GMM_ONLY_TAG}_user_profiles.jsonl").exists()
        has_align = (FEATURE_BASE / cat / f"{GMM_ALIGN_TAG}_user_profiles.jsonl").exists()

        primary_tag = GMM_ALIGN_TAG if has_align else (GMM_ONLY_TAG if has_only else None)
        primary_label = "gmm+align" if has_align else "gmm-only"
        if primary_tag is None:
            log(f"{cat}: no data, skipping")
            continue

        try:
            all_stats.append(run_for_category(cat, primary_tag, primary_label))
        except Exception as e:
            log(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    summary_file = OUTPUT_BASE / "user_latent_variance_summary.json"
    summary_file.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\nSummary: {summary_file}")

    log("\n" + "=" * 90)
    log("=== Per-User Latent Variance Summary ===")
    log("=" * 90)
    log(f"{'Category':<25} | {'N users':>8} | {'median':>8} | {'p99':>8} | {'p99/median':>11} | {'max/median':>11}")
    log("-" * 90)
    for s in all_stats:
        st = s["stats"]
        log(f"{s['category']:<25} | {s['n_users']:>8d} | {st['median']:>8.3f} | {st['p99']:>8.3f} | "
            f"{st['p99_median_ratio']:>10.1f}x | {st['max']/max(st['median'],1e-9):>10.1f}x")
    log("=" * 90)


if __name__ == "__main__":
    main()
