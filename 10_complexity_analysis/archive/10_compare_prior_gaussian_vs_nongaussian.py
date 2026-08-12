#!/usr/bin/env python3
"""Compare Gaussian (2-GMM) vs 5 non-Gaussian user priors on stability/interpretability/complexity.

Groups compared (6 total on Baby_Products):
  1. 2-GMM (Gaussian family)              — gmm tag, diagonal_gmm mode
  2. Student-t (single, heavy-tailed)     — student_t tag, diagonal_student_t mode
  3. Laplace (single, sharp peak)         — laplace tag, diagonal_laplace mode
  4. Logistic (single, smooth tails)      — logistic tag, diagonal_logistic mode
  5. Student-t GMM K=2 (multi-modal heavy) — student_t_gmm tag, diagonal_student_t_gmm mode
  6. Student-t strict VIB                 — student_t_strict tag, diagonal_student_t mode,
                                            Student-t encoder (one-MC-sample log p)

For each group we compute 7 metrics:
  - 5 stability: accept_rate, top1_recomputed_agreement, per_user_score_std,
                 threshold_margin, training_final_loss
  - 1 interpretability: diagonal_mean_correlation (latent-feature alignment)
  - 1 complexity: num_params (per user)

Output:
  - result/.../interpretability_paper/prior_gaussian_vs_nongaussian.json
  - result/.../interpretability_paper/prior_gaussian_vs_nongaussian.png
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = os.environ.get("CATEGORY_OVERRIDE", "Baby_Products")
FEATURE_BASE = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / CATEGORY
OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / "interpretability_paper" / CATEGORY
OUTPUT_JSON = OUTPUT_DIR / "prior_gaussian_vs_nongaussian.json"
OUTPUT_PNG = OUTPUT_DIR / "prior_gaussian_vs_nongaussian.png"

GROUPS = [
    {"key": "gmm_2", "label": "2-GMM (Gaussian)", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_gmm", "cov_mode": "diagonal_gmm", "family": "Gaussian"},
    {"key": "student_t", "label": "Student-t", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t", "cov_mode": "diagonal_student_t", "family": "Non-Gaussian"},
    {"key": "laplace", "label": "Laplace", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_laplace", "cov_mode": "diagonal_laplace", "family": "Non-Gaussian"},
    {"key": "logistic", "label": "Logistic", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_logistic", "cov_mode": "diagonal_logistic", "family": "Non-Gaussian"},
    {"key": "student_t_gmm", "label": "Student-t GMM (K=2)", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t_gmm", "cov_mode": "diagonal_student_t_gmm", "family": "Non-Gaussian"},
    {"key": "student_t_strict", "label": "Student-t strict VIB", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t_strict", "cov_mode": "diagonal_student_t", "encoder_dist": "student_t", "family": "Non-Gaussian"},
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"count": 0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def _params_per_user(cov_mode: str) -> int:
    """Number of learnable params per user for each mode (latent_dim=20, K=2 for GMM)."""
    D = 20
    K = 2
    if cov_mode == "diagonal_gmm":
        return (D + D) * K + K  # 82
    if cov_mode == "diagonal_student_t":
        return D + D + 1  # 41
    if cov_mode == "diagonal_laplace":
        return D + D  # 40
    if cov_mode == "diagonal_logistic":
        return D + D  # 40
    if cov_mode == "diagonal_student_t_gmm":
        return (D + D + 1) * K + K  # 84
    if cov_mode == "diagonal":
        return D + D  # 40
    if cov_mode == "full":
        return D + D * (D + 1) // 2  # 230
    return 0


def _accept_rate(sel: list[dict], rej: list[dict]) -> float:
    total = len(sel) + len(rej)
    if total == 0:
        raise ValueError("no records to compute accept rate")
    return len(sel) / total


def _final_loss(tag: str) -> float | None:
    path = FEATURE_BASE / f"{tag}_epoch_details.jsonl"
    if not path.exists():
        return None
    rows = _load_jsonl(path)
    if not rows:
        return None
    return float(rows[-1].get("total_loss_mean"))


def _load_group(group: dict) -> dict | None:
    tag = group["tag"]
    sel_path = FEATURE_BASE / f"{tag}_selected_query_records.jsonl"
    rej_path = FEATURE_BASE / f"{tag}_rejected_query_records.jsonl"
    profile_path = FEATURE_BASE / f"{tag}_user_profiles.jsonl"
    if not sel_path.exists() or not rej_path.exists() or not profile_path.exists():
        return None
    return {
        "selected": _load_jsonl(sel_path),
        "rejected": _load_jsonl(rej_path),
        "profile": _load_jsonl(profile_path),
        "final_loss": _final_loss(tag),
    }


def _per_user_all_candidate_scores(
    profile_rows: list[dict],
    tag: str,
    cov_mode: str,
    candidate_query_file: Path,
) -> dict[str, list[tuple[str, float]]]:
    """Re-rank every (user, candidate) pair using the saved model and return scores."""
    import torch
    _sys_path = str(Path(__file__).resolve().parent / "common")
    if _sys_path not in sys.path:
        sys.path.insert(0, _sys_path)
    import train_vades_lite_sentence_latent_threshold as M  # noqa: E402

    M.REPO_ROOT = REPO_ROOT
    M.CATEGORY = CATEGORY
    M.OUTPUT_TAG = tag
    M.DEVICE = M.infer_device()
    M.COVARIANCE_MODE = cov_mode
    # Group-specific encoder_dist: only t-t strict uses Student-t encoder.
    group_encoder_dist = None
    for _g in GROUPS:
        if _g["tag"] == tag and "encoder_dist" in _g:
            group_encoder_dist = _g["encoder_dist"]
            break
    M.ENCODER_DIST = group_encoder_dist or "gaussian"
    if cov_mode in {"diagonal_gmm", "diagonal_student_t_gmm"}:
        M.GMM_COMPONENTS = 2

    cov_tag = f"_{cov_mode}" if cov_mode != "diagonal" else ""
    encoder_path = FEATURE_BASE / f"vades_encoder_{tag}{cov_tag}.pt"
    user_table_path = FEATURE_BASE / f"vades_user_table_{tag}{cov_tag}.pt"
    encoder_ckpt = torch.load(encoder_path, map_location=M.DEVICE, weights_only=False)
    encoder_state = {k.replace("_orig_mod.", ""): v for k, v in encoder_ckpt["model_state_dict"].items()}
    user_table_ckpt = torch.load(user_table_path, map_location=M.DEVICE, weights_only=False)
    user_table_state = {k.replace("_orig_mod.", ""): v for k, v in user_table_ckpt["model_state_dict"].items()}

    if cov_mode == "full":
        encoder = M.SentenceEncoderFull(
            input_dim=encoder_ckpt["input_dim"],
            hidden_dim=encoder_ckpt["hidden_dim"],
            latent_dim=encoder_ckpt["latent_dim"],
        ).to(M.DEVICE)
    elif group_encoder_dist == "student_t":
        encoder = M.SentenceEncoderStudentT(
            input_dim=encoder_ckpt["input_dim"],
            hidden_dim=encoder_ckpt["hidden_dim"],
            latent_dim=encoder_ckpt["latent_dim"],
        ).to(M.DEVICE)
    else:
        encoder = M.SentenceEncoder(
            input_dim=encoder_ckpt["input_dim"],
            hidden_dim=encoder_ckpt["hidden_dim"],
            latent_dim=encoder_ckpt["latent_dim"],
        ).to(M.DEVICE)
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    if cov_mode == "diagonal_gmm":
        user_table = M.UserDistributionTableGMM(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
            num_components=user_table_ckpt.get("gmm_components", 2),
        ).to(M.DEVICE)
    elif cov_mode == "diagonal_student_t":
        user_table = M.UserDistributionTableStudentT(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(M.DEVICE)
    elif cov_mode == "diagonal_laplace":
        user_table = M.UserDistributionTableLaplace(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(M.DEVICE)
    elif cov_mode == "diagonal_logistic":
        user_table = M.UserDistributionTableLogistic(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(M.DEVICE)
    elif cov_mode == "diagonal_student_t_gmm":
        user_table = M.UserDistributionTableStudentTGMM(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
            num_components=user_table_ckpt.get("gmm_components", 2),
        ).to(M.DEVICE)
    elif cov_mode == "full":
        user_table = M.UserDistributionTableFull(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(M.DEVICE)
    else:
        user_table = M.UserDistributionTable(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(M.DEVICE)
    user_table.load_state_dict(user_table_state)
    user_table.eval()

    with candidate_query_file.open("r", encoding="utf-8") as f:
        cand_rows = [json.loads(line) for line in f if line.strip()]
    user_to_index = {row["user_id"]: i for i, row in enumerate(profile_rows)}
    feature_names = list(cand_rows[0]["features"].keys())

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in cand_rows:
        grouped[row["user_id"]].append(row)

    user_query_scores: dict[str, list[tuple[str, float]]] = {}
    with torch.no_grad():
        for user_id, cands in grouped.items():
            if user_id not in user_to_index:
                continue
            user_idx = user_to_index[user_id]
            scores: list[tuple[str, float]] = []
            for cand in cands:
                feat = np.asarray([float(cand["features"][n]) for n in feature_names], dtype=np.float64).reshape(1, -1)
                feat_t = torch.tensor(feat, dtype=torch.float32, device=M.DEVICE)
                # Encoder output: 4-tuple for Student-t (mu, logvar, df, recon),
                # 3-tuple for Gaussian (mu, logvar, recon).
                enc_out = encoder(feat_t)
                if len(enc_out) == 4:
                    mu, logvar, df, _ = enc_out
                    # Reparam sample to mirror training-time scoring.
                    z = M.reparameterize_student_t(mu, logvar, df)
                else:
                    mu, logvar, _ = enc_out
                    df = None
                    z = None
                disp = logvar
                if cov_mode == "diagonal_gmm":
                    u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                    u_lv = user_table.user_logvar[user_idx].unsqueeze(0)
                    u_mix = user_table.mix_logits[user_idx].unsqueeze(0)
                    log_p = M.gmm_log_likelihood(mu, disp, u_mu, u_lv, u_mix).item()
                    score = -log_p
                elif cov_mode == "diagonal_student_t":
                    u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                    u_ls = user_table.user_log_scale[user_idx].unsqueeze(0)
                    u_df = user_table.df_all()[user_idx].unsqueeze(0)
                    if df is not None and z is not None:
                        # Strict t-t VIB: score reparam z under user's Student-t.
                        log_p = M.student_t_log_p_matrix(z, u_mu, u_ls, u_df).item()
                    else:
                        # Gaussian encoder: point estimate at μ_q.
                        log_p = M.student_t_log_likelihood(mu, disp, u_mu, u_ls, u_df).item()
                    score = -log_p
                elif cov_mode == "diagonal_laplace":
                    u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                    u_lb = user_table.user_log_b[user_idx].unsqueeze(0)
                    log_p = M.laplace_log_likelihood(mu, disp, u_mu, u_lb).item()
                    score = -log_p
                elif cov_mode == "diagonal_logistic":
                    u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                    u_ls = user_table.user_log_s[user_idx].unsqueeze(0)
                    log_p = M.logistic_log_likelihood(mu, disp, u_mu, u_ls).item()
                    score = -log_p
                elif cov_mode == "diagonal_student_t_gmm":
                    u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                    u_ls = user_table.user_log_scale[user_idx].unsqueeze(0)
                    u_df = user_table.df_all()[user_idx].unsqueeze(0)
                    u_mix = user_table.mix_logits[user_idx].unsqueeze(0)
                    log_p = M.student_t_gmm_log_likelihood(mu, disp, u_mu, u_ls, u_df, u_mix).item()
                    score = -log_p
                else:
                    score = M.diagonal_gaussian_kl(
                        mu, disp,
                        user_table.user_mu.weight[user_idx].unsqueeze(0),
                        user_table.user_logvar.weight[user_idx].unsqueeze(0),
                    ).item()
                scores.append((cand["query"], float(score)))
            scores.sort(key=lambda x: x[1])
            user_query_scores[user_id] = scores
    return user_query_scores


def _compute_interpretability(group: dict) -> dict:
    """diagonal_mean_correlation: 20x20 Pearson between user-mu and per-user feature mean.

    For each user, the user_mu (single dist) or the first component's mu (mixture dist)
    is the 20-dim latent. The per-user feature mean is the average of the user's
    per-sentence 20-dim features. Diagonal of the 20x20 Pearson matrix = alignment.
    """
    tag = group["tag"]
    cov_mode = group["cov_mode"]
    profile_path = FEATURE_BASE / f"{tag}_user_profiles.jsonl"
    if not profile_path.exists():
        return {"diagonal_mean_correlation": None}
    profile_rows = _load_jsonl(profile_path)
    if not profile_rows:
        return {"diagonal_mean_correlation": None}

    # Get user-level latent mu [U, 20] — match the original
    # 12_interpretability_3_experiments.py convention: use the first component's mu
    # (component_mus[0]) for mixture modes. For single-distribution modes, use user_mu
    # directly. The first component carries the dominant user style signal even though
    # the model also uses the second component (this is consistent with the VADES paper
    # Figure 3 design where they show one component's latent for clarity).
    user_latent = []
    for row in profile_rows:
        if "user_mu" in row:
            user_latent.append(row["user_mu"])
        elif "component_mus" in row:
            user_latent.append(row["component_mus"][0])
        else:
            return {"diagonal_mean_correlation": None}
    user_latent = np.asarray(user_latent, dtype=np.float64)  # [U, 20]
    user_ids = [row["user_id"] for row in profile_rows]

    # Per-user feature mean from sentences
    sentence_path = FEATURE_BASE / f"{tag}_sentences.jsonl"
    if not sentence_path.exists():
        return {"diagonal_mean_correlation": None}
    sentence_rows = _load_jsonl(sentence_path)
    if not sentence_rows:
        return {"diagonal_mean_correlation": None}
    feature_names = list(sentence_rows[0]["features"].keys())
    user_feat_acc: dict[str, np.ndarray] = {}
    user_feat_cnt: dict[str, int] = defaultdict(int)
    for row in sentence_rows:
        uid = row["user_id"]
        f = np.asarray([row["features"][n] for n in feature_names], dtype=np.float64)
        if uid not in user_feat_acc:
            user_feat_acc[uid] = np.zeros(20, dtype=np.float64)
        user_feat_acc[uid] += f
        user_feat_cnt[uid] += 1
    user_feat_mean = {uid: user_feat_acc[uid] / max(user_feat_cnt[uid], 1) for uid in user_feat_acc}

    # Build aligned [U, 20] matrices on common users
    common = [u for u in user_ids if u in user_feat_mean]
    if len(common) < 10:
        return {"diagonal_mean_correlation": None}
    latent_mat = np.stack([user_latent[user_ids.index(u)] for u in common], axis=0)
    feat_mat = np.stack([user_feat_mean[u] for u in common], axis=0)

    corr = np.zeros((20, 20), dtype=np.float64)
    for i in range(20):
        for j in range(20):
            x = latent_mat[:, i]
            y = feat_mat[:, j]
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                corr[i, j] = 0.0
            else:
                corr[i, j] = float(np.corrcoef(x, y)[0, 1])
    diag_mean = float(np.mean(np.diag(corr)))
    off_diag = corr - np.diag(np.diag(corr))
    offdiag_mean = float(np.mean(off_diag))
    return {
        "diagonal_mean_correlation": diag_mean,
        "offdiag_mean_correlation": offdiag_mean,
        "diag_offdiag_ratio": diag_mean / abs(offdiag_mean) if abs(offdiag_mean) > 1e-9 else None,
    }


def compute_for_group(group: dict, candidate_query_file: Path) -> tuple[dict, dict]:
    log(f"[{group['key']}] loading run records...")
    run = _load_group(group)
    if run is None:
        raise FileNotFoundError(f"missing run files for {group['tag']}")

    metrics: dict = {
        "category": CATEGORY,
        "label": group["label"],
        "family": group["family"],
        "tag": group["tag"],
        "cov_mode": group["cov_mode"],
    }

    metrics["num_params_per_user"] = _params_per_user(group["cov_mode"])
    metrics["accept_rate"] = _accept_rate(run["selected"], run["rejected"])
    metrics["selected_range_score"] = _summarize([r["range_score"] for r in run["selected"]])
    metrics["rejected_range_score"] = _summarize([r["range_score"] for r in run["rejected"]])
    metrics["training_final_loss"] = run["final_loss"]

    log(f"[{group['key']}] recomputing per-user all-10-candidate scores...")
    user_scores = _per_user_all_candidate_scores(
        run["profile"], group["tag"], group["cov_mode"], candidate_query_file,
    )
    user_stds = [float(np.std([s for _, s in v])) for v in user_scores.values() if v]
    metrics["per_user_score_std"] = _summarize(user_stds)

    log(f"[{group['key']}] computing interpretability (diagonal correlation)...")
    metrics["interpretability"] = _compute_interpretability(group)

    return metrics, user_scores


def main() -> None:
    log("=" * 80)
    log("6 组分布对比：1 Gaussian (2-GMM) + 5 non-Gaussian (Student-t / Laplace / Logistic / Student-t GMM / Student-t strict VIB)")
    log("=" * 80)

    candidate_query_file = FEATURE_BASE / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
    if not candidate_query_file.exists():
        raise FileNotFoundError(f"missing candidate query file: {candidate_query_file}")

    all_metrics: dict = {"category": CATEGORY, "groups": {}}
    all_user_scores: dict = {}
    for group in GROUPS:
        try:
            metrics, user_scores = compute_for_group(group, candidate_query_file)
            all_metrics["groups"][group["key"]] = metrics
            all_user_scores[group["key"]] = user_scores
            log(f"[{group['key']}] ✓ accept_rate={metrics['accept_rate']:.4f} per_user_std_mean={metrics['per_user_score_std'].get('mean', 'N/A')}")
        except FileNotFoundError as e:
            log(f"[{group['key']}] SKIPPED: {e}")
            all_metrics["groups"][group["key"]] = {"error": str(e), "label": group["label"]}

    # top-1 agreement vs 2-GMM (the Gaussian reference)
    if "gmm_2" in all_user_scores:
        ref_key = "gmm_2"
        ref_scores = all_user_scores[ref_key]
        for key, scores in all_user_scores.items():
            if key == ref_key:
                continue
            common = set(scores) & set(ref_scores)
            if not common:
                all_metrics["groups"][key]["top1_agreement_vs_2gmm"] = None
                continue
            agrees = sum(1 for u in common if scores[u][0][0] == ref_scores[u][0][0])
            all_metrics["groups"][key]["top1_agreement_vs_2gmm"] = {
                "ref": ref_key,
                "overlap_users": int(len(common)),
                "agree": int(agrees),
                "agreement_rate": float(agrees / len(common)),
            }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n结果 JSON: {OUTPUT_JSON}")

    # Console summary table
    log("\n" + "=" * 100)
    log("=== 6 组分布对比汇总 (Baby_Products) ===")
    log("=" * 100)
    header = f"{'组':<25} | {'family':<11} | {'params/u':>8} | {'accept':>7} | {'top1 vs 2G':>11} | {'std mean':>9} | {'diag corr':>9} | {'final loss':>10}"
    log(header)
    log("-" * len(header))
    for g in GROUPS:
        m = all_metrics["groups"].get(g["key"], {})
        if "error" in m:
            log(f"{g['label']:<20} | {g['family']:<11} | N/A")
            continue
        params = m.get("num_params_per_user", "N/A")
        ar = m.get("accept_rate", "N/A")
        top1 = m.get("top1_agreement_vs_2gmm")
        top1_str = f"{top1['agreement_rate']:.3f}" if top1 else "—"
        std_mean = m.get("per_user_score_std", {}).get("mean")
        std_str = f"{std_mean:.4f}" if std_mean is not None else "N/A"
        diag = (m.get("interpretability") or {}).get("diagonal_mean_correlation")
        diag_str = f"{diag:.3f}" if diag is not None else "N/A"
        fl = m.get("training_final_loss")
        fl_str = f"{fl:.4f}" if fl is not None else "N/A"
        log(f"{g['label']:<25} | {g['family']:<11} | {params:>8} | {ar:>7.4f} | {top1_str:>11} | {std_str:>9} | {diag_str:>9} | {fl_str:>10}")
    log("=" * 100)

    # Plot bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [g["label"] for g in GROUPS if "error" not in all_metrics["groups"].get(g["key"], {})]
        families = [g["family"] for g in GROUPS if "error" not in all_metrics["groups"].get(g["key"], {})]
        accept = [all_metrics["groups"][g["key"]]["accept_rate"] for g in GROUPS if "error" not in all_metrics["groups"].get(g["key"], {})]
        std_means = [all_metrics["groups"][g["key"]]["per_user_score_std"].get("mean") for g in GROUPS if "error" not in all_metrics["groups"].get(g["key"], {})]
        final_losses = [all_metrics["groups"][g["key"]].get("training_final_loss") for g in GROUPS if "error" not in all_metrics["groups"].get(g["key"], {})]
        top1_agrees = [all_metrics["groups"][g["key"]].get("top1_agreement_vs_2gmm", {}).get("agreement_rate") if g["key"] != "gmm_2" else 1.0 for g in GROUPS if "error" not in all_metrics["groups"].get(g["key"], {})]
        params = [all_metrics["groups"][g["key"]]["num_params_per_user"] for g in GROUPS if "error" not in all_metrics["groups"].get(g["key"], {})]

        fig, axes = plt.subplots(1, 5, figsize=(28, 5))
        colors = ["#3a7ca5" if f == "Gaussian" else "#d97742" for f in families]
        for ax, vals, title in zip(
            axes,
            [accept, std_means, final_losses, top1_agrees, params],
            ["accept_rate", "per_user_score_std (mean)", "training_final_loss", "top-1 agreement vs 2-GMM", "# params per user"],
        ):
            ax.bar(range(len(labels)), vals, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle(f"Gaussian (2-GMM) vs 5 non-Gaussian user priors — {CATEGORY}", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
        log(f"图: {OUTPUT_PNG}")
    except ImportError:
        log("[warn] matplotlib 不可用，跳过 PNG 生成")


if __name__ == "__main__":
    main()
