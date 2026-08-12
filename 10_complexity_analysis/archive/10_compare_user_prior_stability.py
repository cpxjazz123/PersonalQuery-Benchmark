#!/usr/bin/env python3
"""Compare single-Gaussian vs 2-GMM user-prior VADES runs on stability metrics.

Reads two pre-computed run directories under
`result/personal_query/10_complexity_analysis_clause_features/Baby_Products/`:
  - diagonal run:  OUTPUT_TAG with VADES_COVARIANCE_MODE=diagonal
  - 2-GMM run:     OUTPUT_TAG with VADES_COVARIANCE_MODE=diagonal_gmm

Emits:
  - JSON with 6 stability-focused metrics
  - Console table comparing the two modes

Stability focus (per user request: 60% Normal AIC doesn't matter; we want
to know if the GMM prior gives more stable / discriminative ranking):
  1. accept_rate                              (per mode + Δ)
  2. selected / rejected range_score         (mean / std / q50 / q90)
  3. top1_recomputed_agreement               (diagonal vs gmm on best of 10)
  4. threshold_margin                        (per-accepted gap to holdout threshold)
  5. per_user_score_std                      (GMM should give more spread → more
                                              discriminative; or less if GMM
                                              collapses the prior)
  6. training_final_loss                     (reference)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORIES = ["Baby_Products", "Pet_Supplies", "Grocery_and_Gourmet_Food"]
DIAG_TAG = "vades_lite_sentence_user_distribution_train10_holdout10"
GMM_TAG = "vades_lite_sentence_user_distribution_train10_holdout10_gmm"
AGGREGATE_OUTPUT_FILE = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / "ablation_single_vs_2gmm_all_categories.json"


def _feature_dir_for(category: str) -> Path:
    return REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / category


def _candidate_file_for(category: str) -> Path:
    return _feature_dir_for(category) / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _summarize_scores(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("no scores to summarize")
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q50": float(np.quantile(arr, 0.5)),
        "q90": float(np.quantile(arr, 0.9)),
    }


def _accept_rate(sel: list[dict], rej: list[dict]) -> float:
    total = len(sel) + len(rej)
    if total == 0:
        raise ValueError("no records to compute accept rate")
    return len(sel) / total


def _final_training_loss(tag: str, feature_dir: Path) -> float | None:
    """Last epoch's total_loss_mean from {tag}_epoch_details.jsonl."""
    path = feature_dir / f"{tag}_epoch_details.jsonl"
    if not path.exists():
        return None
    rows = _load_jsonl(path)
    if not rows:
        return None
    return float(rows[-1].get("total_loss_mean"))


def _load_runs(category: str) -> tuple[dict, dict]:
    """Load the saved records of both runs and return them in two dicts keyed by mode."""
    feature_dir = _feature_dir_for(category)
    runs: dict[str, dict] = {}
    for mode_name, tag in (("diagonal", DIAG_TAG), ("gmm", GMM_TAG)):
        sel = _load_jsonl(feature_dir / f"{tag}_selected_query_records.jsonl")
        rej = _load_jsonl(feature_dir / f"{tag}_rejected_query_records.jsonl")
        profile = _load_jsonl(feature_dir / f"{tag}_user_profiles.jsonl")
        try:
            summary = json.loads((feature_dir / f"{tag}_summary.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            summary = {}
        runs[mode_name] = {
            "tag": tag,
            "selected": sel,
            "rejected": rej,
            "profile": profile,
            "summary": summary,
            "final_loss": _final_training_loss(tag, feature_dir),
        }
        log(f"[{mode_name}] selected={len(sel)}, rejected={len(rej)}, users={len(profile)}")
    return runs["diagonal"], runs["gmm"]


def _per_user_all_candidate_scores(
    profile_rows: list[dict],
    user_table_path: Path,
    candidate_query_file: Path,
    cov_mode: str,
    output_tag: str,
    category: str,
) -> dict[str, list[tuple[str, float]]]:
    """Re-rank every (user, candidate) pair using the saved model and return scores.

    Returns {user_id: [(query_text, range_score), ...]} — 10 entries per user,
    sorted by ascending range_score (best first).
    """
    import torch
    _sys_path = str(Path(__file__).resolve().parent / "common")
    if _sys_path not in sys.path:
        sys.path.insert(0, _sys_path)
    import train_vades_lite_sentence_latent_threshold as train_module

    train_module.REPO_ROOT = REPO_ROOT
    train_module.CATEGORY = category
    train_module.OUTPUT_TAG = output_tag
    train_module.DEVICE = train_module.infer_device()
    train_module.COVARIANCE_MODE = cov_mode
    if cov_mode == "diagonal_gmm":
        train_module.GMM_COMPONENTS = 2  # default

    cov_tag = f"_{cov_mode}" if cov_mode != "diagonal" else ""
    encoder_path = user_table_path.with_name(f"vades_encoder_{output_tag}{cov_tag}.pt")
    encoder_ckpt = torch.load(encoder_path, map_location=train_module.DEVICE, weights_only=False)
    encoder_state = {k.replace("_orig_mod.", ""): v for k, v in encoder_ckpt["model_state_dict"].items()}
    user_table_ckpt = torch.load(user_table_path, map_location=train_module.DEVICE, weights_only=False)
    user_table_state = {k.replace("_orig_mod.", ""): v for k, v in user_table_ckpt["model_state_dict"].items()}

    if cov_mode == "full":
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
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    if cov_mode == "diagonal_gmm":
        user_table = train_module.UserDistributionTableGMM(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
            num_components=user_table_ckpt.get("gmm_components", train_module.GMM_COMPONENTS),
        ).to(train_module.DEVICE)
    elif cov_mode == "full":
        user_table = train_module.UserDistributionTableFull(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    else:
        user_table = train_module.UserDistributionTable(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    user_table.load_state_dict(user_table_state)
    user_table.eval()

    with open(candidate_query_file, "r", encoding="utf-8") as f:
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
                feat_t = torch.tensor(feat, dtype=torch.float32, device=train_module.DEVICE)
                mu, disp, _ = encoder(feat_t)
                if cov_mode == "diagonal_gmm":
                    user_mu = user_table.user_mu[user_idx].unsqueeze(0)
                    user_logvar = user_table.user_logvar[user_idx].unsqueeze(0)
                    user_mix = user_table.mix_logits[user_idx].unsqueeze(0)
                    log_p_xu = train_module.gmm_log_likelihood(mu, disp, user_mu, user_logvar, user_mix).item()
                    score = -log_p_xu
                elif cov_mode == "full":
                    user_L = user_table.get_L()[user_idx].unsqueeze(0)
                    score = train_module.multivariate_gaussian_kl(mu, disp, user_table.user_mu.weight[user_idx].unsqueeze(0), user_L).item()
                else:
                    user_logvar = user_table.user_logvar.weight[user_idx].unsqueeze(0)
                    score = train_module.diagonal_gaussian_kl(mu, disp, user_table.user_mu.weight[user_idx].unsqueeze(0), user_logvar).item()
                scores.append((cand["query"], float(score)))
            scores.sort(key=lambda x: x[1])
            user_query_scores[user_id] = scores
    return user_query_scores


def _per_user_acceptance_threshold(
    profile_rows: list[dict],
    user_table_path: Path,
    dataset_dir: Path,
    output_tag: str,
    cov_mode: str,
    abs_threshold_quantile: float,
    category: str,
) -> dict[str, float]:
    """Recompute the per-user holdout-based abs threshold for the GMM case.

    We re-encode all sentences for the user, compute -log p(sent|user) on the
    holdout subset, and take the configured quantile. Used so the GMM
    threshold margin metric is consistent with the scoring function actually
    used at inference time.
    """
    import torch
    _sys_path = str(Path(__file__).resolve().parent / "common")
    if _sys_path not in sys.path:
        sys.path.insert(0, _sys_path)
    import train_vades_lite_sentence_latent_threshold as train_module

    train_module.REPO_ROOT = REPO_ROOT
    train_module.CATEGORY = category
    train_module.OUTPUT_TAG = output_tag
    train_module.DEVICE = train_module.infer_device()
    train_module.COVARIANCE_MODE = cov_mode

    sentence_file = dataset_dir / f"{output_tag}_sentences.jsonl"
    if not sentence_file.exists():
        raise FileNotFoundError(f"missing {sentence_file}")
    with sentence_file.open("r", encoding="utf-8") as f:
        sentence_rows = [json.loads(line) for line in f if line.strip()]

    cov_tag = f"_{cov_mode}" if cov_mode != "diagonal" else ""
    encoder_path = user_table_path.with_name(f"vades_encoder_{output_tag}{cov_tag}.pt")
    encoder_ckpt = torch.load(encoder_path, map_location=train_module.DEVICE, weights_only=False)
    encoder_state = {k.replace("_orig_mod.", ""): v for k, v in encoder_ckpt["model_state_dict"].items()}
    user_table_ckpt = torch.load(user_table_path, map_location=train_module.DEVICE, weights_only=False)
    user_table_state = {k.replace("_orig_mod.", ""): v for k, v in user_table_ckpt["model_state_dict"].items()}

    encoder = train_module.SentenceEncoder(
        input_dim=encoder_ckpt["input_dim"],
        hidden_dim=encoder_ckpt["hidden_dim"],
        latent_dim=encoder_ckpt["latent_dim"],
    ).to(train_module.DEVICE)
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    if cov_mode == "diagonal_gmm":
        user_table = train_module.UserDistributionTableGMM(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
            num_components=user_table_ckpt.get("gmm_components", 2),
        ).to(train_module.DEVICE)
    else:
        user_table = train_module.UserDistributionTable(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    user_table.load_state_dict(user_table_state)
    user_table.eval()

    # Use the first MAX_HOLDOUT_SENTENCES_PER_USER (=5) per user as holdout.
    train_module.MAX_HOLDOUT_SENTENCES_PER_USER = 5
    train_n = train_module.TRAIN_SENTENCES_PER_USER
    holdout_rows: dict[str, list[dict]] = defaultdict(list)
    for row in sentence_rows:
        if len(holdout_rows[row["user_id"]]) < train_module.MAX_HOLDOUT_SENTENCES_PER_USER:
            holdout_rows[row["user_id"]].append(row)
    # crude: just take rows beyond train_n per user from sentence_rows order
    holdout_rows = defaultdict(list)
    per_user_seen: dict[str, int] = defaultdict(int)
    for row in sentence_rows:
        uid = row["user_id"]
        if per_user_seen[uid] >= train_n and len(holdout_rows[uid]) < train_module.MAX_HOLDOUT_SENTENCES_PER_USER:
            holdout_rows[uid].append(row)
        per_user_seen[uid] += 1

    feature_names = list(sentence_rows[0]["features"].keys())
    user_to_index = {row["user_id"]: i for i, row in enumerate(profile_rows)}

    thresholds: dict[str, float] = {}
    with torch.no_grad():
        for user_id, rows in holdout_rows.items():
            if user_id not in user_to_index:
                continue
            user_idx = user_to_index[user_id]
            scores: list[float] = []
            for row in rows:
                feat = np.asarray([float(row["features"][n]) for n in feature_names], dtype=np.float64).reshape(1, -1)
                feat_t = torch.tensor(feat, dtype=torch.float32, device=train_module.DEVICE)
                mu, disp, _ = encoder(feat_t)
                if cov_mode == "diagonal_gmm":
                    log_p_xu = train_module.gmm_log_likelihood(
                        mu, disp,
                        user_table.user_mu[user_idx].unsqueeze(0),
                        user_table.user_logvar[user_idx].unsqueeze(0),
                        user_table.mix_logits[user_idx].unsqueeze(0),
                    ).item()
                    scores.append(-log_p_xu)
                else:
                    user_logvar = user_table.user_logvar.weight[user_idx].unsqueeze(0)
                    scores.append(train_module.diagonal_gaussian_kl(
                        mu, disp,
                        user_table.user_mu.weight[user_idx].unsqueeze(0),
                        user_logvar,
                    ).item())
            thresholds[user_id] = float(np.quantile(np.asarray(scores, dtype=np.float64), abs_threshold_quantile))
    return thresholds


def compute_for_category(category: str) -> dict:
    """Compute all 6 stability metrics for a single category."""
    feature_dir = _feature_dir_for(category)
    candidate_query_file = _candidate_file_for(category)

    diag, gmm = _load_runs(category)

    metrics: dict = {"category": category}

    # 1) accept_rate
    metrics["accept_rate"] = {
        "diagonal": _accept_rate(diag["selected"], diag["rejected"]),
        "gmm": _accept_rate(gmm["selected"], gmm["rejected"]),
        "delta_gmm_minus_diag": None,
    }
    metrics["accept_rate"]["delta_gmm_minus_diag"] = (
        metrics["accept_rate"]["gmm"] - metrics["accept_rate"]["diagonal"]
    )

    # 2) selected / rejected range_score distribution
    metrics["selected_range_score"] = {
        "diagonal": _summarize_scores([row["range_score"] for row in diag["selected"]]),
        "gmm": _summarize_scores([row["range_score"] for row in gmm["selected"]]),
    }
    metrics["rejected_range_score"] = {
        "diagonal": _summarize_scores([row["range_score"] for row in diag["rejected"]]),
        "gmm": _summarize_scores([row["range_score"] for row in gmm["rejected"]]),
    }

    # 6) training final loss
    metrics["training_final_loss"] = {
        "diagonal": diag["final_loss"],
        "gmm": gmm["final_loss"],
    }

    # 3) top1 recomputed agreement — re-rank all 10 candidates with each model
    log(f"[{category}] 重新计算每用户 10 候选的 range_score (diagonal)...")
    diag_scores = _per_user_all_candidate_scores(
        diag["profile"], feature_dir / f"vades_user_table_{DIAG_TAG}.pt",
        candidate_query_file, "diagonal", DIAG_TAG, category,
    )
    log(f"[{category}] 重新计算每用户 10 候选的 range_score (gmm)...")
    gmm_scores = _per_user_all_candidate_scores(
        gmm["profile"], feature_dir / f"vades_user_table_{GMM_TAG}_diagonal_gmm.pt",
        candidate_query_file, "diagonal_gmm", GMM_TAG, category,
    )
    common = set(diag_scores) & set(gmm_scores)
    if not common:
        raise ValueError(f"[{category}] no overlapping users for top-1 recompute")
    agrees = sum(1 for u in common if diag_scores[u][0][0] == gmm_scores[u][0][0])
    metrics["top1_recomputed_agreement"] = {
        "overlap_users": int(len(common)),
        "agree": int(agrees),
        "agreement_rate": float(agrees / len(common)),
    }

    # 5) per-user score std — discriminative power within a user's 10 candidates
    diag_user_stds = [float(np.std([s for _, s in v])) for v in diag_scores.values()]
    gmm_user_stds = [float(np.std([s for _, s in v])) for v in gmm_scores.values()]
    metrics["per_user_score_std"] = {
        "diagonal": _summarize_scores(diag_user_stds),
        "gmm": _summarize_scores(gmm_user_stds),
        "delta_gmm_minus_diag_mean": float(np.mean(gmm_user_stds) - np.mean(diag_user_stds)),
    }

    # 4) threshold margin — for each accepted query, gap to its holdout threshold
    abs_thresh_q = float(diag["summary"].get("holdout_abs_threshold_quantile", 0.95)) if diag["summary"] else 0.95
    log(f"[{category}] 重新校准 diagonal 与 gmm 的 per-user abs threshold (q={abs_thresh_q})...")
    diag_thresholds = _per_user_acceptance_threshold(
        diag["profile"], feature_dir / f"vades_user_table_{DIAG_TAG}.pt",
        feature_dir, DIAG_TAG, "diagonal", abs_thresh_q, category,
    )
    gmm_thresholds = _per_user_acceptance_threshold(
        gmm["profile"], feature_dir / f"vades_user_table_{GMM_TAG}_diagonal_gmm.pt",
        feature_dir, GMM_TAG, "diagonal_gmm", abs_thresh_q, category,
    )

    def _margins(sel_records: list[dict], thresholds: dict[str, float]) -> list[float]:
        margins: list[float] = []
        for row in sel_records:
            uid = row["user_id"]
            if uid in thresholds:
                margins.append(float(thresholds[uid] - row["range_score"]))
        return margins

    diag_margins = _margins(diag["selected"], diag_thresholds)
    gmm_margins = _margins(gmm["selected"], gmm_thresholds)
    metrics["threshold_margin"] = {
        "diagonal": _summarize_scores(diag_margins) if diag_margins else {"count": 0},
        "gmm": _summarize_scores(gmm_margins) if gmm_margins else {"count": 0},
        "note": "threshold - range_score; positive = accepted with safety margin",
    }
    return metrics


def main() -> None:
    log("=" * 60)
    log("单高斯 vs 2-GMM 用户先验 稳定性对比 - 三域")
    log("=" * 60)

    all_metrics: dict = {"categories": {}}
    for cat in CATEGORIES:
        log(f"\n>>> 处理 category: {cat}")
        try:
            all_metrics["categories"][cat] = compute_for_category(cat)
        except FileNotFoundError as e:
            log(f"[{cat}] 跳过（缺少产物）: {e}")
            all_metrics["categories"][cat] = {"error": str(e)}

    AGGREGATE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    AGGREGATE_OUTPUT_FILE.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\n聚合结果: {AGGREGATE_OUTPUT_FILE}")

    # 控制台三域对比表
    log("\n" + "=" * 80)
    log("=== 三域稳定性指标对比 ===")
    log("=" * 80)
    header = f"{'指标':<35} | {'Baby':>10} | {'Pet':>10} | {'Grocery':>10}"
    log(header)
    log("-" * len(header))
    for metric_key, label in [
        ("accept_rate.delta_gmm_minus_diag", "accept_rate Δ"),
        ("top1_recomputed_agreement.agreement_rate", "top-1 agreement (gmm)"),
        ("training_final_loss.diagonal", "final loss (diag)"),
        ("training_final_loss.gmm", "final loss (gmm)"),
        ("per_user_score_std.delta_gmm_minus_diag_mean", "per-user score std Δ"),
        ("per_user_score_std.gmm.mean", "per-user std mean (gmm)"),
    ]:
        row = f"{label:<35}"
        for cat in CATEGORIES:
            m = all_metrics["categories"].get(cat, {})
            if "error" in m:
                row += f" | {'N/A':>10}"
                continue
            parts = metric_key.split(".")
            v = m
            for p in parts:
                v = v.get(p) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, float):
                row += f" | {v:>10.4f}"
            else:
                row += f" | {str(v):>10}"
        log(row)
    log("=" * 80)


if __name__ == "__main__":
    main()
