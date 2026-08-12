#!/usr/bin/env python3
"""Compare diagonal vs full-multivariate Gaussian query-selection results.

Reads two pre-computed run directories under
`result/personal_query/10_complexity_analysis_clause_features/<Cat>/`:
  - diagonal run: files with tag suffix `_ablation_diag`
  - full run:     files with tag suffix `_ablation_full`

Emits:
  - JSON with 5 metrics (accept_rate, score distribution, threshold margin,
    top-1 ranking agreement, per-user logvar-vs-trace stability)
  - Console table comparing the two modes
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
DIAG_TAG = "vades_lite_sentence_user_distribution_train10_holdout10_ablation_diag"
FULL_TAG = "vades_lite_sentence_user_distribution_train10_holdout10_ablation_full"
FEATURE_DIR = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / CATEGORY
OUTPUT_FILE = FEATURE_DIR / "ablation_diagonal_vs_full.json"


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


def _group_by_user(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(row["user_id"], []).append(row)
    return grouped


def _accept_rate(sel: list[dict], rej: list[dict]) -> float:
    total = len(sel) + len(rej)
    if total == 0:
        raise ValueError("no records to compute accept rate")
    return len(sel) / total


def _top1_agreement(sel_diag: list[dict], sel_full: list[dict], rej_diag: list[dict], rej_full: list[dict]) -> dict:
    """Compare top-1 per-user across two modes.

    For each user, the chosen candidate is the one with smallest range_score
    across their 10 candidates. We reconstruct per-user top-1 by combining
    selected + rejected (the one record per user stored after selection).
    """
    diag_by_user = {row["user_id"]: row for row in sel_diag}
    for row in rej_diag:
        diag_by_user.setdefault(row["user_id"], row)
    full_by_user = {row["user_id"]: row for row in sel_full}
    for row in rej_full:
        full_by_user.setdefault(row["user_id"], row)

    common = set(diag_by_user) & set(full_by_user)
    if not common:
        raise ValueError("no overlapping users for top-1 comparison")
    common_sorted = sorted(common)
    diag_pick = [diag_by_user[u]["query"] for u in common_sorted]
    full_pick = [full_by_user[u]["query"] for u in common_sorted]
    agreements = sum(1 for d, f in zip(diag_pick, full_pick) if d == f)
    return {
        "overlap_users": int(len(common)),
        "agree": int(agreements),
        "agreement_rate": float(agreements / len(common)),
    }


def _per_user_top1_from_candidates(
    user_profile_rows: list[dict],
    user_index_to_id: dict[int, str],
    user_table_path: Path,
    candidate_query_file: Path,
    cov_mode: str,
    output_tag: str,
) -> dict[str, str]:
    """Recompute per-user top-1 candidate directly from the two checkpoint artifacts.

    Used as a sanity check: even if selected/rejected records store only one
    row per user, we need every user's *best* candidate to compute agreement
    on top-1. Since each user has 10 candidates, and the function only sees
    `sel`+`rej` (one row each), we use a stronger comparison: re-rank all 10
    candidates using the saved query_logvar / query_mu. This requires
    re-loading the checkpoint.
    """
    import torch
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
    import train_vades_lite_sentence_latent_threshold as train_module

    train_module.REPO_ROOT = REPO_ROOT
    train_module.CATEGORY = CATEGORY
    train_module.OUTPUT_TAG = output_tag
    train_module.DEVICE = train_module.infer_device()
    train_module.COVARIANCE_MODE = cov_mode

    cov_tag = f"_{cov_mode}" if cov_mode != "diagonal" else ""
    encoder_path = user_table_path.with_name(f"vades_encoder_{output_tag}{cov_tag}.pt")
    encoder_ckpt = torch.load(encoder_path, map_location=train_module.DEVICE, weights_only=False)
    state_dict = encoder_ckpt["model_state_dict"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    user_table_ckpt = torch.load(user_table_path, map_location=train_module.DEVICE, weights_only=False)
    ut_state_dict = user_table_ckpt["model_state_dict"]
    new_ut_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ut_state_dict.items()}

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
    encoder.load_state_dict(new_state_dict)
    encoder.eval()

    if cov_mode == "full":
        user_table = train_module.UserDistributionTableFull(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    else:
        user_table = train_module.UserDistributionTable(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(train_module.DEVICE)
    user_table.load_state_dict(new_ut_state_dict)
    user_table.eval()

    user_mu_table = user_table.user_mu.weight.detach()

    with open(candidate_query_file, "r", encoding="utf-8") as f:
        cand_rows = [json.loads(line) for line in f if line.strip()]

    user_to_index = {row["user_id"]: i for i, row in enumerate(user_profile_rows)}
    feature_names = list(cand_rows[0]["features"].keys())

    grouped: dict[str, list[dict]] = {}
    for row in cand_rows:
        grouped.setdefault(row["user_id"], []).append(row)

    top1: dict[str, str] = {}
    with torch.no_grad():
        for user_id, cands in grouped.items():
            if user_id not in user_to_index:
                continue
            user_idx = user_to_index[user_id]
            user_mu = user_mu_table[user_idx].unsqueeze(0)
            if cov_mode == "full":
                user_L = user_table.get_L()[user_idx].unsqueeze(0)
            else:
                user_logvar = user_table.user_logvar.weight[user_idx].unsqueeze(0)
            best_score = None
            best_query = None
            for cand in cands:
                feat = np.asarray([float(cand["features"][n]) for n in feature_names], dtype=np.float64).reshape(1, -1)
                feat_t = torch.tensor(feat, dtype=torch.float32, device=train_module.DEVICE)
                mu, disp, _ = encoder(feat_t)
                if cov_mode == "full":
                    score = train_module.multivariate_gaussian_kl(mu, disp, user_mu, user_L).item()
                else:
                    score = train_module.diagonal_gaussian_kl(mu, disp, user_mu, user_logvar).item()
                if best_score is None or score < best_score:
                    best_score = score
                    best_query = cand["query"]
            if best_query is not None:
                top1[user_id] = best_query
    return top1


def main() -> None:
    log("=" * 60)
    log("对角 vs 完整多元高斯 消融对比")
    log("=" * 60)

    sel_diag = _load_jsonl(FEATURE_DIR / f"{DIAG_TAG}_selected_query_records.jsonl")
    rej_diag = _load_jsonl(FEATURE_DIR / f"{DIAG_TAG}_rejected_query_records.jsonl")
    sel_full = _load_jsonl(FEATURE_DIR / f"{FULL_TAG}_selected_query_records.jsonl")
    rej_full = _load_jsonl(FEATURE_DIR / f"{FULL_TAG}_rejected_query_records.jsonl")

    profile_diag = _load_jsonl(FEATURE_DIR / f"{DIAG_TAG}_user_profiles.jsonl")
    profile_full = _load_jsonl(FEATURE_DIR / f"{FULL_TAG}_user_profiles.jsonl")

    summary_diag = json.loads((FEATURE_DIR / f"{DIAG_TAG}_summary.json").read_text(encoding="utf-8"))
    summary_full = json.loads((FEATURE_DIR / f"{FULL_TAG}_summary.json").read_text(encoding="utf-8"))

    log(f"对角: selected={len(sel_diag)}, rejected={len(rej_diag)}, users={len(profile_diag)}")
    log(f"完整: selected={len(sel_full)}, rejected={len(rej_full)}, users={len(profile_full)}")

    metrics: dict = {"category": CATEGORY}

    metrics["accept_rate"] = {
        "diagonal": _accept_rate(sel_diag, rej_diag),
        "full": _accept_rate(sel_full, rej_full),
        "delta_full_minus_diag": None,
    }
    metrics["accept_rate"]["delta_full_minus_diag"] = (
        metrics["accept_rate"]["full"] - metrics["accept_rate"]["diagonal"]
    )

    metrics["selected_range_score"] = {
        "diagonal": _summarize_scores([row["range_score"] for row in sel_diag]),
        "full": _summarize_scores([row["range_score"] for row in sel_full]),
    }
    metrics["rejected_range_score"] = {
        "diagonal": _summarize_scores([row["range_score"] for row in rej_diag]),
        "full": _summarize_scores([row["range_score"] for row in rej_full]),
    }

    # Threshold margin: ratio of range_score to abs_threshold
    diag_thresh = summary_diag.get("abs_threshold_quantile")
    full_thresh = summary_full.get("abs_threshold_quantile")
    metrics["abs_threshold_quantile"] = {
        "diagonal": diag_thresh,
        "full": full_thresh,
    }

    metrics["per_user_logvar_diag_vs_trace"] = {
        "description": "Sum of user_logvar entries (diag) and trace(LL^T) (full) per user; should be comparable in magnitude.",
        "diagonal": {
            "mean_of_sum": float(np.mean([sum(p["user_logvar"]) for p in profile_diag])),
            "median_of_sum": float(np.median([sum(p["user_logvar"]) for p in profile_diag])),
        },
        "full": {
            "mean_of_sum_diag": float(np.mean([sum(p["user_logvar"]) for p in profile_full])),
            "median_of_sum_diag": float(np.median([sum(p["user_logvar"]) for p in profile_full])),
        },
    }

    log("重新计算每用户 top-1 候选并比较...")
    diag_top1 = _per_user_top1_from_candidates(
        profile_diag, {}, FEATURE_DIR / f"vades_user_table_{DIAG_TAG}.pt",
        FEATURE_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl", "diagonal", DIAG_TAG,
    )
    full_top1 = _per_user_top1_from_candidates(
        profile_full, {}, FEATURE_DIR / f"vades_user_table_{FULL_TAG}_full.pt",
        FEATURE_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl", "full", FULL_TAG,
    )
    common = set(diag_top1) & set(full_top1)
    if not common:
        raise ValueError("no overlapping users for top-1 recompute")
    agrees = sum(1 for u in common if diag_top1[u] == full_top1[u])
    metrics["top1_recomputed_agreement"] = {
        "overlap_users": int(len(common)),
        "agree": int(agrees),
        "agreement_rate": float(agrees / len(common)),
    }

    metrics["training_final_loss"] = {
        "diagonal": summary_diag.get("training_epochs", [{}])[-1].get("total_loss_mean"),
        "full": summary_full.get("training_epochs", [{}])[-1].get("total_loss_mean"),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"\n结果已保存: {OUTPUT_FILE}")
    log("\n=== 关键指标对比 ===")
    log(f"accept rate:  对角={metrics['accept_rate']['diagonal']:.4f} | full={metrics['accept_rate']['full']:.4f} | Δ={metrics['accept_rate']['delta_full_minus_diag']:+.4f}")
    log(f"top-1 agreement (recomputed): {metrics['top1_recomputed_agreement']['agreement_rate']:.4f} ({metrics['top1_recomputed_agreement']['agree']}/{metrics['top1_recomputed_agreement']['overlap_users']})")
    log(f"selected score mean:  对角={metrics['selected_range_score']['diagonal']['mean']:.4f} | full={metrics['selected_range_score']['full']['mean']:.4f}")
    log(f"rejected score mean:  对角={metrics['rejected_range_score']['diagonal']['mean']:.4f} | full={metrics['rejected_range_score']['full']['mean']:.4f}")
    log(f"final training loss:  对角={metrics['training_final_loss']['diagonal']} | full={metrics['training_final_loss']['full']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
