#!/usr/bin/env python3
"""Compute per-user mean held-out log-likelihood per latent dim under each prior.

Direct "fit" measure: for each user, the encoder maps the user's sentences to
z ∈ R^D; the user prior p(z|u) is evaluated at each z; the average per-dim
log p is the fit score. Higher = better fit.

This is the most direct way to answer "which prior best fits the true user
distribution", because:
- It uses the same encoder inputs across all groups
- It normalizes by latent dim, so the units (per-dim nats) are comparable
  across Gaussian / Student-t / Laplace / Logistic / GMM families
- It doesn't use the candidate-query mechanism, so it's not a discriminative
  metric (like accept_rate); it's a generative fit metric

Output: JSON with per-group mean per-dim log-likelihood + per-group summary
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = os.environ.get("CATEGORY_OVERRIDE", "Baby_Products")
FEATURE_BASE = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / CATEGORY
SENTENCE_FILE = FEATURE_BASE / "vades_lite_sentence_user_distribution_train10_holdout10_student_t_strict_sentences.jsonl"
# Reuse the same sentence file (symlinked) for all groups since features are
# the same. The file path is the strict tag's symlink which points to the
# shared parse.

sys.path.insert(0, str(REPO_ROOT / "PersoanlQuery" / "10_complexity_analysis" / "common"))
import train_vades_lite_sentence_latent_threshold as M  # noqa: E402


GROUPS = [
    {"key": "gmm_2", "label": "2-GMM (Gaussian)", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_gmm", "cov_mode": "diagonal_gmm", "encoder_dist": "gaussian"},
    {"key": "student_t", "label": "Student-t", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t", "cov_mode": "diagonal_student_t", "encoder_dist": "gaussian"},
    {"key": "laplace", "label": "Laplace", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_laplace", "cov_mode": "diagonal_laplace", "encoder_dist": "gaussian"},
    {"key": "logistic", "label": "Logistic", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_logistic", "cov_mode": "diagonal_logistic", "encoder_dist": "gaussian"},
    {"key": "student_t_gmm", "label": "Student-t GMM (K=2)", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t_gmm", "cov_mode": "diagonal_student_t_gmm", "encoder_dist": "gaussian"},
    {"key": "student_t_strict", "label": "Student-t strict VIB", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t_strict", "cov_mode": "diagonal_student_t", "encoder_dist": "student_t"},
]


def log(msg: str) -> None:
    from datetime import datetime
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _score_user(model, group: dict, dataset: dict, profile_rows: list[dict]) -> dict:
    """For each user, run all their (held-out + train) sentences through encoder,
    score z under the user's prior, return per-dim mean log-likelihood per user.
    """
    M.REPO_ROOT = REPO_ROOT
    M.CATEGORY = CATEGORY
    M.OUTPUT_TAG = group["tag"]
    M.DEVICE = M.infer_device()
    M.COVARIANCE_MODE = group["cov_mode"]
    M.ENCODER_DIST = group["encoder_dist"]

    cov_tag = f"_{group['cov_mode']}" if group["cov_mode"] != "diagonal" else ""
    encoder_path = FEATURE_BASE / f"vades_encoder_{group['tag']}{cov_tag}.pt"
    user_table_path = FEATURE_BASE / f"vades_user_table_{group['tag']}{cov_tag}.pt"
    encoder_ckpt = torch.load(encoder_path, map_location=M.DEVICE, weights_only=False)
    encoder_state = {k.replace("_orig_mod.", ""): v for k, v in encoder_ckpt["model_state_dict"].items()}
    user_table_ckpt = torch.load(user_table_path, map_location=M.DEVICE, weights_only=False)
    user_table_state = {k.replace("_orig_mod.", ""): v for k, v in user_table_ckpt["model_state_dict"].items()}

    # Build encoder
    if group["cov_mode"] == "full":
        encoder = M.SentenceEncoderFull(
            input_dim=encoder_ckpt["input_dim"],
            hidden_dim=encoder_ckpt["hidden_dim"],
            latent_dim=encoder_ckpt["latent_dim"],
        ).to(M.DEVICE)
    elif group["encoder_dist"] == "student_t":
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

    # Build user table
    if group["cov_mode"] == "diagonal_gmm":
        user_table = M.UserDistributionTableGMM(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
            num_components=user_table_ckpt.get("gmm_components", 2),
        ).to(M.DEVICE)
    elif group["cov_mode"] == "diagonal_student_t":
        user_table = M.UserDistributionTableStudentT(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(M.DEVICE)
    elif group["cov_mode"] == "diagonal_laplace":
        user_table = M.UserDistributionTableLaplace(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(M.DEVICE)
    elif group["cov_mode"] == "diagonal_logistic":
        user_table = M.UserDistributionTableLogistic(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(M.DEVICE)
    elif group["cov_mode"] == "diagonal_student_t_gmm":
        user_table = M.UserDistributionTableStudentTGMM(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
            num_components=user_table_ckpt.get("gmm_components", 2),
        ).to(M.DEVICE)
    else:
        user_table = M.UserDistributionTableFull(
            num_users=user_table_ckpt["num_users"],
            latent_dim=user_table_ckpt["latent_dim"],
        ).to(M.DEVICE)
    user_table.load_state_dict(user_table_state)
    user_table.eval()

    cov_mode = group["cov_mode"]
    D = encoder_ckpt["latent_dim"]

    # Per-user per-dim mean log p
    user_per_dim_logp: dict[str, list[float]] = defaultdict(list)

    # Group sentences by user for batched scoring
    user_to_sentence_idx: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset["sentence_rows"]):
        user_to_sentence_idx[row["user_id"]].append(idx)

    user_to_index = dataset["user_to_index"]
    feature_names = list(dataset["sentence_rows"][0]["features"].keys())
    scaled_features = dataset["scaled_features"]  # [N, D_features]

    with torch.no_grad():
        for user_id, sentence_indices in user_to_sentence_idx.items():
            if user_id not in user_to_index:
                continue
            user_idx = user_to_index[user_id]
            batch_features = torch.tensor(scaled_features[sentence_indices], dtype=torch.float32, device=M.DEVICE)

            # Encoder forward
            enc_out = encoder(batch_features)
            if len(enc_out) == 4:
                mu, logvar, df, _ = enc_out
                # For strict VIB: use reparam z (matches training-time scoring)
                z = M.reparameterize_student_t(mu, logvar, df)
            else:
                mu, logvar, _ = enc_out
                df = None
                z = None

            # Score under user prior
            if cov_mode == "diagonal_gmm":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)        # [1, K, D]
                u_lv = user_table.user_logvar[user_idx].unsqueeze(0)    # [1, K, D]
                u_mix = user_table.mix_logits[user_idx].unsqueeze(0)    # [1, K]
                # log p_t(x|u) per dim per component; mix via logsumexp
                # gmm_log_likelihood signature: (mu_q, logvar_q, user_mu, user_logvar, mix_logits)
                log_p = M.gmm_log_likelihood(mu, logvar, u_mu, u_lv, u_mix)  # [B]
                # Sum over latent dim is inside gmm_log_likelihood; per-dim is log_p / D
                per_dim = (log_p / D).cpu().numpy()  # [B]
            elif cov_mode == "diagonal_student_t":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)            # [1, D]
                u_ls = user_table.user_log_scale[user_idx].unsqueeze(0)     # [1, D]
                u_df = user_table.df_all()[user_idx].unsqueeze(0)          # [1]
                if df is not None and z is not None:
                    # strict VIB path: reparam z
                    log_p = M.student_t_log_p_matrix(z, u_mu, u_ls, u_df)  # [B]
                else:
                    log_p = M.student_t_log_likelihood(mu, logvar, u_mu, u_ls, u_df)  # [B]
                per_dim = (log_p / D).cpu().numpy()
            elif cov_mode == "diagonal_laplace":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                u_lb = user_table.user_log_b[user_idx].unsqueeze(0)
                log_p = M.laplace_log_likelihood(mu, logvar, u_mu, u_lb)
                per_dim = (log_p / D).cpu().numpy()
            elif cov_mode == "diagonal_logistic":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                u_ls = user_table.user_log_s[user_idx].unsqueeze(0)
                log_p = M.logistic_log_likelihood(mu, logvar, u_mu, u_ls)
                per_dim = (log_p / D).cpu().numpy()
            elif cov_mode == "diagonal_student_t_gmm":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                u_ls = user_table.user_log_scale[user_idx].unsqueeze(0)
                u_df = user_table.df_all()[user_idx].unsqueeze(0)
                u_mix = user_table.mix_logits[user_idx].unsqueeze(0)
                log_p = M.student_t_gmm_log_likelihood(mu, logvar, u_mu, u_ls, u_df, u_mix)
                per_dim = (log_p / D).cpu().numpy()
            else:
                raise NotImplementedError(f"unsupported cov_mode {cov_mode}")

            user_per_dim_logp[user_id].extend(per_dim.tolist())

    # Aggregate
    per_user_means = {uid: float(np.mean(v)) for uid, v in user_per_dim_logp.items() if v}
    all_means = list(per_user_means.values())
    return {
        "n_users": len(per_user_means),
        "n_sentences": sum(len(v) for v in user_per_dim_logp.values()),
        "global_mean_per_dim_logp": float(np.mean(all_means)) if all_means else None,
        "user_mean_std": float(np.std(all_means)) if all_means else None,
        "user_mean_q50": float(np.quantile(all_means, 0.50)) if all_means else None,
        "user_mean_q10": float(np.quantile(all_means, 0.10)) if all_means else None,
        "user_mean_q90": float(np.quantile(all_means, 0.90)) if all_means else None,
    }


def main() -> None:
    log("=" * 80)
    log("Per-user per-dim log-likelihood under each learned prior (true fit)")
    log("=" * 80)

    # Build dataset once (sentence features are shared)
    log("Building shared dataset...")
    M.REPO_ROOT = REPO_ROOT
    M.CATEGORY = CATEGORY
    M.INPUT_DIR = FEATURE_BASE
    M.OUTPUT_TAG = "vades_lite_sentence_user_distribution_train10_holdout10_student_t_strict"

    sent_rows = _load_jsonl(SENTENCE_FILE)
    feature_names = list(sent_rows[0]["features"].keys())
    user_ids, dataset = M.build_training_dataset(sent_rows, feature_names)
    log(f"  dataset: {len(user_ids)} users, {len(dataset['sentence_rows'])} sentences")

    all_results: dict = {"category": CATEGORY, "groups": {}}
    for group in GROUPS:
        log(f"[{group['key']}] scoring per-user fit...")
        try:
            metrics = _score_user(None, group, dataset, [])
            all_results["groups"][group["key"]] = {
                "label": group["label"],
                "cov_mode": group["cov_mode"],
                "encoder_dist": group["encoder_dist"],
                **metrics,
            }
            log(f"  ✓ mean per-dim log p = {metrics['global_mean_per_dim_logp']:.4f} nats")
        except Exception as e:
            log(f"  ✗ error: {e}")
            all_results["groups"][group["key"]] = {"label": group["label"], "error": str(e)}

    # Print ranking
    log("\n" + "=" * 80)
    log("Fit ranking (higher per-dim log p = better fit)")
    log("=" * 80)
    ranking = []
    for k, v in all_results["groups"].items():
        if "error" not in v and v.get("global_mean_per_dim_logp") is not None:
            ranking.append((v["global_mean_per_dim_logp"], k, v["label"]))
    ranking.sort(reverse=True)
    log(f"{'rank':<5} {'per-dim log p':>14}  {'group':<25}")
    for i, (score, k, label) in enumerate(ranking, 1):
        log(f"{i:<5} {score:>14.4f}  {label}")
    log("=" * 80)

    # Save JSON
    out_path = FEATURE_BASE.parent / "interpretability_paper" / CATEGORY / "fit_quality.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    log(f"\nJSON: {out_path}")


if __name__ == "__main__":
    main()
