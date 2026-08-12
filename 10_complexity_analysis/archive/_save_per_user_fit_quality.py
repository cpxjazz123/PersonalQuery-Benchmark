#!/usr/bin/env python3
"""Compute per-user mean log p (encoder-based) for all 6 user-prior groups and
save to JSON. This is the proper "fit quality" metric — scores sentence-level
mu_q under the user's own prior (not user_mu under its own prior, which is just
self-consistency).

Output: result/.../interpretability_paper/<CAT>/per_user_fit_quality.json
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = os.environ.get("CATEGORY_OVERRIDE", "Baby_Products")
FEATURE_BASE = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / CATEGORY
SENTENCE_FILE = FEATURE_BASE / "vades_lite_sentence_user_distribution_train10_holdout10_student_t_strict_sentences.jsonl"
INTERP = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / "interpretability_paper" / CATEGORY

sys.path.insert(0, str(REPO_ROOT / "PersoanlQuery" / "10_complexity_analysis" / "common"))
import train_vades_lite_sentence_latent_threshold as M  # noqa: E402

GROUPS = [
    {"key": "gmm_2",           "label": "2-GMM (Gaussian)",     "tag": "vades_lite_sentence_user_distribution_train10_holdout10_gmm",            "cov_mode": "diagonal_gmm",           "encoder_dist": "gaussian"},
    {"key": "student_t",       "label": "Student-t",            "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t",        "cov_mode": "diagonal_student_t",      "encoder_dist": "gaussian"},
    {"key": "laplace",         "label": "Laplace",              "tag": "vades_lite_sentence_user_distribution_train10_holdout10_laplace",          "cov_mode": "diagonal_laplace",        "encoder_dist": "gaussian"},
    {"key": "logistic",        "label": "Logistic",             "tag": "vades_lite_sentence_user_distribution_train10_holdout10_logistic",         "cov_mode": "diagonal_logistic",       "encoder_dist": "gaussian"},
    {"key": "student_t_gmm",   "label": "Student-t GMM (K=2)",  "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t_gmm",    "cov_mode": "diagonal_student_t_gmm",  "encoder_dist": "gaussian"},
    {"key": "student_t_strict","label": "Student-t strict VIB", "tag": "vades_lite_sentence_user_distribution_train10_holdout10_student_t_strict", "cov_mode": "diagonal_student_t",      "encoder_dist": "student_t"},
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _per_user_logp(group: dict, dataset: dict) -> dict[str, float]:
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
    if group["encoder_dist"] == "student_t":
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
    cov_mode = group["cov_mode"]
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
    else:
        raise NotImplementedError(f"unsupported {cov_mode}")
    user_table.load_state_dict(user_table_state)
    user_table.eval()

    D = encoder_ckpt["latent_dim"]
    user_per_dim_logp: dict[str, list[float]] = defaultdict(list)
    user_to_sentence_idx: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(dataset["sentence_rows"]):
        user_to_sentence_idx[row["user_id"]].append(idx)
    user_to_index = dataset["user_to_index"]
    scaled_features = dataset["scaled_features"]

    with torch.no_grad():
        for user_id, sentence_indices in user_to_sentence_idx.items():
            if user_id not in user_to_index:
                continue
            user_idx = user_to_index[user_id]
            batch_features = torch.tensor(scaled_features[sentence_indices], dtype=torch.float32, device=M.DEVICE)
            enc_out = encoder(batch_features)
            if len(enc_out) == 4:
                mu, logvar, df, _ = enc_out
                z = M.reparameterize_student_t(mu, logvar, df)
            else:
                mu, logvar, _ = enc_out
                df = None
                z = None

            if cov_mode == "diagonal_gmm":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                u_lv = user_table.user_logvar[user_idx].unsqueeze(0)
                u_mix = user_table.mix_logits[user_idx].unsqueeze(0)
                log_p = M.gmm_log_likelihood(mu, logvar, u_mu, u_lv, u_mix)
            elif cov_mode == "diagonal_student_t":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                u_ls = user_table.user_log_scale[user_idx].unsqueeze(0)
                u_df = user_table.df_all()[user_idx].unsqueeze(0)
                if df is not None and z is not None:
                    log_p = M.student_t_log_p_matrix(z, u_mu, u_ls, u_df)
                else:
                    log_p = M.student_t_log_likelihood(mu, logvar, u_mu, u_ls, u_df)
            elif cov_mode == "diagonal_laplace":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                u_lb = user_table.user_log_b[user_idx].unsqueeze(0)
                log_p = M.laplace_log_likelihood(mu, logvar, u_mu, u_lb)
            elif cov_mode == "diagonal_logistic":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                u_ls = user_table.user_log_s[user_idx].unsqueeze(0)
                log_p = M.logistic_log_likelihood(mu, logvar, u_mu, u_ls)
            elif cov_mode == "diagonal_student_t_gmm":
                u_mu = user_table.user_mu[user_idx].unsqueeze(0)
                u_ls = user_table.user_log_scale[user_idx].unsqueeze(0)
                u_df = user_table.df_all()[user_idx].unsqueeze(0)
                u_mix = user_table.mix_logits[user_idx].unsqueeze(0)
                log_p = M.student_t_gmm_log_likelihood(mu, logvar, u_mu, u_ls, u_df, u_mix)
            else:
                raise NotImplementedError(cov_mode)
            per_dim = (log_p / D).cpu().numpy()
            user_per_dim_logp[user_id].extend(per_dim.tolist())

    return {uid: float(np.mean(v)) for uid, v in user_per_dim_logp.items() if v}


def main() -> None:
    log("=" * 80)
    log(f"Per-user fit quality (encoder-based) for {CATEGORY}")
    log("=" * 80)

    log("Building shared dataset...")
    M.REPO_ROOT = REPO_ROOT
    M.CATEGORY = CATEGORY
    M.INPUT_DIR = FEATURE_BASE
    M.OUTPUT_TAG = "vades_lite_sentence_user_distribution_train10_holdout10_student_t_strict"

    sent_rows = _load_jsonl(SENTENCE_FILE)
    feature_names = list(sent_rows[0]["features"].keys())
    user_ids, dataset = M.build_training_dataset(sent_rows, feature_names)
    log(f"  dataset: {len(user_ids)} users, {len(dataset['sentence_rows'])} sentences")

    out: dict = {"category": CATEGORY, "groups": {}}
    for g in GROUPS:
        log(f"[{g['key']}] scoring per-user fit quality...")
        per_user = _per_user_logp(g, dataset)
        if per_user:
            arr = np.array(list(per_user.values()))
            out["groups"][g["key"]] = {
                "label": g["label"],
                "cov_mode": g["cov_mode"],
                "per_user_mean_logp": per_user,
                "n_users": len(per_user),
                "global_mean": float(arr.mean()),
                "q10": float(np.quantile(arr, 0.10)),
                "q50": float(np.quantile(arr, 0.50)),
                "q90": float(np.quantile(arr, 0.90)),
            }
            log(f"  ✓ mean={arr.mean():.4f}, q10={np.quantile(arr, 0.10):.4f}, q90={np.quantile(arr, 0.90):.4f}")

    INTERP.mkdir(parents=True, exist_ok=True)
    out_path = INTERP / "per_user_fit_quality.json"
    with out_path.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
