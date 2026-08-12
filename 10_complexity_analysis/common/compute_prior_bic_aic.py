#!/usr/bin/env python3
"""Compute BIC/AIC for GMM vs other priors to validate Table 3 claims.

Usage:
    python compute_prior_bic_aic.py --category Baby_Products

This addresses the reviewer concern from iter #38/#41:
  "GMM prior 'best fits' is circular — GMM has more free parameters (2-component)
   than single-component priors, so higher likelihood is expected. BIC/AIC with
   proper model-complexity penalty is needed."

BIC = k * log(n) - 2 * log(L)
AIC = 2 * k - 2 * log(L)

Where k = number of free parameters, n = number of data points.

Parameter counts (per-user diagonal distributions, latent_dim = LATENT_DIM):
  - GMM (K=2):        k = K*d (means) + K*d (log variances) + K (mixing logits via softmax)
                      = 4*d + 2 for K=2
  - t-distribution:   k = d (mean) + d (log_scale) + 1 (df) = 2*d + 1
  - Laplace:          k = d (mean) + d (log_b) = 2*d
  - Logistic:         k = d (mean) + d (log_s) = 2*d

These match the actual nn.Module shapes in train_vades_lite_sentence_latent_threshold.py:
  - UserDistributionTableGMM:        user_mu [U,K,D] + user_logvar [U,K,D] + mix_logits [U,K]
  - UserDistributionTableStudentT:   user_mu [U,D] + user_log_scale [U,D] + user_df_raw [U]
  - UserDistributionTableLaplace:     user_mu [U,D] + user_log_b [U,D]
  - UserDistributionTableLogistic:    user_mu [U,D] + user_log_s [U,D]
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from scipy.stats import kruskal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/fs04/ar57/wenyu")


def compute_bic_aic_for_prior(
    log_likelihood: float,
    n_params: int,
    n_samples: int,
) -> tuple[float, float]:
    """Compute BIC and AIC from log-likelihood, number of params, and sample count."""
    bic = n_params * np.log(n_samples) - 2.0 * log_likelihood
    aic = 2.0 * n_params - 2.0 * log_likelihood
    return float(bic), float(aic)


def load_latent_representations(category: str) -> dict | None:
    """Load VAE latent representations and training info from Stage 10 output.

    Expected files:
      - result/personal_query/12_complexity_analysis_clause_features/<cat>/<tag>_user_profiles.jsonl
      - result/personal_query/12_complexity_analysis_clause_features/<cat>/<tag>_sentences.jsonl
    """
    base = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / category
    if not base.exists():
        return None

    # Find the most recent output_tag directory
    tag_dirs = [d for d in base.iterdir() if d.is_dir()]
    if not tag_dirs:
        return None
    latest_tag = sorted(tag_dirs)[-1].name

    profile_file = base / latest_tag / "user_profiles.jsonl"
    sentence_file = base / latest_tag / "sentences.jsonl"

    if not profile_file.exists() or not sentence_file.exists():
        return None

    profiles = []
    with open(profile_file) as f:
        for line in f:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))

    sentences = []
    with open(sentence_file) as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(json.loads(line))

    return {
        "profiles": profiles,
        "sentences": sentences,
        "tag": latest_tag,
    }


def fit_user_distribution(
    profile: dict, sentences: list
) -> dict:
    """Fit a user-specific distribution (GMM / t / Laplace / Logistic) on user sentences.

    Iter #179: extended compute_prior_bic_aic.py to actually fit and compute
    per-user log-likelihood, instead of stub fallback.
    Iter #181: add t-distribution fitting (4 priors total, matching paper Table 3).

    Param counts match nn.Module shapes in
    train_vades_lite_sentence_latent_threshold.py:
      - GMM:        UserDistributionTableGMM
      - t-dist:     UserDistributionTableStudentT (df via sigmoid, df=2+8*sig)
      - Laplace:    UserDistributionTableLaplace
      - Logistic:   UserDistributionTableLogistic

    The stub _stub_synthetic_vae_latent.py only emits GMM-relevant fields
    (mu, logvar, mix_logits). For t-distribution fitting we approximate
    `df_raw` from a heuristic (default 0 → sigmoid 0.5 → df=6) since the
    stub doesn't carry t-specific parameters. Real Stage 12 outputs would
    carry user_df_raw per the train_vades_lite schema.
    """
    mu = np.asarray(profile['mu'])             # [K, D]
    logvar = np.asarray(profile['logvar'])     # [K, D]
    mix_logits = np.asarray(profile['mix_logits'])  # [K]

    user_latents = np.array([s['latent'] for s in sentences])  # [n_user, D]
    n_user = user_latents.shape[0]

    # ---------- GMM log-likelihood ----------
    diff_sq = (user_latents[None, :, :] - mu[:, None, :]) ** 2  # [K, n, D]
    log_phi = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * logvar[:, None, :]
        - 0.5 * diff_sq / np.exp(logvar)[:, None, :]
    )
    log_phi = log_phi.sum(axis=-1)  # [K, n_user]

    # softmax of mix_logits
    logits = mix_logits - mix_logits.max()
    log_weights = logits - np.log(np.exp(logits).sum())
    gmm_log_lik = np.logaddexp.reduce(
        log_weights[:, None] + log_phi, axis=0
    ).sum()

    # ---------- Laplace log-likelihood ----------
    lap_mu = mu.mean(axis=0)
    lap_std = user_latents.std(axis=0)
    lap_b = lap_std / np.sqrt(2)
    lap_log_b = np.log(lap_b + 1e-12)
    abs_diff = np.abs(user_latents - lap_mu[None, :])
    lap_log_lik = (-np.log(2) - lap_log_b[None, :] - abs_diff / lap_b[None, :]).sum()

    # ---------- Logistic log-likelihood ----------
    log_mu = lap_mu
    log_s = lap_std * np.sqrt(3) / np.pi
    log_log_s = np.log(log_s + 1e-12)
    z = (user_latents - log_mu[None, :]) / log_s[None, :]
    log_log_lik = (
        -log_log_s[None, :] - z - 2 * np.log1p(np.exp(-z))
    ).sum()

    # ---------- t-distribution log-likelihood (iter #181) ----------
    # user_mu [D] + user_log_scale [D] + user_df_raw [1] per
    # UserDistributionTableStudentT (train_vades_lite_sentence_latent_threshold.py:568-583).
    # df mapped via sigmoid: df = 2 + 8 * sigmoid(df_raw), giving df in [2, 10].
    # Use empirical mean for mu, empirical std for log_scale, df_raw=0 (df=6 default).
    t_mu = lap_mu
    t_log_scale = np.log(lap_std + 1e-12)  # [D]
    t_df_raw = float(profile.get('df_raw', 0.0))  # default 0 → df=6 via sigmoid
    t_df = 2.0 + 8.0 / (1.0 + np.exp(-t_df_raw))  # sigmoid → df in [2, 10]
    # log p(x|mu, scale, df) per dim = log Γ((df+1)/2) - log Γ(df/2)
    #   - 0.5 log(df π) - log(scale) - ((df+1)/2) log(1 + (x-mu)^2 / (df * scale^2))
    z_t = (user_latents - t_mu[None, :]) / np.exp(t_log_scale[None, :])  # [n_user, D]
    z_t_sq_scaled = z_t ** 2 / t_df
    # scipy.special.gammaln for stable log-gamma
    from scipy.special import gammaln
    log_norm = gammaln(0.5 * (t_df + 1)) - gammaln(0.5 * t_df) \
               - 0.5 * np.log(t_df * np.pi) - t_log_scale[None, :]
    t_log_lik = (
        log_norm
        - 0.5 * (t_df + 1) * np.log1p(z_t_sq_scaled)
    ).sum()

    return {
        'n_user': n_user,
        'gmm_log_lik': float(gmm_log_lik),
        't_log_lik': float(t_log_lik),
        'laplace_log_lik': float(lap_log_lik),
        'logistic_log_lik': float(log_log_lik),
    }


def _gmm_loglik_against(profile: dict, held_out_latents: np.ndarray) -> float:
    """Helper: evaluate GMM log-likelihood on a held-out set using existing profile.

    iter #187: extract GMM-log-likelihood-on-X function to enable held-out eval.
    """
    mu = np.asarray(profile['mu'])
    logvar = np.asarray(profile['logvar'])
    mix_logits = np.asarray(profile['mix_logits'])

    n = held_out_latents.shape[0]
    if n == 0:
        return float("nan")
    diff_sq = (held_out_latents[None, :, :] - mu[:, None, :]) ** 2
    log_phi = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * logvar[:, None, :]
        - 0.5 * diff_sq / np.exp(logvar)[:, None, :]
    ).sum(axis=-1)
    logits = mix_logits - mix_logits.max()
    log_weights = logits - np.log(np.exp(logits).sum())
    return float(np.logaddexp.reduce(log_weights[:, None] + log_phi, axis=0).sum())


def _laplace_loglik_against(profile: dict, held_out_latents: np.ndarray, train_latents: np.ndarray) -> float:
    mu = np.asarray(profile['mu']).mean(axis=0)
    std = train_latents.std(axis=0)
    lap_b = std / np.sqrt(2)
    lap_log_b = np.log(lap_b + 1e-12)
    abs_diff = np.abs(held_out_latents - mu[None, :])
    return float((-np.log(2) - lap_log_b[None, :] - abs_diff / lap_b[None, :]).sum())


def _logistic_loglik_against(profile: dict, held_out_latents: np.ndarray, train_latents: np.ndarray) -> float:
    mu = np.asarray(profile['mu']).mean(axis=0)
    s = train_latents.std(axis=0) * np.sqrt(3) / np.pi
    log_s = np.log(s + 1e-12)
    z = (held_out_latents - mu[None, :]) / s[None, :]
    return float((-log_s[None, :] - z - 2 * np.log1p(np.exp(-z))).sum())


def _t_loglik_against(profile: dict, held_out_latents: np.ndarray, train_latents: np.ndarray) -> float:
    from scipy.special import gammaln
    mu = np.asarray(profile['mu']).mean(axis=0)
    std = train_latents.std(axis=0)
    log_scale = np.log(std + 1e-12)
    df_raw = float(profile.get('df_raw', 0.0))
    df = 2.0 + 8.0 / (1.0 + np.exp(-df_raw))
    z = (held_out_latents - mu[None, :]) / np.exp(log_scale[None, :])
    z_sq_scaled = z ** 2 / df
    log_norm = gammaln(0.5 * (df + 1)) - gammaln(0.5 * df) \
               - 0.5 * np.log(df * np.pi) - log_scale[None, :]
    return float((log_norm - 0.5 * (df + 1) * np.log1p(z_sq_scaled)).sum())


def held_out_loglik_evaluation(
    profile: dict,
    sentences: list,
    held_out_frac: float = 0.2,
    rng_seed: int = 42,
) -> dict:
    """iter #187: held-out log-likelihood evaluation to detect in-sample overfit.

    Reviewer concern (P0 iter #38): "GMM prior 结论是循环论证" — paper Table 3
    values are computed on full sentences (in-sample); params are fit on the
    same data they are evaluated on.

    This function:
      1. Splits sentences into train + held_out (default 80/20, fixed seed).
      2. Evaluates per-prior log-likelihood on held_out using profile params
         (mu, logvar, mix_logits) — params are treated as if they were fit
         on full data; we measure held-out set fit quality.
      3. Compares in-sample GMM log-likelihood vs held-out GMM log-likelihood;
         ratio > 1 indicates GMM overfits (in-sample adv not preserved OOD).

    Returns dict with 4 held-out log-likelihoods + overfit_ratio_gmm.
    """
    import random as _random
    all_latents = np.array([s['latent'] for s in sentences])
    n_total = all_latents.shape[0]
    if n_total < 5:
        return {"error": "too few sentences for held-out split", "n_total": int(n_total)}

    rng = _random.Random(rng_seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    n_held_out = max(1, int(round(held_out_frac * n_total)))
    held_out_idx = indices[:n_held_out]
    train_idx = indices[n_held_out:]

    held_out_latents = all_latents[held_out_idx]
    train_latents = all_latents[train_idx]

    h_gmm = _gmm_loglik_against(profile, held_out_latents)
    h_t = _t_loglik_against(profile, held_out_latents, train_latents)
    h_lap = _laplace_loglik_against(profile, held_out_latents, train_latents)
    h_log = _logistic_loglik_against(profile, held_out_latents, train_latents)
    i_gmm = _gmm_loglik_against(profile, all_latents)
    overfit_ratio = (i_gmm / h_gmm) if h_gmm != 0 else float("nan")

    return {
        'n_train': int(len(train_idx)),
        'n_held_out': int(len(held_out_idx)),
        'held_out_gmm_loglik': float(h_gmm),
        'held_out_t_loglik': float(h_t),
        'held_out_laplace_loglik': float(h_lap),
        'held_out_logistic_loglik': float(h_log),
        'in_sample_gmm_loglik': float(i_gmm),
        'overfit_ratio_gmm': float(overfit_ratio),
    }


def main_3domain() -> None:
    """iter #182: run BIC/AIC framework on all 3 domains + report 3-domain mean.

    This mirrors paper §3.4 Table 3 "3-domain log p" column, which reports
    the mean across Baby_Products / Grocery_and_Gourmet_Food / Pet_Supplies.
    Useful for direct comparison to paper numbers when synthetic data is used.
    """
    categories = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
    print("=" * 80)
    print(f"3-domain BIC/AIC + Table 3 schema (synthetic VAE latent, iter #182)")
    print("=" * 80)

    # Collect per-domain per-prior metrics
    domain_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for cat in categories:
        print(f"\n--- {cat} ---")
        print(f"Generating synthetic latent (100 users × 10 sents) ...")
        from _stub_synthetic_vae_latent import OUTPUT_ROOT  # type: ignore
        import subprocess
        # Generate stub data for this category
        subprocess.run(
            ["python3", str(Path(__file__).parent / "_stub_synthetic_vae_latent.py"),
             "--category", cat, "--n-users", "100", "--n-sentences-per-user", "10",
             "--tag", "synthetic_iter182"],
            check=True,
        )
        # Run framework on this category
        data = load_latent_representations(cat)
        if data is None:
            print(f"WARNING: failed to load synthetic data for {cat}")
            continue
        # Build per-user dicts
        profiles_by_user = {p['user_id']: p for p in data['profiles']}
        sentences_by_user: dict[str, list] = {}
        for s in data['sentences']:
            sentences_by_user.setdefault(s['user_id'], []).append(s)
        # Fit and accumulate
        totals = {"GMM": 0.0, "t-distribution": 0.0, "Laplace": 0.0, "Logistic": 0.0}
        per_user_lists = {"GMM": [], "t-distribution": [], "Laplace": [], "Logistic": []}
        for uid, profile in profiles_by_user.items():
            sents = sentences_by_user.get(uid, [])
            if len(sents) < 2:
                continue
            result = fit_user_distribution(profile, sents)
            for name in totals:
                totals[name] += result[f"{name.split('-')[0].lower()}_log_lik"] if name == "GMM" else (
                    result["t_log_lik"] if name == "t-distribution" else (
                        result["laplace_log_lik"] if name == "Laplace" else result["logistic_log_lik"]
                    )
                )
                per_user_lists[name].append(result["t_log_lik"] / result["n_user"] if name == "t-distribution"
                                              else result["gmm_log_lik"] / result["n_user"] if name == "GMM"
                                              else result["laplace_log_lik"] / result["n_user"] if name == "Laplace"
                                              else result["logistic_log_lik"] / result["n_user"])
        n_samples_total = sum(len(s) for s in sentences_by_user.values())
        # Compute per-domain log p, intercept, q50
        domain_metrics[cat] = {}
        for name in ("GMM", "t-distribution", "Laplace", "Logistic"):
            ll = totals[name]
            per_user = per_user_lists[name]
            log_p = ll / n_samples_total if n_samples_total > 0 else 0.0
            intercept = min(per_user) if per_user else 0.0
            q50 = float(np.median(per_user)) if per_user else 0.0
            domain_metrics[cat][name] = {
                "log_p": log_p, "intercept": intercept, "q50": q50,
                "n_samples": n_samples_total,
            }
            print(f"  {name:<20} log_p={log_p:8.2f}  intercept={intercept:8.2f}  q50={q50:8.2f}")

    # 3-domain mean
    print()
    print("=" * 80)
    print("3-domain mean (iter #182)")
    print("=" * 80)
    paper_table3 = {
        "GMM": {"log_p": -1.09, "intercept": -0.39, "q50": -1.06},
        "t-distribution": {"log_p": -1.24, "intercept": -0.98, "q50": -1.22},
        "Laplace": {"log_p": -1.22, "intercept": -1.07, "q50": -1.21},
        "Logistic": {"log_p": -1.41, "intercept": -1.30, "q50": -1.40},
    }
    print(f"{'Prior':<20} {'mean log p':<14} {'mean intercept':<16} {'mean q50':<12}")
    print("-" * 64)
    for name in ("GMM", "t-distribution", "Laplace", "Logistic"):
        log_p_list = [domain_metrics[c][name]["log_p"] for c in categories if name in domain_metrics.get(c, {})]
        intercept_list = [domain_metrics[c][name]["intercept"] for c in categories if name in domain_metrics.get(c, {})]
        q50_list = [domain_metrics[c][name]["q50"] for c in categories if name in domain_metrics.get(c, {})]
        if not log_p_list:
            continue
        mean_log_p = sum(log_p_list) / len(log_p_list)
        mean_intercept = sum(intercept_list) / len(intercept_list)
        mean_q50 = sum(q50_list) / len(q50_list)
        paper_v = paper_table3[name]
        print(f"{name:<20} {mean_log_p:<14.2f} {mean_intercept:<16.2f} {mean_q50:<12.2f}  "
              f"[paper: {paper_v['log_p']:.2f}, {paper_v['intercept']:.2f}, {paper_v['q50']:.2f}]")
    print()
    print("NOTE: synthetic data is GMM-generated, so the framework will favor GMM by")
    print("      construction. These numbers validate framework math correctness only.")
    print("      Real empirical verification requires Stage 12 outputs (≈9-21 h GPU).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BIC/AIC for GMM vs other priors")
    parser.add_argument(
        "--category",
        default="Baby_Products",
        choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"],
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="VAE latent dimension (default: 8, matching LATENT_DIM in train script)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of GMM components (default: 2, matching GMM_COMPONENTS)",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="iter #187: also evaluate held-out log-likelihood to detect in-sample overfit",
    )
    parser.add_argument(
        "--held-out-frac",
        type=float,
        default=0.2,
        help="iter #187: held-out fraction (default 0.2)",
    )
    args = parser.parse_args()

    d = args.latent_dim
    K = args.n_components

    n_params = {
        # GMM (K components, diagonal covariance): K*d means + K*d log-vars + K mix logits
        # Mix weights via softmax(logits) over K components: K free params (not K-1).
        "GMM": 2 * K * d + K,  # 4*d + 2 for K=2
        # t-distribution (diagonal): d means + d log_scales + 1 df (df mapped via sigmoid)
        "t-distribution": 2 * d + 1,
        # Laplace (diagonal): d means + d log_bs — no df, no extra scale param
        "Laplace": 2 * d,
        # Logistic (diagonal): d means + d log_ss — no df, no extra scale param
        "Logistic": 2 * d,
    }

    print("=" * 80)
    print(f"BIC/AIC Comparison for {args.category}")
    print(f"Latent dim: {d}, GMM components: {K}")
    print("=" * 80)
    print(f"{'Prior':<20} {'k (params)':<12}")
    print("-" * 32)
    for name, k in n_params.items():
        print(f"{name:<20} {k:<12}")
    print()

    data = load_latent_representations(args.category)
    if data is None:
        print(f"WARNING: No Stage 10 output found for {args.category}")
        print("  Run Stage 10 train script first to generate VAE latent representations.")
        print("  BIC/AIC computation requires the actual log-likelihood values.")
        print("")
        print("Parameter counts used:")
        for name, k in n_params.items():
            print(f"  {name:<20}: k={k}")
        print("")
        print("To complete this analysis, run Stage 10 and re-run this script.")
        return

    # iter #179: per-user fit + sum log-likelihoods across all users
    # iter #180: also collect per-user mean log-likelihood for Table 3 schema
    profiles_by_user = {p['user_id']: p for p in data['profiles']}
    sentences_by_user: dict[str, list] = {}
    for s in data['sentences']:
        sentences_by_user.setdefault(s['user_id'], []).append(s)

    n_users_fitted = 0
    gmm_total = 0.0
    t_total = 0.0
    lap_total = 0.0
    log_total = 0.0
    # iter #180: per-user log-likelihood medians for q50 metric
    # iter #181: include t-distribution
    gmm_per_user: list[float] = []
    t_per_user: list[float] = []
    lap_per_user: list[float] = []
    log_per_user: list[float] = []
    for user_id, profile in profiles_by_user.items():
        sents = sentences_by_user.get(user_id, [])
        if len(sents) < 2:
            continue  # need ≥2 points for variance estimate
        result = fit_user_distribution(profile, sents)
        gmm_total += result['gmm_log_lik']
        t_total += result['t_log_lik']
        lap_total += result['laplace_log_lik']
        log_total += result['logistic_log_lik']
        n_user = result['n_user']
        # iter #180: per-sentence mean log-likelihood (paper Table 3 "log p" = Σ log L / n_samples)
        gmm_per_user.append(result['gmm_log_lik'] / n_user)
        t_per_user.append(result['t_log_lik'] / n_user)
        lap_per_user.append(result['laplace_log_lik'] / n_user)
        log_per_user.append(result['logistic_log_lik'] / n_user)
        n_users_fitted += 1

    n_samples_total = sum(len(s) for s in sentences_by_user.values())
    print(f"Fitted {n_users_fitted} users, {n_samples_total} total latent points")
    print()
    print("--- iter #179: BIC/AIC framework output ---")
    print(f"{'Prior':<20} {'k':<8} {'Σ log L':<14} {'BIC':<12} {'AIC':<12}")
    print("-" * 66)
    prior_totals = {
        "GMM": gmm_total,
        "t-distribution": t_total,
        "Laplace": lap_total,
        "Logistic": log_total,
    }
    for name, k in n_params.items():
        if name not in prior_totals:
            continue
        ll = prior_totals[name]
        bic = k * np.log(n_samples_total) - 2.0 * ll
        aic = 2.0 * k - 2.0 * ll
        print(f"{name:<20} {k:<8} {ll:<14.2f} {bic:<12.2f} {aic:<12.2f}")
    print()
    # iter #180: paper §3.4 Table 3 schema (3-domain log p, intercept, q50)
    # iter #181: include t-distribution row
    print("--- iter #180: paper Table 3 schema (per-sentence log p, intercept, q50) ---")
    print(f"{'Prior':<20} {'log p (nat/sent)':<18} {'intercept':<12} {'q50':<10}")
    print("-" * 60)
    paper_table3 = {
        "GMM": {"log_p": -1.09, "intercept": -0.39, "q50": -1.06},
        "t-distribution": {"log_p": -1.24, "intercept": -0.98, "q50": -1.22},
        "Laplace": {"log_p": -1.22, "intercept": -1.07, "q50": -1.21},
        "Logistic": {"log_p": -1.41, "intercept": -1.30, "q50": -1.40},
    }
    prior_per_user_map = {
        "GMM": (gmm_total, gmm_per_user),
        "t-distribution": (t_total, t_per_user),
        "Laplace": (lap_total, lap_per_user),
        "Logistic": (log_total, log_per_user),
    }
    for name in ("GMM", "t-distribution", "Laplace", "Logistic"):
        ll_total, per_user = prior_per_user_map[name]
        # log p = Σ log L / n_samples_total (per-sentence mean)
        log_p = ll_total / n_samples_total if n_samples_total > 0 else 0.0
        # intercept = log-likelihood at "low-variance baseline":
        # paper §3.4 says GMM intercept=-0.39 leads by 0.59-0.91 nats.
        # Approximation: per-user mean log-likelihood at the lowest-variance user
        # (= worst-case user fit) — proxy for baseline.
        intercept = min(per_user) if per_user else 0.0
        # q50 = median per-user log-likelihood
        q50 = float(np.median(per_user)) if per_user else 0.0
        paper_v = paper_table3[name]
        print(f"{name:<20} {log_p:<18.2f} {intercept:<12.2f} {q50:<10.2f}  "
              f"[paper: log_p={paper_v['log_p']:.2f}, intercept={paper_v['intercept']:.2f}, "
              f"q50={paper_v['q50']:.2f}]")
    print()
    print("NOTE: synthetic data is GMM-generated, so BIC and log p will favor GMM by")
    print("      construction. These numbers validate the framework's math correctness,")
    print("      not paper §3.4 GMM-best claim. Real empirical verification requires")
    print("      Stage 12 outputs (≈9-21 h GPU, see iter #86/96 lineage gap).")
    print()
    if n_params["GMM"] > 0 and n_params["Laplace"] > 0:
        # GMM advantage = how much better log-likelihood GMM needs to be
        # to overcome higher param count
        n = n_samples_total
        penalty_diff_gmm_minus_lap = (n_params["GMM"] - n_params["Laplace"]) * np.log(n)
        ll_diff = gmm_total - lap_total
        print(f"GMM advantage in BIC penalty: {penalty_diff_gmm_minus_lap:.2f} "
              f"(GMM needs log-L advantage ≥ this to win on BIC)")
        print(f"Observed log-L difference GMM - Laplace: {ll_diff:.2f}")
        if ll_diff > penalty_diff_gmm_minus_lap:
            print("→ GMM wins on BIC (better likelihood overcomes higher param count)")
        else:
            print("→ Laplace/Logistic wins on BIC (lower penalty wins)")

    # iter #187: held-out log-likelihood evaluation
    if args.held_out:
        print()
        print("=" * 70)
        print(f"iter #187: Held-out log-likelihood evaluation ({args.held_out_frac:.0%} holdout)")
        print("=" * 70)
        # Use last user's profile (representative) + all sentences collected
        if not user_profiles or not sentences_by_user:
            print("  No users/sentences available — skipping held-out eval")
        else:
            last_uid = list(user_profiles.keys())[-1]
            profile = user_profiles[last_uid]
            sents = sentences_by_user.get(last_uid, [])
            if len(sents) < 5:
                print(f"  User {last_uid} has only {len(sents)} sentences — skipping")
            else:
                held = held_out_loglik_evaluation(
                    profile, sents, held_out_frac=args.held_out_frac
                )
                if "error" in held:
                    print(f"  SKIP: {held['error']}")
                else:
                    print(f"  n_train={held['n_train']}, n_held_out={held['n_held_out']}")
                    print(f"  held-out GMM       log-lik = {held['held_out_gmm_loglik']:.2f}")
                    print(f"  held-out t         log-lik = {held['held_out_t_loglik']:.2f}")
                    print(f"  held-out Laplace   log-lik = {held['held_out_laplace_loglik']:.2f}")
                    print(f"  held-out Logistic  log-lik = {held['held_out_logistic_loglik']:.2f}")
                    print(f"  in-sample GMM      log-lik = {held['in_sample_gmm_loglik']:.2f}")
                    print(f"  GMM overfit ratio (in-sample / held-out) = {held['overfit_ratio_gmm']:.3f}")
                    if held['overfit_ratio_gmm'] > 1.5:
                        print(f"  ⚠ GMM overfit warning: ratio > 1.5 indicates in-sample advantage")
                    # Verdict: which prior wins on held-out?
                    priors_ho = {
                        "GMM": held['held_out_gmm_loglik'],
                        "t": held['held_out_t_loglik'],
                        "Laplace": held['held_out_laplace_loglik'],
                        "Logistic": held['held_out_logistic_loglik'],
                    }
                    best = max(priors_ho, key=priors_ho.get)
                    print(f"  → Held-out winner: {best} (not GMM-prior-bias-protected)")


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--3domain", action="store_true",
                         help="iter #182: run all 3 domains + report 3-domain mean")
    _args, _rest = _parser.parse_known_args()
    if _args.__dict__.get("3domain"):
        # Re-parse the original argparse with our preset
        sys.argv = [sys.argv[0], "--3domain", *_rest]
        # Easiest: just call main_3domain directly
        main_3domain()
    else:
        main()
