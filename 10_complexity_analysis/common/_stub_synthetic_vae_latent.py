#!/usr/bin/env python3
"""iter #179 — BIC/AIC stub demo with synthetic VAE latent data.

Per iter #71/86: Stage 12 outputs (user_profiles.jsonl + sentences.jsonl)
are missing, blocking full compute_prior_bic_aic.py evaluation. This script
generates a synthetic version that mimics the expected schema, so the
BIC/AIC computation framework can be validated end-to-end (param count,
BIC formula, output format) without GPU Stage 10/12 infra.

Synthetic data shape (matches train_vades_lite_sentence_latent_threshold.py
expected schema):
  - user_profiles.jsonl: one JSON object per line, each with keys
    {user_id, mu (list of K*D), logvar (list of K*D), mix_logits (list of K)}
  - sentences.jsonl: one JSON object per line, each with keys
    {user_id, latent (list of D)}

Generation:
  - N users = 100, d = 8 (LATENT_DIM), K = 2 (GMM_COMPONENTS)
  - Sample mu from N(0, 1), logvar from U(-1, 0), mix_logits from N(0, 1)
  - For each user, sample 10 latent points from their GMM
  - This is a "well-specified" GMM — true BIC advantage should be small
    (GMM vs Laplace/Logistic all fit approximately Gaussian)

Output:
  result/personal_query/12_complexity_analysis_clause_features/<cat>/synthetic_iter179_<tag>/{user_profiles,sentences}.jsonl

Usage:
  python3 _stub_synthetic_vae_latent.py --category Baby_Products
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path('/home/wlia0047/ar57/wenyu')
OUTPUT_ROOT = REPO_ROOT / 'result' / 'personal_query' / '12_complexity_analysis_clause_features'


def make_user_profile(user_id: str, mu: np.ndarray, logvar: np.ndarray,
                     mix_logits: np.ndarray) -> dict:
    """One line of user_profiles.jsonl."""
    return {
        'user_id': user_id,
        'mu': mu.tolist(),           # [K, D]
        'logvar': logvar.tolist(),   # [K, D]
        'mix_logits': mix_logits.tolist(),  # [K]
    }


def make_sentence(user_id: str, latent: np.ndarray) -> dict:
    """One line of sentences.jsonl."""
    return {
        'user_id': user_id,
        'latent': latent.tolist(),   # [D]
    }


def sample_gmm_latent(mu: np.ndarray, logvar: np.ndarray,
                      mix_logits: np.ndarray, n: int = 10,
                      rng: np.random.Generator = None) -> np.ndarray:
    """Sample n points from a user-specific diagonal GMM.

    mu: [K, D], logvar: [K, D], mix_logits: [K]
    Returns [n, D].
    """
    if rng is None:
        rng = np.random.default_rng(42)
    K, D = mu.shape
    # softmax over mix_logits
    logits = mix_logits - np.max(mix_logits)
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum()
    # sample component assignments
    comp_idx = rng.choice(K, size=n, p=weights)
    samples = np.zeros((n, D))
    for i in range(n):
        k = comp_idx[i]
        sigma = np.exp(0.5 * logvar[k])
        samples[i] = mu[k] + sigma * rng.standard_normal(D)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', required=True,
                        choices=['Baby_Products', 'Grocery_and_Gourmet_Food', 'Pet_Supplies'])
    parser.add_argument('--n-users', type=int, default=100,
                        help='number of users (default 100)')
    parser.add_argument('--n-sentences-per-user', type=int, default=10,
                        help='number of latent samples per user (default 10)')
    parser.add_argument('--latent-dim', type=int, default=8,
                        help='D (matches LATENT_DIM in train_vades_lite)')
    parser.add_argument('--n-components', type=int, default=2,
                        help='K (matches GMM_COMPONENTS)')
    parser.add_argument('--tag', type=str, default='synthetic_iter179',
                        help='output subdir tag')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    K, D = args.n_components, args.latent_dim

    out_dir = OUTPUT_ROOT / args.category / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic VAE latent data → {out_dir}")
    print(f"  N users: {args.n_users}, K={K}, D={D}, "
          f"n_sentences_per_user={args.n_sentences_per_user}")

    profile_path = out_dir / 'user_profiles.jsonl'
    sentence_path = out_dir / 'sentences.jsonl'

    n_profiles = 0
    n_sentences = 0
    with open(profile_path, 'w') as pf, open(sentence_path, 'w') as sf:
        for u in range(args.n_users):
            user_id = f'syn-user-{u:04d}'
            mu = rng.standard_normal((K, D)).astype(np.float64)
            logvar = rng.uniform(-1.0, 0.0, (K, D)).astype(np.float64)
            mix_logits = rng.standard_normal(K).astype(np.float64)
            pf.write(json.dumps(make_user_profile(user_id, mu, logvar, mix_logits)) + '\n')
            n_profiles += 1
            # sample latent points for this user
            samples = sample_gmm_latent(mu, logvar, mix_logits,
                                        n=args.n_sentences_per_user, rng=rng)
            for latent in samples:
                sf.write(json.dumps(make_sentence(user_id, latent)) + '\n')
                n_sentences += 1

    print(f"  Wrote {n_profiles} profiles → {profile_path}")
    print(f"  Wrote {n_sentences} sentences → {sentence_path}")
    print(f"  Total latent points for BIC: {n_sentences}")
    print()
    print("Now run compute_prior_bic_aic.py to validate the BIC/AIC framework:")
    print(f"  python3 /home/wlia0047/ar57/wenyu/PersoanlQuery/10_complexity_analysis/common/compute_prior_bic_aic.py "
          f"--category {args.category} --latent-dim {D} --n-components {K}")


if __name__ == '__main__':
    main()