#!/usr/bin/env python3
"""Smoke test for iter #187 held-out log-likelihood evaluation.

Tests:
  1. Helper functions: _gmm_loglik_against, _laplace_loglik_against,
     _logistic_loglik_against, _t_loglik_against produce valid (non-NaN) values.
  2. held_out_loglik_evaluation returns correct schema with 4 priors + overfit ratio.
  3. Held-out split ratio: n_train + n_held_out == n_total.
  4. Determinism: same seed → same held-out split.
  5. Overfit ratio sanity: GMM should have ratio >= 1 (in-sample >= held-out, since
     params are fit on full data).
"""

import ast
from pathlib import Path

_SCRIPT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/10_complexity_analysis/common/compute_prior_bic_aic.py")


def _import_module():
    # Use AST-derived extraction to avoid executing main() which calls load functions.
    import importlib.util
    spec = importlib.util.spec_from_file_location("cba187", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Skip the __main__ block by stubbing __name__
    saved_main = _SCRIPT.read_text(encoding="utf-8")
    src = saved_main.replace("if __name__ == \"__main__\":", "if False:")
    spec.loader.exec_module(mod)


def main():
    print("=" * 60)
    print("iter #187 held-out log-likelihood smoke test")
    print("=" * 60)

    # We use simpler approach: exec the helper functions in isolation
    # by extracting them from the source via ast.FunctionDef.
    src = _SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    helpers_to_test = [
        "_gmm_loglik_against",
        "_laplace_loglik_against",
        "_logistic_loglik_against",
        "_t_loglik_against",
        "held_out_loglik_evaluation",
    ]
    helper_src = "\n".join(
        ast.unparse(node) for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helpers_to_test
    )
    assert "def held_out_loglik_evaluation" in helper_src, "function not found"

    import numpy as np
    from scipy.special import gammaln

    ns = {"np": np, "gammaln": gammaln}
    exec(helper_src, ns)
    held_out_loglik_evaluation = ns["held_out_loglik_evaluation"]

    # Case 1: synthetic GMM-style data (well-specified GMM)
    # Profile = 2 components, mu1=[0,0], mu2=[3,3], logvar=log(1) = 0
    profile = {
        "mu": np.array([[0.0, 0.0], [3.0, 3.0]]),
        "logvar": np.array([[0.0, 0.0], [0.0, 0.0]]),
        "mix_logits": np.array([0.0, 0.0]),
    }
    # generate 100 sentences
    rng = np.random.default_rng(42)
    n_each = 50
    sents_c1 = rng.normal(0, 1, (n_each, 2)).tolist()
    sents_c2 = rng.normal(3, 1, (n_each, 2)).tolist()
    all_latents = sents_c1 + sents_c2
    sentences = [{"latent": latent} for latent in all_latents]

    print(f"\n[Case 1-2] synthetic GMM-sampled latent (n={len(sentences)})")
    print(f"  profile: 2 components, mu=[[0,0], [3,3]]")

    # Case 1: schema validity
    held = held_out_loglik_evaluation(profile, sentences, held_out_frac=0.2, rng_seed=42)
    print(f"\n[Case 1] result schema")
    print(f"  n_train = {held['n_train']}, n_held_out = {held['n_held_out']}")
    print(f"  held_out_gmm_loglik = {held['held_out_gmm_loglik']:.2f}")
    print(f"  held_out_t_loglik = {held['held_out_t_loglik']:.2f}")
    print(f"  held_out_laplace_loglik = {held['held_out_laplace_loglik']:.2f}")
    print(f"  held_out_logistic_loglik = {held['held_out_logistic_loglik']:.2f}")
    print(f"  in_sample_gmm_loglik = {held['in_sample_gmm_loglik']:.2f}")
    print(f"  overfit_ratio_gmm = {held['overfit_ratio_gmm']:.3f}")
    assert "held_out_gmm_loglik" in held
    assert "held_out_t_loglik" in held
    assert "held_out_laplace_loglik" in held
    assert "held_out_logistic_loglik" in held
    assert "in_sample_gmm_loglik" in held
    assert "overfit_ratio_gmm" in held
    assert all(isinstance(held[k], float) for k in (
        "held_out_gmm_loglik", "held_out_t_loglik", "held_out_laplace_loglik",
        "held_out_logistic_loglik", "in_sample_gmm_loglik", "overfit_ratio_gmm",
    ))

    # Case 2: train + held_out = total
    print(f"\n[Case 2] split invariant: n_train + n_held_out == n_total")
    assert held["n_train"] + held["n_held_out"] == 100, \
        f"split mismatch: {held['n_train']} + {held['n_held_out']} != 100"
    # 0.2 * 100 = 20
    assert held["n_held_out"] == 20, f"expected 20 held-out, got {held['n_held_out']}"
    print(f"  ✓ n_train={held['n_train']} + n_held_out={held['n_held_out']} = 100")

    # Case 3: determinism — same seed should produce same split
    print(f"\n[Case 3] determinism (seed=42)")
    held_a = held_out_loglik_evaluation(profile, sentences, held_out_frac=0.2, rng_seed=42)
    held_b = held_out_loglik_evaluation(profile, sentences, held_out_frac=0.2, rng_seed=42)
    assert held_a["held_out_gmm_loglik"] == held_b["held_out_gmm_loglik"]
    assert held_a["held_out_laplace_loglik"] == held_b["held_out_laplace_loglik"]
    print(f"  ✓ Two runs with seed=42 produce identical values")
    print(f"    GMM held-out: {held_a['held_out_gmm_loglik']:.4f}")

    # Case 4: different seed → different split
    held_diff = held_out_loglik_evaluation(profile, sentences, held_out_frac=0.2, rng_seed=99)
    assert held_a["held_out_gmm_loglik"] != held_diff["held_out_gmm_loglik"], \
        "Different seed should give different GMM held-out value"
    print(f"\n[Case 4] seed sensitivity")
    print(f"  seed=42 → GMM held-out = {held_a['held_out_gmm_loglik']:.4f}")
    print(f"  seed=99 → GMM held-out = {held_diff['held_out_gmm_loglik']:.4f}")
    print(f"  ✓ Different seeds produce different splits")

    # Case 5: overfit ratio sanity
    print(f"\n[Case 5] GMM overfit ratio sanity")
    ratio = held["overfit_ratio_gmm"]
    print(f"  ratio (in-sample / held-out) = {ratio:.3f}")
    # ratio >= 1 because profile is fit on full data (incl held_out)
    # So in-sample log-lik should be larger than held-out log-lik
    assert ratio >= 1.0, f"GMM overfit ratio should be >= 1.0, got {ratio}"
    # and ratio should be finite / not crazy (5x ratio on small n=100 is plausible)
    assert 1.0 <= ratio < 10.0, f"GMM overfit ratio out of expected range: {ratio}"
    print(f"  ✓ ratio in [1.0, 10.0) — params favor in-sample as expected")

    # Case 6: held-out winner may differ from in-sample
    priors_ho = {
        "GMM": held["held_out_gmm_loglik"],
        "t": held["held_out_t_loglik"],
        "Laplace": held["held_out_laplace_loglik"],
        "Logistic": held["held_out_logistic_loglik"],
    }
    best_ho = max(priors_ho, key=priors_ho.get)
    print(f"\n[Case 6] held-out winner")
    print(f"  priors on held-out: {priors_ho}")
    print(f"  winner = {best_ho} (synthetic GMM data, so GMM should typically win)")
    # Synthetic is well-specified GMM, so GMM should win on held-out too
    assert best_ho == "GMM", f"synthetic GMM data: GMM should win held-out, got {best_ho}"

    # Case 7: too few sentences edge case
    print(f"\n[Case 7] too few sentences edge case (n=3 < 5)")
    held_tiny = held_out_loglik_evaluation(profile, sentences[:3], held_out_frac=0.2)
    print(f"  result = {held_tiny}")
    assert "error" in held_tiny, "should produce error for n_total < 5"
    print(f"  ✓ error returned for n<5: {held_tiny['error']}")

    print("\n" + "=" * 60)
    print("ALL CASES PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()