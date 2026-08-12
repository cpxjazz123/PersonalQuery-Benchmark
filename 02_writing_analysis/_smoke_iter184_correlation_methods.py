#!/usr/bin/env python3
"""Smoke test for iter #184 correlation methods.

Tests _pearson, _spearman, _kendall_tau, _bootstrap_ci, _permutation_test
on synthetic data:
  - perfect correlation: r ≈ 1.0
  - anti-correlation: r ≈ -1.0
  - zero correlation (independent noise): r ≈ 0, perm_p > 0.05

Validates the 4 functions added by iter #184 work correctly.
"""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Inline import to avoid loading real Stage 1/6 data
import importlib.util
spec = importlib.util.spec_from_file_location(
    'pilot', _SCRIPT_DIR / 'pilot_review_vs_query_correlation.py'
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_pearson = mod._pearson
_spearman = mod._spearman
_kendall_tau = mod._kendall_tau
_bootstrap_ci = mod._bootstrap_ci
_permutation_test = mod._permutation_test


def main():
    print("=" * 60)
    print("iter #184 correlation methods smoke test")
    print("=" * 60)

    # Case 1: perfect linear correlation
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    print(f"\n[Case 1] perfect linear: xs={xs}, ys={ys}")
    p = _pearson(xs, ys)
    s = _spearman(xs, ys)
    k = _kendall_tau(xs, ys)
    print(f"  pearson={p:.4f}, spearman={s:.4f}, kendall={k:.4f}")
    assert abs(p - 1.0) < 1e-9, f"perfect linear pearson must be 1.0, got {p}"
    assert abs(s - 1.0) < 1e-9, f"perfect linear spearman must be 1.0, got {s}"
    assert abs(k - 1.0) < 1e-9, f"perfect linear kendall must be 1.0, got {k}"

    # Case 2: anti-correlation
    ys_anti = [10.0, 8.0, 6.0, 4.0, 2.0]
    print(f"\n[Case 2] anti-correlation: ys={ys_anti}")
    p = _pearson(xs, ys_anti)
    s = _spearman(xs, ys_anti)
    k = _kendall_tau(xs, ys_anti)
    print(f"  pearson={p:.4f}, spearman={s:.4f}, kendall={k:.4f}")
    assert abs(p - (-1.0)) < 1e-9, f"anti-corr pearson must be -1.0, got {p}"
    assert abs(s - (-1.0)) < 1e-9, f"anti-corr spearman must be -1.0, got {s}"
    assert abs(k - (-1.0)) < 1e-9, f"anti-corr kendall must be -1.0, got {k}"

    # Case 3: independent noise (no correlation)
    # Use a fixed seed to ensure reproducibility
    import random
    rng = random.Random(123)
    xs_rand = [rng.gauss(0, 1) for _ in range(50)]
    ys_rand = [rng.gauss(0, 1) for _ in range(50)]  # independent draws
    print(f"\n[Case 3] independent noise (n=50)")
    p = _pearson(xs_rand, ys_rand)
    s = _spearman(xs_rand, ys_rand)
    k = _kendall_tau(xs_rand, ys_rand)
    print(f"  pearson={p:+.4f}, spearman={s:+.4f}, kendall={k:+.4f}")
    # Permutation p-value should be > 0.05 (cannot reject H0: ρ=0)
    pp = _permutation_test(xs_rand, ys_rand, _pearson, n_perm=2000)
    print(f"  perm_p(pearson)={pp:.3f} (expect > 0.05 → cannot reject H0)")
    assert pp > 0.05, f"independent noise: perm_p must be > 0.05, got {pp}"

    # Case 4: bootstrap CI on perfect correlation
    p, lo, hi = _bootstrap_ci(xs, ys, _pearson, n_boot=2000)
    print(f"\n[Case 4] bootstrap CI on perfect correlation")
    print(f"  pearson={p:.4f}, CI95=[{lo:.4f}, {hi:.4f}]")
    assert abs(p - 1.0) < 1e-9
    assert abs(lo - 1.0) < 1e-9, f"CI low must be 1.0 for perfect corr, got {lo}"
    assert abs(hi - 1.0) < 1e-9, f"CI high must be 1.0 for perfect corr, got {hi}"

    # Case 5: bootstrap CI on moderate correlation
    xs_m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ys_m = [2.5, 4.0, 5.8, 7.5, 9.0, 10.2, 12.5, 14.0, 15.8, 17.5]
    p, lo, hi = _bootstrap_ci(xs_m, ys_m, _pearson, n_boot=2000)
    print(f"\n[Case 5] bootstrap CI on moderate correlation (n=10)")
    print(f"  pearson={p:.4f}, CI95=[{lo:.4f}, {hi:.4f}]")
    assert lo < p < hi, f"CI must bracket point estimate, got lo={lo}, p={p}, hi={hi}"

    # Case 6: permutation test on strong correlation (must reject H0)
    # n=5 perfect corr → 5!=120 perms; some perms may yield |r|=1 by chance
    # so perm_p ≤ 0.05 is the correct threshold for "significant at α=0.05"
    pp = _permutation_test(xs, ys, _pearson, n_perm=2000)
    print(f"\n[Case 6] perm_p on perfect correlation (n=5) = {pp:.3f} (expect ≤ 0.05)")
    assert pp <= 0.05, f"perfect corr perm_p must be ≤ 0.05, got {pp}"

    print("\n" + "=" * 60)
    print("ALL CASES PASS")
    print("=" * 60)


if __name__ == '__main__':
    main()