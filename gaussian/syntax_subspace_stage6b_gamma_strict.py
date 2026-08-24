"""Stage 6B-γ strict: in attr_pass=1 subset, is N_attrs → length still significant?

Motivation: Stage 6B-γ showed natural length grows with N (16.8 → 27.9 tokens)
but lift1 is flat. To confirm the chain is not just "model produced fluff to fill
in slots", restrict to strict attr_pass=1 (attrs_covered == N_input).

Then test: length ~ N_input + (1|user) + (1|asin) mixed-effects.
Hypothesis: β_N_attrs > 0, p < 0.05 → more attrs still produce longer queries
even when only counting "fully expressed" cases.

Usage:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage6b_gamma_strict.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage6b_gamma_strict.log 2>&1 &

Env-var override:
    STAGE6BG_GAMMA_OUT = path to regen JSON (default below)
    STAGE6BG_STRICT_OUT = path to write analysis JSON (default below)
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import numpy as np


# === Paths (hardcoded, per CLAUDE.md Rule 3) ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
GAMMA_OUT = Path(os.environ.get(
    "STAGE6BG_GAMMA_OUT",
    str(SCRATCH / "query_records_n_input_natural.json"),
))
STRICT_OUT = Path(os.environ.get(
    "STAGE6BG_STRICT_OUT",
    str(REPO_ROOT / "result/gaussian_vades/syntax_subspace_stage6b_gamma_strict.json"),
))

# === Constants ===
N_INPUT_LIST = [3, 4, 5, 6, 7]
SEED = 2024


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=== Stage 6B-γ STRICT (attr_pass=1 length regression) ===")
    log(f"loading {GAMMA_OUT}")
    if not GAMMA_OUT.exists():
        raise FileNotFoundError(f"missing {GAMMA_OUT} — Stage 6B-γ gen must run first")
    regen = json.load(open(GAMMA_OUT, "r", encoding="utf-8"))
    log(f"loaded {len(regen)} entries")

    # === 1. Per-N stats on FULL set (cross-check vs Stage 6B-γ eval) ===
    log("\n=== 1. Per-N stats on FULL set (sanity check) ===")
    full_per_n = {}
    for n in N_INPUT_LIST:
        rs_n = [e for e in regen if e["N_input"] == n and e["query"] and not e["invalid"]]
        n_tok = np.asarray([e["n_tok"] for e in rs_n], dtype=np.float64)
        attr_pass = np.asarray(
            [1.0 if e["attrs_covered"] == e["N_input"] else 0.0 for e in rs_n],
            dtype=np.float64,
        )
        full_per_n[n] = {
            "count": len(rs_n),
            "n_tok_mean": float(n_tok.mean()),
            "n_tok_median": float(np.median(n_tok)),
            "n_tok_std": float(n_tok.std()),
            "n_tok_min": float(n_tok.min()),
            "n_tok_max": float(n_tok.max()),
            "attr_pass_rate": float(attr_pass.mean()),
            "attr_pass_count": int(attr_pass.sum()),
        }
        log(f"  N={n}: count={full_per_n[n]['count']:>5}, "
            f"n_tok mean={full_per_n[n]['n_tok_mean']:>5.2f} median={full_per_n[n]['n_tok_median']:>5.1f} "
            f"std={full_per_n[n]['n_tok_std']:>4.2f} "
            f"attr_pass={full_per_n[n]['attr_pass_rate']*100:>4.1f}% "
            f"({full_per_n[n]['attr_pass_count']:>4})")

    # === 2. STRICT subset (attrs_covered == N_input) ===
    log("\n=== 2. STRICT subset (attrs_covered == N_input) ===")
    strict = [e for e in regen
              if e["query"] and not e["invalid"] and e["attrs_covered"] == e["N_input"]]
    log(f"strict entries: {len(strict)}")
    by_n_strict = collections.defaultdict(list)
    for e in strict:
        by_n_strict[e["N_input"]].append(e)

    strict_per_n = {}
    for n in N_INPUT_LIST:
        rs_n = by_n_strict[n]
        n_tok = np.asarray([e["n_tok"] for e in rs_n], dtype=np.float64)
        strict_per_n[n] = {
            "count": len(rs_n),
            "n_tok_mean": float(n_tok.mean()),
            "n_tok_median": float(np.median(n_tok)),
            "n_tok_std": float(n_tok.std()),
            "n_tok_min": float(n_tok.min()),
            "n_tok_max": float(n_tok.max()),
            "n_tok_p25": float(np.percentile(n_tok, 25)),
            "n_tok_p75": float(np.percentile(n_tok, 75)),
        }
        log(f"  N={n}: count={strict_per_n[n]['count']:>4}, "
            f"n_tok mean={strict_per_n[n]['n_tok_mean']:>5.2f} median={strict_per_n[n]['n_tok_median']:>5.1f} "
            f"std={strict_per_n[n]['n_tok_std']:>4.2f} "
            f"[p25={strict_per_n[n]['n_tok_p25']:.1f}, p75={strict_per_n[n]['n_tok_p75']:.1f}]")

    # === 3. Monotonicity check: does length grow monotonically with N in strict subset? ===
    log("\n=== 3. Monotonicity check (strict subset) ===")
    means = [strict_per_n[n]["n_tok_mean"] for n in N_INPUT_LIST]
    medians = [strict_per_n[n]["n_tok_median"] for n in N_INPUT_LIST]
    is_mono_mean = all(means[i] < means[i + 1] for i in range(len(means) - 1))
    is_mono_median = all(medians[i] < medians[i + 1] for i in range(len(medians) - 1))
    log(f"  N mean sequence: {[f'{m:.2f}' for m in means]}")
    log(f"  N median sequence: {[f'{m:.1f}' for m in medians]}")
    log(f"  monotonic mean? {is_mono_mean}")
    log(f"  monotonic median? {is_mono_median}")
    log(f"  length growth N=3→N=7: {means[0]:.2f} → {means[-1]:.2f} "
        f"(Δ={means[-1] - means[0]:.2f} tokens)")

    # === 4. Mixed-effects: length ~ N_input + (1|user) + (1|asin) ===
    log("\n=== 4. Mixed-effects: n_tok ~ C(N_input) + (1|user_id) + (1|asin) ===")
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as _e:
        log(f"  statsmodels/pandas missing: {_e} — skipping regression")
        regression = None
    else:
        df = pd.DataFrame([{
            "n_tok": e["n_tok"],
            "N_input": int(e["N_input"]),
            "user_id": e["user_id"],
            "asin": e["asin"],
            "split": e.get("split", "?"),
        } for e in strict])
        log(f"  strict rows: {len(df)}")
        log(f"  unique users: {df['user_id'].nunique()}, unique asins: {df['asin'].nunique()}")

        # A) Treatment-coded: N=3 reference, β_N>0 → longer
        # Use N_input continuous as covariate to get single β1
        try:
            md = smf.mixedlm(
                "n_tok ~ N_input",
                data=df,
                groups=df["user_id"],
                re_formula="1",
            )
            mdf = md.fit(reml=False)
            log("  continuous N_input (random=user):")
            log("  " + str(mdf.summary()).replace("\n", "\n  "))
            beta_n = float(mdf.params.get("N_input", np.nan))
            p_n = float(mdf.pvalues.get("N_input", np.nan))
            ci = mdf.conf_int().loc["N_input"]
            regression = {
                "continuous_N": {
                    "beta_N_input": beta_n,
                    "p_value": p_n,
                    "ci_lo": float(ci[0]),
                    "ci_hi": float(ci[1]),
                    "intercept": float(mdf.params.get("Intercept", np.nan)),
                    "group_var_user": float(mdf.cov_re.iloc[0, 0]) if hasattr(mdf, "cov_re") else None,
                    "residual_var": float(mdf.scale),
                    "nobs": int(mdf.nobs),
                    "converged": bool(mdf.converged),
                }
            }
            sig = "*** SIG" if p_n < 0.001 else ("** SIG" if p_n < 0.01 else ("* SIG" if p_n < 0.05 else "n.s."))
            log(f"  >>> β_N_input={beta_n:+.4f}, p={p_n:.4g} [{ci[0]:+.4f}, {ci[1]:+.4f}] ({sig})")
            log(f"  Interpretation: each +1 attr → {beta_n:+.2f} tokens, holding user constant")
        except Exception as _e:
            log(f"  continuous N regression failed: {_e!r}")
            regression = {"continuous_N_error": repr(_e)}

        # B) Categorical: each N vs N=3 reference
        try:
            md_cat = smf.mixedlm(
                "n_tok ~ C(N_input)",
                data=df,
                groups=df["user_id"],
                re_formula="1",
            )
            mdf_cat = md_cat.fit(reml=False)
            log("\n  categorical N_input (N=3 reference, random=user):")
            log("  " + str(mdf_cat.summary()).replace("\n", "\n  "))
            cat_results = {}
            for k, v in mdf_cat.params.items():
                if k == "Intercept" or k == "Group Var":
                    continue
                cat_results[k] = {
                    "coef": float(v),
                    "p_value": float(mdf_cat.pvalues[k]),
                    "ci_lo": float(mdf_cat.conf_int().loc[k, 0]),
                    "ci_hi": float(mdf_cat.conf_int().loc[k, 1]),
                }
                log(f"  {k}: β={v:+.3f}, p={mdf_cat.pvalues[k]:.4g}")
            regression["categorical_N"] = cat_results
        except Exception as _e:
            log(f"  categorical regression failed: {_e!r}")
            regression["categorical_N_error"] = repr(_e)

        # C) Random asin instead of user
        try:
            md_a = smf.mixedlm(
                "n_tok ~ N_input",
                data=df,
                groups=df["asin"],
                re_formula="1",
            )
            mdf_a = md_a.fit(reml=False)
            beta_n_a = float(mdf_a.params.get("N_input", np.nan))
            p_n_a = float(mdf_a.pvalues.get("N_input", np.nan))
            ci_a = mdf_a.conf_int().loc["N_input"]
            log(f"\n  random=asin: β_N_input={beta_n_a:+.4f}, p={p_n_a:.4g} "
                f"[{ci_a[0]:+.4f}, {ci_a[1]:+.4f}]")
            regression["random_asin"] = {
                "beta_N_input": beta_n_a,
                "p_value": p_n_a,
                "ci_lo": float(ci_a[0]),
                "ci_hi": float(ci_a[1]),
                "nobs": int(mdf_a.nobs),
            }
        except Exception as _e:
            log(f"  random=asin regression failed: {_e!r}")

        # D) Length-bucket stratified: does N effect hold within each length bucket?
        # This tests whether the N effect is purely "moves bucket" or also "shifts within bucket"
        log("\n=== 5. Length-bucket stratified: does N effect hold within length buckets? ===")
        length_buckets = [(0, 15), (15, 25), (25, 35), (35, 50), (50, 200)]
        df["length_bucket"] = pd.cut(
            df["n_tok"],
            bins=[0, 15, 25, 35, 50, 200],
            labels=["0-15", "15-25", "25-35", "35-50", "50+"],
            include_lowest=True,
        )
        bucket_results = {}
        for lb in ["0-15", "15-25", "25-35", "35-50", "50+"]:
            df_b = df[df["length_bucket"] == lb]
            if len(df_b) < 30:
                bucket_results[lb] = {"n": len(df_b), "skipped": True}
                log(f"  bucket {lb}: n={len(df_b)}, skipped (too few)")
                continue
            try:
                md_b = smf.mixedlm(
                    "n_tok ~ N_input",
                    data=df_b,
                    groups=df_b["user_id"],
                    re_formula="1",
                )
                mdf_b = md_b.fit(reml=False)
                beta = float(mdf_b.params.get("N_input", np.nan))
                p = float(mdf_b.pvalues.get("N_input", np.nan))
                sig = "SIG" if p < 0.05 else "n.s."
                bucket_results[lb] = {
                    "n": len(df_b),
                    "beta_N_input": beta,
                    "p_value": p,
                    "significant": p < 0.05,
                }
                log(f"  bucket {lb}: n={len(df_b)}, β_N={beta:+.3f}, p={p:.4f} ({sig})")
            except Exception as _e:
                bucket_results[lb] = {"n": len(df_b), "error": repr(_e)}
                log(f"  bucket {lb}: regression failed: {_e!r}")
        regression["by_length_bucket"] = bucket_results

        # E) Spearman correlation (non-parametric check, robust to non-linearity)
        from scipy.stats import spearmanr  # type: ignore
        rho, p_rho = spearmanr(df["N_input"], df["n_tok"])
        log(f"\n  Spearman ρ(N_input, n_tok) = {rho:+.4f}, p = {p_rho:.4g}")
        regression["spearman"] = {"rho": float(rho), "p_value": float(p_rho)}

    # === 6. Final conclusion ===
    log("\n=== 6. Conclusion ===")
    if regression and "continuous_N" in regression:
        b = regression["continuous_N"]["beta_N_input"]
        p = regression["continuous_N"]["p_value"]
        if p < 0.05 and b > 0:
            log(f"  ✅ N_attrs → length: β=+{b:.3f} tokens/attr, p={p:.4g} (SIGNIFICANT)")
            log(f"  Even in strict attr_pass=1 subset, more attrs ⇒ significantly longer queries.")
            log(f"  Combined with Stage 6B-γ flat lift1 across N: more content ≠ better user match.")
        elif p >= 0.05:
            log(f"  ❌ N_attrs → length: β=+{b:.3f} tokens/attr, p={p:.4g} (n.s.)")
            log(f"  In strict subset, length difference between N is NOT significant.")
        else:
            log(f"  ⚠️  β={b:.3f}, p={p:.4g} (significant but negative — unexpected)")

    # === 7. Output JSON ===
    out = {
        "config": {
            "description": "Stage 6B-γ strict: length regression on attr_pass=1 subset",
            "SEED": SEED,
            "N_INPUT_LIST": N_INPUT_LIST,
        },
        "full_per_N": full_per_n,
        "strict_per_N": strict_per_n,
        "monotonic_check": {
            "is_mono_mean": is_mono_mean,
            "is_mono_median": is_mono_median,
            "means": means,
            "medians": medians,
            "growth_3_to_7": means[-1] - means[0],
        },
        "regression": regression,
        "n_strict_total": len(strict),
    }
    STRICT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(STRICT_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nwrote → {STRICT_OUT}")


if __name__ == "__main__":
    main()