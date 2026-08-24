"""Stage 7C: Robustness check of Stage 7B findings

Critical concern: Same user contributes up to 5 strict queries — these are NOT
independent observations (share user syntax habits). Stage 7B used only (1|asin)
which underestimates SE on D_syn.

This script runs 4 robustness settings on the same 768 queries:

1. **Main (replicate Stage 7B)**: (1|asin), all 768 queries
2. **+ user RE**: (1|asin) + (1|user) via vc_formula — controls user dependence
3. **user ≥2 strict**: filter out 25 single-query users → 743 queries
4. **drop tiny asin**: drop asin with only 4 unique queries (B000BNCA4K)

For each setting, report per-1-SD D_syn coef + n_tok coef + interactions.

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage7c_robust.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7c_robust.log 2>&1 &
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np


# === Paths (hardcoded) ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
RETRIEVAL_IN = SCRATCH / "stage7_retrieval_results.json"
REGEN_IN = SCRATCH / "stage7_regen.json"
OUT = REPO_ROOT / "result/gaussian_vades/syntax_subspace_stage7c_robust.json"

# === Constants ===
RETRIEVERS = ["minilm", "bm25"]
SEED = 2024


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fit_mixedlm(df, formula, group_col, vc_formula=None, label=""):
    """Fit MixedLM with optional variance components for nested RE."""
    import statsmodels.formula.api as smf
    try:
        kwargs = dict(
            data=df,
            groups=df[group_col],
            re_formula="1",
        )
        if vc_formula is not None:
            kwargs["vc_formula"] = vc_formula
        md = smf.mixedlm(formula, **kwargs)
        mdf = md.fit(reml=False)
        log(f"  {label}: converged={mdf.converged}, nobs={mdf.nobs}, "
            f"groups={df[group_col].nunique()}")
        return mdf, None
    except Exception as e:
        log(f"  {label}: FAILED {e!r}")
        return None, repr(e)


def extract_per_sd_report(mdf, n_sd_dsyn, n_sd_ntok):
    """Extract per-1-SD effect sizes from fitted model."""
    if mdf is None:
        return None
    out = {
        "params": {k: float(v) for k, v in mdf.params.items()},
        "pvalues": {k: float(v) for k, v in mdf.pvalues.items()},
        "ci": {k: [float(mdf.conf_int().loc[k, 0]), float(mdf.conf_int().loc[k, 1])]
               for k in mdf.params.index},
        "per_1SD": {},
    }
    out["per_1SD"]["D_syn_per_SD"] = {
        "coef_RR": out["params"].get("D_syn_c"),
        "p": out["pvalues"].get("D_syn_c"),
    }
    out["per_1SD"]["n_tok_per_SD"] = {
        "coef_RR": out["params"].get("n_tok_c"),
        "p": out["pvalues"].get("n_tok_c"),
    }
    out["per_1SD"]["D_syn_per_unit"] = {
        "coef_RR_per_unit": out["params"].get("D_syn_c") / n_sd_dsyn,
    }
    out["per_1SD"]["n_tok_per_token"] = {
        "coef_RR_per_token": out["params"].get("n_tok_c") / n_sd_ntok,
    }
    out["per_1SD"]["retriever_x_D_syn_per_SD"] = {
        "coef_RR": out["params"].get("C(retriever)[T.minilm]:D_syn_c"),
        "p": out["pvalues"].get("C(retriever)[T.minilm]:D_syn_c"),
    }
    out["per_1SD"]["retriever_x_n_tok_per_SD"] = {
        "coef_RR": out["params"].get("C(retriever)[T.minilm]:n_tok_c"),
        "p": out["pvalues"].get("C(retriever)[T.minilm]:n_tok_c"),
    }
    return out


def main():
    log("=== Stage 7C: Robustness check for Stage 7B ===")

    # === 1. Load Stage 7B D_syn ===
    log("\n=== 1. Loading Stage 7B data ===")
    syn_b = json.load(open(REPO_ROOT / "result/gaussian_vades/syntax_subspace_stage7b_syn.json"))
    # Stage 7B already computed D_syn and saved long-format rows in mixedlm_full
    # But we need raw rows. Re-load retrieval + regen + D_syn.
    retrieval = json.load(open(RETRIEVAL_IN, "r", encoding="utf-8"))["results"]
    regen = json.load(open(REGEN_IN, "r", encoding="utf-8"))["entries"]

    # Need D_syn for each retrieval entry. Reload via the same path as 7B.
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    from gaussian_vades import _syntax_subspace_prepare  # type: ignore
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=48, random_state=42)
    pca.fit(P["X_scaled"][train_idx])

    QUERY_FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"
    feat_map = {}
    import gzip, hashlib
    if QUERY_FEAT_CACHE.exists():
        with gzip.open(QUERY_FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]
    log(f"  feature cache: {len(feat_map)} entries")

    def fk(t):
        return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()

    # Project
    asin_query_z = {}
    miss = 0
    for e in retrieval:
        feats = feat_map.get(fk(e["query"]))
        if not feats:
            miss += 1
            continue
        vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
        vec_scaled = scaler.transform(vec[None, :])[0]
        z = pca.transform(vec_scaled[None, :])[0]
        asin_query_z[(e["asin"], e["query"])] = z
    log(f"  projections: {len(asin_query_z)}, missed: {miss}")

    asin_to_zs = collections.defaultdict(list)
    for (a, q), z in asin_query_z.items():
        asin_to_zs[a].append(z)
    asin_centroid = {a: np.stack(zs, axis=0).mean(axis=0) for a, zs in asin_to_zs.items()}

    d_syn_by_key = {(a, q): float(np.linalg.norm(z - asin_centroid[a]))
                    for (a, q), z in asin_query_z.items()}

    # Build long-format DF
    rows = []
    for e in retrieval:
        ds = d_syn_by_key.get((e["asin"], e["query"]))
        if ds is None:
            continue
        for r in RETRIEVERS:
            rows.append({
                "asin": e["asin"],
                "user_id": e["user_id"],
                "retriever": r,
                "RR": e[f"rr_{r}"],
                "n_tok": e["n_tok"],
                "D_syn": ds,
            })

    import pandas as pd
    df = pd.DataFrame(rows)
    df["D_syn_c"] = (df["D_syn"] - df["D_syn"].mean()) / df["D_syn"].std()
    df["n_tok_c"] = (df["n_tok"] - df["n_tok"].mean()) / df["n_tok"].std()

    n_sd_dsyn = float(df["D_syn"].std())
    n_sd_ntok = float(df["n_tok"].std())
    log(f"  rows: {len(df)}, asins: {df['asin'].nunique()}, users: {df['user_id'].nunique()}")
    log(f"  D_syn SD: {n_sd_dsyn:.4f}, n_tok SD: {n_sd_ntok:.4f}")

    # Counts per asin / user
    asin_queries = df[df["retriever"] == "minilm"].groupby("asin").size()
    user_queries = df[df["retriever"] == "minilm"].groupby("user_id").size()
    log(f"  queries per asin: median={asin_queries.median():.1f}, "
        f"min={asin_queries.min()}, max={asin_queries.max()}")
    log(f"  queries per user: median={user_queries.median():.1f}, "
        f"min={user_queries.min()}, max={user_queries.max()}")

    # === 2. Setting 1: Main (1|asin), all data (replicate 7B) ===
    log("\n=== 2. Setting 1: Main (1|asin), all data ===")
    formula = "RR ~ C(retriever) + D_syn_c + n_tok_c + C(retriever):D_syn_c + C(retriever):n_tok_c"
    m1, err1 = fit_mixedlm(df, formula, "asin", label="Setting1")
    s1 = extract_per_sd_report(m1, n_sd_dsyn, n_sd_ntok) if m1 else {"error": err1}
    if m1:
        for k in m1.params.index:
            log(f"    {k}: coef={m1.params[k]:+.5f}, p={m1.pvalues[k]:.4g}")

    # === 3. Setting 2: Add user RE via vc_formula ===
    log("\n=== 3. Setting 2: (1|asin) + user_variance_component ===")
    # statsmodels vc_formula needs to encode user_id as factor
    # For categorical VC, format: {"vc_name": "0+C(cat_col)"}
    m2, err2 = fit_mixedlm(
        df, formula, "asin",
        vc_formula={"user": "0+C(user_id)"},
        label="Setting2",
    )
    s2 = extract_per_sd_report(m2, n_sd_dsyn, n_sd_ntok) if m2 else {"error": err2}
    if m2:
        for k in m2.params.index:
            log(f"    {k}: coef={m2.params[k]:+.5f}, p={m2.pvalues[k]:.4g}")
        if hasattr(m2, 'vcp'):
            for vn, vcp in m2.vcp.items() if hasattr(m2, 'vcp') else []:
                log(f"    VC[{vn}]: {vcp}")

    # === 4. Setting 3: filter user ≥2 strict ===
    log("\n=== 4. Setting 3: filter users with ≥2 strict queries ===")
    users_ge2 = set(user_queries[user_queries >= 2].index)
    df_ge2 = df[df["user_id"].isin(users_ge2)].copy()
    # Recompute centering after filter (so D_syn_c/n_tok_c reflect this subset)
    df_ge2["D_syn_c"] = (df_ge2["D_syn"] - df_ge2["D_syn"].mean()) / df_ge2["D_syn"].std()
    df_ge2["n_tok_c"] = (df_ge2["n_tok"] - df_ge2["n_tok"].mean()) / df_ge2["n_tok"].std()
    log(f"  filtered rows: {len(df_ge2)} (was {len(df)}), users: {df_ge2['user_id'].nunique()} "
        f"(was {df['user_id'].nunique()})")
    n_sd_dsyn_ge2 = float(df_ge2["D_syn"].std())
    n_sd_ntok_ge2 = float(df_ge2["n_tok"].std())
    m3, err3 = fit_mixedlm(df_ge2, formula, "asin", label="Setting3")
    s3 = extract_per_sd_report(m3, n_sd_dsyn_ge2, n_sd_ntok_ge2) if m3 else {"error": err3}
    if m3:
        for k in m3.params.index:
            log(f"    {k}: coef={m3.params[k]:+.5f}, p={m3.pvalues[k]:.4g}")

    # === 5. Setting 4: drop tiny asin (B000BNCA4K with 4 queries) ===
    log("\n=== 5. Setting 4: drop tiny asin ===")
    # Find asins with <10 unique queries
    small_asins = set(asin_queries[asin_queries < 10].index)
    log(f"  dropping small asins ({len(small_asins)}): {small_asins}")
    df_no_tiny = df[~df["asin"].isin(small_asins)].copy()
    df_no_tiny["D_syn_c"] = (df_no_tiny["D_syn"] - df_no_tiny["D_syn"].mean()) / df_no_tiny["D_syn"].std()
    df_no_tiny["n_tok_c"] = (df_no_tiny["n_tok"] - df_no_tiny["n_tok"].mean()) / df_no_tiny["n_tok"].std()
    log(f"  filtered rows: {len(df_no_tiny)} (was {len(df)}), asins: {df_no_tiny['asin'].nunique()}")
    n_sd_dsyn_no = float(df_no_tiny["D_syn"].std())
    n_sd_ntok_no = float(df_no_tiny["n_tok"].std())
    m4, err4 = fit_mixedlm(df_no_tiny, formula, "asin", label="Setting4")
    s4 = extract_per_sd_report(m4, n_sd_dsyn_no, n_sd_ntok_no) if m4 else {"error": err4}
    if m4:
        for k in m4.params.index:
            log(f"    {k}: coef={m4.params[k]:+.5f}, p={m4.pvalues[k]:.4g}")

    # === 6. Per-retriever D_syn (per setting 1) — within-retriever robustness ===
    log("\n=== 6. Per-retriever D_syn (setting 1) ===")
    per_ret = {}
    for r in RETRIEVERS:
        df_r = df[df["retriever"] == r].copy()
        try:
            md = __import__('statsmodels.formula.api', fromlist=['mixedlm']).mixedlm(
                "RR ~ D_syn_c + n_tok_c",
                data=df_r,
                groups=df_r["asin"],
                re_formula="1",
            )
            mdf = md.fit(reml=False)
            per_ret[r] = {
                "D_syn_per_SD": {"coef": float(mdf.params["D_syn_c"]), "p": float(mdf.pvalues["D_syn_c"])},
                "n_tok_per_SD": {"coef": float(mdf.params["n_tok_c"]), "p": float(mdf.pvalues["n_tok_c"])},
            }
            log(f"  {r}: D_syn coef={mdf.params['D_syn_c']:+.5f}, p={mdf.pvalues['D_syn_c']:.4g}")
        except Exception as _e:
            per_ret[r] = {"error": repr(_e)}
            log(f"  {r}: failed ({_e!r})")

    # === 7. Side-by-side summary table ===
    log("\n=== 7. Side-by-side summary ===")
    log(f"{'Setting':<25} {'n_rows':>8} {'D_syn coef':>12} {'D_syn p':>10} {'n_tok coef':>12} {'n_tok p':>10} "
        f"{'R×D_syn coef':>14} {'R×D_syn p':>10} {'R×n_tok coef':>14} {'R×n_tok p':>10}")
    log("-" * 130)
    for label, mdf_obj, sd_d, sd_n, n_rows in [
        ("Setting1 (1|asin)", m1, n_sd_dsyn, n_sd_ntok, len(df)),
        ("Setting2 (+user RE)", m2, n_sd_dsyn, n_sd_ntok, len(df)),
        ("Setting3 (user ≥2)", m3, n_sd_dsyn_ge2, n_sd_ntok_ge2, len(df_ge2)),
        ("Setting4 (drop tiny asin)", m4, n_sd_dsyn_no, n_sd_ntok_no, len(df_no_tiny)),
    ]:
        if mdf_obj is None:
            log(f"{label:<25} FAIL")
            continue
        ds_c = mdf_obj.params.get("D_syn_c", float("nan"))
        ds_p = mdf_obj.pvalues.get("D_syn_c", float("nan"))
        nt_c = mdf_obj.params.get("n_tok_c", float("nan"))
        nt_p = mdf_obj.pvalues.get("n_tok_c", float("nan"))
        rxd_c = mdf_obj.params.get("C(retriever)[T.minilm]:D_syn_c", float("nan"))
        rxd_p = mdf_obj.pvalues.get("C(retriever)[T.minilm]:D_syn_c", float("nan"))
        rxn_c = mdf_obj.params.get("C(retriever)[T.minilm]:n_tok_c", float("nan"))
        rxn_p = mdf_obj.pvalues.get("C(retriever)[T.minilm]:n_tok_c", float("nan"))
        log(f"{label:<25} {n_rows:>8} {ds_c:>+12.5f} {ds_p:>10.4g} {nt_c:>+12.5f} {nt_p:>10.4g} "
            f"{rxd_c:>+14.5f} {rxd_p:>10.4g} {rxn_c:>+14.5f} {rxn_p:>10.4g}")

    # === 8. Conclusion ===
    log("\n=== 8. Conclusion ===")
    log("  Direction check (D_syn should be POSITIVE if robust):")
    for label, mdf_obj in [
        ("Setting1", m1), ("Setting2 (+user RE)", m2),
        ("Setting3 (user ≥2)", m3), ("Setting4 (drop tiny asin)", m4),
    ]:
        if mdf_obj is None:
            continue
        ds_c = mdf_obj.params.get("D_syn_c", float("nan"))
        ds_p = mdf_obj.pvalues.get("D_syn_c", float("nan"))
        direction = "POS ✓" if ds_c > 0 else "NEG ✗"
        sig = "SIG" if ds_p < 0.05 else "n.s."
        log(f"    {label}: coef={ds_c:+.5f}, p={ds_p:.4g} → {direction} {sig}")

    log("\n  Retriever × D_syn should be n.s. if universal syntax effect:")
    for label, mdf_obj in [
        ("Setting1", m1), ("Setting2 (+user RE)", m2),
        ("Setting3 (user ≥2)", m3), ("Setting4 (drop tiny asin)", m4),
    ]:
        if mdf_obj is None:
            continue
        rxd_p = mdf_obj.pvalues.get("C(retriever)[T.minilm]:D_syn_c", float("nan"))
        rxd_c = mdf_obj.params.get("C(retriever)[T.minilm]:D_syn_c", float("nan"))
        sig = "SIG" if rxd_p < 0.05 else "n.s."
        log(f"    {label}: coef={rxd_c:+.5f}, p={rxd_p:.4g} → {sig}")

    log("\n  Retriever × n_tok should be SIGNIFICANT (length is retriever-specific):")
    for label, mdf_obj in [
        ("Setting1", m1), ("Setting2 (+user RE)", m2),
        ("Setting3 (user ≥2)", m3), ("Setting4 (drop tiny asin)", m4),
    ]:
        if mdf_obj is None:
            continue
        rxn_p = mdf_obj.pvalues.get("C(retriever)[T.minilm]:n_tok_c", float("nan"))
        rxn_c = mdf_obj.params.get("C(retriever)[T.minilm]:n_tok_c", float("nan"))
        sig = "SIG" if rxn_p < 0.05 else "n.s."
        log(f"    {label}: coef={rxn_c:+.5f}, p={rxn_p:.4g} → {sig}")

    # === 9. Save JSON ===
    out = {
        "config": {
            "description": "Stage 7C: 4-setting robustness check of Stage 7B D_syn effect",
            "settings": [
                "Setting1: replicate Stage 7B (1|asin), all 768 queries",
                "Setting2: + user variance component (vc_formula)",
                "Setting3: filter user ≥2 strict queries",
                "Setting4: drop asins with <10 unique queries",
            ],
        },
        "data_summary": {
            "n_rows_total": len(df),
            "n_unique_asins": df["asin"].nunique(),
            "n_unique_users": df["user_id"].nunique(),
            "D_syn_SD": n_sd_dsyn,
            "n_tok_SD": n_sd_ntok,
            "queries_per_asin": {"median": float(asin_queries.median()), "min": int(asin_queries.min()), "max": int(asin_queries.max())},
            "queries_per_user": {"median": float(user_queries.median()), "min": int(user_queries.min()), "max": int(user_queries.max())},
            "users_with_1_query": int((user_queries == 1).sum()),
            "users_with_ge2_queries": int((user_queries >= 2).sum()),
            "small_asins_dropped_in_Setting4": list(small_asins),
        },
        "setting1_main": s1,
        "setting2_user_re": s2,
        "setting3_user_ge2": s3,
        "setting4_drop_tiny_asin": s4,
        "per_retriever_setting1": per_ret,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()