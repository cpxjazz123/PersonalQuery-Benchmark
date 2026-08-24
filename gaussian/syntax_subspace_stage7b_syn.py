"""Stage 7B: PCA48 D_syn × Retriever interaction — true syntax mechanism

Stage 7A used n_tok as proxy for syntactic complexity. Stage 7B does the
proper job: extract PCA48 features for stage7 queries, compute item-centered
D_syn(q, p) = ||z_q - z̄_p||, run mixed-effects:

    RR ~ Retriever + D_syn + n_tok + Retriever×D_syn + Retriever×n_tok + (1|asin)

The key test: Retriever × D_syn coef — different architectures respond
differently to syntactic variation (controlling for length).

Setup reuses Stage 6B-γ's frozen PCA48 (StandardScaler + 48d PCA fit on
sentence-level features from 9982 rewrites).

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage7b_syn.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7b_syn.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


# === Paths (hardcoded) ===
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
RETRIEVAL_IN = SCRATCH / "stage7_retrieval_results.json"
REGEN_IN = SCRATCH / "stage7_regen.json"
QUERY_FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"
OUT = REPO_ROOT / "result/gaussian_vades/syntax_subspace_stage7b_syn.json"

# === Constants ===
RETRIEVERS = ["minilm", "bm25"]
SEED = 2024
PCA_DIM = 48


def log(msg: str) -> None:
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=== Stage 7B: PCA48 D_syn × Retriever ===")

    # === 1. Load frozen PCA48 (scaler + PCA components) ===
    log("\n=== 1. Loading frozen PCA48 from Stage 5B ===")
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "gaussian"))
    from gaussian_vades import _syntax_subspace_prepare  # type: ignore

    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]  # 182d, alphabetical
    X_pool = P["X"]  # [N_pool, 182] sentences from 9982 rewrites
    train_idx = P["train_idx"]  # 5000 indices used to fit scaler
    log(f"  pool: {X_pool.shape}, scaler mean shape: {scaler.mean_.shape}")
    log(f"  feature count: {len(fnames)}")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 fit on {len(train_idx)} train idx, EV={pca.explained_variance_ratio_.sum():.4f}")

    # === 2. Extract 182d features for stage7 queries ===
    log("\n=== 2. Extracting 182d features for stage7 queries ===")
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    from main import per_sentence_features_v2  # type: ignore

    # Load regen to get query texts (raw, not strict-filtered yet — we'll filter later)
    regen = json.load(open(REGEN_IN, "r", encoding="utf-8"))["entries"]
    log(f"  regen entries: {len(regen)}")

    def feat_key(t: str) -> str:
        return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()

    # Load cache
    feat_map: dict = {}
    if QUERY_FEAT_CACHE.exists():
        with gzip.open(QUERY_FEAT_CACHE, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rec = json.loads(line)
                feat_map[rec["k"]] = rec["v"]
        log(f"  cache loaded: {len(feat_map)} features")

    # Find unique queries needing features
    unique_texts = sorted({e["query"] for e in regen if e["query"]})
    new_texts = [t for t in unique_texts if feat_key(t) not in feat_map]
    log(f"  unique queries: {len(unique_texts)}, new: {len(new_texts)}")

    if new_texts:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        log(f"  extracting features for {len(new_texts)} new queries via spaCy...")
        for i, doc in enumerate(nlp.pipe(new_texts, batch_size=128, n_process=1)):
            t = new_texts[i]
            k = feat_key(t)
            try:
                feats = per_sentence_features_v2(doc)
                feats = feats if feats is not None else {}
            except Exception as _e:
                feats = {}
            numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
            # Filter to fnames (same set as PCA training)
            filtered = {n: numeric.get(n, 0.0) for n in fnames}
            feat_map[k] = filtered
            if (i + 1) % 200 == 0:
                log(f"    {i + 1}/{len(new_texts)}")
        # Save cache
        QUERY_FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(QUERY_FEAT_CACHE, "wt", encoding="utf-8") as f:
            f.write("#META {\"version\": \"v1\", \"n_entries\": " + str(len(feat_map)) + "}\n")
            for k, v in feat_map.items():
                f.write(json.dumps({"k": k, "v": v}) + "\n")
        log(f"  saved cache: {len(feat_map)} entries")

    # === 3. Project to PCA48 + compute item-centered D_syn ===
    log("\n=== 3. Project to PCA48 + compute D_syn per (asin, query) ===")
    # Build per-(asin, query) PCA48 vectors
    asin_query_z = {}  # (asin, query_text) → z (48,)
    miss_feat = 0
    for e in regen:
        if not e["query"]:
            continue
        k = feat_key(e["query"])
        feats = feat_map.get(k)
        if not feats:
            miss_feat += 1
            continue
        vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
        vec_scaled = scaler.transform(vec[None, :])[0]
        z = pca.transform(vec_scaled[None, :])[0]
        asin_query_z[(e["asin"], e["query"])] = z
    log(f"  projections: {len(asin_query_z)}, missed (no feat): {miss_feat}")

    # Per-asin centroid
    asin_to_zs = collections.defaultdict(list)
    for (a, q), z in asin_query_z.items():
        asin_to_zs[a].append(z)
    asin_centroid = {a: np.stack(zs, axis=0).mean(axis=0) for a, zs in asin_to_zs.items()}
    log(f"  asins with centroid: {len(asin_centroid)}")

    # D_syn(q, p) = ||z_q - z̄_p||
    d_syn_by_key = {}
    for (a, q), z in asin_query_z.items():
        d_syn_by_key[(a, q)] = float(np.linalg.norm(z - asin_centroid[a]))

    # === 4. Merge with retrieval results ===
    log("\n=== 4. Merging with retrieval results ===")
    retrieval = json.load(open(RETRIEVAL_IN, "r", encoding="utf-8"))["results"]
    log(f"  retrieval entries: {len(retrieval)}")

    rows = []  # long-format: one row per (entry, retriever)
    n_no_dsyn = 0
    for e in retrieval:
        d_syn = d_syn_by_key.get((e["asin"], e["query"]))
        if d_syn is None:
            n_no_dsyn += 1
            continue
        for r in RETRIEVERS:
            rows.append({
                "asin": e["asin"],
                "user_id": e["user_id"],
                "retriever": r,
                "RR": e[f"rr_{r}"],
                "n_tok": e["n_tok"],
                "D_syn": d_syn,
            })
    log(f"  long-format rows: {len(rows)}, skipped (no D_syn): {n_no_dsyn}")
    log(f"  unique asins: {len({r['asin'] for r in rows})}, unique users: {len({r['user_id'] for r in rows})}")

    # === 5. Diagnostics: D_syn distribution ===
    log("\n=== 5. D_syn diagnostics ===")
    d_syns = np.asarray([r["D_syn"] for r in rows])
    n_toks = np.asarray([r["n_tok"] for r in rows])
    log(f"  D_syn: min={d_syns.min():.3f}, mean={d_syns.mean():.3f}, "
        f"median={np.median(d_syns):.3f}, max={d_syns.max():.3f}, std={d_syns.std():.3f}")
    log(f"  n_tok: min={n_toks.min()}, mean={n_toks.mean():.2f}, "
        f"median={np.median(n_toks):.1f}, max={n_toks.max()}")
    from scipy.stats import spearmanr
    rho_l, p_l = spearmanr(d_syns, n_toks)
    log(f"  Spearman ρ(D_syn, n_tok) = {rho_l:+.4f}, p = {p_l:.4g}")

    # === 6. Per-asin D_syn breakdown (sanity check) ===
    log("\n=== 6. Per-asin D_syn summary (top-5 by count) ===")
    asin_dsyn = collections.defaultdict(list)
    for r in rows:
        asin_dsyn[r["asin"]].append(r["D_syn"])
    asin_table = sorted(
        [{"asin": a, "n": len(ds), "mean": float(np.mean(ds)),
          "std": float(np.std(ds)), "max": float(np.max(ds))}
         for a, ds in asin_dsyn.items()],
        key=lambda x: -x["n"],
    )[:5]
    for row in asin_table:
        log(f"  {row['asin']} n={row['n']}: D_syn mean={row['mean']:.3f} std={row['std']:.3f} max={row['max']:.3f}")

    # === 7. Mixed-effects: RR ~ Retriever + D_syn + n_tok + Retriever × D_syn + Retriever × n_tok + (1|asin) ===
    log("\n=== 7. Mixed-effects: RR ~ Retriever + D_syn + n_tok + interactions + (1|asin) ===")
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as _e:
        log(f"  statsmodels/pandas missing: {_e}")
        mixedlm_results = {"error": repr(_e)}
    else:
        df = pd.DataFrame(rows)
        log(f"  rows: {len(df)}, asins: {df['asin'].nunique()}")

        # Centering D_syn within asin so it's deviation-from-item-centroid (already)
        # but also rescale for numerical stability
        df["D_syn_c"] = (df["D_syn"] - df["D_syn"].mean()) / df["D_syn"].std()
        df["n_tok_c"] = (df["n_tok"] - df["n_tok"].mean()) / df["n_tok"].std()

        try:
            md = smf.mixedlm(
                "RR ~ C(retriever) + D_syn_c + n_tok_c + C(retriever):D_syn_c + C(retriever):n_tok_c",
                data=df,
                groups=df["asin"],
                re_formula="1",
            )
            mdf = md.fit(reml=False)
            log("  MixedLM full model:")
            log("  " + str(mdf.summary()).replace("\n", "\n  "))
            mixedlm_results = {
                "nobs": int(mdf.nobs),
                "converged": bool(mdf.converged),
                "params": {k: float(v) for k, v in mdf.params.items()},
                "pvalues": {k: float(v) for k, v in mdf.pvalues.items()},
                "ci": {k: [float(mdf.conf_int().loc[k, 0]), float(mdf.conf_int().loc[k, 1])]
                       for k in mdf.params.index},
            }
            log("  Highlighted effects:")
            for k in mdf.params.index:
                p = mdf.pvalues[k]
                coef = mdf.params[k]
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
                log(f"    {k}: coef={coef:+.4f}, p={p:.4g} ({sig})")

            # Critical: separate Retriever × D_syn test
            key_term = "C(retriever)[T.minilm]:D_syn_c"
            if key_term in mdf.pvalues:
                p_crit = float(mdf.pvalues[key_term])
                coef_crit = float(mdf.params[key_term])
                log(f"\n  >>> KEY: Retriever×D_syn interaction: coef={coef_crit:+.4f}, p={p_crit:.4g}")
                if p_crit < 0.05:
                    log(f"  >>> SIG: BM25 and MiniLM respond differently to syntactic distance")
                else:
                    log(f"  >>> n.s.: No detectable retriever-level difference in syntactic sensitivity")

        except Exception as _e:
            log(f"  mixedlm failed: {_e!r}")
            mixedlm_results = {"error": repr(_e)}

    # === 8. Reduced model: drop interactions, see what D_syn alone does ===
    log("\n=== 8. Reduced model: RR ~ Retriever + D_syn + n_tok + (1|asin) ===")
    reduced_results = {}
    try:
        md_r = smf.mixedlm(
            "RR ~ C(retriever) + D_syn_c + n_tok_c",
            data=df,
            groups=df["asin"],
            re_formula="1",
        )
        mdf_r = md_r.fit(reml=False)
        reduced_results = {
            "params": {k: float(v) for k, v in mdf_r.params.items()},
            "pvalues": {k: float(v) for k, v in mdf_r.pvalues.items()},
            "converged": bool(mdf_r.converged),
            "nobs": int(mdf_r.nobs),
        }
        for k in mdf_r.params.index:
            p = mdf_r.pvalues[k]
            coef = mdf_r.params[k]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            log(f"    {k}: coef={coef:+.4f}, p={p:.4g} ({sig})")
    except Exception as _e:
        log(f"  reduced model failed: {_e!r}")
        reduced_results = {"error": repr(_e)}

    # === 9. Per-retriever D_syn effect (within each retriever, RR ~ D_syn + n_tok + (1|asin)) ===
    log("\n=== 9. Per-retriever D_syn effect ===")
    per_retriever = {}
    for r in RETRIEVERS:
        df_r = df[df["retriever"] == r].copy()
        try:
            md_pr = smf.mixedlm(
                "RR ~ D_syn_c + n_tok_c",
                data=df_r,
                groups=df_r["asin"],
                re_formula="1",
            )
            mdf_pr = md_pr.fit(reml=False)
            per_retriever[r] = {
                "D_syn_coef": float(mdf_pr.params.get("D_syn_c", 0.0)),
                "D_syn_p": float(mdf_pr.pvalues.get("D_syn_c", 1.0)),
                "n_tok_coef": float(mdf_pr.params.get("n_tok_c", 0.0)),
                "n_tok_p": float(mdf_pr.pvalues.get("n_tok_c", 1.0)),
                "converged": bool(mdf_pr.converged),
                "nobs": int(mdf_pr.nobs),
            }
            log(f"  {r}: D_syn coef={per_retriever[r]['D_syn_coef']:+.4f}, "
                f"p={per_retriever[r]['D_syn_p']:.4g}; "
                f"n_tok coef={per_retriever[r]['n_tok_coef']:+.4f}, "
                f"p={per_retriever[r]['n_tok_p']:.4g}")
        except Exception as _e:
            per_retriever[r] = {"error": repr(_e)}
            log(f"  {r}: failed ({_e!r})")

    # === 10. Conclusion ===
    log("\n=== 10. Final conclusion ===")
    out = {
        "config": {
            "description": "Stage 7B: PCA48 D_syn × Retriever interaction",
            "RETRIEVERS": RETRIEVERS,
            "PCA_DIM": PCA_DIM,
            "SEED": SEED,
        },
        "dsyn_distribution": {
            "min": float(d_syns.min()),
            "mean": float(d_syns.mean()),
            "median": float(np.median(d_syns)),
            "max": float(d_syns.max()),
            "std": float(d_syns.std()),
        },
        "spearman_dsyn_ntok": {"rho": float(rho_l), "p_value": float(p_l)},
        "asin_dsyn_top5": asin_table,
        "mixedlm_full": mixedlm_results,
        "mixedlm_reduced": reduced_results,
        "per_retriever": per_retriever,
        "n_rows": len(rows),
        "n_unique_asins": len({r["asin"] for r in rows}),
        "n_unique_users": len({r["user_id"] for r in rows}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()