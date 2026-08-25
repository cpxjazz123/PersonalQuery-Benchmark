"""Stage 10J — Out-of-sample Decision-Boundary Validation.

Stage 10I-2 found Spearman ρ = −0.872 between `minilm_abs_G_min` and `rr_std`,
but a reviewer concern is: "of course boundary proximity predicts volatility,
you used the SAME queries to define both." This stage validates the finding
with two independent tests:

1. **Split-half**: For each ASIN's 10 personalized queries, randomly split
   into 5 boundary-estimation queries + 5 volatility-evaluation queries.
   Repeat N=500 times. Report mean Spearman ρ with bootstrap CI.

2. **Canonical query**: For each ASIN, use the Stage 8.5 `random` query
   (a non-user-specific pool candidate) as a canonical query. Compute
   |G_canonical|. Predict Stage 10H-B's personalized volatility from this
   single boundary measurement.

A reviewer-acceptable result requires:
- Split-half ρ remains negative and statistically significant
- Canonical ρ remains negative (smaller magnitude OK)
- The boundary signal is NOT an artifact of using the same queries

Inputs:
  - hj82_scratch2/.../stage10i2_per_query_gap.jsonl.gz (per-query G + ranks)
  - hj82_scratch2/.../stage10h_b_per_query.json (per-query BM25 + MiniLM ranks)
  - hj82_scratch2/.../stage10h_b_volatility.json (per-ASIN volatility metrics)
  - hj82_scratch2/.../stage8_5_selection.json (for canonical random queries)
  - hj82_scratch2/.../stage10i_corpus_embeds.npy (217K × 384 MiniLM corpus)
  - hj82_scratch2/.../stage7_corpus_docs.jsonl.gz

Outputs:
  - hj82_scratch2/.../stage10j_splithalf.json
  - hj82_scratch2/.../stage10j_canonical.json
  - hj82_scratch2/.../stage10j_summary.json
"""

from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
PER_Q_GAP = SCRATCH / "stage10i2_per_query_gap.jsonl.gz"
PER_Q_BASELINE = SCRATCH / "stage10h_b_per_query.json"
VOLATILITY_IN = SCRATCH / "stage10h_b_volatility.json"
SELECTION_IN = SCRATCH / "stage8_5_selection.json"
CORPUS_EMBEDS = SCRATCH / "stage10i_corpus_embeds.npy"
CORPUS_DOCS = SCRATCH / "stage7_corpus_docs.jsonl.gz"

OUTPUT_SPLITHALF = SCRATCH / "stage10j_splithalf.json"
OUTPUT_CANONICAL = SCRATCH / "stage10j_canonical.json"
OUTPUT_SUMMARY = SCRATCH / "stage10j_summary.json"

N_SPLIT_REPS = 500
SEED = 2024


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=== Stage 10J — Out-of-sample Decision-Boundary Validation ===")

    # === 1. Load per-query gap + per-query retrieval ranks ===
    log("\n=== 1. Loading per-query data ===")
    per_q_gap_raw = []
    with gzip.open(PER_Q_GAP, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            per_q_gap_raw.append(json.loads(line))
    log(f"  per-query gap records: {len(per_q_gap_raw)}")

    # Build (asin, user_id) -> record map
    gap_map = {(r["asin"], r["user_id"]): r for r in per_q_gap_raw}

    # Load Stage 10H-B per-query (for rank info)
    per_q_b = json.load(open(PER_Q_BASELINE))
    log(f"  per-query (Stage 10H-B): {len(per_q_b)} entries")

    # Group by ASIN
    asin_to_queries = defaultdict(list)
    for r in per_q_b:
        asin_to_queries[r["asin"]].append(r)

    # Load volatility
    vol_data = json.load(open(VOLATILITY_IN))
    per_asin_vol = vol_data["per_asin"]
    log(f"  volatility ASINs: {len(per_asin_vol)}")

    # Load Stage 8.5 selection (for canonical random queries)
    sel_data = json.load(open(SELECTION_IN))
    asin_user_to_canonical = {}
    for e in sel_data["entries"]:
        canonical_q = e.get("random", {}).get("query", "")
        if canonical_q:
            asin_user_to_canonical[(e["asin"], e["user_id"])] = canonical_q
    log(f"  canonical queries (Stage 8.5 random): {len(asin_user_to_canonical)}")

    # === 2. Split-half validation ===
    log("\n=== 2. Split-half validation (5 boundary + 5 volatility, 500 reps) ===")
    rng = np.random.RandomState(SEED)

    def boundary_metric(g_list, m):
        arr = np.array([abs(g) for g in g_list if g is not None], dtype=np.float64)
        if len(arr) == 0:
            return None
        if m == "min":
            return float(arr.min())
        if m == "mean":
            return float(arr.mean())
        if m == "frac_bnd":
            return float((arr < 0.05).mean())
        raise ValueError(m)

    def volatility_metric(rank_list, m):
        # rank_list is a list of (bm25_rank, minilm_rank) tuples
        valid = [(b, m_) for b, m_ in rank_list if b is not None and m_ is not None]
        if len(valid) < 2:
            return None
        if m == "rr_std":
            rrs = [1.0 / r[1] for r in valid]  # use MiniLM rank for RR
            return float(np.std(rrs, ddof=1))
        if m == "hit1_flip":
            hits = [1 if r[1] == 1 else 0 for r in valid]
            pairs = list(combinations(range(len(hits)), 2))
            if not pairs:
                return 0.0
            return sum(1 for i, j in pairs if hits[i] != hits[j]) / len(pairs)
        if m == "hit5_flip":
            hits = [1 if r[1] <= 5 else 0 for r in valid]
            pairs = list(combinations(range(len(hits)), 2))
            if not pairs:
                return 0.0
            return sum(1 for i, j in pairs if hits[i] != hits[j]) / len(pairs)
        raise ValueError(m)

    # Pre-build per-ASIN: list of (minilm_G, bm25_G, bm25_rank, minilm_rank)
    asin_data = {}
    for asin, recs in asin_to_queries.items():
        rows = []
        for r in recs:
            gap = gap_map.get((asin, r["user_id"]), {})
            rows.append({
                "minilm_G": gap.get("minilm_G"),
                "bm25_G": gap.get("bm25_G"),
                "bm25_rank": r.get("bm25_rank"),
                "minilm_rank": r.get("minilm_rank"),
            })
        asin_data[asin] = rows

    # For each rep, randomly split each ASIN into 2 halves
    boundary_metrics = ["min", "mean", "frac_bnd"]
    volatility_metrics = ["rr_std", "hit1_flip", "hit5_flip"]
    rep_results = []

    for rep_idx in range(N_SPLIT_REPS):
        rep_data = {}  # asin -> {boundary: {metric: val}, volatility: {metric: val}}
        for asin, rows in asin_data.items():
            n = len(rows)
            idx_perm = rng.permutation(n)
            half = n // 2
            boundary_rows = [rows[i] for i in idx_perm[:half]]
            volatility_rows = [rows[i] for i in idx_perm[half:half*2]]

            boundary_g = [r["minilm_G"] for r in boundary_rows]
            boundary_g_bm25 = [r["bm25_G"] for r in boundary_rows]
            volatility_ranks = [(r["bm25_rank"], r["minilm_rank"]) for r in volatility_rows]

            rep_data[asin] = {
                "boundary": {
                    "minilm_min": boundary_metric(boundary_g, "min"),
                    "minilm_mean": boundary_metric(boundary_g, "mean"),
                    "minilm_frac": boundary_metric(boundary_g, "frac_bnd"),
                    "bm25_min": boundary_metric(boundary_g_bm25, "min"),
                },
                "volatility": {
                    "minilm_rr_std": volatility_metric(volatility_ranks, "rr_std"),
                    "minilm_hit1_flip": volatility_metric(volatility_ranks, "hit1_flip"),
                    "minilm_hit5_flip": volatility_metric(volatility_ranks, "hit5_flip"),
                },
            }
        rep_results.append(rep_data)

    # Compute Spearman ρ for each rep, each (boundary, volatility) pair
    from scipy.stats import spearmanr

    pairs_to_compute = [
        ("minilm_min", "minilm_rr_std"),
        ("minilm_mean", "minilm_rr_std"),
        ("minilm_frac", "minilm_rr_std"),
        ("bm25_min", "minilm_rr_std"),
        ("minilm_min", "minilm_hit1_flip"),
        ("minilm_mean", "minilm_hit1_flip"),
        ("minilm_frac", "minilm_hit1_flip"),
        ("minilm_min", "minilm_hit5_flip"),
        ("minilm_mean", "minilm_hit5_flip"),
        ("minilm_frac", "minilm_hit5_flip"),
    ]

    log("  Computing Spearman ρ over 500 reps...")
    rho_distributions = {p: [] for p in pairs_to_compute}
    for rep_data in rep_results:
        for b_key, v_key in pairs_to_compute:
            xs, ys = [], []
            for asin, d in rep_data.items():
                bv = d["boundary"].get(b_key)
                vv = d["volatility"].get(v_key)
                if bv is None or vv is None:
                    continue
                xs.append(bv)
                ys.append(vv)
            if len(xs) < 10:
                continue
            rho, _ = spearmanr(xs, ys)
            rho_distributions[(b_key, v_key)].append(float(rho))

    splithalf_summary = {}
    log(f"\n  Split-half Spearman ρ (n={N_SPLIT_REPS} reps):")
    log(f"    {'boundary':<14} {'volatility':<22} {'mean ρ':>10} {'std':>8} {'frac<0':>8} {'frac>0':>8} {'sig_frac':>10}")
    for b_key, v_key in pairs_to_compute:
        dist = rho_distributions[(b_key, v_key)]
        if not dist:
            continue
        mean_rho = float(np.mean(dist))
        std_rho = float(np.std(dist))
        # Fraction of reps where ρ < 0 (consistent direction with full data)
        frac_neg = float(np.mean([r < 0 for r in dist]))
        frac_pos = float(np.mean([r > 0 for r in dist]))
        # Significance per rep: p < 0.05
        sig_frac = None
        if dist:
            p_dist = []
            for rep_data in rep_results:
                xs, ys = [], []
                for asin, d in rep_data.items():
                    bv = d["boundary"].get(b_key)
                    vv = d["volatility"].get(v_key)
                    if bv is None or vv is None:
                        continue
                    xs.append(bv)
                    ys.append(vv)
                if len(xs) < 10:
                    continue
                _, p = spearmanr(xs, ys)
                p_dist.append(p)
            sig_frac = float(np.mean([p < 0.05 for p in p_dist]))
        log(f"    {b_key:<14} {v_key:<22} {mean_rho:>10.3f} {std_rho:>8.3f} {frac_neg:>8.3f} {frac_pos:>8.3f} {sig_frac:>10.3f}")
        splithalf_summary[f"{b_key}__{v_key}"] = {
            "n_reps": len(dist),
            "mean_rho": mean_rho,
            "std_rho": std_rho,
            "frac_negative": frac_neg,
            "frac_positive": frac_pos,
            "sig_fraction_p_lt_0.05": sig_frac,
            "rho_distribution": dist[:50],  # save first 50 for inspection
        }

    # === 3. Canonical query: encode Stage8.5 random queries + compute G_canonical ===
    log("\n=== 3. Canonical query: encode Stage 8.5 random + compute G_canonical ===")

    # Collect one canonical query per (asin, user) — but we want ONE per ASIN for prediction
    # Use the FIRST entry per ASIN as canonical (or aggregate)
    asin_to_canonical_q = {}
    for e in sel_data["entries"]:
        if e["asin"] not in asin_to_canonical_q:
            asin_to_canonical_q[e["asin"]] = e.get("random", {}).get("query", "")

    # Encode with MiniLM
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    corpus_embeds = np.load(CORPUS_EMBEDS)
    log(f"  corpus_embeds: {corpus_embeds.shape}")

    # Build asin -> corpus index
    asin_to_doc = {}
    with gzip.open(CORPUS_DOCS, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("asin")
            doc = rec.get("doc", "") or rec.get("text", "")
            if asin and doc:
                asin_to_doc[asin] = doc[:1000]
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}

    canonical_asins = sorted(asin_to_canonical_q.keys())
    canonical_queries = [asin_to_canonical_q[a] for a in canonical_asins]
    t0 = time.time()
    canonical_embeds = model.encode(
        canonical_queries, batch_size=128, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )
    log(f"  canonical encoded: {time.time() - t0:.1f}s ({len(canonical_queries)} queries)")

    t0 = time.time()
    canon_gpu = torch.from_numpy(canonical_embeds).cuda()
    doc_gpu = torch.from_numpy(corpus_embeds).cuda()
    canon_sims = canon_gpu @ doc_gpu.T  # [N_canon, D]
    log(f"  canonical sim matrix: {time.time() - t0:.1f}s, shape={tuple(canon_sims.shape)}")

    # Per ASIN: compute G_canonical
    canonical_g = {}  # asin -> G_canonical
    canonical_g_abs = {}  # asin -> |G_canonical|
    for i, asin in enumerate(canonical_asins):
        tgt_idx = asin_to_idx.get(asin, -1)
        if tgt_idx < 0:
            continue
        sc = canon_sims[i]
        tgt_score = float(sc[tgt_idx].item())
        rank_target = int((sc > sc[tgt_idx]).sum().item()) + 1
        sc_masked = sc.clone()
        sc_masked[tgt_idx] = -1e9
        top2_vals, _ = torch.topk(sc_masked, k=2)
        score_rank1 = float(top2_vals[0].item())
        score_rank2 = float(top2_vals[1].item())
        if rank_target > 1:
            G = tgt_score - score_rank1
        else:
            G = tgt_score - score_rank2
        canonical_g[asin] = G
        canonical_g_abs[asin] = abs(G)

    del canon_sims, canon_gpu, doc_gpu, canonical_embeds, corpus_embeds, model
    torch.cuda.empty_cache()

    log(f"  G_canonical computed for {len(canonical_g)} ASINs")
    log(f"  G_canonical distribution: min={min(canonical_g.values()):.4f}, "
        f"mean={np.mean(list(canonical_g.values())):.4f}, "
        f"max={max(canonical_g.values()):.4f}")
    log(f"  |G_canonical| distribution: min={min(canonical_g_abs.values()):.4f}, "
        f"mean={np.mean(list(canonical_g_abs.values())):.4f}, "
        f"max={max(canonical_g_abs.values()):.4f}")

    # Spearman correlation: canonical G vs volatility
    log("\n  Spearman ρ (G_canonical ~ volatility):")
    canonical_spearman = {}
    for y_key in ["minilm_rr_std", "minilm_hit1_flip_rate", "minilm_hit5_flip_rate"]:
        xs, ys = [], []
        for asin in canonical_g:
            if asin not in per_asin_vol:
                continue
            y = per_asin_vol[asin].get(y_key)
            if y is None:
                continue
            xs.append(canonical_g[asin])
            ys.append(y)
        if len(xs) < 10:
            continue
        rho, p = spearmanr(xs, ys)
        canonical_spearman[f"G_canonical__{y_key}"] = {"rho": float(rho), "p": float(p), "n": len(xs)}
        log(f"    G_canonical (signed)   vs {y_key:<26} ρ={rho:+.3f} p={p:.2e}")

        xs_abs, ys_abs = [], []
        for asin in canonical_g_abs:
            if asin not in per_asin_vol:
                continue
            y = per_asin_vol[asin].get(y_key)
            if y is None:
                continue
            xs_abs.append(canonical_g_abs[asin])
            ys_abs.append(y)
        if len(xs_abs) >= 10:
            rho_abs, p_abs = spearmanr(xs_abs, ys_abs)
            canonical_spearman[f"abs_G_canonical__{y_key}"] = {
                "rho": float(rho_abs), "p": float(p_abs), "n": len(xs_abs)
            }
            log(f"    |G_canonical| (abs)    vs {y_key:<26} ρ={rho_abs:+.3f} p={p_abs:.2e}")

    # === 4. Save outputs ===
    log("\n=== 4. Saving outputs ===")
    with open(OUTPUT_SPLITHALF, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 10J: split-half validation",
                "n_reps": N_SPLIT_REPS,
                "SEED": SEED,
                "split_rule": "5 boundary + 5 volatility, random per ASIN",
            },
            "results": splithalf_summary,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_SPLITHALF}")

    with open(OUTPUT_CANONICAL, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 10J: canonical query (Stage 8.5 random)",
                "G_definition": "G = score_target - score_nearest_competitor",
            },
            "G_canonical": canonical_g,
            "abs_G_canonical": canonical_g_abs,
            "spearman": canonical_spearman,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_CANONICAL}")

    # === 5. Final summary ===
    summary = {
        "splithalf": {
            k: {
                "mean_rho": v["mean_rho"],
                "std_rho": v["std_rho"],
                "frac_negative": v["frac_negative"],
                "frac_positive": v["frac_positive"],
                "sig_fraction_p_lt_0.05": v["sig_fraction_p_lt_0.05"],
            }
            for k, v in splithalf_summary.items()
        },
        "canonical": canonical_spearman,
    }
    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_SUMMARY}")

    log("\n=== Stage 10J complete ===")


if __name__ == "__main__":
    main()