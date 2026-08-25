"""Stage 10I-2 — Margin Audit: Correct Decision Gap Definition.

Stage 10I found MiniLM margin_mean has ρ=+0.822 with rr_std (opposite of
expected direction). The current margin = score_target - score_best_competitor
where best_competitor is the top-1 retrieved ASIN — but the top-1 may NOT be
the nearest competitor in the decision boundary.

This stage audits whether the unexpected sign is an artifact of the
operationalization. We redefine decision gap:

    G(q) = score_target - score_nearest_competitor
    where nearest_competitor is the highest-scoring ASIN ≠ target
    (i.e., score_rank1 if rank_target > 1, else score_rank2)

Then per-ASIN boundary proximity metrics:
    - min_abs_G: closest any query gets to boundary
    - frac_near_boundary: share of queries with |G| < ε
    - mean_abs_G, G_mean, G_std

Three-group diagnostic:
    - High volatility (rr_std top 10)
    - Low volatility (rr_std bottom 10)
    - High margin (margin_mean top 10 — for diagnosis of the original ρ > 0)

Re-correlation with corrected G. Spearman ρ for:
    - min_abs_G / mean_abs_G / frac_near_boundary / G_mean vs rr_std, hit1_flip

Inputs:
  - hj82_scratch2/.../stage10i_corpus_embeds.npy (217K × 384)
  - hj82_scratch2/.../stage10i_minilm_scores.jsonl.gz (per-query target+top5)
  - hj82_scratch2/.../stage10i_bm25_scores.jsonl.gz (per-query BM25 top5)
  - hj82_scratch2/.../stage10h_b_volatility.json
  - hj82_scratch2/.../stage10i_features.json
  - hj82_scratch2/.../stage10h_b_per_query.json

Outputs:
  - hj82_scratch2/.../stage10i2_per_query_gap.jsonl.gz
  - hj82_scratch2/.../stage10i2_boundary_metrics.json
  - hj82_scratch2/.../stage10i2_diagnostic_groups.json
  - hj82_scratch2/.../stage10i2_spearman.json
"""

from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
CORPUS_EMBEDS = SCRATCH / "stage10i_corpus_embeds.npy"
VOLATILITY_IN = SCRATCH / "stage10h_b_volatility.json"
FEATURES_IN = SCRATCH / "stage10i_features.json"
PER_QUERY_IN = SCRATCH / "stage10h_b_per_query.json"
CORPUS_DOCS_IN = SCRATCH / "stage7_corpus_docs.jsonl.gz"

OUTPUT_PER_Q = SCRATCH / "stage10i2_per_query_gap.jsonl.gz"
OUTPUT_BOUNDARY = SCRATCH / "stage10i2_boundary_metrics.json"
OUTPUT_GROUPS = SCRATCH / "stage10i2_diagnostic_groups.json"
OUTPUT_SPEARMAN = SCRATCH / "stage10i2_spearman.json"

EPS_BOUNDARY = 0.05


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=== Stage 10I-2 — Margin Audit ===")

    # === 1. Load per-query + corpus ===
    log("\n=== 1. Loading per-query + corpus ===")
    per_query = json.load(open(PER_QUERY_IN))
    log(f"  per-query: {len(per_query)} entries")

    asin_to_doc = {}
    with gzip.open(CORPUS_DOCS_IN, "rt", encoding="utf-8") as f:
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
    log(f"  corpus: {len(asins)} ASINs")
    target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in per_query])
    queries = [r["query"] for r in per_query]

    # === 2. MiniLM: compute G_minilm ===
    log("\n=== 2. MiniLM: full sim matrix + G ===")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    corpus_embeds = np.load(CORPUS_EMBEDS)
    log(f"  corpus_embeds: {corpus_embeds.shape}")

    t0 = time.time()
    q_embeds = model.encode(
        queries, batch_size=512, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True
    )
    log(f"  queries encoded: {time.time() - t0:.1f}s")

    t0 = time.time()
    q_embeds_gpu = torch.from_numpy(q_embeds).cuda()
    doc_embeds_gpu = torch.from_numpy(corpus_embeds).cuda()
    sims = q_embeds_gpu @ doc_embeds_gpu.T  # [1000, 217722]
    log(f"  sim matrix: {time.time() - t0:.1f}s, shape={tuple(sims.shape)}")

    per_query_gap = {}
    for i, r in enumerate(per_query):
        key = (r["asin"], r["user_id"])
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            per_query_gap[key] = {
                "asin": r["asin"], "user_id": r["user_id"],
                "minilm_rank_target": None, "minilm_score_target": None,
                "minilm_G": None,
            }
            continue
        sc = sims[i]
        tgt_score = float(sc[tgt_idx].item())
        rank_target = int((sc > sc[tgt_idx]).sum().item()) + 1
        sc_masked = sc.clone()
        sc_masked[tgt_idx] = -1e9
        top2_vals, top2_idxs = torch.topk(sc_masked, k=2)
        score_rank1 = float(top2_vals[0].item())
        score_rank2 = float(top2_vals[1].item())
        # G definition: nearest_competitor is rank1 (if target not rank1) or rank2 (if target is rank1)
        if rank_target > 1:
            G = tgt_score - score_rank1
        else:
            G = tgt_score - score_rank2
        per_query_gap[key] = {
            "asin": r["asin"], "user_id": r["user_id"],
            "minilm_rank_target": rank_target,
            "minilm_score_target": tgt_score,
            "minilm_score_rank1": score_rank1,
            "minilm_score_rank2": score_rank2,
            "minilm_G": float(G),
        }

    del sims, q_embeds_gpu, doc_embeds_gpu, q_embeds, corpus_embeds, model
    torch.cuda.empty_cache()
    log(f"  computed G_minilm for {len(per_query_gap)} queries")

    # === 3. BM25: re-retrieve full corpus for target score + G_bm25 ===
    log("\n=== 3. BM25: re-retrieve full corpus for target score ===")
    import bm25s

    corpus_texts = [asin_to_doc[a] for a in asins]
    t0 = time.time()
    log("  tokenizing + building BM25 index...")
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    log(f"  BM25 index built: {time.time() - t0:.1f}s")

    t0 = time.time()
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    log(f"  query tokenize: {time.time() - t0:.1f}s")

    t0 = time.time()
    bm25_raw = retriever.retrieve(query_tokens, k=len(corpus_texts), show_progress=False)
    log(f"  BM25 retrieve: {time.time() - t0:.1f}s")

    for i, r in enumerate(per_query):
        key = (r["asin"], r["user_id"])
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            per_query_gap[key]["bm25_rank_target"] = None
            per_query_gap[key]["bm25_score_target"] = None
            per_query_gap[key]["bm25_G"] = None
            continue
        scores_arr = bm25_raw.scores[i]
        idxs_arr = bm25_raw.documents[i]
        tgt_pos = (idxs_arr == tgt_idx)
        if not tgt_pos.any():
            per_query_gap[key]["bm25_rank_target"] = None
            per_query_gap[key]["bm25_score_target"] = None
            per_query_gap[key]["bm25_G"] = None
            continue
        rank_target = int(np.where(tgt_pos)[0][0]) + 1
        score_target = float(scores_arr[tgt_pos][0])
        # Find score_rank1 and score_rank2 (masking target)
        if rank_target == 1:
            # rank1 is target itself; rank2 = scores_arr[1]
            score_rank1 = score_target
            score_rank2 = float(scores_arr[1]) if len(scores_arr) > 1 else 0.0
        else:
            # rank1 = scores_arr[0]
            score_rank1 = float(scores_arr[0])
            # rank2: find next that isn't target
            next_idx = 1
            while next_idx < len(idxs_arr) and idxs_arr[next_idx] == tgt_idx:
                next_idx += 1
            score_rank2 = float(scores_arr[next_idx]) if next_idx < len(idxs_arr) else 0.0
        if rank_target > 1:
            G = score_target - score_rank1
        else:
            G = score_target - score_rank2
        per_query_gap[key].update({
            "bm25_rank_target": rank_target,
            "bm25_score_target": score_target,
            "bm25_score_rank1": score_rank1,
            "bm25_score_rank2": score_rank2,
            "bm25_G": float(G),
        })

    del retriever, corpus_tokens, query_tokens, bm25_raw
    import gc; gc.collect()

    with gzip.open(OUTPUT_PER_Q, "wt", encoding="utf-8") as f:
        for key in per_query_gap:
            f.write(json.dumps(per_query_gap[key], ensure_ascii=False) + "\n")
    log(f"  wrote → {OUTPUT_PER_Q}")

    # === 4. Per-ASIN boundary metrics ===
    log("\n=== 4. Per-ASIN boundary metrics ===")
    asin_to_gaps_minilm = defaultdict(list)
    asin_to_gaps_bm25 = defaultdict(list)
    for key, rec in per_query_gap.items():
        if rec.get("minilm_G") is not None:
            asin_to_gaps_minilm[rec["asin"]].append(rec["minilm_G"])
        if rec.get("bm25_G") is not None:
            asin_to_gaps_bm25[rec["asin"]].append(rec["bm25_G"])

    def boundary_metrics(gaps):
        arr = np.array(gaps, dtype=np.float64)
        if len(arr) == 0:
            return None
        return {
            "n": len(arr),
            "G_mean": float(arr.mean()),
            "G_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "G_min": float(arr.min()),
            "G_max": float(arr.max()),
            "abs_G_mean": float(np.abs(arr).mean()),
            "abs_G_min": float(np.abs(arr).min()),
            "abs_G_max": float(np.abs(arr).max()),
            "frac_near_boundary_005": float((np.abs(arr) < EPS_BOUNDARY).mean()),
            "frac_near_boundary_001": float((np.abs(arr) < 0.01).mean()),
        }

    boundary = {}
    for asin in set(list(asin_to_gaps_minilm.keys()) + list(asin_to_gaps_bm25.keys())):
        boundary[asin] = {
            "minilm": boundary_metrics(asin_to_gaps_minilm.get(asin, [])),
            "bm25": boundary_metrics(asin_to_gaps_bm25.get(asin, [])),
        }
    log(f"  boundary metrics for {len(boundary)} ASINs")

    with open(OUTPUT_BOUNDARY, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 10I-2: per-ASIN boundary proximity metrics",
                "EPS_BOUNDARY": EPS_BOUNDARY,
                "G_definition": "G = score_target - score_nearest_competitor (rank1 if target!=rank1, else rank2)",
            },
            "metrics": boundary,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_BOUNDARY}")

    # === 5. Three-group diagnostic ===
    log("\n=== 5. Three-group diagnostic ===")
    vol_data = json.load(open(VOLATILITY_IN))
    feat_data = json.load(open(FEATURES_IN))
    per_asin_vol = vol_data["per_asin"]
    feat_map = feat_data["features"]
    asins_sorted_by_rr_std = sorted(per_asin_vol.keys(), key=lambda a: -per_asin_vol[a]["minilm_rr_std"])
    asins_sorted_by_margin = sorted(per_asin_vol.keys(), key=lambda a: -(feat_map[a].get("minilm_margin_mean") or -1))

    high_vol = asins_sorted_by_rr_std[:10]
    low_vol = asins_sorted_by_rr_std[-10:]
    high_margin = asins_sorted_by_margin[:10]

    log(f"\n  high_vol ASINs (rr_std top 10):")
    log(f"    {high_vol}")
    log(f"  low_vol ASINs (rr_std bottom 10):")
    log(f"    {low_vol}")
    log(f"  high_margin ASINs (margin top 10):")
    log(f"    {high_margin}")

    diagnostic = {}
    for group_name, group_asins in [
        ("high_volatility", high_vol),
        ("low_volatility", low_vol),
        ("high_margin", high_margin),
    ]:
        rows = []
        for asin in group_asins:
            vol = per_asin_vol[asin]
            bm = boundary[asin]
            m = bm["minilm"]
            rows.append({
                "asin": asin,
                "minilm_rr_std": vol["minilm_rr_std"],
                "minilm_hit1_flip_rate": vol["minilm_hit1_flip_rate"],
                "minilm_margin_mean_original": feat_map[asin].get("minilm_margin_mean"),
                "minilm_abs_G_mean": m["abs_G_mean"] if m else None,
                "minilm_abs_G_min": m["abs_G_min"] if m else None,
                "minilm_G_mean": m["G_mean"] if m else None,
                "minilm_frac_near_boundary": m["frac_near_boundary_005"] if m else None,
                "minilm_rank_median": vol.get("minilm_rank_median"),
            })
        diagnostic[group_name] = rows

        log(f"\n  === {group_name} ===")
        log(f"    {'asin':<14} {'rr_std':>9} {'hit1_flp':>9} {'margin':>9} {'abs_G_mean':>11} {'abs_G_min':>11} {'G_mean':>9} {'frac_bnd':>9} {'rank_med':>9}")
        for r in rows:
            log(f"    {r['asin']:<14} "
                f"{r['minilm_rr_std']*100:>8.3f}% "
                f"{r['minilm_hit1_flip_rate']*100:>8.3f}% "
                f"{(r['minilm_margin_mean_original'] or 0):>9.3f} "
                f"{(r['minilm_abs_G_mean'] or 0):>11.4f} "
                f"{(r['minilm_abs_G_min'] or 0):>11.4f} "
                f"{(r['minilm_G_mean'] or 0):>9.4f} "
                f"{(r['minilm_frac_near_boundary'] or 0):>9.3f} "
                f"{(r['minilm_rank_median'] or 0):>9.1f}")

    with open(OUTPUT_GROUPS, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 10I-2: three-group diagnostic",
                "groups": ["high_volatility", "low_volatility", "high_margin"],
            },
            "diagnostic": diagnostic,
        }, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {OUTPUT_GROUPS}")

    # Mann-Whitney U: high_vol vs low_vol on abs_G_min
    log("\n  Mann-Whitney U test: high_vol vs low_vol on minilm_abs_G_min")
    from scipy.stats import mannwhitneyu
    high_abs_G_min = [r["minilm_abs_G_min"] for r in diagnostic["high_volatility"] if r["minilm_abs_G_min"] is not None]
    low_abs_G_min = [r["minilm_abs_G_min"] for r in diagnostic["low_volatility"] if r["minilm_abs_G_min"] is not None]
    if len(high_abs_G_min) >= 5 and len(low_abs_G_min) >= 5:
        u_stat, u_p = mannwhitneyu(high_abs_G_min, low_abs_G_min, alternative="two-sided")
        log(f"    U={u_stat:.1f}, p={u_p:.4f}")
        log(f"    high_vol mean abs_G_min: {np.mean(high_abs_G_min):.4f}")
        log(f"    low_vol mean abs_G_min:  {np.mean(low_abs_G_min):.4f}")

    # === 6. Spearman re-correlation ===
    log("\n=== 6. Spearman re-correlation ===")
    from scipy.stats import spearmanr

    rows_for_corr = []
    for asin in sorted(boundary.keys()):
        vol = per_asin_vol.get(asin, {})
        m = boundary[asin]["minilm"]
        if m is None:
            continue
        rows_for_corr.append({
            "asin": asin,
            "minilm_rr_std": vol.get("minilm_rr_std"),
            "minilm_hit1_flip_rate": vol.get("minilm_hit1_flip_rate"),
            "minilm_hit5_flip_rate": vol.get("minilm_hit5_flip_rate"),
            "minilm_G_mean": m["G_mean"],
            "minilm_abs_G_mean": m["abs_G_mean"],
            "minilm_abs_G_min": m["abs_G_min"],
            "minilm_frac_near_boundary": m["frac_near_boundary_005"],
            "minilm_margin_mean_original": feat_map[asin].get("minilm_margin_mean"),
        })

    Y_COLS = ["minilm_rr_std", "minilm_hit1_flip_rate", "minilm_hit5_flip_rate"]
    X_COLS = [
        "minilm_G_mean", "minilm_abs_G_mean", "minilm_abs_G_min",
        "minilm_frac_near_boundary",
        "minilm_margin_mean_original",
    ]

    spearman_results = {}
    log(f"\n  Spearman ρ (volatility ~ G-variable, n={len(rows_for_corr)}):")
    log(f"    {'predictor':<32} " + " ".join(f"{y[:24]:>25}" for y in Y_COLS))
    for x_col in X_COLS:
        spearman_results[x_col] = {}
        cells = []
        for y_col in Y_COLS:
            pairs = [(r[x_col], r[y_col]) for r in rows_for_corr
                     if r[x_col] is not None and r[y_col] is not None]
            if len(pairs) < 5:
                cells.append(f"{'N/A':>25}")
                spearman_results[x_col][y_col] = {"rho": None, "p": None}
                continue
            x_vals = np.array([p[0] for p in pairs])
            y_vals = np.array([p[1] for p in pairs])
            rho, p = spearmanr(x_vals, y_vals)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else " "))
            cells.append(f"ρ={rho:+.3f} p={p:.2e} {sig}")
            spearman_results[x_col][y_col] = {"rho": float(rho), "p": float(p)}
        log(f"    {x_col:<32} " + " ".join(f"{c:>25}" for c in cells))

    with open(OUTPUT_SPEARMAN, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 10I-2: Spearman with corrected decision gap G",
                "Y_COLS": Y_COLS,
                "X_COLS": X_COLS,
                "n_asins": len(rows_for_corr),
            },
            "spearman": spearman_results,
        }, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {OUTPUT_SPEARMAN}")

    log("\n=== Stage 10I-2 complete ===")


if __name__ == "__main__":
    main()