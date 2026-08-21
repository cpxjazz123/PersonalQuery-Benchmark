#!/usr/bin/env python3
"""Phase 15.B: Semantic Preservation — style must NOT regress content.

User feedback (iter_037 followup): StyleVector paper missing semantic preservation
analysis. Adding 2 metrics:
  1. Semantic similarity: cand vs product attrs string (cosine via sentence-bert)
  2. Attrs completeness: are all input attrs present in cand text?

If A_sampled lifts style but regresses semantic → pipeline unusable.
Verdict: A_sampled semantic ≥ D_off (no regression > 5%)

Reuses:
  - phase13_d_e2e_queries.jsonl (960 candidates)
  - sentence-transformers all-MiniLM-L6-v2 (~80MB, fast)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_CANDIDATES = OUT_DIR / "phase13_d_e2e_queries.jsonl"

OUT_REPORT = OUT_DIR / "phase15_b_semantic.json"
OUT_PER_CAND = OUT_DIR / "phase15_b_per_cand.jsonl"

CONDITIONS = ["A_sampled", "B_mean", "C_shuffled", "D_off"]
SEED = 42
ENCODER_BATCH = 128
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def bootstrap_ci(values, n: int = 2000, seed: int = SEED):
    arr = np.array(list(values))
    if len(arr) == 0:
        return 0.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)
    n_obs = len(arr)
    boot_means = []
    for _ in range(n):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_means.append(float(arr[idx].mean()))
    bm = np.array(boot_means)
    return float(arr.mean()), (float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)))


def attrs_completeness(cand_text: str, attrs: dict) -> dict:
    """Check whether each attr value appears (case-insensitive substring) in cand text.

    Returns:
      coverage: fraction of attr values present in cand text (0..1)
      missing: list of attr names whose values are missing
    """
    cand_lower = cand_text.lower()
    n_total = 0
    n_present = 0
    missing = []
    for k, v in attrs.items():
        if not v or not str(v).strip():
            continue
        n_total += 1
        v_str = str(v).strip()
        if v_str.lower() in cand_lower:
            n_present += 1
        else:
            missing.append(k)
    coverage = n_present / max(n_total, 1)
    return {"coverage": coverage, "missing": missing, "n_attrs": n_total, "n_present": n_present}


def main():
    log("=" * 70)
    log("Phase 15.B: Semantic Preservation — style must NOT regress content")
    log("=" * 70)

    np.random.seed(SEED)

    # === [1] Load candidates ===
    log("[1] Loading Phase 13.D candidates ...")
    candidates: list[dict] = []
    with IN_CANDIDATES.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    log(f"  total candidates: {len(candidates)}")

    # === [2] Load sentence-bert ===
    log(f"[2] Loading sentence-bert: {SBERT_MODEL} ...")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    from sentence_transformers import SentenceTransformer
    try:
        sbert = SentenceTransformer(SBERT_MODEL, device="cuda:0")
    except Exception as e:
        log(f"  local load failed ({e}), trying download ...")
        os.environ.pop("HF_HUB_OFFLINE", None)
        sbert = SentenceTransformer(SBERT_MODEL, device="cuda:0")
    log(f"  loaded, dim={sbert.get_sentence_embedding_dimension()}")

    # === [3] Encode product attrs (per pair) ===
    log("[3] Encoding product attrs (per pair) ...")
    pair_uids = []
    seen = set()
    for c in candidates:
        if c["user_id"] not in seen:
            seen.add(c["user_id"])
            pair_uids.append(c["user_id"])

    pair_attrs_text: dict[str, str] = {}
    for c in candidates:
        if c["user_id"] in pair_attrs_text:
            continue
        attrs = c["attrs"]
        attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
        pair_attrs_text[c["user_id"]] = attr_str

    attr_texts = [pair_attrs_text[u] for u in pair_uids]
    attr_embs = sbert.encode(
        attr_texts, convert_to_numpy=True, batch_size=ENCODER_BATCH,
        show_progress_bar=False, normalize_embeddings=False,
    )
    log(f"  attr_embs: {attr_embs.shape}")
    attr_embs_norm = l2_normalize(attr_embs.astype(np.float32))
    uid_to_attr_idx = {u: i for i, u in enumerate(pair_uids)}

    # === [4] Encode candidates (q_final_post) ===
    log("[4] Encoding candidate q_final_post ...")
    cand_texts = [c.get("q_final_post") or "" for c in candidates]
    cand_embs = sbert.encode(
        cand_texts, convert_to_numpy=True, batch_size=ENCODER_BATCH,
        show_progress_bar=False, normalize_embeddings=False,
    )
    log(f"  cand_embs: {cand_embs.shape}")
    cand_embs_norm = l2_normalize(cand_embs.astype(np.float32))

    # === [5] Per-cand semantic sim + attrs completeness ===
    log("[5] Per-cand semantic similarity + attrs completeness ...")
    n = len(candidates)
    sem_sim_arr = np.zeros(n, dtype=np.float64)
    coverage_arr = np.zeros(n, dtype=np.float64)
    missing_attr_count = np.zeros(n, dtype=np.int32)
    per_cand_records = []

    for i, c in enumerate(candidates):
        attr_idx = uid_to_attr_idx[c["user_id"]]
        sem_sim_arr[i] = float(np.dot(cand_embs_norm[i], attr_embs_norm[attr_idx]))
        comp = attrs_completeness(c.get("q_final_post") or "", c.get("attrs") or {})
        coverage_arr[i] = comp["coverage"]
        missing_attr_count[i] = len(comp["missing"])
        per_cand_records.append({
            "cand_idx_global": i,
            "user_id": c["user_id"],
            "asin": c["asin"],
            "condition": c["condition"],
            "q_styled": c.get("q_styled", "")[:200],
            "q_final_post": c.get("q_final_post", "")[:200],
            "semantic_sim_to_attrs": float(sem_sim_arr[i]),
            "attrs_coverage": float(coverage_arr[i]),
            "n_attrs_missing": int(missing_attr_count[i]),
            "attrs_missing": comp["missing"],
        })

    log(f"  semantic sim: mean={sem_sim_arr.mean():.4f}, "
        f"std={sem_sim_arr.std():.4f}, range=[{sem_sim_arr.min():.4f}, {sem_sim_arr.max():.4f}]")
    log(f"  attrs coverage: mean={coverage_arr.mean():.4f} (perfect=1.0)")

    # === [6] Per-cond aggregate ===
    log("[6] Per-cond aggregate ...")
    per_cond: dict[str, dict] = {}
    for cond in CONDITIONS:
        idxs = [i for i, c in enumerate(candidates) if c["condition"] == cond]
        per_cond[cond] = {
            "n_candidates": len(idxs),
            "mean_semantic_sim": float(sem_sim_arr[idxs].mean()),
            "median_semantic_sim": float(np.median(sem_sim_arr[idxs])),
            "mean_attrs_coverage": float(coverage_arr[idxs].mean()),
            "pct_perfect_coverage": float((coverage_arr[idxs] == 1.0).mean()),
            "mean_n_attrs_missing": float(missing_attr_count[idxs].mean()),
        }
        log(f"  {cond}: sem_sim={per_cond[cond]['mean_semantic_sim']:.4f}, "
            f"coverage={per_cond[cond]['mean_attrs_coverage']:.4f} "
            f"(perfect={per_cond[cond]['pct_perfect_coverage']*100:.1f}%)")

    # === [7] Paired bootstrap diffs (per pair) ===
    log("[7] Paired bootstrap diffs (per-pair) ...")
    pair_to_sem_mean: dict[tuple[str, str], float] = {}
    pair_to_cov_mean: dict[tuple[str, str], float] = {}
    for r in per_cand_records:
        key = (r["user_id"], r["condition"])
        pair_to_sem_mean.setdefault(key, []).append(r["semantic_sim_to_attrs"])
        pair_to_cov_mean.setdefault(key, []).append(r["attrs_coverage"])

    pair_to_sem_mean = {k: float(np.mean(v)) for k, v in pair_to_sem_mean.items()}
    pair_to_cov_mean = {k: float(np.mean(v)) for k, v in pair_to_cov_mean.items()}

    diffs = {}
    for cond_b in CONDITIONS:
        if cond_b == "A_sampled":
            continue
        sem_diffs = []
        cov_diffs = []
        for uid in pair_uids:
            sem_diffs.append(pair_to_sem_mean[(uid, "A_sampled")] - pair_to_sem_mean[(uid, cond_b)])
            cov_diffs.append(pair_to_cov_mean[(uid, "A_sampled")] - pair_to_cov_mean[(uid, cond_b)])
        d_sem, d_sem_ci = bootstrap_ci(sem_diffs)
        d_cov, d_cov_ci = bootstrap_ci(cov_diffs)
        diffs[f"A_sampled_vs_{cond_b}"] = {
            "semantic_diff": d_sem,
            "semantic_diff_ci95": list(d_sem_ci),
            "semantic_diff_excludes_0": (d_sem_ci[0] > 0 and d_sem_ci[1] > 0)
                                          or (d_sem_ci[0] < 0 and d_sem_ci[1] < 0),
            "coverage_diff": d_cov,
            "coverage_diff_ci95": list(d_cov_ci),
            "coverage_diff_excludes_0": (d_cov_ci[0] > 0 and d_cov_ci[1] > 0)
                                        or (d_cov_ci[0] < 0 and d_cov_ci[1] < 0),
        }
        log(f"  A vs {cond_b}: sem_diff={d_sem:+.4f} [{d_sem_ci[0]:+.4f},{d_sem_ci[1]:+.4f}], "
            f"cov_diff={d_cov:+.4f} [{d_cov_ci[0]:+.4f},{d_cov_ci[1]:+.4f}]")

    # === [8] Verdict ===
    a_sem = per_cond["A_sampled"]["mean_semantic_sim"]
    d_sem = per_cond["D_off"]["mean_semantic_sim"]
    a_cov = per_cond["A_sampled"]["mean_attrs_coverage"]
    d_cov = per_cond["D_off"]["mean_attrs_coverage"]
    a_vs_d = diffs["A_sampled_vs_D_off"]

    if a_vs_d["semantic_diff_excludes_0"] and a_vs_d["semantic_diff"] > 0:
        verdict = "GO"
        verdict_reason = (
            f"A_sampled SEMANTIC LIFTS over D_off CI excludes 0: "
            f"diff={a_vs_d['semantic_diff']:+.4f}"
        )
    elif a_vs_d["semantic_diff_ci95"][1] < -0.05:  # regress > 5%
        verdict = "NO-GO_regression"
        verdict_reason = (
            f"A_sampled SEMANTIC REGRESSES by >5%: diff={a_vs_d['semantic_diff']:+.4f} "
            f"CI [{a_vs_d['semantic_diff_ci95'][0]:+.4f}, {a_vs_d['semantic_diff_ci95'][1]:+.4f}]"
        )
    elif a_vs_d["semantic_diff"] < -0.02:
        verdict = "PARTIAL-GO_with_caution"
        verdict_reason = (
            f"A_sampled semantic slightly lower but not significantly: diff={a_vs_d['semantic_diff']:+.4f}"
        )
    else:
        verdict = "GO_no_regress"
        verdict_reason = (
            f"A_sampled semantic preserved (diff={a_vs_d['semantic_diff']:+.4f}, CI includes 0). "
            f"A={a_sem:.4f}, D={d_sem:.4f}"
        )
    log(f"\n  FINAL VERDICT: {verdict}")
    log(f"  reason: {verdict_reason}")

    # === [9] Save ===
    out = {
        "phase": "15.B",
        "encoder": SBERT_MODEL,
        "feature_space": "all-MiniLM-L6-v2 384d cosine (cand vs attrs string)",
        "n_candidates": len(candidates),
        "n_conditions": len(CONDITIONS),
        "n_pairs": len(pair_uids),
        "per_cond": per_cond,
        "paired_diffs": diffs,
        "decision_logic": {
            "GO": "A_sampled semantic CI > 0",
            "GO_no_regress": "no significant regression",
            "PARTIAL-GO_with_caution": "small negative diff but CI includes 0",
            "NO-GO_regression": "CI < -0.05 (regress > 5%)",
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }
    OUT_REPORT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"  report → {OUT_REPORT}")

    with OUT_PER_CAND.open("w") as f:
        for r in per_cand_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"  per_cand → {OUT_PER_CAND}")

    log("=" * 70)
    log("PHASE 15.B COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()