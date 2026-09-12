"""Phase L8.24 — Raw 3.39M user sentence-count distribution (Stage 02 source).

Per user 2026-09-11:
    "从 Stage 02 重新统计3,386,206用户的句子分布"

OBJECTIVE
    Stream uid_to_sentences.json (1.46 GB) with ijson; for each uid,
    record array length = n_sents. No PCFG, no min-sents, no encoder.
    This is the FULL distribution BEFORE any filter.

OUTPUT
    - Percentile summary (min/p1/p5/p10/p25/median/p75/p90/p95/p99/p99.5/p99.9/max)
    - Log-binned histogram
    - Tukey outlier bounds in log10-space (k=1.5)
    - Candidate cutoffs evaluated from raw distribution ONLY
    - Recommended h_min / h_max (frozen, before encoder)

NOTE on memory
    3.3M users × ~10B each (int64 n_sents) ≈ 27 MB; total int array
    fits in < 100 MB.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import ijson
import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
INPUT_JSON = Path("/fs04/ar57/wenyu/PersoanlQuery/result/02_user_review_sentence_extract/uid_to_sentences.json")
OUT_DIR = REPO_ROOT / "result/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    log(f"streaming {INPUT_JSON}")
    log(f"  size: {INPUT_JSON.stat().st_size/1e9:.2f} GB")

    counts = []
    t1 = time.time()
    n_users = 0
    last_log = time.time()
    with open(INPUT_JSON, "rb") as f:
        # ijson parses {key: [array]} as items (key, value) pairs
        for uid, sents in ijson.kvitems(f, ""):
            counts.append(len(sents))
            n_users += 1
            if time.time() - last_log > 30:
                log(f"  scanned {n_users/1e6:.2f}M users in {time.time()-t1:.1f}s")
                last_log = time.time()
    log(f"  DONE scanning {n_users} users in {time.time()-t1:.1f}s")

    counts = np.array(counts, dtype=np.int64)
    log(f"\n  raw cohort: n_users = {n_users:,}")
    log(f"  h_u: min={counts.min()}  max={counts.max()}  "
        f"mean={counts.mean():.2f}  median={np.median(counts):.0f}  "
        f"sum={int(counts.sum()):,}")

    # Percentile summary — finer for upper tail
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    pct_vals = {q: float(np.percentile(counts, q)) for q in qs}
    log("\n  h_u percentile summary (raw cohort, no filter):")
    for q in qs:
        log(f"    p{q:>5g}  = {pct_vals[q]:.0f}")

    # Tukey outlier rule (log10)
    log_arr = np.log10(counts[counts > 0])
    q1, q3 = np.percentile(log_arr, [25, 75])
    iqr = q3 - q1
    k = 1.5
    lo_log = 10 ** (q1 - k * iqr)
    hi_log = 10 ** (q3 + k * iqr)
    log("\n  Tukey outlier rule (log10, k=1.5):")
    log(f"    Q1(log10)={q1:.3f}  Q3(log10)={q3:.3f}  IQR={iqr:.3f}")
    log(f"    bounds: [{lo_log:.1f}, {hi_log:.1f}]")
    n_in_tukey = int(((counts >= lo_log) & (counts <= hi_log)).sum())
    log(f"    in-bounds users: {n_in_tukey:,} / {n_users:,} "
        f"({100.0 * n_in_tukey / n_users:.1f}%)")

    # Candidate cutoffs — purely from raw distribution
    def in_range(lo, hi):
        return int(((counts >= lo) & (counts <= hi)).sum())

    candidates = [
        ("h >= 1 (raw cohort, no floor)",
         lambda c: c >= 1, 1, counts.max()),
        ("h >= 5 (personalization min)",
         lambda c: c >= 5, 5, counts.max()),
        ("h >= 10 (paper-style min)",
         lambda c: c >= 10, 10, counts.max()),
        ("h >= 30 (mid-tier personalization)",
         lambda c: c >= 30, 30, counts.max()),
        ("h >= 50 (current PCFG cache floor/2)",
         lambda c: c >= 50, 50, counts.max()),
        ("h >= 80 (Stage 03 MIN_SENTS=80)",
         lambda c: c >= 80, 80, counts.max()),
        ("h >= 100 (current PCFG cache floor)",
         lambda c: c >= 100, 100, counts.max()),
        ("h in [p5, p99]",
         lambda c: (c >= pct_vals[5]) & (c <= pct_vals[99]),
         pct_vals[5], pct_vals[99]),
        ("h in [p10, p99]",
         lambda c: (c >= pct_vals[10]) & (c <= pct_vals[99]),
         pct_vals[10], pct_vals[99]),
        ("h in [p25, p99]",
         lambda c: (c >= pct_vals[25]) & (c <= pct_vals[99]),
         pct_vals[25], pct_vals[99]),
        (f"Tukey [{lo_log:.1f}, {hi_log:.1f}]",
         lambda c: (c >= lo_log) & (c <= hi_log),
         lo_log, hi_log),
    ]
    log("\n  candidate cutoffs (raw distribution ONLY):")
    log(f"    {'name':>38s}  | {'h_min':>7s} | {'h_max':>7s} | "
        f"{'n_users':>10s} | {'pct':>6s} | {'med':>7s}")
    log("    " + "-" * 95)
    eval_rows = []
    for name, mask_fn, lo, hi in candidates:
        mask = mask_fn(counts)
        n_pass = int(mask.sum())
        pct = 100.0 * n_pass / n_users
        h_pass = counts[mask]
        log(f"    {name:>38s}  | {lo:>7.0f} | {hi:>7.0f} | "
            f"{n_pass:>10,d} | {pct:>5.2f}% | "
            f"{np.median(h_pass):>7.0f}")
        eval_rows.append({
            "name": name,
            "h_min": float(lo),
            "h_max": float(hi),
            "n_users": n_pass,
            "pct_users": round(pct, 4),
            "h_median": float(np.median(h_pass)) if n_pass > 0 else None,
            "h_mean": float(h_pass.mean()) if n_pass > 0 else None,
        })

    # Log-binned histogram (full range from 1 to max)
    log_edges = np.logspace(0, np.log10(counts.max() + 1), 40)
    hist, _ = np.histogram(counts, bins=log_edges)
    log("\n  log-binned histogram (h_u, raw):")
    max_count = hist.max()
    bar_max = 50
    for i in range(len(hist)):
        lo_e = log_edges[i]
        hi_e = log_edges[i + 1]
        bar_len = int(bar_max * hist[i] / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        log(f"    [{lo_e:8.1f}, {hi_e:8.1f}] : {hist[i]:>10,d} {bar}")

    # Save histogram + summary
    np.savez(OUT_DIR / "pre_encoder_h_u_raw_histogram.npz",
             edges=log_edges, counts=hist)
    summary = {
        "n_users_total": int(n_users),
        "source": "Stage 02 uid_to_sentences.json (no PCFG / no filter)",
        "h_u_percentiles": {f"p{q}": pct_vals[q] for q in qs},
        "tukey_outlier_rule_log10_k1.5": {
            "Q1_log10": float(q1),
            "Q3_log10": float(q3),
            "IQR_log10": float(iqr),
            "lower_bound": float(lo_log),
            "upper_bound": float(hi_log),
            "n_in_bounds": n_in_tukey,
        },
        "candidate_cutoffs": eval_rows,
        "recommended_cutoff": {
            "rule": "[p5, p99]",
            "h_min": int(pct_vals[5]),
            "h_max": int(pct_vals[99]),
            "n_users": int(((counts >= pct_vals[5]) & (counts <= pct_vals[99])).sum()),
            "rationale": (
                "raw 全量分布,无任何 PCFG/min-sents/encoder 介入;"
                "p5 切掉极低活跃长尾,p99 切掉 top-1% 超活跃 outlier"
            ),
        },
        "compares_to_pcfg_cache": {
            "pcfg_cache_floor_100_users": 3664,
            "stage03_min80_users": 5558,
            "raw_dist_pct_at_floor_100":
                float(100.0 * (counts >= 100).sum() / n_users),
            "raw_dist_pct_at_floor_80":
                float(100.0 * (counts >= 80).sum() / n_users),
        },
    }
    with open(OUT_DIR / "pre_encoder_raw_distribution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n  wrote summary → "
        f"{OUT_DIR / 'pre_encoder_raw_distribution_summary.json'}")

    log(f"\nALL DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()