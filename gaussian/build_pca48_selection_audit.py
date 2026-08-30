#!/usr/bin/env python3
"""Phase 4.C — Stage 4 strict alignment + transfer diagnostics.

Per condition:
  1. Run Stage 4 strict alignment → selection entry has d_self + margin.
  2. Read pair_records (exemplar_d_self, query) and align by (user, asin, k).
  3. Per-pair transfer_gain = d_query_self(random) - d_query_self(condition)
  4. Spearman ρ(d_exemplar_self, d_query_self).

Decision (per user spec):
  GO            min_d_self drop ≥ 1.0 AND PASSED% up AND ρ > 0
  PARTIAL-GO    F3/exclusivity up but min_d_self drop < 0.5
  NO-GO         nearest ≈ random on min_d_self
  STRONG NO-GO  nearest ≈ farthest on min_d_self

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
LOG_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/logs")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

R_95 = 8.073
CONDITIONS = ["random", "nearest", "farthest"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pca48_audit] {msg}", flush=True)


def classify(d: float, m: float) -> str:
    if m > 0 and d > R_95: return "F1"
    if m <= 0 and d <= R_95: return "F2"
    if m <= 0 and d > R_95: return "F3"
    return "PASSED"


def ensure_cohort_alias() -> None:
    """Stage4 reads stage8_5_asins<SUFFIX>.json — alias Phase 4.B cohort."""
    cohort = SCRATCH / "stage8_5_asins_exemplar_count_sweep.json"
    if not cohort.exists():
        raise FileNotFoundError(f"missing cohort: {cohort}")
    for cond in CONDITIONS:
        alias = SCRATCH / f"stage8_5_asins_pca48_{cond}.json"
        if not alias.exists():
            import shutil
            shutil.copy2(cohort, alias)


def run_stage4(cond: str) -> dict:
    pool = SCRATCH / f"pool_pca48_selection_{cond}.json"
    sel = SCRATCH / f"stage8_5_selection_pca48_{cond}.json"
    if sel.exists():
        log(f"  Stage 4 {cond}: reusing {sel.name}")
        return {"sel_path": str(sel), "reused": True}
    feat_cache = SCRATCH / f"stage7b_query_features_pca48_{cond}.jsonl.gz"
    env = os.environ.copy()
    env["POOL_IN_LOCAL"] = str(pool)
    env["POOL_FEAT_CACHE"] = str(feat_cache)
    env["ASINS_OUT_SUFFIX"] = f"_pca48_{cond}"
    env["SEL_OUT_SUFFIX"] = f"_pca48_{cond}"
    env["STAGE4_USER_FILTER"] = "1"
    log_path = LOG_DIR / f"stage4_select_pca48_{cond}.log"
    t0 = time.time()
    with open(log_path, "w") as fl:
        rc = subprocess.run([PYTHON, "select_query/syntax_subspace_select_strict_alignment.py"],
                            cwd=str(REPO_ROOT), env=env, stdout=fl, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - t0
    log(f"  Stage 4 {cond} done in {elapsed:.1f}s rc={rc}")
    if rc != 0:
        raise RuntimeError(f"Stage 4 {cond} failed rc={rc}, see {log_path}")
    return {"sel_path": str(sel), "stage4_sec": elapsed}


def load_selection(cond: str) -> dict:
    """Load Stage4 selection → {(user, asin) → (d_self, margin, method)}."""
    sel = json.load(open(SCRATCH / f"stage8_5_selection_pca48_{cond}.json"))
    out = {}
    for e in sel["entries"]:
        if e.get("selection_method") == "no_strict_candidate":
            continue
        key = (e["user_id"], e["asin"])
        out[key] = {
            "d_self": float(e["selected_distance"]),
            "margin": float(e["selected_margin"]),
            "method": e["selection_method"],
        }
    return out


def load_pair_records(cond: str) -> dict:
    """Load pool_pca48_selection_{cond}.json → {(user, asin) → list of {k, exemplar_d_self, query}}."""
    pool = json.load(open(SCRATCH / f"pool_pca48_selection_{cond}.json"))
    out = {}
    for r in pool["pair_records"]:
        key = (r["user_id"], r["asin"])
        out.setdefault(key, []).append({
            "k": r["k"],
            "exemplar_d_self": r.get("exemplar_d_self"),
            "exemplar_text": r["exemplar_text"],
            "exemplar_source_asin": r.get("exemplar_source_asin"),
            "query": r["query"],
            "strict": r.get("strict", False),
        })
    return out


def audit_one(cond: str) -> dict:
    """Per-condition audit.

    Stage4 gives per-(user, asin) selection; pair_records has per-k records.
    For transfer diagnostics we treat each k as an independent data point,
    attributing the ASIN-level selected (d_self, margin) to each k.
    """
    sel_map = load_selection(cond)
    pr_map = load_pair_records(cond)
    common_keys = set(sel_map) & set(pr_map)

    buckets = {"F1":[], "F2":[], "F3":[], "PASSED":[]}
    exemplar_ds = []
    query_ds = []
    rows = []
    for key in common_keys:
        s = sel_map[key]
        d = s["d_self"]; m = s["margin"]
        cat = classify(d, m)
        buckets[cat].append((d, m))
        # Each (user, asin) expands into K=4 rows
        for p in pr_map[key]:
            ed = p.get("exemplar_d_self")
            if ed is not None:
                exemplar_ds.append(ed)
                query_ds.append(d)
            rows.append({
                "user_id": key[0], "asin": key[1], "k": p["k"],
                "exemplar_d_self": ed, "query_d_self": d,
                "margin": m, "category": cat,
            })

    n_pass = len(buckets["PASSED"]); n_f1 = len(buckets["F1"])
    n_f2 = len(buckets["F2"]); n_f3 = len(buckets["F3"])
    n_valid = n_pass + n_f1 + n_f2 + n_f3
    n_fail = n_f1 + n_f2 + n_f3

    f3_ds = sorted([x[0] for x in buckets["F3"]])
    f3_ms = sorted([x[1] for x in buckets["F3"]])

    # exemplar ρ(query distance) — computed over (user, asin) not (user, asin, k)
    rho = None; pval = None
    if len(exemplar_ds) >= 5:
        try:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(exemplar_ds, query_ds)
        except Exception:
            rho = float(np.corrcoef(exemplar_ds, query_ds)[0, 1])

    # median d_query_self (all strict (user, asin), regardless of category)
    all_query_ds = sorted([s["d_self"] for s in sel_map.values()])
    # median d_exemplar_self (over (user, asin), each takes one exemplar)
    all_exemplar_ds = []
    for key, ps in pr_map.items():
        if key in sel_map and ps and ps[0].get("exemplar_d_self") is not None:
            all_exemplar_ds.append(ps[0]["exemplar_d_self"])
    all_exemplar_ds.sort()

    log(f"  {cond}: PASSED={n_pass} ({100*n_pass/max(1,n_valid):.1f}%) | "
        f"F1={n_f1} F2={n_f2} F3={n_f3} ({100*n_f3/max(1,n_fail):.1f}% of fails)")
    if f3_ds:
        log(f"    F3 min_d_self median={f3_ds[len(f3_ds)//2]:.3f} Δd median={f3_ms[len(f3_ms)//2]:.3f}")
    if all_query_ds:
        log(f"    ALL strict query d_self median={all_query_ds[len(all_query_ds)//2]:.3f}")
    if all_exemplar_ds:
        log(f"    ALL exemplar d_self median={all_exemplar_ds[len(all_exemplar_ds)//2]:.3f}")
    if rho is not None:
        log(f"    ρ(exemplar d, query d) = {rho:+.3f}  (p={pval:.3g} if scipy available)")

    return {
        "condition": cond,
        "n_valid": n_valid, "n_pass": n_pass, "n_f1": n_f1, "n_f2": n_f2, "n_f3": n_f3,
        "PASSED_pct_valid": n_pass/max(1,n_valid),
        "F3_pct_fails": n_f3/max(1,n_fail),
        "F3_min_d_self_median": f3_ds[len(f3_ds)//2] if f3_ds else None,
        "F3_max_M_median": f3_ms[len(f3_ms)//2] if f3_ms else None,
        "ALL_query_d_self_median": all_query_ds[len(all_query_ds)//2] if all_query_ds else None,
        "ALL_exemplar_d_self_median": all_exemplar_ds[len(all_exemplar_ds)//2] if all_exemplar_ds else None,
        "spearman_rho_exemplar_query": float(rho) if rho is not None else None,
        "spearman_pval": float(pval) if pval is not None else None,
        "rows": rows,
    }


def paired_transfer(nearest: dict, random_: dict, farthest: dict) -> dict:
    """Per-(user, asin) transfer: d_query_self(random) - d_query_self(condition).

    Each k-row uses the same ASIN-level selected d_self, so we deduplicate
    to (user, asin) pairs before computing transfer.
    """
    def collect(d):
        return {r["user_id"]+"_"+r["asin"]: r["query_d_self"] for r in d["rows"]}
    nr = collect(nearest); rr = collect(random_); fr = collect(farthest)

    keys = set(nr) & set(rr)
    transfer_nearest_vs_random = sorted([rr[k] - nr[k] for k in keys])
    keys2 = set(nr) & set(fr)
    transfer_nearest_vs_farthest = sorted([fr[k] - nr[k] for k in keys2])

    out = {}
    if transfer_nearest_vs_random:
        med = transfer_nearest_vs_random[len(transfer_nearest_vs_random)//2]
        out["nearest_vs_random"] = {
            "n_pairs": len(transfer_nearest_vs_random),
            "median": med,
            "mean": float(np.mean(transfer_nearest_vs_random)),
            "frac_positive": sum(1 for x in transfer_nearest_vs_random if x > 0) / len(transfer_nearest_vs_random),
            "frac_negative": sum(1 for x in transfer_nearest_vs_random if x < 0) / len(transfer_nearest_vs_random),
        }
        log(f"  transfer nearest-vs-random: median={med:+.3f}  pos%={out['nearest_vs_random']['frac_positive']*100:.1f}%")
    if transfer_nearest_vs_farthest:
        med2 = transfer_nearest_vs_farthest[len(transfer_nearest_vs_farthest)//2]
        out["nearest_vs_farthest"] = {
            "n_pairs": len(transfer_nearest_vs_farthest),
            "median": med2,
            "mean": float(np.mean(transfer_nearest_vs_farthest)),
            "frac_positive": sum(1 for x in transfer_nearest_vs_farthest if x > 0) / len(transfer_nearest_vs_farthest),
            "frac_negative": sum(1 for x in transfer_nearest_vs_farthest if x < 0) / len(transfer_nearest_vs_farthest),
        }
        log(f"  transfer nearest-vs-farthest: median={med2:+.3f}  pos%={out['nearest_vs_farthest']['frac_positive']*100:.1f}%")
    return out


def main():
    log("=== Phase 4.C — PCA48-nearest selection audit ===")
    ensure_cohort_alias()

    # Stage 4 per condition
    for cond in CONDITIONS:
        run_stage4(cond)

    # Audit per condition
    audits = {cond: audit_one(cond) for cond in CONDITIONS}

    # Transfer diagnostics
    log("\n--- Per-pair transfer diagnostics ---")
    transfer = paired_transfer(audits["nearest"], audits["random"], audits["farthest"])

    # Decision
    nearest = audits["nearest"]; random_ = audits["random"]; farthest = audits["farthest"]
    min_d_n = nearest["ALL_query_d_self_median"]
    min_d_r = random_["ALL_query_d_self_median"]
    min_d_f = farthest["ALL_query_d_self_median"]
    pass_n = nearest["PASSED_pct_valid"]
    pass_r = random_["PASSED_pct_valid"]
    f3_n = nearest["F3_pct_fails"]
    f3_r = random_["F3_pct_fails"]
    rho_n = nearest["spearman_rho_exemplar_query"]

    drop_nr = (min_d_r - min_d_n) if (min_d_r and min_d_n) else 0
    drop_nf = (min_d_f - min_d_n) if (min_d_f and min_d_n) else 0
    pass_gain = pass_n - pass_r
    rho_val = rho_n if rho_n is not None else 0

    log(f"\n=== DECISION DIAGNOSTICS ===")
    log(f"  min_d_self:    nearest={min_d_n:.3f}  random={min_d_r:.3f}  farthest={min_d_f:.3f}")
    log(f"  drop nearest-random: {drop_nr:+.3f}   drop nearest-farthest: {drop_nf:+.3f}")
    log(f"  PASSED%:       nearest={100*pass_n:.1f}  random={100*pass_r:.1f}  gain={100*pass_gain:+.1f}pp")
    log(f"  F3%:           nearest={100*f3_n:.1f}  random={100*f3_r:.1f}")
    log(f"  ρ(exemplar, query): nearest={rho_val:+.3f}")

    if drop_nf is not None and abs(drop_nf) < 0.3:
        decision = "STRONG_NO_GO"
        rationale = (f"nearest 与 farthest 在 min_d_self 上几乎一样 (Δ={drop_nf:+.3f}). "
                     f"LLM 没有把 exemplar 的 PCA48 syntax location 传递到 query. "
                     f"下一步: Phase 5 — explicit syntax-conditioned generation.")
    elif drop_nr >= 1.0 and pass_gain >= 0.005 and rho_val > 0:
        decision = "GO"
        rationale = (f"nearest 比 random 降 min_d_self {drop_nr:+.3f} (≥1.0 ✓), "
                     f"PASSED% 升 {100*pass_gain:+.1f}pp ✓, ρ={rho_val:+.3f} > 0 ✓. "
                     f"PCA48-nearest selection 是有效杠杆.")
    elif drop_nr >= 0.5 and rho_val > 0:
        decision = "PARTIAL_GO"
        rationale = (f"nearest 降 min_d_self {drop_nr:+.3f} (<1.0, 信号弱但方向对). "
                     f"继续做 top-k nearest sampling 或 density-aware selection.")
    elif abs(drop_nr) < 0.3 and rho_val <= 0:
        decision = "NO_GO"
        rationale = (f"nearest 与 random 在 min_d_self 上几乎一样 (Δ={drop_nr:+.3f}), "
                     f"且 ρ≤0. PCA48-nearest selection 不解决问题.")
    else:
        decision = "AMBIGUOUS"
        rationale = "Mixed signals; need more analysis"

    log(f"\n  DECISION: {decision}")
    log(f"  {rationale}")

    out_doc = {
        "description": (
            "Phase 4.C PCA48-nearest exemplar selection. Locked N_EXEMPLARS=1. "
            "Three conditions: random / nearest / farthest. Same 50 ASINs × 167 "
            "(user, asin) pairs. Leakage control: exclude user sentences from "
            "target ASIN. Each pair records (exemplar_d_self, query_d_self) for "
            "transfer diagnostics."
        ),
        "config": {
            "N_EXEMPLARS": 1,
            "K_PER_PAIR": 4,
            "PILOT_N_ASIN": 50,
            "leakage_control": "exclude user sentences from target ASIN",
            "R_95": R_95,
        },
        "audits": {cond: {k: v for k, v in audits[cond].items() if k != "rows"}
                   for cond in CONDITIONS},
        "transfer_diagnostics": transfer,
        "decision_diagnostics": {
            "min_d_self": {"nearest": min_d_n, "random": min_d_r, "farthest": min_d_f},
            "PASSED%": {"nearest": pass_n, "random": pass_r},
            "F3%": {"nearest": f3_n, "random": f3_r},
            "rho_nearest": rho_val,
        },
        "decision": decision,
        "rationale": rationale,
        "next_step_hint": {
            "GO": "Phase 4.D — top-k nearest sampling + density-aware selection, "
                  "then scale to 2174 ASIN cohort",
            "PARTIAL_GO": "Phase 4.D — try top-k nearest with k ∈ {3,5,8}",
            "NO_GO": "Phase 5 — explicit syntax-conditioned generation "
                     "(syntax feature template, e.g. passive-voice 25-35 words)",
            "STRONG_NO_GO": "Stop optimizing exemplar entirely. "
                            "LLM cannot transfer PCA48 syntax location. "
                            "Move to syntax-conditioned generation.",
            "AMBIGUOUS": "Inspect per-pair ρ bootstrap CI; check if signal is "
                          "concentrated in a subset",
        },
    }
    out_path = REPO_ROOT / "result/gaussian/pca48_selection_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {out_path}")


if __name__ == "__main__":
    main()