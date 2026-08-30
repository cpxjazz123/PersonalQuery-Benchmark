#!/usr/bin/env python3
"""Phase 5 — Stage 4 strict alignment + paired causal transfer audit.

4 conditions: control / random / target / anti_target.
Causal chain: target_constraints vs anti_target → both vs control.

Per pair (user, asin):
    d_self[cond] for cond in {control, random, target, anti_target}
    Δd_target = d_self[target] - d_self[control]
    Δd_anti   = d_self[anti_target] - d_self[control]
    Δd_causal = d_self[target] - d_self[anti_target]   (净 target effect)

Decision:
    GO            median Δd_causal ≤ -1.0 (target pulls closer than anti_target)
    PARTIAL-GO    median Δd_causal in [-0.5, -0.1]
    NO-GO         median Δd_causal ≈ 0
    STRONG NO-GO  target ≈ anti_target in d_self distribution
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
LOG_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/logs")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

R_95 = 8.073
CONDITIONS = ["control", "random", "target", "anti_target"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [constraints_audit] {msg}", flush=True)


def classify(d: float, m: float) -> str:
    if m > 0 and d > R_95: return "F1"
    if m <= 0 and d <= R_95: return "F2"
    if m <= 0 and d > R_95: return "F3"
    return "PASSED"


def ensure_cohort_alias() -> None:
    cohort = SCRATCH / "stage8_5_asins_exemplar_count_sweep.json"
    for cond in CONDITIONS:
        alias = SCRATCH / f"stage8_5_asins_constraints_{cond}.json"
        if not alias.exists():
            shutil.copy2(cohort, alias)


def run_stage4(cond: str) -> dict:
    pool = SCRATCH / f"pool_constraints_{cond}.json"
    sel = SCRATCH / f"stage8_5_selection_constraints_{cond}.json"
    if sel.exists():
        log(f"  Stage 4 {cond}: reusing {sel.name}")
        return {"sel_path": str(sel), "reused": True}
    feat_cache = SCRATCH / f"stage7b_query_features_constraints_{cond}.jsonl.gz"
    env = os.environ.copy()
    env["POOL_IN_LOCAL"] = str(pool)
    env["POOL_FEAT_CACHE"] = str(feat_cache)
    env["ASINS_OUT_SUFFIX"] = f"_constraints_{cond}"
    env["SEL_OUT_SUFFIX"] = f"_constraints_{cond}"
    env["STAGE4_USER_FILTER"] = "1"
    log_path = LOG_DIR / f"stage4_select_constraints_{cond}.log"
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
    sel = json.load(open(SCRATCH / f"stage8_5_selection_constraints_{cond}.json"))
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


def audit_one(cond: str) -> dict:
    sel_map = load_selection(cond)
    buckets = {"F1":[], "F2":[], "F3":[], "PASSED":[]}
    rows = []
    for key, s in sel_map.items():
        d, m = s["d_self"], s["margin"]
        cat = classify(d, m)
        buckets[cat].append((d, m))
        rows.append({
            "user_id": key[0], "asin": key[1],
            "d_self": d, "margin": m, "category": cat,
        })

    n_pass = len(buckets["PASSED"]); n_f1 = len(buckets["F1"])
    n_f2 = len(buckets["F2"]); n_f3 = len(buckets["F3"])
    n_valid = n_pass + n_f1 + n_f2 + n_f3
    n_fail = n_f1 + n_f2 + n_f3

    f3_ds = sorted([x[0] for x in buckets["F3"]])
    all_query_ds = sorted([s["d_self"] for s in sel_map.values()])
    med_d = all_query_ds[len(all_query_ds)//2] if all_query_ds else None

    log(f"  {cond}: PASSED={n_pass} ({100*n_pass/max(1,n_valid):.1f}%) | "
        f"F1={n_f1} F2={n_f2} F3={n_f3} ({100*n_f3/max(1,n_fail):.1f}% of fails) | "
        f"ALL query d_self median={med_d:.3f}" if med_d else f"  {cond}: empty")

    return {
        "condition": cond, "n_valid": n_valid,
        "n_pass": n_pass, "n_f1": n_f1, "n_f2": n_f2, "n_f3": n_f3,
        "PASSED_pct_valid": n_pass/max(1,n_valid),
        "F3_pct_fails": n_f3/max(1,n_fail),
        "ALL_query_d_self_median": med_d,
        "rows": rows,
    }


def paired_transfer(audits: dict) -> dict:
    """Per-(user, asin) transfer: d_self[cond] - d_self[control]."""
    ctrl = {r["user_id"]+"_"+r["asin"]: r["d_self"] for r in audits["control"]["rows"]}
    out = {}
    for cond in ["random", "target", "anti_target"]:
        c = {r["user_id"]+"_"+r["asin"]: r["d_self"] for r in audits[cond]["rows"]}
        keys = set(ctrl) & set(c)
        diffs = sorted([c[k] - ctrl[k] for k in keys])
        out[f"{cond}_vs_control"] = {
            "n_pairs": len(diffs),
            "median": diffs[len(diffs)//2] if diffs else None,
            "mean": float(np.mean(diffs)) if diffs else None,
            "frac_negative": sum(1 for x in diffs if x < 0) / max(1, len(diffs)),
        }
    # Causal effect: target - anti_target
    t = {r["user_id"]+"_"+r["asin"]: r["d_self"] for r in audits["target"]["rows"]}
    a = {r["user_id"]+"_"+r["asin"]: r["d_self"] for r in audits["anti_target"]["rows"]}
    keys = set(t) & set(a)
    causal = sorted([a[k] - t[k] for k in keys])  # positive = target < anti_target (target closer)
    out["anti_target_vs_target"] = {
        "n_pairs": len(causal),
        "median": causal[len(causal)//2] if causal else None,
        "mean": float(np.mean(causal)) if causal else None,
        "frac_positive": sum(1 for x in causal if x > 0) / max(1, len(causal)),
    }
    return out


def main():
    log("=== Phase 5 — Explicit syntax-conditioned generation audit ===")
    ensure_cohort_alias()

    for cond in CONDITIONS:
        run_stage4(cond)

    audits = {cond: audit_one(cond) for cond in CONDITIONS}
    transfer = paired_transfer(audits)

    log("\n--- Per-pair transfer diagnostics ---")
    for k, v in transfer.items():
        log(f"  {k}: median={v['median']:+.3f}  "
            f"frac_neg={(v.get('frac_negative', v.get('frac_positive',0)))*100:.1f}%")

    # Decision
    target_med = audits["target"]["ALL_query_d_self_median"]
    anti_med = audits["anti_target"]["ALL_query_d_self_median"]
    ctrl_med = audits["control"]["ALL_query_d_self_median"]
    causal_med = transfer["anti_target_vs_target"]["median"]

    log(f"\n=== DECISION DIAGNOSTICS ===")
    log(f"  median query d_self: control={ctrl_med:.3f}  target={target_med:.3f}  anti_target={anti_med:.3f}")
    log(f"  causal effect (anti_target - target) median: {causal_med:+.3f}")
    log(f"    positive = target closer to user (target pulled closer than anti)")
    log(f"    negative = anti_target closer (constraint is backfiring)")

    if causal_med >= 1.0:
        decision = "GO"
        rationale = (f"target_constraints 显著拉近 (causal effect {causal_med:+.3f}). "
                     f"explicit syntax control 可控 PCA48 location. 下一步: 扩到全 2174 ASIN.")
    elif causal_med >= 0.5:
        decision = "PARTIAL_GO"
        rationale = (f"target_constraints 部分起效 (causal effect {causal_med:+.3f}). "
                     f"继续优化 constraint 模板或增加 top-K.")
    elif causal_med >= -0.3:
        decision = "NO_GO"
        rationale = (f"target ≈ anti_target (causal effect {causal_med:+.3f}). "
                     f"LLM 无法被 NL instructions 推到具体 PCA48 location.")
    else:
        decision = "STRONG_NO_GO"
        rationale = (f"anti_target 比 target 更接近 user (causal effect {causal_med:+.3f}). "
                     f"constraint 反而把 query 推离. 结论: explicit syntax control "
                     f"不能控制 PCA48 location, 瓶颈是 LLM controllability / "
                     f"syntax representation gap.")

    log(f"\n  DECISION: {decision}")
    log(f"  {rationale}")

    out_doc = {
        "description": (
            "Phase 5 explicit syntax-conditioned generation. Locked N_EXEMPLARS=0, "
            "same 50 ASINs × 169 (user, asin) pairs. 4 conditions: control (no "
            "syntax constraint), random (5 random F3 features), target (top-5 "
            "deviating F3 features from μ_u in cohort reverse-mapped to F3), "
            "anti_target (target features with sign flipped). Tests causal "
            "chain: explicit syntax control → controllable PCA48 location."
        ),
        "config": {
            "N_EXEMPLARS": 0, "K_PER_PAIR": 4, "TOP_K_CONSTRAINTS": 5,
            "TEMP": 0.7, "PILOT_N_ASIN": 50, "R_95": R_95,
        },
        "audits": {cond: {k: v for k, v in audits[cond].items() if k != "rows"}
                   for cond in CONDITIONS},
        "transfer_diagnostics": transfer,
        "decision_diagnostics": {
            "control_d_self_median": ctrl_med,
            "target_d_self_median": target_med,
            "anti_target_d_self_median": anti_med,
            "causal_effect_median": causal_med,
        },
        "decision": decision,
        "rationale": rationale,
        "next_step_hint": {
            "GO": "Phase 5.B — increase TOP_K_CONSTRAINTS to {8, 12}, "
                  "test if more constraints = better. Then scale to 2174 ASIN cohort.",
            "PARTIAL_GO": "Phase 5.B — try more diverse F3 features or longer "
                          "constraint descriptions.",
            "NO_GO": "Phase 6 — explore alternative generation paths "
                     "(e.g. constrained decoding, syntax-aware finetuning, "
                     "or per-user syntax templates).",
            "STRONG_NO_GO": "Stop optimizing generation. The bottleneck is the "
                             "syntax representation's disconnect from controllable "
                             "generation. Document the gap as a finding.",
        },
    }
    out_path = REPO_ROOT / "result/gaussian/pca48_constraints_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {out_path}")


if __name__ == "__main__":
    main()