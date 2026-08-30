#!/usr/bin/env python3
"""Phase 4.B Stage 4 + Audit for each N_EXEMPLARS pool.

对每个 N_EXEMPLARS pool:
1. 跑 Stage 4 strict alignment (用 POOL_IN_LOCAL env)
2. 读 selection entry, 提取:
   - F3% / min_d_self median / max_M median (== Δd median)
   - PASSED%
3. 写到 result/gaussian/exemplar_count_sweep_audit.json

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
LOG_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/logs")
PYTHON = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"

R_95 = 8.073

# Same cohort construction as Phase 4 (50 ASINs, 169 pairs)
PILOT_N_ASIN = 50
N_SAMPLE_USERS = 20
RANDOM_SEED_ASIN = 42
RANDOM_SEED_USERS = 0
COHORT_ASINS = SCRATCH / "stage8_5_asins_strict34_intersect2174_ksweep_K200.json"

# N_EXEMPLARS values to audit
N_EXEMPLARS_VALUES = [1, 2, 3, 5, 8]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [count_audit] {msg}", flush=True)


def write_cohort() -> None:
    """Write 50-ASIN cohort subset with 169 (user, asin) pairs."""
    cohort = json.load(open(COHORT_ASINS))
    rng_asin = random.Random(RANDOM_SEED_ASIN)
    s = sorted(cohort["asins"], key=lambda x: x["asin"])
    rng_asin.shuffle(s)
    kept_asins = s[:PILOT_N_ASIN]

    # First pass: collect all users (same logic as Phase 4 generation)
    all_users = set()
    rng_users = random.Random(RANDOM_SEED_USERS)
    for entry in kept_asins:
        users = entry["users_sampled"][:N_SAMPLE_USERS]
        rng_users_local = random.Random(RANDOM_SEED_USERS)
        rng_users_local.shuffle(users)
        for u in users:
            all_users.add(u)

    # Find user_sentences map
    sent_by_user = {}
    with open(SCRATCH / "sentences_for_rewrite_10k.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("sentence_text", "").strip()
            wc = len(text.split())
            if wc < 5 or wc > 60:
                continue
            sent_by_user.setdefault(rec["user_id"], []).append(text)
    valid_users = all_users & set(sent_by_user.keys())
    log(f"  cohort: {len(kept_asins)} ASINs, {len(valid_users)} users with sentences")

    # Write cohort subset
    kept_asins_full = []
    for entry in kept_asins:
        users = entry["users_sampled"][:N_SAMPLE_USERS]
        rng_users_local = random.Random(RANDOM_SEED_USERS)
        rng_users_local.shuffle(users)
        users = [u for u in users if u in valid_users]
        entry2 = dict(entry)
        entry2["users_sampled"] = users
        entry2["n_users_quality"] = len(users)
        kept_asins_full.append(entry2)
    out_doc = {
        "config": {"filter": "strict34_intersect_exemplar_count_sweep"},
        "n_asins": len(kept_asins_full),
        "n_users_total": sum(e["n_users_quality"] for e in kept_asins_full),
        "asins": kept_asins_full,
    }
    with open(SCRATCH / "stage8_5_asins_exemplar_count_sweep.json", "w") as f:
        json.dump(out_doc, f, ensure_ascii=False)
    log(f"  wrote cohort → stage8_5_asins_exemplar_count_sweep.json")


def run_stage4(N: int) -> dict:
    if N == 2:
        # Phase 4 already produced this selection in commit a10987d.
        # Reuse directly — same pool, same config, same cohort.
        src = SCRATCH / "stage8_5_selection_exemplar_pilot.json"
        dst = SCRATCH / "stage8_5_selection_exemplar_N2.json"
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)
            log(f"  reused Phase 4 selection: {src.name} -> {dst.name}")
        return {"N": 2, "stage4_sec": 0.0, "sel_path": str(dst), "reused": True}

    pool_path = SCRATCH / f"pool_exemplar_anchor_N{N}.json"
    sel_out = SCRATCH / f"stage8_5_selection_exemplar_N{N}.json"
    feat_cache = SCRATCH / f"stage7b_query_features_exemplar_N{N}.jsonl.gz"
    log(f"=== Stage 4 N={N}: pool={pool_path.name} ===")

    # Stage4 reads cohort from stage8_5_asins<SUFFIX>.json — alias our
    # 50-ASIN pilot cohort to the per-N suffix so the same 169-pair cohort
    # is used regardless of N_EXEMPLARS.
    asins_alias = SCRATCH / f"stage8_5_asins_exemplar_N{N}.json"
    cohort_for_filter = SCRATCH / "stage8_5_asins_exemplar_count_sweep.json"
    if not asins_alias.exists():
        import shutil
        shutil.copy2(cohort_for_filter, asins_alias)
        log(f"  aliased {cohort_for_filter.name} -> {asins_alias.name}")

    env = os.environ.copy()
    env["POOL_IN_LOCAL"] = str(pool_path)
    env["POOL_FEAT_CACHE"] = str(feat_cache)
    env["ASINS_OUT_SUFFIX"] = f"_exemplar_N{N}"
    env["SEL_OUT_SUFFIX"] = f"_exemplar_N{N}"
    env["STAGE4_USER_FILTER"] = "1"
    log_path = LOG_DIR / f"stage4_select_exemplar_N{N}.log"
    t0 = time.time()
    with open(log_path, "w") as fl:
        rc = subprocess.run([PYTHON, "select_query/syntax_subspace_select_strict_alignment.py"],
                            cwd=str(REPO_ROOT), env=env, stdout=fl, stderr=subprocess.STDOUT).returncode
    elapsed = time.time() - t0
    log(f"  Stage 4 N={N} done in {elapsed:.1f}s rc={rc}")
    if rc != 0:
        raise RuntimeError(f"Stage 4 N={N} failed rc={rc}, see {log_path}")
    return {"N": N, "stage4_sec": elapsed, "sel_path": str(sel_out)}


def classify(d, m):
    if m > 0 and d > R_95: return "F1"
    if m <= 0 and d <= R_95: return "F2"
    if m <= 0 and d > R_95: return "F3"
    return "PASSED"


def audit_one(N: int) -> dict:
    sel_path = SCRATCH / f"stage8_5_selection_exemplar_N{N}.json"
    if not sel_path.exists():
        raise FileNotFoundError(f"selection required: {sel_path}")
    with open(sel_path) as f:
        sd = json.load(f)
    buckets = {"F1":[], "F2":[], "F3":[], "PASSED":[]}
    all_margins = []
    for e in sd["entries"]:
        sel_d = e.get("selected_distance"); sel_m = e.get("selected_margin")
        if sel_d is None or sel_m is None: continue
        cat = classify(float(sel_d), float(sel_m))
        if e.get("selection_method") == "strict_personalized_mahal_margin_max":
            cat = "PASSED"
        if e.get("selection_method") != "no_strict_candidate" and cat != "PASSED":
            continue
        buckets[cat].append({"d": float(sel_d), "m": float(sel_m)})
        all_margins.append(float(sel_m))
    n_pass = len(buckets["PASSED"]); n_f1 = len(buckets["F1"])
    n_f2 = len(buckets["F2"]); n_f3 = len(buckets["F3"])
    n_valid = n_pass + n_f1 + n_f2 + n_f3
    n_fail = n_f1 + n_f2 + n_f3

    f3_d = sorted([x["d"] for x in buckets["F3"]])
    f3_m = sorted([x["m"] for x in buckets["F3"]])

    log(f"  N={N}: PASSED={n_pass} ({100*n_pass/max(1,n_valid):.2f}%) "
        f"| F1={n_f1} F2={n_f2} F3={n_f3} ({100*n_f3/max(1,n_fail):.1f}% of fails)")
    if f3_d:
        log(f"    F3 min_d_self median: {f3_d[len(f3_d)//2]:.3f}")
        log(f"    F3 max_M (=Δd) median: {f3_m[len(f3_m)//2]:.3f}")
    if all_margins:
        sorted_m = sorted(all_margins)
        log(f"    All-pairs Δd median: {sorted_m[len(sorted_m)//2]:.3f}")

    return {
        "N_EXEMPLARS": N,
        "n_valid": n_valid,
        "n_pass": n_pass,
        "n_f1": n_f1, "n_f2": n_f2, "n_f3": n_f3,
        "PASSED_pct_valid": n_pass/max(1,n_valid),
        "F3_pct_fails": n_f3/max(1,n_fail),
        "F3_min_d_self_median": f3_d[len(f3_d)//2] if f3_d else None,
        "F3_max_M_median": f3_m[len(f3_m)//2] if f3_m else None,
        "all_pairs_delta_d_median": sorted(all_margins)[len(all_margins)//2] if all_margins else None,
    }


def main():
    log(f"=== Phase 4.B — N_EXEMPLARS sweep audit ===")
    log(f"  N values: {N_EXEMPLARS_VALUES}")
    write_cohort()

    # Stage 4 for each N
    for N in N_EXEMPLARS_VALUES:
        if not (SCRATCH / f"stage8_5_selection_exemplar_N{N}.json").exists():
            run_stage4(N)

    # Audit each N
    audits = {}
    for N in N_EXEMPLARS_VALUES:
        audits[str(N)] = audit_one(N)

    # Add corrected baseline reference
    audits["0_corrected_baseline"] = {
        "description": "reference: corrected baseline pilot (no exemplar)",
        "F3_pct_fails": 0.857,
        "F3_min_d_self_median": 11.12,
        "F3_max_M_median": -2.09,
        "PASSED_pct_valid": 0.0451,
    }

    # Sweep trend + decision
    f3_trend = [audits[str(N)]["F3_pct_fails"] for N in N_EXEMPLARS_VALUES]
    mind_trend = [audits[str(N)]["F3_min_d_self_median"] for N in N_EXEMPLARS_VALUES]
    margin_trend = [audits[str(N)]["F3_max_M_median"] for N in N_EXEMPLARS_VALUES]
    pass_trend = [audits[str(N)]["PASSED_pct_valid"] for N in N_EXEMPLARS_VALUES]

    log(f"\n=== SWEEP TREND ===")
    log(f"  N     : {N_EXEMPLARS_VALUES}")
    log(f"  F3%   : {[f'{x*100:.1f}' for x in f3_trend]}")
    log(f"  min_d : {[f'{x:.3f}' if x else 'N/A' for x in mind_trend]}")
    log(f"  Δd    : {[f'{x:.3f}' if x else 'N/A' for x in margin_trend]}")
    log(f"  PASSED: {[f'{x*100:.1f}' for x in pass_trend]}")

    # Decision (per user spec)
    f3_drop_N2_to_N8 = (f3_trend[1] - f3_trend[-1]) * 100  # N=2 → N=8 in pp
    mind_delta_N2_to_N8 = (mind_trend[1] - mind_trend[-1]) if (mind_trend[1] and mind_trend[-1]) else None
    margin_delta_N2_to_N8 = (margin_trend[-1] - margin_trend[1]) if (margin_trend[1] and margin_trend[-1]) else None

    if f3_drop_N2_to_N8 >= 5.0 and mind_delta_N2_to_N8 and mind_delta_N2_to_N8 >= 0.5:
        decision = "GO_EXEMPLAR_QUANTITY_BOTTLENECK"
        rationale = (f"F3 持续下降 (N=2→8: Δ={f3_drop_N2_to_N8:.1f}pp), "
                     f"min_d_self 也下降 (Δ={mind_delta_N2_to_N8:.2f}) — "
                     f"exemplar 数量确实是瓶颈")
    elif f3_drop_N2_to_N8 >= 5.0 and (mind_delta_N2_to_N8 is None or mind_delta_N2_to_N8 < 0.5):
        decision = "PARTIAL_EXCLUSIVITY_ONLY"
        rationale = (f"F3 下降 (Δ={f3_drop_N2_to_N8:.1f}pp) 但 min_d_self 没怎么动 "
                    f"(Δ={mind_delta_N2_to_N8 if mind_delta_N2_to_N8 is not None else 'N/A'}) — "
                    f"exemplar 只提升 exclusivity, 不进 PCA48 区域. "
                    f"下一步: PCA48-nearest exemplar selection")
    elif abs(f3_drop_N2_to_N8) < 3.0:
        decision = "STOP_SCALING_EXEMPLARS"
        rationale = (f"F3 在 N=2→8 几乎不变 (Δ={f3_drop_N2_to_N8:+.1f}pp), "
                     f"数量不是瓶颈. 下一步: exemplar selection (PCA48-nearest)")
    else:
        decision = "AMBIGUOUS"
        rationale = "Mixed signals; recommend combined PCA48-nearest + larger N"

    log(f"\n  DECISION: {decision}")
    log(f"  {rationale}")

    out = {
        "description": (
            "Phase 4.B N_EXEMPLARS sweep {1,2,3,5,8}. Same 50 ASINs, same 169 pairs, "
            "same K_PER_PAIR=4. Only N_EXEMPLARS varies. Spread-sample (no consecutive). "
            "Each pool run through Stage 4 strict alignment (T=34, F3+PCA48, "
            "M>0 + d_self≤R_95=8.073). Δd = d_nearest_other - d_self = selected_margin."
        ),
        "config": {
            "PILOT_N_ASIN": PILOT_N_ASIN,
            "N_SAMPLE_USERS": N_SAMPLE_USERS,
            "K_PER_PAIR": 4,
            "R_95": R_95,
            "TEMP": 0.7,
        },
        "N_EXEMPLARS_values": N_EXEMPLARS_VALUES,
        "audits": audits,
        "trend": {
            "F3_pct_fails": {str(N): f3_trend[i] for i, N in enumerate(N_EXEMPLARS_VALUES)},
            "F3_min_d_self_median": {str(N): mind_trend[i] for i, N in enumerate(N_EXEMPLARS_VALUES)},
            "F3_max_M_median": {str(N): margin_trend[i] for i, N in enumerate(N_EXEMPLARS_VALUES)},
            "PASSED_pct_valid": {str(N): pass_trend[i] for i, N in enumerate(N_EXEMPLARS_VALUES)},
        },
        "decision": decision,
        "rationale": rationale,
        "next_step_hint": {
            "GO_EXEMPLAR_QUANTITY_BOTTLENECK":
                "Adopt best N (likely N=8) and run full-scale 2174 ASIN.",
            "PARTIAL_EXCLUSIVITY_ONLY":
                "Move to PCA48-nearest exemplar selection: pick exemplars that "
                "lie in the user's PCA48 region (computed via user_gaussians.json).",
            "STOP_SCALING_EXEMPLARS":
                "Quantity is not the bottleneck. Switch to exemplar SELECTION quality.",
            "AMBIGUOUS":
                "Combine PCA48-nearest + larger N.",
        },
    }
    out_path = REPO_ROOT / "result/gaussian/exemplar_count_sweep_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {out_path}")


if __name__ == "__main__":
    main()