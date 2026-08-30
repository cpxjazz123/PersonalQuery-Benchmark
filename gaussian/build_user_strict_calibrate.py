#!/usr/bin/env python3
"""High-Confidence Gaussian Quality Calibration — Precision-priority CV-NLL threshold.

用户指令 2026-08-30: 不要 ROC 平衡阈值, 改用 **Precision-priority** calibration:

  T_p = max T : Precision(T) ≥ p%,  其中 p ∈ {95, 97.5, 99}

Calibration set (70% by sha1 hash): 找 T_95, T_97.5, T_99
Validation set (30%): 验证 Precision 在独立 user 上稳定
ASIN coverage: 每个 threshold 下还剩多少 ASIN 有 ≥2/≥5/≥10 quality users

最终推荐 threshold: validation precision ≥ 97.5% 同时仍保留足够 ASIN coverage 的最宽松 threshold.

**输入**:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians_reliability.json
  (delta_nll_ci_lo, reliable)
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians_cv.json
  (cv_nll, cv_nll_std)
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json
  (per-ASIN user list → ASIN coverage)

**输出**:
- /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/strict_calibration.json
  (operating points table + ASIN coverage + recommendation)
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_quality_strict_users.json
  (recommended threshold 的 user_id 列表, 下游 Stage 4 cohort 用)

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

RELIABILITY_OUT = SCRATCH / "stage8_5_user_gaussians_reliability.json"
CV_OUT = SCRATCH / "stage8_5_user_gaussians_cv.json"
ASINS_JSON = SCRATCH / "stage8_5_asins.json"

CALIBRATION_OUT = REPO_ROOT / "result" / "gaussian" / "strict_calibration.json"
STRICT_USERS_OUT = SCRATCH / "stage8_5_quality_strict_users.json"

# --- Calibration config (硬编码) ---
SPLIT_HASH_SEED = "calibrate_v1"   # MUST match build_user_calibrate.py
CALIBRATION_FRAC = 0.7
T_SCAN_MIN = 10
T_SCAN_MAX = 39
T_SCAN_STEPS = 30  # T = 10, 11, ..., 39

PRECISION_TARGETS = [0.95, 0.975, 0.99]
RECOMMENDED_TARGET = 0.975  # 用户推荐目标

# Recommended final threshold: validation precision ≥ RECOMMENDED_TARGET 且最大 n_asins_ge5
MIN_ASINS_GE5 = 100  # 推荐至少保留 100 个 ASIN 有 ≥5 quality users


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [strict_cal] {msg}", flush=True)


def user_split_bucket(uid: str) -> int:
    """Must match build_user_calibrate.user_split_bucket."""
    h = hashlib.sha1((SPLIT_HASH_SEED + "|" + uid).encode("utf-8")).hexdigest()
    return int(h, 16) % 100 < int(CALIBRATION_FRAC * 100)


def main():
    log(f"=== High-Confidence Gaussian Calibration (precision-priority) ===")
    log(f"  SPLIT_HASH_SEED={SPLIT_HASH_SEED}, CALIBRATION_FRAC={CALIBRATION_FRAC}")
    log(f"  T_SCAN=[{T_SCAN_MIN}, {T_SCAN_MAX}], steps={T_SCAN_STEPS}")
    log(f"  Precision targets: {PRECISION_TARGETS}")
    log(f"  Recommended target: {RECOMMENDED_TARGET} (with ≥{MIN_ASINS_GE5} ASINs at ≥5 users)")

    for f in [RELIABILITY_OUT, CV_OUT, ASINS_JSON]:
        if not f.exists():
            raise FileNotFoundError(f"required: {f}")

    # ====== Load caches ======
    log(f"\n=== Loading caches ===")
    t0 = time.time()
    with open(RELIABILITY_OUT, "r", encoding="utf-8") as f:
        rdoc = json.load(f)
    rel_users = rdoc["users"]
    log(f"  reliability: {len(rel_users)} users in {time.time()-t0:.1f}s")
    t0 = time.time()
    with open(CV_OUT, "r", encoding="utf-8") as f:
        cdoc = json.load(f)
    cv_users = cdoc["users"]
    log(f"  cv: {len(cv_users)} users in {time.time()-t0:.1f}s")
    t0 = time.time()
    with open(ASINS_JSON, "r", encoding="utf-8") as f:
        asins_doc = json.load(f)
    asins = asins_doc.get("asins", [])
    log(f"  asins: {len(asins)} ASINs in {time.time()-t0:.1f}s")

    # ====== Build (uid → cv_nll, reliable) tuples ======
    log(f"\n=== Building user tuples ===")
    common = set(rel_users.keys()) & set(cv_users.keys())
    log(f"  intersection (rel ∩ cv): {len(common)} users")
    user_data: dict = {}  # uid → {cv_nll, reliable, cv_nll_std, delta_nll_mean, delta_nll_ci_lo}
    for uid in common:
        rg = rel_users[uid]
        cg = cv_users[uid]
        cv_nll = cg.get("cv_nll")
        ci_lo = rg.get("delta_nll_ci_lo")
        if cv_nll is None or ci_lo is None:
            continue
        user_data[uid] = {
            "cv_nll": float(cv_nll),
            "cv_nll_std": float(cg.get("cv_nll_std") or 0.0),
            "delta_nll_mean": float(rg.get("delta_nll_mean") or 0.0),
            "delta_nll_ci_lo": float(ci_lo),
            "delta_nll_ci_hi": float(rg.get("delta_nll_ci_hi") or 0.0),
            "reliable": int(rg.get("reliable") or 0),
        }
    log(f"  users with both cv_nll + ci_lo: {len(user_data)}")

    # ====== Build ASIN → set(uid) map for ASIN coverage ======
    log(f"\n=== Building ASIN → users map ===")
    import ast
    asin_to_users: dict = {}
    n_asins_with_users = 0
    for entry in asins:
        a = entry.get("asin")
        us = entry.get("users_sampled")
        if not a or not us:
            continue
        try:
            uid_list = ast.literal_eval(us) if isinstance(us, str) else us
        except (ValueError, SyntaxError):
            continue
        if not uid_list:
            continue
        asin_to_users[a] = set(uid_list)
        n_asins_with_users += 1
    log(f"  ASINs with users_sampled: {n_asins_with_users}")

    # ====== 70/30 split (matches build_user_calibrate) ======
    cal_uids = [u for u in user_data if user_split_bucket(u) == 0]
    val_uids = [u for u in user_data if user_split_bucket(u) == 1]
    log(f"  split: calibration={len(cal_uids)}, validation={len(val_uids)}")

    cal_cv_nll = np.array([user_data[u]["cv_nll"] for u in cal_uids])
    cal_label = np.array([user_data[u]["reliable"] for u in cal_uids])
    val_cv_nll = np.array([user_data[u]["cv_nll"] for u in val_uids])
    val_label = np.array([user_data[u]["reliable"] for u in val_uids])
    log(f"  calibration: {int(cal_label.sum())} reliable ({cal_label.mean():.1%})")
    log(f"  validation:  {int(val_label.sum())} reliable ({val_label.mean():.1%})")

    # Pre-compute full user_data dict for lookup
    cal_uid_set = set(cal_uids)
    val_uid_set = set(val_uids)

    # ====== Step 1: scan T on calibration set ======
    log(f"\n=== Calibration scan: T ∈ [{T_SCAN_MIN}, {T_SCAN_MAX}] ===")
    T_grid = list(range(T_SCAN_MIN, T_SCAN_MAX + 1))
    cal_results = []
    for T in T_grid:
        pred = (cal_cv_nll <= T)
        n_pred = int(pred.sum())
        if n_pred == 0:
            cal_results.append({"T": T, "n_pred": 0, "precision": None, "recall": 0.0, "F1": None})
            log(f"  T={T:>3}: n=0 (skip)")
            continue
        tp = int((pred & (cal_label == 1)).sum())
        fp = int((pred & (cal_label == 0)).sum())
        fn = int(((~pred) & (cal_label == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        cal_results.append({
            "T": T, "n_pred": n_pred, "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "F1": f1,
        })
        log(f"  T={T:>3}: n={n_pred:>6} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

    # ====== Step 2: find T_p = max T : Precision(T) ≥ p ======
    log(f"\n=== Find T_p = max T : Precision(T) ≥ p ===")
    targets_found = {}
    for p in PRECISION_TARGETS:
        candidates = [r for r in cal_results
                      if r["precision"] is not None and r["precision"] >= p]
        if not candidates:
            log(f"  p={p:.3f}: NOT ACHIEVABLE in scan range")
            targets_found[p] = None
            continue
        best = max(candidates, key=lambda r: r["T"])  # max T
        log(f"  p={p:.3f}: T_p={best['T']} (P={best['precision']:.4f}, R={best['recall']:.4f}, "
            f"n_cal={best['n_pred']})")
        targets_found[p] = best

    # ====== Step 3: validate on 30% held-out ======
    log(f"\n=== Validation on 30% held-out ===")
    val_results = {}
    for p in PRECISION_TARGETS:
        T_p = targets_found[p]
        if T_p is None:
            val_results[p] = None
            continue
        T = T_p["T"]
        pred = (val_cv_nll <= T)
        n_pred = int(pred.sum())
        if n_pred == 0:
            val_results[p] = {"T": T, "n_pass": 0}
            log(f"  T_p={T} (p={p}): 0 users pass in validation (FAIL)")
            continue
        tp = int((pred & (val_label == 1)).sum())
        fp = int((pred & (val_label == 0)).sum())
        fn = int(((~pred) & (val_label == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        val_results[p] = {
            "T": T, "TP": tp, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "F1": f1, "n_pass": n_pred,
        }
        log(f"  T_p={T} (target p={p}): n={n_pred} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

    # ====== Step 4: ASIN coverage at each threshold ======
    log(f"\n=== ASIN coverage ===")
    # Build full uid pool = {uid in user_data (both cal+val)} - those who pass the threshold
    # For ASIN coverage, we care about ALL users who pass (calibration ∪ validation)

    def compute_asin_coverage(threshold: float) -> dict:
        """For a given cv_nll threshold, count ASINs with ≥2/≥5/≥10 quality users."""
        # Step 1: get all uids that pass (in cal OR val)
        passing_uids = {u for u, d in user_data.items() if d["cv_nll"] <= threshold}
        # Step 2: count per ASIN
        per_asin_counts = []
        n_asins_ge2 = 0
        n_asins_ge5 = 0
        n_asins_ge10 = 0
        n_asins_ge1 = 0
        for a, uids in asin_to_users.items():
            n_passing = len(uids & passing_uids)
            if n_passing >= 1:
                n_asins_ge1 += 1
                per_asin_counts.append(n_passing)
            if n_passing >= 2:
                n_asins_ge2 += 1
            if n_passing >= 5:
                n_asins_ge5 += 1
            if n_passing >= 10:
                n_asins_ge10 += 1
        per_asin_arr = np.array(per_asin_counts) if per_asin_counts else np.array([0])
        return {
            "threshold": threshold,
            "n_users_passing": len(passing_uids),
            "n_asins_ge1": n_asins_ge1,
            "n_asins_ge2": n_asins_ge2,
            "n_asins_ge5": n_asins_ge5,
            "n_asins_ge10": n_asins_ge10,
            "per_asin_quality_users_P50": float(np.percentile(per_asin_arr, 50)) if len(per_asin_arr) > 1 else 0.0,
            "per_asin_quality_users_P75": float(np.percentile(per_asin_arr, 75)) if len(per_asin_arr) > 1 else 0.0,
            "per_asin_quality_users_P90": float(np.percentile(per_asin_arr, 90)) if len(per_asin_arr) > 1 else 0.0,
        }

    coverage_at = {}
    # For T_95, T_97.5, T_99, plus T=39 (current ROC) and a few reference T
    thresholds_for_coverage = []
    for p in PRECISION_TARGETS:
        if targets_found[p] is not None:
            thresholds_for_coverage.append((f"T_{p}", targets_found[p]["T"]))
    thresholds_for_coverage.append(("T_39_current", 39))

    for label, T in thresholds_for_coverage:
        cov = compute_asin_coverage(T)
        coverage_at[label] = cov
        log(f"  {label} (T={T}): users={cov['n_users_passing']:>6} "
            f"asins≥1={cov['n_asins_ge1']:>5} ≥2={cov['n_asins_ge2']:>5} "
            f"≥5={cov['n_asins_ge5']:>5} ≥10={cov['n_asins_ge10']:>5}")

    # ====== Step 5: recommendation ======
    log(f"\n=== Recommendation ===")
    # 找 validation precision ≥ RECOMMENDED_TARGET 且 asins_ge5 ≥ MIN_ASINS_GE5 的最宽松 threshold
    # 从最大 p 开始, 如果 validation precision 不达标, 降到下一个
    rec_candidates = []
    for p in PRECISION_TARGETS:
        vr = val_results.get(p)
        cov = coverage_at.get(f"T_{p}")
        if vr is None or cov is None:
            continue
        if vr["precision"] >= RECOMMENDED_TARGET and cov["n_asins_ge5"] >= MIN_ASINS_GE5:
            rec_candidates.append((p, vr["T"], vr["precision"], cov))
    # Choose largest T (most permissive)
    if rec_candidates:
        rec = max(rec_candidates, key=lambda x: x[1])  # max T
        recommended_T, recommended_p_target = rec[1], rec[0]
        rec_val_precision = rec[2]
        rec_cov = rec[3]
        log(f"  recommended T={recommended_T} (target p={recommended_p_target}): "
            f"val_precision={rec_val_precision:.4f}, asins_ge5={rec_cov['n_asins_ge5']}")
    else:
        # Fallback: largest available
        available = [(p, targets_found[p]["T"]) for p in PRECISION_TARGETS
                     if targets_found[p] is not None]
        if available:
            recommended_T = available[-1][1]
            recommended_p_target = available[-1][0]
            rec_val_precision = val_results[recommended_p_target]["precision"]
            rec_cov = coverage_at[f"T_{recommended_p_target}"]
            log(f"  fallback recommendation T={recommended_T} (target p={recommended_p_target}): "
                f"val_precision={rec_val_precision:.4f}, asins_ge5={rec_cov['n_asins_ge5']}")
        else:
            recommended_T = None
            recommended_p_target = None

    # ====== Step 6: write strict users list ======
    STRICT_USERS_OUT.parent.mkdir(parents=True, exist_ok=True)
    if recommended_T is not None:
        strict_uids = sorted([u for u, d in user_data.items() if d["cv_nll"] <= recommended_T])
        with open(STRICT_USERS_OUT, "w", encoding="utf-8") as f:
            json.dump({
                "description": (
                    f"High-confidence Gaussian quality cohort: users with cv_nll <= {recommended_T}. "
                    f"Calibration precision target p={recommended_p_target}, "
                    f"validation precision={rec_val_precision:.4f}."
                ),
                "threshold_cv_nll": recommended_T,
                "precision_target": recommended_p_target,
                "validation_precision": rec_val_precision,
                "n_users": len(strict_uids),
                "users": strict_uids,
            }, f, ensure_ascii=False, indent=2)
        log(f"  wrote → {STRICT_USERS_OUT} ({len(strict_uids)} users)")

    # ====== Write summary ======
    CALIBRATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": (
            f"High-confidence Gaussian quality calibration. "
            f"Ground truth label = reliable iff delta_nll_ci_lo > 0. "
            f"70/30 split (sha1 user_id hash, seed=calibrate_v1). "
            f"Targets: T_p = max T s.t. calibration Precision(T) >= p for p in {{95%, 97.5%, 99%}}. "
            f"Recommended threshold = largest T whose validation Precision >= 97.5% AND "
            f"ASIN coverage with >=5 quality users >= {MIN_ASINS_GE5}."
        ),
        "split_seed": SPLIT_HASH_SEED,
        "n_users_with_both": len(user_data),
        "n_calibration": len(cal_uids),
        "n_validation": len(val_uids),
        "calibration_reliable_frac": float(cal_label.mean()),
        "validation_reliable_frac": float(val_label.mean()),
        "calibration_scan": cal_results,
        "operating_points": {
            f"T_{p}": {
                "calibration": targets_found[p],
                "validation": val_results[p],
                "coverage": coverage_at.get(f"T_{p}"),
            } for p in PRECISION_TARGETS
        },
        "current_ROC": {
            "T": 39,
            "validation": {
                "precision": 0.928,
                "recall": 0.723,
                "F1": 0.812,
            },
            "coverage": coverage_at.get("T_39_current"),
        },
        "recommendation": {
            "threshold_cv_nll": recommended_T,
            "precision_target": recommended_p_target,
            "validation_precision": rec_val_precision if recommended_T is not None else None,
            "coverage": rec_cov if recommended_T is not None else None,
            "rationale": (
                f"Largest T satisfying: (a) validation precision >= {RECOMMENDED_TARGET}, "
                f"(b) ASINs with >=5 quality users >= {MIN_ASINS_GE5}."
            ),
        },
        "elapsed_sec": time.time() - t0,
    }
    with open(CALIBRATION_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {CALIBRATION_OUT}")


if __name__ == "__main__":
    main()
