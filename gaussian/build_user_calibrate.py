#!/usr/bin/env python3
"""ROC-based CV-NLL threshold calibration.

用户指令 2026-08-30: 完全 abandon 人工阈值 (0.8/0.9/60/70)。用 Gaussian Predictive Reliability
的 ΔNLL + bootstrap CI 作为 ground truth 标签, 训练一个 CV-NLL 阈值 T*。

方法:
  Step 1: 70/30 split (calibration / validation) by user_id hash
  Step 2: 在 calibration set (70%) 上:
            - 用 delta_nll_ci_lo > 0 作为 ground truth label (1=reliable, 0=unreliable)
            - 扫描 T ∈ [20, 200], J(T) = TPR(T) - FPR(T)
            - 找 T* = argmax J(T)
  Step 3: 在 validation set (30%) 上验证 T*:
            - precision = P(ci_lo > 0 | cv_nll <= T*)
            - recall = P(cv_nll <= T* | ci_lo > 0)
            - F1 = 2*P*R/(P+R)
  Step 4: 输出 ROC curve points + T* recommendation

**输入**:
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_user_gaussians_reliability.json
    (字段: cv_nll, cv_nll_std, delta_nll_mean, delta_nll_ci_lo, reliable)

**输出**:
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/cv_nll_calibration.json
    (T*, ROC curve, validation metrics, recommended Stage 4 cohort)

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

RELIABILITY_OUT = SCRATCH / "stage8_5_user_gaussians_reliability.json"
CV_OUT = SCRATCH / "stage8_5_user_gaussians_cv.json"
CALIBRATION_OUT = REPO_ROOT / "result" / "gaussian" / "cv_nll_calibration.json"

# --- Calibration config (硬编码) ---
SPLIT_HASH_SEED = "calibrate_v1"   # hash seed for 70/30 split (deterministic)
CALIBRATION_FRAC = 0.7             # 70% for T* search, 30% for validation
T_SCAN_MIN = 20.0
T_SCAN_MAX = 200.0
T_SCAN_STEPS = 181                 # scan T = 20, 21, ..., 200 (1.0 step)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [calibrate] {msg}", flush=True)


def user_split_bucket(uid: str) -> int:
    """Deterministic bucket assignment: 0 = calibration, 1 = validation."""
    h = hashlib.sha1((SPLIT_HASH_SEED + "|" + uid).encode("utf-8")).hexdigest()
    return int(h, 16) % 100 < int(CALIBRATION_FRAC * 100)


def main():
    log(f"=== ROC-based CV-NLL calibration ===")
    log(f"  SPLIT_HASH_SEED={SPLIT_HASH_SEED}, CALIBRATION_FRAC={CALIBRATION_FRAC}")
    log(f"  T_SCAN=[{T_SCAN_MIN}, {T_SCAN_MAX}], steps={T_SCAN_STEPS}")

    if not RELIABILITY_OUT.exists():
        raise FileNotFoundError(f"reliability cache required: {RELIABILITY_OUT}")
    if not CV_OUT.exists():
        raise FileNotFoundError(f"CV cache required (for cv_nll): {CV_OUT}")

    log(f"  loading reliability cache: {RELIABILITY_OUT}")
    t0 = time.time()
    with open(RELIABILITY_OUT, "r", encoding="utf-8") as f:
        rdoc = json.load(f)
    rel_users = rdoc["users"]
    rel_meta = rdoc.get("reliability_meta", {})
    log(f"  loaded {len(rel_users)} users from reliability cache in {time.time() - t0:.1f}s")

    log(f"  loading CV cache: {CV_OUT}")
    t0 = time.time()
    with open(CV_OUT, "r", encoding="utf-8") as f:
        cdoc = json.load(f)
    cv_users = cdoc["users"]
    log(f"  loaded {len(cv_users)} users from CV cache in {time.time() - t0:.1f}s")

    # Merge: keep users present in BOTH caches (intersection by uid)
    pairs = []  # (uid, cv_nll, cv_nll_std, delta_nll_mean, delta_nll_ci_lo, reliable)
    common = set(rel_users.keys()) & set(cv_users.keys())
    log(f"  intersection (both caches): {len(common)} users")
    for uid in common:
        rg = rel_users[uid]
        cg = cv_users[uid]
        cv_nll = cg.get("cv_nll")
        ci_lo = rg.get("delta_nll_ci_lo")
        if cv_nll is None or ci_lo is None:
            continue
        pairs.append((
            uid,
            float(cv_nll),
            float(cg.get("cv_nll_std") or 0.0),
            float(rg.get("delta_nll_mean") or 0.0),
            float(ci_lo),
            int(rg.get("reliable") or 0),
        ))
    log(f"  users with both cv_nll + delta_nll_ci_lo: {len(pairs)}")

    # 70/30 split
    cal_pairs = [p for p in pairs if user_split_bucket(p[0]) == 0]
    val_pairs = [p for p in pairs if user_split_bucket(p[0]) == 1]
    log(f"  split: calibration={len(cal_pairs)}, validation={len(val_pairs)}")

    # Ground truth: delta_nll_ci_lo > 0 → reliable (label=1)
    cal_cv_nll = np.array([p[1] for p in cal_pairs])
    cal_label = np.array([1 if p[4] > 0 else 0 for p in cal_pairs])
    val_cv_nll = np.array([p[1] for p in val_pairs])
    val_label = np.array([1 if p[4] > 0 else 0 for p in val_pairs])
    log(f"  calibration: {cal_label.sum()} reliable, {(1 - cal_label).sum()} unreliable "
        f"({cal_label.mean():.1%} reliable)")
    log(f"  validation:  {val_label.sum()} reliable, {(1 - val_label).sum()} unreliable "
        f"({val_label.mean():.1%} reliable)")

    # ====== Step 1: ROC scan on calibration set ======
    log(f"\n=== ROC scan on calibration set ===")
    T_grid = np.linspace(T_SCAN_MIN, T_SCAN_MAX, T_SCAN_STEPS)
    roc_points = []
    best_J = -np.inf
    best_T = None
    for T in T_grid:
        # Predict reliable iff cv_nll <= T (low NLL → good fit → likely reliable)
        pred = (cal_cv_nll <= T).astype(np.int32)
        tp = int(((pred == 1) & (cal_label == 1)).sum())
        fp = int(((pred == 1) & (cal_label == 0)).sum())
        fn = int(((pred == 0) & (cal_label == 1)).sum())
        tn = int(((pred == 0) & (cal_label == 0)).sum())
        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        J = tpr - fpr
        roc_points.append({
            "T": float(T),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "TPR": float(tpr), "FPR": float(fpr),
            "J": float(J),
            "precision": tp / max(1, tp + fp),
            "recall": tpr,
        })
        if J > best_J:
            best_J = J
            best_T = float(T)
    log(f"  best T* = {best_T} (J={best_J:.3f})")

    # Compute AUC via trapezoidal
    sorted_roc = sorted(roc_points, key=lambda r: r["FPR"])
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    for p in sorted_roc:
        auc += (p["FPR"] - prev_fpr) * (p["TPR"] + prev_tpr) / 2.0
        prev_fpr = p["FPR"]
        prev_tpr = p["TPR"]
    log(f"  AUC (cv_nll → reliable) = {auc:.3f}")

    # ====== Step 2: validation on 30% set ======
    log(f"\n=== validation on 30% held-out set at T*={best_T} ===")
    val_pred = (val_cv_nll <= best_T).astype(np.int32)
    val_tp = int(((val_pred == 1) & (val_label == 1)).sum())
    val_fp = int(((val_pred == 1) & (val_label == 0)).sum())
    val_fn = int(((val_pred == 0) & (val_label == 1)).sum())
    val_tn = int(((val_pred == 0) & (val_label == 0)).sum())
    val_precision = val_tp / max(1, val_tp + val_fp)
    val_recall = val_tp / max(1, val_tp + val_fn)
    val_f1 = 2 * val_precision * val_recall / max(1e-9, val_precision + val_recall)
    val_accuracy = (val_tp + val_tn) / max(1, len(val_label))
    log(f"  TP={val_tp}, FP={val_fp}, FN={val_fn}, TN={val_tn}")
    log(f"  precision={val_precision:.3f}, recall={val_recall:.3f}, F1={val_f1:.3f}, accuracy={val_accuracy:.3f}")

    # ====== Step 3: also report baseline human-set thresholds for comparison ======
    log(f"\n=== baseline comparison: human-set cv_nll thresholds ===")
    baseline_results = {}
    for T_human in [40, 50, 60, 70, 80, 100, 139.8]:  # 139.8 = q95 from CV
        pred_h = (val_cv_nll <= T_human).astype(np.int32)
        tp_h = int(((pred_h == 1) & (val_label == 1)).sum())
        fp_h = int(((pred_h == 1) & (val_label == 0)).sum())
        fn_h = int(((pred_h == 0) & (val_label == 1)).sum())
        tn_h = int(((pred_h == 0) & (val_label == 0)).sum())
        prec_h = tp_h / max(1, tp_h + fp_h)
        rec_h = tp_h / max(1, tp_h + fn_h)
        f1_h = 2 * prec_h * rec_h / max(1e-9, prec_h + rec_h)
        baseline_results[str(T_human)] = {
            "T": T_human, "TP": tp_h, "FP": fp_h, "FN": fn_h, "TN": tn_h,
            "precision": prec_h, "recall": rec_h, "F1": f1_h,
            "n_pass": int(pred_h.sum()),
        }
        log(f"  T={T_human}: P={prec_h:.3f} R={rec_h:.3f} F1={f1_h:.3f} n_pass={int(pred_h.sum())}")

    # ====== Step 4: cohort size at T* ======
    cal_pass = int((cal_cv_nll <= best_T).sum())
    val_pass = int((val_cv_nll <= best_T).sum())
    total_pass = cal_pass + val_pass
    log(f"\n=== cohort at T*={best_T}: calibration={cal_pass}, validation={val_pass}, total={total_pass} ===")

    # ====== write output ======
    CALIBRATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "description": (
            f"ROC-based CV-NLL threshold calibration. Ground truth label = reliable iff "
            f"delta_nll_ci_lo > 0 (from Gaussian Predictive Reliability). 70/30 split by "
            f"sha1(user_id + seed). T* = argmax (TPR - FPR) on calibration set. "
            f"Validation on held-out 30%."
        ),
        "split_seed": SPLIT_HASH_SEED,
        "calibration_frac": CALIBRATION_FRAC,
        "n_total_with_both": len(pairs),
        "n_calibration": len(cal_pairs),
        "n_validation": len(val_pairs),
        "n_reliable_calibration": int(cal_label.sum()),
        "n_reliable_validation": int(val_label.sum()),
        "T_star": best_T,
        "best_J_calibration": float(best_J),
        "auc_calibration": float(auc),
        "validation_at_T_star": {
            "TP": val_tp, "FP": val_fp, "FN": val_fn, "TN": val_tn,
            "precision": val_precision,
            "recall": val_recall,
            "F1": val_f1,
            "accuracy": val_accuracy,
            "n_pass": val_pass,
        },
        "baseline_human_thresholds": baseline_results,
        "roc_points": roc_points,
        "reliability_meta": rel_meta,
        "elapsed_sec": time.time() - t0,
    }
    with open(CALIBRATION_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {CALIBRATION_OUT}")


if __name__ == "__main__":
    main()
