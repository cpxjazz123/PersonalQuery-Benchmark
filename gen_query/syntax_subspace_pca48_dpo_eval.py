#!/usr/bin/env python3
"""Phase 6.B.2.3 — DPO Eval: PC sweep + target/anti + absolute distance.

用户指令 2026-08-30: DPO 成功后必须重跑 3 项 controllability tests:
1. PC directional sweep: ρ median > 0.2 (Phase 6.A baseline -0.255)
2. Target vs Anti-target: Δd < 0 with bootstrap CI not crossing 0
3. Absolute distance: median d should decrease from 5+ → 4

加载 dpo_final.pt (Phase 6.B.2.2 训练结果), 跑 same tests as Phase 6.A eval
+ extra absolute distance check (single-sample median d).

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
DATA_DIR = SCRATCH / "pca48_dataset"
CKPT_DIR = SCRATCH / "pca48_dpo_ckpt"
LOG_DIR = SCRATCH / "logs"
OUT_DIR = REPO_ROOT / "result" / "gaussian"
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Eval config (locked) ===
N_PC_SWEEP_CONTENTS = 5
N_TOP_PCS = 5
PC_LEVELS = [-2, -1, 0, 1, 2]
N_TAR_ANTI_USERS = 50
N_ABS_DIST_PAIRS = 100  # new: absolute distance check on N pairs
GEN_BATCH = 4  # batch_contents
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
TOP_P = 0.95
SEED = 42

PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [dpo_eval] {msg}", flush=True)


def compute_pca48_of_text(
    text, pca_components, pca_mean, fnames_all, fnames_f3, col_idx_f3,
    scaler_mean, scaler_scale,
):
    import spacy
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2
    nlp = spacy.load("en_core_web_sm")
    if "textcat" in nlp.pipe_names:
        nlp.remove_pipe("textcat")
    doc = nlp(text)
    try:
        d = per_sentence_features_v2(doc)
        if d is None:
            return None
        feats = np.zeros(len(fnames_all), dtype=np.float64)
        for j, fn in enumerate(fnames_all):
            feats[j] = float(d.get(fn, 0.0))
    except Exception:
        return None
    f3 = feats[col_idx_f3]
    f3_scaled = (f3 - scaler_mean) / np.maximum(scaler_scale, 1e-12)
    z = (f3_scaled - pca_mean) @ pca_components.T
    return z.astype(np.float32)


def main():
    log("=== Phase 6.B.2.3 — DPO Eval (3 tests) ===")

    # 1. Load dataset (use same held-out as build_pref)
    log("loading dataset ...")
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    log(f"  {len(samples)} samples")

    # 2. Load DPO ckpt
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix

    log("loading Qwen2-7B + LoRA + projector (DPO ckpt) ...")
    model = Qwen2WithPrefix()
    log(f"  loading ckpt from {CKPT_DIR / 'dpo_final.pt'}")
    model.load_projector(str(CKPT_DIR / "dpo_final.pt"))
    model.eval()

    # 3. Load user Gaussians
    log("loading user Gaussians ...")
    gauss = json.load(open(GAUSS_PATH))
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])
    users_dict = gauss["users"]
    all_mu = np.asarray([u["mu"] for u in users_dict.values()], dtype=np.float64)
    mu_global_pca48 = all_mu.mean(axis=0)

    # ============================================================
    # TEST 1: PC directional sweep
    # ============================================================
    log("\n=== TEST 1: PC directional sweep ===")
    explained_var = np.asarray(gauss["pca_explained_variance_ratio"])
    top_pc_idx = np.argsort(-explained_var)[:N_TOP_PCS].tolist()

    held_out = samples[-N_PC_SWEEP_CONTENTS:]
    held_out_attrs = [s["c_i"] for s in held_out]

    pc_sweep_results = []
    t0 = time.time()
    for pc_idx in top_pc_idx:
        for content_idx, attrs in enumerate(held_out_attrs):
            z_base = mu_global_pca48.astype(np.float32).copy()
            pc_std = all_mu[:, pc_idx].std()
            if pc_std < 1e-3:
                pc_std = 1.0

            zs = []
            for k in PC_LEVELS:
                z = z_base.copy()
                z[pc_idx] += float(k) * float(pc_std)
                zs.append(z)
            zs_arr = np.stack(zs)

            torch.manual_seed(SEED + pc_idx * 100 + content_idx)
            zs_torch = torch.from_numpy(zs_arr).to("cuda")
            attrs_batch = [attrs] * len(zs)

            gens = model.generate(
                zs_torch, attrs_batch,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )
            pc_sweep_results.append({
                "pc_idx": int(pc_idx),
                "content_idx": content_idx,
                "generations": gens,
            })
    log(f"  PC sweep generation done in {time.time() - t0:.0f}s "
        f"({len(pc_sweep_results)} batches)")

    # Compute rho
    pc_summary = []
    for pc_idx in top_pc_idx:
        all_pc_target = []
        all_pc_gen = []
        rows = [r for r in pc_sweep_results if r["pc_idx"] == pc_idx]
        for ci, row in enumerate(rows):
            for k_target, z_used, gen_text in zip(PC_LEVELS,
                np.stack([mu_global_pca48.astype(np.float32).copy() +
                         (k * all_mu[:, pc_idx].std() if all_mu[:, pc_idx].std() > 1e-3 else 0) *
                         np.eye(PCA_DIM)[pc_idx].astype(np.float32)
                         for k in PC_LEVELS]),
                row["generations"]):
                z_gen = compute_pca48_of_text(gen_text, pca_components, pca_mean,
                                              fnames_all, fnames_f3, col_idx_f3,
                                              scaler_mean, scaler_scale)
                if z_gen is None:
                    continue
                all_pc_target.append(float(z_used[pc_idx]))
                all_pc_gen.append(float(z_gen[pc_idx]))

        if len(all_pc_target) < 5:
            continue
        from scipy.stats import spearmanr
        rho, pval = spearmanr(all_pc_target, all_pc_gen)
        pc_summary.append({
            "pc_idx": int(pc_idx),
            "explained_var": float(explained_var[pc_idx]),
            "n_samples": len(all_pc_target),
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
        })
        log(f"  PC {pc_idx:2d} (var={explained_var[pc_idx]:.3f}): "
            f"ρ={rho:+.3f}  p={pval:.3g}")

    # ============================================================
    # TEST 2: Target vs Anti-target
    # ============================================================
    log("\n=== TEST 2: Target/Anti-target ===")
    cohort_user_ids = [s["user_id"] for s in samples[:N_TAR_ANTI_USERS * 2]]
    cohort_user_ids = list(dict.fromkeys(cohort_user_ids))[:N_TAR_ANTI_USERS]
    valid_pairs = []
    for u in cohort_user_ids:
        if u not in users_dict:
            continue
        user_sents = [s for s in samples if s["user_id"] == u]
        if not user_sents:
            continue
        valid_pairs.append({"user_id": u, "attrs": user_sents[0]["c_i"]})
    log(f"  valid pairs: {len(valid_pairs)}")

    target_anti_results = []
    t1 = time.time()
    for i, pair in enumerate(valid_pairs):
        u = pair["user_id"]
        attrs = pair["attrs"]
        mu_u = np.asarray(users_dict[u]["mu"], dtype=np.float32)
        z_target = mu_u.copy()
        z_anti = (2 * mu_global_pca48 - mu_u).astype(np.float32)

        torch.manual_seed(SEED + i)
        zs_torch = torch.stack([torch.from_numpy(z_target),
                                 torch.from_numpy(z_anti)]).to("cuda")

        gens = model.generate(
            zs_torch, [attrs, attrs],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )
        gen_target, gen_anti = gens
        z_gen_target = compute_pca48_of_text(gen_target, pca_components, pca_mean,
                                             fnames_all, fnames_f3, col_idx_f3,
                                             scaler_mean, scaler_scale)
        z_gen_anti = compute_pca48_of_text(gen_anti, pca_components, pca_mean,
                                           fnames_all, fnames_f3, col_idx_f3,
                                           scaler_mean, scaler_scale)
        if z_gen_target is None or z_gen_anti is None:
            continue
        d_target_to_target = float(np.linalg.norm(z_gen_target - z_target))
        d_anti_to_target = float(np.linalg.norm(z_gen_anti - z_target))
        d_target_to_anti = float(np.linalg.norm(z_gen_target - z_anti))
        d_anti_to_anti = float(np.linalg.norm(z_gen_anti - z_anti))

        target_anti_results.append({
            "user_id": u,
            "attrs": attrs,
            "gen_target": gen_target,
            "gen_anti": gen_anti,
            "d_target_to_target": d_target_to_target,
            "d_anti_to_target": d_anti_to_target,
            "d_target_to_anti": d_target_to_anti,
            "d_anti_to_anti": d_anti_to_anti,
        })
        if i < 3 or i % 10 == 0:
            log(f"  [{i+1}/{len(valid_pairs)}] user={u[:8]}... "
                f"d(y_t,z_t)={d_target_to_target:.3f}  d(y_a,z_t)={d_anti_to_target:.3f}  "
                f"Δ={d_anti_to_target - d_target_to_target:+.3f}")
    log(f"  target/anti done in {time.time() - t1:.0f}s")

    # ============================================================
    # TEST 3: Absolute distance (single-sample, NEW)
    # ============================================================
    log("\n=== TEST 3: Absolute distance (single-sample median d) ===")
    # Use samples[1500:1500+N_ABS_DIST_PAIRS] to avoid overlap with training
    abs_pairs = samples[1500:1500 + N_ABS_DIST_PAIRS]
    log(f"  using {len(abs_pairs)} pairs for absolute distance")

    abs_results = []
    t2 = time.time()
    for bi in range(0, len(abs_pairs), GEN_BATCH):
        batch_samples = abs_pairs[bi:bi + GEN_BATCH]
        bc = len(batch_samples)
        zs_list = []
        attrs_list = []
        for sample in batch_samples:
            z = np.asarray(sample["z_48"], dtype=np.float32)
            zs_list.append(torch.from_numpy(z))
            attrs_list.append(sample["c_i"])
        zs_batched = torch.stack(zs_list, dim=0).to("cuda")

        torch.manual_seed(SEED + bi + 10000)
        gens = model.generate(
            zs_batched, attrs_list,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=False,  # greedy for reproducibility
        )
        for j, sample in enumerate(batch_samples):
            z_target = np.asarray(sample["z_48"], dtype=np.float32)
            z_gen = compute_pca48_of_text(gens[j], pca_components, pca_mean,
                                          fnames_all, fnames_f3, col_idx_f3,
                                          scaler_mean, scaler_scale)
            if z_gen is None:
                continue
            d = float(np.linalg.norm(z_gen - z_target))
            abs_results.append({
                "pair_idx": 1500 + bi + j,
                "d": d,
                "gen": gens[j],
            })

    log(f"  absolute distance done in {time.time() - t2:.0f}s "
        f"({len(abs_results)} valid pairs)")

    if abs_results:
        ds = [r["d"] for r in abs_results]
        log(f"  median d (single-sample): {np.median(ds):.3f}, mean: {np.mean(ds):.3f}")
        log(f"  d < 1: {sum(1 for d in ds if d < 1)} ({100*sum(1 for d in ds if d < 1)/len(ds):.1f}%)")
        log(f"  d < 2: {sum(1 for d in ds if d < 2)} ({100*sum(1 for d in ds if d < 2)/len(ds):.1f}%)")
        log(f"  d < 3: {sum(1 for d in ds if d < 3)} ({100*sum(1 for d in ds if d < 3)/len(ds):.1f}%)")

    # ============================================================
    # DECISION
    # ============================================================
    log("\n=== DECISION ===")

    # Test 1: PC sweep
    valid_pc = [p for p in pc_summary if p["spearman_rho"] is not None]
    median_rho = float(np.median([p["spearman_rho"] for p in valid_pc])) if valid_pc else 0.0
    max_rho = float(np.max([p["spearman_rho"] for p in valid_pc])) if valid_pc else 0.0
    log(f"  Test 1 (PC sweep): median ρ={median_rho:+.3f}, max ρ={max_rho:+.3f}")

    # Test 2: Target/anti
    if target_anti_results:
        deltas = sorted([r["d_anti_to_target"] - r["d_target_to_target"]
                        for r in target_anti_results])
        med_delta = deltas[len(deltas) // 2]
        mean_delta = float(np.mean(deltas))
        frac_negative = sum(1 for x in deltas if x < 0) / len(deltas)
        rng = np.random.default_rng(SEED)
        boot_means = []
        for _ in range(2000):
            idx = rng.integers(0, len(deltas), size=len(deltas))
            boot_means.append(float(np.mean([deltas[i] for i in idx])))
        boot_means.sort()
        ci_low = boot_means[int(0.025 * len(boot_means))]
        ci_high = boot_means[int(0.975 * len(boot_means))]
        log(f"  Test 2 (target/anti): median Δd={med_delta:+.3f}, "
            f"CI [{ci_low:+.3f}, {ci_high:+.3f}], frac_neg={frac_negative*100:.1f}%")
    else:
        med_delta = 0.0
        ci_low, ci_high = 0.0, 0.0
        frac_negative = 0.0

    # Test 3: Absolute distance
    if abs_results:
        ds = [r["d"] for r in abs_results]
        med_abs = float(np.median(ds))
        mean_abs = float(np.mean(ds))
        log(f"  Test 3 (absolute d): median={med_abs:.3f}, mean={mean_abs:.3f} "
            f"(Phase 6.A d_min median=5.03)")
    else:
        med_abs = 99.0

    # Decision criteria
    go_test1 = median_rho > 0.2  # Phase 6.A baseline -0.255
    go_test2 = (med_delta < 0) and (ci_high < 0)
    go_test3 = med_abs < 5.0  # should decrease from Phase 6.A's ~7 (single-sample) or 5.03 (best of 8)

    if go_test1 and go_test2 and go_test3:
        decision = "STRONG_GO"
        rationale = (f"ρ={median_rho:+.3f}>0.2 ✓; "
                     f"target/anti Δd={med_delta:+.3f}<0 CI[{ci_low:+.3f},{ci_high:+.3f}] 不跨 0 ✓; "
                     f"abs d={med_abs:.3f}<5.0 ✓.")
    elif go_test1 and go_test2:
        decision = "PARTIAL_GO"
        rationale = (f"ρ={median_rho:+.3f}>0.2 ✓; "
                     f"target/anti Δd={med_delta:+.3f}<0 ✓; "
                     f"但 abs d={med_abs:.3f} 仍 ≥ 5.0 (directional control 起效但 reachable region 不足).")
    elif go_test1:
        decision = "DIRECTIONAL_ONLY"
        rationale = (f"ρ={median_rho:+.3f}>0.2 但 target/anti 不显著. "
                     f"Direction 学会但 user style 不能 pinpoint.")
    else:
        decision = "NO_GO"
        rationale = (f"ρ={median_rho:+.3f} 不显著, target/anti Δd={med_delta:+.3f} CI 跨 0. "
                     f"DPO 没建立 z_t → direction 映射. 下一步: iterative DPO (Phase 6.B.3) 或换 mechanism.")

    log(f"\n  DECISION: {decision}")
    log(f"  {rationale}")

    # Save
    out_doc = {
        "description": (
            "Phase 6.B.2.3 — DPO Eval (3 tests). "
            "Loads DPO ckpt from pca48_dpo_ckpt/dpo_final.pt. "
            "Test 1: PC directional sweep (5 PCs × 5 contents × 5 levels). "
            "Test 2: target vs anti-target (50 users). "
            f"Test 3: absolute distance on {N_ABS_DIST_PAIRS} pairs (single-sample)."
        ),
        "config": {
            "ckpt": "dpo_final.pt",
            "N_PC_SWEEP_CONTENTS": N_PC_SWEEP_CONTENTS,
            "N_TOP_PCS": N_TOP_PCS,
            "PC_LEVELS": PC_LEVELS,
            "N_TAR_ANTI_USERS": N_TAR_ANTI_USERS,
            "N_ABS_DIST_PAIRS": N_ABS_DIST_PAIRS,
            "TEMPERATURE": TEMPERATURE,
            "TOP_P": TOP_P,
        },
        "test1_pc_sweep": {
            "summary": pc_summary,
            "median_rho": median_rho,
            "max_rho": max_rho,
        },
        "test2_target_anti": {
            "n_pairs": len(target_anti_results),
            "median_delta_d": float(med_delta),
            "mean_delta_d": float(mean_delta),
            "frac_negative": float(frac_negative),
            "bootstrap_95_ci": [ci_low, ci_high],
            "rows": [
                {k: v for k, v in r.items() if k not in ("z_target", "z_anti")}
                for r in target_anti_results[:30]
            ],
        },
        "test3_absolute_distance": {
            "n_valid": len(abs_results),
            "median_d": med_abs,
            "mean_d": mean_abs,
            "phase_6a_d_min_median": 5.03,  # for reference
            "frac_d_lt_1": float(sum(1 for r in abs_results if r["d"]<1) / len(abs_results)),
            "frac_d_lt_2": float(sum(1 for r in abs_results if r["d"]<2) / len(abs_results)),
            "frac_d_lt_3": float(sum(1 for r in abs_results if r["d"]<3) / len(abs_results)),
        },
        "decision": decision,
        "rationale": rationale,
    }
    out_path = OUT_DIR / "pca48_dpo_audit.json"
    with open(out_path, "w") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()
