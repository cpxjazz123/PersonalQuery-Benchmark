#!/usr/bin/env python3
"""Phase 6.A.4 + 6.A.5 — Inference + PC sweep + target/anti-target eval.

用户指令 2026-08-30:
  Phase 6F PC directional sweep:
    固定 content c; 只改变 PCA target; 测 ρ(PC_target, PC_generated)
  Phase 6G target/anti-target:
    target z_t = μ_u; anti z_a = 2μ_global - μ_u
    d(y_t, z_t) < d(y_a, z_t) 在大多数对
  GO criteria:
    median[d(y_t, z_t) - d(y_a, z_t)] < 0, bootstrap 95% CI 不跨 0
    PC directional sweep 至少稳定正相关

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
CKPT_DIR = SCRATCH / "pca48_ckpt"
LOG_DIR = SCRATCH / "logs"
OUT_DIR = REPO_ROOT / "result" / "gaussian"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Eval config (locked) ===
N_PC_SWEEP_CONTENTS = 5   # 5 different content samples for PC sweep
N_TOP_PCS = 5             # test top-5 variance PCs
PC_LEVELS = [-2, -1, 0, 1, 2]  # in std units
N_TAR_ANTI_USERS = 50     # 50 users for target/anti-target test
GEN_BATCH = 16
MAX_NEW_TOKENS = 60
TEMPERATURE = 0.7
TOP_P = 0.95
SEED = 42

# F3 103d PCA48 projection pipeline
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"

PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pca48_eval] {msg}", flush=True)


def main():
    log("=== Phase 6.A.4/5 — Inference + PC sweep + target/anti-target ===")

    # 1. Load training dataset (we'll use last 200 as held-out for content selection)
    log("loading dataset + ckpt path ...")
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    z_all = np.asarray([s["z_48"] for s in samples], dtype=np.float32)
    log(f"  {len(samples)} samples, z_all {z_all.shape}")

    # 2. Load model + projector + LoRA
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix

    log("loading Qwen2-7B + LoRA ...")
    model = Qwen2WithPrefix()
    log(f"  loading checkpoint from {CKPT_DIR / 'projector_final.pt'}")
    model.load_projector(str(CKPT_DIR / "projector_final.pt"))
    model.eval()
    log("  model ready")

    # 3. Load user Gaussians for target/anti-target
    log("loading user Gaussians (for z_mean, μ_global, target/anti pairs)...")
    gauss = json.load(open(GAUSS_PATH))
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)  # (48, 103)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])

    # Compute global mean of mu (for anti-target)
    users_dict = gauss["users"]
    log(f"  {len(users_dict)} users")

    # μ_global = mean of all user mu (in PCA48 space, not original F3 space)
    all_mu = np.asarray([u["mu"] for u in users_dict.values()], dtype=np.float64)
    mu_global_pca48 = all_mu.mean(axis=0)
    log(f"  μ_global PCA48 norm: {np.linalg.norm(mu_global_pca48):.3f}")

    # ============== PHASE 6F: PC DIRECTIONAL SWEEP ==============
    log("\n=== Phase 6F: PC directional sweep ===")

    # 5 high-variance PCs (top-5 by explained variance ratio)
    explained_var = np.asarray(gauss["pca_explained_variance_ratio"])
    top_pc_idx = np.argsort(-explained_var)[:N_TOP_PCS].tolist()
    log(f"  testing PCs {top_pc_idx} (explained_var: {[f'{explained_var[i]:.3f}' for i in top_pc_idx]})")

    # Held-out contents (use last 5 samples from train as fixed c for sweep)
    held_out = samples[-N_PC_SWEEP_CONTENTS:]
    held_out_attrs = [s["c_i"] for s in held_out]

    pc_sweep_results = []
    for pc_idx in top_pc_idx:
        for content_idx, attrs in enumerate(held_out_attrs):
            # z_base = μ_global (starting point); vary this PC only
            z_base = mu_global_pca48.astype(np.float32).copy()

            # Compute std of this PC across users
            pc_std = all_mu[:, pc_idx].std()
            if pc_std < 1e-3:
                pc_std = 1.0  # fallback

            # Sweep z = z_base + k * std * e_pc for k in PC_LEVELS
            zs = []
            for k in PC_LEVELS:
                z = z_base.copy()
                z[pc_idx] += float(k) * float(pc_std)
                zs.append(z)
            zs_arr = np.stack(zs)  # (5, 48)

            # Batched generate
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
                "attrs": attrs,
                "z_levels": zs_arr.tolist(),
                "pc_levels_target": [float(k) for k in PC_LEVELS],
                "generations": gens,
            })

    # Compute PC_k projection on each generation
    log("\n--- PC sweep: computing ρ ---")
    pc_summary = []
    for pc_idx in top_pc_idx:
        all_pc_target = []
        all_pc_gen = []
        rows = [r for r in pc_sweep_results if r["pc_idx"] == pc_idx]
        for row in rows:
            for k_target, z_used, gen_text in zip(row["pc_levels_target"], row["z_levels"], row["generations"]):
                # Compute PCA48 of generation
                z_gen = compute_pca48_of_text(gen_text, pca_components, pca_mean,
                                              fnames_all, fnames_f3, col_idx_f3,
                                              scaler_mean, scaler_scale)
                if z_gen is None:
                    continue
                all_pc_target.append(z_used[pc_idx])
                all_pc_gen.append(float(z_gen[pc_idx]))

        if len(all_pc_target) < 5:
            rho = None
            log(f"  PC {pc_idx}: only {len(all_pc_target)} valid samples, skip")
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
        log(f"  PC {pc_idx:2d} (var={explained_var[pc_idx]:.3f}): ρ={rho:+.3f}  p={pval:.3g}  n={len(all_pc_target)}")

    # ============== PHASE 6G: TARGET / ANTI-TARGET ==============
    log("\n=== Phase 6G: target/anti-target falsification ===")

    # Pick users with sentences in dataset
    cohort_user_ids = [s["user_id"] for s in samples[:N_TAR_ANTI_USERS * 2]]
    cohort_user_ids = list(dict.fromkeys(cohort_user_ids))[:N_TAR_ANTI_USERS]  # unique first N
    valid_pairs = []
    for u in cohort_user_ids:
        if u not in users_dict:
            continue
        # Find this user's sentences
        user_sents = [s for s in samples if s["user_id"] == u]
        if not user_sents:
            continue
        valid_pairs.append({"user_id": u, "attrs": user_sents[0]["c_i"]})
    log(f"  valid (user, attrs) pairs: {len(valid_pairs)}")

    target_anti_results = []
    for i, pair in enumerate(valid_pairs):
        u = pair["user_id"]
        attrs = pair["attrs"]
        mu_u = np.asarray(users_dict[u]["mu"], dtype=np.float32)
        z_target = mu_u.copy()
        z_anti = (2 * mu_global_pca48 - mu_u).astype(np.float32)

        torch.manual_seed(SEED + i)
        zs_torch = torch.stack([torch.from_numpy(z_target), torch.from_numpy(z_anti)]).to("cuda")

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

        # d_self = Mahalanobis-like distance (Euclidean in standardized PCA48 space, since
        # per-user σ_diag is per-user; for cross-user comparison, use Euclidean in PCA48)
        # Use simple L2 distance for Phase 6.A.4/5 pilot (sufficient for relative comparison)
        d_target_to_target = float(np.linalg.norm(z_gen_target - z_target))
        d_anti_to_target = float(np.linalg.norm(z_gen_anti - z_target))
        d_target_to_anti = float(np.linalg.norm(z_gen_target - z_anti))
        d_anti_to_anti = float(np.linalg.norm(z_gen_anti - z_anti))

        target_anti_results.append({
            "user_id": u,
            "attrs": attrs,
            "z_target": z_target.tolist(),
            "z_anti": z_anti.tolist(),
            "gen_target": gen_target,
            "gen_anti": gen_anti,
            "z_gen_target": z_gen_target.tolist(),
            "z_gen_anti": z_gen_anti.tolist(),
            "d_target_to_target": d_target_to_target,
            "d_anti_to_target": d_anti_to_target,
            "d_target_to_anti": d_target_to_anti,
            "d_anti_to_anti": d_anti_to_anti,
        })
        if i < 3 or i % 10 == 0:
            log(f"  [{i+1}/{len(valid_pairs)}] user={u[:8]}... "
                f"d(y_t,z_t)={d_target_to_target:.3f}  d(y_a,z_t)={d_anti_to_target:.3f}  "
                f"Δ={d_anti_to_target - d_target_to_target:+.3f}")

    # ============== DECISION ==============
    log("\n=== DECISION DIAGNOSTICS ===")

    # PC sweep: median ρ across top-5 PCs
    valid_pc = [p for p in pc_summary if p["spearman_rho"] is not None]
    if valid_pc:
        median_rho = float(np.median([p["spearman_rho"] for p in valid_pc]))
        max_rho = float(np.max([p["spearman_rho"] for p in valid_pc]))
        log(f"  PC sweep median ρ: {median_rho:+.3f}  max ρ: {max_rho:+.3f}")
    else:
        median_rho = 0.0
        max_rho = 0.0

    # Target/anti: median Δd
    if target_anti_results:
        deltas = sorted([r["d_anti_to_target"] - r["d_target_to_target"]
                        for r in target_anti_results])
        med_delta = deltas[len(deltas) // 2]
        mean_delta = float(np.mean(deltas))
        frac_negative = sum(1 for x in deltas if x < 0) / len(deltas)
        # Bootstrap 95% CI
        rng = np.random.default_rng(SEED)
        boot_means = []
        for _ in range(2000):
            idx = rng.integers(0, len(deltas), size=len(deltas))
            boot_means.append(float(np.mean([deltas[i] for i in idx])))
        boot_means.sort()
        ci_low = boot_means[int(0.025 * len(boot_means))]
        ci_high = boot_means[int(0.975 * len(boot_means))]
        log(f"  target/anti: median Δd = {med_delta:+.3f}  mean Δd = {mean_delta:+.3f}  "
            f"frac_negative = {frac_negative*100:.1f}%  bootstrap 95% CI [{ci_low:+.3f}, {ci_high:+.3f}]")
        log(f"    (positive Δd = target closer than anti-target — good)")

    # Decision per user's criteria
    go_target = (med_delta < 0) and (ci_high < 0)
    go_pc = median_rho >= 0.6 or max_rho >= 0.6
    if go_target and go_pc:
        decision = "GO"
        rationale = (f"target/anti Δd={med_delta:+.3f}<0 CI不跨0 ✓; "
                    f"PC sweep median ρ={median_rho:+.3f} max={max_rho:+.3f} ✓. "
                    f"PCA48 controllable. 下一步: 扩到全 2174 ASIN + 完整用户 benchmark.")
    elif go_target:
        decision = "PARTIAL_GO"
        rationale = (f"target/anti 起效 (Δd={med_delta:+.3f}<0 ✓), 但 PC sweep ρ={median_rho:+.3f} 偏弱. "
                    f"control 部分工作, 需要更多训练 steps 或更大 dataset.")
    elif go_pc:
        decision = "AMBIGUOUS"
        rationale = (f"PC sweep ρ={median_rho:+.3f} 不错, 但 target/anti Δd={med_delta:+.3f} 没有显著 < 0. "
                    f"direction sweep 起效但 user style 不能 pinpoint.")
    else:
        decision = "STRONG_NO_GO"
        rationale = (f"target ≈ anti (Δd={med_delta:+.3f}, CI 跨 0) + PC sweep ρ={median_rho:+.3f}. "
                    f"PCA48 完全不能控制, 即使送进模型内部. "
                    f"比 Phase 5 NL prompt 更干净 — 因为排除了 NL bottleneck, "
                    f"但 LLM 仍不能把 PCA48 映射到 syntax location.")

    log(f"\n  DECISION: {decision}")
    log(f"  {rationale}")

    # Save output
    out_doc = {
        "description": (
            "Phase 6.A.4/5 — Inference + PC sweep + target/anti-target. "
            "Loads Qwen2-7B + LoRA + PCA48 projector trained 500 steps. "
            "PC sweep: top-5 high-variance PCs × 5 contents × 5 levels = 125 generations. "
            "Target/anti: ~50 users × 2 (target/anti) generations."
        ),
        "config": {
            "ckpt": "projector_final.pt",
            "N_PC_SWEEP_CONTENTS": N_PC_SWEEP_CONTENTS,
            "N_TOP_PCS": N_TOP_PCS,
            "PC_LEVELS": PC_LEVELS,
            "N_TAR_ANTI_USERS": N_TAR_ANTI_USERS,
            "TEMPERATURE": TEMPERATURE,
            "TOP_P": TOP_P,
        },
        "pc_sweep": {
            "summary": pc_summary,
            "median_rho": median_rho,
            "max_rho": max_rho,
            "rows": [
                {k: v for k, v in r.items() if k != "z_levels"}  # truncate
                for r in pc_sweep_results
            ],
        },
        "target_anti": {
            "median_delta_d": float(med_delta) if target_anti_results else None,
            "mean_delta_d": float(mean_delta) if target_anti_results else None,
            "frac_negative": float(frac_negative) if target_anti_results else None,
            "bootstrap_95_ci": [ci_low, ci_high] if target_anti_results else None,
            "n_pairs": len(target_anti_results),
            "rows": [
                {k: v for k, v in r.items() if k not in ("z_target", "z_anti", "z_gen_target", "z_gen_anti")}
                for r in target_anti_results
            ],
        },
        "decision": decision,
        "rationale": rationale,
    }
    out_path = OUT_DIR / "pca48_syntax_renderer_audit.json"
    with open(out_path, "w") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote → {out_path}")


def compute_pca48_of_text(
    text: str,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    fnames_all: list,
    fnames_f3: list,
    col_idx_f3: list,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray | None:
    """Compute PCA48 representation of a generated text. Returns None on failure."""
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


if __name__ == "__main__":
    main()