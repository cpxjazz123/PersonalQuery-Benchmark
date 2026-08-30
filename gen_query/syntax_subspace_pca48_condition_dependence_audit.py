#!/usr/bin/env python3
"""Phase 6.B.2-audit — Condition-dependence rate + Phase 6.A preference audit.

用户指令 2026-08-30:
1800-pair 失败可能因为 Phase 6.A 完全忽略 z_t → 不同 target 的 preference
互相冲突(gradient cancel),而不是 impl bug.

Audit 1: Phase 6.A 是否有 z_t preference?
  对每个 pair (z_t, c, y+, y-) 计算:
    ρ(ref_logprob_y+, -d_y+)  — Spearman
    ρ(ref_logprob_y-, -d_y-)
  如果 ρ < 0 且 |ρ| > 0.2 → Phase 6.A 有 z_t preference, DPO 可放大
  如果 ρ ≈ 0 → Phase 6.A 没 z_t preference, preference conflict 主因

Audit 2: d 与 length 的相关性(排除 length bias)
  ρ(-d_y+, length_y+)
  如果 ρ 接近 1 → d 是 length proxy → preference signal 是 length 偏差

Audit 3: 重新计算 z(y+), z(y-) for 1800 pairs, 然后:
  - 在 8 candidates 中,z_t_self 下的 closest ≠ z_t_other 下的 closest?
  - report condition-dependence rate = P(preference flips under different z_t)

Audit 4 (主要): 条件依赖率
  任意两个 z_t 值 (z_A, z_B),在 z_A 下 y+ 是 c_i,在 z_B 下 y+ 是 c_j。
  i ≠ j 的比例 = 条件依赖率。
  高 (e.g. > 50%) → task 真的要求读取 z_t
  低 (e.g. < 20%) → candidates dominated by length/format/content,无 z_t signal

为 audit 3/4,需要:
  - 重新跑 200 个 pair × 8 candidates 的 generation (Phase A 模式)
  - 计算 z(y_i) for each candidate (Phase B F3+PCA48)
  - 计算 z_t_A vs z_t_B 下 argmin 的分歧

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
PREF_DIR = SCRATCH / "pca48_preferences"
LOG_DIR = SCRATCH / "logs"
CKPT_DIR = SCRATCH / "pca48_ckpt"
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Audit config ===
N_PAIRS = 200       # subset for audit 3/4
N_CANDIDATES = 8
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 50
SEED = 42
BATCH_CONTENTS = 4
PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [cond_audit] {msg}", flush=True)


def main():
    log("=== Phase 6.B.2-audit — Condition-dependence ===")

    # ============================================================
    # Audit 1+2: Spearman on existing 1800 pairs (no generation needed)
    # ============================================================
    log("\n[Audit 1+2] Loading preference dataset for ref_logprob vs d analysis ...")
    pref_path = PREF_DIR / "preference_dataset.jsonl"
    pairs = []
    with open(pref_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  loaded {len(pairs)} pairs")

    # Per-pair: ref_logprob_y+, ref_logprob_y-, d_y+, d_y-
    rp_plus = np.array([p["ref_logprob_y_plus"] for p in pairs])
    rp_minus = np.array([p["ref_logprob_y_minus"] for p in pairs])
    d_plus = np.array([p["d_y_plus"] for p in pairs])
    d_minus = np.array([p["d_y_minus"] for p in pairs])
    len_plus = np.array([len(p["y_plus"]) for p in pairs])
    len_minus = np.array([len(p["y_minus"]) for p in pairs])

    # All (ref_logprob, d) pairs for both y+ and y-
    all_ref = np.concatenate([rp_plus, rp_minus])
    all_d = np.concatenate([d_plus, d_minus])
    all_len = np.concatenate([len_plus, len_minus])

    from scipy.stats import spearmanr

    # Audit 1a: ρ(ref_logprob, -d) — does lower d → higher ref_logprob?
    rho_preference, pval_preference = spearmanr(all_ref, -all_d)
    log(f"\n[Audit 1a] ρ(ref_logprob, -d) on {len(all_ref)} candidates:")
    log(f"  Spearman ρ = {rho_preference:+.4f}, p = {pval_preference:.2e}")
    log(f"  interpretation: ρ<0 → Phase 6.A prefers z_t-close sentences (good)")

    # Per-pair preference: ρ(ref_logprob diff, -d diff)
    rp_diff = rp_plus - rp_minus
    d_diff = d_plus - d_minus  # negative since d_y+ < d_y-
    rho_per_pair, pval_per_pair = spearmanr(rp_diff, -d_diff)
    log(f"\n[Audit 1b] Per-pair ρ(ref_logprob diff, -d diff):")
    log(f"  Spearman ρ = {rho_per_pair:+.4f}, p = {pval_per_pair:.2e}")
    log(f"  (positive = model prefers closer y+ over farther y-)")

    # Audit 2: length bias
    rho_len_ref, _ = spearmanr(all_ref, -all_len)
    rho_len_d, _ = spearmanr(all_d, all_len)
    log(f"\n[Audit 2] length bias:")
    log(f"  ρ(ref_logprob, -length) = {rho_len_ref:+.4f}  (model prefers shorter)")
    log(f"  ρ(d, length) = {rho_len_d:+.4f}  (do PCA48-close candidates also shorter?)")
    log(f"  if |ρ(d, length)| > 0.3 → d is partially length proxy → preference signal contaminated")

    # Partial correlation: residual ref_logprob after regressing out length
    from numpy.polynomial import polynomial as P
    # Simple linear regression: ref_logprob = a + b * length
    coeffs_ref_len = np.polyfit(all_len, all_ref, 1)
    ref_resid = all_ref - np.polyval(coeffs_ref_len, all_len)
    rho_partial, pval_partial = spearmanr(ref_resid, -all_d)
    log(f"\n[Audit 1c] Length-controlled ρ(ref_logprob residual, -d):")
    log(f"  Spearman ρ = {rho_partial:+.4f}, p = {pval_partial:.2e}")
    log(f"  interpretation: this is the pure z_t-preference (length removed)")
    log(f"  if |ρ_partial| > 0.15 → Phase 6.A genuinely prefers z_t-close sentences")

    # ============================================================
    # Audit 3+4: Condition-dependence rate via cross-pair comparison
    # ============================================================
    # Need: z(y_i) for each candidate from N_PAIRS × N_CANDIDATES generations
    log(f"\n[Audit 3+4] Condition-dependence via regeneration of {N_PAIRS} pairs ...")
    log(f"  loading dataset + Qwen2-7B + projector (Phase 6.A ckpt) ...")

    # Load samples
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    log(f"  {len(samples)} samples")

    held_out = samples[:N_PAIRS]

    # Load Qwen + projector
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix
    model = Qwen2WithPrefix()
    log(f"  loading ckpt from {CKPT_DIR / 'projector_final.pt'}")
    model.load_projector(str(CKPT_DIR / 'projector_final.pt'))
    model.eval()

    # Load user Gaussians
    gauss = json.load(open(GAUSS_PATH))
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])

    # Phase A: regenerate candidates
    log(f"\nphase A: generating {N_CANDIDATES} candidates × {N_PAIRS} pairs ...")
    all_pairs_candidates = []
    t0 = time.time()
    n_batches = (N_PAIRS + BATCH_CONTENTS - 1) // BATCH_CONTENTS
    for bi in range(n_batches):
        batch_samples = held_out[bi * BATCH_CONTENTS: (bi + 1) * BATCH_CONTENTS]
        bc = len(batch_samples)
        zs_list = []
        attrs_list = []
        for sample in batch_samples:
            z = np.asarray(sample["z_48"], dtype=np.float32)
            zs_list.append(torch.from_numpy(z))
            attrs_list.extend([sample["c_i"]] * N_CANDIDATES)
        zs_batched = torch.stack(zs_list, dim=0)
        zs_expanded = (zs_batched.unsqueeze(1)
                       .expand(bc, N_CANDIDATES, -1)
                       .reshape(bc * N_CANDIDATES, -1)
                       .contiguous().to("cuda"))
        torch.manual_seed(SEED + bi)
        candidates_flat = model.generate(
            zs_expanded, attrs_list,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )
        for j, sample in enumerate(batch_samples):
            pi = bi * BATCH_CONTENTS + j
            cands_j = candidates_flat[j * N_CANDIDATES: (j + 1) * N_CANDIDATES]
            all_pairs_candidates.append({
                "pair_idx": pi,
                "z_target": sample["z_48"],
                "candidates": cands_j,
            })
    log(f"  generation done in {time.time() - t0:.0f}s")

    # Phase B: F3+PCA48 for candidates
    log(f"\nphase B: F3+PCA48 for {len(all_pairs_candidates) * N_CANDIDATES} candidates ...")
    import spacy
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2

    nlp = spacy.load("en_core_web_sm")
    if "textcat" in nlp.pipe_names:
        nlp.remove_pipe("textcat")

    flat_texts = []
    for pc in all_pairs_candidates:
        flat_texts.extend(pc["candidates"])

    flat_z = []
    BATCH = 256
    t1 = time.time()
    for st in range(0, len(flat_texts), BATCH):
        batch_texts = flat_texts[st:st + BATCH]
        docs = list(nlp.pipe(batch_texts, batch_size=BATCH, n_process=8))
        for doc in docs:
            try:
                d = per_sentence_features_v2(doc)
                if d is None:
                    flat_z.append(None)
                    continue
                feats = np.zeros(len(fnames_all), dtype=np.float64)
                for j, fn in enumerate(fnames_all):
                    feats[j] = float(d.get(fn, 0.0))
                f3 = feats[col_idx_f3]
                f3_scaled = (f3 - scaler_mean) / np.maximum(scaler_scale, 1e-12)
                z = (f3_scaled - pca_mean) @ pca_components.T
                flat_z.append(z.astype(np.float32))
            except Exception:
                flat_z.append(None)
    log(f"  F3 done in {time.time() - t1:.0f}s")

    # Audit 3: per-pair, check if z_t_self vs z_t_other give different argmin
    log("\n[Audit 3] Condition-dependence rate ...")
    n_total = 0
    n_flip = 0
    per_pair_stats = []
    for pi, pc in enumerate(all_pairs_candidates):
        z_target = np.asarray(pc["z_target"], dtype=np.float32)
        cand_zs = []
        for j in range(N_CANDIDATES):
            idx = pi * N_CANDIDATES + j
            if idx < len(flat_z) and flat_z[idx] is not None:
                cand_zs.append((j, flat_z[idx]))
        if len(cand_zs) < 4:
            continue
        # self: argmin d(z_cand, z_target_self)
        ds_self = [(j, float(np.linalg.norm(z - z_target))) for j, z in cand_zs]
        ds_self_sorted = sorted(ds_self, key=lambda x: x[1])
        j_self = ds_self_sorted[0][0]

        # For each OTHER pair's z_target, find argmin
        for other_pi in range(len(all_pairs_candidates)):
            if other_pi == pi:
                continue
            other_z = np.asarray(all_pairs_candidates[other_pi]["z_target"], dtype=np.float32)
            ds_other = [(j, float(np.linalg.norm(z - other_z))) for j, z in cand_zs]
            j_other = sorted(ds_other, key=lambda x: x[1])[0][0]
            n_total += 1
            if j_other != j_self:
                n_flip += 1

        # Also record: per-pair, the rank-1 / rank-3 candidates under self
        per_pair_stats.append({
            "pair_idx": pi,
            "self_argmin_idx": j_self,
            "self_d_argmin": ds_self_sorted[0][1],
            "self_d_rank3": ds_self_sorted[2][1] if len(ds_self_sorted) >= 3 else None,
            "delta_d_self": ds_self_sorted[2][1] - ds_self_sorted[0][1] if len(ds_self_sorted) >= 3 else None,
        })

    flip_rate = n_flip / max(n_total, 1)
    log(f"  total comparisons: {n_total}")
    log(f"  preference flips: {n_flip} ({100 * flip_rate:.1f}%)")

    # Audit 4: condition-dependence strength — how does z_target_self differ from z_target_other in PCA48?
    z_targets = np.stack([np.asarray(pc["z_target"], dtype=np.float32)
                          for pc in all_pairs_candidates])
    pairwise_z_dists = np.linalg.norm(z_targets[:, None, :] - z_targets[None, :, :], axis=-1)
    np.fill_diagonal(pairwise_z_dists, np.nan)
    log(f"\n[Audit 4] PCA48 z_target pairwise distance:")
    log(f"  median dist: {np.nanmedian(pairwise_z_dists):.2f}")
    log(f"  P25 / P75:   {np.nanpercentile(pairwise_z_dists, 25):.2f} / "
        f"{np.nanpercentile(pairwise_z_dists, 75):.2f}")
    log(f"  (z_t's are spread out enough to test condition dependence)")

    # ============================================================
    # DECISION
    # ============================================================
    log("\n=== DECISION ===")
    log(f"\n[Audit 1c — pure z_t preference after removing length]:")
    log(f"  length-controlled ρ = {rho_partial:+.4f}")
    if abs(rho_partial) > 0.2:
        log("  → Phase 6.A HAS genuine z_t preference (after removing length bias)")
        log("  → DPO CAN amplify it")
    elif abs(rho_partial) > 0.1:
        log("  → Phase 6.A has weak z_t preference")
        log("  → DPO may help, but limited")
    else:
        log("  → Phase 6.A has NO z_t preference (after removing length bias)")
        log("  → preference conflict hypothesis CONFIRMED")
        log("  → MUST use condition-contrastive loss (Step 4)")

    log(f"\n[Audit 3 — condition-dependence rate]:")
    log(f"  flip rate = {100 * flip_rate:.1f}%")
    if flip_rate > 0.5:
        log("  → task genuinely requires reading z_t (high cond-dep)")
    elif flip_rate > 0.3:
        log("  → moderate condition dependence")
    else:
        log("  → low condition dependence → candidates mostly length/format-driven")

    log(f"\n[length bias (Audit 2)]:")
    log(f"  ρ(d, length) = {rho_len_d:+.4f}")
    if abs(rho_len_d) > 0.3:
        log("  ⚠️  d correlates with length → preference signal is partially length")

    # Save
    out = {
        "description": (
            "Phase 6.B.2-audit — Condition-dependence. "
            f"Uses {len(pairs)} existing pairs + regenerates {N_PAIRS} pairs for condition test. "
            "Audit 1: Phase 6.A's preference for z_t-close sentences. "
            "Audit 2: Length bias. "
            "Audit 3: Condition-dependence flip rate. "
            "Audit 4: z_target pairwise distances."
        ),
        "audit_1_preference": {
            "rho_ref_minus_d_all": float(rho_preference),
            "pval_ref_minus_d_all": float(pval_preference),
            "rho_per_pair_diff": float(rho_per_pair),
            "pval_per_pair_diff": float(pval_per_pair),
            "rho_length_controlled": float(rho_partial),
            "pval_length_controlled": float(pval_partial),
        },
        "audit_2_length_bias": {
            "rho_ref_minus_length": float(rho_len_ref),
            "rho_d_minus_length": float(rho_len_d),
        },
        "audit_3_condition_dependence": {
            "n_pairs": len(per_pair_stats),
            "n_total_comparisons": n_total,
            "n_flip": n_flip,
            "flip_rate": float(flip_rate),
        },
        "audit_4_z_target_spread": {
            "median_pairwise_dist": float(np.nanmedian(pairwise_z_dists)),
            "p25_pairwise_dist": float(np.nanpercentile(pairwise_z_dists, 25)),
            "p75_pairwise_dist": float(np.nanpercentile(pairwise_z_dists, 75)),
        },
        "diagnosis": {
            "has_zt_preference": abs(rho_partial) > 0.15,
            "high_condition_dependence": flip_rate > 0.5,
            "length_contaminated": abs(rho_len_d) > 0.3,
            "next_step": (
                "Step 4 condition-contrastive loss"
                if abs(rho_partial) < 0.1
                else "Step 2 length/format-matched + Step 3 small-LR DPO"
            ),
        },
    }
    out_path = LOG_DIR / "phase6b2_audit_condition_dependence.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()