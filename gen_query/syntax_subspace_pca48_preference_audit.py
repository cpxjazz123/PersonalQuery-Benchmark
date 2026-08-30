#!/usr/bin/env python3
"""Phase 6.B.2a — PCA48 Preference Signal Audit (FAST batched).

用户指令 2026-08-30: Phase 6.B.2 DPO 前的可行性审计.
不要立即训练 DPO, 先检查当前模型是否能在同一个 (content, z_target)
下产生足够 diverse 的 8 条 candidates.

加速 (vs Phase 6.B.2a v1):
  - BATCH_CONTENTS=4: 4 contents × 8 candidates = batch 32 (vs batch 8)
  - N_PAIRS=300 (vs 500): 仍 statistically meaningful
  - MAX_NEW_TOKENS=50 (vs 60): queries 通常 ≤ 40 tokens

诊断:
  对 N=300 (content, z_target) 对, 生成 N=8 stochastic candidates.
  对每对计算:
    d_i = ||z(y_i) - z_target||   (PCA48 distance)
    Δd = max(d_i) - min(d_i)       (spread)
  报告:
    median Δd (across pairs)
    frac pairs with Δd > 1.0
    median d_min (best achievable)
  决策:
    DPO_GO: median Δd > 0.5 AND frac Δd>1 > 50%
      → generation capacity 够, DPO 可学 conditional selection
    DPO_PARTIAL: median Δd > 0.3 但 Δd>1 < 50%
      → partial spread, DPO 仍可能 work 但需精细调参
    DPO_NO_GO: median Δd < 0.2
      → 8 candidates 全 cluster, no preference signal

输出: result/gaussian/pca48_preference_audit.json

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import spacy
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
DATA_DIR = SCRATCH / "pca48_dataset"
CKPT_DIR = SCRATCH / "pca48_ckpt"
LOG_DIR = SCRATCH / "logs"
OUT_DIR = REPO_ROOT / "result" / "gaussian"
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"

# === FAST config ===
N_PAIRS = 300
N_CANDIDATES = 8
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 50
SEED = 42
BATCH_CONTENTS = 4  # 4 contents × 8 candidates = batch 32 per forward

PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pca48_pref_audit] {msg}", flush=True)


def main():
    log("=== Phase 6.B.2a FAST — PCA48 Preference Signal Audit ===")

    log("loading dataset ...")
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    log(f"  {len(samples)} samples loaded")

    held_out = samples[:N_PAIRS]
    log(f"  using first {N_PAIRS} samples as held-out pairs")

    log("loading Qwen2-7B + LoRA + PCA48 projector (Phase 6.A ckpt) ...")
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix

    model = Qwen2WithPrefix()
    log(f"  loading ckpt from {CKPT_DIR / 'projector_final.pt'}")
    model.load_projector(str(CKPT_DIR / "projector_final.pt"))
    model.eval()

    log("loading user Gaussians (F3 PCA48 projector) ...")
    gauss = json.load(open(GAUSS_PATH))
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])

    # === Phase A: Batched generation (4 contents × 8 candidates per batch) ===
    log(f"\ngenerating {N_CANDIDATES} candidates × {N_PAIRS} pairs "
        f"(batch {BATCH_CONTENTS} contents × {N_CANDIDATES} = {BATCH_CONTENTS * N_CANDIDATES}) ...")
    all_pairs_candidates = []  # list of dict per pair
    t0 = time.time()
    n_batches = (N_PAIRS + BATCH_CONTENTS - 1) // BATCH_CONTENTS
    for bi in range(n_batches):
        batch_samples = held_out[bi * BATCH_CONTENTS: (bi + 1) * BATCH_CONTENTS]
        bc = len(batch_samples)

        # Build batched inputs: each content × N_CANDIDATES
        zs_list = []
        attrs_list = []
        for sample in batch_samples:
            z = np.asarray(sample["z_48"], dtype=np.float32)
            zs_list.append(torch.from_numpy(z))
            attrs_list.extend([sample["c_i"]] * N_CANDIDATES)
        zs_batched = torch.stack(zs_list, dim=0)  # (bc, 48)
        # Expand to (bc * N_CANDIDATES, 48)
        zs_expanded = (zs_batched.unsqueeze(1)
                       .expand(bc, N_CANDIDATES, -1)
                       .reshape(bc * N_CANDIDATES, -1)
                       .contiguous().to("cuda"))
        # attrs_list already (bc * N_CANDIDATES)

        torch.manual_seed(SEED + bi)
        candidates_flat = model.generate(
            zs_expanded, attrs_list,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )
        # Reshape back: (bc, N_CANDIDATES)
        for j, sample in enumerate(batch_samples):
            pi = bi * BATCH_CONTENTS + j
            cands_j = candidates_flat[j * N_CANDIDATES: (j + 1) * N_CANDIDATES]
            all_pairs_candidates.append({
                "pair_idx": pi,
                "attrs": sample["c_i"],
                "z_target": sample["z_48"],
                "candidates": cands_j,
            })

        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            done_pairs = min((bi + 1) * BATCH_CONTENTS, N_PAIRS)
            log(f"  batch {bi+1}/{n_batches} ({done_pairs}/{N_PAIRS} pairs, "
                f"{elapsed:.0f}s, {elapsed / done_pairs:.2f}s/pair)")

    log(f"\ngeneration done in {time.time() - t0:.0f}s")

    # === Phase B: Batch F3 + PCA48 computation ===
    log(f"\nbatch F3 + PCA48 for {len(all_pairs_candidates) * N_CANDIDATES} candidates ...")
    nlp = spacy.load("en_core_web_sm")
    if "textcat" in nlp.pipe_names:
        nlp.remove_pipe("textcat")
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2

    flat_texts = []
    flat_meta = []
    for pc in all_pairs_candidates:
        for j, txt in enumerate(pc["candidates"]):
            flat_texts.append(txt)
            flat_meta.append((pc["pair_idx"], j))

    log(f"  total candidates: {len(flat_texts)}")
    t1 = time.time()
    flat_z = []
    BATCH = 256
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
        if (st // BATCH) % 4 == 0:
            log(f"  F3 batch {st // BATCH + 1}/{(len(flat_texts) - 1) // BATCH + 1} "
                f"({min(st + BATCH, len(flat_texts))}/{len(flat_texts)}, "
                f"{time.time() - t1:.0f}s)")

    log(f"  F3 done in {time.time() - t1:.0f}s")

    # === Phase C: Aggregate per-pair preference stats ===
    log("\n=== Aggregate preference stats ===")
    pair_records = []
    for pc in all_pairs_candidates:
        pi = pc["pair_idx"]
        z_target = np.asarray(pc["z_target"], dtype=np.float32)
        ds_cand = []
        for j, txt in enumerate(pc["candidates"]):
            flat_idx = pi * N_CANDIDATES + j
            if flat_idx >= len(flat_z):
                continue
            z = flat_z[flat_idx]
            if z is None:
                continue
            d = float(np.linalg.norm(z - z_target))
            ds_cand.append(d)

        if len(ds_cand) < 2:
            continue

        d_min = min(ds_cand)
        d_max = max(ds_cand)
        d_mean = float(np.mean(ds_cand))
        delta_d = d_max - d_min
        y_plus_idx = int(np.argmin(ds_cand))
        y_minus_idx = int(np.argmax(ds_cand))
        hit_count_1 = sum(1 for d in ds_cand if d < 1.0)
        hit_count_2 = sum(1 for d in ds_cand if d < 2.0)
        hit_count_3 = sum(1 for d in ds_cand if d < 3.0)

        pair_records.append({
            "pair_idx": pi,
            "attrs": pc["attrs"],
            "d_min": d_min,
            "d_max": d_max,
            "d_mean": d_mean,
            "delta_d": delta_d,
            "y_plus_text": pc["candidates"][y_plus_idx],
            "y_minus_text": pc["candidates"][y_minus_idx],
            "d_y_plus": ds_cand[y_plus_idx],
            "d_y_minus": ds_cand[y_minus_idx],
            "all_ds": ds_cand,
            "hit_count_d_lt_1": hit_count_1,
            "hit_count_d_lt_2": hit_count_2,
            "hit_count_d_lt_3": hit_count_3,
        })

    delta_ds = [r["delta_d"] for r in pair_records]
    d_mins = [r["d_min"] for r in pair_records]
    d_means = [r["d_mean"] for r in pair_records]
    spread_1 = [d for d in delta_ds if d > 1.0]
    spread_2 = [d for d in delta_ds if d > 2.0]
    log(f"  total pairs (≥2 valid candidates): {len(pair_records)}")
    log(f"  Δd (max-min spread) median: {np.median(delta_ds):.3f}, "
        f"mean: {np.mean(delta_ds):.3f}, max: {max(delta_ds):.3f}")
    log(f"  Δd > 1.0: {len(spread_1)} ({100*len(spread_1)/len(pair_records):.1f}%)")
    log(f"  Δd > 2.0: {len(spread_2)} ({100*len(spread_2)/len(pair_records):.1f}%)")
    log(f"  d_min median: {np.median(d_mins):.3f}, mean: {np.mean(d_mins):.3f}")
    log(f"  d_mean median: {np.median(d_means):.3f}")
    log(f"  best hit@1 (d<1): "
        f"{sum(1 for r in pair_records if r['d_min']<1)} ({100*sum(1 for r in pair_records if r['d_min']<1)/len(pair_records):.1f}%)")
    log(f"  best hit@1 (d<2): "
        f"{sum(1 for r in pair_records if r['d_min']<2)} ({100*sum(1 for r in pair_records if r['d_min']<2)/len(pair_records):.1f}%)")
    log(f"  best hit@1 (d<3): "
        f"{sum(1 for r in pair_records if r['d_min']<3)} ({100*sum(1 for r in pair_records if r['d_min']<3)/len(pair_records):.1f}%)")

    median_delta = float(np.median(delta_ds))
    frac_spread_1 = len(spread_1) / len(pair_records)
    frac_spread_2 = len(spread_2) / len(pair_records)
    if median_delta > 0.5 and frac_spread_1 > 0.5:
        decision = "DPO_GO"
        rationale = (f"median Δd={median_delta:.3f}>0.5 AND {frac_spread_1*100:.1f}% pairs have Δd>1.0. "
                     f"Model 已能产生 diverse PCA48 candidates, DPO 可学 conditional selection.")
    elif median_delta > 0.3:
        decision = "DPO_PARTIAL"
        rationale = (f"median Δd={median_delta:.3f}>0.3 but only {frac_spread_1*100:.1f}% pairs have Δd>1.0. "
                     f"Partial spread, DPO 仍可能 work 但需精细调参.")
    else:
        decision = "DPO_NO_GO"
        rationale = (f"median Δd={median_delta:.3f}<0.3. 8 candidates 全 cluster in PCA48 space, "
                     f"no preference signal. DPO 没有 sufficient positive/negative pairs. "
                     f"Next: 需要换 generation mechanism (constrained decoding / full finetune / hybrid).")

    log(f"\nDECISION: {decision}")
    log(f"  {rationale}")

    out_doc = {
        "description": (
            "Phase 6.B.2a — PCA48 Preference Signal Audit (FAST batched). "
            f"For each of {N_PAIRS} (content, z_target) pairs, "
            f"generate {N_CANDIDATES} stochastic candidates (T={TEMPERATURE}). "
            f"Batch {BATCH_CONTENTS} contents × {N_CANDIDATES} candidates = "
            f"batch {BATCH_CONTENTS * N_CANDIDATES} per forward. "
            f"Compute z(y_i) via F3+PCA48 evaluator, d_i = L2(z(y_i), z_target). "
            f"Report Δd = max - min spread, d_min, hit count."
        ),
        "config": {
            "N_PAIRS": N_PAIRS, "N_CANDIDATES": N_CANDIDATES,
            "BATCH_CONTENTS": BATCH_CONTENTS,
            "TEMPERATURE": TEMPERATURE, "TOP_P": TOP_P,
            "MAX_NEW_TOKENS": MAX_NEW_TOKENS, "SEED": SEED,
            "ckpt": "projector_final.pt (Phase 6.A)",
        },
        "aggregate": {
            "n_pairs_valid": len(pair_records),
            "delta_d_median": median_delta,
            "delta_d_mean": float(np.mean(delta_ds)),
            "delta_d_max": float(max(delta_ds)),
            "frac_delta_d_gt_1": frac_spread_1,
            "frac_delta_d_gt_2": frac_spread_2,
            "d_min_median": float(np.median(d_mins)),
            "d_min_mean": float(np.mean(d_mins)),
            "d_mean_median": float(np.median(d_means)),
            "best_hit_at_1_lt1": int(sum(1 for r in pair_records if r['d_min']<1)),
            "best_hit_at_1_lt2": int(sum(1 for r in pair_records if r['d_min']<2)),
            "best_hit_at_1_lt3": int(sum(1 for r in pair_records if r['d_min']<3)),
        },
        "decision": decision,
        "rationale": rationale,
        "pair_records_sample": pair_records[:20],
    }
    out_path = OUT_DIR / "pca48_preference_audit.json"
    with open(out_path, "w") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()
