#!/usr/bin/env python3
"""Phase 6.B.2.1 — Build offline preference dataset for DPO.

用户指令 2026-08-30: 不要直接每 step 在线生成. 先构造固定 preference dataset:
  对每 (content, z_target): 生成 8 candidates
  y+ = argmin d (closest)
  y- = rank-3 (hard negative, NOT argmax — argmax 太容易区分)
  ref_logprob_y+, ref_logprob_y- (Phase 6.A frozen)

输出: scratch2/pca48_preferences/preference_dataset.jsonl

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
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
DATA_DIR = SCRATCH / "pca48_dataset"
CKPT_DIR = SCRATCH / "pca48_ckpt"
LOG_DIR = SCRATCH / "logs"
PREF_DIR = SCRATCH / "pca48_preferences"
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"

PREF_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Config (locked) ===
N_PAIRS = 2000
N_CANDIDATES = 8
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 50
SEED = 42
BATCH_CONTENTS = 4  # 4 contents × 8 candidates = batch 32 per forward
HARD_NEGATIVE_RANK = 3  # 1-indexed: 1=argmin(used as y+), 3=hard negative
MIN_DELTA_D = 0.3  # filter pairs where y+/y- too similar (no preference signal)
PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [build_pref] {msg}", flush=True)


def compute_logprob(model, z: torch.Tensor, attrs: list[str], target_text: str) -> float:
    """Compute sum of log P(target | prefix(z) + prompt(attrs)) under model."""
    device = next(model.model.parameters()).device
    B = z.shape[0]
    assert B == 1, "compute_logprob assumes B=1"
    z_d = z.to(device=device, dtype=model.dtype)
    prefix_embeds = model.projector(z_d)  # (1, prefix_len, H)

    # Tokenize prompt + target
    orig_padding = model.tokenizer.padding_side
    model.tokenizer.padding_side = "right"
    try:
        prompt = model._build_prompt([attrs])[0]
        prompt_enc = model.tokenizer(prompt, return_tensors="pt",
                                     add_special_tokens=False).to(device)
        target_enc = model.tokenizer(target_text, return_tensors="pt",
                                     add_special_tokens=False).to(device)
    finally:
        model.tokenizer.padding_side = orig_padding

    prompt_ids = prompt_enc.input_ids
    target_ids = target_enc.input_ids

    embed_layer = model.model.get_input_embeddings()
    prompt_embeds = embed_layer(prompt_ids).to(model.dtype)
    target_embeds = embed_layer(target_ids).to(model.dtype)

    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)
    out = model.model(inputs_embeds=inputs_embeds, use_cache=False)
    logits = out.logits  # (1, total_len, V)

    # logits at position prefix_len + prompt_len - 1 + t predicts target[t]
    prefix_len = model.prefix_len
    prompt_len = prompt_ids.shape[1]
    target_len = target_ids.shape[1]

    pred_logits = logits[0, prefix_len + prompt_len - 1:
                            prefix_len + prompt_len - 1 + target_len, :].float()  # (T, V)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    target_logprobs = log_probs.gather(1, target_ids[0].unsqueeze(-1)).squeeze(-1)
    return float(target_logprobs.sum().item())


def main():
    log("=== Phase 6.B.2.1 — Build offline preference dataset ===")

    log("loading dataset ...")
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    log(f"  {len(samples)} samples loaded")

    held_out = samples[:N_PAIRS]
    log(f"  using first {N_PAIRS} samples as preference pairs")

    log("loading Qwen2-7B + LoRA + PCA48 projector (Phase 6.A ckpt, used for both gen and ref) ...")
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix

    model = Qwen2WithPrefix()
    log(f"  loading ckpt from {CKPT_DIR / 'projector_final.pt'}")
    model.load_projector(str(CKPT_DIR / 'projector_final.pt'))
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

    # === Phase A: Batched generation ===
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
                "attrs": sample["c_i"],
                "z_target": sample["z_48"],
                "candidates": cands_j,
            })

        if (bi + 1) % 10 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            done_pairs = min((bi + 1) * BATCH_CONTENTS, N_PAIRS)
            log(f"  batch {bi+1}/{n_batches} ({done_pairs}/{N_PAIRS}, "
                f"{elapsed:.0f}s, {elapsed / done_pairs:.2f}s/pair)")

    log(f"\ngeneration done in {time.time() - t0:.0f}s")

    # === Phase B: Batch F3 + PCA48 for distance computation ===
    log(f"\nphase B: F3 + PCA48 for {len(all_pairs_candidates) * N_CANDIDATES} candidates ...")
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
    log(f"  F3 done in {time.time() - t1:.0f}s")

    # === Phase C: Per-pair: pick y+ / y-, compute ref logprobs ===
    log(f"\nphase C: selecting y+/y- + computing ref logprobs ...")
    preference_pairs = []
    skipped_weak = 0
    skipped_invalid = 0
    t2 = time.time()
    for pi, pc in enumerate(all_pairs_candidates):
        z_target = np.asarray(pc["z_target"], dtype=np.float32)

        # Compute d for each candidate
        ds_cand = []
        for j in range(N_CANDIDATES):
            flat_idx = pi * N_CANDIDATES + j
            if flat_idx >= len(flat_z):
                ds_cand.append(None)
                continue
            z = flat_z[flat_idx]
            if z is None:
                ds_cand.append(None)
                continue
            ds_cand.append(float(np.linalg.norm(z - z_target)))

        if sum(d is not None for d in ds_cand) < 4:
            skipped_invalid += 1
            continue

        # Sort by d (ascending), keep candidates with valid d
        valid_idx = [i for i, d in enumerate(ds_cand) if d is not None]
        valid_ds = [ds_cand[i] for i in valid_idx]
        sorted_pairs = sorted(zip(valid_ds, valid_idx), key=lambda x: x[0])

        if len(sorted_pairs) < HARD_NEGATIVE_RANK:
            skipped_invalid += 1
            continue

        d_y_plus, y_plus_idx = sorted_pairs[0]
        d_y_minus, y_minus_idx = sorted_pairs[HARD_NEGATIVE_RANK - 1]  # rank 3 = index 2
        delta_d = d_y_minus - d_y_plus

        if delta_d < MIN_DELTA_D:
            skipped_weak += 1
            continue

        y_plus_text = pc["candidates"][y_plus_idx]
        y_minus_text = pc["candidates"][y_minus_idx]

        # Compute ref logprobs
        z_target_t = torch.from_numpy(z_target).unsqueeze(0).to("cuda")
        ref_lp_y_plus = compute_logprob(model, z_target_t, pc["attrs"], y_plus_text)
        ref_lp_y_minus = compute_logprob(model, z_target_t, pc["attrs"], y_minus_text)

        preference_pairs.append({
            "pair_idx": pi,
            "attrs": pc["attrs"],
            "z_target": pc["z_target"],
            "y_plus": y_plus_text,
            "y_minus": y_minus_text,
            "ref_logprob_y_plus": ref_lp_y_plus,
            "ref_logprob_y_minus": ref_lp_y_minus,
            "d_y_plus": d_y_plus,
            "d_y_minus": d_y_minus,
            "delta_d": delta_d,
        })

        if (pi + 1) % 200 == 0:
            elapsed = time.time() - t2
            log(f"  preference pair {pi+1}/{N_PAIRS} ({elapsed:.0f}s, "
                f"{elapsed / (pi + 1):.2f}s/pair, "
                f"kept={len(preference_pairs)} skipped_weak={skipped_weak} "
                f"skipped_invalid={skipped_invalid})")

    log(f"\npreference building done in {time.time() - t2:.0f}s")
    log(f"  total preference pairs: {len(preference_pairs)} "
        f"(from {N_PAIRS} input pairs)")
    log(f"  skipped (weak Δd<{MIN_DELTA_D}): {skipped_weak}")
    log(f"  skipped (invalid candidates): {skipped_invalid}")

    # Stats
    if preference_pairs:
        delta_ds = [p["delta_d"] for p in preference_pairs]
        log(f"  Δd (y- minus y+) median: {np.median(delta_ds):.3f}, "
            f"mean: {np.mean(delta_ds):.3f}")

    # Save
    out_path = PREF_DIR / "preference_dataset.jsonl"
    with open(out_path, "w") as f:
        for p in preference_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log(f"\nwrote → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
        f"{len(preference_pairs)} pairs)")

    # Stats JSON
    stats_path = PREF_DIR / "preference_dataset_stats.json"
    stats = {
        "n_input_pairs": N_PAIRS,
        "n_preference_pairs": len(preference_pairs),
        "skipped_weak_delta_d": skipped_weak,
        "skipped_invalid": skipped_invalid,
        "min_delta_d_threshold": MIN_DELTA_D,
        "hard_negative_rank": HARD_NEGATIVE_RANK,
        "delta_d_median": float(np.median(delta_ds)) if preference_pairs else None,
        "delta_d_mean": float(np.mean(delta_ds)) if preference_pairs else None,
        "delta_d_p25": float(np.percentile(delta_ds, 25)) if preference_pairs else None,
        "delta_d_p75": float(np.percentile(delta_ds, 75)) if preference_pairs else None,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log(f"wrote → {stats_path}")


if __name__ == "__main__":
    main()
