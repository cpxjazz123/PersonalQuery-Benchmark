#!/usr/bin/env python3
"""Phase 6.B.0 — PCA48 Linear Probe Diagnostic on Qwen2-7B.

诊断: Qwen2-7B hidden states 能否预测 PCA48?
  若 yes → Phase 6.A 失败因为 conditioning collapse (L_LM-only 没强迫使用 PCA48)
  若 no  → representation mismatch, 需换 generation mechanism

设计:
  Step 1: extract + pool hidden states (last-token + mean) from Qwen2-7B
  Step 2: per (layer, pool) train linear probe → z_PCA48 (80/20 split)
  Step 3: held-out R² per PC + Pearson + Spearman
  Step 4: 决策

输出: result/gaussian/pca48_probe_diagnostic.json

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
DATA_DIR = SCRATCH / "pca48_dataset"
HIDDEN_DIR = SCRATCH / "pca48_hidden"
LOG_DIR = SCRATCH / "logs"
OUT_DIR = REPO_ROOT / "result" / "gaussian"
HIDDEN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

QWEN_PATH = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"

PROBE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 27]  # embedding + 7 mid/late layers
POOLINGS = ["last_token", "mean"]
N_PROBE_EPOCHS = 30
LR = 1e-3
TEST_FRAC = 0.2
SEED = 42
BATCH_SIZE = 8
MAX_SEQ_LEN = 80
PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pca48_probe] {msg}", flush=True)


def main():
    log("=== Phase 6.B.0 — PCA48 Linear Probe Diagnostic ===")

    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    z_all = np.asarray([s["z_48"] for s in samples], dtype=np.float32)
    texts = [s["s_i"] for s in samples]
    log(f"  {len(samples)} samples, z_all {z_all.shape}, "
        f"z std per-dim mean={z_all.std(axis=0).mean():.3f}")

    log("loading Qwen2-7B ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    log("  loaded")

    hidden_path = HIDDEN_DIR / f"pooled_layers{'_'.join(map(str, PROBE_LAYERS))}_N{len(texts)}.pt"
    if hidden_path.exists():
        log(f"loading cached pooled hidden states from {hidden_path}")
        pooled_cache = torch.load(hidden_path, map_location="cpu")
    else:
        log(f"extracting + pooling hidden states for {len(texts)} sentences ...")
        t0 = time.time()
        pooled_cache = {layer: {p: [] for p in POOLINGS} for layer in PROBE_LAYERS}
        with torch.no_grad():
            for st in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[st:st + BATCH_SIZE]
                enc = tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_SEQ_LEN,
                ).to("cuda")
                out = model(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    output_hidden_states=True,
                )
                attn = enc.attention_mask
                for layer in PROBE_LAYERS:
                    h = out.hidden_states[layer].float()  # (B, T, H)
                    B = h.shape[0]
                    last_idx = attn.sum(dim=1) - 1  # (B,)
                    last_tok = h[torch.arange(B, device="cuda"), last_idx]  # (B, H)
                    mask = attn.unsqueeze(-1).float()  # (B, T, 1)
                    mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    pooled_cache[layer]["last_token"].append(last_tok.cpu())
                    pooled_cache[layer]["mean"].append(mean.cpu())
                if (st // BATCH_SIZE) % 50 == 0:
                    log(f"  batch {st // BATCH_SIZE + 1}/{(len(texts)-1)//BATCH_SIZE+1}, "
                        f"{min(st+BATCH_SIZE, len(texts))}/{len(texts)} ({time.time()-t0:.0f}s)")
        for layer in PROBE_LAYERS:
            for p in POOLINGS:
                pooled_cache[layer][p] = torch.cat(pooled_cache[layer][p], dim=0)
        torch.save(pooled_cache, hidden_path)
        log(f"  saved → {hidden_path} ({time.time()-t0:.0f}s)")

    H_dim = pooled_cache[PROBE_LAYERS[0]]["last_token"].shape[-1]
    log(f"  hidden dim: {H_dim}")

    N = len(texts)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    n_test = int(N * TEST_FRAC)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    z_train = z_all[train_idx]
    z_test = z_all[test_idx]
    log(f"  split: train={len(train_idx)}, test={len(test_idx)}")

    log("\ntraining probes ...")
    results = []
    best_r2 = -np.inf
    best_cfg = None
    device = "cuda"
    for layer in PROBE_LAYERS:
        for pool in POOLINGS:
            log(f"  L{layer} {pool}")
            h_train = pooled_cache[layer][pool][train_idx].to(device)
            h_test = pooled_cache[layer][pool][test_idx].to(device)
            zt = torch.from_numpy(z_train).to(device)
            zte = torch.from_numpy(z_test).to(device)

            probe = nn.Linear(H_dim, PCA_DIM, bias=True).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=LR)
            for epoch in range(N_PROBE_EPOCHS):
                perm2 = torch.randperm(h_train.shape[0], device=device)
                for s in range(0, h_train.shape[0], 32):
                    idx = perm2[s:s+32]
                    pred = probe(h_train[idx])
                    loss = nn.functional.mse_loss(pred, zt[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

            probe.eval()
            with torch.no_grad():
                pred_test = probe(h_test).cpu().numpy()
            r2_per_pc = 1 - ((z_test - pred_test) ** 2).sum(axis=0) / \
                          ((z_test - z_test.mean(axis=0)) ** 2).sum(axis=0)
            from scipy.stats import pearsonr, spearmanr
            pearson_per_pc = []
            spearman_per_pc = []
            for j in range(PCA_DIM):
                rp, _ = pearsonr(z_test[:, j], pred_test[:, j])
                rs, _ = spearmanr(z_test[:, j], pred_test[:, j])
                pearson_per_pc.append(float(rp))
                spearman_per_pc.append(float(rs))
            mean_r2 = float(np.mean(r2_per_pc))
            mean_pearson = float(np.mean(np.abs(pearson_per_pc)))
            mean_spearman = float(np.mean(np.abs(spearman_per_pc)))
            log(f"    R²={mean_r2:+.3f}, |Pearson|={mean_pearson:.3f}, |Spearman|={mean_spearman:.3f}")
            results.append({
                "layer": int(layer),
                "pool": pool,
                "mean_r2": mean_r2,
                "mean_abs_pearson": mean_pearson,
                "mean_abs_spearman": mean_spearman,
                "r2_per_pc": r2_per_pc.tolist(),
                "pearson_per_pc": pearson_per_pc,
                "spearman_per_pc": spearman_per_pc,
            })
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_cfg = (layer, pool)

    log(f"\nBest: L{best_cfg[0]} {best_cfg[1]} R²={best_r2:+.3f}")

    best_result = max(results, key=lambda r: r["mean_r2"])
    log(f"\nPer-PC R² for best (L{best_result['layer']} {best_result['pool']}):")
    for j in range(PCA_DIM):
        log(f"  PC{j:2d}: R²={best_result['r2_per_pc'][j]:+.3f}  "
            f"r={best_result['pearson_per_pc'][j]:+.3f}")

    if best_r2 > 0.3:
        decision = "PCA48_IN_QWEN"
        rationale = (f"Best probe R²={best_r2:+.3f} > 0.3. PCA48 info IS in Qwen hidden states. "
                     f"Phase 6.A failed due to conditioning collapse: optimizer 没强迫模型使用 PCA48. "
                     f"Next: Phase 6.B.1 PCA-aware loss (L = L_LM + λ·L_PCA) "
                     f"or Phase 6.B.2 DPO preference.")
    elif best_r2 > 0.1:
        decision = "WEAK_PCA48"
        rationale = (f"Best probe R²={best_r2:+.3f} ∈ [0.1, 0.3]. PCA48 info PARTIALLY in Qwen. "
                     f"Need MLP probe / ensemble, or better representation.")
    else:
        decision = "PCA48_NOT_IN_QWEN"
        rationale = (f"Best probe R²={best_r2:+.3f} < 0.1. Strong evidence PCA48 representation "
                     f"与 Qwen 隐藏空间不匹配. Next: Phase 6.B.3 hard control "
                     f"(posterior constrained decoding / template retrieval).")

    log(f"\nDECISION: {decision}")
    log(f"  {rationale}")

    out_doc = {
        "description": (
            "Phase 6.B.0 — PCA48 Linear Probe Diagnostic. "
            f"Linear probe (Qwen hidden → PCA48) per (layer ∈ {PROBE_LAYERS}) × "
            f"(pool ∈ {POOLINGS}). 5000 sentences, 80/20 train/test split."
        ),
        "config": {
            "PROBE_LAYERS": PROBE_LAYERS, "POOLINGS": POOLINGS,
            "N_PROBE_EPOCHS": N_PROBE_EPOCHS, "LR": LR,
            "TEST_FRAC": TEST_FRAC, "BATCH_SIZE": BATCH_SIZE,
            "MAX_SEQ_LEN": MAX_SEQ_LEN, "SEED": SEED,
        },
        "results": results,
        "best_cfg": {"layer": best_cfg[0], "pool": best_cfg[1]},
        "best_r2": best_r2,
        "decision": decision,
        "rationale": rationale,
    }
    out_path = OUT_DIR / "pca48_probe_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()
