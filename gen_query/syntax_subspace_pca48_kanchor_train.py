#!/usr/bin/env python3
"""Phase 6.C.2 — K-anchor controller training (smoke v1, 用户设计).

用户指令 2026-08-30:
- KMeans K=32 anchors (frozen, from train PCA48)
- 32x3584 learnable anchor_embeddings + softmax(-d²/τ)
- L = L_LM only (no align / DPO / CC / mirror)

Smoke config (Rule 18):
- MAX_STEPS=100 (smoke 验证 pipeline 跑通)
- BATCH_SIZE=2, LR=2e-5, τ=1.0
- ckpt 输出: /home/wlia0047/hj82_scratch2/wenyu/pca48_kanchor_ckpt/kanchor_smoke.pt
- losses 输出: /home/wlia0047/hj82_scratch2/wenyu/logs/phase6c2_kanchor_smoke_losses.json

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
CKPT_DIR = SCRATCH / "pca48_kanchor_ckpt"
LOG_DIR = SCRATCH / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# === Train config (SMOKE per Rule 18, 用户锁定 v1) ===
ANCHORS_PATH = SCRATCH / "pca48_kanchor" / "anchors.npy"
N_EPOCHS = 1
BATCH_SIZE = 2
LR = 2e-5
MAX_STEPS = 100         # smoke: 100 steps (用户: "第一版 100-200 steps 看 loss 是否下降")
TEMPERATURE = 1.0       # softmax τ
SEED = 42
LOG_EVERY = 25          # 100 / 25 = 4 log lines


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [kanchor_train] {msg}", flush=True)


def main():
    log("=== Phase 6.C.2 — K-anchor controller training (SMOKE v1) ===")

    torch.manual_seed(SEED)

    # Load dataset
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    log(f"  loaded {len(samples)} training pairs")

    # Load model
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_kanchor import Qwen2WithKAnchor

    log("  loading Qwen2-7B + LoRA + K-anchor controller ...")
    model = Qwen2WithKAnchor(
        anchors_path=str(ANCHORS_PATH),
        temperature=TEMPERATURE,
    )
    model.eval()  # LoRA + controller 在 forward 时自动激活训练模式(它们 requires_grad)
    log(f"  loaded K-anchor model")

    # Optimizer: only trainable params (anchor_embeddings + LoRA)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR)

    # Pad tokenizer
    model.tokenizer.padding_side = "right"

    # Loss tracking
    losses = {"total": [], "lm": []}
    weight_entropies = []
    t_start = time.time()

    log(f"  training: {MAX_STEPS} steps × {BATCH_SIZE} pairs, L=L_LM only, τ={TEMPERATURE}")

    rng = np.random.default_rng(SEED)
    idx_pool = rng.integers(0, len(samples), size=MAX_STEPS * BATCH_SIZE * 4)

    pool_ptr = 0
    for step in range(MAX_STEPS):
        batch_indices = idx_pool[pool_ptr: pool_ptr + BATCH_SIZE]
        pool_ptr += BATCH_SIZE
        if pool_ptr + BATCH_SIZE > len(idx_pool):
            pool_ptr = 0

        batch = [samples[int(i)] for i in batch_indices]
        z = torch.tensor(
            np.stack([np.asarray(s["z_48"], dtype=np.float32) for s in batch]),
            dtype=torch.float32, device="cuda",
        )
        attrs_list = [s["c_i"] for s in batch]
        target_texts = [s["s_i"] for s in batch]

        # Forward
        loss_lm, weights = model(
            z, attrs_list, target_texts,
            return_components=True,
        )

        # Track weight entropy (诊断 anchor collapse)
        # H(w) = -Σ w_k log w_k (with w_k > 0)
        eps = 1e-12
        entropy_per_sample = -(weights * torch.log(weights + eps)).sum(dim=-1)  # (B,)
        weight_entropies.append(float(entropy_per_sample.mean().item()))

        # Backward
        optimizer.zero_grad()
        loss_lm.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()

        losses["total"].append(float(loss_lm.item()))
        losses["lm"].append(float(loss_lm.item()))

        if (step + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t_start
            avg_loss = np.mean(losses["total"][-LOG_EVERY:])
            avg_H = np.mean(weight_entropies[-LOG_EVERY:])
            log(f"  step {step+1}/{MAX_STEPS} elapsed={elapsed:.0f}s  "
                f"L_LM={avg_loss:.3f}  H(w)={avg_H:.3f}")

    # Save ckpt
    final_path = CKPT_DIR / "kanchor_smoke.pt"
    model.save_controller(str(final_path))
    log(f"  saved ckpt → {final_path}")

    # Save losses + entropies
    out = {
        "losses": losses,
        "weight_entropies": weight_entropies,
        "config": {
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "temperature": TEMPERATURE,
            "seed": SEED,
        },
    }
    with open(LOG_DIR / "phase6c2_kanchor_smoke_losses.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"  saved losses → {LOG_DIR / 'phase6c2_kanchor_smoke_losses.json'}")

    # Summary
    log(f"\n=== TRAIN SUMMARY ===")
    log(f"  total steps: {MAX_STEPS}  total time: {time.time() - t_start:.0f}s")
    log(f"  final L_LM avg (last 25): {np.mean(losses['lm'][-LOG_EVERY:]):.4f}")
    log(f"  final H(w) avg (last 25): {np.mean(weight_entropies[-LOG_EVERY:]):.4f}")
    log(f"  H(w) reference: log(32)={np.log(32):.3f} (max), 0 (degenerate)")


if __name__ == "__main__":
    main()
