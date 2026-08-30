#!/usr/bin/env python3
"""Phase 6.A.3 — Training loop for PCA48 Syntax Controller.

用户指令 2026-08-30:
  Qwen2-7B 冻结 + PCA48 projector (trainable) + LoRA r=16 q/k/v/o
  Loss = L_LM only (Phase 6.A pilot, 暂不加 L_control / DPO)

训练:
  - Dataset: scratch2/pca48_dataset/train.jsonl (5000 samples)
  - AdamW, lr=1e-4 (LoRA + projector 标准值)
  - Batch size: 4 (Qwen2-7B bf16 ~14GB + activations)
  - Steps: 500 (1 epoch ≈ 1250 steps; pilot 半 epoch)
  - Save checkpoint 每 100 step

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
TRAIN_JSONL = SCRATCH / "pca48_dataset" / "train.jsonl"
LOG_DIR = SCRATCH / "logs"
CKPT_DIR = SCRATCH / "pca48_ckpt"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 6.A.3 训练配置 (硬编码)
N_STEPS = 500
LOG_EVERY = 10
CKPT_EVERY = 100
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 30
GRAD_ACCUM = 1
SEED = 42
PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pca48_train] {msg}", flush=True)


def main():
    log("=== Phase 6.A.3 — Train PCA48 Syntax Controller (L_LM only) ===")

    # 1. Load dataset
    log(f"loading dataset: {TRAIN_JSONL}")
    samples = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            samples.append(json.loads(line))
    log(f"  loaded {len(samples)} samples")

    z_all = np.asarray([s["z_48"] for s in samples], dtype=np.float32)
    attrs_all = [s["c_i"] for s in samples]
    target_all = [s["s_i"] for s in samples]

    log(f"  z_all shape: {z_all.shape}, attrs: {len(attrs_all)}, targets: {len(target_all)}")

    # 2. Build model
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix

    log("loading Qwen2-7B + LoRA + PCA48 projector ...")
    t0 = time.time()
    model = Qwen2WithPrefix()
    log(f"  model loaded in {time.time() - t0:.1f}s")

    # 3. Optimizer
    trainable_params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    log(f"  trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=LR, weight_decay=WEIGHT_DECAY
    )

    # 4. LR warmup
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 5. Training loop
    log(f"\n--- training {N_STEPS} steps, batch={BATCH_SIZE}, lr={LR} ---")
    rng = np.random.default_rng(SEED)
    losses = []
    t_start = time.time()
    for step in range(N_STEPS):
        # Sample batch
        idx = rng.choice(len(samples), size=BATCH_SIZE, replace=False)
        z = torch.from_numpy(z_all[idx]).to("cuda")
        attrs_batch = [attrs_all[i] for i in idx]
        target_batch = [target_all[i] for i in idx]

        loss = model(z, attrs_batch, target_batch)
        loss_val = float(loss.detach())
        losses.append(loss_val)

        loss.backward()
        if GRAD_ACCUM > 1 and (step + 1) % GRAD_ACCUM != 0:
            continue
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if (step + 1) % LOG_EVERY == 0:
            recent = losses[-LOG_EVERY:]
            elapsed = time.time() - t_start
            log(f"  step {step + 1:4d}/{N_STEPS} | loss={np.mean(recent):.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f} | "
                f"{elapsed:.0f}s elapsed ({elapsed / (step + 1):.2f}s/step)")

        if (step + 1) % CKPT_EVERY == 0:
            ckpt_path = CKPT_DIR / f"projector_step{step + 1}.pt"
            model.save_projector(str(ckpt_path))
            log(f"  ckpt saved → {ckpt_path}")

    # Final ckpt
    final_ckpt = CKPT_DIR / "projector_final.pt"
    model.save_projector(str(final_ckpt))
    log(f"\nfinal ckpt saved → {final_ckpt}")

    # Save loss curve
    log_path = LOG_DIR / "phase6a3_train_losses.json"
    with open(log_path, "w") as f:
        json.dump({
            "losses": losses,
            "config": {
                "N_STEPS": N_STEPS, "BATCH_SIZE": BATCH_SIZE, "LR": LR,
                "WARMUP_STEPS": WARMUP_STEPS, "LORA_R": 16,
                "PREFIX_LEN": 8, "PCA_DIM": PCA_DIM,
            },
            "final_loss_mean_last_50": float(np.mean(losses[-50:])),
            "first_loss_mean_first_50": float(np.mean(losses[:50])),
        }, f, indent=2)
    log(f"loss curve → {log_path}")

    log(f"\n=== Training done in {time.time() - t_start:.0f}s ===")
    log(f"  final loss (last 50): {np.mean(losses[-50:]):.4f}")
    log(f"  first loss (first 50): {np.mean(losses[:50]):.4f}")


if __name__ == "__main__":
    main()