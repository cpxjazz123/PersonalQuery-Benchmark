#!/usr/bin/env python3
"""Phase 6.B.2.2 — DPO Training on offline preference dataset.

用户指令 2026-08-30: offline DPO with precomputed ref logprobs (Phase 6.A frozen).
- L = L_DPO, β=0.1, LR=2e-5, 2 epochs, batch=4 pairs (8 forward passes/step)
- Reference model: Phase 6.A ckpt, frozen
- Trainable: PCA48 projector + LoRA (Qwen2-7B frozen)
- Loss: -logsigmoid(β * ((log π_θ(y+|x) - log π_ref(y+|x)) - (log π_θ(y-|x) - log π_ref(y-|x))))

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
DATA_DIR = SCRATCH / "pca48_dataset"
PREF_DIR = SCRATCH / "pca48_preferences"
LOG_DIR = SCRATCH / "logs"
INIT_CKPT = SCRATCH / "pca48_ckpt" / "projector_final.pt"
CKPT_DIR = SCRATCH / "pca48_dpo_ckpt"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# === DPO config (locked) ===
N_EPOCHS = 2
BATCH_SIZE = 4   # pairs per step (8 forward passes)
LR = 2e-5
BETA = 0.1
WARMUP_STEPS = 30
GRAD_CLIP = 1.0
LOG_EVERY = 10
CKPT_EVERY = 100
SEED = 42
MAX_PAIRS = 1500  # use first 1500 from preference dataset

PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [dpo_train] {msg}", flush=True)


def compute_logprob(model, z: torch.Tensor, attrs: list[str], target_text: str) -> torch.Tensor:
    """Compute sum of log P(target | prefix(z) + prompt(attrs)) under model.
    Returns a torch.Tensor (scalar) WITH gradient flow, on cuda, dtype=float32.

    B=1 only (called per-pair).
    """
    device = next(model.model.parameters()).device
    z_d = z.to(device=device, dtype=model.dtype)
    prefix_embeds = model.projector(z_d)

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
    logits = out.logits.float()  # (1, total, V)

    prefix_len = model.prefix_len
    prompt_len = prompt_ids.shape[1]
    target_len = target_ids.shape[1]

    pred_logits = logits[0, prefix_len + prompt_len - 1:
                            prefix_len + prompt_len - 1 + target_len, :]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    target_logprobs = log_probs.gather(1, target_ids[0].unsqueeze(-1)).squeeze(-1)
    # KEEP as tensor (with gradient), do NOT call .item()
    return target_logprobs.sum()


def dpo_loss(curr_lp_plus, curr_lp_minus, ref_lp_plus, ref_lp_minus, beta):
    """
    DPO loss:
      L = -logsigmoid(β * (log π_θ(y+|x)/π_θ(y-|x) - log π_ref(y+|x)/π_ref(y-|x)))
    Returns (loss, logratios) where logratios is the implicit reward difference.
    """
    pi_logratios = curr_lp_plus - curr_lp_minus
    ref_logratios = ref_lp_plus - ref_lp_minus
    logratios = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logratios).mean()
    return loss, logratios


def main():
    log("=== Phase 6.B.2.2 — DPO Training (offline) ===")

    # 1. Load preference dataset
    pref_path = PREF_DIR / "preference_dataset.jsonl"
    log(f"loading preference dataset from {pref_path}")
    pairs = []
    with open(pref_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  loaded {len(pairs)} preference pairs")

    if MAX_PAIRS and len(pairs) > MAX_PAIRS:
        pairs = pairs[:MAX_PAIRS]
        log(f"  truncated to first {len(pairs)} pairs (MAX_PAIRS={MAX_PAIRS})")

    # 2. Load model (initialized from Phase 6.A)
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix
    log("loading Qwen2-7B + LoRA + projector ...")
    model = Qwen2WithPrefix()
    log(f"  loading init ckpt from {INIT_CKPT}")
    model.load_projector(str(INIT_CKPT))
    # Note: model.train() enables LoRA training (gradients)
    model.train()

    # 3. Optimizer + scheduler
    trainable_params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    log(f"  trainable params: {n_trainable:,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.0)

    n_steps = (len(pairs) * N_EPOCHS) // BATCH_SIZE
    log(f"  total steps: {n_steps} ({len(pairs)} pairs × {N_EPOCHS} epochs / batch {BATCH_SIZE})")

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 4. Training loop
    log(f"\n--- training {n_steps} steps, β={BETA}, LR={LR} ---")
    rng = np.random.default_rng(SEED)
    losses = []
    t_start = time.time()
    step = 0
    for epoch in range(N_EPOCHS):
        indices = rng.permutation(len(pairs))
        for st in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = [pairs[i] for i in indices[st:st + BATCH_SIZE]]
            actual_bs = len(batch_pairs)

            # Compute current logprobs (per pair, 2 forward passes each)
            # IMPORTANT: keep curr_lp as torch.Tensor (with gradient),
            # ref_lp can be detached (precomputed by Phase 6.A).
            curr_lp_plus_list = []
            curr_lp_minus_list = []
            ref_lp_plus_list = []
            ref_lp_minus_list = []
            for p in batch_pairs:
                z_t = torch.from_numpy(np.asarray(p["z_target"], dtype=np.float32)).unsqueeze(0).to("cuda")
                lp_plus = compute_logprob(model, z_t, p["attrs"], p["y_plus"])
                lp_minus = compute_logprob(model, z_t, p["attrs"], p["y_minus"])
                curr_lp_plus_list.append(lp_plus)  # torch.Tensor with grad
                curr_lp_minus_list.append(lp_minus)  # torch.Tensor with grad
                ref_lp_plus_list.append(p["ref_logprob_y_plus"])
                ref_lp_minus_list.append(p["ref_logprob_y_minus"])

            # Stack into batch tensors (curr_lp keep gradient; ref_lp detached constants)
            curr_lp_plus = torch.stack(curr_lp_plus_list)   # (B,) requires_grad=True via chain
            curr_lp_minus = torch.stack(curr_lp_minus_list)  # (B,) requires_grad=True via chain
            ref_lp_plus = torch.tensor(ref_lp_plus_list, dtype=torch.float32, device="cuda")
            ref_lp_minus = torch.tensor(ref_lp_minus_list, dtype=torch.float32, device="cuda")

            loss, logratios = dpo_loss(curr_lp_plus, curr_lp_minus,
                                       ref_lp_plus, ref_lp_minus, beta=BETA)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            losses.append(float(loss.detach()))
            step += 1

            if step % LOG_EVERY == 0:
                recent = losses[-LOG_EVERY:]
                elapsed = time.time() - t_start
                log(f"  step {step}/{n_steps} | loss={np.mean(recent):+.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"logratio={logratios.mean().item():+.3f} | "
                    f"{elapsed:.0f}s ({elapsed/step:.2f}s/step)")

            if step % CKPT_EVERY == 0:
                ckpt_path = CKPT_DIR / f"dpo_step{step}.pt"
                model.save_projector(str(ckpt_path))
                log(f"  ckpt saved → {ckpt_path}")

    # Final ckpt
    final_ckpt = CKPT_DIR / "dpo_final.pt"
    model.save_projector(str(final_ckpt))
    log(f"\nfinal ckpt saved → {final_ckpt}")

    # Save loss curve
    log_path = LOG_DIR / "phase6b2_2_dpo_losses.json"
    with open(log_path, "w") as f:
        json.dump({
            "losses": losses,
            "config": {
                "N_EPOCHS": N_EPOCHS, "BATCH_SIZE": BATCH_SIZE, "LR": LR,
                "BETA": BETA, "WARMUP_STEPS": WARMUP_STEPS,
                "MAX_PAIRS": MAX_PAIRS,
            },
            "final_loss_mean_last_50": float(np.mean(losses[-50:])),
            "first_loss_mean_first_50": float(np.mean(losses[:50])),
        }, f, indent=2)
    log(f"loss curve → {log_path}")

    log(f"\n=== DPO training done in {time.time() - t_start:.0f}s ===")
    log(f"  final loss (last 50): {np.mean(losses[-50:]):+.4f}")
    log(f"  first loss (first 50): {np.mean(losses[:50]):+.4f}")


if __name__ == "__main__":
    main()
