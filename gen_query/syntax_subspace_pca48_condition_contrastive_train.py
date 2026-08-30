#!/usr/bin/env python3
"""Phase 6.B.4 — Condition-Contrastive Loss training.

用户指令 2026-08-30:
Phase 6.A 完全忽略 z_t (length-controlled ρ=-0.04). DPO 无法放大没有的 signal.
必须用 condition-contrastive loss 强制 model 区分不同 z_t:

  L_CC = -log σ(β · (log π_θ(y+|c, z_correct) - log π_θ(y+|c, z_wrong)))

其中 z_wrong 来自 batch 内另一 pair 的 z_t (cyclic shift), 但 attrs 保持 pair[i]
(只 z 不同, attrs 和 y+ 都相同 — 强制 model 必须用 prefix 信息).

组合 loss:
  L_total = L_DPO + λ · L_CC

L_DPO 教 "y+ 比 y- 概率更高" (preference).
L_CC 教 "在同一个 attrs + y+ 下, 正确 z 比错误 z 给出更高概率" (conditioning).

z_wrong 选 cyclic shift (i+1) % B:保证 (i,j) 配对一一对应, 无自配对, 无随机种子敏感.

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
PREF_DIR = SCRATCH / "pca48_preferences"
LOG_DIR = SCRATCH / "logs"
INIT_CKPT = SCRATCH / "pca48_ckpt" / "projector_final.pt"
CKPT_DIR = SCRATCH / "pca48_cc_ckpt"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# === Condition-Contrastive config (locked) ===
N_EPOCHS = 2
BATCH_SIZE = 4           # pairs per step
LR = 2e-5                # 2× standard 1e-5 → 2 losses combined
BETA = 0.1               # DPO beta
LAMBDA_CC = 1.0          # weight on L_CC (vs L_DPO)
WARMUP_STEPS = 30
GRAD_CLIP = 1.0
LOG_EVERY = 10
CKPT_EVERY = 100
SEED = 42
MAX_PAIRS = 1500

PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [cc_train] {msg}", flush=True)


def compute_logprob(model, z: torch.Tensor, attrs: list[str], target_text: str,
                    eval_mode: bool = True) -> torch.Tensor:
    """B=1 forward. eval_mode=True (no dropout) for deterministic r."""
    orig_training = model.training
    if eval_mode:
        model.eval()
    try:
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
        logits = out.logits.float()

        prefix_len = model.prefix_len
        prompt_len = prompt_ids.shape[1]
        target_len = target_ids.shape[1]

        pred_logits = logits[0, prefix_len + prompt_len - 1:
                                prefix_len + prompt_len - 1 + target_len, :]
        log_probs = F.log_softmax(pred_logits, dim=-1)
        target_logprobs = log_probs.gather(1, target_ids[0].unsqueeze(-1)).squeeze(-1)
        return target_logprobs.sum()
    finally:
        if orig_training:
            model.train()


def main():
    log("=== Phase 6.B.4 — Condition-Contrastive Loss Training ===")

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

    # 2. Load model
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix
    log("loading Qwen2-7B + LoRA + projector (Phase 6.A init) ...")
    model = Qwen2WithPrefix()
    log(f"  loading init ckpt from {INIT_CKPT}")
    model.load_projector(str(INIT_CKPT))
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
    log(f"\n--- training {n_steps} steps, β={BETA}, λ_CC={LAMBDA_CC}, LR={LR} ---")
    log(f"  per step: B={BATCH_SIZE} pairs × 3 forwards = {BATCH_SIZE * 3} forward passes")
    log(f"  L_DPO: log P(y+|z,attrs) > log P(y-|z,attrs)   (preference)")
    log(f"  L_CC:  log P(y+|attrs,z_correct) > log P(y+|attrs,z_wrong)  (conditioning)")

    rng = np.random.default_rng(SEED)
    losses_dpo = []
    losses_cc = []
    losses_total = []
    logratios_history = []   # DPO log-ratio (Δ_θ - Δ_ref)
    cc_diff_history = []      # CC diff (lp_yplus_zcorrect - lp_yplus_zwrong)

    t_start = time.time()
    step = 0
    for epoch in range(N_EPOCHS):
        indices = rng.permutation(len(pairs))
        for st in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = [pairs[i] for i in indices[st:st + BATCH_SIZE]]
            actual_bs = len(batch_pairs)

            # Build z tensor (B, 48)
            z_targets = torch.stack([
                torch.from_numpy(np.asarray(p["z_target"], dtype=np.float32))
                for p in batch_pairs
            ]).to("cuda")
            # z_wrong: cyclic shift
            z_wrongs = torch.stack([z_targets[(i + 1) % actual_bs] for i in range(actual_bs)])

            # Forward 1: y+ with z_correct (use attrs[i] for both y+ and CC)
            # Forward 2: y- with z_correct (for DPO)
            # Forward 3: y+ with z_wrong (attrs[i], same as y+/z_correct attrs)
            lp_yplus_zc = []
            lp_yminus_zc = []
            lp_yplus_zw = []
            for i, p in enumerate(batch_pairs):
                z_c_i = z_targets[i].unsqueeze(0)
                z_w_i = z_wrongs[i].unsqueeze(0)
                lp_yplus_zc.append(compute_logprob(model, z_c_i, p["attrs"], p["y_plus"]))
                lp_yminus_zc.append(compute_logprob(model, z_c_i, p["attrs"], p["y_minus"]))
                lp_yplus_zw.append(compute_logprob(model, z_w_i, p["attrs"], p["y_plus"]))

            # Stack tensors
            curr_lp_plus_zc = torch.stack(lp_yplus_zc)    # (B,) grad
            curr_lp_minus_zc = torch.stack(lp_yminus_zc)  # (B,) grad
            curr_lp_plus_zw = torch.stack(lp_yplus_zw)    # (B,) grad

            # ref logprobs (cached constants)
            ref_lp_plus = torch.tensor(
                [p["ref_logprob_y_plus"] for p in batch_pairs],
                dtype=torch.float32, device="cuda"
            )
            ref_lp_minus = torch.tensor(
                [p["ref_logprob_y_minus"] for p in batch_pairs],
                dtype=torch.float32, device="cuda"
            )

            # DPO loss
            pi_logratios = curr_lp_plus_zc - curr_lp_minus_zc
            ref_logratios = ref_lp_plus - ref_lp_minus
            dpo_logratio = pi_logratios - ref_logratios
            loss_dpo = -F.logsigmoid(BETA * dpo_logratio).mean()

            # CC loss: log P(y+|attrs,z_correct) > log P(y+|attrs,z_wrong)
            cc_diff = curr_lp_plus_zc - curr_lp_plus_zw
            loss_cc = -F.logsigmoid(BETA * cc_diff).mean()

            # Combined
            loss = loss_dpo + LAMBDA_CC * loss_cc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            losses_dpo.append(float(loss_dpo.detach()))
            losses_cc.append(float(loss_cc.detach()))
            losses_total.append(float(loss.detach()))
            logratios_history.append(float(dpo_logratio.mean().detach()))
            cc_diff_history.append(float(cc_diff.mean().detach()))
            step += 1

            if step % LOG_EVERY == 0:
                elapsed = time.time() - t_start
                recent_dpo = losses_dpo[-LOG_EVERY:]
                recent_cc = losses_cc[-LOG_EVERY:]
                recent_total = losses_total[-LOG_EVERY:]
                recent_logratio = logratios_history[-LOG_EVERY:]
                recent_cc_diff = cc_diff_history[-LOG_EVERY:]
                log(f"  step {step}/{n_steps} | "
                    f"loss_dpo={np.mean(recent_dpo):+.4f} "
                    f"loss_cc={np.mean(recent_cc):+.4f} "
                    f"loss_total={np.mean(recent_total):+.4f} | "
                    f"logratio={np.mean(recent_logratio):+.3f} "
                    f"cc_diff={np.mean(recent_cc_diff):+.3f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed:.0f}s ({elapsed/step:.2f}s/step)")

            if step % CKPT_EVERY == 0:
                ckpt_path = CKPT_DIR / f"cc_step{step}.pt"
                model.save_projector(str(ckpt_path))
                log(f"  ckpt saved → {ckpt_path}")

    # Final ckpt
    final_ckpt = CKPT_DIR / "cc_final.pt"
    model.save_projector(str(final_ckpt))
    log(f"\nfinal ckpt saved → {final_ckpt}")

    # Save training log
    log_path = LOG_DIR / "phase6b4_cc_losses.json"
    with open(log_path, "w") as f:
        json.dump({
            "losses_dpo": losses_dpo,
            "losses_cc": losses_cc,
            "losses_total": losses_total,
            "logratios": logratios_history,
            "cc_diffs": cc_diff_history,
            "config": {
                "N_EPOCHS": N_EPOCHS, "BATCH_SIZE": BATCH_SIZE, "LR": LR,
                "BETA": BETA, "LAMBDA_CC": LAMBDA_CC,
                "WARMUP_STEPS": WARMUP_STEPS, "MAX_PAIRS": MAX_PAIRS,
            },
            "final_means_last_50": {
                "loss_dpo": float(np.mean(losses_dpo[-50:])),
                "loss_cc": float(np.mean(losses_cc[-50:])),
                "loss_total": float(np.mean(losses_total[-50:])),
                "logratio": float(np.mean(logratios_history[-50:])),
                "cc_diff": float(np.mean(cc_diff_history[-50:])),
            },
            "first_means_first_50": {
                "loss_dpo": float(np.mean(losses_dpo[:50])),
                "loss_cc": float(np.mean(losses_cc[:50])),
                "loss_total": float(np.mean(losses_total[:50])),
                "logratio": float(np.mean(logratios_history[:50])),
                "cc_diff": float(np.mean(cc_diff_history[:50])),
            },
        }, f, indent=2)
    log(f"loss curve → {log_path}")

    log(f"\n=== Training done in {time.time() - t_start:.0f}s ===")
    log(f"  final loss_dpo:  {np.mean(losses_dpo[-50:]):+.4f} (init {np.mean(losses_dpo[:50]):+.4f})")
    log(f"  final loss_cc:   {np.mean(losses_cc[-50:]):+.4f} (init {np.mean(losses_cc[:50]):+.4f})")
    log(f"  final loss_total:{np.mean(losses_total[-50:]):+.4f}")
    log(f"  final logratio:  {np.mean(logratios_history[-50:]):+.3f} (DPO Δ_θ-Δ_ref)")
    log(f"  final cc_diff:   {np.mean(cc_diff_history[-50:]):+.3f} "
        f"(CC lp(y+|z_correct)-lp(y+|z_wrong), POSITIVE = z_correct > z_wrong)")


if __name__ == "__main__":
    main()
