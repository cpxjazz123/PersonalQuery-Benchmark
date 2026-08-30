#!/usr/bin/env python3
"""Phase 6.B.2-debug — Single-pair DPO overfit test.

用户指令 2026-08-30:
- ref_logprob(y+) ≈ ref_logprob(y-) 是 NORMAL DPO baseline (log 2),不是梯度弱
- 真正异常是 logratio 越训越负 → 必须查 DPO 实现是否反向
- 第一步: 单 pair overfit,确认 Δ_θ = log π_θ(y+) - log π_θ(y-) INCREASES
- 失败模式:
  - chosen/rejected 顺序反
  - loss 正负号反
  - logged logratio 定义反
  - reference 与 policy 共享 trainable LoRA
  - completion mask 错
  - ref logprob 没 freeze

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

LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Single-pair overfit config (locked) ===
PAIR_IDX = 0
N_STEPS = 100
LR = 2e-5
BETA = 0.1
GRAD_CLIP = 1.0
SEED = 42
LOG_EVERY = 1

PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [overfit] {msg}", flush=True)


def compute_logprob(model, z: torch.Tensor, attrs: list[str], target_text: str) -> torch.Tensor:
    """Same as in dpo_train.py — returns torch.Tensor with gradient (B=1)."""
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

    # logit at position prefix_len + prompt_len - 1 + t predicts target[t]
    pred_logits = logits[0, prefix_len + prompt_len - 1:
                            prefix_len + prompt_len - 1 + target_len, :]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    target_logprobs = log_probs.gather(1, target_ids[0].unsqueeze(-1)).squeeze(-1)
    return target_logprobs.sum()


def main():
    log(f"=== Phase 6.B.2-debug — Single-pair overfit test (pair[{PAIR_IDX}]) ===")

    # 1. Load pair
    pref_path = PREF_DIR / "preference_dataset.jsonl"
    log(f"loading preference dataset from {pref_path}")
    pairs = []
    with open(pref_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    pair = pairs[PAIR_IDX]
    log(f"  pair[{PAIR_IDX}]:")
    log(f"    d_y+={pair['d_y_plus']:.3f}  d_y-={pair['d_y_minus']:.3f}  delta_d={pair['delta_d']:.3f}")
    log(f"    ref_logprob y+={pair['ref_logprob_y_plus']:.3f}  y-={pair['ref_logprob_y_minus']:.3f}")
    log(f"    ref Δ = ref_logprob(y+) - ref_logprob(y-) = "
        f"{pair['ref_logprob_y_plus'] - pair['ref_logprob_y_minus']:+.3f}")
    log(f"    attrs (first 60 chars): {str(pair['attrs'])[:60]}")
    log(f"    y+ ({len(pair['y_plus'])} chars): {pair['y_plus'][:80]}")
    log(f"    y- ({len(pair['y_minus'])} chars): {pair['y_minus'][:80]}")

    # 2. Load model
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix
    log("loading Qwen2-7B + LoRA + projector (Phase 6.A init) ...")
    model = Qwen2WithPrefix()
    log(f"  loading init ckpt from {INIT_CKPT}")
    model.load_projector(str(INIT_CKPT))

    # 3. Initial sanity: compare eval-mode Δ (no dropout) to ref Δ
    model.eval()
    with torch.no_grad():
        z_t = torch.from_numpy(np.asarray(pair["z_target"], dtype=np.float32)).unsqueeze(0).to("cuda")
        lp_plus_eval0 = compute_logprob(model, z_t, pair["attrs"], pair["y_plus"])
        lp_minus_eval0 = compute_logprob(model, z_t, pair["attrs"], pair["y_minus"])
        delta_eval0 = float(lp_plus_eval0.item() - lp_minus_eval0.item())
        ref_delta = pair["ref_logprob_y_plus"] - pair["ref_logprob_y_minus"]
    log(f"\n[eval-mode init check]")
    log(f"  eval logp(y+)={lp_plus_eval0.item():.3f}  logp(y-)={lp_minus_eval0.item():.3f}")
    log(f"  eval Δ_θ(init) = {delta_eval0:+.3f}")
    log(f"  ref Δ = {ref_delta:+.3f}")
    log(f"  diff (eval - ref) = {delta_eval0 - ref_delta:+.3f} (should be ~0 if ckpt loaded correctly)")

    # 4. Train
    model.train()
    trainable_params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    log(f"\n[train mode] {n_trainable:,} trainable params")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.0)

    z_t = torch.from_numpy(np.asarray(pair["z_target"], dtype=np.float32)).unsqueeze(0).to("cuda")

    log(f"\n[overfit training: {N_STEPS} steps, β={BETA}, LR={LR}, single pair]")
    log(f"  expectation: Δ_θ should INCREASE (toward +∞), loss should DECREASE toward 0")

    delta_history = []
    loss_history = []
    grad_norm_history = []
    t0 = time.time()

    for step in range(N_STEPS):
        # Forward: compute logprob in TRAIN mode (dropout active)
        lp_plus = compute_logprob(model, z_t, pair["attrs"], pair["y_plus"])
        lp_minus = compute_logprob(model, z_t, pair["attrs"], pair["y_minus"])
        delta_theta = lp_plus - lp_minus  # log π_θ(y+) - log π_θ(y-)
        ref_delta = torch.tensor(pair["ref_logprob_y_plus"] - pair["ref_logprob_y_minus"],
                                  dtype=torch.float32, device="cuda")

        logratio = delta_theta - ref_delta
        loss = -F.logsigmoid(BETA * logratio)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP).item())
        optimizer.step()

        delta_history.append(float(delta_theta.detach()))
        loss_history.append(float(loss.detach()))
        grad_norm_history.append(grad_norm)

        if (step + 1) % LOG_EVERY == 0 or step < 5:
            log(f"  step {step+1:3d}: loss={float(loss.detach()):+.4f}  "
                f"Δ_θ={float(delta_theta.detach()):+.3f}  "
                f"ref Δ={ref_delta.item():+.3f}  "
                f"logratio={float(logratio.detach()):+.3f}  "
                f"grad_norm={grad_norm:.3f}")

    log(f"\ntraining done in {time.time() - t0:.0f}s")

    # 5. Final sanity: eval mode Δ after training
    model.eval()
    with torch.no_grad():
        lp_plus_evalF = compute_logprob(model, z_t, pair["attrs"], pair["y_plus"])
        lp_minus_evalF = compute_logprob(model, z_t, pair["attrs"], pair["y_minus"])
        delta_evalF = float(lp_plus_evalF.item() - lp_minus_evalF.item())
    log(f"\n[final eval-mode Δ check]")
    log(f"  eval logp(y+)={lp_plus_evalF.item():.3f}  logp(y-)={lp_minus_evalF.item():.3f}")
    log(f"  eval Δ_θ(final) = {delta_evalF:+.3f}")
    log(f"  delta (final - init) = {delta_evalF - delta_eval0:+.3f}  "
        f"(POSITIVE = model learned to prefer y+ over y-, "
        f"NEGATIVE = DPO reversed or sign error)")
    log(f"  ref Δ = {ref_delta:+.3f}  (DPO expectation: eval Δ_final >> ref Δ)")

    # 6. Decision
    delta_change = delta_evalF - delta_eval0
    final_vs_ref = delta_evalF - ref_delta

    if delta_change > 1.0 and final_vs_ref > 5.0:
        verdict = "DPO_SIGN_OK"
        rationale = (f"Δ_θ increased by {delta_change:+.3f}, final Δ_θ={delta_evalF:+.3f} "
                     f"much larger than ref Δ={ref_delta:+.3f}. DPO signal direction is correct. "
                     f"Original 1800-pair failure was batch-level noise, not sign error.")
    elif delta_change > 0.5:
        verdict = "DPO_SIGN_WEAK_OK"
        rationale = (f"Δ_θ increased by {delta_change:+.3f} but only modestly. "
                     f"DPO works but weak signal. Consider larger β or longer training.")
    elif delta_change > -0.5:
        verdict = "DPO_SIGN_STUCK"
        rationale = (f"Δ_θ essentially unchanged ({delta_change:+.3f}). "
                     f"Loss oscillates near log 2. DPO not learning even on single pair. "
                     f"Possible: gradient too small (β too low), or sign issue masked by dropout.")
    else:
        verdict = "DPO_SIGN_REVERSED"
        rationale = (f"Δ_θ DECREASED by {delta_change:+.3f}. "
                     f"DPO is learning WRONG direction. Sign error / chosen-rejected swap / "
                     f"completion mask / LoRA not loaded from ckpt.")

    log(f"\n=== VERDICT: {verdict} ===")
    log(f"  {rationale}")

    # Save
    out = {
        "pair_idx": PAIR_IDX,
        "n_steps": N_STEPS,
        "beta": BETA,
        "lr": LR,
        "init": {
            "eval_logp_y_plus": float(lp_plus_eval0.item()),
            "eval_logp_y_minus": float(lp_minus_eval0.item()),
            "eval_delta": delta_eval0,
            "ref_delta": ref_delta,
        },
        "final": {
            "eval_logp_y_plus": float(lp_plus_evalF.item()),
            "eval_logp_y_minus": float(lp_minus_evalF.item()),
            "eval_delta": delta_evalF,
        },
        "delta_change": delta_change,
        "final_vs_ref": final_vs_ref,
        "delta_theta_per_step": delta_history,
        "loss_per_step": loss_history,
        "grad_norm_per_step": grad_norm_history,
        "verdict": verdict,
        "rationale": rationale,
    }
    out_path = LOG_DIR / "phase6b2_debug_single_pair_overfit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()