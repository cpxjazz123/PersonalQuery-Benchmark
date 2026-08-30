#!/usr/bin/env python3
"""Phase 6.B.2-debug-2 — 8-pair fixed-batch DPO overfit.

用户指令 2026-08-30: 在 single-pair overfit (PASS) 之后,先 sanity check batching:
- 固定 8 个 preference pairs
- 重复训练 50 step (每次都用这 8 pair)
- 观察:
    mean(r) = mean(Δ_θ - Δ_ref)  → 应该 0 → 正值 → 持续 ↑
    P(r > 0)                     → 应该 50% → 70% → 80%+
- 关键: **eval-mode deterministic forward**(否则 r 会被 dropout noise 污染)

如果这关也通过 → Step 2 (重建 length/format-matched preference) + Step 3 (小 LR DPO)
如果失败 → 问题在 batch implementation / masking / optimizer.

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

# === 8-pair fixed-batch overfit config (locked) ===
PAIR_INDICES = list(range(8))  # first 8 pairs
N_STEPS = 50
LR = 2e-5
BETA = 0.1
GRAD_CLIP = 1.0
SEED = 42
LOG_EVERY = 1

PCA_DIM = 48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [8pair] {msg}", flush=True)


def compute_logprob(model, z: torch.Tensor, attrs: list[str], target_text: str,
                    eval_mode: bool = True) -> torch.Tensor:
    """Returns torch.Tensor (scalar) with gradient. B=1.
    eval_mode=True: model.eval() during forward (deterministic, no dropout).
    """
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
        # restore original training flag
        if orig_training:
            model.train()


def main():
    log(f"=== Phase 6.B.2-debug-2 — 8-pair fixed-batch overfit ===")

    # 1. Load 8 fixed pairs
    pref_path = PREF_DIR / "preference_dataset.jsonl"
    log(f"loading preference dataset from {pref_path}")
    pairs = []
    with open(pref_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    fixed_pairs = [pairs[i] for i in PAIR_INDICES]
    log(f"  selected {len(fixed_pairs)} pairs (indices {PAIR_INDICES})")

    # Show the 8 pairs
    for i, p in enumerate(fixed_pairs):
        d_plus, d_minus, dd = p["d_y_plus"], p["d_y_minus"], p["delta_d"]
        ref_dp, ref_dm = p["ref_logprob_y_plus"], p["ref_logprob_y_minus"]
        ref_delta = ref_dp - ref_dm
        log(f"  pair[{i}]: d_y+={d_plus:.2f} d_y-={d_minus:.2f} Δd={dd:.2f}  "
            f"ref Δ={ref_delta:+.3f}  |y+|={len(p['y_plus'])} |y-|={len(p['y_minus'])}")

    # 2. Load model
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from syntax_subspace_pca48_projector import Qwen2WithPrefix
    log("loading Qwen2-7B + LoRA + projector (Phase 6.A init) ...")
    model = Qwen2WithPrefix()
    log(f"  loading init ckpt from {INIT_CKPT}")
    model.load_projector(str(INIT_CKPT))

    # 3. Eval-mode initial check (deterministic, no dropout)
    log("\n[eval-mode init check]")
    model.eval()
    init_deltas = []
    init_ref_deltas = []
    with torch.no_grad():
        for i, p in enumerate(fixed_pairs):
            z_t = torch.from_numpy(np.asarray(p["z_target"], dtype=np.float32)).unsqueeze(0).to("cuda")
            lp_plus = compute_logprob(model, z_t, p["attrs"], p["y_plus"], eval_mode=True)
            lp_minus = compute_logprob(model, z_t, p["attrs"], p["y_minus"], eval_mode=True)
            delta_theta = float(lp_plus.item() - lp_minus.item())
            ref_delta = p["ref_logprob_y_plus"] - p["ref_logprob_y_minus"]
            init_deltas.append(delta_theta)
            init_ref_deltas.append(ref_delta)
    log(f"  init eval Δ_θ per pair: {[f'{d:+.2f}' for d in init_deltas]}")
    log(f"  ref Δ per pair:         {[f'{d:+.2f}' for d in init_ref_deltas]}")
    log(f"  init eval Δ_θ mean: {np.mean(init_deltas):+.3f}  (should ≈ ref Δ mean {np.mean(init_ref_deltas):+.3f})")

    # 4. Train (use train() for gradient, but forward uses eval-mode per compute_logprob flag)
    log(f"\n[train] {N_STEPS} steps, β={BETA}, LR={LR}, fixed 8-pair batch")
    model.train()
    trainable_params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    log(f"  {n_trainable:,} trainable params")

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.0)

    log("  expectation:")
    log("    mean(r) = mean(Δ_θ - Δ_ref)  →  0 → positive → increasing")
    log("    P(r > 0)                       →  50% → 70% → 80%+")

    # Pre-extract z tensors
    z_targets = [torch.from_numpy(np.asarray(p["z_target"], dtype=np.float32)).unsqueeze(0).to("cuda")
                 for p in fixed_pairs]
    ref_deltas_t = torch.tensor(
        [p["ref_logprob_y_plus"] - p["ref_logprob_y_minus"] for p in fixed_pairs],
        dtype=torch.float32, device="cuda"
    )

    history = {
        "mean_r": [],
        "median_r": [],
        "P_r_pos": [],
        "loss": [],
        "grad_norm": [],
        "per_pair_r_per_step": [],  # (step, pair_idx) -> r
    }
    t0 = time.time()

    for step in range(N_STEPS):
        # Forward: 8 pairs × 2 forward passes each (y+ and y-) in eval-mode deterministic
        curr_lp_plus_list = []
        curr_lp_minus_list = []
        for i, p in enumerate(fixed_pairs):
            lp_plus = compute_logprob(model, z_targets[i], p["attrs"], p["y_plus"], eval_mode=True)
            lp_minus = compute_logprob(model, z_targets[i], p["attrs"], p["y_minus"], eval_mode=True)
            curr_lp_plus_list.append(lp_plus)
            curr_lp_minus_list.append(lp_minus)

        curr_lp_plus = torch.stack(curr_lp_plus_list)
        curr_lp_minus = torch.stack(curr_lp_minus_list)
        delta_theta = curr_lp_plus - curr_lp_minus  # (8,)
        r = delta_theta - ref_deltas_t               # (8,) — what DPO optimizes

        loss = -F.logsigmoid(BETA * r).mean()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP).item())
        optimizer.step()

        # Log
        r_np = r.detach().cpu().numpy()
        mean_r = float(np.mean(r_np))
        median_r = float(np.median(r_np))
        p_pos = float(np.mean(r_np > 0))
        loss_v = float(loss.detach())
        history["mean_r"].append(mean_r)
        history["median_r"].append(median_r)
        history["P_r_pos"].append(p_pos)
        history["loss"].append(loss_v)
        history["grad_norm"].append(grad_norm)
        history["per_pair_r_per_step"].append(r_np.tolist())

        if (step + 1) % LOG_EVERY == 0 or step < 5:
            log(f"  step {step+1:3d}: loss={loss_v:+.4f}  mean(r)={mean_r:+.3f}  "
                f"median(r)={median_r:+.3f}  P(r>0)={p_pos*100:.1f}%  grad_norm={grad_norm:.3f}")

    log(f"\ntraining done in {time.time() - t0:.0f}s")

    # 5. Final eval-mode check
    log("\n[final eval-mode Δ check]")
    model.eval()
    final_deltas = []
    with torch.no_grad():
        for i, p in enumerate(fixed_pairs):
            z_t = torch.from_numpy(np.asarray(p["z_target"], dtype=np.float32)).unsqueeze(0).to("cuda")
            lp_plus = compute_logprob(model, z_t, p["attrs"], p["y_plus"], eval_mode=True)
            lp_minus = compute_logprob(model, z_t, p["attrs"], p["y_minus"], eval_mode=True)
            delta_theta = float(lp_plus.item() - lp_minus.item())
            final_deltas.append(delta_theta)
    final_r_per_pair = [final_deltas[i] - init_ref_deltas[i] for i in range(8)]
    log(f"  final eval Δ_θ per pair: {[f'{d:+.2f}' for d in final_deltas]}")
    log(f"  final r per pair:        {[f'{d:+.2f}' for d in final_r_per_pair]}")
    log(f"  final eval Δ_θ mean: {np.mean(final_deltas):+.3f}  "
        f"vs init {np.mean(init_deltas):+.3f}  change={np.mean(final_deltas) - np.mean(init_deltas):+.3f}")
    log(f"  final mean(r) = {np.mean(final_r_per_pair):+.3f}")
    log(f"  final P(r>0) = {np.mean([1 if r > 0 else 0 for r in final_r_per_pair]) * 100:.1f}%")

    # 6. Decision
    final_mean_r = float(np.mean(final_r_per_pair))
    final_p_pos = float(np.mean([1 if r > 0 else 0 for r in final_r_per_pair]))
    initial_mean_r = float(np.mean(np.array(init_deltas) - np.array(init_ref_deltas)))
    trajectory_positive = (
        history["mean_r"][0] < history["mean_r"][N_STEPS // 2]
        and history["mean_r"][N_STEPS // 2] < history["mean_r"][-1]
    )
    p_pos_increasing = (
        history["P_r_pos"][0] <= history["P_r_pos"][N_STEPS // 2]
        and history["P_r_pos"][N_STEPS // 2] <= history["P_r_pos"][-1]
    )

    if final_mean_r > 10 and final_p_pos >= 0.875:
        verdict = "BATCH_OK"
        rationale = (f"After {N_STEPS} steps: mean(r)={final_mean_r:+.3f}>10, "
                     f"P(r>0)={final_p_pos*100:.1f}%>=87.5%. 8-pair batch learns preference. "
                     f"Batching / masking / optimizer all correct. "
                     f"Original 1800-pair failure was preference dataset issue, not impl bug.")
    elif final_mean_r > 0 and final_p_pos >= 0.75:
        verdict = "BATCH_WEAK_OK"
        rationale = (f"After {N_STEPS} steps: mean(r)={final_mean_r:+.3f}>0, "
                     f"P(r>0)={final_p_pos*100:.1f}%>=75%. Weak signal. "
                     f"Batching works but needs more steps or larger β.")
    elif final_p_pos > 0.5 and trajectory_positive:
        verdict = "BATCH_TOO_SMALL"
        rationale = (f"After {N_STEPS} steps: P(r>0)={final_p_pos*100:.1f}%, mean(r) increasing. "
                     f"DPO works but needs more steps. Try 100+ steps.")
    else:
        verdict = "BATCH_BUG"
        rationale = (f"After {N_STEPS} steps: mean(r)={final_mean_r:+.3f}, "
                     f"P(r>0)={final_p_pos*100:.1f}%. 8-pair batch does NOT learn. "
                     f"Bug in batch implementation / masking / dropout / reference freezing. "
                     f"Need to investigate.")

    log(f"\n=== VERDICT: {verdict} ===")
    log(f"  {rationale}")

    # Save (using only floats, no tensors)
    out = {
        "n_pairs": len(fixed_pairs),
        "pair_indices": PAIR_INDICES,
        "n_steps": N_STEPS,
        "beta": BETA,
        "lr": LR,
        "init": {
            "eval_delta_per_pair": init_deltas,
            "ref_delta_per_pair": init_ref_deltas,
            "init_mean_r": initial_mean_r,
        },
        "final": {
            "eval_delta_per_pair": final_deltas,
            "final_mean_r": final_mean_r,
            "final_p_pos": final_p_pos,
        },
        "trajectory": {
            "mean_r": history["mean_r"],
            "median_r": history["median_r"],
            "P_r_pos": history["P_r_pos"],
            "loss": history["loss"],
            "grad_norm": history["grad_norm"],
        },
        "verdict": verdict,
        "rationale": rationale,
    }
    out_path = LOG_DIR / "phase6b2_debug_8pair_overfit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()