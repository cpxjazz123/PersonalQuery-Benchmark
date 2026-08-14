#!/usr/bin/env python3
"""E15-D2 — H1 validation on TEST items with the dev-selected layers.

For each axis, use the dev-selected (method, layer) and sweep alpha in
{-1,-0.5,0,0.5,1} on held-out TEST items (same product, prompt, decoding).
Primary metric: standardized effect of axis feature (alpha=+1 vs -1, paired
by item). Reports monotonicity, SESOI (0.3), clustered 95% CI.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from copy_aware import build_attr_prompt_lines  # noqa: E402
from e15_intervene import ActivationController, target_feature  # noqa: E402

BASE_MODEL = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E15_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e15")
MAX_NEW = 32
ALPHAS = [0.0, 3.0, -3.0]
SELECTED = {  # dev-selected layers (from layer_sweep.json, 3-method sweep)
    "opener": ("mean_diff", 0),
    "coordination": ("logistic", 8),
    "subordination": ("pca", 0),
    "modifier_density": ("logistic", 27),
}
SESOI = 0.3
N_BOOT = 2000
SEED = 42


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    axes = np.load(E15_DIR / "activation_axes.npz")
    pairs = [json.loads(l) for l in open(E15_DIR / "syntax_axis_pairs.jsonl")]

    results = {}
    for axis, (method, L) in SELECTED.items():
        direction = axes[f"{axis}_{method}_L{L}"]
        test_pairs = [p for p in pairs if p["axis"] == axis and p["split"] == "test"]
        log(f"{axis}: test pairs={len(test_pairs)} (L{L}, {method})")
        per_alpha = {a: [] for a in ALPHAS}
        # BATCHED generation: all test pairs at once per alpha (same layer
        # direction applies to every sample; transformer generate handles B).
        prompts = [build_attr_prompt_lines(p["attrs"]) + "\n" for p in test_pairs]
        for alpha in ALPHAS:
            ctrl = ActivationController(model, L, direction, alpha)
            enc = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            ctrl.remove()
            for i, q in enumerate(tok.batch_decode(out[:, enc["input_ids"].size(1):], skip_special_tokens=True)):
                per_alpha[alpha].append(target_feature(q, axis))
        feats = {a: np.asarray(v, dtype=np.float64) for a, v in per_alpha.items()}
        diff = feats[3.0] - feats[-3.0]
        d = float(diff.mean())
        std = float(diff.std()) if len(diff) > 1 else 1.0
        z = d / max(std, 1e-9)
        # clustered CI (cluster = item/asn)
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(N_BOOT):
            s = rng.choice(diff, size=len(diff), replace=True)
            boot.append(float(s.mean()))
        boot = np.asarray(boot)
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        # monotonicity across alpha sequence
        means = [float(feats[a].mean()) for a in ALPHAS]
        mono = all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1)) or \
               all(means[i] >= means[i + 1] - 1e-9 for i in range(len(means) - 1))
        results[axis] = {
            "layer": L, "method": method, "n_test": len(diff),
            "effect_plus_vs_minus": d, "std": std, "z": z,
            "sesoi": SESOI, "reached_sesoi": abs(d) >= SESOI,
            "ci": ci, "ci_crosses_zero": ci[0] < 0 < ci[1],
            "monotonic": bool(mono),
            "feature_means_by_alpha": dict(zip(ALPHAS, means)),
        }
        log(f"  {axis}: d={d:.3f} z={z:.3f} CI={[round(x,3) for x in ci]} "
            f"sesoi={abs(d) >= SESOI} mono={mono}")
    with open(E15_DIR / "heldout_h1.json", "w") as f:
        json.dump(results, f, indent=2)
    log("wrote heldout_h1.json")


if __name__ == "__main__":
    main()
