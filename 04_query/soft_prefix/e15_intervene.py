#!/usr/bin/env python3
"""E15-D — Causal activation intervention (H1) + layer selection (C3) + M1/M2.

Frozen Qwen2.5-1.5B with a forward hook that adds alpha*direction to the
hidden states of a chosen layer at every generated token position:
    h' = h + alpha * direction

For each axis we sweep {method x layer} on DEV items (fixed product, prompt,
decoding) and select the best (method, layer) per axis by the target feature
effect. Test items are never used for selection (C3). M1 = hook off
(alpha=0), M2 = single-axis positive/negative intervention.
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
from e15_build_pairs import opener_class  # noqa: E402

BASE_MODEL = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E15_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e15")
N_LAYERS = 28
ALPHAS = [0.0, 3.0, -3.0]
SEED = 42
MAX_NEW = 32
# coarse sweep first: sampled layers + mean_diff only; fine sweep afterwards
SWEEP_LAYERS = [0, 4, 8, 12, 16, 20, 24, 27]
SWEEP_METHODS = ["mean_diff", "logistic", "pca"]
MAX_DEV_PAIRS_PER_AXIS = 10


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class ActivationController:
    def __init__(self, model, layer: int, direction: np.ndarray, alpha: float):
        self.hook_handle = None
        self.direction = torch.as_tensor(direction, dtype=torch.bfloat16, device="cuda:0")
        self.alpha = alpha

        def hook(module, args, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + self.alpha * self.direction[None, None, :]
                return (h,) + output[1:]
            return output + self.alpha * self.direction[None, None, :]

        self.hook_handle = model.model.layers[layer].register_forward_hook(hook)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def target_feature(query: str, axis: str) -> float:
    if axis == "opener":
        return 1.0 if opener_class(query) == "adverbial" else 0.0
    sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))
    from extract_clause_features_single_query import extract_clause_features
    f = extract_clause_features(query)
    key = {"coordination": "coordination_count", "subordination": "advcl_count",
           "modifier_density": "modifier_density"}[axis]
    return float(f.get(key, 0.0))


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    axes = np.load(E15_DIR / "activation_axes.npz")
    pairs = [json.loads(l) for l in open(E15_DIR / "syntax_axis_pairs.jsonl")]
    dev_items = {p["asin"] for p in pairs if p["split"] == "dev"}
    log(f"dev items: {len(dev_items)}")

    axis_names = ["opener", "coordination", "subordination", "modifier_density"]
    methods = ["mean_diff", "logistic", "pca"]
    results = {}

    for axis in axis_names:
        best = None
        sweep = []
        dev_pairs_axis = [p for p in pairs if p["axis"] == axis and p["asin"] in dev_items][:MAX_DEV_PAIRS_PER_AXIS]
        log(f"{axis}: dev pairs={len(dev_pairs_axis)}")
        for method in SWEEP_METHODS:
            for L in SWEEP_LAYERS:
                direction = axes[f"{axis}_{method}_L{L}"]
                eff_pos = []
                eff_neg = []
                for p in dev_pairs_axis:
                    attrs = p["attrs"]
                    prompt = build_attr_prompt_lines(attrs) + "\n"
                    # generate with alpha=+1 and alpha=-1 (M2), same prompt
                    gen = {}
                    for alpha in (3.0, -3.0):
                        ctrl = ActivationController(model, L, direction, alpha)
                        ids = tok.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
                        with torch.no_grad():
                            out = model.generate(input_ids=ids, max_new_tokens=MAX_NEW, do_sample=False,
                                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                        ctrl.remove()
                        q = tok.decode(out[0][ids.size(1):], skip_special_tokens=True)
                        gen[alpha] = q
                    eff_pos.append(target_feature(gen[3.0], axis))
                    eff_neg.append(target_feature(gen[-3.0], axis))
                eff_pos = np.asarray(eff_pos, dtype=np.float64)
                eff_neg = np.asarray(eff_neg, dtype=np.float64)
                diff = eff_pos - eff_neg
                d = float(diff.mean()) if len(diff) else 0.0
                std = float(diff.std()) if len(diff) > 1 else 1.0
                z = d / max(std, 1e-9)
                sweep.append({"axis": axis, "method": method, "layer": L,
                              "mean_diff": d, "std": std, "z": z, "n": len(diff)})
                if best is None or z > best["z"]:
                    best = sweep[-1]
        results[axis] = {"sweep": sweep, "selected": best}
        log(f"{axis}: selected {best['method']} L{best['layer']} z={best['z']:.3f} (n={best['n']})")

    with open(E15_DIR / "layer_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    log("wrote layer_sweep.json")


if __name__ == "__main__":
    main()
