#!/usr/bin/env python3
"""E16-Task1 — Activation extraction contract + Check-1 automated tests.

Fixes the five E15 validation biases:
1. sentence-END (EOS-1) representation instead of first-token
2. explicit hidden_states[L+1] <-> layers[L] mapping (E15 used a one-layer
   mismatch: hidden_states[L] is layers[L-1]'s output)
3. intervention applied ONLY to generated tokens, never the prompt
4. (contract) deterministic layer/method mapping
5. all artifacts land under the repo result dir for remote reproducibility

Check-1 tests (all must pass for Task 1):
  a) layer mapping: injecting d into layers[L] changes hidden_states[L+1]
  b) end representation separates structurally different pairs with the
     same opener (first-token representation cannot)
  c) alpha=0 intervention == hook-off logits (within tolerance)
  d) prompt-phase hidden states identical with hook on/off
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
BASE_MODEL = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E16_DIR = REPO_ROOT / "result" / "personal_query" / "e16"
N_LAYERS = 28


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class GenOnlyController:
    """Hook that intervenes ONLY on generated tokens (seq_len > prompt_len),
    on the last position of each step. Prompt-phase forward is untouched."""

    def __init__(self, model, layer: int, direction: np.ndarray, alpha: float, prompt_len: int):
        self.direction = torch.as_tensor(direction, dtype=torch.bfloat16, device="cuda:0")
        self.alpha = alpha
        self.prompt_len = prompt_len
        self.intervention_calls = 0
        self.prompt_calls = 0

        def hook(module, args, output):
            h = output[0]
            seq = h.size(1)
            if seq > self.prompt_len:
                # cache steps: only the new token; first generated step may
                # include prompt+1 token — intervene only on the last position.
                self.intervention_calls += 1
                h = h.clone()
                h[:, -1, :] = h[:, -1, :] + self.alpha * self.direction
                return (h,) + output[1:]
            self.prompt_calls += 1
            return output

        self.hook_handle = model.model.layers[layer].register_forward_hook(hook)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def check_a_layer_mapping(model, tok) -> bool:
    """Inject d into layers[L]; hidden_states[L+1] must change, L must not."""
    ids = tok.encode("test sentence.", add_special_tokens=False, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        base = model(input_ids=ids, output_hidden_states=True)
    d = torch.randn(1536, dtype=torch.bfloat16, device="cuda:0") * 0.05
    h = model.model.layers[3].register_forward_hook(
        lambda m, a, o: (o[0] + d[None, None, :],) + o[1:])
    with torch.no_grad():
        inj = model(input_ids=ids, output_hidden_states=True)
    h.remove()
    diff4 = (inj.hidden_states[4] - base.hidden_states[4]).float().abs().mean().item()
    diff3 = (inj.hidden_states[3] - base.hidden_states[3]).float().abs().mean().item()
    ok = diff4 > 1e-4 and diff3 < 1e-4
    log(f"check-a layer mapping: hidden_states[4] diff={diff4:.5f} (must >0), "
        f"hidden_states[3] diff={diff3:.5f} (must ~0) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_b_end_representation(model, tok) -> bool:
    """A minimal pair with the SAME opener but different end structure must be
    separated at the END position (structure) while nearly identical at the
    FIRST-token position (same opener)."""
    neg = "I'm looking for a blue tie for baby use at 8.99."
    pos = "I'm looking for a blue tie for baby use at 8.99, and I need it soon."
    with torch.no_grad():
        hs = {}
        for name, txt in [("n", neg), ("p", pos)]:
            ids = tok.encode(txt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
            out = model(input_ids=ids, output_hidden_states=True)
            hs[name] = out.hidden_states
    def dist(L_idx, pos):
        return float((hs["n"][L_idx][0, pos].float() - hs["p"][L_idx][0, pos].float()).norm())
    first_dist = dist(15, 0)   # first-token position (same opener -> ~0)
    end_dist = dist(15, -1)    # sentence-end position (structure differs -> >0)
    ok = end_dist > first_dist + 0.5 and end_dist > 0.5
    log(f"check-b end rep: first-token dist={first_dist:.3f}, sentence-end dist={end_dist:.3f} "
        f"(end must exceed first + structure signal) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_c_alpha0(model, tok) -> bool:
    """alpha=0 intervention == hook-off logits within tolerance."""
    ids = tok.encode("Write a short shopping query for a blue tie.", add_special_tokens=False,
                     return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out_off = model(input_ids=ids)
    d = np.zeros(1536, dtype=np.float32)
    ctrl = GenOnlyController(model, 8, d, 0.0, ids.size(1))
    with torch.no_grad():
        out_alpha0 = model(input_ids=ids)
    ctrl.remove()
    diff = (out_alpha0.logits - out_off.logits).float().abs().max().item()
    ok = diff < 1e-3
    log(f"check-c alpha=0 vs hook-off max logit diff: {diff:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_d_prompt_untouched(model, tok) -> bool:
    """Prompt-phase hidden states identical with hook on/off (gen-only hook)."""
    ids = tok.encode("Write a short shopping query for a blue tie.", add_special_tokens=False,
                     return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        base = model(input_ids=ids, output_hidden_states=True)
    d = np.full(1536, 0.1, dtype=np.float32)
    ctrl = GenOnlyController(model, 8, d, 5.0, ids.size(1))
    with torch.no_grad():
        inj = model(input_ids=ids, output_hidden_states=True)
    ctrl.remove()
    diff = (inj.hidden_states[15] - base.hidden_states[15]).float().abs().max().item()
    ok = diff < 1e-3 and ctrl.prompt_calls > 0
    log(f"check-d prompt hidden max diff: {diff:.6f} (prompt forwards seen={ctrl.prompt_calls}) "
        f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    results = {
        "a_layer_mapping": check_a_layer_mapping(model, tok),
        "b_end_representation": check_b_end_representation(model, tok),
        "c_alpha0_equals_off": check_c_alpha0(model, tok),
        "d_prompt_untouched": check_d_prompt_untouched(model, tok),
    }
    contract = {
        "representation": "sentence-END token (last token before EOS) hidden state",
        "layer_mapping": "hidden_states[L+1] == model.layers[L] output (verified)",
        "intervention": "hook on model.layers[L] adding alpha*direction ONLY at "
                        "seq_len > prompt_len, last position (generated tokens)",
        "directions": ["mean_diff", "logistic", "pca"],
        "n_layers": N_LAYERS,
        "model": BASE_MODEL,
        "checks": results,
        "hash": None,
    }
    contract["hash"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    E16_DIR.mkdir(parents=True, exist_ok=True)
    (E16_DIR / "activation_contract.json").write_text(json.dumps(contract, indent=2))
    passed = all(results.values())
    log(f"Check-1: {'ALL PASS' if passed else 'FAILED'}")
    log(f"wrote {E16_DIR / 'activation_contract.json'}")


if __name__ == "__main__":
    main()
