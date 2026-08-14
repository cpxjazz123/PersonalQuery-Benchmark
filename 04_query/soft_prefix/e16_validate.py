#!/usr/bin/env python3
"""E16-Task3 — Formal H1 confirmation: dev layer/method selection then a
SINGLE test2 evaluation, with pre-registered gates.

Uses the corrected contract:
- sentence-END directions (e16_activation_axes.npz, train-only)
- GenOnlyController (intervention only on generated tokens)
- dev (191 pairs) selects layer/method per axis; test2 (234 pairs) evaluated
  exactly once; item-clustered bootstrap CI; SESOI=0.3; Bonferroni (4 axes).
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
from e16_activation_contract import GenOnlyController  # noqa: E402
from e16_build_valid_pairs import opener_class  # noqa: E402

BASE_MODEL = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E16_DIR = REPO_ROOT / "result" / "personal_query" / "e16"
N_LAYERS = 28
METHODS = ["mean_diff", "logistic", "pca"]
SWEEP_LAYERS = list(range(0, 28, 4))  # 7 coarse layers for dev selection
ALPHAS = [3.0, -3.0]
SESOI = 0.3
N_BOOT = 2000
SEED = 42
MAX_NEW = 32


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def target_feature(query: str, axis: str) -> float:
    if axis == "opener":
        return 1.0 if opener_class(query) == "adverbial" else 0.0
    sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))
    from extract_clause_features_single_query import extract_clause_features
    f = extract_clause_features(query)
    key = {"coordination": "coordination_count", "subordination": "advcl_count",
           "modifier_density": "modifier_density"}[axis]
    return float(f.get(key, 0.0))


def generate_batch(prompts, model, tok, ctrl, max_new=MAX_NEW):
    enc = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return [tok.decode(o[enc["input_ids"].size(1):], skip_special_tokens=True) for o in out]


def sweep_dev(model, tok, axes, pairs, selected_out) -> dict:
    selected = {}
    dev_sweep = {}
    for axis in ["opener", "coordination", "subordination", "modifier_density"]:
        dev_pairs = [p for p in pairs if p["axis"] == axis and p["split"] == "dev"][:8]
        best = None
        for method in METHODS:
            for L in SWEEP_LAYERS:
                direction = axes[f"{axis}_{method}_L{L}"]
                feats = {a: [] for a in ALPHAS}
                prompts = [build_attr_prompt_lines(p["attrs"]) + "\n" for p in dev_pairs]
                max_p = max(len(tok.encode(x, add_special_tokens=False)) for x in prompts)
                for alpha in ALPHAS:
                    ctrl = GenOnlyController(model, L, direction, alpha, max_p)
                    qs = generate_batch(prompts, model, tok, ctrl)
                    ctrl.remove()
                    feats[alpha].extend(target_feature(q, axis) for q in qs)
                fpos = np.asarray(feats[3.0], dtype=np.float64)
                fneg = np.asarray(feats[-3.0], dtype=np.float64)
                diff = fpos - fneg
                d = float(diff.mean()) if len(diff) else 0.0
                z = d / max(float(diff.std()), 1e-9) if len(diff) > 1 else 0.0
                dev_sweep[f"{axis}_{method}_L{L}"] = {"d": d, "z": z, "n": len(diff)}
                if best is None or abs(z) > abs(best[2]):
                    best = (method, L, z, d)
        selected[axis] = {"method": best[0], "layer": best[1], "z": best[2], "d": best[3]}
        log(f"dev {axis}: selected {best[0]} L{best[1]} z={best[2]:.3f} d={best[3]:.3f} (n={len(dev_pairs)})")
    with open(selected_out, "w") as f:
        json.dump({"selected": selected, "sweep": dev_sweep}, f, indent=2)
    return selected


def run_test2(model, tok, axes, pairs, selected) -> dict:
    results = {}
    for axis, sel in selected.items():
        method, L = sel["method"], sel["layer"]
        direction = axes[f"{axis}_{method}_L{L}"]
        test_pairs = [p for p in pairs if p["axis"] == axis and p["split"] == "test2"]
        log(f"test2 {axis}: {len(test_pairs)} pairs (L{L} {method})")
        prompts = [build_attr_prompt_lines(p["attrs"]) + "\n" for p in test_pairs]
        feats = {a: [] for a in ALPHAS}
        for alpha in ALPHAS:
            ctrl = GenOnlyController(model, L, direction, alpha,
                                     len(tok.encode(prompts[0], add_special_tokens=False)))
            qs = generate_batch(prompts, model, tok, ctrl)
            ctrl.remove()
            for q in qs:
                feats[alpha].append(target_feature(q, axis))
        diff = np.asarray(feats[3.0], dtype=np.float64) - np.asarray(feats[-3.0], dtype=np.float64)
        d = float(diff.mean())
        std = float(diff.std()) if len(diff) > 1 else 1.0
        z = d / max(std, 1e-9)
        rng = np.random.default_rng(SEED)
        boot = [float(rng.choice(diff, size=len(diff), replace=True).mean()) for _ in range(N_BOOT)]
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        results[axis] = {
            "layer": L, "method": method, "n_test2": len(diff),
            "effect": d, "std": std, "z": z,
            "ci": ci, "ci_crosses_zero": ci[0] < 0 < ci[1],
            "reached_sesoi": abs(d) >= SESOI,
            "direction_correct": d > 0,
        }
        log(f"  {axis}: d={d:.3f} z={z:.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}] sesoi={abs(d) >= SESOI}")
    return results


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    axes = np.load(E16_DIR / "e16_activation_axes.npz")
    pairs = [json.loads(l) for l in open(E16_DIR / "valid_pairs.jsonl")]
    log(f"pairs: {len(pairs)}")

    selected = sweep_dev(model, tok, axes, pairs, E16_DIR / "dev_sweep.json")
    results = run_test2(model, tok, axes, pairs, selected)
    with open(E16_DIR / "heldout_h1.json", "w") as f:
        json.dump({"results": results, "selected": selected}, f, indent=2)
    log("wrote heldout_h1.json")

    # gated decision
    passed = [a for a, r in results.items()
              if r["direction_correct"] and r["reached_sesoi"] and not r["ci_crosses_zero"]]
    decision = {
        "verdict": "GO" if passed else "NO-GO for corrected H1 method",
        "passed_axes": passed,
        "sesoi": SESOI,
        "note": "GO requires >=1 pre-registered axis with correct direction, SESOI, "
                "Bonferroni-adjusted item-clustered CI not crossing zero",
    }
    with open(E16_DIR / "e16_decision.md", "w") as f:
        f.write(json.dumps(decision, indent=2))
    log(f"DECISION: {decision['verdict']} (passed: {passed})")


if __name__ == "__main__":
    main()
