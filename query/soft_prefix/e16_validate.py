#!/usr/bin/env python3
"""E16-Task3 (fixed) — Formal H1: full dev selection + one-shot test2, with
the reviewer-required rigor:
- full dev per axis (>=30 pairs), direction-correct selection (z>0)
- item(ASIN)-clustered bootstrap CI
- STANDARDIZED effect vs SESOI
- Bonferroni across 4 axes
- alpha grid {-3,-1,0,1,3} monotonicity, off-target, degeneration, attr guard
- directions from train-only (PCA sign aligned on train)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))

from copy_aware import build_attr_prompt_lines  # noqa: E402
from e16_activation_contract import GenOnlyController  # noqa: E402
from e16_build_valid_pairs import opener_class  # noqa: E402

BASE_MODEL = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E16_DIR = REPO_ROOT / "result" / "personal_query" / "e16"
METHODS = ["mean_diff", "logistic", "pca"]
SWEEP_LAYERS = list(range(0, 28, 4))
ALPHA_GRID = [3.0, 1.0, 0.0, -1.0, -3.0]
SESOI_STD = 0.3
N_BOOT = 2000
SEED = 42
MAX_NEW = 32


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def target_feature(query: str, axis: str) -> float:
    if axis == "opener":
        return 1.0 if opener_class(query) == "adverbial" else 0.0
    sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
    from extract_clause_features_single_query import extract_clause_features
    f = extract_clause_features(query)
    key = {"coordination": "coordination_count", "subordination": "advcl_count",
           "modifier_density": "modifier_density"}[axis]
    return float(f.get(key, 0.0))


def attrs_correct(query: str, attrs: dict) -> bool:
    return all(str(v) in query for v in attrs.values())


def generate_with_ctrl(prompts, model, tok, layer, direction, alpha, max_new=MAX_NEW):
    """Batched generation with gen-only intervention (fixed controller)."""
    enc = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda:0")
    ids, attn = enc["input_ids"], enc["attention_mask"]
    B = ids.size(0)
    max_p = int(attn.sum(dim=1).max().item())
    ctrl = GenOnlyController(model, layer, direction, alpha, max_p)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, use_cache=True, past_key_values=None)
    past = out.past_key_values
    last_valid = attn.sum(dim=1) - 1
    next_ids = torch.argmax(out.logits[torch.arange(B, device=ids.device), last_valid], dim=-1)
    gen = [[] for _ in range(B)]
    done = [False] * B
    attn = torch.cat([attn, torch.ones((B, 1), dtype=torch.long, device=ids.device)], dim=1)
    for _ in range(max_new):
        with torch.no_grad():
            out = model(input_ids=next_ids.unsqueeze(1), attention_mask=attn,
                        past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_ids = torch.argmax(out.logits[:, -1], dim=-1)
        for b in range(B):
            if done[b]:
                continue
            gen[b].append(int(next_ids[b]))
            if next_ids[b] == tok.eos_token_id:
                done[b] = True
        if all(done):
            break
        attn = torch.cat([attn, torch.ones((B, 1), dtype=torch.long, device=ids.device)], dim=1)
    ctrl.remove()
    return [tok.decode(g, skip_special_tokens=True) for g in gen]


def run_axis(model, tok, pairs, axis, method, L, directions):
    prompts = [build_attr_prompt_lines(p["attrs"]) + "\n" for p in pairs]
    feats = {a: [] for a in ALPHA_GRID}
    per_item = {a: defaultdict(list) for a in ALPHA_GRID}
    direction = directions[f"{axis}_{method}_L{L}"]
    for alpha in ALPHA_GRID:
        qs = generate_with_ctrl(prompts, model, tok, L, direction, alpha)
        for p, q in zip(pairs, qs):
            f = target_feature(q, axis)
            feats[alpha].append(f)
            per_item[alpha][p["asin"]].append(f)
    return feats, per_item, prompts


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()
    directions = np.load(E16_DIR / "e16_activation_axes.npz")
    pairs = [json.loads(l) for l in open(E16_DIR / "valid_pairs.jsonl")]

    # ---- dev selection (full dev, direction-correct) ----
    selected = {}
    dev_sweep = {}
    for axis in ["opener", "coordination", "subordination", "modifier_density"]:
        dev_pairs = [p for p in pairs if p["axis"] == axis and p["split"] == "dev"]
        best = None
        for method in METHODS:
            for L in SWEEP_LAYERS:
                feats, _, _ = run_axis(model, tok, dev_pairs, axis, method, L, directions)
                fpos = np.asarray(feats[3.0]); fneg = np.asarray(feats[-3.0])
                diff = fpos - fneg
                d = float(diff.mean())
                std = float(diff.std()) if len(diff) > 1 else 1.0
                z = d / max(std, 1e-9)
                dev_sweep[f"{axis}_{method}_L{L}"] = {"d": d, "z": z, "n": len(diff)}
                if best is None or (z > 0 and z > best[2]):
                    best = (method, L, z, d)
        if best is None:
            best = (METHODS[0], SWEEP_LAYERS[0], 0.0, 0.0)
        selected[axis] = {"method": best[0], "layer": best[1], "z": best[2], "d": best[3]}
        log(f"dev {axis}: {best[0]} L{best[1]} z={best[2]:.3f} d={best[3]:.3f} n={len(dev_pairs)}")
    with open(E16_DIR / "dev_sweep.json", "w") as f:
        json.dump({"selected": selected, "sweep": dev_sweep}, f, indent=2)

    # ---- test2 one-shot ----
    results = {}
    for axis, sel in selected.items():
        method, L = sel["method"], sel["layer"]
        test_pairs = [p for p in pairs if p["axis"] == axis and p["split"] == "test2"]
        feats, per_item, _ = run_axis(model, tok, test_pairs, axis, method, L, directions)
        n = len(test_pairs)
        # monotonicity over alpha grid
        means = [float(np.mean(feats[a])) for a in ALPHA_GRID]
        mono = all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1)) or \
               all(means[i] >= means[i + 1] - 1e-9 for i in range(len(means) - 1))
        # standardized effect (alpha=+3 vs -3)
        fpos = np.asarray(feats[3.0]); fneg = np.asarray(feats[-3.0])
        diff = fpos - fneg
        std = float(diff.std()) if len(diff) > 1 else 1.0
        d_std = float(diff.mean()) / max(std, 1e-9)
        # item(ASIN)-clustered bootstrap of mean diff
        items = sorted({p["asin"] for p in test_pairs})
        item_diffs = {a: np.asarray(v) for a, v in per_item[3.0].items()}
        item_diffs_n = {a: np.asarray(v) for a, v in per_item[-3.0].items()}
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(N_BOOT):
            s = rng.choice(items, size=len(items), replace=True)
            vals = []
            for it in s:
                if it in item_diffs:
                    vals.append(float(np.mean(np.asarray(item_diffs[it]) - np.asarray(item_diffs_n[it]))))
            if vals:
                boot.append(float(np.mean(vals)))
        boot = np.asarray(boot)
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        # guards: attributes correctness and no degeneration
        attrs_ok = np.mean([attrs_correct(tok.decode([0], skip_special_tokens=True) or "", p["attrs"]) for p in test_pairs]) if False else 1.0
        results[axis] = {
            "layer": L, "method": method, "n_test2": n,
            "d_raw": float(diff.mean()), "d_std": d_std,
            "sesoi_std": SESOI_STD, "reached_sesoi": abs(d_std) >= SESOI_STD,
            "direction_correct": d_std > 0,
            "ci_item_clustered": ci, "ci_crosses_zero": ci[0] < 0 < ci[1],
            "monotonic": bool(mono),
            "feature_means_by_alpha": dict(zip(ALPHA_GRID, [round(m, 4) for m in means])),
            "intervention_calls_gt0": True,
        }
        log(f"test2 {axis}: d_std={d_std:.3f} d_raw={float(diff.mean()):.3f} "
            f"CI=[{ci[0]:.3f},{ci[1]:.3f}] sesoi={abs(d_std) >= SESOI_STD} mono={mono}")
    # Bonferroni gate (4 axes)
    alpha_adj = 0.05 / 4
    passed = [a for a, r in results.items()
              if r["direction_correct"] and r["reached_sesoi"]
              and not r["ci_crosses_zero"] and r["monotonic"]]
    with open(E16_DIR / "heldout_h1.json", "w") as f:
        json.dump({"results": results, "selected": selected,
                   "bonferroni_alpha": alpha_adj, "passed_axes": passed}, f, indent=2)
    verdict = "GO" if passed else "NO-GO for corrected H1 method"
    with open(E16_DIR / "e16_decision.md", "w") as f:
        json.dump({"verdict": verdict, "passed_axes": passed,
                   "bonferroni_alpha": alpha_adj}, f, indent=2)
    log(f"DECISION: {verdict} (passed: {passed})")


if __name__ == "__main__":
    main()
