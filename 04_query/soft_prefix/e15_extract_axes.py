#!/usr/bin/env python3
"""E15-C1/C2 — Extract per-layer hidden activations and syntax-axis directions.

For each content-identical minimal pair (pos/neg), run a frozen forward pass
over [attribute prompt + pair sentence] and record the hidden state at the
pair sentence's FIRST TOKEN position (the position that predicts the first
generated token) for all 28 layers. Directions (mean-difference / logistic /
PCA) are fit on TRAIN pairs only; dev/test are never used for fitting or sign
selection (checked).

Outputs:
  e15/activation_manifest.json
  e15/activations.npz        (act_pos, act_neg, pair meta: axis, split)
  e15/activation_axes.npz    (directions per axis per method, layer-wise)
  e15/axis_fit_metrics.json
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
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))

from copy_aware import build_attr_prompt_lines  # noqa: E402

BASE_MODEL = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E15_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/e15")
PAIRS = E15_DIR / "syntax_axis_pairs.jsonl"
SEED = 42
N_LAYERS = 28


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    pairs = [json.loads(l) for l in open(PAIRS)]
    log(f"pairs: {len(pairs)}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()

    attrs_map = {}
    for p in pairs:
        attrs_map[(p["user_id"], p["asin"])] = p["attrs"]
    # prompt template: attribute lines + "Write a natural shopping query."
    def build_input_ids(sentence: str, attrs: dict):
        prompt = build_attr_prompt_lines(attrs) + "\n" + sentence
        return tok.encode(prompt, add_special_tokens=False)

    meta = []
    act_pos = np.zeros((len(pairs), N_LAYERS, 1536), dtype=np.float32)
    act_neg = np.zeros((len(pairs), N_LAYERS, 1536), dtype=np.float32)
    layer_map = {}  # layer index -> model layer name offset (all 28 layers)
    with torch.no_grad():
        for i, p in enumerate(pairs):
            attrs = attrs_map[(p["user_id"], p["asin"])]
            ids_p = build_input_ids(p["positive"], attrs)
            ids_n = build_input_ids(p["negative"], attrs)
            seq_p = torch.tensor([ids_p], device="cuda:0")
            seq_n = torch.tensor([ids_n], device="cuda:0")
            out_p = model(input_ids=seq_p, output_hidden_states=True, use_cache=False)
            out_n = model(input_ids=seq_n, output_hidden_states=True, use_cache=False)
            # first-token position of the sentence: after the attribute prompt
            prompt_len = len(tok.encode(build_attr_prompt_lines(attrs) + "\n", add_special_tokens=False))
            for L in range(N_LAYERS):
                hp = out_p.hidden_states[L][0, prompt_len].float().cpu().numpy()
                hn = out_n.hidden_states[L][0, prompt_len].float().cpu().numpy()
                act_pos[i, L] = hp
                act_neg[i, L] = hn
            meta.append({"pair_idx": i, "axis": p["axis"], "split": p["split"],
                         "asin": p["asin"], "user_id": p["user_id"]})
            if (i + 1) % 100 == 0:
                log(f"  extracted {i + 1}/{len(pairs)}")
    log("extraction done")

    E15_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(E15_DIR / "activations.npz", act_pos=act_pos, act_neg=act_neg)
    with open(E15_DIR / "activation_manifest.json", "w") as f:
        json.dump({
            "model": BASE_MODEL,
            "model_hash": hashlib.sha256(Path(BASE_MODEL + "/config.json").read_bytes()).hexdigest() if Path(BASE_MODEL + "/config.json").exists() else None,
            "n_layers": N_LAYERS,
            "n_pairs": len(pairs),
            "token_position": "first token of pair sentence (after attr prompt)",
            "dtype": "bf16->float32",
            "pairs": meta,
        }, f, indent=2)

    # C2: fit directions on TRAIN pairs only (all three methods, per layer)
    train_idx = [i for i, m in enumerate(meta) if m["split"] == "train"]
    axes = sorted({m["axis"] for m in meta})
    directions = {}
    fit_metrics = {}
    for axis in axes:
        ax_idx = [i for i in train_idx if meta[i]["axis"] == axis]
        D = act_pos[ax_idx] - act_neg[ax_idx]  # [n_train, L, D]
        directions[axis] = {}
        for L in range(N_LAYERS):
            dL = D[:, L]
            # mean-difference direction (normalized)
            d_mean = dL.mean(axis=0)
            directions[axis][f"mean_diff_L{L}"] = d_mean / max(np.linalg.norm(d_mean), 1e-9)
            # logistic direction
            X = np.concatenate([act_pos[ax_idx, L], act_neg[ax_idx, L]])
            y = np.concatenate([np.ones(len(ax_idx)), np.zeros(len(ax_idx))])
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(X, y)
            w = clf.coef_[0].astype(np.float32)
            directions[axis][f"logistic_L{L}"] = w / max(np.linalg.norm(w), 1e-9)
            # pca direction (first component of the diff vectors)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pca.fit(dL)
            v = pca.components_[0].astype(np.float32)
            directions[axis][f"pca_L{L}"] = v / max(np.linalg.norm(v), 1e-9)
        fit_metrics[axis] = {"n_train_pairs": len(ax_idx), "layers": N_LAYERS}
        log(f"  fitted {axis}: {len(ax_idx)} train pairs")
    np.savez(E15_DIR / "activation_axes.npz", **{f"{a}_{m}": v for a, ds in directions.items() for m, v in ds.items()})
    with open(E15_DIR / "axis_fit_metrics.json", "w") as f:
        json.dump(fit_metrics, f, indent=2)
    log("wrote axes + metrics")


if __name__ == "__main__":
    main()
