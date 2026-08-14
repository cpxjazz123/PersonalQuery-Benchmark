#!/usr/bin/env python3
"""E16-Task1 — Sentence-END representation extraction with layer alignment.

Fixes E15 biases: END-token (last token before EOS) hidden state, and
hidden_states[L+1] <-> layers[L] mapping. Directions (mean_diff/logistic/pca)
fit on TRAIN pairs only. Outputs land under result/personal_query/e16/ for
remote reproducibility.
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
BASE_MODEL = "/fs04/scratch2/ar57/wenyu/hf_home/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
E16_DIR = REPO_ROOT / "result" / "personal_query" / "e16"
PAIRS = E16_DIR / "valid_pairs.jsonl"
N_LAYERS = 28


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    pairs = [json.loads(l) for l in open(PAIRS)]
    log(f"valid pairs: {len(pairs)}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0", trust_remote_code=True)
    model.eval()

    act_pos = np.zeros((len(pairs), N_LAYERS, 1536), dtype=np.float32)
    act_neg = np.zeros((len(pairs), N_LAYERS, 1536), dtype=np.float32)
    meta = []
    with torch.no_grad():
        for i, p in enumerate(pairs):
            for sign, arr in [("positive", act_pos), ("negative", act_neg)]:
                txt = p[sign]
                ids = tok.encode(txt, add_special_tokens=False, return_tensors="pt").to("cuda:0")
                out = model(input_ids=ids, output_hidden_states=True)
                # sentence-END representation: last token (EOS-1) position
                # layer alignment: hidden_states[L+1] == layers[L] output
                for L in range(N_LAYERS):
                    arr[i, L] = out.hidden_states[L + 1][0, -1].float().cpu().numpy()
            meta.append({"pair_idx": i, "axis": p["axis"], "split": p["split"],
                         "asin": p["asin"], "user_id": p["user_id"]})
            if (i + 1) % 200 == 0:
                log(f"  extracted {i + 1}/{len(pairs)}")
    log("extraction done")

    np.savez(E16_DIR / "e16_activations.npz", act_pos=act_pos, act_neg=act_neg)
    with open(E16_DIR / "e16_activation_manifest.json", "w") as f:
        json.dump({
            "model": BASE_MODEL,
            "representation": "sentence-END token (last token, EOS-1)",
            "layer_mapping": "hidden_states[L+1] == layers[L] (verified in activation_contract.json)",
            "n_layers": N_LAYERS,
            "n_pairs": len(pairs),
            "pairs": meta,
        }, f, indent=2)

    train_idx = [i for i, m in enumerate(meta) if m["split"] == "train"]
    axes = sorted({m["axis"] for m in meta})
    directions = {}
    fit_metrics = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    for axis in axes:
        ax_idx = [i for i in train_idx if meta[i]["axis"] == axis]
        D = act_pos[ax_idx] - act_neg[ax_idx]
        directions[axis] = {}
        for L in range(N_LAYERS):
            dL = D[:, L]
            d_mean = dL.mean(axis=0)
            directions[axis][f"mean_diff_L{L}"] = d_mean / max(float(np.linalg.norm(d_mean)), 1e-9)
            X = np.concatenate([act_pos[ax_idx, L], act_neg[ax_idx, L]])
            y = np.concatenate([np.ones(len(ax_idx)), np.zeros(len(ax_idx))])
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(X, y)
            w = clf.coef_[0].astype(np.float32)
            directions[axis][f"logistic_L{L}"] = w / max(float(np.linalg.norm(w)), 1e-9)
            pca = PCA(n_components=1)
            pca.fit(dL)
            v = pca.components_[0].astype(np.float32)
            directions[axis][f"pca_L{L}"] = v / max(float(np.linalg.norm(v)), 1e-9)
        fit_metrics[axis] = {"n_train_pairs": len(ax_idx)}
        log(f"  fitted {axis}: {len(ax_idx)} train pairs")
    np.savez(E16_DIR / "e16_activation_axes.npz",
             **{f"{a}_{m}": v for a, ds in directions.items() for m, v in ds.items()})
    with open(E16_DIR / "e16_axis_fit_metrics.json", "w") as f:
        json.dump(fit_metrics, f, indent=2)
    log("wrote e16 axes + metrics")


if __name__ == "__main__":
    main()
