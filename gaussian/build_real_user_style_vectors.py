#!/usr/bin/env python3
"""Build real user style vectors from VADES Qwen residuals.

For each user:
1. Mean-pool all sentence residuals (layer 16/20/24/26) → 3584d per user
2. Write user_style_vectors_real.jsonl with format:
   {user_id, residual_mean_layer_16, residual_mean_layer_20, residual_mean_layer_24, residual_mean_layer_26}

These are the actual user style signals (mean residual = user - neutral),
used as the bias vector for hidden-state injection in generate_strict_nohallu.py.

Reads:  /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/residual_hidden_10k.npz
        /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/rewrites_10k.jsonl
Writes: /home/wlia0047/hj82_scratch2/wenyu/user_style_steering/user_style_vectors_real.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

SCRATCH_G = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
SCRATCH_U = Path("/home/wlia0047/hj82_scratch2/wenyu/user_style_steering")
SCRATCH_U.mkdir(parents=True, exist_ok=True)

REWRITES = SCRATCH_G / "rewrites_10k.jsonl"
NPZ_FILE = SCRATCH_G / "residual_hidden_10k.npz"
OUT = SCRATCH_U / "user_style_vectors_real.jsonl"

LAYERS = [16, 20, 24, 26]


def main() -> None:
    # Load sentence → uid mapping (dedup same as extract_residual_hidden.py)
    print("[load] sentence→uid from rewrites_10k.jsonl (dedup)...")
    sent_to_uid: dict[str, str] = {}
    with open(REWRITES, "r") as f:
        for line in f:
            r = json.loads(line)
            sent = r.get("sentence_text", "")
            uid = r.get("user_id", "")
            if sent and sent not in sent_to_uid:
                sent_to_uid[sent] = uid
    print(f"  {len(sent_to_uid)} unique sentences, {len(set(sent_to_uid.values()))} unique uids")

    # Load residuals npz
    print(f"[load] residuals from {NPZ_FILE} ...")
    npz = np.load(NPZ_FILE, allow_pickle=True)
    sentences = npz["sentences"].tolist()
    layer_resids = {L: npz[f"residual_layer_{L}"] for L in LAYERS}
    print(f"  shape per layer: {layer_resids[LAYERS[0]].shape}")

    # Build per-user mean residual per layer
    print("[group] per-user mean residual per layer...")
    user_layer_vecs: dict[str, dict[int, list[list[float]]]] = {
        uid: {L: [] for L in LAYERS} for uid in set(sent_to_uid.values())
    }
    n_mapped = 0
    for sent_idx, sent in enumerate(sentences):
        uid = sent_to_uid.get(sent)
        if uid is None:
            continue
        for L in LAYERS:
            user_layer_vecs[uid][L].append(layer_resids[L][sent_idx])
        n_mapped += 1
    print(f"  mapped {n_mapped} / {len(sentences)} sentences to uids")

    # Mean pool + write
    print(f"[write] {len(user_layer_vecs)} users → {OUT}")
    with open(OUT, "w", encoding="utf-8") as fh:
        for uid, layer_dict in user_layer_vecs.items():
            row = {"user_id": uid}
            for L in LAYERS:
                vecs = layer_dict[L]
                if not vecs:
                    mean_vec = [0.0] * 3584
                else:
                    arr = np.stack(vecs, axis=0).astype(np.float32)
                    mean_vec = arr.mean(axis=0).tolist()
                row[f"residual_mean_layer_{L}"] = mean_vec
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  written. file size: {OUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
