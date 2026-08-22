#!/usr/bin/env python3
"""Phase 14.P-1: Encode all 11 conds' cands through Qwen at layer 26.

Output: phase14_p_cand_residuals_qwen.npy shape (2640, 5, 3584) — same format as phase14_f cache.

Reuses:
  - phase14_f_cand_residuals_qwen.npy for 720 cands (A22_a0.5 + A14_a1.0 + D_off)
  - phase14_b_layer_alpha_sweep.jsonl for all 2640 cand texts
  - phase13_a_neutral_hiddens.npz for global neutral
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_JSONL = OUT_DIR / "phase14_b_layer_alpha_sweep.jsonl"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
IN_CAND_RESID_3CONDS = OUT_DIR / "phase14_f_cand_residuals_qwen.npy"  # 720 cands

OUT_CAND_RESID = OUT_DIR / "phase14_p_cand_residuals_qwen.npy"

LAYERS_5 = [8, 14, 18, 22, 26]
HIDDEN = 3584
BATCH_SIZE = 32

# 3 conds already cached
CACHED_CONDS = ["A22_a0.5", "A14_a1.0", "D_off"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=" * 70)
    log("Phase 14.P-1: Encode all 11 conds' cands through Qwen at layer 26")
    log("=" * 70)

    # === [1] Load all candidates ===
    log(f"[1] Loading candidates from {IN_JSONL.name} ...")
    candidates: list[dict] = []
    with IN_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            candidates.append(r)
    log(f"  total candidates: {len(candidates)}")

    # === [2] Load Qwen client ===
    log("[2] Loading Qwen local client ...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    log("  client loaded")

    # === [3] Encode all candidates through Qwen at 5 layers ===
    log(f"[3] Encoding {len(candidates)} candidates at {LAYERS_5} layers (mean-pool) ...")
    texts = [c.get("q_styled") or "" for c in candidates]

    hdict = client.get_hidden_states(
        texts,
        layers=LAYERS_5,
        batch_size=BATCH_SIZE,
        max_length=256,
    )
    # hdict[layer] is [N, H]; stack to [N, len(layers), H]
    hiddens_all = np.stack([hdict[l] for l in LAYERS_5], axis=1).astype(np.float32)
    log(f"  hiddens_all shape: {hiddens_all.shape}")

    # === [4] Subtract global neutral ===
    log("[4] Loading global neutral ...")
    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)  # [2976, 28, 3584]
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)  # [5, 3584]

    cand_residuals_all = hiddens_all - global_neutral[None, :, :]  # [2640, 5, 3584]
    log(f"  cand_residuals_all shape: {cand_residuals_all.shape}")

    # === [5] Sanity check vs cached 720 ===
    log("[5] Sanity check: compare vs cached 720 cands (first 3 conds) ...")
    cached = np.load(IN_CAND_RESID_3CONDS)  # [720, 5, 3584]
    # Cached corresponds to first 720 cands in candidates list (A22_a0.5, A14_a1.0, D_off)
    n_cached = sum(1 for c in candidates if c["condition"] in CACHED_CONDS)
    log(f"  expected cached n: {n_cached}")
    log(f"  loaded cached shape: {cached.shape}")
    # Compare first 5 cands
    diff = np.abs(cand_residuals_all[:5] - cached[:5]).max()
    log(f"  max diff first 5: {diff:.6f} (should be ~0)")

    # === [6] Save ===
    np.save(OUT_CAND_RESID, cand_residuals_all)
    log(f"  saved → {OUT_CAND_RESID}")

    # === [7] Stats per cond ===
    log("[7] Stats per cond (layer 26 mean norm) ...")
    from collections import defaultdict
    by_cond = defaultdict(list)
    for ci, c in enumerate(candidates):
        by_cond[c["condition"]].append(ci)

    for cond in sorted(by_cond.keys()):
        idxs = by_cond[cond]
        norms = np.linalg.norm(cand_residuals_all[idxs, LAYERS_5.index(26)], axis=-1)
        log(f"  {cond}: n={len(idxs)}, norm mean={norms.mean():.2f}, std={norms.std():.2f}")

    log("=" * 70)
    log("PHASE 14.P-1 COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()