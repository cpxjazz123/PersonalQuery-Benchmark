#!/usr/bin/env python3
"""Phase 14.Q-6 enc: encode Q6 generated queries via Qwen 5 layers (transformers)."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")

IN_GEN = OUT_DIR / "phase14_q6_generated.jsonl"
IN_NEUTRAL_HIDDENS = OUT_DIR / "phase13_a_neutral_hiddens.npz"
OUT_CAND_RESID = OUT_DIR / "phase14_q6_cand_residuals_qwen.npy"

LAYERS_5 = [8, 14, 18, 22, 26]
HIDDEN = 3584
BATCH_SIZE = 32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log("=" * 70)
    log("Phase 14.Q-6 encode: Qwen 5 layers")
    log("=" * 70)

    log("[1] Loading generated queries ...")
    records = []
    with IN_GEN.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    log(f"  records: {len(records)}")

    log("[2] Loading Qwen client (transformers) ...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)

    log(f"[3] Encoding {len(records)} queries at {LAYERS_5} layers ...")
    texts = [r.get("q_styled") or "" for r in records]

    hdict = client.get_hidden_states(
        texts,
        layers=LAYERS_5,
        batch_size=BATCH_SIZE,
        max_length=256,
    )
    hiddens_all = np.stack([hdict[l] for l in LAYERS_5], axis=1).astype(np.float32)
    log(f"  hiddens_all shape: {hiddens_all.shape}")

    log("[4] Loading global neutral ...")
    npz_n = np.load(IN_NEUTRAL_HIDDENS, allow_pickle=True)
    neutral_vecs = npz_n["vecs"].astype(np.float32)
    global_neutral = neutral_vecs[:, LAYERS_5, :].mean(axis=0)

    cand_residuals = hiddens_all - global_neutral[None, :, :]
    log(f"  cand_residuals shape: {cand_residuals.shape}")

    np.save(OUT_CAND_RESID, cand_residuals)
    log(f"  → {OUT_CAND_RESID}")

    from collections import defaultdict
    by_cond = defaultdict(list)
    for ri, r in enumerate(records):
        by_cond[r["condition"]].append(ri)

    for cond in sorted(by_cond.keys()):
        idxs = by_cond[cond]
        norms = np.linalg.norm(cand_residuals[idxs, LAYERS_5.index(26)], axis=-1)
        log(f"  {cond}: n={len(idxs)}, norm mean={norms.mean():.2f}, std={norms.std():.2f}")

    log("=" * 70)
    log("PHASE 14.Q-6 ENCODE COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
