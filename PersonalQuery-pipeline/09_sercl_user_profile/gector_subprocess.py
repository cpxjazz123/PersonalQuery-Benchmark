#!/usr/bin/env python3
"""GECToR-2024 RoBERTa-large batch corrector for SErCL Stage 2.

Runs in the required pq_env interpreter. Reads and writes fixed JSONL paths;
the parent launches this script once per category so the model is loaded once
for that category's batched sentences.

Inputs: $PQ_SCRATCH/tmp/gec_in.jsonl
  Each line: {"i": <int>, "text": "<original sentence>"}

Outputs: $PQ_SCRATCH/tmp/gec_out.jsonl
  Each line: {"i": <int>, "text": "<corrected sentence>"}
  (preserves original order index `i`; caller must sort by i)

Usage (Rule 3: no parameters):
  $PQ_PYTHON $(dirname "$0")/gector_subprocess.py

Path resolution:
  REPO_ROOT = directory containing this script's parent
  SCRATCH   = $PQ_SCRATCH (default: <repo_root>/scratch)
  HF_CACHE_ROOT = $PQ_HF_CACHE (default: <repo_root>/scratch/hf_cache)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

# ----- Paths (Rule A: REPO_ROOT; Rule C: no absolute path defaults) -----
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("PQ_SCRATCH", str(REPO_ROOT / "scratch")))
HF_CACHE_ROOT = Path(os.environ.get("PQ_HF_CACHE", str(REPO_ROOT / "scratch" / "hf_cache")))
INPUT_JSONL = SCRATCH / "tmp" / "gec_in.jsonl"
OUTPUT_JSONL = SCRATCH / "tmp" / "gec_out.jsonl"
WEIGHTS = HF_CACHE_ROOT / "gector" / "gector-2024-roberta-large.th"
VOCAB_DIR = HF_CACHE_ROOT / "gector" / "vocab"
HF_CACHE = str(HF_CACHE_ROOT)

# GECToR hyperparams
GEC_TRANSFORMER = "roberta-large"
GEC_MAX_LENGTH = 80
GEC_BATCH_SIZE = 64
GEC_N_ITER = 3
GEC_USE_BF16 = True
GEC_MIN_ERROR_PROB = 0.0
GEC_KEEP_CONFIDENCE = 0.0

# Force HF cache to scratch2
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_HUB_CACHE"] = HF_CACHE

# Make gotutiyan/gector importable
sys.path.insert(0, str(SCRATCH / "external" / "gector" / "src"))

import torch
from transformers import AutoTokenizer
from gector import GECToR, predict, load_verb_dict


def log(msg: str) -> None:
    print(f"[gector {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _heartbeat(stop: threading.Event, start_time: float, n_total: int) -> None:
    """Print elapsed time + estimated rate every 30s while predict runs.
    GECToR only logs once per iteration end, so for iter 0 on large inputs we add this.
    """
    while not stop.is_set():
        elapsed = time.time() - start_time
        # very rough estimate: assume linear throughput ~100 sent/s (lower bound for safety)
        approx_done = min(n_total, int(elapsed * 100))
        pct = approx_done / n_total * 100
        print(f"[gector {time.strftime('%H:%M:%S')}] [heartbeat] "
              f"elapsed={elapsed:.0f}s  est_progress≈{pct:.0f}% "
              f"(rough, real iter count shown by gector when each iter finishes)",
              flush=True)
        stop.wait(30.0)


def main() -> None:
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"input JSONL not found: {INPUT_JSONL}")
    INPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    # Read inputs
    log(f"reading inputs: {INPUT_JSONL}")
    items = []
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if not items:
        raise RuntimeError(f"input JSONL is empty: {INPUT_JSONL}")
    log(f"  loaded {len(items)} sentences")

    # Sort by index to ensure deterministic order, but we'll preserve original `i`
    srcs = [it["text"] for it in items]

    # Load model
    log(f"loading GECToR: {WEIGHTS.name}")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GECToR.from_official_pretrained(
        str(WEIGHTS),
        special_tokens_fix=1,
        transformer_model=GEC_TRANSFORMER,
        vocab_path=str(VOCAB_DIR),
        max_length=GEC_MAX_LENGTH,
    ).to(device)
    log(f"  loaded in {time.time()-t0:.1f}s, device={device}")

    # Convert model to bf16 for ~2× inference speedup (RoBERTa sm_86 native bf16 support)
    if GEC_USE_BF16 and device.type == "cuda":
        log(f"  converting model to bfloat16 (A40 sm_86 supports native bf16)")
        model = model.to(torch.bfloat16)
        # Re-tie weights in correct dtype if needed
        log(f"  ✓ model dtype={next(model.parameters()).dtype}")

    log("loading roberta-large tokenizer (add_prefix_space=True)")
    tokenizer = AutoTokenizer.from_pretrained(GEC_TRANSFORMER, add_prefix_space=True)

    log(f"loading verb vocab: {VOCAB_DIR}/verb-form-vocab.txt")
    encode_v, decode_v = load_verb_dict(str(VOCAB_DIR / "verb-form-vocab.txt"))

    # Predict
    log(f"predicting {len(srcs)} sentences (batch={GEC_BATCH_SIZE}, n_iter={GEC_N_ITER})")
    t0 = time.time()
    # Start heartbeat thread (prints every 30s while iter 0 grinds through large input)
    _stop_hb = threading.Event()
    _hb = threading.Thread(target=_heartbeat, args=(_stop_hb, t0, len(srcs)), daemon=True)
    _hb.start()
    corrected = predict(
        model, tokenizer, srcs, encode_v, decode_v,
        keep_confidence=GEC_KEEP_CONFIDENCE,
        min_error_prob=GEC_MIN_ERROR_PROB,
        n_iteration=GEC_N_ITER,
        batch_size=GEC_BATCH_SIZE,
    )
    _stop_hb.set()
    _hb.join(timeout=1.0)
    dt = time.time() - t0
    rate = len(srcs) / (dt + 1e-9)
    log(f"  done in {dt:.1f}s, rate={rate:.1f}/s")

    # Write outputs (preserve original `i`)
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for it, c in zip(items, corrected):
            f.write(json.dumps({"i": it["i"], "text": c}, ensure_ascii=False) + "\n")
    log(f"wrote → {OUTPUT_JSONL} ({len(items)} lines)")


if __name__ == "__main__":
    main()
