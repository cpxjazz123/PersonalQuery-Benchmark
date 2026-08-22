#!/usr/bin/env python3
"""Extract Qwen hidden-state residuals for VADES training data.

对每条 user review sentence:
1. 提取 user sentence 的 Qwen hidden state (per layer, mean-pool)
2. 提取 neutral rewrite 的 Qwen hidden state (per layer, mean-pool)
3. residual_l = user_hidden_l - neutral_hidden_l (per layer)
4. 按 user 分组保存 npz, 供后续 diagonal_residual_llm mode 用

注意: 必须在 generate_neutral_rewrites.py 跑完后才能跑 (依赖 rewrites.jsonl)
"""
import json
import os
import sys
import time
from pathlib import Path

# 必须在 torch/vllm import 前设置
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
SCRATCH.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

# === 配置 (硬编码, Rule 3) ===
REWRITES_FILE = SCRATCH / "rewrites_10k.jsonl"
OUTPUT_FILE = SCRATCH / "residual_hidden_10k.npz"
HIDDEN_BATCH = int(os.environ.get("VADES_HIDDEN_BATCH", "16"))
HIDDEN_MAX_LENGTH = int(os.environ.get("VADES_HIDDEN_MAX_LENGTH", "128"))
# Qwen2-7B 共 28 层 (0-27); 选 Phase 14.F SOTA layer 26 + 中间层 16 + 后层 20
LAYERS = [int(x) for x in os.environ.get("VADES_HIDDEN_LAYERS", "16,20,24,26").split(",")]


def load_rewrites() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(REWRITES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("sentence_text") and d.get("rewrite"):
                    out[d["sentence_text"]] = d["rewrite"]
            except Exception:
                continue
    return out


def main() -> int:
    print(f"[main] starting at {time.strftime('%H:%M:%S')}")
    rewrites = load_rewrites()
    if not rewrites:
        print(f"[main] ✗ {REWRITES_FILE} 无内容, 先跑 generate_neutral_rewrites.py")
        return 1
    print(f"[main] loaded {len(rewrites)} rewrites")
    sentences = list(rewrites.keys())
    neutral_texts = [rewrites[s] for s in sentences]
    user_texts = sentences  # 直接复用原句

    # === 加载 Qwen (transformers 后端, 走 get_hidden_states) ===
    print(f"[main] loading Qwen hidden backend (layers={LAYERS})...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    # 默认 model 已通过 QwenLocalClient._hidden_backend property 懒加载
    # 触发一次访问让模型实际加载
    _ = client._hidden_backend.model
    hidden_dim = client._hidden_backend.model.config.hidden_size
    n_layer = client._hidden_backend.model.config.num_hidden_layers
    print(f"[main] Qwen loaded: hidden_dim={hidden_dim}, n_layer={n_layer}")

    # === 提取 user hidden ===
    print(f"[main] extracting user_hidden ({len(user_texts)} texts, layers={LAYERS})...")
    t0 = time.time()
    user_hidden = client.get_hidden_states(
        user_texts, layers=LAYERS,
        batch_size=HIDDEN_BATCH, max_length=HIDDEN_MAX_LENGTH,
    )
    print(f"[main] user_hidden done in {time.time()-t0:.1f}s")
    for l in LAYERS:
        print(f"  layer {l}: shape={user_hidden[l].shape}, dtype={user_hidden[l].dtype}, "
              f"norm_mean={float((user_hidden[l]**2).sum(axis=1).mean()**0.5):.2f}")

    # === 提取 neutral hidden ===
    print(f"[main] extracting neutral_hidden ({len(neutral_texts)} texts)...")
    t0 = time.time()
    neutral_hidden = client.get_hidden_states(
        neutral_texts, layers=LAYERS,
        batch_size=HIDDEN_BATCH, max_length=HIDDEN_MAX_LENGTH,
    )
    print(f"[main] neutral_hidden done in {time.time()-t0:.1f}s")
    for l in LAYERS:
        print(f"  layer {l}: shape={neutral_hidden[l].shape}, "
              f"norm_mean={float((neutral_hidden[l]**2).sum(axis=1).mean()**0.5):.2f}")

    # === 算 residual ===
    residual: dict[int, "np.ndarray"] = {}
    print(f"[main] computing residual = user - neutral...")
    for l in LAYERS:
        residual[l] = (user_hidden[l] - neutral_hidden[l]).astype("float32")
        r = residual[l]
        print(f"  layer {l}: shape={r.shape}, "
              f"norm_mean={float((r**2).sum(axis=1).mean()**0.5):.2f}, "
              f"max_abs={float(abs(r).max()):.2f}")

    # === 释放 user/neutral 内存(避免后续 OOM) ===
    del user_hidden, neutral_hidden

    # === 保存: 仅存 residual (不存 user/neutral, 节省 ~4x 内存) ===
    save_kwargs = {
        "sentences": np.asarray(sentences, dtype=object),
        "layers": np.asarray(LAYERS, dtype=np.int32),
        "hidden_dim": np.asarray([hidden_dim], dtype=np.int32),
    }
    for l in LAYERS:
        save_kwargs[f"residual_layer_{l}"] = residual[l]
    print(f"[main] saving {OUTPUT_FILE} ...")
    np.savez_compressed(OUTPUT_FILE, **save_kwargs)
    print(f"[main] ✓ saved {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size//1024} KB)")
    print(f"[main] done at {time.strftime('%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())