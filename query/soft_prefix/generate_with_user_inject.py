#!/usr/bin/env python3
"""Generate queries with per-user Gaussian mean injected into Qwen hidden layer.

Pipeline:
  1. 加载 Gaussian VADES 训练产出的 user profile (residual_llm_20u_gaussian)
     -> UserVectorProvider mode="vades" 拿到每用户 20-dim user_mu
  2. 加载/初始化 projection layer (Linear 20 -> Qwen hidden_dim, frozen random)
     路径: /home/wlia0047/hj82_scratch2/wenyu/user_style_steering/projection_layer.pt
     不存在则首次创建 + save (不训练, 随机 init 0.02;后续可接 trainer)
  3. 加载 query_records.json, 对每条 record 拿到 (user_id, attrs_used)
     构造 (system, user_prompt) 走 llm_client.generate_with_hidden_injection
     injection_per_row = α * (user_mu @ W_proj),None 行作为 D_off baseline
  4. 输出 result/query_records_with_query_inject.json (新增字段 y_plus_query_inject)

设计参考 (Phase 14.F SOTA layer 26 + Phase 14.B layer×α sweep):
  - 默认注入中间层 layer 16 (Qwen2-7B 共 28 层),后续 sweep 可调 8/12/16/20/24/26
  - 默认 α=1.0,后续 sweep 0.5/1.0/2.0

注意 (AGENTS.md Rule 7 严禁 fallback):
  - projection_layer.pt 不存在时显式 init + save, 不静默用零向量 (那等于 D_off 退化)
  - 没有 user profile 的用户用 zero vector (mode="vades" 的标准行为, UserVectorProvider
    返回 (zeros, has_vector=False), 不是 fallback

依赖:
  - llm_client.py::QwenLocalClient.generate_with_hidden_injection (transformers path)
  - query/soft_prefix/user_style_vectors.py::UserVectorProvider
  - result/residual_llm_20u_gaussian_user_profiles.jsonl (VADES 训练产出)

路径全部硬编码 (Rule 3), 不接受 CLI 参数。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/user_style_steering")
SCRATCH.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))

# === 硬编码路径 (Rule 3) ===
RECORDS_IN = REPO_ROOT / "result/query_records.json"
RECORDS_OUT = REPO_ROOT / "result/query_records_with_query_inject.json"
PROFILE_JSONL = REPO_ROOT / "result/residual_llm_20u_gaussian_user_profiles.jsonl"
PROJECTION_PT = SCRATCH / "projection_layer.pt"
GENERATION_LOG = SCRATCH / "generate_inject.log"

# === 硬编码实验配置 (Rule 3, 不接受 CLI 覆盖) ===
INJECTION_LAYERS = [int(x) for x in os.environ.get("INJECT_LAYERS", "16").split(",")]
INJECTION_ALPHA = float(os.environ.get("INJECT_ALPHA", "1.0"))
GEN_BATCH = int(os.environ.get("INJECT_GEN_BATCH", "8"))
GEN_MAX_NEW = int(os.environ.get("INJECT_GEN_MAX_NEW", "96"))
GEN_TEMP = float(os.environ.get("INJECT_GEN_TEMP", "0.5"))
GEN_TOP_P = float(os.environ.get("INJECT_GEN_TOP_P", "0.95"))
GEN_REPETITION_PENALTY = float(os.environ.get("INJECT_GEN_REP_PENALTY", "1.05"))
MAX_RECORDS = int(os.environ.get("INJECT_GEN_MAX_RECORDS", "500"))
MAX_INPUT_LENGTH = 384

# Amazon-style 自然 search query 模板 (与 generate_query_targets.py 一致, 让两路
# baseline 与 inject 的 prompt 分布对齐, 唯一差别是 inject 端的 hidden bias)
GEN_SYSTEM = (
    "You write realistic Amazon-style search queries. Output ONLY the query "
    "text — no preamble, no quotes, no sentence like 'I'm looking for'. "
    "Match the style of an Amazon search bar input: short, intent-driven, "
    "keywords joined by commas or natural phrasing. Mention only the most "
    "important 2-4 attributes; not every field. Never start with 'I'."
)


def build_user_content(attrs: dict) -> str:
    """与 generate_query_targets.py::build_user_content 一致的 prompt 拼接。"""
    lines = ["Product attributes:"]
    for k, v in attrs.items():
        s = str(v).strip() if v else ""
        if s:
            lines.append(s)
    lines.append("Write a short Amazon-style search query (no preamble).")
    return "\n".join(lines)


def load_or_init_projection(in_dim: int, out_dim: int) -> "torch.Tensor":
    """Load frozen projection W: (out_dim, in_dim). 不存在则 init + save。

    Init: normal(mean=0, std=0.02), 符合 transformer convention (类似 nn.Linear
    default init); 不训练 (frozen), 后续可接 trainer (尚未实现)。

    返回: torch.Tensor shape (out_dim, in_dim), dtype=bfloat16 (与 Qwen 一致)
    """
    import torch
    if PROJECTION_PT.exists():
        W = torch.load(PROJECTION_PT, map_location="cpu")
        assert W.shape == (out_dim, in_dim), (
            f"projection shape mismatch: file={W.shape}, expected ({out_dim},{in_dim})"
        )
        return W
    print(f"[proj] projection 不存在, init random normal(0, 0.02): {PROJECTION_PT}")
    g = torch.Generator().manual_seed(42)
    W = torch.empty(out_dim, in_dim)
    torch.nn.init.normal_(W, mean=0.0, std=0.02, generator=g)
    PROJECTION_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(W, PROJECTION_PT)
    print(f"[proj] 已保存 {PROJECTION_PT} (shape={tuple(W.shape)})")
    return W


def main() -> int:
    import torch
    t0 = time.time()

    # === 1) 加载 user profile (VADES Gaussian mean) ===
    print(f"[main] loading user profiles: {PROFILE_JSONL}", flush=True)
    if not PROFILE_JSONL.exists():
        raise FileNotFoundError(
            f"VADES profile 不存在: {PROFILE_JSONL}\n"
            f"先跑 gaussian/gaussian_vades.py 输出 tag=residual_llm_20u_gaussian"
        )
    from user_style_vectors import (
        UserVectorProvider, l2_normalize, VADES_LATENT_DIM,
    )
    profiles: dict[str, dict] = {}
    with open(PROFILE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "user_mu" not in row:
                continue
            mu_list = row["user_mu"]
            if not isinstance(mu_list, list):
                raise ValueError(
                    f"user_mu 必须是 list, 实际 {type(mu_list).__name__}; "
                    f"user_id={row.get('user_id', '?')}"
                )
            mu_t = torch.as_tensor(mu_list, dtype=torch.float32)
            profiles[row["user_id"]] = {"user_mu": mu_t}

    print(f"[main] loaded {len(profiles)} user profiles (VADES_LATENT_DIM={VADES_LATENT_DIM})")

    # === 2) 加载 Qwen 拿到 hidden_dim ===
    print(f"[main] loading Qwen (transformers path, with_vllm=False)...", flush=True)
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    hidden_dim = client._hidden_backend.model.config.hidden_size
    print(f"[main] hidden_dim={hidden_dim}, layers={INJECTION_LAYERS}")

    # === 3) 加载 / 初始化 projection ===
    W = load_or_init_projection(in_dim=VADES_LATENT_DIM, out_dim=hidden_dim)
    # _hidden_backend.dtype 是字符串 ("bfloat16" 等), 需先 map 成 torch.dtype
    _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    target_dtype = _DTYPE_MAP.get(client._hidden_backend.dtype)
    if target_dtype is None:
        raise ValueError(
            f"unsupported hidden backend dtype: {client._hidden_backend.dtype!r}"
        )
    W_dev = W.to(device=client._hidden_backend.device, dtype=target_dtype)
    print(f"[main] projection shape={tuple(W.shape)}, on device {W_dev.device}, dtype={target_dtype}")

    # === 4) 加载 query_records ===
    print(f"[main] loading records: {RECORDS_IN}", flush=True)
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    if MAX_RECORDS and len(records) > MAX_RECORDS:
        records = records[:MAX_RECORDS]
    print(f"[main] {len(records)} records to generate")

    # === 5) 增量加载: 已生成的跳过 ===
    existing: dict[tuple[str, str], str] = {}
    if RECORDS_OUT.exists():
        prev = json.load(open(RECORDS_OUT, "r", encoding="utf-8"))
        for r in prev:
            if r.get("y_plus_query_inject") and "user_id" in r and "asin" in r:
                existing[(r["user_id"], r["asin"])] = r["y_plus_query_inject"]
        print(f"[main] {len(existing)} already-generated (增量跳过)")

    # === 6) 准备 todo ===
    todo_records: list[dict] = []
    todo_prompts: list[str] = []
    todo_injections: list = []  # (hidden_dim,) tensor 或 None
    n_no_profile = 0
    n_zero_norm = 0
    for r in records:
        uid, asin = r["user_id"], r["asin"]
        if (uid, asin) in existing:
            continue
        attrs = r.get("attrs_used", {})
        if not attrs:
            continue
        prof = profiles.get(uid)
        if prof is None:
            n_no_profile += 1
            # 没 profile: 显式 zero vector + has_vector=False (UserVectorProvider 标准行为)
            bias = torch.zeros(hidden_dim, dtype=torch.float32)
        else:
            mu_t = prof["user_mu"]  # (20,) float32 tensor
            norm = float(mu_t.norm())
            if norm < 1e-12:
                n_zero_norm += 1
                bias = torch.zeros(hidden_dim, dtype=torch.float32)
            else:
                mu_l2 = mu_t / norm  # L2-normalized, (20,)
                # bias = (W @ mu_l2), shape (hidden_dim,)
                bias = W_dev @ mu_l2.to(device=W_dev.device, dtype=W_dev.dtype)
        todo_records.append(r)
        todo_prompts.append(build_user_content(attrs))
        todo_injections.append(bias)
    print(
        f"[main] todo={len(todo_records)}, "
        f"no_profile={n_no_profile}, zero_norm={n_zero_norm}, "
        f"already={len(existing)}",
        flush=True,
    )

    if not todo_records:
        print(f"[main] ✓ 无 todo, 退出")
        return 0

    # === 7) 写出初始结果 (已生成 fill) ===
    out_records: list[dict] = []
    for r in records:
        if (r["user_id"], r["asin"]) in existing:
            r2 = dict(r)
            r2["y_plus_query_inject"] = existing[(r["user_id"], r["asin"])]
            out_records.append(r2)
        else:
            out_records.append(dict(r))

    # === 8) 分批生成 ===
    new_rows: list[dict] = []
    for i in range(0, len(todo_prompts), GEN_BATCH):
        chunk_prompts = todo_prompts[i:i + GEN_BATCH]
        chunk_records = todo_records[i:i + GEN_BATCH]
        chunk_inj = todo_injections[i:i + GEN_BATCH]
        try:
            queries = client.generate_with_hidden_injection(
                system_text=GEN_SYSTEM,
                user_texts=chunk_prompts,
                injection_per_row=chunk_inj,
                injection_layers=INJECTION_LAYERS,
                injection_alpha=INJECTION_ALPHA,
                max_new_tokens=GEN_MAX_NEW,
                temperature=GEN_TEMP,
                top_p=GEN_TOP_P,
                repetition_penalty=GEN_REPETITION_PENALTY,
                batch_size=GEN_BATCH,
                max_input_length=MAX_INPUT_LENGTH,
            )
        except Exception as exc:
            import traceback
            print(f"[main] ✗ batch failed at chunk {i}: {exc!r}")
            traceback.print_exc()
            raise
        for r, q in zip(chunk_records, queries):
            r["y_plus_query_inject"] = q
            new_rows.append(r)
        done = min(i + GEN_BATCH, len(todo_prompts))
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        eta = (len(todo_prompts) - done) / max(rate, 1e-6)
        print(
            f"[main] {done}/{len(todo_prompts)} ({rate:.2f}/s, ETA {eta:.0f}s)",
            flush=True,
        )

    # === 9) fill back 到 out_records ===
    for r in new_rows:
        for orec in out_records:
            if orec["user_id"] == r["user_id"] and orec["asin"] == r["asin"]:
                orec["y_plus_query_inject"] = r["y_plus_query_inject"]
                break

    # === 10) 写出 ===
    RECORDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_records, open(RECORDS_OUT, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    n_with_inject = sum(1 for r in out_records if r.get("y_plus_query_inject"))
    print(f"[main] ✓ wrote {RECORDS_OUT}: {n_with_inject}/{len(out_records)} with inject")
    print(f"[main] total {time.time()-t0:.1f}s")
    print(f"[main] layers={INJECTION_LAYERS}, alpha={INJECTION_ALPHA}, batch={GEN_BATCH}")
    for r in out_records[:3]:
        print(f"\n--- asin={r['asin']} (user={r['user_id']}) ---")
        print(f"  attrs: {r['attrs_used']}")
        print(f"  y_plus_query_inject: {r.get('y_plus_query_inject', '')[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
