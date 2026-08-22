#!/usr/bin/env python3
"""Generate queries with per-user 真实 Qwen hidden residual 注入 layer output.

背景 (用户指令 2026-08-22 修正):
  旧版用 VADES 20-dim user_mu + random Linear(20 -> 3584) projection 注入,
  projection 未训练 → 注入 noise → 输出 broken (alpha=1.0 案例:
  ""Ajable pen, refillable ink cartridges...").

  正确路线 (参考 ACL 2025 StyleVector / EACL 2024 Style Vectors):
    直接用 build_user_style_vector_real.py 产出的 真实 Qwen hidden residual
    (3584-dim) 作为注入 bias, 不需要 projector:
      h_l' = h_l + α * Δh_u

模式 (INJECT_MODE):
  - direct_mean (B 路线): Δh_u = user 在 layer L 的 mean residual (3584-dim, 确定性)
  - pca_gaussian (C 路线): z_u ~ N(mu_u, Sigma_u) on 20-dim PCA space
                          → Δh_u = z_u @ U.T + mean_residual (3584-dim, 可采样)

依赖:
  - /home/wlia0047/hj82_scratch2/wenyu/user_style_steering/user_style_vectors_real.jsonl
    (B 路线, build_user_style_vector_real.py 产出)
  - /home/wlia0047/hj82_scratch2/wenyu/user_style_steering/pca_model.npz
    (C 路线, 同上)
  - llm_client.py::QwenLocalClient.generate_with_hidden_injection

α 起步范围 (用户建议, 不要 1.0): 0.05/0.1/0.2/0.3/0.5
- direct_mean: 用户 mean norm 30-70, Qwen hidden norm 60-100 → α=0.3 让 hidden 加 ~20
- pca_gaussian: 采样 norm 与原始 residual 一致 (~200), α 必须更小 (0.05-0.1)

路径全部硬编码 (Rule 3), 不接受 CLI 参数; 通过 INJECT_* 环境变量覆盖。
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
RECORDS_OUT_DEFAULT = REPO_ROOT / "result/query_records_with_query_inject.json"
# 可选 OUT_SUFFIX (env-var) 给 output 加后缀, 避免 sweep 时互相覆盖
# 例: INJECT_OUT_SUFFIX=b_L26_a0.5 → result/query_records_with_query_inject_b_L26_a0.5.json
OUT_SUFFIX = os.environ.get("INJECT_OUT_SUFFIX", "")
RECORDS_OUT = (
    REPO_ROOT / "result" / f"query_records_with_query_inject_{OUT_SUFFIX}.json"
    if OUT_SUFFIX else RECORDS_OUT_DEFAULT
)
USER_STYLE_VECTORS_JSONL = SCRATCH / "user_style_vectors_real.jsonl"
PCA_MODEL_NPZ = SCRATCH / "pca_model.npz"
GENERATION_LOG = SCRATCH / "generate_inject.log"

# === 实验配置 (env-var 覆盖硬编码默认值) ===
INJECT_MODE = os.environ.get("INJECT_MODE", "direct_mean")  # direct_mean | pca_gaussian
INJECT_LAYERS = [int(x) for x in os.environ.get("INJECT_LAYERS", "16").split(",")]
INJECT_ALPHA = float(os.environ.get("INJECT_ALPHA", "0.3"))
GEN_BATCH = int(os.environ.get("INJECT_GEN_BATCH", "8"))
GEN_MAX_NEW = int(os.environ.get("INJECT_GEN_MAX_NEW", "96"))
GEN_TEMP = float(os.environ.get("INJECT_GEN_TEMP", "0.5"))
GEN_TOP_P = float(os.environ.get("INJECT_GEN_TOP_P", "0.95"))
GEN_REPETITION_PENALTY = float(os.environ.get("INJECT_GEN_REP_PENALTY", "1.05"))
MAX_RECORDS = int(os.environ.get("INJECT_GEN_MAX_RECORDS", "500"))
MAX_INPUT_LENGTH = 384
PCA_SAMPLE_SEED = int(os.environ.get("INJECT_PCA_SEED", "42"))  # C 路线采样种子
# 用户指令 2026-08-22: 屏蔽 CJK 字符 token, 防止注入风格偏移引入中文噪声
MASK_CJK = os.environ.get("INJECT_MASK_CJK", "1") == "1"

# 第一人称自然语言 query 模板 (用户指令 2026-08-22: 不是关键词罗列, 是描述
# 自己需求的第一人称 query)。Allow "I want / I'm looking for / Need a"
# 这类搜索意图驱动表达, 2-4 个关键属性, < 80 字符。
GEN_SYSTEM = (
    "You write realistic first-person product search queries as if you are the "
    "customer typing into a search box. Output ONLY the query text — no "
    "preamble, no quotes. Express what YOU (the shopper) are looking for in "
    "natural phrasing like 'I'm looking for', 'I want', 'I need', 'Searching "
    "for', 'Looking for'. Mention ALL 5 provided attributes in the query — "
    "each must appear naturally in the sentence. Keep it short (< 120 chars)."
)


def build_user_content(attrs: dict) -> str:
    lines = ["Product attributes:"]
    for k, v in attrs.items():
        s = str(v).strip() if v else ""
        if s:
            lines.append(s)
    lines.append("Write a short Amazon-style search query (no preamble).")
    return "\n".join(lines)


class DirectMeanProvider:
    """B 路线: Δh_u = user 在指定 layer 的 mean residual (3584-dim, 确定性)。"""

    def __init__(self, jsonl_path: Path, layers: list[int]):
        self.layers = layers
        self.profiles: dict[str, dict[int, list[float]]] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                uid = row["user_id"]
                self.profiles[uid] = {}
                for L in layers:
                    key = f"residual_mean_layer_{L}"
                    if key in row:
                        self.profiles[uid][L] = row[key]

    def get_bias(self, user_id: str, layer: int) -> "torch.Tensor | None":
        import torch
        prof = self.profiles.get(user_id)
        if prof is None or layer not in prof:
            return None
        # 直接用真实 3584-dim residual mean
        return torch.as_tensor(prof[layer], dtype=torch.float32)


class PCAGaussianProvider:
    """C 路线: z_u ~ N(mu_u, Sigma_u) on 20-dim PCA space → inverse → 3584-dim Δh_u。"""

    def __init__(self, pca_path: Path, layers: list[int], seed: int = 42):
        import torch
        npz = np.load(pca_path, allow_pickle=True)
        self.pca_components = torch.as_tensor(npz["pca_components"], dtype=torch.float32)  # (20, 3584)
        self.pca_mean = torch.as_tensor(npz["pca_mean"], dtype=torch.float32)  # (3584,)
        self.user_ids = [str(u) for u in npz["user_ids"]]
        self.user_mu = torch.as_tensor(npz["user_mu"], dtype=torch.float32)  # (n_users, 20)
        self.user_cov = torch.as_tensor(npz["user_cov"], dtype=torch.float32)  # (n_users, 20, 20)
        self.uid_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.layers = layers

    def get_bias(self, user_id: str, layer: int) -> "torch.Tensor | None":
        """返回 (3584,) tensor: Δh_u = z_u @ U.T + mean, 其中 z_u ~ N(mu_u, Sigma_u)。"""
        import torch
        idx = self.uid_to_idx.get(user_id)
        if idx is None:
            return None
        mu_u = self.user_mu[idx].numpy()  # (20,)
        cov_u = self.user_cov[idx].numpy()  # (20, 20)
        # sample z_u from N(mu, cov)
        z_u = self.rng.multivariate_normal(mu_u, cov_u)  # (20,)
        z_t = torch.as_tensor(z_u, dtype=torch.float32)
        # Δh_u = z @ U.T + mean   (U is (20, 3584), U.T is (3584, 20))
        delta_h = z_t @ self.pca_components.T + self.pca_mean  # (3584,)
        return delta_h


def main() -> int:
    import torch
    t0 = time.time()

    # === 1) 加载 Qwen 拿 hidden_dim ===
    print(f"[main] loading Qwen (transformers path)...", flush=True)
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    hidden_dim = client._hidden_backend.model.config.hidden_size
    print(f"[main] hidden_dim={hidden_dim}, layers={INJECT_LAYERS}, mode={INJECT_MODE}, alpha={INJECT_ALPHA}")
    if hidden_dim != 3584:
        raise ValueError(f"unexpected hidden_dim={hidden_dim}, 仅支持 Qwen2-7B (3584)")

    # === 2) 加载 user style vector provider ===
    if INJECT_MODE == "direct_mean":
        if not USER_STYLE_VECTORS_JSONL.exists():
            raise FileNotFoundError(
                f"{USER_STYLE_VECTORS_JSONL} 不存在, 先跑 query/soft_prefix/build_user_style_vector_real.py"
            )
        provider = DirectMeanProvider(USER_STYLE_VECTORS_JSONL, INJECT_LAYERS)
        print(f"[main] DirectMeanProvider: {len(provider.profiles)} users, layers={INJECT_LAYERS}")
    elif INJECT_MODE == "pca_gaussian":
        if not PCA_MODEL_NPZ.exists():
            raise FileNotFoundError(
                f"{PCA_MODEL_NPZ} 不存在, 先跑 query/soft_prefix/build_user_style_vector_real.py"
            )
        provider = PCAGaussianProvider(PCA_MODEL_NPZ, INJECT_LAYERS, seed=PCA_SAMPLE_SEED)
        print(f"[main] PCAGaussianProvider: {len(provider.user_ids)} users, "
              f"pca_dim=20, layers={INJECT_LAYERS}, seed={PCA_SAMPLE_SEED}")
    else:
        raise ValueError(f"unknown INJECT_MODE={INJECT_MODE!r}, 必须 direct_mean | pca_gaussian")

    # === 3) 加载 query_records ===
    print(f"[main] loading records: {RECORDS_IN}", flush=True)
    records = json.load(open(RECORDS_IN, "r", encoding="utf-8"))
    if MAX_RECORDS and len(records) > MAX_RECORDS:
        records = records[:MAX_RECORDS]
    print(f"[main] {len(records)} records to generate")

    # === 4) 增量加载 ===
    existing: dict[tuple[str, str], str] = {}
    if RECORDS_OUT.exists():
        prev = json.load(open(RECORDS_OUT, "r", encoding="utf-8"))
        for r in prev:
            if r.get("y_plus_query_inject") and "user_id" in r and "asin" in r:
                existing[(r["user_id"], r["asin"])] = r["y_plus_query_inject"]
        print(f"[main] {len(existing)} already-generated (增量跳过)")

    # === 5) 准备 todo ===
    todo_records: list[dict] = []
    todo_prompts: list[str] = []
    todo_injections: list = []  # (3584,) tensor 或 None
    n_no_profile = 0
    for r in records:
        uid, asin = r["user_id"], r["asin"]
        if (uid, asin) in existing:
            continue
        attrs = r.get("attrs_used", {})
        if not attrs:
            continue
        # 拿每 user 在每个注入 layer 的 bias
        # 因为 injection_layers 可能是多层, 我们对每条 record 取所有 layer 的 bias 列表
        # 但 generate_with_hidden_injection 的 injection_per_row 是 per-row 的 (hidden_dim,)
        # 多层注入: 每层拿一次 bias, 但 llm_client 接口只支持每行一个 bias 应用于所有指定 layer
        # → 简化: 取第一层 bias (INJECT_LAYERS[0]); 后续扩展可对每层分别 hook
        bias = provider.get_bias(uid, INJECT_LAYERS[0])
        if bias is None:
            n_no_profile += 1
            bias = torch.zeros(hidden_dim, dtype=torch.float32)
        todo_records.append(r)
        todo_prompts.append(build_user_content(attrs))
        todo_injections.append(bias)
    print(
        f"[main] todo={len(todo_records)}, no_profile={n_no_profile}, "
        f"already={len(existing)}, mode={INJECT_MODE}",
        flush=True,
    )

    if not todo_records:
        print(f"[main] ✓ 无 todo, 退出")
        return 0

    # === 6) 写出初始结果 ===
    out_records: list[dict] = []
    for r in records:
        if (r["user_id"], r["asin"]) in existing:
            r2 = dict(r)
            r2["y_plus_query_inject"] = existing[(r["user_id"], r["asin"])]
            out_records.append(r2)
        else:
            out_records.append(dict(r))

    # === 7) 分批生成 ===
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
                injection_layers=INJECT_LAYERS,
                injection_alpha=INJECT_ALPHA,
                max_new_tokens=GEN_MAX_NEW,
                temperature=GEN_TEMP,
                top_p=GEN_TOP_P,
                repetition_penalty=GEN_REPETITION_PENALTY,
                batch_size=GEN_BATCH,
                max_input_length=MAX_INPUT_LENGTH,
                mask_cjk=MASK_CJK,
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

    # === 8) fill back ===
    for r in new_rows:
        for orec in out_records:
            if orec["user_id"] == r["user_id"] and orec["asin"] == r["asin"]:
                orec["y_plus_query_inject"] = r["y_plus_query_inject"]
                break

    # === 9) 写出 ===
    RECORDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_records, open(RECORDS_OUT, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    n_with_inject = sum(1 for r in out_records if r.get("y_plus_query_inject"))
    print(f"[main] ✓ wrote {RECORDS_OUT}: {n_with_inject}/{len(out_records)} with inject")
    print(f"[main] total {time.time()-t0:.1f}s")
    print(f"[main] mode={INJECT_MODE}, layers={INJECT_LAYERS}, alpha={INJECT_ALPHA}, batch={GEN_BATCH}")
    # 打印 bias norm 用于 sanity check
    norms = [float(b.norm()) for b in todo_injections if b is not None]
    if norms:
        print(f"[main] bias norm stats: min={min(norms):.1f}, "
              f"max={max(norms):.1f}, mean={sum(norms)/len(norms):.1f}")
    for r in out_records[:3]:
        print(f"\n--- asin={r['asin']} (user={r['user_id']}) ---")
        print(f"  attrs: {r['attrs_used']}")
        print(f"  y_plus_query_inject: {r.get('y_plus_query_inject', '')[:200]}")
    return 0


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
