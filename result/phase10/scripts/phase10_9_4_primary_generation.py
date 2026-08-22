#!/usr/bin/env python3
"""Phase 10.9.4: Primary Generation — 60 dev × Target/Random × 5 candidates.

用户指定要求:
  - 60 dev users (与训练重叠 — 结果只作为 dev, 测试留给 fresh 60 test)
  - 相同商品、prompt、seed、解码参数 (跨 Target/Random 共享)
  - Target-style + Random-style 各 5 候选 = 10 candidates / pair
  - 不添加 prefix noise
  - 不进行 Mahalanobis rerank (直接保存 LLM 原始输出 + 强制 attr 兜底)
  - 统计单位 = 用户或用户×商品 (Phase 10.9.5 处理)

ckpt: phase10_9_2_projector.pt (Counterfactual DPO, epoch 16)

输出:
  - phase10_9_4_candidates_60.jsonl: 每行 1 candidate
    {user_id, asin, cond, candidate_index, candidate_query, candidate_query_raw,
      attrs, attr_pass, needed_append, prompt}
  - phase10_9_4_meta.json: 汇总
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PROJ_CKPT = OUT_DIR / "phase10_9_2_projector.pt"
PAIRS_FILE = OUT_DIR / "phase10_pairs_60.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
DEV_USER_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"

OUT_CANDS = OUT_DIR / "phase10_9_4_candidates_60.jsonl"
OUT_META = OUT_DIR / "phase10_9_4_meta.json"
LOG_OUT = OUT_DIR / "phase10_9_4_primary_generation.log"

# === 硬编码 ===
N_CANDIDATES_PER_COND = 5
CONDITIONS = ["D_target_style", "C_random_style"]   # 顺序: target 在前 (主测量)
# pair.prompts 实际 key 是 'C_target_style' / 'B_random_style' (老命名)
COND_TO_PROMPT_KEY = {
    "D_target_style": "C_target_style",
    "C_random_style": "B_random_style",
}
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.8
TOP_P = 0.95
SEED = 42

ATTR_FIELDS = ["Brand", "Color", "Material"]   # 2026-08-19: 移除数字属性 (Item Weight + Product Dimensions)

SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query "
    "that mentions every listed attribute of the product by its exact value "
    "(brand name, weight, dimensions, color, material). Keep the query under 25 words."
)

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)


def log(m: str) -> None:
    print(f"[phase10-9.4] {m}", flush=True)


def check_attr_pass(query: str, attrs: Dict[str, str]) -> bool:
    """检查 5 attrs 是否都在 query 中 (容忍复数变体 + 数字格式)."""
    import re
    if not query:
        return False
    q_lower = query.lower()
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", q_lower)
    for k, v in attrs.items():
        if not v:
            return False
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        all_tokens = v_clean.split()
        tokens = []
        for t in all_tokens:
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
            elif k == "Brand" and re.match(r"^[a-zA-Z]+$", t) and len(t) >= 2:
                tokens.append(t)
        if not tokens:
            return False
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True; break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True; break
            if (t_lower + "s") in q_clean:
                found = True; break
        if not found:
            return False
    return True


def append_missing_attrs(query: str, attrs: Dict[str, str]) -> str:
    """E30.14 风格: 缺失 attrs 用 clause 追加到 query 末尾."""
    import re
    if check_attr_pass(query, attrs):
        return query
    if not query:
        # 兜底: 构造一个最小 attribute-listing query
        clauses = []
        for k, v in attrs.items():
            if v:
                words = str(v).split()[:4]
                cleaned = " ".join(w.rstrip(":;,.") for w in words).strip()
                if cleaned:
                    clauses.append(f"{k} {cleaned}")
        if clauses:
            return "Looking for " + ", ".join(clauses[:-1]) + f", and {clauses[-1]}."
        return ""
    missing = []
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", query.lower())
    for k, v in attrs.items():
        if not v:
            continue
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        all_tokens = v_clean.split()
        tokens = []
        for t in all_tokens:
            if len(t) >= 3:
                tokens.append(t)
            elif re.match(r"^\d+(\.\d+)?$", t):
                tokens.append(t)
        found = False
        for t in tokens:
            t_lower = t.lower()
            if t_lower in q_clean:
                found = True; break
            if t_lower.endswith("s") and len(t_lower) > 3 and t_lower[:-1] in q_clean:
                found = True; break
            if (t_lower + "s") in q_clean:
                found = True; break
        if not found:
            words = str(v).split()[:4]
            cleaned = " ".join(w.rstrip(":;,.") for w in words).strip()
            if cleaned:
                missing.append(f"{k} {cleaned}")
    if not missing:
        return query
    if len(missing) == 1:
        clause = f"with {missing[0]}"
    elif len(missing) == 2:
        clause = f"with {missing[0]} and {missing[1]}"
    else:
        clause = "with " + ", ".join(missing[:-1]) + f", and {missing[-1]}"
    return query.rstrip(" .") + f", {clause}."


def load_pairs() -> List[dict]:
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def load_user_z() -> Dict[str, np.ndarray]:
    """Load z_u = [mu, logvar] (40d) per user_id."""
    z_map = {}
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            z_map[row["user_id"]] = np.concatenate(
                [row["user_mu"], row["user_logvar"]]
            ).astype(np.float32)
    return z_map


def main():
    log("=" * 70)
    log("Phase 10.9.4: Primary Generation (60 dev × Target/Random × 5)")
    log("=" * 70)

    # === 1. Load pairs + dev users + z map ===
    log("[1] Loading pairs / dev users / user z ...")
    all_pairs = load_pairs()
    dev_ids = sorted(json.loads(DEV_USER_FILE.read_text())["user_ids"])
    dev_set = set(dev_ids)
    log(f"  all pairs: {len(all_pairs)}, dev users: {len(dev_ids)}")

    # 只取 dev 用户的 pairs (Phase 10.8.1 锁定的 60 用户)
    pairs = [p for p in all_pairs if p["user_id"] in dev_set]
    log(f"  dev-filtered pairs: {len(pairs)}")

    # 加载 z 表
    z_map = load_user_z()
    missing_z = [p["user_id"] for p in pairs if p["user_id"] not in z_map]
    missing_r = [p["random_user_id"] for p in pairs if p["random_user_id"] not in z_map]
    if missing_z or missing_r:
        raise RuntimeError(f"Missing z for users: target={len(missing_z)} random={len(missing_r)}")
    log(f"  z_map loaded: {len(z_map)} users")

    # === 2. Load projector ===
    log("[2] Loading projector ...")
    if not PROJ_CKPT.exists():
        raise FileNotFoundError(f"{PROJ_CKPT} 不存在; 请先运行 phase10_9_2_train_counterfactual.py")
    ckpt = torch.load(PROJ_CKPT, map_location="cpu", weights_only=False)
    from query.soft_prefix.projector import SoftPrefixProjector

    projector = SoftPrefixProjector(
        user_dim=ckpt["user_dim"], hidden_dim=ckpt["hidden_dim"],
        num_tokens=ckpt["num_tokens"], model_dim=ckpt["model_dim"],
        dtype=torch.float32, gate_init=ckpt["gate_init"],
    )
    projector.load_state_dict(ckpt["state_dict"])
    projector.eval()
    dpo_a = ckpt.get('dpo_loss_a') or 0.0
    epoch = ckpt.get('epoch', '?')
    log(f"  projector loaded (epoch={epoch}, dpo_a={dpo_a:.4f})")

    # === 3. Load Qwen ===
    log("[3] Loading Qwen2-7B (frozen) via _HiddenBackend ...")
    from llm_client import _HiddenBackend
    try:
        _HiddenBackend.reset()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    backend = _HiddenBackend.get(QWEN_MODEL_PATH)
    model = backend.model
    tokenizer = backend.tokenizer
    DEVICE = next(model.parameters()).device
    projector = projector.to(DEVICE)
    log(f"  Qwen on {DEVICE}, dtype={backend.dtype}")

    emb_m = model.get_input_embeddings()
    NUM_TOKENS = ckpt["num_tokens"]

    # === 4. Generation function (batched: B = N_CANDIDATES_PER_COND × N_PROMPTS) ===
    @torch.no_grad()
    def generate_batch(prompts: List[str], prefix: torch.Tensor) -> List[str]:
        """prefix: [B, K, H]. prompts: len B. → List[str] len B.

        同一 (user, asin) 下: 5 candidates 共享同一 prefix (target or random),
        只是不同的 sampling (model.generate 内置 sampling diversity 由 T=0.8 控制).
        """
        B = len(prompts)
        assert prefix.size(0) == B, f"prefix B={prefix.size(0)} != prompts B={B}"
        chat_prompts = []
        for p in prompts:
            cp = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p},
                ],
                tokenize=False, add_generation_prompt=True,
            )
            chat_prompts.append(cp)
        prompt_ids_list = [tokenizer.encode(cp, add_special_tokens=False) for cp in chat_prompts]
        max_p = max(len(x) for x in prompt_ids_list)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.full((B, max_p), pad_id, dtype=torch.long, device=DEVICE)
        am = torch.zeros((B, max_p), dtype=torch.long, device=DEVICE)
        for b, pl in enumerate(prompt_ids_list):
            ids[b, max_p - len(pl):] = torch.tensor(pl, dtype=torch.long, device=DEVICE)
            am[b, max_p - len(pl):] = 1

        text_emb = emb_m(ids).to(model.dtype)
        full_emb = torch.cat([prefix.to(model.dtype), text_emb], dim=1)
        full_am = torch.cat(
            [torch.ones((B, NUM_TOKENS), dtype=torch.long, device=DEVICE), am], dim=1,
        )
        out = model.generate(
            inputs_embeds=full_emb,
            attention_mask=full_am,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = []
        for b in range(B):
            text = tokenizer.decode(out[b].tolist(), skip_special_tokens=True).strip()
            text = text.split("\n")[0]
            decoded.append(text)
        return decoded

    # === 5. Main loop ===
    # 每个 (user, asin, cond): prefix = projector(z_target) or projector(z_random)
    # 5 candidates 共享同一 prefix, prompts 相同 (含 target_style_desc or random_style_desc)
    # 同 cond 5 candidate 一次 forward (B=5)
    log(f"[4] Generating: {len(pairs)} pairs × {len(CONDITIONS)} conds × {N_CANDIDATES_PER_COND} = "
        f"{len(pairs) * len(CONDITIONS) * N_CANDIDATES_PER_COND} candidates")
    OUT_CANDS.parent.mkdir(parents=True, exist_ok=True)
    fout = OUT_CANDS.open("w")
    n_total = 0
    n_attr_pass = 0
    n_needed_append = 0
    t0 = time.time()

    for pi, pair in enumerate(pairs):
        if pi % 5 == 0:
            log(f"  [{pi+1}/{len(pairs)}] user={pair['user_id'][:10]} asin={pair['asin']} "
                f"({time.time()-t0:.1f}s)")

        attrs = pair["attrs"]
        z_target = z_map[pair["user_id"]]
        z_random = z_map[pair["random_user_id"]]

        # 准备 prefix (target + random), 每 cond 一份
        cond_z = {
            "D_target_style": z_target,
            "C_random_style": z_random,
        }

        for cond in CONDITIONS:
            z = cond_z[cond]
            # prefix: 1 个 → 复制 5 份 (同 prefix, 不同 sampling)
            z_t = torch.as_tensor(z, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prefix_one = projector(z_t).to(model.dtype)  # [1, K, H]
            prefix_b = prefix_one.repeat(N_CANDIDATES_PER_COND, 1, 1)  # [5, K, H]

            # prompt: 用对应 cond 的 prompt (含 target_style_desc / random_style_desc)
            prompt_key = COND_TO_PROMPT_KEY[cond]
            prompt_template = pair["prompts"][prompt_key]

            # 5 candidate 的 prompts (内容一致, 只靠 sampling diversity)
            prompts = [prompt_template] * N_CANDIDATES_PER_COND

            try:
                outputs = generate_batch(prompts, prefix_b)
            except Exception as e:
                log(f"    [WARN] generate failed user={pair['user_id'][:10]} cond={cond}: {e}")
                outputs = [""] * N_CANDIDATES_PER_COND

            for ci, raw_query in enumerate(outputs):
                raw_query = (raw_query or "").strip().strip('"').strip("'")
                attr_pass_raw = check_attr_pass(raw_query, attrs)
                query_kept = append_missing_attrs(raw_query, attrs)
                attr_pass = check_attr_pass(query_kept, attrs)
                n_total += 1
                if attr_pass:
                    n_attr_pass += 1
                if not attr_pass_raw:
                    n_needed_append += 1

                fout.write(json.dumps({
                    "user_id": pair["user_id"],
                    "asin": pair["asin"],
                    "cond": cond,
                    "candidate_index": ci,
                    "candidate_query": query_kept,
                    "candidate_query_raw": raw_query,
                    "attrs": attrs,
                    "attr_pass": attr_pass,
                    "attr_pass_raw": attr_pass_raw,
                    "needed_append": not attr_pass_raw,
                    "prompt_used": prompt_template[:200],  # 只存前 200 char (避免 json 过大)
                    "z_target_norm": float(np.linalg.norm(z_target)),
                    "z_random_norm": float(np.linalg.norm(z_random)),
                }, ensure_ascii=False) + "\n")

        # 每 10 pair 主动 gc + empty_cache (防止 GPU OOM)
        if (pi + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fout.close()
    log(f"  done in {time.time()-t0:.1f}s")

    # === 6. Meta ===
    meta = {
        "n_pairs": len(pairs),
        "n_conditions": len(CONDITIONS),
        "n_candidates_per_cond": N_CANDIDATES_PER_COND,
        "n_total_candidates": n_total,
        "n_attr_pass": n_attr_pass,
        "attr_pass_rate": n_attr_pass / n_total if n_total else 0,
        "n_needed_append": n_needed_append,
        "needed_append_rate": n_needed_append / n_total if n_total else 0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "proj_ckpt": str(PROJ_CKPT),
        "conditions": CONDITIONS,
        "note": "60 dev × Target/Random × 5 — 无 prefix noise, 无 rerank. "
                "Phase 10.9.5 在这上面做 scale-free 评估 (rank diff / win rate / percentile / semantic sim / attr).",
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {OUT_META}")

    log("=" * 70)
    log(f"汇总: total={n_total}, attr_pass={n_attr_pass} ({meta['attr_pass_rate']:.2%}), "
        f"needed_append={n_needed_append}")
    log("=" * 70)


if __name__ == "__main__":
    main()