#!/usr/bin/env python3
"""Phase 10.9.4d: Copy-Aware Primary Generation.

设计:
  - 加载 phase10_9_2_projector.pt (frozen, 已 DPO 训练好) + phase10_9_4c_copyhead.pt (新训练)
  - 用 CopyAwareGenerator (projector + copy head + Qwen) 跑 batched generate_batch
  - 3 attrs (Brand/Color/Material, 不含数字属性)
  - 60 dev × (D_target_style + C_random_style) × 5 candidates = 600 candidates
  - 不 prefix noise, 不 Mahalanobis rerank

ckpt:
  - phase10_9_2_projector.pt (DPO, epoch 16, 40d → 8 tokens × 3584d)
  - phase10_9_4c_copyhead.pt (新训练, 3 epochs on 54 chosen_query)

输出:
  - phase10_9_4d_candidates_60.jsonl
  - phase10_9_4d_meta.json
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PROJ_CKPT = OUT_DIR / "phase10_9_2_projector.pt"
COPY_CKPT = OUT_DIR / "phase10_9_4c_copyhead.pt"
PAIRS_FILE = OUT_DIR / "phase10_pairs_60.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
DEV_USER_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"

OUT_CANDS = OUT_DIR / "phase10_9_4d_candidates_60.jsonl"
OUT_META = OUT_DIR / "phase10_9_4d_meta.json"
LOG_OUT = OUT_DIR / "phase10_9_4d_copyaware_generation.log"

# === 硬编码 ===
ATTR_FIELDS = ["Brand", "Color", "Material"]   # 2026-08-19: 移除数字属性
N_CANDIDATES_PER_COND = 5
CONDITIONS = ["D_target_style", "C_random_style"]
COND_TO_PROMPT_KEY = {
    "D_target_style": "C_target_style",
    "C_random_style": "B_random_style",
}
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 20
SEED = 42

SYSTEM_PROMPT = (
    "You are a shopping query writer. Write one short natural shopping query. "
    "OUTPUT LANGUAGE: ENGLISH ONLY. "
    "NO Chinese characters. NO Japanese characters. NO Korean characters. "
    "All words in the query must be in English. "
    "Mention every listed attribute of the product by its exact value "
    "(brand name, color, material). Keep the query under 25 words. "
    "Use only Latin alphabet letters, digits, spaces, and standard punctuation."
)

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)


def log(m):
    print(f"[phase10-9.4d] {m}", flush=True)


def check_attr_pass(query: str, attrs: Dict[str, str]) -> bool:
    if not query or not attrs:
        return False
    q_lower = query.lower()
    q_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", q_lower)
    for k, v in attrs.items():
        if not v:
            return False
        v_clean = re.sub(r"[^a-zA-Z0-9./\- ]", " ", str(v)).strip()
        tokens = []
        for t in v_clean.split():
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
    if check_attr_pass(query, attrs):
        return query
    if not query:
        clauses = []
        for k, v in attrs.items():
            if v:
                words = str(v).split()[:3]
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
        tokens = []
        for t in v_clean.split():
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
            words = str(v).split()[:3]
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


def filter_attrs(attrs_5: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in attrs_5.items() if k in ATTR_FIELDS and v}


def main():
    log("=" * 70)
    log("Phase 10.9.4d: CopyAware Primary Generation (3 attrs)")
    log("=" * 70)

    # === 1. Load pairs + z_map ===
    log("[1] Loading pairs / dev users / z_map ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    dev_ids = sorted(json.loads(DEV_USER_FILE.read_text())["user_ids"])
    dev_set = set(dev_ids)
    pairs = [p for p in pairs if p["user_id"] in dev_set]
    log(f"  dev-filtered pairs: {len(pairs)}")

    z_map = {}
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            z_map[row["user_id"]] = np.concatenate([row["user_mu"], row["user_logvar"]]).astype(np.float32)
    log(f"  z_map: {len(z_map)} users")

    # === 2. Load projector + copy head + Qwen ===
    log("[2] Loading projector (frozen) + copy head (frozen) + Qwen ...")
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
    VOCAB_SIZE = model.config.vocab_size
    MODEL_DIM = model.config.hidden_size
    log(f"  Qwen on {DEVICE}, dim={MODEL_DIM}, vocab={VOCAB_SIZE}")

    from query.soft_prefix.projector import SoftPrefixProjector
    from query.soft_prefix.copy_aware import CopyAwareHead
    from query.soft_prefix.copy_aware_generate import CopyAwareGenerator

    proj_ckpt = torch.load(PROJ_CKPT, map_location=DEVICE, weights_only=False)
    projector = SoftPrefixProjector(
        user_dim=proj_ckpt["user_dim"], hidden_dim=proj_ckpt["hidden_dim"],
        num_tokens=proj_ckpt["num_tokens"], model_dim=proj_ckpt["model_dim"],
        dtype=torch.float32, gate_init=proj_ckpt["gate_init"],
    ).to(DEVICE)
    projector.load_state_dict(proj_ckpt["state_dict"])
    projector.eval()
    # 把 projector 转成 model dtype (bf16), 否则 copy_aware_generate.generate_batch 内的
    # self.projector(z.to(target_dtype)) 会 dtype mismatch
    projector = projector.to(model.dtype)
    for p in projector.parameters():
        p.requires_grad = False

    copy_head = CopyAwareHead(MODEL_DIM, VOCAB_SIZE, dtype=torch.bfloat16).to(DEVICE)
    copy_head.load_state_dict(torch.load(COPY_CKPT, map_location=DEVICE, weights_only=False))
    copy_head.eval()
    for p in copy_head.parameters():
        p.requires_grad = False

    NUM_TOKENS = proj_ckpt["num_tokens"]
    log(f"  projector (frozen) + copy_head (frozen), K={NUM_TOKENS}")

    # === 3. CopyAwareGenerator ===
    log("[3] Constructing CopyAwareGenerator ...")
    generator = CopyAwareGenerator(model, projector, copy_head, device=DEVICE)
    eos_id = tokenizer.eos_token_id

    # === 4. Generation loop ===
    log(f"[4] Generating: {len(pairs)} pairs × {len(CONDITIONS)} conds × {N_CANDIDATES_PER_COND} "
        f"= {len(pairs) * len(CONDITIONS) * N_CANDIDATES_PER_COND} candidates")

    OUT_CANDS.parent.mkdir(parents=True, exist_ok=True)
    fout = OUT_CANDS.open("w")
    n_total = 0
    n_attr_pass = 0
    n_needed_append = 0
    t0 = time.time()

    for pi, pair in enumerate(pairs):
        if pi % 10 == 0:
            log(f"  [{pi+1}/{len(pairs)}] user={pair['user_id'][:10]} asin={pair['asin']} "
                f"({time.time()-t0:.1f}s)")
        attrs_5 = pair["attrs"]
        attrs_3 = filter_attrs(attrs_5)
        if len(attrs_3) < 2:
            log(f"    [WARN] user {pair['user_id'][:10]} attrs_3 too sparse, skip")
            continue
        z_target = z_map[pair["user_id"]]
        z_random = z_map[pair["random_user_id"]]

        cond_z = {
            "D_target_style": z_target,
            "C_random_style": z_random,
        }

        for cond in CONDITIONS:
            z = cond_z[cond]
            z_in = z  # 完整 40d vector
            prompt_key = COND_TO_PROMPT_KEY[cond]
            # 用 pair 里原 prompt (含 style_desc), 但 attrs 替换为 3 字段
            # pair.prompts[cond] 是固定 5 字段 prompt, 我们重新构造 3 字段版本
            from query.soft_prefix.copy_aware import build_attr_prompt_lines
            # build_attr_prompt_lines 假设 key 是 "A1/A2" 格式 (int(k[1:]))
            # 我们的 key 是 "Brand/Color/Material", 用 wrapper 避免排序失败
            def _build_lines_3(a):
                lines = ["Product attributes:"]
                for key in sorted(a.keys()):
                    lines.append(f"{key}: {a[key]}")
                lines.append("Write a natural shopping query that mentions every attribute.")
                return "\n".join(lines)
            attr_prompt_body = _build_lines_3(attrs_3)
            # 嵌入原 style desc: 把它加在 attrs 后
            original_prompt = pair["prompts"][prompt_key]
            # 原 prompt 形如: "Product attributes:\n{5attrs}\n\n{style_desc}\n\nWrite..."
            # 简化: 取原 prompt 的 style_desc 部分, 重新拼接
            style_match = re.search(r"\n\n(Your query should match.*?)Write a short", original_prompt, re.DOTALL)
            if style_match:
                style_desc_block = "\n\n" + style_match.group(1) + "\n\n"
            else:
                style_desc_block = "\n\n"
            user_content = (attr_prompt_body + style_desc_block
                            + "IMPORTANT: Write a short shopping query in ENGLISH ONLY. "
                            + "Do NOT use any Chinese, Japanese, or Korean characters. "
                            + "Use English words only. The query must include every attribute listed above.")
            chat_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False, add_generation_prompt=True,
            )

            # CopyAwareGenerator.generate_batch: 单 batch 共享同一 prefix
            # 这里 5 candidate 共享 prefix, prompt 一致 (靠 sampling diversity)
            # 但 generate_batch 内 bsz = len(prompts) → 需要 5 个 prompt
            # copy head 会在不同 sample 上学不同 attrs → 但 attrs 相同, 所以 prompt 相同
            prompts = [chat_prompt] * N_CANDIDATES_PER_COND
            user_vecs = [z_in] * N_CANDIDATES_PER_COND
            attrs_list = [attrs_3] * N_CANDIDATES_PER_COND

            try:
                outputs = generator.generate_batch(
                    prompt_strs=prompts,
                    attrs_list=attrs_list,
                    user_vecs=user_vecs,
                    tokenizer=tokenizer,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=eos_id,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                )
            except Exception as e:
                log(f"    [WARN] generate failed user={pair['user_id'][:10]} cond={cond}: {e}")
                outputs = [""] * N_CANDIDATES_PER_COND

            for ci, raw_query in enumerate(outputs):
                raw_query = (raw_query or "").strip().strip('"').strip("'")
                raw_query = raw_query.split("\n")[0]
                # check 用 3 字段 attrs
                attr_pass_raw = check_attr_pass(raw_query, attrs_3)
                query_kept = append_missing_attrs(raw_query, attrs_3)
                attr_pass = check_attr_pass(query_kept, attrs_3)
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
                    "attrs": attrs_3,
                    "attrs_5": attrs_5,
                    "attr_pass": attr_pass,
                    "attr_pass_raw": attr_pass_raw,
                    "needed_append": not attr_pass_raw,
                    "prompt_used": user_content[:200],
                    "z_target_norm": float(np.linalg.norm(z_target)),
                    "z_random_norm": float(np.linalg.norm(z_random)),
                }, ensure_ascii=False) + "\n")

        # 每 10 pair 主动 gc + empty_cache
        if (pi + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fout.close()
    log(f"  done in {time.time()-t0:.1f}s")

    # === 5. Meta ===
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
        "top_k": TOP_K,
        "seed": SEED,
        "proj_ckpt": str(PROJ_CKPT),
        "copy_ckpt": str(COPY_CKPT),
        "conditions": CONDITIONS,
        "attr_fields": ATTR_FIELDS,
        "method": "CopyAwareGenerator (projector + copy head + Qwen)",
        "note": "60 dev × Target/Random × 5 — CopyAwareGenerator, 3 attrs (no numeric). "
                "Phase 10.9.5b 在这上面做 scale-free 评估.",
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {OUT_META}")
    log("=" * 70)
    log(f"汇总: total={n_total}, attr_pass={n_attr_pass} ({meta['attr_pass_rate']:.2%}), "
        f"needed_append={n_needed_append}")
    log("=" * 70)


if __name__ == "__main__":
    main()