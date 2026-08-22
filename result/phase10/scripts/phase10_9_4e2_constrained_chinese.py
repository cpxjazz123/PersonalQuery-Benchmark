#!/usr/bin/env python3
"""Phase 10.9.4e: Constrained Chinese-Ban Generation.

设计动机 (diag_chinese_origin.py 验证):
  - 100% 中文来自 raw LLM 生成 (APPEND_INTRODUCED = 0)
  - D_target vs C_random 中文率几乎相同 (16.67% vs 17.00%) → prefix 不是元凶
  - attrs / prompt_used 都 0 中文 → 上下文已全英文
  - prompt 工程 (v1=42% → v4=17%) 边际收益接近 0, 残余是 Qwen2-7B-Instruct
    base 模型的多语言先验; 20d VADES 句法特征不含语言脚本维度.

修复策略 (constrained decoding):
  - 在 mixed_logits 后 + sampling 前, 把所有 CJK token id 的 logits 设为 -inf
  - Qwen 物理上无法在 decode 时选中 CJK token → 100% 没有中文
  - 这是 hard constraint, 不依赖 prompt / 用户风格 / 属性构造

实现:
  - 扩展 CopyAwareGenerator.generate_batch: 新增 cjk_mask: Optional[torch.Tensor]
    参数 (None = 无约束, backward compatible)
  - 一次性扫描 tokenizer vocab, 标记所有 decoded 含 CJK 的 token id
  - generation loop 同 phase10_9_4d, 但 generate_batch 传入 cjk_mask

数据:
  - 60 dev users × (D_target_style + C_random_style) × 5 candidates = 600 candidates
  - 3 attrs (Brand / Color / Material), ATTR_FIELDS 与 4d 对齐

ckpt:
  - phase10_9_2_projector.pt (DPO, frozen)
  - phase10_9_4c_copyhead.pt (新训练, frozen)

输出:
  - phase10_9_4e2_candidates_60.jsonl
  - phase10_9_4e2_meta.json
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
COPY_CKPT = OUT_DIR / "phase10_9_4f2_copyhead.pt"
PAIRS_FILE = OUT_DIR / "phase10_pairs_60.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
DEV_USER_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"

OUT_CANDS = OUT_DIR / "phase10_9_4e2_candidates_60.jsonl"
OUT_META = OUT_DIR / "phase10_9_4e2_meta.json"
LOG_OUT = OUT_DIR / "phase10_9_4e2_constrained_chinese.log"

# === 硬编码 (与 4d 对齐) ===
ATTR_FIELDS = ["Brand", "Color", "Material"]
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

# CJK ranges: Hiragana + Katakana + CJK Unified Ideographs + Hangul + CJK Symbols
CJK_RE = re.compile(r"[　-〿぀-ゟ゠-ヿ一-鿿가-힯]")


def log(m):
    print(f"[phase10-9.4e2] {m}", flush=True)


def build_cjk_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """一次性扫描 vocab, 标记所有 raw BPE bytes 含 non-ASCII (>= 0x80) 的 token id.

    设计动机 (修复 v1 漏检):
      Qwen2 BPE 是 byte-level. '白' 单独 encode → token 99243 (raw BPE = 'çĻ½',
      decoded = '白'). 但 '白' 在长序列里被 BPE 拆成 [68294, 121]: 单个 decode
      是 ' �' / '�' (含 U+FFFD replacement char), 不在 CJK_RE 范围 — 漏检.

    正确方法: raw BPE token 的每个 char 是 1 字节 (GPT-2 byte encoder), 通过
    byte_decoder 反查字节 → 判断是否含 >= 0x80 字节. 这会 ban 掉:
      - 完整 CJK/Hiragana/Katakana/Hangul token
      - 跨 token 拆分的 partial CJK (关键的遗漏修复)
      - 副作用: accented Latin / emoji / Cyrillic / Arabic 等都被 ban
        (英文 shopping query 不需要这些, 安全起见直接 ban)

    返回 ([V] bool tensor, n_banned).
    """
    bpe = tokenizer.backend_tokenizer
    vocab = bpe.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}

    # GPT-2 byte-level BPE byte_decoder: 把 byte 0xXX 映射到 unicode chr(256 + n)
    # 反向: byte_decoder[chr(c)] = c
    printable = (
        list(range(ord('!'), ord('~') + 1))
        + list(range(ord('¡'), ord('¬') + 1))
        + list(range(ord('®'), ord('ÿ') + 1))
    )
    byte_decoder = {}
    n = 0
    for b in range(256):
        if b not in printable:
            byte_decoder[chr(256 + n)] = b
            n += 1
    for c in printable:
        byte_decoder[chr(c)] = c

    cjk_mask = torch.zeros(vocab_size, dtype=torch.bool)
    n_banned = 0
    for tid in range(vocab_size):
        raw = inv_vocab.get(tid)
        if raw is None:
            continue
        raw_bytes = bytes([byte_decoder.get(c, ord(c)) for c in raw])
        if any(b >= 0x80 for b in raw_bytes):
            cjk_mask[tid] = True
            n_banned += 1
    return cjk_mask, n_banned


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
    """DEPRECATED — 用过 fallback append. 项目硬规则 (Rule 7 / AGENTS.md) 禁止 fallback.

    保留函数签名仅为兼容 (返回原 query), 实际不再调用. 真正的 raw 不合规
    candidate 应该让 generator 自然产出, 而非被后处理掩盖.
    """
    return query


def filter_attrs(attrs_5: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in attrs_5.items() if k in ATTR_FIELDS and v}


def main():
    log("=" * 70)
    log("Phase 10.9.4e: CopyAware + CJK-Constrained Generation")
    log("=" * 70)

    # === 1. Load pairs / dev users / z_map ===
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
    projector = projector.to(model.dtype)
    for p in projector.parameters():
        p.requires_grad = False

    copy_head = CopyAwareHead(MODEL_DIM, VOCAB_SIZE, dtype=torch.bfloat16).to(DEVICE)
    copy_head.load_state_dict(torch.load(COPY_CKPT, map_location=DEVICE, weights_only=False))
    copy_head.eval()
    for p in copy_head.parameters():
        p.requires_grad = False

    NUM_TOKENS = proj_ckpt["num_tokens"]
    log(f"  projector + copy_head loaded, K={NUM_TOKENS}")

    # === 3. Build CJK token mask ===
    log("[3] Building CJK token mask (one-shot vocab scan) ...")
    cjk_mask, n_banned = build_cjk_token_mask(tokenizer, VOCAB_SIZE)
    cjk_mask = cjk_mask.to(DEVICE)
    log(f"  CJK banned tokens: {n_banned} / {VOCAB_SIZE} ({n_banned / VOCAB_SIZE:.2%})")

    # === 4. CopyAwareGenerator ===
    log("[4] Constructing CopyAwareGenerator ...")
    generator = CopyAwareGenerator(model, projector, copy_head, device=DEVICE)
    eos_id = tokenizer.eos_token_id

    # === 5. Generation loop ===
    log(f"[5] Generating: {len(pairs)} pairs × {len(CONDITIONS)} conds × {N_CANDIDATES_PER_COND} "
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
            z_in = z
            prompt_key = COND_TO_PROMPT_KEY[cond]

            def _build_lines_3(a):
                lines = ["Product attributes:"]
                for key in sorted(a.keys()):
                    lines.append(f"{key}: {a[key]}")
                lines.append("Write a natural shopping query that mentions every attribute.")
                return "\n".join(lines)

            attr_prompt_body = _build_lines_3(attrs_3)
            original_prompt = pair["prompts"][prompt_key]
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
                    cjk_mask=cjk_mask,        # <-- KEY: hard constraint
                )
            except Exception as e:
                log(f"    [WARN] generate failed user={pair['user_id'][:10]} cond={cond}: {e}")
                outputs = [""] * N_CANDIDATES_PER_COND

            for ci, raw_query in enumerate(outputs):
                raw_query = (raw_query or "").strip().strip('"').strip("'")
                raw_query = raw_query.split("\n")[0]
                # NO FALLBACK: candidate_query = raw_query, attr_pass = raw pass check
                attr_pass = check_attr_pass(raw_query, attrs_3)
                n_total += 1
                if attr_pass:
                    n_attr_pass += 1
                if not attr_pass:
                    n_needed_append += 1

                fout.write(json.dumps({
                    "user_id": pair["user_id"],
                    "asin": pair["asin"],
                    "cond": cond,
                    "candidate_index": ci,
                    "candidate_query": raw_query,
                    "candidate_query_raw": raw_query,
                    "attrs": attrs_3,
                    "attrs_5": attrs_5,
                    "attr_pass": attr_pass,
                    "needed_append": not attr_pass,
                    "prompt_used": user_content[:200],
                    "z_target_norm": float(np.linalg.norm(z_target)),
                    "z_random_norm": float(np.linalg.norm(z_random)),
                }, ensure_ascii=False) + "\n")

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
        "top_k": TOP_K,
        "seed": SEED,
        "proj_ckpt": str(PROJ_CKPT),
        "copy_ckpt": str(COPY_CKPT),
        "conditions": CONDITIONS,
        "attr_fields": ATTR_FIELDS,
        "method": "CopyAwareGenerator + non-ASCII hard mask (raw BPE byte-level) — NO FALLBACK",
        "n_cjk_banned_tokens": n_banned,
        "vocab_size": VOCAB_SIZE,
        "cjk_banned_rate": n_banned / VOCAB_SIZE,
        "fallback_appended": False,
        "note": "60 dev × Target/Random × 5 — CopyAwareGenerator + cjk_mask 参数. "
                "在 mixed_logits 后 + sampling 前把 CJK token logits 设为 -inf. "
                "Phase 10.9.5c 在这上面做 scale-free 评估.",
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {OUT_META}")

    # 自检: 中文率 + raw 属性率 (no fallback)
    n_zh = 0
    with OUT_CANDS.open() as f:
        for line in f:
            r = json.loads(line)
            if CJK_RE.search(r["candidate_query"]):
                n_zh += 1
    log(f"  self-check: {n_zh}/{n_total} = {n_zh / n_total:.2%} candidates contain CJK")

    log("=" * 70)
    log(f"汇总: total={n_total}, attr_pass={n_attr_pass} ({meta['attr_pass_rate']:.2%}), "
        f"needed_append={n_needed_append}, cjk_banned={n_banned}")
    log("=" * 70)


if __name__ == "__main__":
    main()
