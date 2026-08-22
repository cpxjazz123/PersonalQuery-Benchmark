#!/usr/bin/env python3
"""Phase 10.9.4i: Hard Pointer Action + Coverage Constraint (CLEAN VERSION).

设计动机:
  - 4h (5952 records): soft sigmoid gate ptr loss stuck ~0.69 (sigmoid 0.5)
  - 4f / 4f2 (60 records): same issue
  - 结论: 软概率 gate 不是正确归纳偏置

新架构: Hard Pointer Action + Coverage Constraint (NO soft sigmoid gate)
  - LM 自由生成普通文本 (受 user style prefix 控制, P7 保证不变)
  - coverage state: per attr key 标记是否已 copy
  - 末期硬保证: 当 step >= max_new_tokens - SAFETY_STEPS 且 coverage 未全 True:
    强制 COPY 剩余未覆盖 attrs (从 src verbatim emit)
  - 早 EOS 防护: coverage 未全 True 时禁 EOS (model 不能中途停止)
  - LM 自然 emit attribute 也会被 detection 接管 (保证 coverage 准确)
  - 完全不用 sigmoid gate / soft probability

实现:
  - 完全复用 projector + Qwen
  - 不需要 CopyAwareHead (gate 删除)
  - 单条 generate loop (slow), correctness 100%

数据: 同 4e (60 dev × Target/Random × 5 candidates = 600)
ckpt:
  - phase10_9_2_projector.pt (frozen)

输出:
  - phase10_9_4i2_candidates_60.jsonl
  - phase10_9_4i2_meta.json
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"  # Phase 10.10 scale-up
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
DEV_USER_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"

OUT_CANDS = OUT_DIR / "phase10_10_candidates_n10.jsonl"
OUT_META = OUT_DIR / "phase10_10_n10_meta.json"
LOG_OUT = OUT_DIR / "phase10_10_1_generate_n10.log"

# === 硬编码 ===
ATTR_FIELDS = ["Brand", "Color", "Material"]
N_CANDIDATES_PER_COND = 10  # Phase 10.10: 10 candidates (rerank pool)
# Phase 10.10: NO style prefix at all. Single generation pool per pair.
# We only generate A_no_style, then Phase 10.10.2 will rerank by maha_target.
CONDITIONS = ["A_no_style"]
COND_TO_PROMPT_KEY = {
    "A_no_style": "A_no_style",
}
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 20
SEED = 42
# 末期硬保证触发: 当剩余 step 数 <= SAFETY_STEPS 且 coverage 未全 True → 强制 emit
SAFETY_STEPS = 8

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

CJK_RE = re.compile(r"[　-〿぀-ゟ゠-ヿ一-鿿가-힯]")


def log(m):
    print(f"[phase10-10.1] {m}", flush=True)


def build_cjk_token_mask(tokenizer, vocab_size: int):
    bpe = tokenizer.backend_tokenizer
    vocab = bpe.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}

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


def filter_attrs(attrs_5):
    return {k: v for k, v in attrs_5.items() if k in ATTR_FIELDS and v}


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


def prepare_attr_keys(attrs: Dict[str, str], tokenizer):
    """Return ordered list of attr token sequences.

    Returns:
      attr_keys: List[List[int]] — token ids for each attr value (verbatim from prompt vocab)
      attr_str_keys: List[str] — "Brand" / "Color" / "Material" (key names)
      attr_str_values: List[str] — original attr value strings (e.g. "Munchkin", "Stainless Steel")
    """
    attr_keys = []
    attr_str_keys = []
    attr_str_values = []
    for key in ["Brand", "Color", "Material"]:
        v = attrs.get(key)
        if not v:
            continue
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        if ids:
            attr_keys.append(ids)
            attr_str_keys.append(key)
            attr_str_values.append(str(v))
    return attr_keys, attr_str_keys, attr_str_values


@torch.no_grad()
def hard_pointer_generate_batch(
    model, tokenizer,
    prompt_str: str, attrs: Dict[str, str],
    n_candidates: int,
    device,
    max_new_tokens: int, eos_token_id: int,
    temperature: float, top_k: int,
    cjk_mask: torch.Tensor,
    projector=None, user_vec: np.ndarray = None, num_tokens: int = 0,
) -> List[str]:
    """Batched hard pointer action + coverage constraint decoder (Phase 10.10: NO soft prefix).

    同时生成 n_candidates 个候选 (共享同一个 prompt), 速度比逐条快 ~5x.

    Phase 10.10: 去掉 projector/user_vec/num_tokens, 纯自由生成.
    核心原则:
      1. LM 自由生成普通文本 (no style prefix)
      2. coverage state 跟踪每 attr 是否已 copy (per sample)
      3. 自然 emit trigger: 当 LM emit 某 attr 第一 token → 自动 push 剩余 token 到 forced queue
      4. 末期硬保证: 剩余 step <= SAFETY_STEPS 时强制 COPY 剩余 attrs
      5. 早 EOS 防护: coverage 未全 True 时禁 EOS
    """
    target_dtype = next(model.parameters()).dtype
    B = n_candidates

    # 1. encode prompt (共享)
    prompt_ids_1d = tokenizer.encode(prompt_str, add_special_tokens=False)
    prompt_len = len(prompt_ids_1d)
    # batch repeat
    prompt_ids = torch.tensor([prompt_ids_1d] * B, dtype=torch.long, device=device)

    # 2. 准备 attr token sequences (verbatim from tokenizer vocab)
    attr_keys, attr_str_keys, attr_str_values = prepare_attr_keys(attrs, tokenizer)
    n_attrs = len(attr_keys)

    # 3. 准备 prefix + first forward (Phase 10.10: NO soft prefix)
    emb = model.get_input_embeddings()
    text_emb = emb(prompt_ids).to(target_dtype)
    # Phase 10.10: skip soft prefix entirely (post-hoc rerank paradigm)
    full_emb = text_emb
    attention_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=device)
    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0).expand(B, -1)
    # num_tokens forced to 0 for compatibility with later code
    num_tokens = 0

    out = model(
        inputs_embeds=full_emb, attention_mask=attention_mask,
        position_ids=position_ids, use_cache=True,
        past_key_values=None,
    )
    past = out.past_key_values
    # first_gen_logits: [B, V] — 取每行最后有效位置 (right-padding 时)
    # 但这里所有 sample 同 prompt_len, 无 padding, 所以直接取 -1
    first_gen_logits = out.logits[:, -1]  # [B, V]

    # 4. per-sample generate state
    coverage = [[False] * n_attrs for _ in range(B)]  # coverage[b][k]
    generated_ids = [[] for _ in range(B)]  # generated_ids[b] = List[int]
    forced_copy_queue = [[] for _ in range(B)]  # forced_copy_queue[b] = List[int]

    def _sample_token(logits_b, b):
        """GEN action: sample for sample b with CJK + EOS + temperature + top_k."""
        m = logits_b.clone().float()
        if cjk_mask is not None:
            m = m.masked_fill(cjk_mask.to(m.device, dtype=torch.bool), float("-inf"))
        # coverage 未满时禁 EOS
        if not all(coverage[b]):
            m[eos_token_id] = float("-inf")
        if temperature > 0:
            m = m / max(temperature, 1e-4)
            if top_k > 0:
                v = torch.topk(m, min(top_k, m.size(-1))).values[-1]
                m = m.clone(); m[m < v] = float("-inf")
            return int(torch.multinomial(m.softmax(dim=-1), 1).item())
        return int(torch.argmax(m))

    def _check_trigger(b, last_id):
        """Per-sample string-level trigger (more precise than token-level).

        改进:
          1. 累积 generated_ids[b] 后用 tokenizer.decode 取 last N tokens → 拼成 text
          2. 对每个未 covered attr_k, 计算 attr value 在累积 text 中的最长公共前缀 (LCP)
          3. case A: LCP == len(attr_value) → 完全 emit, mark coverage=True
          4. case B: 0 < LCP < len(attr_value) → LM 在 emit attr 过程中, push 剩余 token 到 forced queue
          5. case C: LCP == 0 → 不触发
        """
        # case A & B: 累积 text
        max_attr_len = max((len(k) for k in attr_keys), default=0)
        recent_ids = generated_ids[b][-max(8, max_attr_len * 2):]
        try:
            recent_text = tokenizer.decode(recent_ids, skip_special_tokens=True)
        except Exception:
            return False

        for k in range(n_attrs):
            if coverage[b][k]:
                continue
            attr_value = attr_str_values[k]
            if not attr_value:
                continue
            attr_lower = attr_value.lower()
            text_lower = recent_text.lower()
            if attr_lower in text_lower:
                coverage[b][k] = True
                return True
            lcp_len = 0
            for j in range(1, len(attr_lower) + 1):
                if j <= len(text_lower) and text_lower[-j:].startswith(attr_lower[:j]):
                    lcp_len = j
                else:
                    break
            if lcp_len > 0 and lcp_len < len(attr_lower):
                attr_remaining_str = attr_value[lcp_len:]
                if attr_remaining_str:
                    remaining_ids = tokenizer.encode(attr_remaining_str, add_special_tokens=False)
                    if remaining_ids:
                        coverage[b][k] = True
                        forced_copy_queue[b].extend(remaining_ids)
                        return True
        return False

    # 5. first token (per sample)
    next_ids = torch.zeros(B, dtype=torch.long, device=device)
    for b in range(B):
        tid = _sample_token(first_gen_logits[b], b)
        next_ids[b] = tid
        generated_ids[b].append(tid)
        _check_trigger(b, tid)
    attention_mask = torch.cat([attention_mask, torch.ones((B, 1), dtype=attention_mask.dtype, device=device)], dim=1)

    # 6. main loop
    done = [next_ids[b].item() == eos_token_id for b in range(B)]

    for step in range(max_new_tokens - 1):
        if all(done):
            break

        # === A. 末期硬保证 (per sample) ===
        remaining_steps = max_new_tokens - step - 1
        if remaining_steps <= SAFETY_STEPS:
            for b in range(B):
                if done[b]:
                    continue
                if not all(coverage[b]):
                    for k in range(n_attrs):
                        if not coverage[b][k]:
                            coverage[b][k] = True
                            forced_copy_queue[b].extend(attr_keys[k])
                            break

        # === B. 构造 next_input (per sample, 看 forced queue 还是 gen) ===
        # 用 input_ids 路径 (复用 KV cache), 不重新 embed (Qwen 内部 embed)
        # 关键: Qwen 在 input_ids 模式下自动 embed; 在 inputs_embeds 模式下不 embed
        # 这里用 input_ids (与 prefix 不冲突, 因为 prefix 已在 past 里)

        # === B.1 决定 next_ids: forced > gen ===
        # 先做 forward 拿 gen_logits (用于不 in forced queue 的 sample)
        need_gen_mask = torch.zeros(B, dtype=torch.bool, device=device)
        forced_ids = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            if done[b]:
                forced_ids[b] = eos_token_id
                continue
            if forced_copy_queue[b]:
                forced_ids[b] = forced_copy_queue[b].pop(0)
            else:
                need_gen_mask[b] = True

        # 对 need_gen_mask=True 的 sample, 先 forward 拿 logits
        if need_gen_mask.any():
            # 临时 forward (用 input_ids=next_ids where need_gen_mask)
            # 但 forward 必须是 batched, 不能跳 sample
            # 简化: 整个 batch forward, 但只取 need_gen_mask 的 logits
            input_for_forward = next_ids.clone()
            # 替换 done sample 的 next_id 为 pad token (避免影响 attention)
            # 但 attention_mask 已是有效, done sample 不再更新
            out_step = model(
                input_ids=input_for_forward.unsqueeze(1),
                attention_mask=attention_mask,
                use_cache=True, past_key_values=past,
            )
            past = out_step.past_key_values
            gen_logits = out_step.logits[:, -1, :]  # [B, V]
            for b in range(B):
                if need_gen_mask[b]:
                    tid = _sample_token(gen_logits[b], b)
                    forced_ids[b] = tid

        # === C. 写入 generated + trigger detection ===
        for b in range(B):
            tid = int(forced_ids[b].item())
            if done[b]:
                continue
            generated_ids[b].append(tid)
            if tid == eos_token_id:
                done[b] = True
            else:
                _check_trigger(b, tid)

        # === D. 更新 next_ids + attention_mask ===
        next_ids = forced_ids
        # done sample 用 pad token 维持形状
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
        for b in range(B):
            if done[b]:
                next_ids[b] = pad_id
        attention_mask = torch.cat([attention_mask, torch.ones((B, 1), dtype=attention_mask.dtype, device=device)], dim=1)

    # 7. decode
    outs = []
    for b in range(B):
        text = tokenizer.decode(generated_ids[b], skip_special_tokens=True)
        outs.append(text.strip())
    return outs


def main():
    log("=" * 70)
    log("Phase 10.9.4i2: Hard Pointer Action + Coverage (NO sigmoid gate)")
    log("=" * 70)

    # === 1. Load pairs / z_map (Phase 10.10: no dev filter, n=1000 already excludes dev) ===
    log("[1] Loading pairs / z_map ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    log(f"  total pairs (n=1000 scale-up, already excludes dev 60): {len(pairs)}")

    z_map = {}
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            row = json.loads(line)
            z_map[row["user_id"]] = np.concatenate([row["user_mu"], row["user_logvar"]]).astype(np.float32)
    log(f"  z_map: {len(z_map)} users")

    # === 2. Load projector + Qwen ===
    log("[2] Loading projector (frozen) + Qwen ...")
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

    # Phase 10.10: NO projector (no soft prefix). LLM 自由生成.
    projector = None
    NUM_TOKENS = 0
    log(f"  Phase 10.10: NO soft prefix, NO projector")

    # === 3. Build CJK token mask ===
    log("[3] Building CJK token mask (one-shot vocab scan) ...")
    cjk_mask, n_banned = build_cjk_token_mask(tokenizer, VOCAB_SIZE)
    cjk_mask = cjk_mask.to(DEVICE)
    log(f"  CJK banned tokens: {n_banned} / {VOCAB_SIZE} ({n_banned / VOCAB_SIZE:.2%})")

    # === 4. Generation loop ===
    log(f"[4] Generating: {len(pairs)} pairs × {len(CONDITIONS)} conds × {N_CANDIDATES_PER_COND} "
        f"= {len(pairs) * len(CONDITIONS) * N_CANDIDATES_PER_COND} candidates")

    OUT_CANDS.parent.mkdir(parents=True, exist_ok=True)
    fout = OUT_CANDS.open("w")
    n_total = 0
    n_attr_pass = 0
    n_needed_append = 0
    t0 = time.time()
    eos_id = tokenizer.eos_token_id

    for pi, pair in enumerate(pairs):
        if pi % 5 == 0:
            log(f"  [{pi+1}/{len(pairs)}] user={pair['user_id'][:10]} asin={pair['asin']} "
                f"({time.time()-t0:.1f}s)")
        attrs_5 = pair["attrs"]
        attrs_3 = filter_attrs(attrs_5)
        if len(attrs_3) < 2:
            log(f"    [WARN] user {pair['user_id'][:10]} attrs_3 too sparse, skip")
            continue
        z_target = z_map[pair["user_id"]]
        z_random = z_map[pair["random_user_id"]]

        # Phase 10.10: single cond (A_no_style), no per-cond z mapping
        for cond in CONDITIONS:
            def _build_lines_3(a):
                lines = ["Product attributes:"]
                for key in sorted(a.keys()):
                    lines.append(f"{key}: {a[key]}")
                lines.append("Write a natural shopping query that mentions every attribute.")
                return "\n".join(lines)

            attr_prompt_body = _build_lines_3(attrs_3)
            # Phase 10.10: use A_no_style prompt, no style block at all
            user_content = (attr_prompt_body
                            + "\n\nIMPORTANT: Write a short shopping query in ENGLISH ONLY. "
                            + "Do NOT use any Chinese, Japanese, or Korean characters. "
                            + "Use English words only. The query must include every attribute listed above.")
            chat_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False, add_generation_prompt=True,
            )

            # batch all N candidates together for speed
            try:
                batch_outputs = hard_pointer_generate_batch(
                    model=model, tokenizer=tokenizer,
                    prompt_str=chat_prompt, attrs=attrs_3,
                    n_candidates=N_CANDIDATES_PER_COND,
                    device=DEVICE,
                    max_new_tokens=MAX_NEW_TOKENS, eos_token_id=eos_id,
                    temperature=TEMPERATURE, top_k=TOP_K,
                    cjk_mask=cjk_mask,
                )
            except Exception as e:
                log(f"    [WARN] generate failed user={pair['user_id'][:10]} cond={cond}: {e}")
                batch_outputs = [""] * N_CANDIDATES_PER_COND

            for ci, raw_query in enumerate(batch_outputs):
                raw_query = (raw_query or "").strip().strip('"').strip("'")
                raw_query = raw_query.split("\n")[0]
                attr_pass = check_attr_pass(raw_query, attrs_3)
                n_total += 1
                if attr_pass:
                    n_attr_pass += 1
                if not attr_pass:
                    n_needed_append += 1

                fout.write(json.dumps({
                    "user_id": pair["user_id"],
                    "asin": pair["asin"],
                    "random_user_id": pair["random_user_id"],
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

        if (pi + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fout.close()
    log(f"  done in {time.time()-t0:.1f}s")

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
        "proj_ckpt": None,  # Phase 10.10: NO projector
        "soft_prefix_mode": "none",  # Phase 10.10: NO soft prefix
        "conditions": CONDITIONS,
        "attr_fields": ATTR_FIELDS,
        "method": "Hard Pointer Action + Coverage Constraint (NO soft sigmoid gate). "
                  "LM 自由生成普通文本 + 末期 SAFETY_STEPS=8 强制 COPY 剩余 attrs. "
                  "Coverage 未全 True 时禁 EOS.",
        "n_cjk_banned_tokens": n_banned,
        "vocab_size": VOCAB_SIZE,
        "cjk_banned_rate": n_banned / VOCAB_SIZE,
        "fallback_appended": False,
        "safety_steps": SAFETY_STEPS,
        "note": "Phase 10.9.5i will evaluate.",
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {OUT_META}")

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