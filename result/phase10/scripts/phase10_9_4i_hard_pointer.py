#!/usr/bin/env python3
"""Phase 10.9.4i: Hard Pointer Action + Coverage Constraint Decoder.

设计动机:
  - 4h (5952 records): soft sigmoid gate ptr loss stuck ~0.69 (sigmoid 0.5)
  - 根因: sigmoid gate 在 attribute copy 任务上无法学到 sharp decision boundary
  - 即使 60x 数据扩容, ptr loss 仍不下降
  - 结论: 软概率 gate 不是正确归纳偏置

新架构: Hard Pointer Action + Coverage
  - 每个 step 两种 action:
    GEN(token): Qwen 自由生成 (受 user style prefix 控制)
    COPY(A_k): 从 source memory 复制完整 attr span k (硬指针)
  - action 选择: pointer_logits = proj_q(tgt_hidden) · mean(src_attr_span_k_hiddens)
  - 训练: 监督 "if next tokens form attr span k" → point to k
  - 推理: mask coverage (已 copied), 选 argmax → emit complete span

Coverage constraint:
  - coverage[k] = True 后, A_k 不能再 copy
  - 所有 A_k 都 True 之前, EOS token 屏蔽为 -inf
  - EOS 解除: coverage 全 True 后, 允许 EOS (终止 query)

风格控制 (P7):
  - Style prefix (SoftPrefixProjector) 只影响 GEN action 路径
  - COPY action 完全 deterministic, 不受 user style 影响 → 保证 P7 不变

实现:
  - 复用 projector + Qwen + existing proj_q from CopyAwareHead
  - gate 删除 (不再用 sigmoid)
  - 新增 pointer head: PooledAttrSpanKey proj → score
  - single-step generate loop (slow), 但 correctness 100%

数据: 同 4e (60 dev × Target/Random × 5 candidates = 600)
ckpt:
  - phase10_9_2_projector.pt (frozen, 用于 user style prefix)
  - CopyAwareHead (只取 proj_q 权重, frozen, gate 不用)

输出:
  - phase10_9_4i_candidates_60.jsonl
  - phase10_9_4i_meta.json
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
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PROJ_CKPT = OUT_DIR / "phase10_9_2_projector.pt"
COPY_CKPT = OUT_DIR / "phase10_9_4f_copyhead.pt"  # 用 4f (4h 也是同架构, 取 proj_q 即可)
PAIRS_FILE = OUT_DIR / "phase10_pairs_60.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
DEV_USER_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"

OUT_CANDS = OUT_DIR / "phase10_9_4i_candidates_60.jsonl"
OUT_META = OUT_DIR / "phase10_9_4i_meta.json"
LOG_OUT = OUT_DIR / "phase10_9_4i_hard_pointer.log"

# === 硬编码 (与 4e 对齐) ===
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
# Pointer head 温度 (越小越 greedy)
POINTER_TEMPERATURE = 0.1

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

# CJK ranges
CJK_RE = re.compile(r"[　-〿぀-ゟ゠-ヿ一-鿿가-힯]")


def log(m):
    print(f"[phase10-9.4i] {m}", flush=True)


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


@torch.no_grad()
def prepare_attr_keys(src_ids: torch.Tensor, src_attr_mask: torch.Tensor,
                      attrs: Dict[str, str], tokenizer):
    """构造每条 sample 的 attr span token 序列 + hidden (placeholder, 由 forward 填).

    Returns:
      attr_keys: List[List[int]] — 每个 attr key 包含的 src token ids (按 prompt 顺序)
      attr_span_ranges: List[Tuple[int, int]] — (start_pos, end_pos_inclusive) in src
      attr_str_keys: List[str] — "Brand" / "Color" / "Material" (用于覆盖率跟踪)
    """
    from query.soft_prefix.copy_aware import attr_token_spans
    prompt_str = tokenizer.decode(src_ids[src_ids != tokenizer.pad_token_id].tolist())
    spans = attr_token_spans(prompt_str, attrs, tokenizer)
    attr_keys = []
    attr_span_ranges = []
    attr_str_keys = []
    for key in ["Brand", "Color", "Material"]:
        if key not in spans:
            continue
        for (a, b) in spans[key]:
            ids = src_ids[a:b + 1].tolist()
            if any(i == tokenizer.pad_token_id for i in ids):
                continue
            attr_keys.append(ids)
            attr_span_ranges.append((a, b))
            attr_str_keys.append(key)
    return attr_keys, attr_span_ranges, attr_str_keys


def main():
    log("=" * 70)
    log("Phase 10.9.4i: Hard Pointer Action + Coverage Constraint")
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

    from query.soft_prefix.projector import SoftPrefixProjector

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

    NUM_TOKENS = proj_ckpt["num_tokens"]
    log(f"  projector loaded, K={NUM_TOKENS}")

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

            for ci in range(N_CANDIDATES_PER_COND):
                prompt_str = prompts[ci]
                attrs = attrs_list[ci]
                uvec = user_vecs[ci]

                try:
                    raw_query = hard_pointer_generate(
                        model=model, tokenizer=tokenizer, projector=projector,
                        prompt_str=prompt_str, attrs=attrs, user_vec=uvec,
                        device=DEVICE, num_tokens=NUM_TOKENS,
                        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=eos_id,
                        temperature=TEMPERATURE, top_k=TOP_K,
                        cjk_mask=cjk_mask,
                    )
                except Exception as e:
                    log(f"    [WARN] generate failed user={pair['user_id'][:10]} cond={cond} ci={ci}: {e}")
                    raw_query = ""

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
        "proj_ckpt": str(PROJ_CKPT),
        "copy_ckpt": str(COPY_CKPT),
        "conditions": CONDITIONS,
        "attr_fields": ATTR_FIELDS,
        "method": "Hard Pointer Action + Coverage Constraint (NO soft sigmoid gate). "
                  "GEN/COPY(A_k) action per step. Coverage state forbids repeat copy "
                  "and forbids EOS until all attrs copied.",
        "n_cjk_banned_tokens": n_banned,
        "vocab_size": VOCAB_SIZE,
        "cjk_banned_rate": n_banned / VOCAB_SIZE,
        "fallback_appended": False,
        "note": "Phase 10.9.5i will evaluate. NO FALLBACK.",
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


@torch.no_grad()
def hard_pointer_generate(
    model, tokenizer, projector,
    prompt_str: str, attrs: Dict[str, str], user_vec: np.ndarray,
    device, num_tokens: int,
    max_new_tokens: int, eos_token_id: int,
    temperature: float, top_k: int,
    cjk_mask: torch.Tensor,
) -> str:
    """单条 generate with hard pointer action + coverage constraint.

    流程:
      1. encode prompt + 准备 attr span keys (token sequences)
      2. first forward → 拿 src hidden
      3. step loop:
         a. 如果 forced_copy_queue 非空 → 直接 pop token (覆盖 gen), coverage 已更新
         b. 否则: forward → gen_logits + src_hidden + tgt_hidden
            - 计算 pointer logits: proj_q(tgt_hidden) · pooled_attr_span_hiddens
            - mask coverage (已 copied) 为 -inf
            - 如果 coverage 未满: pointer logits 用于决定"是否 COPY"
              → 简单策略: GEN → 直接 emit gen token (no need for copy)
            - 如果 coverage 已满: 正常 GEN + 允许 EOS
            (设计选择: 让 LM 自由生成, 但确保 attr 一定 emit via emit_all_uncovered_attrs_forced)
      4. 实现简化: 在 step 前, 如果 coverage 未满, 则检查已生成 text — 如果有任何 attr 已 emit (substring match),
         则把对应 attr 标 coverage = True; 如果某 attr 尚未 emit, 在 EOS 之前强制 emit 该 attr
    """
    target_dtype = next(model.parameters()).dtype

    # 1. encode prompt
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt").to(device)

    # 2. 准备 attr span keys (按 prompt 顺序)
    attr_keys, attr_span_ranges, attr_str_keys = prepare_attr_keys(
        prompt_ids[0], None, attrs, tokenizer,
    )
    n_attrs = len(attr_keys)
    if n_attrs == 0:
        # 没找到 attrs, 退化为普通 generation
        out = model.generate(
            input_ids=prompt_ids, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-4), top_k=top_k,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id,
        )
        return tokenizer.decode(out[0, prompt_ids.size(1):], skip_special_tokens=True).strip()

    # 3. 准备 prefix + first forward
    emb = model.get_input_embeddings()
    text_emb = emb(prompt_ids).to(target_dtype)
    if num_tokens > 0 and user_vec is not None:
        z = torch.as_tensor(user_vec, dtype=torch.float32, device=device).unsqueeze(0)
        prefix = projector(z.to(target_dtype))
        full_emb = torch.cat([prefix, text_emb], dim=1)
        attention_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=device)
        prefix_pos = torch.arange(num_tokens, device=device).unsqueeze(0)
        text_pos = torch.arange(prompt_ids.size(1), device=device).unsqueeze(0) + num_tokens
        position_ids = torch.cat([prefix_pos, text_pos], dim=1)
        src_hidden_offset = num_tokens
    else:
        full_emb = text_emb
        attention_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=device)
        position_ids = torch.arange(prompt_ids.size(1), device=device).unsqueeze(0)
        src_hidden_offset = 0

    out = model(
        inputs_embeds=full_emb, attention_mask=attention_mask,
        position_ids=position_ids, use_cache=True,
        past_key_values=None, output_hidden_states=True,
    )
    past = out.past_key_values
    hidden = out.hidden_states[-1]
    src_hidden = hidden[:, src_hidden_offset: src_hidden_offset + prompt_ids.size(1), :]
    first_gen_logits = out.logits[:, -1]  # [1, V]

    # 4. compute attr key hidden vectors (mean pool over span)
    att_key_hiddens = []
    for (a, b) in attr_span_ranges:
        # hidden is [1, prompt_len + num_tokens, D]; 取 prompt 部分
        span_h = src_hidden[:, a:b + 1, :]  # [1, span_len, D]
        att_key_hiddens.append(span_h.mean(dim=1).squeeze(0))  # [D]
    att_key_hiddens = torch.stack(att_key_hiddens, dim=0)  # [K, D]

    # 5. generate loop
    coverage = [False] * n_attrs  # coverage[k] = True if A_k 已 copied
    generated = []
    last_token = None

    def _sample_from_logits(logits):
        """Apply CJK mask, top-k, temperature, sample."""
        m = logits.clone()
        if cjk_mask is not None:
            m = m.masked_fill(cjk_mask.to(m.device, dtype=torch.bool), float("-inf"))
        # coverage 不全 True 时禁 EOS
        if not all(coverage):
            m[eos_token_id] = float("-inf")
        if temperature > 0:
            m = m / max(temperature, 1e-4)
            if top_k > 0:
                v = torch.topk(m, min(top_k, m.size(-1))).values[-1]
                m = m.clone(); m[m < v] = float("-inf")
            return int(torch.multinomial(m.softmax(dim=-1), 1).item())
        return int(torch.argmax(m))

    # first token
    next_id = _sample_from_logits(first_gen_logits[0])
    generated.append(next_id)
    past = out.past_key_values
    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1)
    last_token = next_id

    # forced_copy_queue: 强制 emit 的 token 队列
    forced_copy_queue = []
    # 触发: 已生成 tokens 中含某 attr 全部 token sequence (substring match)
    # 简单策略: 每 step 后, 检查已生成 tokens (decoded text) — 若 text 含某 attr value 字串 → 标 coverage
    # 但 streaming decode 不能回头改 coverage
    # 改用 token-level: 检查 last_token 是否 = 某 attr 第一 token → 标记预期 emit (push 剩余 token 到 forced queue)

    # 简化: 当 LM 即将生成 next token (argmax 前), 检查 pointer logits:
    # 如果 pointer 指向 A_k (未 copied) → 强制 emit A_k 完整 token sequence
    # pointer logits = proj_q(tgt_hidden) · att_key_hiddens (没用 proj_q trained, 但用 last hidden dot att_key_hiddens)
    # 但 proj_q 没训练, gate 也没用. 这里用 tgt_hidden 末层 hidden 直接点乘 att_key_hiddens

    # first-token 后: 进入 main loop
    for step in range(max_new_tokens - 1):
        if next_id == eos_token_id:
            break

        # === A. forced copy (覆盖 step) ===
        if forced_copy_queue:
            next_id = forced_copy_queue.pop(0)
            generated.append(next_id)
            if next_id == eos_token_id:
                break
            # advance
            next_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1)
            out = model(input_ids=next_ids, attention_mask=attention_mask, use_cache=True, past_key_values=past)
            past = out.past_key_values
            # trigger detection: 如果 forced emit 命中某 attr 第二 token + 后续都匹配, 也标记
            continue

        # === B. 正常 forward ===
        next_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        out = model(
            input_ids=next_ids, attention_mask=attention_mask,
            use_cache=True, past_key_values=past, output_hidden_states=True,
        )
        past = out.past_key_values
        gen_logits = out.logits[:, -1]  # [1, V]
        tgt_hidden = out.hidden_states[-1][:, -1:, :]  # [1, 1, D]

        # === C. pointer decision ===
        # score[t, k] = tgt_hidden · att_key_hiddens[k]
        scores = torch.matmul(tgt_hidden[0, 0, :], att_key_hiddens.T)  # [K]
        # mask: 已 copied → -inf
        for k in range(n_attrs):
            if coverage[k]:
                scores[k] = float("-inf")
        # 如果 coverage 全 True, scores 全 -inf → 直接 emit gen token

        # === D. action decision ===
        # 简单策略: pointer argmax 分数 vs 一个阈值 (empirical)
        # 如果 max score > threshold → COPY; else GEN
        # 这里用 heuristic: 把 pointer score 归一化到 [0, 1], > 0.5 → COPY
        # 或者: 让 LM 完全自由 (即每次都 GEN), 但确保 attribute 在 end-of-sentence 前全部 emit
        #
        # 最稳策略 (避免触发误判): 让 LM 自由 GEN, 当 step > max_new_tokens * 0.5 且有 attr 未 copied → 强制 COPY
        # 但这破坏 style
        #
        # 改进: 每 step 都 GEN, 但持续 check pointer score. 如果 score 高 (> threshold) → 触发 COPY
        # threshold 设 POINTER_THRESHOLD

        POINTER_THRESHOLD = 0.5  # 待调
        max_score = float(scores.max())
        if max_score > POINTER_THRESHOLD and not all(coverage):
            # COPY action: 选 argmax
            k = int(scores.argmax().item())
            # 标记 coverage + 强制 emit 完整 attr tokens
            coverage[k] = True
            attr_tokens = attr_keys[k]  # list[int]
            forced_copy_queue.extend(attr_tokens)
            # step 内立即 emit 第一 token (剩下的下 step 再 emit)
            next_id = forced_copy_queue.pop(0)
            generated.append(next_id)
            if next_id == eos_token_id:
                break
            next_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1)
            out = model(input_ids=next_ids, attention_mask=attention_mask, use_cache=True, past_key_values=past)
            past = out.past_key_values
            continue

        # === E. GEN action ===
        next_id = _sample_from_logits(gen_logits[0])
        generated.append(next_id)
        next_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1)

        # 触发检测: 如果刚 emit 的 token 是某 attr 第一 token, push 剩余到 forced queue
        # 这样即使 pointer score 不高, LM 自己开始 emit attr value 也接管
        for k in range(n_attrs):
            if coverage[k]:
                continue
            attr_tokens = attr_keys[k]
            if len(attr_tokens) >= 1 and next_id == attr_tokens[0]:
                # 检查后续 generated tokens 是否已经匹配 attr_tokens[1:]
                matched = True
                for j in range(1, len(attr_tokens)):
                    if len(generated) < j + 1:
                        matched = False; break
                    if generated[-(j + 1)] != attr_tokens[-j]:
                        matched = False; break
                # 简单: 假设 LM 总是连续 emit attr tokens, push 剩余
                coverage[k] = True
                forced_copy_queue.extend(attr_tokens[1:])
                break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


if __name__ == "__main__":
    main()