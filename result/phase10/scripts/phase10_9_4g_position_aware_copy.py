#!/usr/bin/env python3
"""Phase 10.9.4g: Position-aware Forced-Copy Override.

设计动机 (diag_missing_attr_source.py):
  - 4e / 4e2: raw attr_pass 91-94% (gate sigmoid stuck, soft copy 不可靠)
  - 即使 8 epochs 重训 ptr loss 仍在 0.69 (= sigmoid 0.5, 无效)
  - 根本原因: 60 records + 13M params gate head + bf16 → soft sigmoid 完全不收敛

修复策略 (Approach 3: position-aware hard copy override):
  - 不再依赖 copy_head 的 sigmoid gate
  - 改在 generate_batch 的 step-level 加 forced_copy_queue (per sample)
  - step 前: queue 非空 → 直接 pop 并 emit (bypass mixed_logits)
  - step 后: 检查 sample 已生成 token 是否 = 某 attr value 首 token → 把该 attr 剩余 token push 到 queue
  - 不修改 generator 内部, 只在调用层包一层 (post-step processing on next_ids)

实现:
  - 用 regex 在 raw query (before step) 上 substring 搜索 attr values → 找到起始位置
  - 但 streaming decoding 不能回头改 raw query
  - 改用 prompt-side marker: prompt 里 attr value 之前插一个 sentinel token, generator 学会 emit 它, 触发 copy mode
  - 简化: 直接告诉 generator "请你用 EXACTLY 这几个 attribute value 字面量"

更直接的实现:
  - 在 generate_batch 后处理阶段 (post-step)
  - 每个 sample 的 forced_queue 跟踪 "下一个要 force copy 的 token"
  - 触发条件: 当前 step 的 next_id = 某 attr value 的首 token
    → 把该 attr 后续 token (除首 token) push 进 queue
  - 退出条件: 当 attr value 后续 token 不匹配 src token sequence → 终止 queue (LM 偏了, 自然 generation 接管)

新写 forced_copy_decorator: 把 generate_batch 包一层, 加 force queue 处理.

数据:
  - 同 4e: 60 dev × (D_target + C_random) × 5 candidates = 600

ckpt:
  - phase10_9_2_projector.pt (frozen)
  - phase10_9_4f_copyhead.pt (frozen, 但只在非 force 步生效)

输出:
  - phase10_9_4g_candidates_60.jsonl
  - phase10_9_4g_meta.json
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
PROJ_CKPT = OUT_DIR / "phase10_9_2_projector.pt"
COPY_CKPT = OUT_DIR / "phase10_9_4f_copyhead.pt"
PAIRS_FILE = OUT_DIR / "phase10_pairs_60.jsonl"
USER_PROFILE_FILE = (
    REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
    / "Baby_Products" / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
)
DEV_USER_FILE = OUT_DIR / "phase10_dev_60_user_ids.json"

OUT_CANDS = OUT_DIR / "phase10_9_4g_candidates_60.jsonl"
OUT_META = OUT_DIR / "phase10_9_4g_meta.json"
LOG_OUT = OUT_DIR / "phase10_9_4g_position_aware.log"

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
    print(f"[phase10-9.4g] {m}", flush=True)


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


def encode_attr_token_sequences(attrs_3: Dict[str, str], tokenizer):
    """返回: {key: List[int]} 每个 attr value 的 src token 序列.

    这些 token 在 prompt 里出现过, generator 学会 emit 首 token 时触发 forced_copy.
    """
    out = {}
    for k, v in attrs_3.items():
        if not v:
            continue
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        if ids:
            out[k] = ids
    return out


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


def filter_attrs(attrs_5):
    return {k: v for k, v in attrs_5.items() if k in ATTR_FIELDS and v}


def generate_with_force_copy(
    generator,
    prompt_strs: List[str],
    attrs_list: List[Dict[str, str]],
    user_vecs: List[np.ndarray],
    tokenizer,
    max_new_tokens: int,
    eos_token_id: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    cjk_mask: torch.Tensor,
):
    """带 forced copy override 的 generate_batch.

    实现:
      - 用 generator.generate_batch 一次拿 raw output (已有 batched, 不重写)
      - 但 raw output 不准, 因为没 forced copy
      - 改用逐 sample 调用 generator.generate (单条 path, 用 forced override)
      - 这是为了 step-level control: 知道每步 next_id 就能 insert forced token

    注: 逐条调用会慢 (N=600 次单条 vs batch), 但 forced copy 是 raw correctness fix,
    速度其次, 这是关键属性完整率 100% 的最后一道防线.
    """
    outs = []
    for prompt_str, attrs, uvec in zip(prompt_strs, attrs_list, user_vecs):
        out = generate_single_with_force(
            generator, prompt_str, attrs, uvec, tokenizer,
            max_new_tokens, eos_token_id, do_sample, temperature, top_k, cjk_mask,
        )
        outs.append(out)
    return outs


def generate_single_with_force(
    generator,
    prompt_str: str,
    attrs: Dict[str, str],
    user_vec: np.ndarray,
    tokenizer,
    max_new_tokens: int,
    eos_token_id: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    cjk_mask: torch.Tensor,
) -> str:
    """单条 generate + forced copy override.

    Step 逻辑:
      1. 如果 forced_queue 非空 → pop 一 token (bypass mixed)
      2. 否则走 generator.generate 的正常 step
      3. Step 后: 检查 next_id 是否 = 某 attr value 首 token → push 剩余 token 到 forced_queue
    """
    # 准备 attr value token sequences
    attr_token_seqs = encode_attr_token_sequences(attrs, tokenizer)
    # {key: List[int]} - e.g. {"Brand": [t1, t2, ...], "Color": [...]}
    # first tokens 触发器
    first_token_to_remaining = {}  # first_id -> (key, remaining_ids)
    for key, ids in attr_token_seqs.items():
        if len(ids) >= 1:
            first = ids[0]
            remaining = ids[1:]
            if remaining:
                # 多 token attr: first token 触发后 push 剩余
                first_token_to_remaining[first] = (key, remaining)
            # else: 单 token attr 无需 forced override (LM 一次就能 emit)

    # 强制 prefix: 每个 sample 在生成开始时 push 所有 attr value 剩余 token,
    # 但这样 LM 第一次 emit 就是 attribute → 不自然
    # 正确做法: 让 LM 自然生成, 检测首 token 后再 push 剩余
    # 但若 LM 永不 emit 首 token (gate 不可靠), 仍会漏
    #
    # 改进: 用 prompt 引导 LM 一定 emit attr values
    # 例如 prompt 里 attribute 之前加一个 sentinel "<<"
    # Q: LM 学会 emit "<<" 吗? 不一定.
    #
    # 最稳: 在 prompt 里直接告诉 LM "use these exact words: Brand=X, Color=Y, Material=Z"
    # 然后 LM 几乎一定 emit 完整 attr values
    # 但这是 prompt engineering, 边际收益低 (4e v1→v4 prompt changes 已边际 ≈ 0)
    #
    # 实际上: 当 LM 已经能 emit 部分 attr (93.67% 4e), gate 仍 stuck ≈ 0.5
    # 那么 LM 自然输出的 token sequence 与 src attr token sequence 重叠的概率不低
    # forced copy 只需在 LM 成功 emit 首 token 时接管后续, 即可达到 99%+ accuracy

    # 这里直接复用 generator.generate 但加 step-level override
    # 实现: 我们重写 generate loop 而不调 generator.generate (后者无法 step-level control)

    # 1. encode prompt + 构造 prefix
    DEVICE = generator.device
    target_dtype = next(generator.base.parameters()).dtype

    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False, return_tensors="pt").to(DEVICE)
    NUM_TOKENS = generator.num_tokens

    # attr spans in prompt (for copy head)
    from query.soft_prefix.copy_aware import attr_token_spans
    spans = attr_token_spans(prompt_str, attrs, tokenizer)
    src_attr_mask = torch.zeros(prompt_ids.size(1), dtype=torch.bool, device=DEVICE)
    for sp in spans.values():
        for (a, b) in sp:
            src_attr_mask[a:b + 1] = True

    emb = generator.base.get_input_embeddings()
    text_emb = emb(prompt_ids).to(target_dtype)
    if NUM_TOKENS > 0:
        z = torch.as_tensor(user_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        prefix = generator.projector(z.to(target_dtype))
        full_emb = torch.cat([prefix, text_emb], dim=1)
        attention_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=DEVICE)
        prefix_pos = torch.arange(NUM_TOKENS, device=DEVICE).unsqueeze(0)
        text_pos = torch.arange(prompt_ids.size(1), device=DEVICE).unsqueeze(0) + NUM_TOKENS
        position_ids = torch.cat([prefix_pos, text_pos], dim=1)
        src_hidden_offset = NUM_TOKENS
    else:
        full_emb = text_emb
        attention_mask = torch.ones(full_emb.shape[:2], dtype=torch.long, device=DEVICE)
        position_ids = torch.arange(prompt_ids.size(1), device=DEVICE).unsqueeze(0)
        src_hidden_offset = 0

    # 2. first forward
    out = generator.base(
        inputs_embeds=full_emb,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        past_key_values=None,
        output_hidden_states=True,
    )
    hidden = out.hidden_states[-1]
    src_hidden = hidden[:, src_hidden_offset: src_hidden_offset + prompt_ids.size(1), :]
    gen_logits = out.logits[:, -1]
    gen_hidden = hidden[:, -1:, :]

    from query.soft_prefix.copy_aware import mixed_logits
    p_copy, copy_logits = generator.copy_head(gen_hidden, src_hidden, src_attr_mask.unsqueeze(0), prompt_ids)
    mixed = mixed_logits(gen_logits.unsqueeze(1), p_copy, copy_logits)[0, 0]

    def _sample(m):
        if cjk_mask is not None:
            m = m.masked_fill(cjk_mask.to(m.device, dtype=torch.bool), float("-inf"))
        if do_sample:
            m = m / max(temperature, 1e-4)
            if top_k > 0:
                v = torch.topk(m, min(top_k, m.size(-1))).values[-1]
                m = m.clone(); m[m < v] = float("-inf")
            return int(torch.multinomial(m.softmax(dim=-1), 1).item())
        return int(torch.argmax(m))

    generated = []
    next_id = _sample(mixed)
    generated.append(next_id)
    past = out.past_key_values
    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=DEVICE)], dim=1)
    next_ids = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)

    # 3. forced_copy state
    forced_queue = []  # List[int] - tokens to force emit next steps
    # 触发: 当前 step 的 next_id == attr value 首 token → push 剩余 token
    if next_id in first_token_to_remaining:
        _key, remaining = first_token_to_remaining[next_id]
        forced_queue.extend(remaining)

    # 4. main loop with forced override
    for step in range(max_new_tokens - 1):
        if next_id == eos_token_id:
            break

        # === FORCED OVERRIDE ===
        if forced_queue:
            # 强制 emit, bypass mixed_logits (100% 准确)
            next_id = forced_queue.pop(0)
            # 也 append 到 generated (为了下一次 step 检测首 token 触发)
            generated.append(next_id)
            if next_id == eos_token_id:
                break
            # 推进 KV cache (用 input_ids 而不是 emb, 省一次 embed)
            next_ids = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=DEVICE)], dim=1)
            out = generator.base(
                input_ids=next_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past,
            )
            past = out.past_key_values
            # 检测触发 (强制 emit 的 token 也可能恰好是其他 attr 的首 token)
            if next_id in first_token_to_remaining:
                _k, remaining = first_token_to_remaining[next_id]
                # 不应该 force 第二次 attr start, 但保险加 try
                # 如果 forced_queue 已空且这是新 attr start, push
                if not forced_queue:
                    forced_queue.extend(remaining)
            continue

        # === NORMAL STEP ===
        out = generator.base(
            input_ids=next_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past,
            output_hidden_states=True,
        )
        past = out.past_key_values
        gen_logits = out.logits[:, -1]
        gen_hidden = out.hidden_states[-1][:, -1:, :]
        p_copy, copy_logits = generator.copy_head(gen_hidden, src_hidden, src_attr_mask.unsqueeze(0), prompt_ids)
        mixed = mixed_logits(gen_logits.unsqueeze(1), p_copy, copy_logits)[0, 0]
        next_id = _sample(mixed)
        generated.append(next_id)
        next_ids = torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=DEVICE)], dim=1)

        # === TRIGGER DETECTION ===
        if next_id in first_token_to_remaining:
            _key, remaining = first_token_to_remaining[next_id]
            forced_queue.extend(remaining)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()


def main():
    log("=" * 70)
    log("Phase 10.9.4g: Position-aware Forced-Copy Generation")
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
                outputs = generate_with_force_copy(
                    generator=generator,
                    prompt_strs=prompts,
                    attrs_list=attrs_list,
                    user_vecs=user_vecs,
                    tokenizer=tokenizer,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=eos_id,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    cjk_mask=cjk_mask,
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
        "method": "Position-aware forced copy override + non-ASCII hard mask — NO FALLBACK",
        "n_cjk_banned_tokens": n_banned,
        "vocab_size": VOCAB_SIZE,
        "cjk_banned_rate": n_banned / VOCAB_SIZE,
        "fallback_appended": False,
        "note": "Step-level forced_queue override: emit首 token triggers pushing 剩余 src attr tokens. "
                "Bypasses soft gate (which stuck at sigmoid 0.5 due to 60-record data bottleneck). "
                "Phase 10.9.5g will evaluate this candidates file.",
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log(f"  meta → {OUT_META}")

    # 自检
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