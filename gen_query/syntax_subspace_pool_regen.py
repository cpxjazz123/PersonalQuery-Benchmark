"""Syntax Subspace — Stage 1 (pool regen only).

两步生成：
  Step 1: vLLM 生成含 N=5 属性的基础 query
  Step 2: TinyStyler 用 per-user 风格向量改写基础 query

用法:
  python gen_query/syntax_subspace_pool_regen.py --stage pool_regen
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import os
import requests

# 常量
HF_HOME = '/fs04/ar57/wenyu/.cache/huggingface'
T5_BASE = 'google/t5-v1_1-large'
TINYSTYLER_WEIGHTS = os.path.join(
    HF_HOME,
    'hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b/tinystyler_model_weights.pt'
)
T5_SNAP = os.path.join(HF_HOME, 'hub/models--google--t5-v1_1-large/snapshots/a98b0fcd0b8137ded40cdf0c0cf0ee884e7c9726')
STYLE_ALPHA = 1.0   # 纯 μ（无噪声），alpha≥0.5 稳定在 d_self↓ + P(d<σ)=100%
MAX_NEW_TOKENS = 72
BATCH_TINYSTYLER = 32
VLLM_URL = 'http://localhost:8800/v1/completions'
QWEN_MODEL = '/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct'
K_POOL = 50
TEMP = 0.7
MAX_TOKENS = 120
MAX_QUERY_TOKENS = 60
N_INPUT = 5
SEED = 42

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import POOL_OUT, REPO_ROOT, log

# ===========================================================================
# Gaussian
# ===========================================================================

GAUSSIAN_NPZ = REPO_ROOT / 'result/gaussian/empirical_user_gaussians.npz'


def _load_gaussian():
    d = np.load(GAUSSIAN_NPZ)
    log(f"  Empirical Gaussian: {len(d['user_ids'])} users, mu={d['mu_768d'].shape}")
    return d['mu_768d'], d['sigma_euclidean'], d['user_ids']


# ===========================================================================
# Prompt 模板
# ===========================================================================

VLLM_PROMPT_TMPL = (
    "system\n"
    "You are an Amazon shopper. Write ONE natural first-person search query "
    "using EXACTLY the {N} attribute values listed below. Write from YOUR OWN perspective "
    "(I, my, looking for, etc.). DO NOT add product type, use case, or extra context. "
    "Output ONLY the query, nothing else.\n\n"
    "Attributes ({N}):\n{attrs}\n\n"
    "assistant\n"
)


def build_attrs_text(attrs: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in attrs.items())


# ===========================================================================
# 5 层 strict filter
# ===========================================================================

_COUNTY_RE = re.compile(r"\b(i'm|im|i|my|me|mine|we|our|us|looking for|looking to|i want|i need|i'm hoping|i'm looking|i'm searching|hoping to|shopping for|searching for|in search of|want to buy|need to find|wanting|needing)\b", re.IGNORECASE)
# 只保留真正自言自语/元评论的 pattern，不误杀正常的 1st-person query 表达
_SELF_TALK_PATTERNS = [
    r"\bcan someone help\b", r"\bcan anyone help\b",
    r"\bany suggestions\??", r"\bany ideas\??",
    r"\bthanks!?\s*$", r"\bthanks again\b", r"\bthank you\b",
    r"\bi appreciate\b", r"\byou'?re amazing\b", r"\byou'?re the best\b",
    r"\blet'?s keep it simple\b", r"\blet'?s try again\b", r"\blet me rephrase\b",
    r"\blet me clarify\b",
    r"\bdoes that sound right\b", r"\bdoes that help\b",
    r"\bthat'?s exactly what i'?m\b", r"\bjust those exact attributes\b",
    r"\bwithout (specifying|adding) any\b",
    r"\bthat'?s what i'?m after\b", r"\bi hope so\b",
    r"\bsound(s)? good\??", r"\bmake(s)? sense\??",
    r"\bhelp me find\b",  # 角色扮演: 问别人帮忙找
    r"\bi'?d appreciate\b",
]
_BAD_PUNCT_RE = re.compile(r"^(here|this|below|sure|okay|ok)[,:]|^attribute[s]?:|^brand:|^color:|^material:|^style:")


def count_attrs_covered(text: str, attrs: dict) -> int:
    if not text:
        return 0
    text_lower = text.lower()
    import re as _re
    text_norm = _re.sub(r"[,;:.\-_/]", " ", text_lower)
    text_norm = _re.sub(r"\s+", " ", text_norm)
    covered = 0
    for v in attrs.values():
        if not v:
            continue
        parts = [p.strip() for p in str(v).split(",") if p.strip()]
        if not parts:
            continue
        all_match = True
        for part in parts:
            part_lower = part.lower()
            part_norm = _re.sub(r"[,;:.\-_/]", " ", part_lower).strip()
            part_norm = _re.sub(r"\s+", " ", part_norm)
            if part_lower not in text_lower and part_norm not in text_norm:
                all_match = False
                break
        if all_match:
            covered += 1
    return covered


def _has_emoji(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        cp = ord(ch)
        if (0x1F000 <= cp <= 0x1FFFF or 0x2600 <= cp <= 0x27BF
                or 0x2300 <= cp <= 0x23FF or 0x1F300 <= cp <= 0x1F5FF
                or 0x1F600 <= cp <= 0x1F64F or 0x1F680 <= cp <= 0x1F6FF
                or 0x1F900 <= cp <= 0x1F9FF or 0x1FA00 <= cp <= 0x1FA6F
                or 0x1FA70 <= cp <= 0x1FAFF or 0x1F1E6 <= cp <= 0x1F1FF):
            return True
    return False


def is_strict(text: str, attrs: dict, n_input: int) -> dict:
    n_cov = count_attrs_covered(text, attrs)
    invalid = bool(_BAD_PUNCT_RE.search(text.lower().strip())) if text else True
    first_p = bool(_COUNTY_RE.search(text)) if text else False
    emoji = _has_emoji(text)
    self_talk = any(re.search(p, text.lower()) for p in _SELF_TALK_PATTERNS) if text else True
    too_long = len(text.split()) > MAX_QUERY_TOKENS if text else True
    strict = (n_cov == n_input) and (not invalid) and first_p and (not emoji) and (not self_talk) and (not too_long)
    return {
        "strict": strict,
        "attrs_covered": n_cov,
        "invalid": invalid,
        "first_person": first_p,
        "emoji": emoji,
        "self_talk": self_talk,
        "too_long": too_long,
        "n_tok": len(text.split()) if text else 0,
    }


# ===========================================================================
# Step 1: vLLM 生成基础 query
# ===========================================================================

def vllm_generate(prompts: List[str], temp: float = TEMP, max_tokens: int = MAX_TOKENS) -> List[str]:
    outputs = []
    bs = 512
    n_total = len(prompts)
    n_chunks = (n_total + bs - 1) // bs
    t_start = time.time()
    for ci in range(0, n_total, bs):
        chunk = prompts[ci:ci + bs]
        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(VLLM_URL, json={
                    "model": QWEN_MODEL,
                    "prompt": chunk,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "top_p": 0.95 if temp > 0 else 1.0,
                    "stop": ["\nuser", "\nassistant", "\nsystem"],
                }, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                for choice in data["choices"]:
                    outputs.append(choice["text"].strip())
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"vLLM failed after 3 attempts: {last_err}")
        elapsed = time.time() - t_start
        done = min(ci + bs, n_total)
        rate = done / max(elapsed, 1e-3)
        eta = (n_total - done) / max(rate, 1e-3)
        log(f"  vLLM chunk {ci//bs+1}/{n_chunks}: {done}/{n_total} ({done/n_total*100:.0f}%, {rate:.0f}/s, ETA {eta:.0f}s)")
    return outputs


# ===========================================================================
# Step 2: TinyStyler 改写注入风格
# ===========================================================================

def _load_tinystyler():
    log("[2] Loading TinyStyler ...")
    os.environ['HF_HOME'] = HF_HOME
    from tinystyler import TinyStyler
    from transformers import T5Tokenizer
    ts_model = TinyStyler(base_model=T5_BASE, use_style=True, ctrl_embed_dim=768)
    saved = torch.load(TINYSTYLER_WEIGHTS, map_location='cpu', weights_only=True)
    saved = {k.replace('module.', ''): v for k, v in saved.items()}
    cur = ts_model.state_dict()
    cur.update(saved)
    ts_model.load_state_dict(cur)
    ts_model.to('cuda:0').eval()
    for p in ts_model.parameters():
        p.requires_grad_(False)
    t5_tokenizer = T5Tokenizer.from_pretrained(T5_SNAP, legacy=True)
    log("  TinyStyler + T5 tokenizer loaded")
    return ts_model, t5_tokenizer


def tinystyler_rewrite(base_queries: List[str], style_vectors: np.ndarray,
                       ts_model, t5_tokenizer, batch_size: int = 32) -> List[str]:
    """TinyStyler 用风格向量改写基础 query。

    TinyStyler.generate() 内部: proj(style * alpha) -> prepend 1 token -> concat with input_embeds -> decode
    输入是 base query text, T5 encoder 将其编码后再注入风格信号。
    """
    n = len(base_queries)
    outputs = []
    pad_id = t5_tokenizer.pad_token_id or 0
    H = ts_model.proj.weight.shape[0]   # 1024

    t_start = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bsz = end - start
        texts = base_queries[start:end]
        z_batch = style_vectors[start:end]

        # tokenize base queries
        toks = [t5_tokenizer(t, padding=False, truncation=True, max_length=256)['input_ids'] for t in texts]
        max_len = max(len(t) for t in toks)
        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device='cuda:0')
        attn_mask = torch.zeros((bsz, max_len), dtype=torch.long, device='cuda:0')
        for i, t in enumerate(toks):
            input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long, device='cuda:0')
            attn_mask[i, :len(t)] = 1

        # style embedding
        style_in = torch.from_numpy(z_batch).to('cuda:0')
        style_scaled = style_in * STYLE_ALPHA

        with torch.no_grad():
            out_ids = ts_model.generate(
                input_ids, attn_mask,
                style=style_scaled,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.2,  # 防止 T5 贪婪解码时重复生成同一短语
            )

        for i in range(bsz):
            text = t5_tokenizer.decode(out_ids[i], skip_special_tokens=True).strip()
            outputs.append(text)

        done = end
        elapsed = time.time() - t_start
        rate = done / max(elapsed, 1e-3)
        log(f"  TinyStyler batch {start//batch_size+1}: {done}/{n} ({done/n*100:.0f}%, {rate:.0f}/s)")

    return outputs


# ===========================================================================
# STAGE 1 — 主流程
# ===========================================================================

def stage_pool_regen():
    log(f"=== STAGE 1: vLLM (Step 1) + TinyStyler (Step 2), K={K_POOL}, N={N_INPUT} ===")

    # -- 加载 product_attributes -----------------------------------------------
    pattrs = json.load(open(REPO_ROOT / "result/product_attributes.json"))
    log(f"  {len(pattrs)} ASINs in product_attributes.json")

    def _has_digit(v):
        return any(c.isdigit() for c in str(v))

    def _get_top_n(attrs_dict, n):
        non_num = [(k, v) for k, v in attrs_dict.items() if not _has_digit(v)]
        if len(non_num) < n:
            return None
        return dict(non_num[:n])

    _MAX_ASINS = 1  # smoke
    asin_list = [a for a in pattrs if _get_top_n(pattrs[a], N_INPUT) is not None][: _MAX_ASINS]
    log(f"  ASINs: {len(asin_list)}")

    # -- 加载 Gaussian -------------------------------------------------------
    mu_768, sigma_arr, user_ids_gauss = _load_gaussian()
    n_users = len(user_ids_gauss)
    rng = np.random.default_rng(SEED)

    # -- 加载模型 -----------------------------------------------------------
    ts_model, t5_tokenizer = _load_tinystyler()

    # -- 构建任务 -----------------------------------------------------------
    asin_attrs = {a: _get_top_n(pattrs[a], N_INPUT) for a in asin_list}
    jobs = [(a, asin_attrs[a]) for a in asin_list for _ in range(K_POOL)]
    log(f"  {len(jobs)} jobs (ASIN={len(asin_list)}, K={K_POOL})")

    # -- Step 1: vLLM 生成基础 query ---------------------------------------
    log("[Step 1] vLLM generating base queries ...")
    vllm_prompts = []
    for a, attrs in jobs:
        attrs_text = build_attrs_text(attrs)
        vllm_prompts.append(VLLM_PROMPT_TMPL.format(N=N_INPUT, attrs=attrs_text))

    base_queries = vllm_generate(vllm_prompts)
    log(f"  vLLM done: {len(base_queries)} base queries")

    # -- Step 2: 采样风格向量 + TinyStyler 改写 ------------------------------
    log("[Step 2] TinyStyler rewriting with style vectors ...")
    uidxs = rng.integers(0, n_users, size=len(jobs))
    mu_batch = mu_768[uidxs].astype(np.float32)
    z_style = mu_batch  # 纯 μ（无噪声），alpha=1.0 已在诊断中确认 d_self↓ + P=100%

    styled_queries = tinystyler_rewrite(base_queries, z_style, ts_model, t5_tokenizer)
    log(f"  TinyStyler done: {len(styled_queries)} styled queries")

    # -- 5 层 strict filter ------------------------------------------------
    pools = collections.defaultdict(list)
    strict_counts = {}
    n_total_strict = 0
    n_cov_full = n_invalid = n_first_p = n_emoji = n_self_talk = n_too_long = 0

    for idx, ((a, attrs), styled) in enumerate(zip(jobs, styled_queries)):
        text = styled.strip() if styled else ""
        if not text:
            continue
        st = is_strict(text, attrs, N_INPUT)
        if st["attrs_covered"] == N_INPUT: n_cov_full += 1
        if not st["invalid"]: n_invalid += 1
        if st["first_person"]: n_first_p += 1
        if not st["emoji"]: n_emoji += 1
        if not st["self_talk"]: n_self_talk += 1
        if not st["too_long"]: n_too_long += 1
        if st["strict"]:
            n_total_strict += 1
            strict_counts[a] = strict_counts.get(a, 0) + 1
        pools[a].append({
            "k": idx % K_POOL,
            "query": text,
            "base_query": base_queries[idx],
            "strict": st["strict"],
            **st,
        })

    log(f"\n=== Pool stats ===")
    log(f"  total: {len(styled_queries)}, strict: {n_total_strict} ({n_total_strict/max(1,len(styled_queries))*100:.1f}%)")
    log(f"  attrs all covered: {n_cov_full}/{len(styled_queries)} ({n_cov_full/max(1,len(styled_queries))*100:.1f}%)")
    log(f"  no invalid punct:  {n_invalid}/{len(styled_queries)}")
    log(f"  1st-person:        {n_first_p}/{len(styled_queries)}")
    log(f"  no emoji:          {n_emoji}/{len(styled_queries)}")
    log(f"  no self-talk:      {n_self_talk}/{len(styled_queries)}")
    log(f"  length <={MAX_QUERY_TOKENS}:  {n_too_long}/{len(styled_queries)}")

    # sample
    import random as _rnd
    _rnd.seed(42)
    all_recs = [r for recs in pools.values() for r in recs]
    strict_recs = [r for r in all_recs if r["strict"]]
    non_strict = [r for r in all_recs if not r["strict"]]
    log(f"\n=== Samples (2 strict + 2 non-strict) ===")
    for r in _rnd.sample(strict_recs, min(2, len(strict_recs))):
        log(f"  [STRICT] base='{r['base_query'][:80]}' -> styled='{r['query'][:100]}'")
    for r in _rnd.sample(non_strict, min(2, len(non_strict))):
        log(f"  [FAIL]   base='{r['base_query'][:80]}' -> styled='{r['query'][:100]}'")

    # -- 保存 ---------------------------------------------------------------
    POOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": f"Stage 1: vLLM base query + TinyStyler style rewrite, K={K_POOL}, N={N_INPUT}",
                "K_POOL": K_POOL, "N_INPUT": N_INPUT, "SEED": SEED,
            },
            "strict_counts": strict_counts,
            "n_asins": len(pools),
            "n_total_strict": n_total_strict,
            "pools": dict(pools),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote -> {POOL_OUT}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["pool_regen"])
    args = parser.parse_args()
    log(f"=== syntax_subspace_pool_regen.py --stage={args.stage} ===")
    if args.stage == "pool_regen":
        stage_pool_regen()


if __name__ == "__main__":
    main()
