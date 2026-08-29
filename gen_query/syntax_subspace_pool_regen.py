"""Syntax Subspace — Stage 1 (pool regen only).

用户指令 2026-08-29: Stage 2 spaCy features 已合并到 select_query/ 下
(select_query/syntax_subspace_select_v6m_strict_alignment.py stage_features()),
本脚本只负责 Stage 1 vLLM pool generation。

归 gen_query/: 用 vLLM LLM 为 100 个 ASIN 各生成 K=50 共享候选查询池。

用法:
  python gen_query/syntax_subspace_pool_regen.py --stage pool_regen

I/O 路径:
  输入: stage8_5_asins.json
  输出: stage8_5_pool.json          (Stage 1)

共享工具 (log, paths, hyperparams) 来自:
  common/syntax_subspace_utils.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import List

import requests

# Ensure common/ is on sys.path so we can import the shared utils
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, POOL_IN, POOL_OUT, REPO_ROOT, SCRATCH, VLLM_URL, MODEL_NAME,
    K_POOL, TEMP, MAX_TOKENS, MAX_QUERY_TOKENS, N_INPUT, PCA_DIM, log, feat_key,
)

# Stage 2 also imports per_sentence_features_v2 from common/syntactic_features
sys.path.insert(0, str(REPO_ROOT / "common"))


# ===========================================================================
# Stage 1 helpers (LLM pool generation)
# ===========================================================================

GEN_SYSTEM_TMPL_NATURAL = (
    "You are an Amazon shopper writing a search query. Use EXACTLY the {N_INPUT} "
    "attribute values listed below verbatim (mention each value once). DO NOT add any "
    "product type (no bottle/clothes/toy/candle/tumbler), use case, personal "
    "context, or inferred property (do not turn a Color into a scent, a "
    "Material into a function, or a Style into a product class). DO NOT "
    "include attribute field names (no 'Brand:', 'material_type:', "
    "'material_composition:', 'main category', 'Style:') in the query — "
    "only the values. Each value keeps the meaning of its attribute name. "
    "Write the query from YOUR OWN first-person perspective as the shopper — "
    "use 'I', 'I'm', 'my', 'looking for', 'I want', 'I need', 'hoping to find', "
    "etc. The query must read as something YOU (the shopper) would type, not "
    "as a third-party product description. Write a natural sentence (any length "
    "is fine). Output ONLY the query, no preamble.\n\n"
    "IMPORTANT: ALL {N_INPUT} values MUST appear in the query (verifiable by reading). "
    "If a value is long or compound (e.g., 'silicone nipple, Polypropylene'), "
    "mention BOTH parts even if joined by 'and' or 'with' instead of a comma. "
    "If a value doesn't fit naturally, REWRITE the sentence — never silently drop it.\n\n"
    "Examples of good queries (1st-person, ≤60 tokens, all attrs embedded):\n"
    "- attrs: Brand: Manhattan Toy | Color: Blue/Green/Orange | Material: Plastic | Manufacturer: Manhattan Toy | Material Type Free: BPA Free\n"
    "  query: I'm looking for a Manhattan Toy BPA Free plastic toy in Blue/Green/Orange, hoping it's made by Manhattan Toy again.\n"
    "- attrs: Brand: Dream On Me | Color: White | Material Type: Polyester | Style: Vinyl Cover | Target gender: Unisex\n"
    "  query: I'm looking for a Dream On Me unisex vinyl cover in white made of polyester.\n"
    "- attrs: Brand: Vulli | Color: Brown/ White | Material: Phthalate Free | Material Type: Phthalate Free | Style: Brown Box\n"
    "  query: I'm looking for a Vulli product that's Brown/ White and made with Phthalate Free material, hoping to find one that comes in a Brown Box style.\n"
    "- attrs: Brand: Munchkin | Material Type: Other | Age Range (Description): Adult | Target gender: Unisex | Scent: Munchkin\n"
    "  query: I'm looking for a Munchkin unisex product for adults with a Munchkin scent, hoping it's made of other material type.\n"
    "- attrs: Brand: Boon | Color: White/Orange | Material: Manmade | Material Type: Manmade | Style: Chair with Seat Pad\n"
    "  query: I'm looking for a Boon chair with seat pad in White/Orange, hoping it's made of manmade material and is also manmade type.\n\n"
    "Attributes ({N_INPUT}):\n{ATTRIBUTES}"
)


def build_user_content(attrs: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in attrs.items())


def make_prompt(attrs: dict, n_input: int, k: int = 0) -> str:
    base = GEN_SYSTEM_TMPL_NATURAL.format(
        N_INPUT=n_input,
        ATTRIBUTES=build_user_content(attrs),
    )
    if k > 0:
        # 避免模型把 "(variant k)" 字面输出。改用 paraphrase 指令隐式鼓励差异
        base += f"\nWrite a fresh paraphrase of these attributes — a different sentence shape than other paraphrases you have produced for this same attribute set."
    return base


def batch_generate_vllm(prompts: List[str], temp: float = TEMP, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Batched vLLM generation.

    用户指令 2026-08-27: 去掉 batch→single→"" fallback 三层降级。
    失败直接 raise,让 Stage 1 重跑 (网络/timeout 问题应在 batch 入口 retry)。

    Returns: list of generated texts, len == len(prompts).
    """
    outputs = []
    full_prompts = []
    for p in prompts:
        full_prompts.append(
            f"system\n{p}\n"
            f"user\n\n"
            f"assistant\n"
        )
    bs = 512   # 用户指令 2026-08-27: 8× 客户端 batch, 减少 HTTP overhead
    n_total = len(prompts)
    n_chunks = (n_total + bs - 1) // bs
    t_start = time.time()
    for ci, i in enumerate(range(0, n_total, bs)):
        chunk = full_prompts[i: i + bs]
        # 重试 3 次 batch 请求 (网络瞬时错误), 都失败则 raise
        last_err = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    VLLM_URL,
                    json={
                        "model": MODEL_NAME,
                        "prompt": chunk,
                        "temperature": temp,
                        "max_tokens": max_tokens,
                        "top_p": 0.95 if temp > 0 else 1.0,
                        # 阻止模型继续角色切换 + 写元注释 ('To clarify', '(Note', etc.)
                        "stop": ["\nuser", "\nUser", "\nassistant", "\nAssistant",
                                 "\nsystem", "\nSystem", " user", " User",
                                 " assistant", " Assistant", " system", " System",
                                 "\n(Note", "\n(Here", "\nLet me",
                                 "\nTo clarify", "\nTo make sure", "\nI need",
                                 "\n\nUser:", "\n\nAssistant:", "\n\nSystem:",
                                 ],
                    },
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                for choice in data["choices"]:
                    outputs.append(choice["text"].strip())
                last_err = None
                break
            except Exception as e:
                last_err = e
                log(f"  batch {i}-{i+bs} attempt {attempt+1}/3 failed: {e!r}")
        if last_err is not None:
            raise RuntimeError(
                f"vLLM batch generation failed for prompts {i}-{i+bs} after 3 attempts: "
                f"{last_err!r}. 不要降级 — 请检查 vLLM server 状态后重跑 Stage 1。"
            ) from last_err
        # 用户指令 2026-08-28: 每个 chunk 完成后 log 进度 + ETA, 这样 14B ~10-12min
        # 生成期间能看到实时进展而不是等结束后才知道
        elapsed = time.time() - t_start
        rate = (i + len(chunk)) / max(elapsed, 1e-3)
        eta_s = (n_total - i - len(chunk)) / max(rate, 1e-3)
        log(f"  chunk {ci+1}/{n_chunks}: prompts {i}-{i+len(chunk)} done "
            f"({(i+len(chunk))/n_total*100:.1f}%, {rate:.0f} prompts/s, "
            f"elapsed {elapsed:.0f}s, ETA {eta_s:.0f}s)")
    log(f"  generation complete: {len(outputs)} outputs in {time.time()-t_start:.0f}s")
    return outputs


def count_attrs_covered(text: str, attrs: dict) -> int:
    """用户指令 2026-08-28: 拆分 value 在逗号后,逐部分 substring 匹配。

    原 strict 检查用整个 value 作为 substring 匹配 query,但 LLM 在 ≤60 token 1st-person
    query 里会自然地把 "silicone nipple, Polypropylene" 写成 "silicone nipple and Polypropylene"
    (逗号 → and 是语法自然)。这种生成是合规的,但原匹配因为不含逗号字符串而误判 ATTRS<5。

    新策略: 把 value 按 "," 拆分, 每个 part 都必须在 query 中出现才算 covered。
    例: "silicone nipple, Polypropylene" → ["silicone nipple", "Polypropylene"]
        query: "...silicone nipple and Polypropylene..." → 两个 part 都在, covered=1

    进一步: 也尝试匹配去掉标点符号后的 part,容忍 "Polypropylene," vs "Polypropylene"
    """
    if not text:
        return 0
    text_lower = text.lower()
    # 规范化 text: 去掉常见标点
    import re as _re
    text_norm = _re.sub(r"[,;:.\-_/]", " ", text_lower)
    text_norm = _re.sub(r"\s+", " ", text_norm)
    covered = 0
    for k, v in attrs.items():
        if not v:
            continue
        parts = [p.strip() for p in str(v).split(",") if p.strip()]
        if not parts:
            continue
        # 双重匹配: 原串 + 去标点串都满足
        all_match = True
        for part in parts:
            part_lower = part.lower()
            part_norm = _re.sub(r"[,;:.\-_/]", " ", part_lower).strip()
            part_norm = _re.sub(r"\s+", " ", part_norm)
            # 任一匹配即可 (verbatim 或去标点)
            if part_lower not in text_lower and part_norm not in text_norm:
                all_match = False
                break
        if all_match:
            covered += 1
    return covered


def has_invalid_punct(text: str) -> bool:
    if not text:
        return True
    bad_patterns = [
        r"^(here|this|below|sure|okay|ok)[,:]",
        r"^attribute[s]?:",
        r"^brand:",
        r"^color:",
        r"^material:",
        r"^style:",
    ]
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in bad_patterns)


# First-person indicators: must be shopper's own perspective, not 3rd-person description.
# Require at least one first-person pronoun (i, my, me, mine, we, our) OR first-person
# verb phrase (looking for, want, need, hoping, searching, shopping for).
# Match as whole words (avoid "I" inside "icon" / "image"; "my" inside "mystery"; etc.).
_FIRST_PERSON_RE = re.compile(
    r"\b(i'm|im|i|my|me|mine|we|our|us|looking for|looking to|i want|i need|i'm hoping|i'm looking|i'm searching|hoping to|shopping for|searching for|in search of|want to buy|need to find|wanting|needing)\b",
    re.IGNORECASE,
)


def has_first_person(text: str) -> bool:
    """True iff query is written in shopper's first-person perspective.

    第三人称描述 ("A modern cotton frame..." / "This product..." / "Goodpick
    modern...") 不算 first-person,即使包含 attribute values 也不通过。
    """
    if not text:
        return False
    return bool(_FIRST_PERSON_RE.search(text))


# 用户指令 2026-08-28: emoji 检测。LLM 在 Baby 类目偶尔输出 🌊😊 之类的 emoji 装饰,
# 不属于搜索 query,必须过滤。Unicode emoji block 主要在 0x1F000-0x1FFFF,但 Misc Symbols
# (0x2600-0x27FF) / Dingbats (0x2700-0x27BF) / Supplemental Symbols (0x1F900-0x1F9FF)
# 等都是常见 emoji 区间。
def has_emoji(text: str) -> bool:
    """True iff text contains emoji characters."""
    if not text:
        return False
    for ch in text:
        cp = ord(ch)
        # 排除范围:覆盖主要 emoji block
        if (0x1F000 <= cp <= 0x1FFFF  # 绝大部分 emoji (Misc Symbols & Pictographs 等)
                or 0x2600 <= cp <= 0x27BF  # Misc Symbols + Dingbats
                or 0x2300 <= cp <= 0x23FF  # Misc Technical (含 ⌚⌛)
                or 0x1F300 <= cp <= 0x1F5FF  # Misc Symbols & Pictographs 子集
                or 0x1F600 <= cp <= 0x1F64F  # Emoticons
                or 0x1F680 <= cp <= 0x1F6FF  # Transport & Map
                or 0x1F900 <= cp <= 0x1F9FF  # Supplemental Symbols & Pictographs
                or 0x1FA00 <= cp <= 0x1FA6F  # Chess Symbols
                or 0x1FA70 <= cp <= 0x1FAFF  # Symbols & Pictographs Ext-A
                or 0x1F1E6 <= cp <= 0x1F1FF):  # Regional Indicator Symbols (国旗)
            return True
    return False


# 用户指令 2026-08-28: 自言自语 / 角色扮演检测。LLM 在 N=5+ 1st-person prompt 下偶尔
# 写出 "Can someone help me find it?", "Thanks!", "Let me rephrase..." 等元评论,
# 这些不是 query 文字,污染 strict 池。检测典型自言自语短语。
_SELF_TALK_PATTERNS = [
    r"\bcan someone help\b",
    r"\bcan anyone help\b",
    r"\bany suggestions\??",
    r"\bany ideas\??",
    r"\bthanks!?\s*$",  # 行末单独 "Thanks"
    r"\bthanks again\b",
    r"\bthank you\b",
    r"\bi appreciate\b",
    r"\byou'?re amazing\b",
    r"\byou'?re the best\b",
    r"\blet'?s keep it simple\b",
    r"\blet'?s try again\b",
    r"\blet me rephrase\b",
    r"\blet me clarify\b",
    r"\bdoes that sound right\b",
    r"\bdoes that help\b",
    r"\bthat'?s exactly what i'?m\b",
    r"\bjust those exact attributes\b",
    r"\bwithout (specifying|adding) any\b",
    r"\bthat'?s what i'?m after\b",
    r"\bi hope so\b",
    r"\bi think that\b.*\bwill work\b",
    r"\bsound(s)? good\??",
    r"\bmake(s)? sense\??",
    r"\bhelp me find\b",  # 角色扮演:问别人帮忙找
    r"\bi'?d appreciate\b",
]


def has_self_talk(text: str) -> bool:
    """True iff text contains self-talk / role-playing / meta-commentary phrases.

    这些短语标志 LLM 失控开始"装用户对话",不是真实 query。
    """
    if not text:
        return True
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in _SELF_TALK_PATTERNS)


def n_tokens_simple(text: str) -> int:
    return len(text.split())


def is_query_too_long(text: str, max_tokens: int = MAX_QUERY_TOKENS) -> bool:
    """True iff query token count exceeds max_tokens (default MAX_QUERY_TOKENS=60)."""
    if not text:
        return True
    return n_tokens_simple(text) > max_tokens


# ===========================================================================
# STAGE 1 — POOL REGEN
# ===========================================================================

def stage_pool_regen():
    log(f"=== STAGE 1 — POOL REGEN: K_POOL={K_POOL} × N={N_INPUT} attrs × ASINs ===")

    log(f"loading {ASINS_IN}")
    asin_data = json.load(open(ASINS_IN))["asins"]
    log(f"  {len(asin_data)} ASINs in stage8_5 set")

    # 用户指令 2026-08-28: 用 product_attributes.json 的 top-N attrs (默认 N=5),而不是
    # stage8_5_asins.json 的 attrs_used (那个只有 1-4 attrs)。
    pattrs_path = REPO_ROOT / "result/product_attributes.json"
    log(f"loading {pattrs_path}")
    pattrs = json.load(open(pattrs_path))
    log(f"  {len(pattrs)} ASINs in product_attributes.json")

    # Import get_top_n_attrs from sibling script (rule 12: 留在 gen_query/)
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
    from _n_ablation_pool_v2 import get_top_n_attrs

    log(f"building prompts with top-{N_INPUT} NON-NUMERIC attrs from product_attributes.json...")
    asin_attrs = {}  # asin → top-N attrs
    n_no_attrs = n_lt_n = 0
    for entry in asin_data:
        a = entry["asin"]
        pa = pattrs.get(a)
        if pa is None:
            n_no_attrs += 1
            continue
        top_n = get_top_n_attrs(pa, N_INPUT)
        # 用户指令 2026-08-28: 如果去掉数值属性后 <5 个, 过滤该 ASIN
        if len(top_n) < N_INPUT:
            n_lt_n += 1
            continue
        asin_attrs[a] = top_n
    log(f"  ASINs with ≥{N_INPUT} non-numeric attrs: {len(asin_attrs)}")
    log(f"  ASINs filtered out: no_attrs={n_no_attrs}, lt_{N_INPUT}_non_numeric={n_lt_n}")

    all_prompts = []
    asin_idx = []
    for a, attrs in asin_attrs.items():
        for k in range(K_POOL):
            prompt = make_prompt(attrs, N_INPUT, k=k)
            all_prompts.append(prompt)
            asin_idx.append((a, k))

    log(f"generating {len(all_prompts)} pool queries via vLLM...")
    all_outputs = batch_generate_vllm(all_prompts, temp=TEMP, max_tokens=MAX_TOKENS)
    log(f"  got {len(all_outputs)} outputs")

    pools = collections.defaultdict(list)
    strict_counts = {}
    n_total_strict = 0
    # 用户指令 2026-08-28: 5 层 strict filter breakdown (3 个新 filter 加进来)
    n_cov_full = n_invalid = n_first_p = n_emoji = n_self_talk = n_too_long = 0

    for (a, k), out in zip(asin_idx, all_outputs):
        attrs = asin_attrs[a]
        text = out.strip() if out else ""
        if not text:
            continue
        n_cov = count_attrs_covered(text, attrs)
        invalid = has_invalid_punct(text)
        first_p = has_first_person(text)
        emoji = has_emoji(text)
        self_talk = has_self_talk(text)
        too_long = is_query_too_long(text)
        # 用户指令 2026-08-28: 5 层 strict = 全 attrs + 无 invalid + 1st-person + 无 emoji + 无自言自语 + 长度 ≤60
        is_strict = (
            (n_cov == N_INPUT) and (not invalid) and first_p
            and (not emoji) and (not self_talk) and (not too_long)
        )
        # 统计每个 filter 失败数
        if n_cov == N_INPUT: n_cov_full += 1
        if not invalid: n_invalid += 1
        if first_p: n_first_p += 1
        if not emoji: n_emoji += 1
        if not self_talk: n_self_talk += 1
        if not too_long: n_too_long += 1
        if is_strict:
            n_total_strict += 1
            strict_counts[a] = strict_counts.get(a, 0) + 1
        pools[a].append({
            "k": k,
            "query": text,
            "strict": is_strict,
            "attrs_covered": n_cov,
            "invalid": invalid,
            "first_person": first_p,
            "emoji": emoji,
            "self_talk": self_talk,
            "too_long": too_long,
            "n_tok": n_tokens_simple(text),
        })
        # 用户指令 2026-08-28: 生成过程中每 4000 outputs 打 running strict rate,
        # 这样 14B ~10-12min 生成时能看到质量,不用等结束后才知道
        if len(pools[a]) % 4000 == 0:
            n_so_far = sum(len(v) for v in pools.values())
            log(f"  [progress] {n_so_far}/{len(all_outputs)} processed "
                f"({n_so_far/len(all_outputs)*100:.0f}%), "
                f"running strict: {n_total_strict/max(1,n_so_far)*100:.1f}%")

    log(f"\n=== Pool stats ===")
    log(f"  total queries: {len(all_outputs)}")
    log(f"  strict: {n_total_strict} ({n_total_strict / max(1, len(all_outputs)) * 100:.1f}%)")
    log(f"  --- filter breakdown (passed count) ---")
    log(f"  attrs all covered: {n_cov_full}/{len(all_outputs)} ({n_cov_full/max(1,len(all_outputs))*100:.1f}%)")
    log(f"  no invalid punct:  {n_invalid}/{len(all_outputs)} ({n_invalid/max(1,len(all_outputs))*100:.1f}%)")
    log(f"  1st-person:        {n_first_p}/{len(all_outputs)} ({n_first_p/max(1,len(all_outputs))*100:.1f}%)")
    log(f"  no emoji:          {n_emoji}/{len(all_outputs)} ({n_emoji/max(1,len(all_outputs))*100:.1f}%)")
    log(f"  no self-talk:      {n_self_talk}/{len(all_outputs)} ({n_self_talk/max(1,len(all_outputs))*100:.1f}%)")
    log(f"  length ≤{MAX_QUERY_TOKENS}:       {n_too_long}/{len(all_outputs)} ({n_too_long/max(1,len(all_outputs))*100:.1f}%)")
    log(f"  ASINs: {len(pools)}")
    if strict_counts:
        log(f"  strict per ASIN: min={min(strict_counts.values())}, "
            f"max={max(strict_counts.values())}, "
            f"mean={sum(strict_counts.values()) / len(strict_counts):.1f}")
    log(f"  ASINs with ≥10 strict: {sum(1 for v in strict_counts.values() if v >= 10)}")
    log(f"  ASINs with ≥20 strict: {sum(1 for v in strict_counts.values() if v >= 20)}")

    # 用户指令 2026-08-28: 立刻打 3 strict + 2 non-strict 样本 query,
    # 不必等 Stage 2-5 跑完就能肉眼验证质量
    import random as _rnd
    _rnd.seed(42)
    all_records = [rec for recs in pools.values() for rec in recs]
    strict_recs = [r for r in all_records if r["strict"]]
    non_strict_recs = [r for r in all_records if not r["strict"]]
    log(f"\n=== Sample queries (3 strict + 2 non-strict) ===")
    if strict_recs:
        for r in _rnd.sample(strict_recs, min(3, len(strict_recs))):
            log(f"  [STRICT] {r['query'][:140]}")
    if non_strict_recs:
        for r in _rnd.sample(non_strict_recs, min(2, len(non_strict_recs))):
            fail_reasons = []
            if r["attrs_covered"] < N_INPUT:
                fail_reasons.append(f"attrs={r['attrs_covered']}/{N_INPUT}")
            if r["invalid"]:
                fail_reasons.append("invalid_punct")
            if not r["first_person"]:
                fail_reasons.append("not_1st_person")
            if r["emoji"]:
                fail_reasons.append("emoji")
            if r["self_talk"]:
                fail_reasons.append("self_talk")
            if r["too_long"]:
                fail_reasons.append(f"too_long({r['n_tok']}tok)")
            log(f"  [FAIL  ] {r['query'][:120]} ({','.join(fail_reasons)})")

    POOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": (
                    f"Stage 8.5: SHARED candidate pool per ASIN (K={K_POOL}, N={N_INPUT} non-numeric attrs, "
                    f"first-person + 5-layer strict filter: attrs/invalid/1p/emoji/self-talk/length≤{MAX_QUERY_TOKENS})"
                ),
                "K_POOL": K_POOL,
                "N_INPUT": N_INPUT,
                "MAX_QUERY_TOKENS": MAX_QUERY_TOKENS,
                "TEMP": TEMP,
                "MAX_TOKENS": MAX_TOKENS,
                "SEED": 2024,
                "MODEL_NAME": MODEL_NAME,
                "ATTR_NUMERIC_FILTER": "value contains digit char → excluded",
                "filters": ["attrs_covered==N", "no_invalid_punct", "first_person",
                            "no_emoji", "no_self_talk", f"length≤{MAX_QUERY_TOKENS}"],
            },
            "strict_counts": strict_counts,
            "n_asins": len(pools),
            "n_total_strict": n_total_strict,
            "pools": dict(pools),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {POOL_OUT}")


# ===========================================================================
# MAIN
# ===========================================================================

STAGE_FUNCTIONS = {
    "pool_regen": stage_pool_regen,
}


def main():
    parser = argparse.ArgumentParser(description="Syntax Subspace — gen_query (Stage 1 pool regen only)")
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGE_FUNCTIONS.keys()),
        help="Which stage to run (Stage 2 features 已迁到 select_query/)",
    )
    args = parser.parse_args()

    log(f"=== syntax_subspace_pool_regen.py — stage={args.stage} ===")

    STAGE_FUNCTIONS[args.stage]()


if __name__ == "__main__":
    main()