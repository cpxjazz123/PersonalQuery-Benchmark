"""Syntax Subspace — Stage 1 (pool regen, spaCy-only, no TinyStyler).

流程:
  Step 1: vLLM 生成含 N=5 属性的基础 query (prompt 注入 target_user 历史句子作为风格锚点)
  Step 2: spaCy 287d 句法特征编码
  Step 3: 287d 余弦距离个性化验证 (d_self vs per-ASIN exclusive threshold)

输出: result/gen_query/pool.json

用法:
  python 04_gen_query/syntax_subspace_pool_regen.py
"""

from __future__ import annotations

import collections
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import POOL_OUT, REPO_ROOT, log
from syntactic_features import per_sentence_features_v2

# ===========================================================================
# 常量
# ===========================================================================

VLLM_URL = "http://localhost:8800/v1/completions"
QWEN_MODEL = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
TEMP = 0.7
MAX_TOKENS = 120
MAX_QUERY_TOKENS = 60
N_INPUT = 5
K_POOL = 50
SEED = 42

SYNTAX_GAUSSIAN = REPO_ROOT / "result/syntax_style_encoder/user_syntax_gaussian.json"
SYNTAX_EXCL = REPO_ROOT / "result/syntax_style_encoder/syntax_exclusive_distance.json"
ASIN_USERS_PATH = REPO_ROOT / "asin_users/asin_to_users.json"
SENT_CACHE = REPO_ROOT / "result/user_review_sentence_extract/uid_to_sentences.pkl"

# 287d CANONICAL_KEYS (per_sentence_features_v2 实际输出,按字母顺序)
CANONICAL_KEYS = [
    'acl', 'advcl', 'advmod_per_verb', 'all_caps_words', 'amod_per_noun',
    'ari_approx', 'branch_factor', 'capitalized_ratio', 'case_per_noun',
    'ccomp', 'chunk_begin_ratio', 'chunk_begin_width', 'chunk_end_ratio',
    'chunk_end_width', 'chunk_mid_width', 'close_p1_', 'close_p2_', 'close_p3_',
    'clpair_acl_advcl', 'clpair_acl_ccomp', 'clpair_acl_relcl', 'clpair_acl_xcomp',
    'clpair_ccomp_advcl', 'clpair_ccomp_xcomp', 'clpair_relcl_advcl',
    'clpair_relcl_ccomp', 'clpair_relcl_xcomp', 'clpair_xcomp_advcl',
    'comma_per_clause', 'compound_per_noun', 'content_word_ratio',
    'dep_ROOT', 'dep_acl', 'dep_advcl', 'dep_advmod', 'dep_advmod_count',
    'dep_agent_count', 'dep_amod', 'dep_amod_count', 'dep_appos', 'dep_attr',
    'dep_aux', 'dep_auxpass', 'dep_case', 'dep_case_count', 'dep_cc',
    'dep_ccomp', 'dep_ccomp_count', 'dep_cobj', 'dep_compound', 'dep_compound_count',
    'dep_conj', 'dep_det', 'dep_det_count', 'dep_diversity', 'dep_diversity_overall',
    'dep_entropy', 'dep_fixed', 'dep_flat', 'dep_flat_count', 'dep_iobj',
    'dep_mark', 'dep_mark_count', 'dep_nmod', 'dep_nmod_count', 'dep_nsubj',
    'dep_nsubj_count', 'dep_nsubjpass', 'dep_nsubjpass_count', 'dep_nummod',
    'dep_nummod_count', 'dep_obj', 'dep_orphan_count', 'dep_poss', 'dep_punct',
    'dep_relcl', 'dep_xcomp',
    'depbg_ADP_NOUN', 'depbg_ADV_VERB', 'depbg_AUX_VERB', 'depbg_ROOT_VERB',
    'depbg_ROOT_ccomp', 'depbg_ROOT_nsubj', 'depbg_ROOT_xcomp', 'depbg_VERB_ADP',
    'depbg_VERB_obj', 'depbg_advcl_VERB', 'depbg_amod_NOUN', 'depbg_cc_CONJ',
    'depbg_compound_NOUN', 'depbg_conj_CONJ', 'depbg_det_NOUN', 'depbg_nmod_NOUN',
    'depbg_nsubj_ADP', 'depbg_nsubj_ADV', 'depbg_nsubj_AUX', 'depbg_nsubj_VERB',
    'depth_eq1', 'depth_eq2', 'depth_eq3', 'depth_eq4', 'depth_eq5', 'depth_var',
    'det_per_noun', 'dist_0_1', 'dist_1_2', 'dist_2_3', 'dist_3_5', 'dist_5_100',
    'ends_comma', 'ends_ellipsis', 'ends_excl', 'ends_period', 'ends_quest',
    'excl_per_clause', 'first_word_lower', 'first_word_upper', 'flesch_approx',
    'gini_word_len', 'has_cond', 'has_conj_and', 'has_conj_or', 'has_dash',
    'has_ellipsis', 'has_paren', 'has_passive', 'has_quote', 'is_interrog',
    'is_left_branching', 'is_wh_question', 'leaf_ratio', 'len_bucket_10',
    'len_bucket_20', 'len_bucket_30', 'len_bucket_40', 'len_bucket_50',
    'main_EXISTS', 'main_SV', 'main_SVA', 'main_SVC', 'main_SVO', 'main_SVOA',
    'main_SVOC', 'main_SVOO', 'main_SVO_IOBJ', 'max_depth', 'max_root_depth',
    'max_word_len', 'mean_dist', 'mean_root_depth', 'mean_word_len',
    'mod_per_noun', 'n_aux', 'n_clause', 'n_colon', 'n_conj', 'n_conj_per_conj',
    'n_coord', 'n_dep_relations', 'n_mod', 'n_multi_punct', 'n_noun',
    'n_np_chunks', 'n_pobj', 'n_pp', 'n_punct_total', 'n_question', 'n_semicol',
    'n_tok', 'n_verb', 'nest_d1', 'nest_d2', 'nest_d3', 'nest_d4', 'nest_d5',
    'nest_ge2', 'nest_ge3', 'nest_max', 'nest_mean', 'nest_std',
    'np_chunk_width_mean', 'open_p1_', 'open_p2_', 'open_p3_', 'paren_depth',
    'pos_ADJ', 'pos_ADP', 'pos_ADV', 'pos_AUX', 'pos_CCONJ', 'pos_CONJ',
    'pos_DET', 'pos_INTJ', 'pos_NOUN', 'pos_NUM', 'pos_PART', 'pos_PRON',
    'pos_PUNCT', 'pos_SCONJ', 'pos_VERB', 'pos_X', 'pos_diversity', 'pos_entropy',
    'posbg_ADJ_ADP', 'posbg_ADJ_NOUN', 'posbg_ADP_DET', 'posbg_ADP_NOUN',
    'posbg_ADV_ADJ', 'posbg_ADV_VERB', 'posbg_AUX_ADJ', 'posbg_AUX_NOUN',
    'posbg_AUX_PART', 'posbg_AUX_VERB', 'posbg_DET_ADJ', 'posbg_DET_NOUN',
    'posbg_NOUN_ADP', 'posbg_NOUN_CCONJ', 'posbg_NOUN_SCONJ', 'posbg_NOUN_VERB',
    'posbg_NUM_NOUN', 'posbg_PART_VERB', 'posbg_PRON_ADP', 'posbg_PRON_AUX',
    'posbg_PRON_VERB', 'posbg_VERB_ADP', 'posbg_VERB_CCONJ', 'posbg_VERB_DET',
    'posbg_VERB_NOUN', 'posbg_VERB_PRON', 'posbg_VERB_SCONJ',
    'postg_ADP_DET_ADJ', 'postg_ADP_DET_NOUN', 'postg_ADV_VERB_DET',
    'postg_AUX_ADJ_ADP', 'postg_AUX_VERB_ADV', 'postg_AUX_VERB_DET',
    'postg_AUX_VERB_NOUN', 'postg_AUX_VERB_PRON', 'postg_DET_ADJ_NOUN',
    'postg_DET_NOUN_VERB', 'postg_NOUN_CCONJ_NOUN', 'postg_PRON_AUX_ADJ',
    'postg_PRON_AUX_VERB', 'postg_VERB_ADP_DET', 'postg_VERB_CCONJ_VERB',
    'postg_VERB_DET_NOUN', 'postg_VERB_PRON_AUX', 'postg_VERB_SCONJ_VERB',
    'pp_depth', 'punct_!', 'punct_"', 'punct_\'', 'punct_(', 'punct_)',
    'punct_,', 'punct_-', 'punct_.', 'punct_:', 'punct_;', 'punct_?',
    'relcl', 'shape_Title', 'shape_UPPER', 'shape_digit', 'shape_lower',
    'shape_mixed', 'shape_other', 'stop_word_ratio', 'subtree_iqr',
    'subtree_max', 'subtree_mean', 'subtree_p25', 'subtree_p50', 'subtree_p75',
    'subtree_skew', 'subtree_std', 'tree_depth_to_width_ratio', 'tree_width_max',
    'tree_width_mean', 'type_token_ratio', 'vp_complexity', 'word_len_iqr',
    'word_len_p25', 'word_len_p75', 'word_len_range', 'word_len_std', 'xcomp',
]


# ===========================================================================
# 工具函数
# ===========================================================================

def feats_to_vector(feats):
    if feats is None:
        return None
    return np.array([float(feats.get(k, 0)) for k in CANONICAL_KEYS], dtype=np.float32)


def extract_syntax_features(text, nlp):
    doc = nlp(text[:500])
    feats = per_sentence_features_v2(doc)
    return feats_to_vector(feats)


def vllm_generate(prompts, temp=TEMP, max_tokens=MAX_TOKENS):
    outputs = []
    bs = 512
    n_total = len(prompts)
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
                for choice in resp.json()["choices"]:
                    outputs.append(choice["text"].strip())
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            outputs.extend([""] * len(chunk))
        done = min(ci + bs, n_total)
        elapsed = time.time() - t_start
        rate = done / max(elapsed, 1e-3)
        log(f"  vLLM {done}/{n_total} ({done/n_total*100:.0f}%, {rate:.0f}/s)")
    return outputs


# ===========================================================================
# 5 层 strict filter
# ===========================================================================

_COUNTY_RE = re.compile(
    r"\b(i'm|im|i|my|me|mine|we|our|us|looking for|looking to|"
    r"i want|i need|i'm hoping|i'm looking|i'm searching|hoping to|"
    r"shopping for|searching for|in search of|want to buy|need to find|"
    r"wanting|needing)\b", re.IGNORECASE)

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
    r"\bhelp me find\b", r"\bi'?d appreciate\b",
]
_BAD_PUNCT_RE = re.compile(
    r"^(here|this|below|sure|okay|ok)[,:]|^attribute[s]?:|^brand:|^color:|^material:|^style:")


_NEGATION_RE = re.compile(
    r"\b(not|no|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|"
    r"won't|wouldn't|can't|couldn't|shouldn't|haven't|hasn't|hadn't)\b")


def count_attrs_covered(text, attrs):
    if not text:
        return 0
    text_lower = text.lower()
    text_norm = re.sub(r"[,;:.\-_/]", " ", text_lower)
    text_norm = re.sub(r"\s+", " ", text_norm)
    covered = 0
    for k, v in attrs.items():
        if not v:
            continue
        v_str = str(v).strip()
        parts = [p.strip() for p in v_str.split(",") if p.strip()]
        if not parts:
            continue

        # Special handling for "No" / "Yes" binary attrs
        # "No" → query should contain negation OR v_str itself
        # "Yes" → query should contain v_str directly (not negated)
        if v_str.lower() == "no":
            # Check if query contains negation near the attribute key or "discontinued"/"required"
            # Match if: negation word is present OR value "no" appears as word
            has_negation = bool(_NEGATION_RE.search(text_lower))
            has_no_word = re.search(r'\bno\b', text_lower) is not None
            has_v_direct = any(p.lower() in text_lower or p.lower() in text_norm for p in parts)
            # Covered if: negation found, OR "no" as word, OR direct value match
            if has_negation or has_no_word or has_v_direct:
                covered += 1
            continue

        if v_str.lower() == "yes":
            # "Yes" → must appear directly (not negated)
            # Check that no negation precedes "yes"
            all_match = True
            for part in parts:
                part_lower = part.lower()
                part_norm = re.sub(r"[,;:.\-_/]", " ", part_lower).strip()
                part_norm = re.sub(r"\s+", " ", part_norm)
                if part_lower not in text_lower and part_norm not in text_norm:
                    all_match = False
                    break
            if all_match:
                covered += 1
            continue

        # Normal case: value appears as-is
        all_match = True
        for part in parts:
            part_lower = part.lower()
            part_norm = re.sub(r"[,;:.\-_/]", " ", part_lower).strip()
            part_norm = re.sub(r"\s+", " ", part_norm)
            if part_lower not in text_lower and part_norm not in text_norm:
                all_match = False
                break
        if all_match:
            covered += 1
    return covered


def _has_emoji(text):
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


def is_strict(text, attrs, n_input):
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
# 主流程
# ===========================================================================

def main():
    import random as rnd
    rnd.seed(SEED)
    np.random.seed(SEED)

    log("=== Stage 1: vLLM + spaCy 287d (no TinyStyler), pool regen ===")

    # -- 加载数据 -----------------------------------------------------------
    log("  Loading Gaussians + exclusive distances ...")
    syntax_gauss = json.load(open(SYNTAX_GAUSSIAN))
    excl_dist = json.load(open(SYNTAX_EXCL))
    asin_users_map = json.load(open(ASIN_USERS_PATH))
    syntax_uids = set(syntax_gauss.keys())
    uid_list = list(syntax_gauss.keys())
    uid_to_idx = {u: i for i, u in enumerate(uid_list)}

    log("  Loading user sentences (for style anchor) ...")
    with open(SENT_CACHE, 'rb') as f:
        uid_to_sents = pickle.load(f)

    log("  Loading spaCy ...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    nlp.max_length = 500000

    # -- Product attributes --------------------------------------------------
    pattrs = json.load(open(REPO_ROOT / "result/product_attributes.json"))
    log(f"  {len(pattrs)} ASINs in product_attributes.json")

    def _get_top_n(attrs_dict, n):
        non_num = [(k, v) for k, v in attrs_dict.items()
                   if not any(c.isdigit() for c in str(v))]
        return non_num[:n] if len(non_num) >= n else None

    # Filter ASINs: must have ≥2 syntax users with exclusive d_self > 0
    asin_list = [a for a in pattrs if _get_top_n(pattrs[a], N_INPUT) is not None]
    asin_list = [a for a in asin_list if
                 sum(1 for u in asin_users_map.get(a, [])
                     if u in syntax_uids and excl_dist.get(a, {}).get(u, -999) > 0) >= 2]
    asin_list = asin_list[:20]  # 20 ASINs
    log(f"  Filtered to {len(asin_list)} ASINs with syntax exclusive > 0")

    # -- 构建 jobs: (asin, attrs, target_uid, anchor_sents) ---------------
    jobs = []
    for a in asin_list:
        attrs = dict(_get_top_n(pattrs[a], N_INPUT))
        cohort = [u for u in asin_users_map.get(a, [])
                  if u in syntax_uids and excl_dist.get(a, {}).get(u, -999) > 0]
        if not cohort:
            continue
        for k in range(K_POOL):
            tu = rnd.choice(cohort)
            # Pick 2 anchor sentences from this user
            sents = uid_to_sents.get(tu, [])
            anchors = [s[:200] for s in sents[:3]]  # up to 3 anchor sentences
            jobs.append((a, attrs, tu, anchors))

    log(f"  {len(jobs)} jobs ({len(asin_list)} ASINs × {K_POOL} K)")

    # -- VLLM prompt template (with style anchor) ---------------------------
    VLLM_PROMPT_TMPL = (
        "system\n"
        "You are an Amazon shopper. Write ONE natural first-person search query "
        "using EXACTLY the {N} attribute values listed below. "
        "Match the WRITING STYLE of the example sentences provided. "
        "Write from YOUR OWN perspective (I, my, looking for, etc.). "
        "DO NOT add product type, use case, or extra context. "
        "Output ONLY the query, nothing else.\n\n"
        "Example sentences (your writing style):\n{anchors}\n\n"
        "Attributes ({N}):\n{attrs}\n\n"
        "assistant\n"
    )

    # -- Build prompts -------------------------------------------------------
    log("[Step 1] Building VLLM prompts with style anchors ...")
    vllm_prompts = []
    for a, attrs, tu, anchors in jobs:
        attrs_text = "\n".join(f"- {k}: {v}" for k, v in attrs.items())
        anchors_text = "\n".join(f"- {s}" for s in anchors[:2]) if anchors else "(none)"
        prompt = VLLM_PROMPT_TMPL.format(
            N=N_INPUT, attrs=attrs_text, anchors=anchors_text)
        vllm_prompts.append(prompt)

    # -- vLLM generate ------------------------------------------------------
    log("[Step 1] vLLM generating queries ...")
    t0 = time.time()
    base_queries = vllm_generate(vllm_prompts)
    log(f"  vLLM done: {len(base_queries)} queries in {time.time()-t0:.1f}s")

    # -- Batch spaCy encode -------------------------------------------------
    log("[Step 2] Encoding queries with spaCy 287d ...")
    query_vecs = []
    for q in base_queries:
        v = extract_syntax_features(q, nlp)
        query_vecs.append(v if v is not None else np.zeros(287, dtype=np.float32))
    query_vecs = np.array(query_vecs)

    # Normalize for cosine distance
    norms_q = np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8
    query_norm = query_vecs / norms_q

    # -- Pre-compute mu batch ------------------------------------------------
    mu_batch = np.array([
        np.array(syntax_gauss[uid_list[uid_to_idx[tu]]]['mu'], dtype=np.float32)
        for _, _, tu, _ in jobs
    ])
    norms_mu = np.linalg.norm(mu_batch, axis=1, keepdims=True) + 1e-8
    mu_norm = mu_batch / norms_mu

    # -- 5-layer strict filter + d_self validation --------------------------
    log("[Step 3] Filtering strict + d_self cosine validation ...")
    pools = collections.defaultdict(list)
    n_total = len(jobs)
    n_strict = n_d_self_pass = n_both = 0
    n_cov_full = n_invalid = n_first_p = n_emoji = n_self_talk = n_too_long = 0

    for idx, (a, attrs, tu, anchors) in enumerate(jobs):
        text = base_queries[idx].strip() if base_queries[idx] else ""
        st = is_strict(text, attrs, N_INPUT)
        if st["attrs_covered"] == N_INPUT: n_cov_full += 1
        if not st["invalid"]: n_invalid += 1
        if st["first_person"]: n_first_p += 1
        if not st["emoji"]: n_emoji += 1
        if not st["self_talk"]: n_self_talk += 1
        if not st["too_long"]: n_too_long += 1

        # d_self cosine distance
        t_eucl = excl_dist.get(a, {}).get(tu, None)
        q_vec = query_norm[idx]
        mu_vec = mu_norm[idx]
        cos_dist = float(1.0 - np.dot(q_vec, mu_vec))
        # t_eucl > 0: user has exclusive zone, convert to cosine threshold
        # t_eucl <= 0: user in shared zone, skip d_self filter (accept all)
        if t_eucl is not None and t_eucl > 0:
            t_cos = t_eucl / 12.0
            # Use stricter of: t_cos or 0.7 (target d_self < 0.7)
            d_self_pass = cos_dist < max(t_cos, 0.7)
        else:
            t_cos = None
            d_self_pass = True  # shared zone: no d_self filter

        strict = st["strict"] and d_self_pass
        if st["strict"]: n_strict += 1
        if d_self_pass: n_d_self_pass += 1
        if strict: n_both += 1

        pools[a].append({
            "k": idx % K_POOL,
            "query": text,
            "target_user": tu,
            "strict": strict,
            "d_self_cos": cos_dist,
            "threshold_cos": t_cos,
            "threshold_eucl": t_eucl,
            **st,
        })

    log(f"\n=== Pool stats (spaCy 287d, cosine distance) ===")
    log(f"  total:           {n_total}")
    log(f"  strict:          {n_strict} ({n_strict/n_total*100:.1f}%)")
    log(f"  d_self_pass:     {n_d_self_pass} ({n_d_self_pass/n_total*100:.1f}%)")
    log(f"  both pass:       {n_both} ({n_both/n_total*100:.1f}%)")
    log(f"  attrs all cover: {n_cov_full}/{n_total}")
    log(f"  1st-person:      {n_first_p}/{n_total}")
    log(f"  no emoji:        {n_emoji}/{n_total}")
    log(f"  no self-talk:    {n_self_talk}/{n_total}")
    log(f"  len <={MAX_QUERY_TOKENS}: {n_too_long}/{n_total}")

    # Samples
    strict_recs = [r for recs in pools.values() for r in recs if r["strict"]]
    non_strict = [r for recs in pools.values() for r in recs if not r["strict"]]
    log(f"\n=== Samples (2 strict + 2 non-strict) ===")
    for r in rnd.sample(strict_recs, min(2, len(strict_recs))):
        log(f"  [STRICT] d_self={r['d_self_cos']:.4f} t_cos={r['threshold_cos']:.4f} | {r['query'][:90]}")
    for r in rnd.sample(non_strict, min(2, len(non_strict))):
        fail = "strict" if not r['strict'] else ("d_self" if not r.get('d_self_pass', True) else "both")
        log(f"  [FAIL:{fail}] d_self={r.get('d_self_cos','?'):.4f} | {r['query'][:90]}")

    # Save
    POOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 1: vLLM + spaCy 287d cosine distance, no TinyStyler",
                "K_POOL": K_POOL, "N_INPUT": N_INPUT, "SEED": SEED,
                "feature_dim": 287, "distance": "cosine",
            },
            "strict_counts": {a: sum(1 for r in recs if r["strict"]) for a, recs in pools.items()},
            "n_asins": len(pools),
            "n_total_strict": n_both,
            "pools": dict(pools),
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote -> {POOL_OUT}")


if __name__ == "__main__":
    main()
