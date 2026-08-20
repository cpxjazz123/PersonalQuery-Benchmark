#!/usr/bin/env python3
"""Phase 10.19.B: Contrastive Exemplar RAG (engine)

新范式 (vs Phase 10.18 iter exemplar search):
  - Phase 10.18: 只给 target 用户的 exemplar 让 LLM 学句法 → margin -21% 但 rank-1 0%
    失败原因: LLM 从 1-3 exemplar 学到 token-level pattern,不是 distributional property
  - Phase 10.19: 对比式 RAG — 同时给 target exemplar + 最近竞争者 exemplar + 差异列表
    让 LLM 学"target 比其他人究竟差在哪",而不是"target 大概长什么样"

三种动态信息:
  1. 目标用户最具代表性的 2 条评论 (exemplar 正例)
  2. 目标用户最容易混淆的 1-2 个其他用户的代表性评论 (exemplar 反例)
  3. 目标用户相对最近竞争者的 top 5-10 个稳定句法差异 (natural language)

输入仍是只含当前商品属性的中性 Query,LLM 只迁移表达组织,不复制旧商品词。
最后用 318d target-vs-nearest-other margin 选候选,并 hard-copy 兜底 attrs_complete。

配置:
  - 30 pairs (sample) × 5 rounds × 8 candidates/round = 1200 generations
  - 集成 hard-copy post-processing (Phase 10.10.6 / e30.14)

输出:
  - phase10_19_iter_log.jsonl (每条: pair, round, cand_idx, query, margin, target_rank, is_best, attrs_complete_post)
  - phase10_19_iter_summary.json
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/vades_prototype")
PAIRS_FILE = OUT_DIR / "phase10_pairs_1000.jsonl"
VADES_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / "Baby_Products"
USER_PROFILE_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_user_profiles.jsonl"
RAW_SENTENCES_FILE = VADES_DIR / "vades_prototype_3000u_v6_raw_sentences.jsonl"
SENT_FEATS_CACHE = OUT_DIR / "phase10_10_6_cache" / "sentence_318d.npy"
SENT_USERS_CACHE = OUT_DIR / "phase10_10_6_cache" / "sentence_318d_users.json"

# === 硬编码 ===
N_PAIRS = 30
N_ROUNDS = 5
N_CANDIDATES_PER_ROUND = 8
K_NEAREST_NEIGHBORS = 3       # 找 3 个最混淆的 other users
N_TARGET_EXEMPLARS = 2        # 2 条 target 用户的代表句
N_COUNTER_EXEMPLARS = 2       # 2 条 (从 K 个最近竞争者合并) 反例
N_DIFF_FEATURES = 8           # top-8 差异 dim
ATTR_FIELDS = ["Brand", "Color", "Material"]
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 20
SEED = 42
# 对比式 prompt 用
DIVERSITY_THRESHOLD = 0.95

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)

CJK_RE = re.compile(r"[　-〿぀-ゟ゠-ヿ一-鿿가-힯]")


def log(m):
    print(f"[phase10-10.19] {m}", flush=True)


# -----------------------------------------------------------------------
# 318d feature name -> natural-language mapping (top relevant dims)
# 仅映射最重要的 30+ dims 让 prompt 易读
# -----------------------------------------------------------------------
FEATURE_NL_MAP = {
    "clause_rate": "subordinate clauses per token",
    "acl_rate": "adnominal clauses per token",
    "advcl_rate": "adverbial clauses per token",
    "ccomp_rate": "clausal complements per token",
    "xcomp_rate": "open clausal complements per token",
    "relcl_rate": "relative clauses per token",
    "modifier_density": "modifiers per token",
    "coordination_density": "coordinators per token",
    "mean_dep_distance": "average dependency distance",
    "depth_variance": "dependency depth variance",
    "median_dep_depth": "median dependency depth",
    "passive_rate": "passive constructions",
    "interrogative_rate": "interrogative sentences",
    "conditional_rate": "conditional sentences",
    "senttype_simple": "simple sentences",
    "senttype_conjunctive": "conjunctive sentences",
    "senttype_complex": "complex sentences",
    "nest_max": "max nesting depth",
    "nest_mean": "mean nesting depth",
    "n_punct_total": "punctuation density",
    "punct_,_and": "comma-and coordination",
    "punctbg_;_and": "semicolon-and",
    "punctbg_,_but": "comma-but",
    "punctbg_!_!": "double exclamation",
    "punctbg_?_?": "double question",
}


def feature_name_to_nl(name: str) -> str:
    """Map 318d feature name to natural-language phrase."""
    if name in FEATURE_NL_MAP:
        return FEATURE_NL_MAP[name]
    if name.startswith("pos_"):
        return f"POS tag {name[4:]}"
    if name.startswith("dep_"):
        return f"dependency {name[4:]}"
    if name.startswith("opener_"):
        return f"sentence-opening {name[7:]}"
    if name.startswith("punct_"):
        return f"punctuation {name[6:]}"
    if name.startswith("main_"):
        return f"main clause pattern {name[5:]}"
    if name.startswith("open_p") or name.startswith("close_p"):
        return name.replace("_", " position ")
    if name.startswith("nest_"):
        return f"nesting {name[5:]}"
    return name.replace("_", " ")


# -----------------------------------------------------------------------
# Hard-copy post-processing (复用 e30.14)
# -----------------------------------------------------------------------
_MAX_ATTR_WORDS = 4
_TRAILING_NOISE_RE = re.compile(r'[":;\s]+$')
_QUOTED_RE = re.compile(r'"[^"]*"')


def _clean_attr_for_append(v: str) -> str:
    v = v.replace('\n', ' ').strip()
    v = _QUOTED_RE.sub('', v)
    v = _TRAILING_NOISE_RE.sub('', v)
    v = ' '.join(v.split())
    words = v.split()
    if len(words) > _MAX_ATTR_WORDS:
        v = ' '.join(words[:_MAX_ATTR_WORDS])
    return v


def _join_list(items):
    items = [str(x) for x in items if x]
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ', '.join(items[:-1]) + f", and {items[-1]}"


def _append_missing_attrs(q: str, attrs: dict) -> str:
    """Append missing attrs as a clause. Returns new query (or original if all present)."""
    q_lower = q.lower()
    missing = []
    for k, v in attrs.items():
        if not v:
            continue
        if str(v).lower() not in q_lower:
            missing.append((k, v))
    if not missing:
        return q
    cleaned = [_clean_attr_for_append(v) for _, v in missing]
    filtered = [cv for cv in cleaned if cv]
    if not filtered:
        return q
    clause = _join_list(filtered)
    new_q = f"{q.rstrip('. ').rstrip(',')}, with {clause}."
    return new_q


def attrs_complete(text: str, attrs: dict, attr_fields: list[str] = ATTR_FIELDS) -> bool:
    """Check if text contains all required attribute values."""
    text_lower = text.lower()
    for k in attr_fields:
        v = attrs.get(k)
        if v and str(v).lower() not in text_lower:
            return False
    return True


def build_lines_3(attrs: dict) -> str:
    lines = ["Product attributes:"]
    for key in sorted(attrs.keys()):
        lines.append(f"{key}: {attrs[key]}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Contrastive prompt builder (核心创新)
# -----------------------------------------------------------------------
def build_contrastive_prompt(
    attrs: dict,
    target_exemplars: list[str],
    counter_exemplars: list[tuple[str, str]],  # (user_id, sentence_text)
    diff_descriptions: list[str],
) -> tuple[str, str]:
    """Build system + user messages for contrastive few-shot generation.

    target_exemplars: 2 representative sentences from TARGET user
    counter_exemplars: 2 sentences from K nearest competitor users
    diff_descriptions: top-N natural-language differences (target vs competitors)
    """
    attr_body = build_lines_3(attrs)

    target_block = "\n".join(f'  T{i+1}. "{ex}"' for i, ex in enumerate(target_exemplars))
    counter_block = "\n".join(
        f'  C{i+1}. (user {uid[:8]}): "{ex}"' for i, (uid, ex) in enumerate(counter_exemplars)
    )

    diff_block = "\n".join(f"  - {d}" for d in diff_descriptions)

    user_content = (
        attr_body
        + "\n\nSTYLE EXAMPLES FROM THIS USER (T) and similar users (C):\n"
        + "TARGET user's typical style:\n" + target_block
        + "\n\nSIMILAR USERS' style (to AVOID — they are NOT this user):\n" + counter_block
        + "\n\nKEY DIFFERENCES — this user, vs similar users, tends to:\n" + diff_block
        + "\n\nCRITICAL RULES:\n"
        + "1. Write a query that follows STYLE T (target user), NOT style C.\n"
        + "2. Emphasize the listed KEY DIFFERENCES — those are what make this user UNIQUE.\n"
        + "3. Do NOT start with template phrases ('Looking for', 'I need', 'I want').\n"
        + "4. Do NOT use template structures ('weighing exactly', 'made entirely of').\n"
        + "5. Do NOT copy brand names, model numbers, or specific words from any example.\n"
        + "6. The query must mention every current product attribute listed above.\n"
        + "7. Use ONLY English characters, Latin alphabet, digits, spaces, and standard punctuation.\n"
        + "8. Keep the query under 25 words.\n\n"
        + "Now write ONE short shopping query for the current product, matching STYLE T and the KEY DIFFERENCES."
    )

    system_prompt = (
        "You are a shopping query writer. You learn style from target examples and "
        "avoid the contrast examples. OUTPUT LANGUAGE: ENGLISH ONLY. "
        "NO Chinese characters. NO Japanese characters. NO Korean characters. "
        "All words in the query must be in English. "
        "Mention every listed attribute of the product by its exact value "
        "(brand name, color, material). Keep the query under 25 words. "
        "Use only Latin alphabet letters, digits, spaces, and standard punctuation."
    )

    return system_prompt, user_content


# -----------------------------------------------------------------------
# Common template phrases (for anti-template filter)
# -----------------------------------------------------------------------
TEMPLATE_PHRASES = [
    "looking for", "i need", "i want", "i'm looking", "i am looking",
    "make sure", "check products", "weighing exactly", "made of pure",
    "made entirely", "made from", "need one", "with an exact", "in exact",
    "with precise", "precise dimensions", "exact weight", "exact dimensions",
    "precise weight", "ensuring it's", "ensure it", "ensure the",
    "must measure", "should measure", "should be", "must be",
]


def contains_template_phrase(text: str, threshold: int = 2) -> bool:
    """Returns True if text contains >= threshold template phrases."""
    text_lower = text.lower()
    n = sum(1 for p in TEMPLATE_PHRASES if p in text_lower)
    return n >= threshold


def contains_forbidden_words(text: str, forbidden: list[str]) -> bool:
    """Check if text contains any forbidden word (case-insensitive substring)."""
    text_lower = text.lower()
    for w in forbidden:
        if w.lower() in text_lower:
            return True
    return False


def extract_exemplar_words(text: str) -> list[str]:
    """Extract distinctive content words (>4 chars, alphanumeric)."""
    words = re.findall(r'\b[A-Za-z][A-Za-z\.\-]{4,}\b', text)
    common_stop = {
        "with", "from", "this", "that", "have", "your", "short", "shopping",
        "query", "every", "include", "must", "also", "should", "write",
        "natural", "english", "only", "looking", "need", "want", "make",
        "sure", "check", "weighing", "exactly", "made", "pure", "entirely",
        "measures", "dimensions", "color", "weight", "pounds", "inches",
        "ounces", "item", "product", "gets", "find", "buy", "size",
        "high", "wide", "long", "large", "small", "looking", "looking",
    }
    return [w for w in words if w.lower() not in common_stop and not w[0].isdigit()]


def main():
    log("=" * 70)
    log("Phase 10.19.B: Contrastive Exemplar RAG (engine)")
    log("=" * 70)

    # === 1. Load pairs (sample N_PAIRS) ===
    log("[1] Loading pairs ...")
    pairs = []
    with PAIRS_FILE.open() as f:
        for line in f:
            pairs.append(json.loads(line))
    pairs = pairs[:N_PAIRS]
    log(f"  pairs: {len(pairs)}")

    # === 2. Load 318d user_mu + LW covariance ===
    log("[2] Loading 318d user_mu + Ledoit-Wolf ...")
    profiles = []
    with USER_PROFILE_FILE.open() as f:
        for line in f:
            profiles.append(json.loads(line))
    user_id_to_idx = {p["user_id"]: i for i, p in enumerate(profiles)}
    n_users = len(profiles)

    sent_feats = np.load(SENT_FEATS_CACHE)
    sent_users = json.loads(SENT_USERS_CACHE.read_text())
    stds = sent_feats.std(axis=0)
    keep_dims = stds > 1e-9
    n_dims_eff = int(keep_dims.sum())
    sent_feats_keep = sent_feats[:, keep_dims].astype(np.float64)
    log(f"  effective dims: {n_dims_eff}")

    user_mu = np.zeros((n_users, n_dims_eff), dtype=np.float64)
    counts = np.zeros(n_users, dtype=np.int32)
    for si in range(len(sent_users)):
        ui = user_id_to_idx.get(sent_users[si])
        if ui is None:
            continue
        user_mu[ui] += sent_feats_keep[si]
        counts[ui] += 1
    valid_user_mask = counts > 0
    user_mu[valid_user_mask] /= counts[valid_user_mask, None]

    from sklearn.covariance import LedoitWolf
    valid_user_mu = user_mu[valid_user_mask]
    lw = LedoitWolf().fit(valid_user_mu)
    cov_shrunk = lw.covariance_
    inv_cov = np.linalg.inv(cov_shrunk + 1e-6 * np.eye(n_dims_eff))
    quad_mu = np.einsum('ij,jk,ik->i', user_mu, inv_cov, user_mu)
    log(f"  users: {n_users}, valid: {valid_user_mask.sum()}")

    # === 3. Pre-compute pairwise Maha distances (n_users x n_users) ===
    log("[3] Pre-computing user-user Maha distances ...")
    # D(i, j) = (mu_i - mu_j)^T inv_cov (mu_i - mu_j)
    # = quad_mu[i] + quad_mu[j] - 2 * mu_i @ inv_cov @ mu_j
    cross_users = user_mu @ inv_cov @ user_mu.T  # [n_users, n_users]
    user_dmat = quad_mu[:, None] + quad_mu[None, :] - 2 * cross_users
    # Zero out invalid users (set distance to inf so they never win)
    invalid_mask = ~valid_user_mask
    user_dmat[invalid_mask, :] = np.inf
    user_dmat[:, invalid_mask] = np.inf
    np.fill_diagonal(user_dmat, np.inf)
    log(f"  user-user dmat: {user_dmat.shape}")

    # === 4. Load raw sentences (for exemplar extraction) ===
    log("[4] Loading raw sentences (43770) ...")
    raw_sentences = []
    with RAW_SENTENCES_FILE.open() as f:
        for line in f:
            raw_sentences.append(json.loads(line))
    log(f"  raw sentences: {len(raw_sentences)}")

    # Index by user_id (exclude holdout)
    user_to_sents = defaultdict(list)
    for s in raw_sentences:
        if s.get("is_holdout"):
            continue
        uid = s["user_id"]
        if uid not in user_id_to_idx:
            continue
        text = (s.get("sentence_text") or "").strip()
        wc = s.get("word_count", 0)
        if not text or wc < 8 or wc > 35:
            continue  # skip too short or too long
        if CJK_RE.search(text):
            continue
        user_to_sents[uid].append(text)

    log(f"  users with usable sents: {len(user_to_sents)}")

    # === 5. For each pair, pre-compute KNN counter-examples + diff list ===
    log("[5] Pre-computing per-pair contrast info ...")
    from syntactic_analysis.extract_syntactic_features import ALL_FEATS_V2
    # Only the kept-dim names matter for diff interpretation
    feat_names = ALL_FEATS_V2
    kept_names = [feat_names[i] for i, k in enumerate(keep_dims) if k]

    pair_contrast = []
    for pi, pair in enumerate(pairs):
        uid = pair["user_id"]
        attrs = pair["attrs"]
        target_idx = user_id_to_idx.get(uid)
        if target_idx is None:
            pair_contrast.append(None)
            continue

        # K nearest other users by Maha distance
        dists = user_dmat[target_idx].copy()
        sorted_idx = np.argsort(dists)
        knn_idx = sorted_idx[:K_NEAREST_NEIGHBORS]
        knn_user_ids = [profiles[j]["user_id"] for j in knn_idx]

        # Mean of KNN users in 318d
        knn_mean = user_mu[knn_idx].mean(axis=0)
        # Diff = target - knn_mean (signed)
        diff = user_mu[target_idx] - knn_mean
        # Top-N dims by |diff|
        abs_diff = np.abs(diff)
        top_diff_local = np.argsort(-abs_diff)[:N_DIFF_FEATURES]
        # Map local kept-dim index back to original 318 index
        global_diff_idx = np.where(keep_dims)[0][top_diff_local]
        diff_descriptions = []
        for rank_local, di in enumerate(top_diff_local):
            name = kept_names[di]
            sign = "+" if diff[di] > 0 else "-"
            nl = feature_name_to_nl(name)
            diff_descriptions.append(f"{sign} {name} ({nl})")

        # Pick 2 target exemplars (random-ish: longest 2 with diverse content)
        target_sents = user_to_sents.get(uid, [])
        target_sents_sorted = sorted(target_sents, key=lambda x: -len(x.split()))[:30]
        # Sample 2
        if len(target_sents_sorted) >= N_TARGET_EXEMPLARS:
            np.random.seed(SEED + pi)
            sel = np.random.choice(len(target_sents_sorted),
                                   size=min(N_TARGET_EXEMPLARS, len(target_sents_sorted)),
                                   replace=False)
            target_exemplars = [target_sents_sorted[i] for i in sel]
        else:
            target_exemplars = target_sents_sorted[:N_TARGET_EXEMPLARS]

        # Pick 1 from each of 2 nearest competitors (total 2 counter-exemplars)
        counter_exemplars = []
        for cuid in knn_user_ids[:N_COUNTER_EXEMPLARS]:
            csents = user_to_sents.get(cuid, [])
            csents_sorted = sorted(csents, key=lambda x: -len(x.split()))[:30]
            if csents_sorted:
                np.random.seed(SEED + pi + hash(cuid) % 10000)
                sel = np.random.choice(len(csents_sorted), size=1, replace=False)
                counter_exemplars.append((cuid, csents_sorted[int(sel[0])]))

        pair_contrast.append({
            "knn_user_ids": knn_user_ids,
            "target_exemplars": target_exemplars,
            "counter_exemplars": counter_exemplars,
            "diff_descriptions": diff_descriptions,
        })

    n_with_contrast = sum(1 for x in pair_contrast if x is not None)
    log(f"  pairs with contrast info: {n_with_contrast}/{N_PAIRS}")

    # === 6. Load Qwen via llm_client ===
    log("[6] Loading Qwen via llm_client ...")
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
    log(f"  Qwen on {DEVICE}")

    # === 7. spaCy for 318d feature extraction ===
    log("[7] Loading spaCy ...")
    import spacy
    from extract_syntactic_features import per_sentence_features_v2, user_features_v2
    try:
        from e22_t2_syntax_encoder_bridge import neutralize_content
    except ImportError:
        neutralize_content = None
    nlp = spacy.load("en_core_web_sm")

    def extract_318d(query: str, attrs: dict) -> np.ndarray | None:
        text = query.strip()
        if not text:
            return None
        if neutralize_content is not None:
            text = neutralize_content(text, attrs)
        doc = nlp(text)
        sfs = [per_sentence_features_v2(s) for s in doc.sents]
        sfs = [s for s in sfs if s is not None]
        if not sfs:
            return None
        v = user_features_v2(sfs)
        return v

    # === 8. Batched generation helper ===
    @torch.no_grad()
    def generate_batch_same_prompt(system_prompt: str, user_content: str, n: int, max_new: int) -> list[str]:
        be = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_content}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        # apply_chat_template returns BatchEncoding; extract input_ids tensor
        prompt_ids = be["input_ids"] if hasattr(be, "__getitem__") and "input_ids" in be else be
        prompt_ids = prompt_ids.to(DEVICE)
        out = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[:, prompt_ids.shape[1]:]
        texts = []
        for i in range(gen_ids.shape[0]):
            txt = tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip()
            txt = CJK_RE.sub("", txt)
            texts.append(txt)
        return texts

    # === 9. Iteration loop ===
    log(f"[9] Iterative loop: {N_PAIRS} pairs × {N_ROUNDS} rounds × {N_CANDIDATES_PER_ROUND} candidates")

    OUT_LOG = OUT_DIR / "phase10_19_iter_log.jsonl"
    fout = OUT_LOG.open("w")

    iter_state = {pi: {"best_margin": -np.inf, "best_rank": n_users,
                       "best_query": None, "best_round": -1,
                       "best_attrs_complete_post": False} for pi in range(N_PAIRS)}

    round_summaries = []

    for r in range(N_ROUNDS):
        log(f"\n--- Round {r} ---")
        t_round = time.time()

        for pi, pair in enumerate(pairs):
            uid = pair["user_id"]
            asin = pair["asin"]
            attrs = pair["attrs"]
            contrast = pair_contrast[pi]
            if contrast is None:
                continue
            target_idx = user_id_to_idx.get(uid)
            if target_idx is None:
                continue

            # Build prompt
            target_exemplars = contrast["target_exemplars"]
            counter_exemplars = contrast["counter_exemplars"]
            diff_descriptions = contrast["diff_descriptions"]

            sys_p, user_content = build_contrastive_prompt(
                attrs, target_exemplars, counter_exemplars, diff_descriptions,
            )

            # Generate
            try:
                queries = generate_batch_same_prompt(sys_p, user_content, N_CANDIDATES_PER_ROUND, MAX_NEW_TOKENS)
            except Exception as e:
                import traceback
                log(f"    [WARN] pair {pi} round {r} generation failed: {type(e).__name__}: {e}")
                log(f"    traceback: {traceback.format_exc().splitlines()[-3:]}")
                continue

            # Collect forbidden words from BOTH target and counter exemplars
            current_brand = str(attrs.get("Brand", "")).lower()
            current_color = str(attrs.get("Color", "")).lower()
            current_material = str(attrs.get("Material", "")).lower()
            forbidden = []
            for ex in target_exemplars + [c[1] for c in counter_exemplars]:
                for w in extract_exemplar_words(ex):
                    wl = w.lower()
                    if wl in (current_brand, current_color, current_material):
                        continue
                    forbidden.append(w)
            forbidden = list(dict.fromkeys(forbidden))[:30]

            # Filter + score
            cand_data = []
            for ci, q in enumerate(queries):
                if not q or len(q.strip()) < 5:
                    continue
                if CJK_RE.search(q):
                    continue
                if forbidden and contains_forbidden_words(q, forbidden):
                    continue

                feat = extract_318d(q, attrs)
                if feat is None:
                    continue
                feat = feat[keep_dims]
                cross = feat @ inv_cov @ user_mu.T
                quad_c = float(feat @ inv_cov @ feat)
                dist = quad_c + quad_mu - 2 * cross
                target_d = float(dist[target_idx])
                target_rank = int((dist < target_d).sum())
                dist_no_target = dist.copy()
                dist_no_target[target_idx] = np.inf
                nearest_other_d = float(dist_no_target.min())
                margin = nearest_other_d - target_d

                is_attr_complete = attrs_complete(q, attrs)

                cand_data.append({
                    "pi": pi, "round": r, "cand_idx": ci,
                    "query": q,
                    "attrs_complete": is_attr_complete,
                    "target_rank": target_rank,
                    "margin": margin,
                })

            if not cand_data:
                continue

            # === Apply hard-copy post-processing for attrs_complete ===
            for c in cand_data:
                if not c["attrs_complete"]:
                    fixed = _append_missing_attrs(c["query"], attrs)
                    c["query_post"] = fixed
                    c["attrs_complete_post"] = attrs_complete(fixed, attrs)
                else:
                    c["query_post"] = c["query"]
                    c["attrs_complete_post"] = True

            # === Diversity-aware selection (top-1 with diversity penalty) ===
            cand_data.sort(key=lambda x: (x["attrs_complete_post"], x["margin"]), reverse=True)

            best = cand_data[0]
            iter_state[pi]["best_margin"] = max(iter_state[pi]["best_margin"], best["margin"])
            if best["target_rank"] < iter_state[pi]["best_rank"]:
                iter_state[pi]["best_rank"] = best["target_rank"]
                iter_state[pi]["best_query"] = best["query_post"]
                iter_state[pi]["best_round"] = r
                iter_state[pi]["best_attrs_complete_post"] = best["attrs_complete_post"]

            # Write log
            for c in cand_data:
                fout.write(json.dumps({
                    "pair_idx": pi, "user_id": uid, "asin": asin, "attrs": attrs,
                    "round": r, "cand_idx": c["cand_idx"],
                    "candidate_query": c["query"],
                    "candidate_query_post": c["query_post"],
                    "attrs_complete_pre": c["attrs_complete"],
                    "attrs_complete_post": c["attrs_complete_post"],
                    "target_rank": c["target_rank"],
                    "margin": c["margin"],
                    "is_best": c["query_post"] == best["query_post"],
                    "knn_user_ids": contrast["knn_user_ids"],
                    "diff_descriptions": diff_descriptions,
                }, ensure_ascii=False) + "\n")

        fout.flush()

        # Round summary
        all_margins = []
        all_ranks = []
        all_complete_post = []
        all_complete_pre = []
        for pi in range(N_PAIRS):
            st = iter_state[pi]
            all_margins.append(st["best_margin"])
            all_ranks.append(st["best_rank"])
            all_complete_post.append(st["best_attrs_complete_post"])
        round_summary = {
            "round": r,
            "elapsed_sec": time.time() - t_round,
            "best_margin_mean": float(np.mean(all_margins)),
            "best_rank_mean": float(np.mean(all_ranks)),
            "rank1_count": int(sum(1 for rk in all_ranks if rk == 0)),
            "rank1_coverage": float(sum(1 for rk in all_ranks if rk == 0) / N_PAIRS),
            "attrs_complete_post_count": int(sum(all_complete_post)),
            "attrs_complete_post_rate": float(sum(all_complete_post) / N_PAIRS),
            "n_pairs": N_PAIRS,
        }
        round_summaries.append(round_summary)
        log(f"  round {r}: margin_mean={round_summary['best_margin_mean']:+.4f}, "
            f"rank1_cov={round_summary['rank1_coverage']:.4f}, "
            f"attrs_complete_post={round_summary['attrs_complete_post_rate']:.4f}, "
            f"elapsed={round_summary['elapsed_sec']:.1f}s")

    fout.close()

    # === Final summary ===
    log("\n" + "=" * 70)
    log("ITERATION CURVE")
    log("=" * 70)
    log(f"  {'Round':>6s} {'Margin':>10s} {'Rank1 Cov':>10s} {'Complete':>10s}")
    for rs in round_summaries:
        log(f"  {rs['round']:>6d} {rs['best_margin_mean']:>+10.4f} "
            f"{rs['rank1_coverage']:>10.4f} {rs['attrs_complete_post_rate']:>10.4f}")

    SUMMARY_OUT = OUT_DIR / "phase10_19_iter_summary.json"
    SUMMARY_OUT.write_text(json.dumps({
        "version": "phase10_19_b_v1",
        "n_pairs": N_PAIRS,
        "n_rounds": N_ROUNDS,
        "n_candidates_per_round": N_CANDIDATES_PER_ROUND,
        "k_nearest_neighbors": K_NEAREST_NEIGHBORS,
        "n_target_exemplars": N_TARGET_EXEMPLARS,
        "n_counter_exemplars": N_COUNTER_EXEMPLARS,
        "n_diff_features": N_DIFF_FEATURES,
        "round_summaries": round_summaries,
        "final_iter_state": {
            pi: {"best_margin": iter_state[pi]["best_margin"],
                 "best_rank": iter_state[pi]["best_rank"],
                 "best_round": iter_state[pi]["best_round"],
                 "best_query": iter_state[pi]["best_query"],
                 "best_attrs_complete_post": iter_state[pi]["best_attrs_complete_post"]}
            for pi in range(N_PAIRS)
        },
    }, ensure_ascii=False, indent=2))
    log(f"  → {SUMMARY_OUT}")
    log(f"  → {OUT_LOG}")


if __name__ == "__main__":
    main()