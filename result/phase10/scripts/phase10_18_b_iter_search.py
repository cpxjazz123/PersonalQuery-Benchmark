#!/usr/bin/env python3
"""Phase 10.18.B: Iterative Exemplar-Guided Style Search (engine)

新范式 (vs Phase 10.16/10.17 soft prefix):
  - Phase 10.16: e22_t3 trained projector 注入 → NO-GO (covariate shift)
  - Phase 10.17: TinyStyler + AnnaWegmann 注入 → NO-GO (LLM 看不到真实 query)

Phase 10.18: 给 LLM 看真实 exemplars 而非抽象统计
  - Round 0: 用 A_no_style prompt 自由生成 K=12 candidates/pair
  - 318d 评分: margin = D_nearest_other - D_target
  - 选 top-1 (with diversity penalty 防句式固定) 作为 exemplar
  - Round 1+: 把 exemplar 嵌入 prompt,告诉 LLM 学句法不抄属性,生成新候选
  - 重复直到满足 rank=1 ∧ margin>0 ∧ attrs_complete

评估指标 (3 维):
  - Rank-1 coverage: 迭代越多 → 持续上升 (目标: > 10%)
  - Mean margin: 迭代越多 → 持续 > 0
  - Syntactic similarity to exemplar: 高 (学句法)
  - Brand/attr copy rate: 低 (不抄内容)

配置:
  - 30 pairs (sample) × 5 rounds × 12 candidates/round = 1800 generations
  - 仅 Phase 10.18.B 引擎,具体 case study 见 Phase 10.18.C

输出:
  - phase10_18_iter_log.jsonl (每条: pair, round, cand_idx, query, margin, target_rank, is_best)
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
SENT_FEATS_CACHE = OUT_DIR / "phase10_10_6_cache" / "sentence_318d.npy"
SENT_USERS_CACHE = OUT_DIR / "phase10_10_6_cache" / "sentence_318d_users.json"

# === 硬编码 ===
N_PAIRS = 30  # subset for demo (target: full 876 in Phase 10.18.C if GO)
N_ROUNDS = 5
N_CANDIDATES_PER_ROUND = 12
EXEMPLAR_PER_PAIR = 3  # 留 3 条高分且彼此不同 (用户说 1-3 条,3 条效果更好)
DIVERSITY_THRESHOLD = 0.95  # 318d cosine 相似度阈值,>0.95 视为同一句式
ATTR_FIELDS = ["Brand", "Color", "Material"]
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 20
SEED = 42
# 12 个多样化 syntactic 提示 (从 Phase 10.12.C 提取),用于 round 0 生成多样 exemplars
SYNTAX_HINTS = [
    "use a simple declarative statement (subject-verb-object, no question)",
    "use an appositive phrase to add detail (e.g., 'a X, which is Y')",
    "start with a determiner (This/The/A/An + noun)",
    "start with a prepositional phrase (With/For/In/On/By + noun)",
    "use a participial phrase (needing/wanting/seeking + noun)",
    "use a relative clause (that/which/where + clause)",
    "use a reduced relative clause (gerund: 'product weighing X')",
    "include a sensory or descriptive adjective (soft, sturdy, lightweight)",
    "include a usage scenario (for babies, for travel, for home)",
    "use a noun phrase (just the product name with attributes, no verb)",
    "include a size qualifier (compact, full-size, mini, large)",
    "include a benefit phrase (easy to clean, durable, gentle on skin)",
]

QWEN_MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct",
)

CJK_RE = re.compile(r"[　-〿぀-ゟ゠-ヿ一-鿿가-힯]")


def log(m):
    print(f"[phase10-10.18] {m}", flush=True)


def build_lines_3(attrs: dict) -> str:
    lines = ["Product attributes:"]
    for key in sorted(attrs.keys()):
        lines.append(f"{key}: {attrs[key]}")
    lines.append("Write a natural shopping query that mentions every attribute.")
    return "\n".join(lines)


# === Few-shot prompt builder ===
def build_fewshot_prompt(attrs: dict, exemplars: list[str], anti_copy_words: list[str]) -> tuple[str, str]:
    """Build system + user messages for few-shot generation.

    exemplars: list of verified query examples (1-3) for the target user.
    anti_copy_words: list of brand/model words from exemplars to forbid.
    """
    attr_body = build_lines_3(attrs)

    exemplar_block = "\n".join(f'  {i+1}. "{ex}"' for i, ex in enumerate(exemplars))

    anti_copy_str = ", ".join(f'"{w}"' for w in anti_copy_words[:20]) if anti_copy_words else "any brand/model words from examples"

    user_content = (
        attr_body
        + "\n\nSTYLE EXAMPLES (this user's previous queries - learn ONLY the syntax/punctuation/tone, NOT content):\n"
        + exemplar_block
        + "\n\nCRITICAL RULES:\n"
        + "1. Learn ONLY the SYNTAX, PUNCTUATION, and TONE from the examples above.\n"
        + "2. Do NOT start with template phrases like 'Looking for', 'I need', 'I want', 'I am looking'.\n"
        + "3. Do NOT use template structures like 'weighing exactly', 'made entirely of', 'precise dimensions'.\n"
        + "4. Do NOT copy any brand name, model number, or specific words from the examples.\n"
        + f"5. Forbid these words from examples: {anti_copy_str}.\n"
        + "6. The query must mention every current product attribute listed above (brand, color, material, etc.).\n"
        + "7. Use ONLY English characters, Latin alphabet, digits, spaces, and standard punctuation.\n"
        + "8. Keep the query under 25 words.\n\n"
        + "Now write ONE short shopping query for the current product, using a DIFFERENT sentence structure than the examples (no template phrases)."
    )

    system_prompt = (
        "You are a shopping query writer. You learn style from examples, not content. "
        "OUTPUT LANGUAGE: ENGLISH ONLY. "
        "NO Chinese characters. NO Japanese characters. NO Korean characters. "
        "All words in the query must be in English. "
        "Mention every listed attribute of the product by its exact value "
        "(brand name, color, material). Keep the query under 25 words. "
        "Use only Latin alphabet letters, digits, spaces, and standard punctuation."
    )

    return system_prompt, user_content


def extract_attr_words(text: str, attrs: dict) -> list[str]:
    """Extract attribute words from text (case-insensitive substrings)."""
    text_lower = text.lower()
    found = []
    for k, v in attrs.items():
        v_lower = str(v).lower()
        # Match whole value (e.g., "TL Care" must appear as phrase)
        if v_lower and v_lower in text_lower:
            found.append(v)
    return found


def contains_forbidden_words(text: str, forbidden: list[str]) -> bool:
    """Check if text contains any forbidden word (from exemplar attrs).

    Returns True if any 4+ char forbidden word appears in text.
    Filters common function words that are not user-specific.
    """
    text_lower = text.lower()
    for w in forbidden:
        if w.lower() in text_lower:
            return True
    return False


# Common template phrases to penalize/avoid in exemplars
TEMPLATE_PHRASES = [
    "looking for",
    "i need",
    "i want",
    "i'm looking",
    "i am looking",
    "make sure",
    "check products",
    "weighing exactly",
    "made of pure",
    "made entirely",
    "made from",
    "i need a",
    "i need one",
    "i need it",
    "need one",
    "with an exact",
    "in exact",
    "with precise",
    "precise dimensions",
    "exact weight",
    "exact dimensions",
    "precise weight",
    "ensuring it's",
    "ensure it",
    "ensure the",
    "must measure",
    "should measure",
    "should be",
    "must be",
]


def attrs_complete(text: str, attrs: dict, attr_fields: list[str] = ATTR_FIELDS) -> bool:
    """Check if text contains all required attribute values."""
    text_lower = text.lower()
    for k in attr_fields:
        v = attrs.get(k)
        if v and str(v).lower() not in text_lower:
            return False
    return True


def main():
    log("=" * 70)
    log("Phase 10.18.B: Iterative Exemplar-Guided Style Search (engine)")
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

    # === 3. Load Qwen via llm_client ===
    log("[3] Loading Qwen via llm_client ...")
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
    log(f"  Qwen on {DEVICE}, vocab={VOCAB_SIZE}")

    # === 4. spaCy for 318d feature extraction ===
    log("[4] Loading spaCy ...")
    import spacy
    from extract_syntactic_features import per_sentence_features_v2, user_features_v2
    try:
        from e22_t2_syntax_encoder_bridge import neutralize_content
    except ImportError:
        neutralize_content = None
    nlp = spacy.load("en_core_web_sm")

    def extract_318d(query: str, attrs: dict) -> np.ndarray | None:
        """Extract 318d feature for a candidate query (post-attr neutralization)."""
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

    # === 5. Batched generation helper ===
    log("[5] Building batched generation helper ...")

    @torch.no_grad()
    def generate_batch_same_prompt(chat_prompt: str, n: int, max_new: int) -> list[str]:
        """Generate n candidates sharing the same chat prompt (batched)."""
        from transformers import LogitsProcessor
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a shopping query writer."},
             {"role": "user", "content": chat_prompt.split("Write a natural shopping query")[0]}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        # Note: we use the prebuilt chat_prompt directly
        prompt_ids = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids.to(DEVICE)
        # Generate n candidates
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
        # Decode only the new tokens
        gen_ids = out[:, prompt_ids.shape[1]:]
        texts = []
        for i in range(gen_ids.shape[0]):
            txt = tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip()
            # Strip CJK artifacts
            txt = CJK_RE.sub("", txt)
            texts.append(txt)
        return texts

    # === 6. Iteration loop ===
    log(f"[6] Iterative loop: {N_PAIRS} pairs × {N_ROUNDS} rounds × {N_CANDIDATES_PER_ROUND} candidates")

    OUT_LOG = OUT_DIR / "phase10_18_iter_log.jsonl"
    fout = OUT_LOG.open("w")

    iter_state = {pi: {"exemplars": [], "best_margin": -np.inf, "best_rank": n_users,
                       "best_query": None, "best_round": -1} for pi in range(N_PAIRS)}

    round_summaries = []

    for r in range(N_ROUNDS):
        log(f"\n--- Round {r} ---")
        t_round = time.time()

        round_records = []  # (pi, cand_idx, query, attrs_complete, target_rank, margin, is_best)

        for pi, pair in enumerate(pairs):
            uid = pair["user_id"]
            asin = pair["asin"]
            attrs = pair["attrs"]
            target_idx = user_id_to_idx.get(uid)
            if target_idx is None:
                continue

            # === Build prompt ===
            if r == 0:
                # Round 0: 12 different syntactic hints, one per candidate (to seed diversity)
                # For pair pi, use SYNTAX_HINTS[pi % 12] as the seed style
                hint_idx = pi % len(SYNTAX_HINTS)
                hint = SYNTAX_HINTS[hint_idx]
                attr_body = build_lines_3(attrs)
                user_content = (
                    f"{attr_body}\n\n"
                    f"STYLE HINT: Your personal writing style tends to {hint}.\n"
                    f"Use your own wording and do not follow a fixed sentence template.\n"
                    f"IMPORTANT: Write a short shopping query in ENGLISH ONLY. "
                    f"Do NOT use any Chinese, Japanese, or Korean characters. "
                    f"Use English words only. The query must include every attribute listed above."
                )
            else:
                # Round 1+: few-shot with exemplars
                exemplars = iter_state[pi]["exemplars"]
                if not exemplars:
                    # Fall back to A_no_style if no exemplar yet
                    attr_body = build_lines_3(attrs)
                    user_content = (attr_body + "\n\nIMPORTANT: Write a short shopping query in ENGLISH ONLY. "
                                    + "Do NOT use any Chinese, Japanese, or Korean characters. "
                                    + "Use English words only. The query must include every attribute listed above.")
                else:
                    # Collect forbidden words from exemplars' attrs (NOT current product attrs)
                    forbidden = []
                    for ex in exemplars:
                        # We treat exemplars as queries of the same user; their brand/model may differ
                        # From any pair's attrs except current product's brand
                        # For simplicity: forbid any 2+-gram word from exemplar (heuristic)
                        words = re.findall(r'\b[A-Za-z][A-Za-z\.\-]{2,}\b', ex)
                        # Filter: only multi-word or distinctive words
                        for w in words:
                            if len(w) >= 4 and w.lower() not in {"with", "from", "this", "that", "have", "your", "short", "shopping", "query", "every", "include", "must", "also", "should", "write", "natural", "english", "only"}:
                                if w.lower() != str(attrs.get("Brand", "")).lower():
                                    forbidden.append(w)
                    # Dedupe
                    forbidden = list(dict.fromkeys(forbidden))[:30]

                    sys_p, user_content = build_fewshot_prompt(attrs, exemplars, forbidden)

            chat_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a shopping query writer. Write one short natural shopping query. OUTPUT LANGUAGE: ENGLISH ONLY. NO Chinese characters. NO Japanese characters. NO Korean characters. All words in the query must be in English. Mention every listed attribute of the product by its exact value. Keep the query under 25 words. Use only Latin alphabet letters, digits, spaces, and standard punctuation."},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False, add_generation_prompt=True,
            )

            # === Generate N candidates ===
            try:
                queries = generate_batch_same_prompt(chat_prompt, N_CANDIDATES_PER_ROUND, MAX_NEW_TOKENS)
            except Exception as e:
                log(f"    [WARN] pair {pi} round {r} generation failed: {e}")
                continue

            # === Filter + score ===
            cand_data = []
            for ci, q in enumerate(queries):
                # Skip empty or CJK-only
                if not q or len(q.strip()) < 5:
                    continue
                if CJK_RE.search(q):
                    continue
                # Skip if contains forbidden content words (anti-copy, only for r >= 1)
                # NOTE: forbid ONLY content words (long distinctive nouns), NOT template phrases.
                # Template phrases ("looking for", "I need") are handled by exemplar selection
                # and prompt instructions, not by content filter.
                if r >= 1 and iter_state[pi]["exemplars"]:
                    current_brand = str(attrs.get("Brand", "")).lower()
                    current_color = str(attrs.get("Color", "")).lower()
                    current_material = str(attrs.get("Material", "")).lower()
                    common_stop = {
                        "with", "from", "this", "that", "have", "your", "short", "shopping",
                        "query", "every", "include", "must", "also", "should", "write",
                        "natural", "english", "only", "looking", "need", "want", "make",
                        "sure", "check", "weighing", "exactly", "made", "pure", "entirely",
                        "measures", "dimensions", "color", "weight", "pounds", "inches",
                        "ounces", "item", "product", "one", "gets", "find", "buy", "size",
                        "high", "wide", "long", "large", "small", "looking", "looking",
                    }
                    forbidden = []
                    for ex in iter_state[pi]["exemplars"]:
                        words = re.findall(r'\b[A-Za-z][A-Za-z\.\-]{4,}\b', ex)
                        for w in words:
                            wl = w.lower()
                            if wl in common_stop:
                                continue
                            if wl in (current_brand, current_color, current_material):
                                continue
                            if wl[0].isdigit():
                                continue
                            forbidden.append(w)
                    forbidden = list(dict.fromkeys(forbidden))[:30]
                    if forbidden and contains_forbidden_words(q, forbidden):
                        continue

                # Extract 318d
                feat = extract_318d(q, attrs)
                if feat is None:
                    continue
                feat = feat[keep_dims]
                # Compute target_rank + margin
                cross = feat @ inv_cov @ user_mu.T
                quad_c = float(feat @ inv_cov @ feat)
                dist = quad_c + quad_mu - 2 * cross
                target_d = float(dist[target_idx])
                target_rank = int((dist < target_d).sum())
                # margin = D(nearest_other) - D(target)
                # nearest_other = min over all users excluding target_idx
                dist_no_target = dist.copy()
                dist_no_target[target_idx] = np.inf
                nearest_other_d = float(dist_no_target.min())
                margin = nearest_other_d - target_d

                is_attr_complete = attrs_complete(q, attrs)

                cand_data.append({
                    "pi": pi, "round": r, "cand_idx": ci,
                    "query": q, "attrs_complete": is_attr_complete,
                    "target_rank": target_rank, "margin": margin,
                })

            if not cand_data:
                continue

            # === Diversity-aware selection (top-1 with diversity penalty) ===
            # Sort by margin desc, then by attr_complete
            cand_data.sort(key=lambda x: (x["attrs_complete"], x["margin"]), reverse=True)

            best = cand_data[0]
            iter_state[pi]["best_margin"] = max(iter_state[pi]["best_margin"], best["margin"])
            if best["target_rank"] < iter_state[pi]["best_rank"]:
                iter_state[pi]["best_rank"] = best["target_rank"]
                iter_state[pi]["best_query"] = best["query"]
                iter_state[pi]["best_round"] = r

            # === Choose exemplar for next round ===
            # Template-aware selection: prefer queries with FEWER template phrases, but allow all
            chosen_exemplars = []
            chosen_feats = []
            # Sort by margin desc, but penalize templated queries
            def exemplar_score(c):
                q_lower = c["query"].lower()
                template_hits = sum(1 for p in TEMPLATE_PHRASES if p in q_lower)
                # Lower template_hits → higher priority; tiebreak by margin
                return (-template_hits, c["margin"])
            cand_sorted = sorted(cand_data, key=exemplar_score, reverse=True)
            for c in cand_sorted:
                if not c["attrs_complete"]:
                    continue
                # Allow margin < 0 (use best candidates as exemplars even if not yet ideal)
                # Diversity check: 318d cosine distance
                feat = extract_318d(c["query"], attrs)
                if feat is None:
                    continue
                feat = feat[keep_dims]
                if chosen_feats:
                    # Compute max cosine similarity to chosen
                    sims = [float(np.dot(feat, cf) / (np.linalg.norm(feat) * np.linalg.norm(cf) + 1e-8))
                            for cf in chosen_feats]
                    if max(sims) > DIVERSITY_THRESHOLD:
                        continue
                chosen_exemplars.append(c["query"])
                chosen_feats.append(feat)
                if len(chosen_exemplars) >= EXEMPLAR_PER_PAIR:
                    break

            # Fallback: if no exemplar passes (e.g., margin all < 0), take top-1 anyway
            if not chosen_exemplars and cand_data:
                chosen_exemplars = [cand_data[0]["query"]]
                feat = extract_318d(cand_data[0]["query"], attrs)
                if feat is not None:
                    chosen_feats = [feat[keep_dims]]

            iter_state[pi]["exemplars"] = chosen_exemplars

            # === Write log ===
            for c in cand_data:
                fout.write(json.dumps({
                    "pair_idx": pi, "user_id": uid, "asin": asin, "attrs": attrs,
                    "round": r, "cand_idx": c["cand_idx"],
                    "candidate_query": c["query"],
                    "attrs_complete": c["attrs_complete"],
                    "target_rank": c["target_rank"],
                    "margin": c["margin"],
                    "is_best": c["query"] == best["query"],
                }, ensure_ascii=False) + "\n")

        fout.flush()

        # === Round summary ===
        all_margins = []
        all_ranks = []
        all_complete = []
        for pi in range(N_PAIRS):
            st = iter_state[pi]
            all_margins.append(st["best_margin"])
            all_ranks.append(st["best_rank"])
            all_complete.append(st["best_query"] is not None and attrs_complete(st["best_query"], pairs[pi]["attrs"]))
        round_summary = {
            "round": r,
            "elapsed_sec": time.time() - t_round,
            "best_margin_mean": float(np.mean(all_margins)),
            "best_rank_mean": float(np.mean(all_ranks)),
            "rank1_count": int(sum(1 for rk in all_ranks if rk == 0)),
            "rank1_coverage": float(sum(1 for rk in all_ranks if rk == 0) / N_PAIRS),
            "attrs_complete_count": int(sum(all_complete)),
            "attrs_complete_rate": float(sum(all_complete) / N_PAIRS),
            "n_pairs": N_PAIRS,
        }
        round_summaries.append(round_summary)
        log(f"  round {r}: margin_mean={round_summary['best_margin_mean']:+.4f}, "
            f"rank1_cov={round_summary['rank1_coverage']:.4f}, "
            f"attrs_complete={round_summary['attrs_complete_rate']:.4f}, "
            f"elapsed={round_summary['elapsed_sec']:.1f}s")

    fout.close()

    # === Final summary ===
    log("\n" + "=" * 70)
    log("ITERATION CURVE")
    log("=" * 70)
    log(f"  {'Round':>6s} {'Margin':>10s} {'Rank1 Cov':>10s} {'Complete':>10s}")
    for rs in round_summaries:
        log(f"  {rs['round']:>6d} {rs['best_margin_mean']:>+10.4f} "
            f"{rs['rank1_coverage']:>10.4f} {rs['attrs_complete_rate']:>10.4f}")

    SUMMARY_OUT = OUT_DIR / "phase10_18_iter_summary.json"
    SUMMARY_OUT.write_text(json.dumps({
        "version": "phase10_18_b_v1",
        "n_pairs": N_PAIRS,
        "n_rounds": N_ROUNDS,
        "n_candidates_per_round": N_CANDIDATES_PER_ROUND,
        "exemplar_per_pair": EXEMPLAR_PER_PAIR,
        "round_summaries": round_summaries,
        "final_iter_state": {
            pi: {"best_margin": iter_state[pi]["best_margin"],
                 "best_rank": iter_state[pi]["best_rank"],
                 "best_round": iter_state[pi]["best_round"],
                 "best_query": iter_state[pi]["best_query"]}
            for pi in range(N_PAIRS)
        },
    }, ensure_ascii=False, indent=2))
    log(f"  → {SUMMARY_OUT}")
    log(f"  → {OUT_LOG}")


if __name__ == "__main__":
    main()
