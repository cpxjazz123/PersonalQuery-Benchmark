#!/usr/bin/env python3
"""318d 句法特征 residual 流水线主脚本。

Pipeline:
  Stage 1: extract_sentences_spacy  — spaCy 切句，输出 sentences_for_rewrite.jsonl
  Stage 2: parse_sentences_to_features — 318d 句法特征提取，输出 sentences_318d_cache.jsonl.gz
  Stage 3: gaussian diagonal_residual — 拟合 per-user Gaussian，生成 query

所有路径硬编码，不接受 CLI 参数。
"""
from __future__ import annotations
import gzip
import json
import os
import sys
import time
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# 硬编码路径
# ══════════════════════════════════════════════════════════════════════════════
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

DATA_SOURCE = Path(os.environ.get("PQ_DATA_SOURCE", str(SCRATCH / "stage1_filtered_users_reviews_10000u.json")))
SENTS_OUT = Path(os.environ.get("PQ_SENTS_OUT", str(SCRATCH / "sentences_for_rewrite_10k.jsonl")))
FEAT_CACHE = Path(os.environ.get("PQ_FEAT_CACHE", str(SCRATCH / "sentences_318d_cache.jsonl.gz")))

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: spaCy 切句
# ══════════════════════════════════════════════════════════════════════════════
def stage1_extract_sentences():
    """spaCy 批量切句，输出 sentences_for_rewrite_10k.jsonl。"""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    MIN_WORDS, MAX_WORDS, SENTS_PER_USER = 5, 60, int(os.environ.get("PQ_SENTS_PER_USER", "15"))

    def normalize_text(text: str) -> str:
        import re
        return re.sub(r"\s+", " ", text.strip())

    print(f"[Stage 1] Loading {DATA_SOURCE}...")
    data = json.load(open(DATA_SOURCE, "r", encoding="utf-8"))
    print(f"[Stage 1] {len(data)} users")

    # 增量
    done_users: set = set()
    if SENTS_OUT.exists():
        with open(SENTS_OUT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_users.add(d["user_id"])
                except Exception:
                    pass
        print(f"[Stage 1] 增量跳过 {len(done_users)} 用户")

    pending: list = []
    user_meta: dict = {}
    for entry in data:
        uid = entry.get("user_id", "?")
        if uid in done_users:
            continue
        asin = entry.get("asin", "?")
        review_texts = []
        for rev in entry.get("reviews", []):
            trg = rev.get("target_reviews", [])
            if isinstance(trg, list):
                review_texts.extend(trg)
            elif isinstance(trg, str):
                review_texts.append(trg)
        if not review_texts:
            continue
        user_meta[uid] = {"asin": asin}
        for t in review_texts:
            pending.append((uid, asin, t))

    print(f"[Stage 1] 待处理 {len(user_meta)} 用户，{len(pending)} 文本")
    if not pending:
        print("[Stage 1] 无待处理文本")
        return

    user_collected: dict = {u: [] for u in user_meta}
    seen_sentences: dict = {u: set() for u in user_meta}
    BATCH = 500
    t0 = time.time()

    for batch_start in range(0, len(pending), BATCH):
        batch = pending[batch_start:batch_start + BATCH]
        texts = [t for _, _, t in batch]
        docs = list(nlp.pipe(texts, batch_size=BATCH))
        for (uid, asin, _), doc in zip(batch, docs):
            for sent in doc.sents:
                if len(user_collected[uid]) >= SENTS_PER_USER:
                    break
                normalized = normalize_text(sent.text)
                if not normalized:
                    continue
                n = len(normalized.split())
                if MIN_WORDS <= n <= MAX_WORDS and normalized not in seen_sentences[uid]:
                    seen_sentences[uid].add(normalized)
                    user_collected[uid].append({
                        "user_id": uid,
                        "asin": asin,
                        "sentence_text": normalized,
                        "word_count": n,
                    })
        done = min(batch_start + BATCH, len(pending))
        if done % 5000 < BATCH or done == len(pending):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (len(pending) - done) / max(rate, 1e-6)
            print(f"  parsed {done}/{len(pending)} ({rate:.0f}/s, ETA {eta:.0f}s)")

    rows = []
    for uid, sents in user_collected.items():
        rows.extend(sents[:SENTS_PER_USER])
    print(f"[Stage 1] 输出 {len(rows)} 句到 {SENTS_OUT}")
    with open(SENTS_OUT, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: 318d 句法特征提取
# ══════════════════════════════════════════════════════════════════════════════
# --- inject parse_sentences_to_features helpers inline (避免跨目录 import 问题) ---
import re
from collections import Counter

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _tokens_no_space(sent):
    return [t for t in sent if not t.is_space]


POS_TAGS = (
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "CONJ",
    "AUX", "NUM", "INTJ", "PART", "PUNCT", "X", "CCONJ", "SCONJ",
)
POS_BIGRAMS = (
    ("DET","NOUN"),("PRON","VERB"),("AUX","VERB"),("VERB","DET"),
    ("VERB","NOUN"),("ADJ","NOUN"),("NOUN","VERB"),("ADV","VERB"),
    ("ADP","DET"),("ADP","NOUN"),("PRON","AUX"),("DET","ADJ"),
    ("NOUN","ADP"),("VERB","ADP"),("VERB","PRON"),("ADV","ADJ"),
    ("ADJ","ADP"),("NOUN","CCONJ"),("VERB","CCONJ"),("NOUN","SCONJ"),
    ("VERB","SCONJ"),("AUX","ADJ"),("AUX","NOUN"),("PRON","VERB"),
    ("DET","NOUN"),("ADV","ADJ"),("AUX","PART"),("PART","VERB"),
    ("NUM","NOUN"),("PRON","ADP"),
)
POS_TRIGRAMS = (
    ("DET","NOUN","VERB"),("PRON","AUX","VERB"),("PRON","AUX","ADJ"),
    ("AUX","VERB","DET"),("AUX","VERB","NOUN"),("AUX","VERB","ADV"),
    ("VERB","DET","NOUN"),("ADP","DET","NOUN"),("VERB","ADP","DET"),
    ("ADP","DET","ADJ"),("DET","ADJ","NOUN"),("VERB","PRON","AUX"),
    ("AUX","ADJ","ADP"),("ADV","VERB","DET"),("AUX","VERB","PRON"),
    ("VERB","CCONJ","VERB"),("NOUN","CCONJ","NOUN"),("VERB","SCONJ","VERB"),
)
DEP_RELS = (
    "nsubj","nsubjpass","obj","iobj","cobj","attr","aux","auxpass",
    "ROOT","det","poss","amod","advmod","nmod","appos","nummod",
    "acl","relcl","ccomp","xcomp","advcl","conj","cc","punct",
    "case","mark","compound","fixed","flat",
)
DEP_BIGRAMS = (
    ("nsubj","VERB"),("VERB","obj"),("nsubj","AUX"),("AUX","VERB"),
    ("det","NOUN"),("nmod","NOUN"),("amod","NOUN"),("compound","NOUN"),
    ("nsubj","ADV"),("ADV","VERB"),("ROOT","nsubj"),("ROOT","VERB"),
    ("VERB","ADP"),("ADP","NOUN"),("nsubj","ADP"),("cc","CONJ"),
    ("conj","CONJ"),("ROOT","ccomp"),("ROOT","xcomp"),("advcl","VERB"),
)
CLAUSE_DEPS = {"acl","relcl","ccomp","xcomp","advcl"}
CLAUSE_PAIRS = (
    ("acl","relcl"),("acl","ccomp"),("acl","xcomp"),("acl","advcl"),
    ("relcl","ccomp"),("relcl","xcomp"),("relcl","advcl"),
    ("ccomp","xcomp"),("ccomp","advcl"),("xcomp","advcl"),
)
PUNCT_TAGS = (",",".",":",";","!","?","'","\"","-","(",")")


def per_sentence_features_v2(sent):
    """Extract ~235-dim per-sentence syntactic features."""
    import numpy as np
    toks = _tokens_no_space(sent)
    n_tok = len(toks)
    if n_tok < 3:
        return None
    feats = {
        "n_tok": n_tok,
        "n_clause": sum(1 for t in sent if t.dep_ in CLAUSE_DEPS),
        "n_coord": sum(1 for t in sent if t.dep_ in ("conj","cc")),
        "n_mod": sum(1 for t in sent if t.dep_ in ("amod","advmod","nmod","appos","nummod","poss","det")),
        "mean_dist": 0.0, "max_depth": 0, "depth_var": 0.0,
        "opener": toks[0].pos_ if toks else "X",
        "stype": "complex" if sum(1 for t in sent if t.dep_ in CLAUSE_DEPS) > 0
                  else ("conjunctive" if sum(1 for t in sent if t.dep_ in ("conj","cc")) > 0
                        else "simple"),
        "has_passive": any(t.dep_ in ("nsubjpass","auxpass") for t in sent),
        "is_interrog": sent.text.rstrip().endswith("?"),
        "has_cond": any(m in sent.text.lower() for m in (" if "," when "," unless "," whenever ")),
        "acl": sum(1 for t in sent if t.dep_ == "acl"),
        "advcl": sum(1 for t in sent if t.dep_ == "advcl"),
        "ccomp": sum(1 for t in sent if t.dep_ == "ccomp"),
        "xcomp": sum(1 for t in sent if t.dep_ == "xcomp"),
        "relcl": sum(1 for t in sent if t.dep_ == "relcl"),
    }
    pos_seq = [t.pos_ for t in toks]
    pos_counts = Counter(pos_seq)
    for p in POS_TAGS:
        feats[f"pos_{p}"] = pos_counts.get(p, 0)
    pos_bigram_counts = Counter()
    for i in range(len(pos_seq)-1):
        pos_bigram_counts[(pos_seq[i], pos_seq[i+1])] += 1
    for bg in POS_BIGRAMS:
        feats[f"posbg_{bg[0]}_{bg[1]}"] = pos_bigram_counts.get(bg, 0)
    pos_trigram_counts = Counter()
    for i in range(len(pos_seq)-2):
        pos_trigram_counts[(pos_seq[i], pos_seq[i+1], pos_seq[i+2])] += 1
    for tg in POS_TRIGRAMS:
        feats[f"postg_{tg[0]}_{tg[1]}_{tg[2]}"] = pos_trigram_counts.get(tg, 0)
    dep_counts = Counter(t.dep_ for t in sent)
    for d in DEP_RELS:
        feats[f"dep_{d}"] = dep_counts.get(d, 0)
    dep_seq = [t.dep_ for t in sent if t.dep_ != "punct"]
    dep_bigram_counts = Counter()
    for i in range(len(dep_seq)-1):
        dep_bigram_counts[(dep_seq[i], dep_seq[i+1])] += 1
    for bg in DEP_BIGRAMS:
        feats[f"depbg_{bg[0]}_{bg[1]}"] = dep_bigram_counts.get(bg, 0)
    root = next((t for t in sent if t.dep_ == "root"), None)
    if root is None:
        feats["main_SVO"] = feats["main_SVC"] = feats["main_SV"] = 0
        feats["main_SVA"] = feats["main_SVOA"] = feats["main_SVOC"] = 0
        feats["main_SVOO"] = feats["main_SVO_IOBJ"] = feats["main_EXISTS"] = 0
    else:
        has_nsubj = any(c.dep_ in ("nsubj","nsubjpass") for c in root.children)
        has_obj = any(c.dep_ in ("obj","iobj") for c in root.children)
        has_attr = any(c.dep_ in ("attr","acomp","oprd") for c in root.children)
        has_obl = any(c.dep_ == "obl" for c in root.children)
        n_objs = sum(1 for c in root.children if c.dep_ in ("obj","iobj"))
        feats["main_SVO"] = int(has_nsubj and n_objs >= 1 and not has_obl)
        feats["main_SVC"] = int(has_nsubj and has_attr)
        feats["main_SV"] = int(has_nsubj and not has_obj and not has_attr)
        feats["main_SVA"] = int(has_nsubj and has_obl and not has_obj)
        feats["main_SVOA"] = int(has_nsubj and n_objs >= 1 and has_obl)
        feats["main_SVOC"] = int(has_nsubj and has_attr and n_objs >= 1)
        feats["main_SVOO"] = int(has_nsubj and n_objs >= 2)
        feats["main_SVO_IOBJ"] = int(has_nsubj and any(c.dep_ == "iobj" for c in root.children))
        feats["main_EXISTS"] = int(any(t.dep_ in ("expl","nsubj") and t.pos_ in ("PRON","DET") and t.text.lower() in ("there","it") for t in sent))
    sent_clause_deps = {t.dep_ for t in sent if t.dep_ in CLAUSE_DEPS}
    for pair in CLAUSE_PAIRS:
        feats[f"clpair_{pair[0]}_{pair[1]}"] = int(pair[0] in sent_clause_deps and pair[1] in sent_clause_deps)
    def height(token, memo):
        if token in memo: return memo[token]
        children = list(token.children)
        if not children: memo[token] = 1; return 1
        h = 1 + max(height(c, memo) for c in children)
        memo[token] = h; return h
    memo = {}
    depths_all = [height(t, memo) for t in sent]
    clause_depths = [height(t, memo) for t in sent if t.dep_ in CLAUSE_DEPS]
    feats["nest_max"] = max(clause_depths) if clause_depths else 0
    feats["nest_mean"] = float(np.mean(clause_depths)) if clause_depths else 0.0
    feats["nest_std"] = float(np.std(clause_depths)) if clause_depths else 0.0
    for d in (1,2,3,4,5):
        feats[f"nest_d{d}"] = int(feats["nest_max"] == d)
    feats["nest_ge2"] = int(feats["nest_max"] >= 2)
    feats["nest_ge3"] = int(feats["nest_max"] >= 3)
    for pos in range(min(3, len(pos_seq))):
        feats[f"open_p{pos+1}_{pos_seq[pos]}"] = 1
    for pos in range(min(3, len(pos_seq))):
        feats[f"close_p{pos+1}_{pos_seq[-(pos+1)]}"] = 1
    dists = [abs(t.head.i - t.i) for t in sent if t.head.i != t.i]
    feats["mean_dist"] = float(np.mean(dists)) if dists else 0.0
    feats["max_depth"] = max(depths_all) if depths_all else 0
    feats["depth_var"] = float(np.var(depths_all)) if depths_all else 0.0
    for lo, hi in [(0,1),(1,2),(2,3),(3,5),(5,100)]:
        feats[f"dist_{lo}_{hi}"] = sum(1 for d in dists if lo <= d < hi)
    for d in (1,2,3,4,5):
        feats[f"depth_eq{d}"] = sum(1 for x in depths_all if x == d)
    text = sent.text
    feats["n_punct_total"] = sum(1 for c in text if c in ",.;:!?\"'-()")
    for p in PUNCT_TAGS:
        feats[f"punct_{p}"] = text.count(p)

    # ── New structural features ────────────────────────────────────────────────────────

    # Subtree / tree-shape statistics
    def subtree_size(t):
        return 1 + sum(subtree_size(c) for c in t.children)
    subtree_sizes = [subtree_size(t) for t in sent]
    feats["subtree_mean"] = float(np.mean(subtree_sizes)) if subtree_sizes else 0.0
    feats["subtree_max"] = max(subtree_sizes) if subtree_sizes else 0
    feats["subtree_std"] = float(np.std(subtree_sizes)) if len(subtree_sizes) > 1 else 0.0
    n_leaf = sum(1 for t in sent if len(list(t.children)) == 0)
    feats["leaf_ratio"] = n_leaf / max(len(sent), 1)

    # Branching factor
    n_internal = len(sent) - n_leaf
    feats["branch_factor"] = (len(sent) - 1) / max(n_internal, 1)

    # Coordination structure
    n_conj = sum(1 for t in sent if t.dep_ == "conj")
    feats["n_conj"] = n_conj
    feats["has_conj_and"] = int(any(t.text.lower() == "and" for t in sent if t.dep_ == "cc"))
    feats["has_conj_or"] = int(any(t.text.lower() in ("or", "nor") for t in sent if t.dep_ == "cc"))

    # Prepositional phrase / pobj
    n_pobj = sum(1 for t in sent if t.dep_ == "pobj")
    n_pp = sum(1 for t in sent if t.dep_ in ("prep",))
    feats["n_pobj"] = n_pobj
    feats["n_pp"] = n_pp
    feats["pp_depth"] = sum(1 for t in sent if t.dep_ == "prep")

    # Noun phrase chunks (spaCy noun_chunks API)
    np_chunks = list(sent.doc.noun_chunks)
    feats["n_np_chunks"] = len(np_chunks)
    feats["np_chunk_width_mean"] = float(np.mean([len(list(nc.root.subtree)) for nc in np_chunks])) if np_chunks else 0.0
    n_noun = sum(1 for t in sent if t.pos_ == "NOUN")
    feats["n_noun"] = n_noun
    feats["mod_per_noun"] = feats["n_mod"] / max(n_noun, 1)

    # Verb phrase / auxiliary complexity
    verbs = [t for t in sent if t.pos_ == "VERB"]
    feats["n_verb"] = len(verbs)
    n_aux = sum(1 for t in sent if t.pos_ == "AUX")
    feats["n_aux"] = n_aux
    feats["vp_complexity"] = n_aux / max(len(verbs), 1)

    # Wh-movement
    wh_words = ("who", "what", "where", "when", "why", "how", "which", "whom", "whose")
    feats["is_wh_question"] = int(any(t.text.lower().rstrip("?") in wh_words for t in sent))

    # Morphological richness
    lemmas = set(t.lemma_.lower() for t in sent if not t.is_punct and not t.is_space)
    feats["type_token_ratio"] = len(lemmas) / max(len([t for t in sent if not t.is_punct and not t.is_space]), 1)

    # Stop-word / content-word ratio
    stop_words = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those"}
    tokens_non_punct = [t for t in sent if not t.is_punct and not t.is_space]
    feats["stop_word_ratio"] = sum(1 for t in tokens_non_punct if t.text.lower() in stop_words) / max(len(tokens_non_punct), 1)
    feats["content_word_ratio"] = 1 - feats["stop_word_ratio"]

    # Word-length statistics
    word_lens = [len(t.text) for t in sent if not t.is_space]
    feats["mean_word_len"] = float(np.mean(word_lens)) if word_lens else 0.0
    feats["max_word_len"] = max(word_lens) if word_lens else 0
    feats["word_len_std"] = float(np.std(word_lens)) if len(word_lens) > 1 else 0.0

    # Punctuation patterns
    feats["comma_per_clause"] = text.count(",") / max(feats["n_clause"], 1)
    feats["has_dash"] = int("-" in text or "—" in text)
    feats["has_quote"] = int("'" in text or '"' in text)
    feats["has_ellipsis"] = int("..." in text)

    # Readability proxies (no semantic knowledge needed)
    feats["flesch_approx"] = (206.835 - 1.015 * feats["n_tok"] / max(feats["n_clause"], 1) - 84.6 * sum(1 for t in sent if t.pos_ == "VERB") / max(feats["n_clause"], 1)) if feats["n_clause"] > 0 else 0.0

    # ARIndex approximation
    feats["ari_approx"] = (4.71 * sum(word_lens) / max(len(word_lens), 1) + 0.5 * feats["n_tok"] / max(feats["n_clause"], 1) - 21.43) if feats["n_clause"] > 0 else 0.0

    # Expanded dependency counts
    feats["dep_nsubj_count"] = sum(1 for t in sent if t.dep_ == "nsubj")
    feats["dep_nsubjpass_count"] = sum(1 for t in sent if t.dep_ == "nsubjpass")
    feats["dep_nummod_count"] = sum(1 for t in sent if t.dep_ == "nummod")

    # Left/right branching
    feats["is_left_branching"] = int(feats["subtree_max"] > feats.get("subtree_mean", 0) * 1.5) if feats.get("subtree_mean", 0) > 0 else 0

    # Prefix/suffix features
    if toks:
        first_tok = toks[0]
        feats["prefix_2"] = first_tok.prefix_ if hasattr(first_tok, "prefix_") else ""
        feats["suffix_2"] = first_tok.suffix_ if hasattr(first_tok, "suffix_") else ""

    # Conjoin complexity
    feats["n_conj_per_conj"] = feats["n_conj"] / max(feats["n_coord"], 1)

    # Sentence length buckets
    feats["len_bucket_10"] = int(feats["n_tok"] <= 10)
    feats["len_bucket_20"] = int(10 < feats["n_tok"] <= 20)
    feats["len_bucket_30"] = int(20 < feats["n_tok"] <= 30)
    feats["len_bucket_40"] = int(30 < feats["n_tok"] <= 40)
    feats["len_bucket_50"] = int(feats["n_tok"] > 40)

    # Embed depth from root
    root_depths = []
    for t in sent:
        depth = 0
        cur = t
        while cur.head != cur and cur.head is not None:
            depth += 1
            cur = cur.head
            if depth > 100:
                break
        root_depths.append(depth)
    feats["mean_root_depth"] = float(np.mean(root_depths)) if root_depths else 0.0
    feats["max_root_depth"] = max(root_depths) if root_depths else 0

    # ── More structural: subtree quantiles ────────────────────────────────────────────
    if subtree_sizes:
        sorted_ss = sorted(subtree_sizes)
        n = len(sorted_ss)
        feats["subtree_p25"] = sorted_ss[int(n * 0.25)]
        feats["subtree_p50"] = sorted_ss[int(n * 0.50)]
        feats["subtree_p75"] = sorted_ss[int(n * 0.75)]
        feats["subtree_iqr"] = feats["subtree_p75"] - feats["subtree_p25"]
        mean_ss = np.mean(subtree_sizes)
        var_ss = np.var(subtree_sizes) if len(subtree_sizes) > 1 else 0.0
        feats["subtree_skew"] = float(np.mean([((s - mean_ss) ** 3) / max(var_ss ** 1.5, 1e-9) for s in subtree_sizes])) if var_ss > 1e-9 else 0.0
    else:
        feats["subtree_p25"] = feats["subtree_p50"] = feats["subtree_p75"] = 0.0
        feats["subtree_iqr"] = feats["subtree_skew"] = 0.0

    # ── Word shape patterns ─────────────────────────────────────────────────────────────
    def word_shape(t):
        s = t.text
        if s.isupper(): return "UPPER"
        if s.islower(): return "lower"
        if s.istitle(): return "Title"
        if s.isdigit(): return "digit"
        if s.isalpha(): return "mixed"
        return "other"
    shape_counts = Counter(word_shape(t) for t in toks)
    for ws in ("lower", "Title", "UPPER", "digit", "mixed", "other"):
        feats[f"shape_{ws}"] = shape_counts.get(ws, 0)

    # ── More punctuation ───────────────────────────────────────────────────────────────
    feats["n_semicol"] = text.count(";")
    feats["n_colon"] = text.count(":")
    feats["excl_per_clause"] = text.count("!") / max(feats["n_clause"], 1)
    feats["n_multi_punct"] = sum(1 for i in range(1, len(text)) if text[i] == text[i-1] and text[i] in "!?.")
    feats["n_question"] = text.count("?")
    feats["capitalized_ratio"] = sum(1 for t in toks if t.text[0].isupper() if t.text) / max(len(toks), 1)
    feats["all_caps_words"] = sum(1 for t in toks if t.text.isupper() and len(t.text) > 1)
    feats["has_paren"] = int("(" in text or ")" in text)
    feats["paren_depth"] = max(0, sum(1 for c in text if c == "(") - sum(1 for c in text if c == ")"))

    # ── More dependency relation features ─────────────────────────────────────────────
    n_tok_safe = max(len(sent), 1)
    extra_deps = ("det", "case", "amod", "advmod", "nmod", "compound", "flat", "mark", "ccomp", "orphan", "agent")
    for d in extra_deps:
        feats[f"dep_{d}_count"] = sum(1 for t in sent if t.dep_ == d)
    # Ratios
    feats["det_per_noun"] = feats.get("dep_det_count", 0) / max(n_noun, 1)
    feats["case_per_noun"] = feats.get("dep_case_count", 0) / max(n_noun, 1)
    feats["amod_per_noun"] = feats.get("dep_amod_count", 0) / max(n_noun, 1)
    feats["advmod_per_verb"] = feats.get("dep_advmod_count", 0) / max(len(verbs), 1)
    feats["compound_per_noun"] = feats.get("dep_compound_count", 0) / max(n_noun, 1)
    feats["dep_diversity"] = len(set(t.dep_ for t in sent)) / n_tok_safe
    feats["n_dep_relations"] = len(set(t.dep_ for t in sent))

    # ── Tree width at each depth ───────────────────────────────────────────────────────
    depth_to_width = {}
    for t, d in zip(sent, root_depths):
        depth_to_width[d] = depth_to_width.get(d, 0) + 1
    if depth_to_width:
        widths = list(depth_to_width.values())
        feats["tree_width_max"] = max(widths)
        feats["tree_width_mean"] = float(np.mean(widths))
        feats["tree_depth_to_width_ratio"] = feats["max_root_depth"] / max(feats["tree_width_max"], 1)
    else:
        feats["tree_width_max"] = feats["tree_width_mean"] = 0.0
        feats["tree_depth_to_width_ratio"] = 0.0

    # ── Entropy of POS and dep distributions ───────────────────────────────────────────
    pos_counter = Counter(t.pos_ for t in sent)
    dep_counter = Counter(t.dep_ for t in sent)
    def entropy(counter):
        total = sum(counter.values())
        if total == 0: return 0.0
        probs = [c / total for c in counter.values()]
        return -sum(p * np.log2(max(p, 1e-9)) for p in probs)
    feats["pos_entropy"] = entropy(pos_counter)
    feats["dep_entropy"] = entropy(dep_counter)
    feats["pos_diversity"] = len(pos_counter) / n_tok_safe
    feats["dep_diversity_overall"] = len(dep_counter) / n_tok_safe

    # ── Chunk / span features ─────────────────────────────────────────────────────────
    if len(toks) >= 3:
        chunk_size = max(1, len(toks) // 3)
        feats["chunk_begin_width"] = len([t for t in toks[:chunk_size] if not t.is_space])
        feats["chunk_mid_width"] = len([t for t in toks[chunk_size:2*chunk_size] if not t.is_space])
        feats["chunk_end_width"] = len([t for t in toks[2*chunk_size:] if not t.is_space])
        feats["chunk_begin_ratio"] = feats["chunk_begin_width"] / max(len(toks), 1)
        feats["chunk_end_ratio"] = feats["chunk_end_width"] / max(len(toks), 1)
    else:
        feats["chunk_begin_width"] = feats["chunk_mid_width"] = feats["chunk_end_width"] = 0
        feats["chunk_begin_ratio"] = feats["chunk_end_ratio"] = 0.0

    # ── Word length quantiles ─────────────────────────────────────────────────────────
    if word_lens:
        sorted_lens = sorted(word_lens)
        n = len(sorted_lens)
        feats["word_len_p25"] = sorted_lens[int(n * 0.25)]
        feats["word_len_p75"] = sorted_lens[int(n * 0.75)]
        feats["word_len_iqr"] = feats["word_len_p75"] - feats["word_len_p25"]
        feats["word_len_range"] = max(word_lens) - min(word_lens)
    else:
        feats["word_len_p25"] = feats["word_len_p75"] = feats["word_len_iqr"] = feats["word_len_range"] = 0.0

    # ── Gini coefficient of word lengths ───────────────────────────────────────────────
    if word_lens and len(word_lens) > 1:
        sorted_lens_s = sorted(word_lens)
        n = len(sorted_lens_s)
        cum = np.cumsum(sorted_lens_s)
        feats["gini_word_len"] = (2 * np.sum((np.arange(1, n+1) * sorted_lens_s)) - (n + 1) * cum[-1]) / (n * cum[-1]) if cum[-1] > 0 else 0.0
    else:
        feats["gini_word_len"] = 0.0

    # ── Capitalization of first word ─────────────────────────────────────────────────
    feats["first_word_lower"] = int(toks[0].text[0].islower()) if toks and toks[0].text else 0
    feats["first_word_upper"] = int(toks[0].text[0].isupper()) if toks and toks[0].text else 0

    # ── Sentence-end punctuation type ────────────────────────────────────────────────
    text_stripped = text.rstrip()
    feats["ends_period"] = int(text_stripped.endswith("."))
    feats["ends_excl"] = int(text_stripped.endswith("!"))
    feats["ends_quest"] = int(text_stripped.endswith("?"))
    feats["ends_comma"] = int(text_stripped.endswith(","))
    feats["ends_ellipsis"] = int(text_stripped.endswith("..."))

    return feats


def stage2_extract_features():
    """对 sentences_for_rewrite_10k.jsonl 做 318d 句法特征提取，输出 gzip 缓存。"""
    import spacy
    import hashlib
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner","lemmatizer","attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    CACHE_META_VERSION = "v3"  # v3: inline per_sentence_features_v2

    def sent_key(text: str) -> str:
        return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()

    # Load existing cache
    cache: dict = {}
    if FEAT_CACHE.exists():
        try:
            with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
                header = f.readline()
                if header.startswith("#META "):
                    meta = json.loads(header[len("#META "):])
                    if meta.get("version") == CACHE_META_VERSION:
                        for line in f:
                            line = line.strip()
                            if line:
                                rec = json.loads(line)
                                cache[rec["k"]] = rec["v"]
                        print(f"[Stage 2] 加载缓存 {len(cache)} 条")
                    else:
                        print(f"[Stage 2] 缓存版本不匹配，清空重建")
                        cache = {}
        except Exception:
            cache = {}

    # Load sentences
    sents = []
    if SENTS_OUT.exists():
        with open(SENTS_OUT, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sents.append(json.loads(line))
                except Exception:
                    pass
    print(f"[Stage 2] {len(sents)} 句子待处理")

    miss_sents = []
    miss_keys = set()
    for s in sents:
        k = sent_key(s.get("sentence_text", ""))
        if k not in cache and k not in miss_keys:
            miss_keys.add(k)
            miss_sents.append(s)

    print(f"[Stage 2] 缓存命中 {len(sents)-len(miss_sents)}，需解析 {len(miss_sents)}")

    if miss_sents:
        t0 = time.time()
        new_feats = 0
        for i in range(0, len(miss_sents), 256):
            batch = miss_sents[i:i+256]
            texts = [s.get("sentence_text","") for s in batch]
            for doc in nlp.pipe(texts, batch_size=256):
                for sent in doc.sents:
                    sf = per_sentence_features_v2(sent)
                    if sf is not None:
                        k = sent_key(sent.text)
                        cache[k] = sf
                        new_feats += 1
            done = min(i+256, len(miss_sents))
            if done % 2000 < 256 or done == len(miss_sents):
                elapsed = time.time()-t0
                rate = done/max(elapsed, 1e-6)
                print(f"  parsed {done}/{len(miss_sents)} ({rate:.0f}/s)")

        print(f"[Stage 2] 新增 {new_feats} 条特征")

    # Save cache
    FEAT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FEAT_CACHE.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        meta = {"version": CACHE_META_VERSION, "n_entries": len(cache)}
        f.write("#META " + json.dumps(meta) + "\n")
        for k, v in cache.items():
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
    tmp.replace(FEAT_CACHE)
    print(f"[Stage 2] 缓存已保存: {FEAT_CACHE} ({len(cache)} 条)")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: 运行 gaussian_vades diagonal_residual
# ══════════════════════════════════════════════════════════════════════════════
def stage3_gaussian_vades():
    """调用 gaussian_vades diagonal_residual 模式。"""
    print("[Stage 3] 运行 gaussian_vades diagonal_residual...")
    print("  提示: VADES_COVARIANCE_MODE=diagonal_residual 已内置到 gaussian_vades.py")
    print("  用法: VADES_COVARIANCE_MODE=diagonal_residual python gaussian/gaussian_vades.py smoke")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    t0 = time.time()
    stage = os.environ.get("STAGE", "all")

    if stage in ("1", "all"):
        stage1_extract_sentences()
    if stage in ("2", "all"):
        stage2_extract_features()
    if stage in ("3", "all"):
        stage3_gaussian_vades()

    print(f"\n[Total] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
