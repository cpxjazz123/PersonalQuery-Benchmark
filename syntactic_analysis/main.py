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

DATA_SOURCE = SCRATCH / "stage1_filtered_users_reviews_10000u.json"
SENTS_OUT = SCRATCH / "sentences_for_rewrite_10k.jsonl"
FEAT_CACHE = SCRATCH / "sentences_318d_cache.jsonl.gz"

# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: spaCy 切句
# ══════════════════════════════════════════════════════════════════════════════
def stage1_extract_sentences():
    """spaCy 批量切句，输出 sentences_for_rewrite_10k.jsonl。"""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    MIN_WORDS, MAX_WORDS, SENTS_PER_USER = 5, 60, 15

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
