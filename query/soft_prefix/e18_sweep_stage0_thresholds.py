#!/usr/bin/env python3
"""E18 Stage 0 Threshold Sweep：在不同 (MIN_WORDS, MAX_WORDS, MIN_LONG_SENTENCES) 配置下
衡量 user style vector 之间的判别力。

策略：
  - 从 raw reviews_Baby_5.json.gz 加载所有 review
  - 对每个配置 (网格扫描)：
    1. 按 Stage 0 规则过滤 user（至少 N 个长度在 [min, max] 词的句子）
    2. 用 spaCy 跑 nlp.pipe 计算每个 user 的 20 个句法特征 + 10 个 opener
    3. 计算 user 间风格向量的判别力（inter-user variance + 配对距离）
  - 输出每个配置的 n_users / inter_var / mean_dist / AUC_like

输出：result/personal_query/e18/e18_stage0_sweep.json
"""
from __future__ import annotations

import gzip
import json
import re
import time
from itertools import product
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
E18 = REPO_ROOT / "result" / "personal_query" / "e18"
E18.mkdir(parents=True, exist_ok=True)

# Same input as Stage 0
REVIEW_FILE = "/fs04/ar57/wenyu/data/Amazon-Reviews-2018/reviews_Baby_5.json.gz"
META_FILE = "/fs04/ar57/wenyu/data/Amazon-Reviews-2018/meta_Baby.jsonl.gz"

# Opener classes (mirror of OPENER_CLASSES)
OPENER_CLASSES = [
    "PRON", "ADV", "DET", "NOUN", "VERB", "CCONJ", "ADP", "PROPN", "NUM", "OTHER",
]

# Grid
MIN_WORDS_GRID = [5, 10, 15, 20, 25]
MAX_WORDS_GRID = [25, 35, 50, 100]
MIN_LONG_SENTENCES_GRID = [5, 10, 15, 20]
SAMPLE_N_USERS = 200  # subsample for speed
SEED = 42


def split_sents(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text or "")


def count_long_sents(text: str, mn: int, mx: int) -> int:
    return sum(1 for s in split_sents(text) if mn <= len(s.split()) <= mx)


def opener_class(word: str) -> str:
    w = word.lower()
    if w in {"i", "we", "you", "they", "he", "she", "it", "my", "me", "our"}:
        return "PRON"
    if w in {"the", "a", "an", "this", "that", "these", "those", "my", "your"}:
        return "DET"
    if w in {"and", "but", "or", "so", "because", "although", "though"}:
        return "CCONJ"
    if w in {"in", "on", "at", "for", "with", "of", "to", "after", "before", "during"}:
        return "ADP"
    if w.endswith("ly"):
        return "ADV"
    return "OTHER"


def first_content_word(text: str) -> str | None:
    """Naive 'first content word' for opener hist (no spaCy)."""
    for tok in re.findall(r"[A-Za-z']+", text):
        if tok.strip():
            return tok
    return None


def dep_features_from_doc(doc) -> list[float]:
    """Compute 20 features per spaCy doc."""
    if not doc or len(list(doc)) == 0:
        return [0.0] * 20
    toks = list(doc)
    n = len(toks)
    # depths (subtree root = 0; depth = -i.head chain length)
    depths = []
    for t in toks:
        d = 0
        h = t
        seen = set()
        while h.head is not h and h.head not in seen and d < 50:
            seen.add(h)
            h = h.head
            d += 1
        depths.append(d)
    depths = np.array(depths, dtype=np.float64)
    max_dep_depth = float(depths.max()) if len(depths) else 0.0
    mean_dep_depth = float(depths.mean()) if len(depths) else 0.0
    dep_tree_height = max_dep_depth
    depth_var = float(depths.var()) if len(depths) else 0.0

    # Subtree / branch
    max_branch = 0
    for t in toks:
        c = len(list(t.children))
        if c > max_branch:
            max_branch = c

    # Counts
    def cnt(dep):
        return sum(1 for t in toks if t.dep_ == dep)

    acl = cnt("acl")
    relcl = cnt("relcl")
    ccomp = cnt("ccomp")
    xcomp = cnt("xcomp")
    advcl = cnt("advcl")
    amod = cnt("amod")
    advmod = cnt("advmod")
    nmod = cnt("nmod")
    compound = cnt("compound")
    coord = cnt("conj")

    # Modifier density = (amod + advmod + nmod) / n_tokens
    mod_density = (amod + advmod + nmod) / max(n, 1)

    # dependency distance: |t.i - t.head.i|
    dists = []
    for t in toks:
        if t.head is not t:
            dists.append(abs(t.i - t.head.i))
    dists = np.array(dists, dtype=np.float64)
    mean_dd = float(dists.mean()) if len(dists) else 0.0
    max_dd = float(dists.max()) if len(dists) else 0.0
    long_dd_ratio = float((dists > 5).sum() / max(len(dists), 1))

    # clause nesting depth (max depth for tokens with dep in {acl, relcl, ccomp, xcomp, advcl})
    clause_depths = [depths[i] for i, t in enumerate(toks) if t.dep_ in {"acl", "relcl", "ccomp", "xcomp", "advcl"}]
    clause_nesting = float(max(clause_depths)) if clause_depths else 0.0

    return [
        max_dep_depth, mean_dep_depth, dep_tree_height, depth_var,
        float(acl), float(relcl), float(ccomp), float(xcomp), float(advcl),
        clause_nesting, mean_dd, max_dd, long_dd_ratio,
        float(amod), float(advmod), float(nmod), float(compound),
        mod_density, float(coord), float(max_branch),
    ]


def user_features(user_texts: list[str], nlp) -> tuple[np.ndarray, np.ndarray]:
    """Given a list of long-sentence texts (already filtered by min/max words),
    compute per-sentence features + opener classes, then aggregate to user vector."""
    if not user_texts:
        return np.zeros(20, dtype=np.float32), np.zeros(10, dtype=np.float32)
    sent_feats = []
    openers = []
    for doc in nlp.pipe(user_texts, batch_size=64):
        if not doc or len(list(doc)) < 3:
            continue
        sent_feats.append(dep_features_from_doc(doc))
        first = first_content_word(doc.text)
        if first:
            openers.append(OPENER_CLASSES.index(opener_class(first)) if opener_class(first) in OPENER_CLASSES else OPENER_CLASSES.index("OTHER"))
    if not sent_feats:
        return np.zeros(20, dtype=np.float32), np.zeros(10, dtype=np.float32)
    fmean = np.mean(sent_feats, axis=0)
    fmean = fmean / (np.linalg.norm(fmean) + 1e-12)
    ohist = np.zeros(10, dtype=np.float32)
    for c in openers:
        ohist[c] += 1
    ohist = ohist / max(len(openers), 1)
    return fmean.astype(np.float32), ohist.astype(np.float32)


def main() -> None:
    print(f"[load] scanning reviews...", flush=True)
    user_long_counts = {}  # user_id -> {config_idx: count}
    user_texts_cache = {}  # user_id -> {min_words, max_words: [sentences]}
    with gzip.open(REVIEW_FILE, "rt") as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            uid = r.get("reviewerID") or r.get("user_id")
            text = r.get("reviewText") or r.get("text") or ""
            if not uid or not text:
                continue
            sents = split_sents(text)
            if not sents:
                continue
            # For each (mn, mx) we may need different subsets; just store per-sentence word counts
            word_counts = [len(s.split()) for s in sents]
            if uid not in user_texts_cache:
                user_texts_cache[uid] = []
            user_texts_cache[uid].extend(zip(word_counts, sents))
            if i % 200000 == 0 and i > 0:
                print(f"  scanned {i} reviews, users={len(user_texts_cache)}", flush=True)
    print(f"[load] users total: {len(user_texts_cache)}", flush=True)

    # Compute per-user long-sentence counts for each (mn, mx) in grid
    print(f"[prep] computing long-sentence counts per (mn, mx)...", flush=True)
    user_sents_by_config = {}  # (uid, mn, mx) -> [sentences]
    user_count_by_config = {}  # (uid, mn, mx) -> int
    for uid, sents in user_texts_cache.items():
        for wc, st in sents:
            for mn in MIN_WORDS_GRID:
                for mx in MAX_WORDS_GRID:
                    if mn <= wc <= mx:
                        user_sents_by_config.setdefault((uid, mn, mx), []).append(st)
        for (uid2, mn, mx), ss in user_sents_by_config.items():
            if uid2 == uid:
                user_count_by_config[(uid, mn, mx)] = len(ss)

    # Load spaCy
    print(f"[nlp] loading spaCy...", flush=True)
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except Exception:
        nlp = spacy.load("en_core_web_sm")
    print(f"[nlp] OK", flush=True)

    rng = np.random.default_rng(SEED)

    # Sweep configurations
    results = []
    total = len(MIN_WORDS_GRID) * len(MAX_WORDS_GRID) * len(MIN_LONG_SENTENCES_GRID)
    print(f"[sweep] {total} configs, each sampling {SAMPLE_N_USERS} users", flush=True)
    done = 0
    t0 = time.time()
    for mn, mx, mls in product(MIN_WORDS_GRID, MAX_WORDS_GRID, MIN_LONG_SENTENCES_GRID):
        done += 1
        # Users with ≥ mls long-sentences in [mn, mx]
        qual = [uid for uid in user_texts_cache
                if user_count_by_config.get((uid, mn, mx), 0) >= mls]
        if len(qual) < 30:
            results.append({"mn": mn, "mx": mx, "mls": mls, "n_users": len(qual),
                            "skipped": "too_few_users"})
            print(f"  [{done}/{total}] ({mn},{mx},{mls}): n_users={len(qual)} SKIP", flush=True)
            continue
        sample = list(rng.choice(qual, size=min(SAMPLE_N_USERS, len(qual)), replace=False))

        # Compute style vectors
        vecs = []
        for uid in sample:
            texts = user_sents_by_config.get((uid, mn, mx), [])[:12]  # MAX_SENTENCES = 12
            fmean, ohist = user_features(texts, nlp)
            vecs.append(np.concatenate([fmean, ohist]))
        mat = np.stack(vecs)
        # Inter-user variance (mean across dims)
        inter_var = float(mat.var(axis=0).mean())
        # Pairwise L2 distance (sample 50 users for speed)
        sub = mat[rng.choice(len(sample), size=min(50, len(sample)), replace=False)]
        from scipy.spatial.distance import pdist
        dists = pdist(sub, metric="euclidean")
        mean_dist = float(dists.mean())
        std_dist = float(dists.std())
        # Dims with std == 0
        n_zero_dims = int((mat.std(axis=0) == 0).sum())

        results.append({
            "mn": mn, "mx": mx, "mls": mls,
            "n_users": len(qual),
            "n_sampled": len(sample),
            "inter_var": inter_var,
            "mean_pair_dist": mean_dist,
            "std_pair_dist": std_dist,
            "n_zero_dims": n_zero_dims,
        })
        print(f"  [{done}/{total}] ({mn},{mx},{mls}): n_users={len(qual)} inter_var={inter_var:.5f} "
              f"mean_dist={mean_dist:.4f} zero_dims={n_zero_dims}", flush=True)

    # Save
    out = {
        "version": "e18-stage0-sweep-v1",
        "seed": SEED,
        "sample_n_users": SAMPLE_N_USERS,
        "review_file": REVIEW_FILE,
        "grid": {
            "min_words": MIN_WORDS_GRID,
            "max_words": MAX_WORDS_GRID,
            "min_long_sentences": MIN_LONG_SENTENCES_GRID,
        },
        "results": results,
        "baseline_existing": {
            "min_words": 15, "max_words": 35, "min_long_sentences": 10,
            "n_users_actual": 7681,
            "inter_var": 0.002613, "auc_step4": 0.4796,
            "d_z_correct_vs_shuffled": -0.1895,
            "note": "computed on e17_style_vectors.json (400 users sampled from 5031 candidates)",
        },
    }
    json.dump(out, open(E18 / "e18_stage0_sweep.json", "w"), indent=2)
    print(f"\n[saved] {E18 / 'e18_stage0_sweep.json'}", flush=True)
    print(f"[done] total time {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
