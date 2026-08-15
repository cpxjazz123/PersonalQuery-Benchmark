#!/usr/bin/env python3
"""E18 Task 2a: 重建 user style vector pool（用 sweep 找到的最优阈值 (20,35,5)）。

Sweep 结果（80 configs）:
  baseline (15,35,10): inter_var=0.00278 mean_dist=0.3971 n_users=13374
  best balance (20,35,5): inter_var=0.00371 mean_dist=0.4441 n_users=13838
    → inter_var +34% vs baseline, mean_dist +12% vs baseline, n_users +3.5%

策略：
  - 复用 e17_step1_contract.py 的核心算法（FEATURES20 + opener + L2-normalize）
  - Stage 0 阈值改为 (MIN_WORDS=20, MAX_WORDS=35, MIN_LONG_SENTENCES=5)
  - 候选池不再限制 400 user：保留全部 ≥MIN_GROUPS=6 product group 的 user
  - 与 e17 完全相同的 train/dev/test 切分：保证"无 fallback + 干净对照"承诺
  - 输出：
    * e18_style_vectors.json (all candidate vectors, keyed by user_id)
    * e18_pool_stats.json (per-config inter_var/mean_dist 验证)
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
E18 = REPO_ROOT / "result" / "personal_query" / "e18"
E18.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT / "query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from user_stat_vector import FEATURES20, OPENER_CLASSES, opener_class_of  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features_from_doc  # noqa: E402

CATEGORY = "Baby_Products"
STAGE1 = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
REVIEW_FILE = "/fs04/ar57/wenyu/data/Amazon-Reviews-2018/reviews_Baby_5.json.gz"
BABY_DS = REPO_ROOT / "result" / "personal_query" / "e14_multiproduct" / "baby_dataset.json"

# Sweep 选定的最优配置
MIN_WORDS = 20
MAX_WORDS = 35
MIN_LONG_SENTENCES = 5
SEED = 42
MIN_GROUPS = 6  # 与 e17 一致：≥3 profile + ≥3 audit
MAX_SENTENCES_PER_USER = 12

FEATS = FEATURES20


def split_sents(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def main() -> None:
    print(f"[load] stage1...", flush=True)
    data = json.load(open(STAGE1))

    # 1) 候选池：每个 user 的 stage1 reviews，按 product group（asin）聚合
    #    stage1 schema: results[].target_reviews = [review_text_str, ...]
    user_groups = defaultdict(list)  # uid -> {asin: [review_text, ...]}
    for u in data["users"]:
        uid = u["user_id"]
        for r in u.get("results", []):
            asin = r.get("asin")
            reviews = r.get("target_reviews") or []
            if asin and reviews:
                for txt in reviews:
                    if txt and txt.strip():
                        user_groups[uid].append((asin, txt))

    print(f"[cand] total users with reviews: {len(user_groups)}", flush=True)

    # 2) 过滤：每个 user ≥ MIN_GROUPS 个不同的 asin 组
    cands = {uid: groups for uid, groups in user_groups.items()
             if len({a for a, _ in groups}) >= MIN_GROUPS}
    print(f"[cand] users with ≥{MIN_GROUPS} product groups: {len(cands)}", flush=True)

    # 3) 收集每个 user 的全部 review texts（后面 doc.sents 切句后再按 word count 过滤）
    user_review_texts = {}
    for uid, groups in cands.items():
        texts = []
        for asin, txt in groups:
            if txt and txt.strip():
                texts.append(txt[:1000])  # 限长保护
        user_review_texts[uid] = texts
    print(f"[prep] users with review texts: {len(user_review_texts)}", flush=True)

    # 4) 加载 spaCy（参考 e17）
    print(f"[nlp] loading spaCy...", flush=True)
    nlp = load_spacy_model()
    print(f"[nlp] OK", flush=True)

    # 5) 计算每个 user 的 30-dim style vector
    #    复用 e17 step1 的做法：对每个 review text 跑 nlp.pipe，
    #    然后按 doc.sents 切句，每句按 [MIN_WORDS, MAX_WORDS] 词数过滤后提取 20 个句法特征 + opener。
    vectors = {}
    feats_per_user = {}
    openers_per_user = {}
    sent_count_dist = []
    user_long_sent_counts = {}
    for i, (uid, review_texts) in enumerate(user_review_texts.items()):
        # review_texts 是 user 全部 stage1 review texts
        sent_feats = []
        openers = []
        for doc in nlp.pipe(review_texts, batch_size=32):
            for sent in doc.sents:
                toks = [tok for tok in sent if not tok.is_punct and not tok.is_space]
                wc = len(toks)
                if wc < MIN_WORDS or wc > MAX_WORDS:
                    continue
                if len(toks) < 3:
                    continue
                try:
                    ex = extract_clause_features_from_doc(sent, sent.text)
                except Exception:
                    continue
                sent_feats.append(np.asarray([float(ex.get(k, 0.0)) for k in FEATS], dtype=np.float32))
                op_idx = opener_class_of(toks[0].text)
                openers.append(OPENER_CLASSES.index(op_idx) if op_idx is not None else OPENER_CLASSES.index("OTHER"))
                if len(sent_feats) >= MAX_SENTENCES_PER_USER:
                    break
            if len(sent_feats) >= MAX_SENTENCES_PER_USER:
                break
        if len(sent_feats) < MIN_LONG_SENTENCES:
            continue
        user_long_sent_counts[uid] = len(sent_feats)
        feats_per_user[uid] = sent_feats
        openers_per_user[uid] = openers
        fmean = np.mean(sent_feats, axis=0)
        n = float(np.linalg.norm(fmean))
        if n > 1e-12:
            fmean = fmean / n
        ohist = np.zeros(len(OPENER_CLASSES), dtype=np.float32)
        for c in openers:
            ohist[c] += 1
        ohist = ohist / max(len(openers), 1)
        vectors[uid] = np.concatenate([fmean, ohist]).astype(np.float32).tolist()
        sent_count_dist.append(len(sent_feats))
        if (i + 1) % 500 == 0:
            print(f"  [vec] {i+1}/{len(user_review_texts)} users done "
                  f"(qual={len(vectors)})", flush=True)

    print(f"[vec] total users with valid vectors: {len(vectors)}", flush=True)

    # 6) 验证：inter-user variance / pairwise distance（应在 sweep 预测范围内）
    mat = np.array([vectors[uid] for uid in vectors], dtype=np.float32)
    inter_var = float(mat.var(axis=0).mean())
    from scipy.spatial.distance import pdist
    sub = mat[np.random.default_rng(SEED).choice(len(mat), size=min(200, len(mat)), replace=False)]
    dists = pdist(sub, metric="euclidean")
    mean_dist = float(dists.mean())
    n_zero_dims = int((mat.std(axis=0) == 0).sum())

    # 7) train / dev / test 切分（与 e17 保持对齐）
    train_set = set()
    if BABY_DS.exists():
        bd = json.load(open(BABY_DS))
        train_set = {r["user_id"] for r in bd}

    rng = np.random.default_rng(SEED)
    all_users = sorted(vectors.keys())
    pool_users = [u for u in all_users if u not in train_set]
    rng.shuffle(pool_users)
    n_test = max(100, len(pool_users) // 5)  # Issue 18 目标 ≥100
    test_users = sorted(pool_users[:n_test])
    leftover = pool_users[n_test:]
    n_dev = min(50, len(leftover) // 5)
    dev_users = sorted(leftover[:n_dev])
    train_users = sorted(set(all_users) - set(test_users) - set(dev_users))

    out = {
        "version": "e18-clean-style-pool-v1",
        "category": CATEGORY,
        "stage0_thresholds": {
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "min_long_sentences": MIN_LONG_SENTENCES,
            "min_groups": MIN_GROUPS,
            "max_sentences_per_user": MAX_SENTENCES_PER_USER,
        },
        "rationale": "selected from 80-config sweep; see e18_stage0_sweep.json",
        "split_seed": SEED,
        "n_pool": len(vectors),
        "n_train": len(train_users),
        "n_dev": len(dev_users),
        "n_test": len(test_users),
        "quality": {
            "inter_var": inter_var,
            "mean_pair_dist": mean_dist,
            "n_zero_dims": n_zero_dims,
            "sweep_baseline_inter_var": 0.00278,
            "sweep_baseline_mean_dist": 0.3971,
        },
        "vectors": vectors,
        "splits": {"train": train_users, "dev": dev_users, "test": test_users},
    }
    with open(E18 / "e18_style_vectors.json", "w") as f:
        json.dump(out, f)
    print(f"\n[saved] {E18 / 'e18_style_vectors.json'} "
          f"({(E18 / 'e18_style_vectors.json').stat().st_size / 1024**2:.1f} MB)", flush=True)

    pool_stats = {
        "version": "e18-pool-stats-v1",
        "thresholds": out["stage0_thresholds"],
        "n_pool": out["n_pool"],
        "n_train": out["n_train"],
        "n_dev": out["n_dev"],
        "n_test": out["n_test"],
        "inter_var": inter_var,
        "mean_pair_dist": mean_dist,
        "n_zero_dims": n_zero_dims,
        "sweep_predicted_inter_var": 0.00371,
        "sweep_predicted_mean_dist": 0.4441,
        "sweep_baseline_inter_var": 0.00278,
        "sweep_baseline_mean_dist": 0.3971,
        "improvement_inter_var_pct": round(100 * (inter_var - 0.00278) / 0.00278, 2),
        "improvement_mean_dist_pct": round(100 * (mean_dist - 0.3971) / 0.3971, 2),
    }
    json.dump(pool_stats, open(E18 / "e18_pool_stats.json", "w"), indent=2)
    print(f"[saved] {E18 / 'e18_pool_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
