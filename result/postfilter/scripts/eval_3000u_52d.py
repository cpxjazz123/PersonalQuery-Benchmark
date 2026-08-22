#!/usr/bin/env python3
"""在 3000 用户上提取 52d 特征, 评估 same-user vs cross-user AUC.

输入: stage1_filtered_users_reviews_3000u.json (2918 users)
输出: expanded_features_3000u_52d.jsonl + eval_3000u_52d.json

时间估计: 3000 × 15 = 45000 句, spaCy 每秒 ~200 句, ~225 秒
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import spacy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
STAGE1_FILE = REPO_ROOT / "result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews_3000u.json"

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
EXPANDED_FILE = OUT_DIR / "expanded_features_3000u_52d.jsonl"
EVAL_OUT = OUT_DIR / "eval_3000u_52d.json"
LOG_FILE = OUT_DIR / "extract_3000u_52d.log"

POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
            "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
FUNCTION_WORDS = [
    "the", "a", "an",
    "of", "in", "to", "for", "with", "on", "at", "by",
    "and", "but", "or",
    "is", "are", "was",
]
SENTENCES_PER_USER = 15
TRAIN_PER_USER = 10
TOTAL_DIM = 17 + 17 + 5 + 3 + 5  # 47d (no reused 20-d features since they need full spaCy parse)


def extract_features(doc, sentence_text: str) -> dict:
    tokens = [t for t in doc if not t.is_space]
    n = len(tokens)
    if n == 0:
        return {f"f{i}": 0.0 for i in range(TOTAL_DIM)}
    pos_counts = Counter(t.pos_ for t in tokens)
    pos_freq = {pos: pos_counts.get(pos, 0) / n for pos in POS_TAGS}
    word_lower = Counter(t.text.lower() for t in tokens if t.is_alpha)
    n_words = sum(word_lower.values())
    func_freq = {fw: word_lower.get(fw, 0) / max(1, n_words) for fw in FUNCTION_WORDS}
    n_unique = len(word_lower)
    word_lengths = [len(t.text) for t in tokens if t.is_alpha]
    mean_word_len = float(np.mean(word_lengths)) if word_lengths else 0.0
    max_word_len = float(max(word_lengths)) if word_lengths else 0.0
    ttr = n_unique / max(1, n_words)
    hapax_count = sum(1 for w, c in word_lower.items() if c == 1)
    hapax_ratio = hapax_count / max(1, len(word_lower))
    content_count = pos_counts.get("NOUN", 0) + pos_counts.get("VERB", 0) + \
                    pos_counts.get("ADJ", 0) + pos_counts.get("ADV", 0)
    content_density = content_count / n
    pn = Counter(t.text for t in tokens if t.pos_ == "PUNCT")
    feats = {}
    for pos in POS_TAGS:
        feats[f"pos_{pos}"] = pos_freq[pos]
    for fw in FUNCTION_WORDS:
        feats[f"func_{fw}"] = func_freq[fw]
    feats["len_word_count"] = float(n)
    feats["len_mean_word_len"] = mean_word_len
    feats["len_max_word_len"] = max_word_len
    feats["len_char_count"] = float(len(sentence_text))
    feats["len_n_unique"] = float(n_unique)
    feats["rich_ttr"] = ttr
    feats["rich_hapax_ratio"] = hapax_ratio
    feats["rich_content_density"] = content_density
    feats["punct_excl"] = float(pn.get("!", 0))
    feats["punct_quest"] = float(pn.get("?", 0))
    feats["punct_semi"] = float(pn.get(";", 0))
    feats["punct_colon"] = float(pn.get(":", 0))
    feats["punct_dash"] = float(pn.get("-", 0) + pn.get("—", 0))
    assert len(feats) == TOTAL_DIM
    return feats


def main():
    log = lambda m: print(f"[3000u] {m}", flush=True)
    log("=" * 70)
    log(f"3000 用户 52d 特征 + 评估")
    log("=" * 70)

    log(f"加载 {STAGE1_FILE} ...")
    with STAGE1_FILE.open() as f:
        users = json.load(f)
    log(f"  {len(users)} users")
    n_users = len(users)

    # 抽取每用户 15 句 (10 train + 5 holdout)
    log("抽取每用户 15 句 ...")
    import re
    all_rows = []
    for u_idx, user in enumerate(users):
        uid = user["user_id"]
        collected: list[dict] = []
        seen: set[str] = set()
        for r_idx, review in enumerate(user["reviews"]):
            for text in review.get("target_reviews", []):
                if not text:
                    continue
                parts = re.split(r"(?<=[.!?])\s+", text.strip())
                for s_idx, sent in enumerate(parts):
                    sent = sent.strip()
                    if not sent or sent in seen:
                        continue
                    wc = len(sent.split())
                    if wc < 3 or wc > 100:
                        continue
                    seen.add(sent)
                    collected.append({
                        "user_id": uid,
                        "review_index": r_idx,
                        "sentence_index": s_idx,
                        "sentence_text": sent,
                        "is_holdout": len(collected) >= TRAIN_PER_USER,
                    })
                    if len(collected) == SENTENCES_PER_USER:
                        break
                if len(collected) == SENTENCES_PER_USER:
                    break
            if len(collected) == SENTENCES_PER_USER:
                break
        if len(collected) < SENTENCES_PER_USER:
            continue
        all_rows.extend(collected)
    train = [r for r in all_rows if not r["is_holdout"]]
    ho = [r for r in all_rows if r["is_holdout"]]
    log(f"  total: {len(all_rows)}, train: {len(train)}, holdout: {len(ho)}")

    # spaCy 提取特征
    log("加载 spaCy ...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    log(f"批量解析 {len(all_rows)} 句 ...")
    EXPANDED_FILE.unlink(missing_ok=True)
    with EXPANDED_FILE.open("w", encoding="utf-8") as f:
        for n_done, (doc, row) in enumerate(zip(nlp.pipe([r["sentence_text"] for r in all_rows], batch_size=128), all_rows)):
            feats = extract_features(doc, row["sentence_text"])
            out = {
                "user_id": row["user_id"],
                "review_index": row["review_index"],
                "sentence_index": row["sentence_index"],
                "sentence_text": row["sentence_text"],
                "is_holdout": row["is_holdout"],
                "expanded_features": feats,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if (n_done + 1) % 5000 == 0:
                log(f"  {n_done+1}/{len(all_rows)} 句")

    log(f"  完成, 已写入 {EXPANDED_FILE}")

    # 评估
    log("=" * 70)
    log("评估 3000 用户 52d 特征")
    log("=" * 70)
    rows = []
    with EXPANDED_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    train = [r for r in rows if not r["is_holdout"]]
    ho = [r for r in rows if r["is_holdout"]]
    feature_keys = sorted(rows[0]["expanded_features"].keys())
    log(f"  feat_dim: {len(feature_keys)}")

    train_X = np.array([[r["expanded_features"][k] for k in feature_keys] for r in train])
    ho_X = np.array([[r["expanded_features"][k] for k in feature_keys] for r in ho])
    scaler = StandardScaler().fit(train_X)
    train_Xs = scaler.transform(train_X)
    ho_Xs = scaler.transform(ho_X)

    # uid → idx
    uid_to_idx = {}
    for i, r in enumerate(train):
        if r["user_id"] not in uid_to_idx:
            uid_to_idx[r["user_id"]] = i
    # 实际去重
    seen = set()
    uid_to_idx = {}
    for r in rows:
        uid = r["user_id"]
        if uid not in seen:
            uid_to_idx[uid] = len(uid_to_idx)
            seen.add(uid)
    n_unique = len(uid_to_idx)
    log(f"  unique users: {n_unique}")

    # per-user mean (in scaled space)
    mu_u = np.zeros((n_unique, train_Xs.shape[1]))
    counts = np.zeros(n_unique)
    for r, x in zip(train, train_Xs):
        uidx = uid_to_idx[r["user_id"]]
        mu_u[uidx] += x
        counts[uidx] += 1
    mu_u /= counts[:, None]

    labels = np.array([uid_to_idx[r["user_id"]] for r in ho])
    results = {}
    for mode in ["shared_sigma_l2", "l2", "cosine"]:
        log(f"\n--- Mode: {mode} ---")
        if mode == "shared_sigma_l2":
            sigma = train_Xs.var(axis=0).clip(min=1e-6)
            D = ((ho_Xs[:, None, :] - mu_u[None, :, :]) ** 2 / sigma[None, None, :]).sum(axis=2)
        elif mode == "l2":
            D = ((ho_Xs[:, None, :] - mu_u[None, :, :]) ** 2).sum(axis=2)
        else:
            ho_n = ho_Xs / (np.linalg.norm(ho_Xs, axis=1, keepdims=True) + 1e-9)
            mu_n = mu_u / (np.linalg.norm(mu_u, axis=1, keepdims=True) + 1e-9)
            D = 1 - (ho_n @ mu_n.T)
        same_d = np.array([D[i, labels[i]] for i in range(len(ho))])
        cross_d = np.zeros(len(ho))
        for i in range(len(ho)):
            d_rest = D[i].copy()
            d_rest[labels[i]] = np.inf
            cross_d[i] = d_rest.min()
        margin = cross_d - same_d
        top1_preds = np.argmin(D, axis=1)
        top1_acc = float((top1_preds == labels).mean())
        y_true_bin = np.zeros_like(D, dtype=int)
        for i, l in enumerate(labels):
            y_true_bin[i, l] = 1
        scores = -D
        aucs = []
        for u in range(n_unique):
            if y_true_bin[:, u].sum() == 0:
                continue
            try:
                aucs.append(roc_auc_score(y_true_bin[:, u], scores[:, u]))
            except ValueError:
                pass
        macro_auc = float(np.mean(aucs))
        # permutation test
        rng = np.random.default_rng(42)
        n_ho = len(ho)
        obs = margin.mean()
        perm_means = np.zeros(200)
        for p in range(200):
            pm = rng.permutation(labels)
            perm_same = np.array([D[i, pm[i]] for i in range(n_ho)])
            perm_cross = np.zeros(n_ho)
            for i in range(n_ho):
                d_rest = D[i].copy()
                d_rest[pm[i]] = np.inf
                perm_cross[i] = d_rest.min()
            perm_means[p] = (perm_cross - perm_same).mean()
        p_value = float((perm_means >= obs).mean())
        results[mode] = {
            "top1_acc": top1_acc,
            "macro_auc": macro_auc,
            "margin_mean": float(margin.mean()),
            "margin_pos_rate": float((margin > 0).mean()),
            "perm_p": p_value,
            "same_d_mean": float(same_d.mean()),
            "cross_d_mean": float(cross_d.mean()),
        }
        log(f"  top-1 acc: {top1_acc*100:.2f}%")
        log(f"  macro AUC: {macro_auc:.4f}")
        log(f"  margin > 0: {results[mode]['margin_pos_rate']*100:.2f}%")
        log(f"  same_d: {results[mode]['same_d_mean']:.3f}, cross_d: {results[mode]['cross_d_mean']:.3f}")
        log(f"  perm p: {p_value:.4f}")

    # 对比 300 / 3000 user
    log("\n" + "=" * 70)
    log("对比 300 user vs 3000 user (52d Cosine)")
    log("=" * 70)
    log(f"{'metric':<25} {'300u':>10} {'3000u':>10}")
    log(f"{'top-1 accuracy':<25} {1.40:>9.2f}% {results['cosine']['top1_acc']*100:>9.2f}%")
    log(f"{'macro AUC':<25} {0.6330:>10.4f} {results['cosine']['macro_auc']:>10.4f}")
    log(f"{'margin > 0 (%)':<25} {1.40:>9.2f}% {results['cosine']['margin_pos_rate']*100:>9.2f}%")

    EVAL_OUT.write_text(json.dumps({
        "n_users": n_unique,
        "n_holdout": len(ho),
        "feat_dim": len(feature_keys),
        "results": results,
    }, indent=2, ensure_ascii=False))
    log(f"\n已写入 {EVAL_OUT}")


if __name__ == "__main__":
    main()
