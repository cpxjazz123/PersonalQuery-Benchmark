#!/usr/bin/env python3
"""Phase 1: Syntax-only subspace exploration.

目标 (用户 2026-08-22 反馈):
  review-space residual 包含 syntax + sentiment + category + review-discourse
  多种信号. 直接注入 query LLM 会把后几类也带进去 → 触发 prompt
  interpretation / 解释句 / product confound.

本脚本:
  1. 加载 299 reviews × 3584-dim residuals (layers 16/20/24/26)
  2. 计算每条 review 的 style feature vector:
     - SYNTAX: sentence_length, clause_count, function_word_ratio,
       coordination_count, relative_clause_count, subordinator_count,
       dependency_depth, noun_ratio, verb_ratio, adj_ratio, adv_ratio,
       pronoun_ratio, avg_word_length
     - NOISE:   sentiment_polarity, sentiment_subjectivity, product_category,
       rating
  3. PCA(20) on residuals (per layer)
  4. 对每个 PCA dim j, 计算与 SYNTAX/NOISE feature 的 |Pearson r|
  5. 输出 "top correlated feature" 表 + syntax_score / noise_score 排名
  6. 标识可保留的 syntax-only dims (syntax_score > noise_score AND > threshold)

输出 (硬编码):
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/
    - pca_layer_16.npz (components, mean, explained_variance_ratio)
    - feature_matrix.npz (per-sentence syntax + noise features)
    - correlation_layer_16.json (top-correlated features per dim)
    - syntax_dims_summary.json (which dims to keep)
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
OUT_DIR = SCRATCH / "syntax_subspace"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 硬编码输入路径 ===
RESIDUAL_NPZ = SCRATCH / "gaussian_vades/residual_hidden.npz"
SENTENCES_JSONL = SCRATCH / "gaussian_vades/sentences_for_rewrite.jsonl"
STAGE1_REVIEWS = REPO_ROOT / "result/stage1_filtered_users_reviews_3000u.json"
META_GZ = Path("/fs04/ar57/wenyu/PersoanlQuery/data/meta_Baby_Products_2023.jsonl.gz")
PRODUCT_ATTRS_JSON = REPO_ROOT / "result/product_attributes.json"

LAYERS = [16, 20, 24, 26]
PCA_DIM = 20
MIN_CORR = 0.10  # |r| 阈值, 低于此认为该 dim 与该 feature 无关联


# === 1. 加载数据 ===
def load_residuals():
    npz = np.load(RESIDUAL_NPZ, allow_pickle=True)
    sentences = list(npz["sentences"])
    residuals = {l: npz[f"residual_layer_{l}"] for l in LAYERS}
    print(f"[load] {len(sentences)} sentences, residuals shape per layer: "
          f"{residuals[LAYERS[0]].shape}")
    return sentences, residuals


def load_user_id_mapping():
    """sentences_for_rewrite.jsonl: {sentence_text, user_id, ...}"""
    sent_to_user = {}
    with open(SENTENCES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sent_to_user[row["sentence_text"]] = row["user_id"]
    print(f"[load] {len(sent_to_user)} sentence→user mappings")
    return sent_to_user


def load_rating_mapping():
    """从 stage1 reviews 拿 (user_id, asin, rating). 仅作为参考表,
    sentence-level rating 实际通过 stage1 review_index 对应回 stage1 reviews."""
    reviews = json.load(open(STAGE1_REVIEWS, "r", encoding="utf-8"))
    # 按 (user_id, asin, review_index) → rating 索引
    rate_idx: dict[tuple[str, str, int], float] = {}
    for r in reviews:
        uid = r.get("user_id")
        asin = r.get("asin")
        # stage1 里 review 可能有 review_index (0-based), 没有就用 overall rating
        if not uid or not asin:
            continue
        rating = r.get("overall") or r.get("rating") or 0.0
        # sentences_for_rewrite.jsonl 的 review_index 0-based
        rate_idx[(uid, asin)] = float(rating)
    print(f"[load] {len(rate_idx)} (user_id, asin)→rating mappings")
    return rate_idx


def load_category_mapping():
    """asin → main_category (top-level from product_attributes.json)."""
    pa = json.load(open(PRODUCT_ATTRS_JSON))
    cat_map = {asin: (attrs.get("Main Category") or attrs.get("top-level") or "Unknown")
               for asin, attrs in pa.items()}
    print(f"[load] {len(cat_map)} asin→category mappings "
          f"(unique cats: {len(set(cat_map.values()))})")
    return cat_map


def load_review_meta():
    """从 sentences_for_rewrite.jsonl 拿 (sentence_text, user_id, asin, rating, ...)."""
    rows = []
    with open(SENTENCES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# === 2. Syntax + noise features per sentence ===
def compute_syntax_features(sentences: list[str]) -> tuple[np.ndarray, list[str]]:
    """spaCy batch parse, 计算每个句子的 syntax feature vector。

    SYNTAX features (目标保留):
      - sentence_length (token count)
      - clause_count (verb count proxy)
      - function_word_ratio (DT, IN, CC, PRP, MD, TO 比例)
      - coordination_count (CC + 并列连词)
      - subordinator_count (IN 从属连词)
      - relative_clause_count (WDT/WP/WP$ + VBN)
      - dependency_depth (max token depth in dep tree)
      - avg_word_length
      - noun_ratio, verb_ratio, adj_ratio, adv_ratio
      - pronoun_ratio
      - punctuation_count (commas, periods, semicolons)

    Returns:
      X_syntax: (n_sentences, n_syntax_features)
      feat_names: list of feature names
    """
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}
    PRONOUNS = {"PRON"}

    rows = []
    for doc in nlp.pipe(sentences, batch_size=64):
        n_tok = max(len(doc), 1)
        pos_counts = {}
        for tok in doc:
            pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1
        function_words = sum(pos_counts.get(p, 0) for p in FUNCTION_POS)
        func_ratio = function_words / n_tok
        # coordination: CCONJ count
        coord = pos_counts.get("CCONJ", 0)
        # subordinator: SCONJ + ADP (IN 传统上是介词/从属连词)
        subord = pos_counts.get("SCONJ", 0) + pos_counts.get("ADP", 0)
        # relative clause: WDT/WP/WP$ + following verb
        rel_count = sum(1 for tok in doc if tok.pos_ == "SCONJ" and
                        tok.dep_ in {"relcl", "advcl"})
        # dependency depth: max token head-distance
        max_depth = 0
        for tok in doc:
            depth = 0
            cur = tok
            while cur.head != cur and depth < 20:
                cur = cur.head
                depth += 1
            max_depth = max(max_depth, depth)
        # average word length
        words = [t.text for t in doc if t.is_alpha]
        avg_wlen = (sum(len(w) for w in words) / max(len(words), 1)) if words else 0
        # pos ratios
        noun_r = pos_counts.get("NOUN", 0) / n_tok
        verb_r = pos_counts.get("VERB", 0) / n_tok
        adj_r = pos_counts.get("ADJ", 0) / n_tok
        adv_r = pos_counts.get("ADV", 0) / n_tok
        pron_r = pos_counts.get("PRON", 0) / n_tok
        # punctuation
        punct = sum(1 for t in doc if t.is_punct)
        # clause_count proxy: verb + (SCONJ for sub-clauses)
        clause_count = pos_counts.get("VERB", 0) + subord

        rows.append([
            float(n_tok),
            float(clause_count),
            func_ratio,
            float(coord),
            float(subord),
            float(rel_count),
            float(max_depth),
            avg_wlen,
            noun_r,
            verb_r,
            adj_r,
            adv_r,
            pron_r,
            float(punct),
        ])

    feat_names = [
        "sentence_length", "clause_count", "function_word_ratio",
        "coordination_count", "subordinator_count", "relative_clause_count",
        "dependency_depth", "avg_word_length", "noun_ratio", "verb_ratio",
        "adj_ratio", "adv_ratio", "pronoun_ratio", "punctuation_count",
    ]
    X = np.array(rows, dtype=np.float64)
    # standardize (zero mean, unit var) 以便 correlation 比较
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    print(f"[spacy] computed {X.shape[1]} syntax features for {X.shape[0]} sentences")
    return X_std, feat_names


def _simple_polarity(text: str) -> tuple[float, float]:
    """极简 lexicon-based polarity + subjectivity。

    目的: 给每条 review 一个 coarse sentiment 信号供 PCA correlation 分析,
    不需要精确的 NLP 分数 (textblob 不可用, 也不值得装)。

    polarity ∈ [-1, 1] = (n_pos - n_neg) / (n_pos + n_neg + 1)
    subjectivity ∈ [0, 1] = (n_pos + n_neg + n_modifiers) / n_words
    """
    POS = {"good", "great", "nice", "love", "loved", "loves", "perfect",
           "excellent", "amazing", "wonderful", "best", "happy", "glad",
           "easy", "comfortable", "recommend", "recommended", "awesome",
           "fantastic", "pretty", "fine", "works", "worked", "working",
           "useful", "helpful", "impressed", "smooth", "satisfied",
           "solid", "sturdy", "durable", "soft", "cute", "enjoy",
           "enjoyed", "enjoying", "like", "liked", "liking", "prefer",
           "preferred", "yes", "yeah", "super", "fine", "well", "good",
           "better", "worth", "convenient", "quality", "beautiful"}
    NEG = {"bad", "poor", "terrible", "awful", "hate", "hated", "worst",
           "horrible", "disappointing", "disappointed", "disappointment",
           "broken", "broke", "breaks", "breaking", "cheap", "flimsy",
           "difficult", "hard", "impossible", "frustrated", "frustrating",
           "frustration", "annoying", "annoyed", "useless", "waste",
           "wasted", "wasteful", "fail", "failed", "failure", "wrong",
           "defective", "defect", "issue", "issues", "problem", "problems",
           "noisy", "loud", "smelly", "smell", "rough", "uncomfortable",
           "painful", "sore", "weak", "leaking", "leaked", "leak", "leaks",
           "return", "returned", "refund", "refunded", "disgusting",
           "horrendous", "dreadful", "ugh", "no", "not", "didn't", "don't"}
    NEGATOR = {"not", "no", "never", "n't", "without"}
    MODIFIER = {"very", "really", "extremely", "quite", "too", "so",
                "absolutely", "totally", "completely", "highly"}

    words = re.findall(r"[a-z']+", text.lower())
    n_words = max(len(words), 1)
    n_pos = 0
    n_neg = 0
    n_mod = 0
    negated = False
    for w in words:
        if w in NEGATOR:
            negated = True
            continue
        if w in MODIFIER:
            n_mod += 1
            continue
        if w in POS:
            n_pos += 1 if not negated else 0
            n_neg += 1 if negated else 0
            negated = False
        elif w in NEG:
            n_neg += 1 if not negated else 0
            n_pos += 1 if negated else 0
            negated = False
        else:
            # window 内如果走了很远, 重置
            if w in {".", ",", ";", "!", "?"}:
                negated = False
    polarity = (n_pos - n_neg) / (n_pos + n_neg + 1.0)
    subjectivity = min(1.0, (n_pos + n_neg + n_mod) / n_words)
    return polarity, subjectivity


def compute_noise_features(rows: list[dict], cat_map: dict,
                           rate_idx: dict[tuple[str, str], float]
                           ) -> tuple[np.ndarray, list[str]]:
    """noise features: sentiment + product_category + rating.

    row schema: {user_id, asin, review_index, sentence_text, word_count}
    缺失 rating / category 时填默认值 (rating=0, category="Unknown")。

    Returns:
      X_noise: (n_sentences, n_noise_features)
      feat_names: list of feature names
    """
    noise_rows = []
    # category -> one-hot index
    cats = sorted(set(cat_map.values()))
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    n_cat = len(cat_to_idx)

    n_missing_cat = 0
    n_missing_rate = 0
    for row in rows:
        text = row["sentence_text"]
        # sentiment (极简 lexicon, 不依赖 textblob)
        polarity, subjectivity = _simple_polarity(text)
        # category (one-hot) - row has asin only
        asin = row.get("asin")
        cat = cat_map.get(asin, "Unknown")
        cat_idx = cat_to_idx.get(cat, -1)
        if cat_idx < 0:
            n_missing_cat += 1
        cat_onehot = np.zeros(n_cat)
        if cat_idx >= 0:
            cat_onehot[cat_idx] = 1.0
        # rating from stage1 join
        uid = row.get("user_id")
        rating = rate_idx.get((uid, asin), 0.0)
        if rating == 0.0:
            n_missing_rate += 1
        # word_count (raw)
        word_count = float(len(text.split()))
        noise_rows.append([polarity, subjectivity, rating, word_count] + cat_onehot.tolist())

    if n_missing_cat:
        print(f"[noise] 警告: {n_missing_cat} 条 sentence asin 未在 product_attrs 中")
    if n_missing_rate:
        print(f"[noise] 警告: {n_missing_rate} 条 sentence rating 缺失 (填 0)")

    feat_names = ["sentiment_polarity", "sentiment_subjectivity",
                  "rating", "word_count"] + [f"cat_{c[:20]}" for c in cats]
    X = np.array(noise_rows, dtype=np.float64)
    # standardize numeric features, leave one-hot as is (already 0/1)
    numeric_idx = [0, 1, 2, 3]
    for idx in numeric_idx:
        col = X[:, idx]
        X[:, idx] = (col - col.mean()) / (col.std() + 1e-8)
    print(f"[noise] computed {X.shape[1]} noise features ({n_cat} cats) for {X.shape[0]} sentences")
    return X, feat_names


# === 3. PCA + correlation analysis ===
def pca(X: np.ndarray, n_dim: int = PCA_DIM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (n_samples, n_features). 返回 (components, mean, explained_var_ratio)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_dim]  # (n_dim, n_features)
    var_total = (S ** 2).sum()
    explained = (S ** 2)[:n_dim] / var_total
    print(f"[pca] {n_dim} dims explain {explained.sum()*100:.2f}% variance "
          f"(per-dim: {explained[:5].round(3).tolist()}, ...)")
    return components, mean, explained


def correlation_analysis(Z: np.ndarray, feats: np.ndarray, feat_names: list[str]
                         ) -> list[list[tuple[str, float]]]:
    """对每个 PCA dim, 计算与所有 features 的 |Pearson r|。

    Returns:
      top_corrs_per_dim: list of n_dim, 每 dim 是 [(feat_name, r), ...] 按 |r| 降序
    """
    n_dim = Z.shape[1]
    n_feat = feats.shape[1]
    top_per_dim = []
    for j in range(n_dim):
        corrs = []
        for k in range(n_feat):
            r, _ = pearsonr(Z[:, j], feats[:, k])
            corrs.append((feat_names[k], float(r)))
        # 按 |r| 降序
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_per_dim.append(corrs)
    return top_per_dim


def main():
    # === 1. 加载数据 ===
    sentences, residuals = load_residuals()
    sent_to_user = load_user_id_mapping()
    rows = load_review_meta()
    cat_map = load_category_mapping()
    rate_idx = load_rating_mapping()

    # 验证 row alignment: sentences[i] == rows[i]["sentence_text"]
    n_aligned = sum(1 for s, r in zip(sentences, rows) if s == r["sentence_text"])
    print(f"[align] {n_aligned}/{len(sentences)} sentences aligned with rows")
    if n_aligned != len(sentences):
        print("  ⚠ sentence-text mismatch; using index alignment")

    # === 2. Syntax features (spaCy batch) ===
    X_syntax, syntax_names = compute_syntax_features(sentences)
    np.savez_compressed(
        OUT_DIR / "syntax_features.npz",
        X=X_syntax, names=np.array(syntax_names),
    )
    print(f"[save] syntax features → {OUT_DIR / 'syntax_features.npz'}")

    # === 3. Noise features (sentiment + category + rating) ===
    X_noise, noise_names = compute_noise_features(rows, cat_map, rate_idx)
    np.savez_compressed(
        OUT_DIR / "noise_features.npz",
        X=X_noise, names=np.array(noise_names),
    )
    print(f"[save] noise features → {OUT_DIR / 'noise_features.npz'}")

    # === 4. PCA per layer + correlation analysis ===
    print()
    print("=" * 80)
    print("【PCA + Correlation Analysis per layer】")
    print("=" * 80)

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        R = residuals[layer]
        comp, mean, explained = pca(R, n_dim=PCA_DIM)
        # project: Z = (R - mean) @ comp.T  (n_samples, n_dim)
        Z = (R - mean) @ comp.T
        np.savez_compressed(
            OUT_DIR / f"pca_layer_{layer}.npz",
            components=comp, mean=mean, explained=explained, Z=Z,
        )

        # syntax correlation
        syntax_corr = correlation_analysis(Z, X_syntax, syntax_names)
        # noise correlation
        noise_corr = correlation_analysis(Z, X_noise, noise_names)

        # per-dim summary
        dim_summary = []
        for j in range(PCA_DIM):
            top_syntax = [(f, r) for f, r in syntax_corr[j] if abs(r) > MIN_CORR][:3]
            top_noise = [(f, r) for f, r in noise_corr[j] if abs(r) > MIN_CORR][:3]
            max_syntax_r = max((abs(r) for _, r in syntax_corr[j]), default=0.0)
            max_noise_r = max((abs(r) for _, r in noise_corr[j]), default=0.0)
            verdict = "KEEP_SYNTAX" if max_syntax_r > max_noise_r and max_syntax_r > MIN_CORR else \
                      "DROP_NOISE" if max_noise_r > MIN_CORR else \
                      "WEAK"
            dim_summary.append({
                "dim": j,
                "explained_var": float(explained[j]),
                "max_syntax_r": max_syntax_r,
                "max_noise_r": max_noise_r,
                "top_syntax": [{"feat": f, "r": r} for f, r in top_syntax],
                "top_noise": [{"feat": f, "r": r} for f, r in top_noise],
                "verdict": verdict,
            })

        # print per-dim table
        print(f"{'dim':<4} {'exp_var':>8} {'max_syn':>8} {'max_noi':>8}  "
              f"{'verdict':<11}  top_syntax | top_noise")
        for d in dim_summary:
            ts = ", ".join(f"{x['feat']}({x['r']:.2f})" for x in d["top_syntax"][:2])
            tn = ", ".join(f"{x['feat']}({x['r']:.2f})" for x in d["top_noise"][:2])
            print(f"{d['dim']:<4} {d['explained_var']:>8.4f} {d['max_syntax_r']:>8.3f} "
                  f"{d['max_noise_r']:>8.3f}  {d['verdict']:<11}  {ts} | {tn}")

        # 保存 JSON
        with open(OUT_DIR / f"correlation_layer_{layer}.json", "w") as f:
            json.dump({
                "layer": layer,
                "explained_variance_total": float(explained.sum()),
                "dims": dim_summary,
            }, f, indent=2)

        # summary: which dims to keep
        keep_dims = [d["dim"] for d in dim_summary if d["verdict"] == "KEEP_SYNTAX"]
        print(f"  → 建议保留 dims (KEEP_SYNTAX): {keep_dims}")

    # 跨 layer 汇总 (找稳定 KEEP 的 dim)
    print()
    print("=" * 80)
    print("【跨 layer 汇总】")
    print("=" * 80)
    layer_keeps = {}
    for layer in LAYERS:
        with open(OUT_DIR / f"correlation_layer_{layer}.json") as f:
            data = json.load(f)
        layer_keeps[layer] = set(d["dim"] for d in data["dims"] if d["verdict"] == "KEEP_SYNTAX")
        print(f"  Layer {layer}: KEEP_SYNTAX dims = {sorted(layer_keeps[layer])}")

    all_keep = set.intersection(*layer_keeps.values()) if layer_keeps else set()
    print(f"\n  跨 layer 稳定 KEEP_SYNTAX: {sorted(all_keep)}")
    summary = {
        "layer_keeps": {str(k): sorted(v) for k, v in layer_keeps.items()},
        "stable_keep": sorted(all_keep),
        "n_stable": len(all_keep),
        "threshold": MIN_CORR,
    }
    with open(OUT_DIR / "syntax_dims_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] → {OUT_DIR / 'syntax_dims_summary.json'}")
    print("\n【下一步建议】")
    if len(all_keep) >= 3:
        print(f"  1. 用 stable_keep dims ({sorted(all_keep)}) 重新 Gaussian fit per user")
        print(f"  2. 投影 Δh = z_syntax @ U_keep.T, 注入 query LLM")
        print(f"  3. 与 v8 (direct residual) 对比")
    else:
        print(f"  ⚠ 稳定 syntax-only dim 太少 ({len(all_keep)}), 考虑:")
        print(f"     - 降低 MIN_CORR (例如 0.08)")
        print(f"     - 引入更多 syntax features (POS ratios, n-gram)")
        print(f"     - 用 partial correlation 排除 confounders")


if __name__ == "__main__":
    main()
