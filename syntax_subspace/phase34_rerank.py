#!/usr/bin/env python3
"""Phase 34 Step 2: Rerank candidates using 3 score families (no Qwen for residual).

输入:
  result/phase34/candidates.json (10 records × K=8 candidates)
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/syntax_gaussian_layer_20.npz
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/syntax_features.npz (norm stats)

每个 candidate 算 3 类 score:
  S_syntax:  log N(f(q) | mu_u, Sigma_u)   ← Layer 20 syntax Gaussian
  S_lexical: function word overlap with user exemplars
  S_coverage: 1 if 所有 attrs 都覆盖 else 0 (binary)

3 conditions (no Qwen residual in light version):
  - baseline:    cand[0] (greedy-like)
  - syntax:      argmax S_syntax (filtered by coverage=1)
  - hybrid:      argmax(0.5 * z(S_syntax) + 0.5 * z(S_lexical)) (filtered by coverage=1)

输出:
  result/phase34/rerank.json
  per record: baseline_q, syntax_q, hybrid_q, all_candidates with scores
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
OUT_DIR = REPO_ROOT / "result/phase34"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === 硬编码 ===
CANDIDATES_JSON = OUT_DIR / "candidates.json"
SYNTAX_GAUSSIAN_NPZ = SCRATCH / "syntax_gaussian_layer_20.npz"
SYNTAX_FEATURES_NPZ = SCRATCH / "syntax_features.npz"
USER_SENTS_JSON = SCRATCH / "user_sents_per_user.json"
OUT_RERANK = OUT_DIR / "rerank_light.json"


# === Syntax features (复用 phase1_explore.py 逻辑) ===
def compute_syntax_features(texts: list[str]) -> np.ndarray:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}
    rows = []
    for doc in nlp.pipe(texts, batch_size=64):
        n_tok = max(len(doc), 1)
        pos_counts = {}
        for tok in doc:
            pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1
        function_words = sum(pos_counts.get(p, 0) for p in FUNCTION_POS)
        func_ratio = function_words / n_tok
        coord = pos_counts.get("CCONJ", 0)
        subord = pos_counts.get("SCONJ", 0) + pos_counts.get("ADP", 0)
        rel_count = sum(1 for tok in doc if tok.pos_ == "SCONJ" and
                        tok.dep_ in {"relcl", "advcl"})
        max_depth = 0
        for tok in doc:
            depth, cur = 0, tok
            while cur.head != cur and depth < 20:
                cur = cur.head
                depth += 1
            max_depth = max(max_depth, depth)
        words = [t.text for t in doc if t.is_alpha]
        avg_wlen = (sum(len(w) for w in words) / max(len(words), 1)) if words else 0
        noun_r = pos_counts.get("NOUN", 0) / n_tok
        verb_r = pos_counts.get("VERB", 0) / n_tok
        adj_r = pos_counts.get("ADJ", 0) / n_tok
        adv_r = pos_counts.get("ADV", 0) / n_tok
        pron_r = pos_counts.get("PRON", 0) / n_tok
        punct = sum(1 for t in doc if t.is_punct)
        clause_count = pos_counts.get("VERB", 0) + subord
        rows.append([float(n_tok), float(clause_count), func_ratio, float(coord),
                     float(subord), float(rel_count), float(max_depth), avg_wlen,
                     noun_r, verb_r, adj_r, adv_r, pron_r, float(punct)])
    return np.array(rows, dtype=np.float64)


# === Lexical: function word overlap ===
FUNCTION_WORDS = {"the", "a", "an", "and", "or", "but", "if", "because", "as",
                  "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did",
                  "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
                  "my", "your", "his", "its", "our", "their",
                  "this", "that", "these", "those",
                  "in", "on", "at", "with", "from", "to", "for", "of", "by", "into",
                  "about", "between", "through", "during", "before", "after",
                  "above", "below", "up", "down", "out", "off", "over", "under"}


def function_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w in FUNCTION_WORDS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


# === Attribute coverage ===
def count_attrs_covered(text: str, attrs: dict) -> int:
    if not attrs:
        return 0
    text_lower = text.lower()
    return sum(1 for v in attrs.values() if v and str(v).strip()
               and str(v).strip().lower() in text_lower)


def main():
    if not CANDIDATES_JSON.exists():
        raise FileNotFoundError(f"先跑 phase34_generate: {CANDIDATES_JSON} 不存在")

    candidates = json.load(open(CANDIDATES_JSON))
    print(f"[load] {len(candidates)} records from {CANDIDATES_JSON}")

    # 加载 syntax Gaussian
    npz = np.load(SYNTAX_GAUSSIAN_NPZ, allow_pickle=True)
    user_ids = [str(u) for u in npz["user_ids"]]
    mu = npz["mu"]  # (n_users, n_keep)
    sigma = npz["sigma"]  # (n_users, n_keep, n_keep)
    uid_to_idx = {u: i for i, u in enumerate(user_ids)}
    n_keep = mu.shape[1]
    print(f"[load] syntax Gaussian: {len(user_ids)} users, n_keep={n_keep}")

    # 加载 user exemplars for lexical
    user_sents = json.load(open(USER_SENTS_JSON))
    print(f"[load] {len(user_sents)} user exemplars")

    # 加载 norm stats (from syntax_features.npz → std, mean)
    sf = np.load(SYNTAX_FEATURES_NPZ, allow_pickle=True)
    X_train = sf["X"]  # (299, 14) standardized
    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0) + 1e-8
    print(f"[load] syntax norm: feat_dim=14")

    # 处理每条 record
    rerank_records = []
    n_baseline_cov, n_syntax_cov, n_hybrid_cov = 0, 0, 0

    for rec in candidates:
        uid = rec["user_id"]
        attrs = rec["attrs_used"]
        cands = rec["candidates"]
        if not cands:
            continue
        u_idx = uid_to_idx.get(uid)

        # 1) syntax features per candidate
        X_cands = compute_syntax_features(cands)  # (K, 14)
        X_cands_z = (X_cands - feat_mean) / feat_std  # z-score (与 user self 同坐标系)

        # 2) S_syntax = log N(X_cands_z | mu_u, sigma_u) only on KEEP dims
        if u_idx is not None:
            mu_u = mu[u_idx]
            sigma_u = sigma[u_idx]
            # 限制到 KEEP dims (mu/sigma 已经是 KEEP subspace)
            try:
                rv = multivariate_normal(mean=mu_u, cov=sigma_u)
                scores_syntax = rv.logpdf(X_cands_z)  # (K,)
            except Exception:
                scores_syntax = np.zeros(len(cands))
        else:
            scores_syntax = np.zeros(len(cands))

        # 3) S_lexical = function word overlap with user exemplars
        u_sents = user_sents.get(uid, [])
        u_fw = set()
        for s in u_sents[:5]:
            u_fw |= function_words(s)
        scores_lexical = np.array([jaccard(function_words(c), u_fw) for c in cands])

        # 4) coverage
        coverages = np.array([count_attrs_covered(c, attrs) for c in cands])
        n_attrs = len(attrs)
        coverages_full = (coverages == n_attrs).astype(float)  # 1 if all attrs covered

        # 5) z-score normalize for hybrid
        def zscore(x: np.ndarray) -> np.ndarray:
            s = x.std()
            return (x - x.mean()) / (s + 1e-8)
        z_syn = zscore(scores_syntax)
        z_lex = zscore(scores_lexical)
        scores_hybrid = 0.5 * z_syn + 0.5 * z_lex

        # 6) select under each condition (filter by coverage_full)
        # baseline: cand[0]
        baseline_idx = 0
        baseline_cov = coverages[baseline_idx]

        # syntax: argmax S_syntax among coverage_full=1, else argmax regardless
        valid_mask = coverages_full > 0
        if valid_mask.any():
            masked = scores_syntax.copy()
            masked[~valid_mask] = -1e9
            syntax_idx = int(np.argmax(masked))
        else:
            syntax_idx = int(np.argmax(scores_syntax))

        # hybrid: argmax hybrid among coverage_full=1
        if valid_mask.any():
            masked_h = scores_hybrid.copy()
            masked_h[~valid_mask] = -1e9
            hybrid_idx = int(np.argmax(masked_h))
        else:
            hybrid_idx = int(np.argmax(scores_hybrid))

        n_baseline_cov += int(baseline_cov == n_attrs)
        n_syntax_cov += int(coverages[syntax_idx] == n_attrs)
        n_hybrid_cov += int(coverages[hybrid_idx] == n_attrs)

        rerank_records.append({
            "user_id": uid,
            "asin": rec["asin"],
            "attrs_used": attrs,
            "n_attrs": n_attrs,
            "candidates": cands,
            "scores_syntax": scores_syntax.tolist(),
            "scores_lexical": scores_lexical.tolist(),
            "scores_hybrid": scores_hybrid.tolist(),
            "coverages": coverages.tolist(),
            "selections": {
                "baseline_idx": baseline_idx,
                "syntax_idx": syntax_idx,
                "hybrid_idx": hybrid_idx,
                "baseline_q": cands[baseline_idx],
                "syntax_q": cands[syntax_idx],
                "hybrid_q": cands[hybrid_idx],
                "baseline_cov": int(baseline_cov),
                "syntax_cov": int(coverages[syntax_idx]),
                "hybrid_cov": int(coverages[hybrid_idx]),
            },
        })

    # 输出
    out = {
        "n_records": len(rerank_records),
        "n_baseline_full_cov": n_baseline_cov,
        "n_syntax_full_cov": n_syntax_cov,
        "n_hybrid_full_cov": n_hybrid_cov,
        "records": rerank_records,
    }
    json.dump(out, open(OUT_RERANK, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] → {OUT_RERANK}")
    print(f"\n=== 覆盖率汇总 ===")
    print(f"  baseline: {n_baseline_cov}/{len(rerank_records)} = {n_baseline_cov/max(1,len(rerank_records))*100:.1f}%")
    print(f"  syntax:   {n_syntax_cov}/{len(rerank_records)} = {n_syntax_cov/max(1,len(rerank_records))*100:.1f}%")
    print(f"  hybrid:   {n_hybrid_cov}/{len(rerank_records)} = {n_hybrid_cov/max(1,len(rerank_records))*100:.1f}%")

    # print sample
    if rerank_records:
        s = rerank_records[0]
        print(f"\n--- sample (asin={s['asin']}, user={s['user_id'][:14]}) ---")
        for cond in ["baseline", "syntax", "hybrid"]:
            sel = s["selections"]
            print(f"  [{cond:>8}] (cov={sel[f'{cond}_cov']}/{s['n_attrs']}): {sel[f'{cond}_q'][:150]}")


if __name__ == "__main__":
    main()
