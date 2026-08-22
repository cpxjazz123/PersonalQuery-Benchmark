#!/usr/bin/env python3
"""Phase 4: 评估 syntax-only Δh vs baseline 在 query-level syntax fidelity.

对比条件:
  - D_off:        strict_v8_n10.json (no injection, baseline)
  - syntax_mean:  query_records_with_query_inject_syntax_L20_a0.3_mean.json
  - syntax_samp:  query_records_with_query_inject_syntax_L20_a0.1_sample.json

每个 query 评估 2 类指标:
  A. SYNTAX fidelity (与 user self sentence 的 syntax 距离):
     - per-query 算 syntax features (spaCy)
     - user mean (来自 sentences_for_rewrite.jsonl 该 user 的所有句)
     - 距离 = L2(z-score(features)) lower = better
  B. SENTIMENT leakage (query 偏离 neutral 的程度):
     - polarity / subjectivity 绝对值
     - 理想 syntax-only inject 应该接近 0 (不引入 sentiment bias)

输出:
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase4_results.json
  per-condition {mean_syntax_dist, mean_sentiment_leak, n_queries, per_query: [...]}
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
OUT_DIR = SCRATCH
OUT_DIR.mkdir(parents=True, exist_ok=True)


# === 输入条件 ===
CONDITIONS = {
    "D_off_baseline": REPO_ROOT / "result/query_records_with_query_inject_strict_v8_n10.json",
    "syntax_L20_mean_a0.3": REPO_ROOT / "result/query_records_with_query_inject_syntax_L20_a0.3_mean.json",
    "syntax_L20_sample_a0.1": REPO_ROOT / "result/query_records_with_query_inject_syntax_L20_a0.1_sample.json",
}


# === SYNTAX FEATURES (复用 phase1_explore 的逻辑) ===
def compute_query_syntax_features(queries: list[str]) -> tuple[np.ndarray, list[str]]:
    """对 query 算 14 维 syntax features。"""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART"}

    rows = []
    for doc in nlp.pipe(queries, batch_size=64):
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
            depth = 0
            cur = tok
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
    return np.array(rows, dtype=np.float64), feat_names


def _simple_polarity(text: str) -> tuple[float, float]:
    """极简 lexicon polarity, 与 phase1 一致。"""
    POS = {"good", "great", "nice", "love", "loved", "loves", "perfect",
           "excellent", "amazing", "wonderful", "best", "happy", "glad",
           "easy", "comfortable", "recommend", "recommended", "awesome",
           "fantastic", "pretty", "fine", "works", "worked", "working",
           "useful", "helpful", "impressed", "smooth", "satisfied",
           "solid", "sturdy", "durable", "soft", "cute", "enjoy",
           "enjoyed", "enjoying", "like", "liked", "liking", "prefer",
           "preferred", "yes", "yeah", "super", "well", "good",
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
    words = re.findall(r"[a-z']+", text.lower())
    n_words = max(len(words), 1)
    n_pos = sum(1 for w in words if w in POS)
    n_neg = sum(1 for w in words if w in NEG)
    polarity = (n_pos - n_neg) / (n_pos + n_neg + 1.0)
    subjectivity = min(1.0, (n_pos + n_neg) / n_words)
    return polarity, subjectivity


def load_user_self_sentences():
    """sentences_for_rewrite.jsonl → user_id → list of sentence_text"""
    user_sents: dict[str, list[str]] = {}
    with open("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/sentences_for_rewrite.jsonl") as f:
        for line in f:
            r = json.loads(line)
            user_sents.setdefault(r["user_id"], []).append(r["sentence_text"])
    return user_sents


def evaluate_condition(name: str, json_path: Path,
                       user_self_sents: dict[str, list[str]],
                       user_syntax_means: dict[str, np.ndarray],
                       syntax_norm_mean: np.ndarray,
                       syntax_norm_std: np.ndarray,
                       ) -> dict:
    print(f"\n=== 评估 {name} ===")
    print(f"  path: {json_path}")
    if not json_path.exists():
        return {"name": name, "error": f"file not found: {json_path}"}

    records = json.load(open(json_path, "r", encoding="utf-8"))
    records = [r for r in records if r.get("y_plus_query_inject")]
    print(f"  records: {len(records)}")

    if not records:
        return {"name": name, "n_queries": 0}

    # 抽取 best query
    queries = [r["y_plus_query_inject"] for r in records]
    user_ids = [r["user_id"] for r in records]

    # 1. syntax features per query
    X_q, feat_names = compute_query_syntax_features(queries)
    # z-score (用 sentence-norm mean/std, 与 user self sentences 同一坐标系)
    X_q_z = (X_q - syntax_norm_mean) / (syntax_norm_std + 1e-8)

    # 2. sentiment per query
    sent_q = np.array([_simple_polarity(q) for q in queries])  # (n, 2)

    # 3. per-query 与 user self mean syntax 的 L2 distance
    syntax_dist = []
    sentiment_leak = []
    valid = 0
    for i, (q, uid) in enumerate(zip(queries, user_ids)):
        u_mean = user_syntax_means.get(uid)
        if u_mean is None:
            continue
        valid += 1
        d = float(np.linalg.norm(X_q_z[i] - u_mean))
        syntax_dist.append(d)
        sentiment_leak.append(float(abs(sent_q[i, 0]) + abs(sent_q[i, 1])))

    syntax_dist = np.array(syntax_dist) if syntax_dist else np.array([0.0])
    sentiment_leak = np.array(sentiment_leak) if sentiment_leak else np.array([0.0])

    # 4. 与 user self sentence mean 的 contrast baseline:
    #    拿该 user 的 30 个 self sentence (从 npz 加载), 算它们之间的距离作为 natural variance floor
    print(f"  syntax_dist (mean ± std): {syntax_dist.mean():.3f} ± {syntax_dist.std():.3f}")
    print(f"  sentiment_leak (mean ± std): {sentiment_leak.mean():.3f} ± {sentiment_leak.std():.3f}")

    return {
        "name": name,
        "n_queries": valid,
        "syntax_dist_mean": float(syntax_dist.mean()),
        "syntax_dist_std": float(syntax_dist.std()),
        "syntax_dist_min": float(syntax_dist.min()),
        "syntax_dist_max": float(syntax_dist.max()),
        "sentiment_leak_mean": float(sentiment_leak.mean()),
        "sentiment_leak_std": float(sentiment_leak.std()),
    }


def main():
    # === 1. 加载 user self sentences + 算 user mean syntax features ===
    print("[main] loading user self sentences...")
    user_sents = load_user_self_sentences()
    print(f"  {len(user_sents)} users")

    all_user_sentences = []
    for uid, sents in user_sents.items():
        all_user_sentences.extend(sents)
    print(f"  {len(all_user_sentences)} total sentences")

    print("[main] computing syntax features for ALL user self sentences...")
    X_all, feat_names = compute_query_syntax_features(all_user_sentences)
    syntax_norm_mean = X_all.mean(axis=0)
    syntax_norm_std = X_all.std(axis=0)
    print(f"  features shape: {X_all.shape}, std range: "
          f"[{syntax_norm_std.min():.3f}, {syntax_norm_std.max():.3f}]")

    # per-user mean (z-scored)
    user_syntax_means: dict[str, np.ndarray] = {}
    idx = 0
    for uid, sents in user_sents.items():
        n = len(sents)
        X_u = X_all[idx:idx + n]
        X_u_z = (X_u - syntax_norm_mean) / (syntax_norm_std + 1e-8)
        user_syntax_means[uid] = X_u_z.mean(axis=0)
        idx += n

    # also: user-to-user natural distance distribution (as floor reference)
    all_means = np.stack([user_syntax_means[u] for u in user_sents.keys()])
    n_users = all_means.shape[0]
    inter_dists = []
    for i in range(min(20, n_users)):
        for j in range(i + 1, min(20, n_users)):
            inter_dists.append(float(np.linalg.norm(all_means[i] - all_means[j])))
    inter_dists = np.array(inter_dists)
    print(f"\n[baseline] user-to-user syntax distance: "
          f"mean={inter_dists.mean():.3f}, std={inter_dists.std():.3f}")
    print(f"  → query-to-own-user distance < {inter_dists.mean():.2f} ≈ 接近 self user")

    # === 2. 评估每个 condition ===
    results = []
    for name, path in CONDITIONS.items():
        r = evaluate_condition(name, path, user_sents, user_syntax_means,
                               syntax_norm_mean, syntax_norm_std)
        results.append(r)

    # === 3. summary ===
    print("\n" + "=" * 80)
    print("【Phase 4 评估结果汇总】")
    print("=" * 80)
    print(f"{'condition':<30} {'n':>4} {'syn_dist':>10} {'sent_leak':>12}")
    for r in results:
        n = r.get("n_queries", 0)
        d = r.get("syntax_dist_mean", 0)
        l = r.get("sentiment_leak_mean", 0)
        print(f"{r['name']:<30} {n:>4} {d:>10.3f} {l:>12.3f}")
    print(f"\nuser-to-user natural distance floor: mean={inter_dists.mean():.3f}")
    print("  → 解读: query-to-own-user 距离越接近 user-to-user 距离, style fidelity 越好")

    # 找出最佳 condition
    if len(results) >= 2:
        baseline = next((r for r in results if r["name"] == "D_off_baseline"), None)
        for r in results:
            if r["name"] == "D_off_baseline":
                continue
            if baseline and r.get("syntax_dist_mean"):
                delta = r["syntax_dist_mean"] - baseline["syntax_dist_mean"]
                verdict = "✓ BETTER" if delta < -0.1 else ("≈ SAME" if abs(delta) < 0.1 else "✗ WORSE")
                print(f"\n  {r['name']} vs baseline: Δsyn_dist = {delta:+.3f}  {verdict}")
                print(f"  {r['name']} sentiment_leak = {r['sentiment_leak_mean']:.3f} "
                      f"vs baseline {baseline['sentiment_leak_mean']:.3f}")

    # save
    with open(OUT_DIR / "phase4_results.json", "w") as f:
        json.dump({
            "user_to_user_natural_dist_mean": float(inter_dists.mean()),
            "user_to_user_natural_dist_std": float(inter_dists.std()),
            "conditions": results,
            "feat_names": feat_names,
        }, f, indent=2)
    print(f"\n[save] → {OUT_DIR / 'phase4_results.json'}")


if __name__ == "__main__":
    main()
