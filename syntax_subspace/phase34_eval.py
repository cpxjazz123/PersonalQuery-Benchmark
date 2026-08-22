#!/usr/bin/env python3
"""Phase 34 Step 3: Evaluate 3-condition rerank on style fidelity + sentiment + coverage.

输入:
  result/phase34/rerank_light.json
  /home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/user_sents_per_user.json

每个 condition 算:
  - n_full_cov: 5/5 attribute coverage records (already in rerank_light)
  - syntax_dist: L2(z(f(top1))) - z(f(user_mean)))  (与 user self sentence 距离)
  - sentiment_leak: |polarity| + |subjectivity|  (越小越 neutral)
  - function_word_overlap: Jaccard 与 user exemplars
  - exemplar_echo: query 中是否复用 user exemplar 原句 (e.g. cand3 in sample)

输出:
  result/phase34/eval.json (per-condition metrics)
  控制台汇总表
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
OUT_DIR = REPO_ROOT / "result/phase34"

RERANK_JSON = OUT_DIR / "rerank_light.json"
USER_SENTS_JSON = SCRATCH / "user_sents_per_user.json"
SYNTAX_FEATURES_NPZ = SCRATCH / "syntax_features.npz"

OUT_EVAL = OUT_DIR / "eval.json"

# === Lexical ===
FUNCTION_WORDS = {"the", "a", "an", "and", "or", "but", "if", "because", "as",
                  "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did",
                  "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
                  "my", "your", "his", "its", "our", "their",
                  "this", "that", "these", "those",
                  "in", "on", "at", "with", "from", "to", "for", "of", "by", "into",
                  "about", "between", "through", "during", "before", "after",
                  "above", "below", "up", "down", "out", "off", "over", "under"}

# === Hallucination words (compact version, same as phase3) ===
HALLU_WORDS = [
    "bottle", "clothing", "clothes", "toy", "candle", "tumbler", "carrier",
    "basket", "diaper", "pacifier", "stroller", "swaddle", "onesie",
    "soap", "lotion", "shampoo", "scented", "fragrance", "decorative",
    "friend", "sister", "brother", "wife", "husband", "mom", "dad",
    "premium", "luxury", "handmade", "artisan", "accessory", "tool", "decor",
]


def function_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w in FUNCTION_WORDS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _simple_polarity(text: str) -> tuple[float, float]:
    POS = {"good", "great", "nice", "love", "loved", "perfect", "excellent",
           "amazing", "wonderful", "best", "happy", "easy", "comfortable",
           "recommend", "awesome", "fantastic", "useful", "helpful",
           "satisfied", "solid", "durable", "soft", "cute", "enjoy", "like",
           "yes", "super", "well", "better", "worth", "convenient", "quality"}
    NEG = {"bad", "poor", "terrible", "awful", "hate", "worst", "horrible",
           "disappointing", "broken", "cheap", "flimsy", "difficult",
           "frustrated", "annoying", "useless", "waste", "fail", "wrong",
           "defective", "issue", "problem", "noisy", "rough", "uncomfortable",
           "painful", "weak", "leaking", "return", "refund", "disgusting",
           "no", "not", "didn't", "don't"}
    words = re.findall(r"[a-z']+", text.lower())
    n = max(len(words), 1)
    n_pos = sum(1 for w in words if w in POS)
    n_neg = sum(1 for w in words if w in NEG)
    polarity = (n_pos - n_neg) / (n_pos + n_neg + 1.0)
    subjectivity = min(1.0, (n_pos + n_neg) / n)
    return polarity, subjectivity


def has_hallucination(text: str, attrs: dict) -> list[str]:
    text_lower = text.lower()
    attr_substrings = set()
    for v in attrs.values():
        s = str(v).strip() if v else ""
        if s:
            attr_substrings.add(s.lower())
            for tok in re.split(r"[\s,/]+", s.lower()):
                if len(tok) >= 4:
                    attr_substrings.add(tok)
    return [w for w in HALLU_WORDS
            if re.search(rf"\b{re.escape(w)}\b", text_lower) and w not in attr_substrings]


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
        function_words_n = sum(pos_counts.get(p, 0) for p in FUNCTION_POS)
        func_ratio = function_words_n / n_tok
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


def has_exemplar_echo(query: str, user_sents: list[str]) -> bool:
    """query 是否包含 user exemplar 原句 (>20 char 复用)。"""
    q_lower = query.lower()
    for s in user_sents[:5]:
        s_short = s[:30].lower()
        if len(s_short) > 20 and s_short in q_lower:
            return True
    return False


def main():
    data = json.load(open(RERANK_JSON))
    records = data["records"]
    print(f"[load] {len(records)} records")

    user_sents = json.load(open(USER_SENTS_JSON))

    # self-contained norm stats: compute on all 299 sentences first, then z-score
    all_sents = []
    for uid, sents in user_sents.items():
        all_sents.extend(sents)
    print(f"[spacy] computing syntax features for {len(all_sents)} user self sentences...")
    X_user_raw = compute_syntax_features(all_sents)
    feat_mean = X_user_raw.mean(axis=0)
    feat_std = X_user_raw.std(axis=0) + 1e-8
    X_user_z = (X_user_raw - feat_mean) / feat_std
    print(f"[norm] feat_mean norm={np.linalg.norm(feat_mean):.3f}, feat_std norm={np.linalg.norm(feat_std):.3f}")
    print(f"[norm] X_user_z mean norm={np.linalg.norm(X_user_z, axis=1).mean():.3f} (should be ~sqrt(14)=3.74)")

    # pre-compute user self sentence mean syntax (per user) on z-scored space
    user_mean_z: dict[str, np.ndarray] = {}
    idx = 0
    for uid, sents in user_sents.items():
        n = len(sents)
        user_mean_z[uid] = X_user_z[idx:idx + n].mean(axis=0)
        idx += n
    print(f"[load] user_mean_z ready for {len(user_mean_z)} users")

    # diagnostic: user-to-user natural distance floor
    rng = np.random.default_rng(0)
    sample_pairs = []
    uids_list = list(user_sents.keys())
    for _ in range(50):
        a, b = rng.choice(uids_list, 2, replace=False)
        sample_pairs.append((a, b))
    floors = [float(np.linalg.norm(user_mean_z[a] - user_mean_z[b])) for a, b in sample_pairs]
    print(f"[floor] user-to-user natural dist mean={np.mean(floors):.3f} (Phase 1 reported 1.849)")

    # eval each condition
    conditions = ["baseline", "syntax", "hybrid"]
    cond_results: dict[str, list] = {c: [] for c in conditions}

    for rec in records:
        uid = rec["user_id"]
        attrs = rec["attrs_used"]
        sel = rec["selections"]
        u_mean = user_mean_z.get(uid)
        u_sents = user_sents.get(uid, [])
        u_fw = set()
        for s in u_sents[:5]:
            u_fw |= function_words(s)

        for cond in conditions:
            q = sel[f"{cond}_q"]
            cov = sel[f"{cond}_cov"]
            # syntax features of top-1 candidate
            f_q = compute_syntax_features([q])[0]
            f_q_z = (f_q - feat_mean) / feat_std
            # distance to user self mean
            syn_dist = float(np.linalg.norm(f_q_z - u_mean)) if u_mean is not None else None
            # sentiment leak
            pol, subj = _simple_polarity(q)
            sent_leak = abs(pol) + subj
            # function word overlap
            fw_overlap = jaccard(function_words(q), u_fw)
            # hallucination
            hallu = has_hallucination(q, attrs)
            # exemplar echo
            echo = has_exemplar_echo(q, u_sents)

            cond_results[cond].append({
                "user_id": uid,
                "asin": rec["asin"],
                "query": q,
                "cov": cov,
                "n_attrs": rec["n_attrs"],
                "syn_dist": syn_dist,
                "sent_leak": float(sent_leak),
                "fw_overlap": float(fw_overlap),
                "hallu_n": len(hallu),
                "hallu_words": hallu,
                "exemplar_echo": echo,
            })

    # aggregate
    summary = {}
    for cond, rows in cond_results.items():
        n = len(rows)
        n_full_cov = sum(1 for r in rows if r["cov"] == r["n_attrs"])
        syn_dists = [r["syn_dist"] for r in rows if r["syn_dist"] is not None]
        sent_leaks = [r["sent_leak"] for r in rows]
        fw_overlaps = [r["fw_overlap"] for r in rows]
        n_hallu = sum(1 for r in rows if r["hallu_n"] > 0)
        n_echo = sum(1 for r in rows if r["exemplar_echo"])
        summary[cond] = {
            "n_records": n,
            "n_full_cov": n_full_cov,
            "full_cov_pct": n_full_cov / max(n, 1) * 100,
            "syn_dist_mean": float(np.mean(syn_dists)) if syn_dists else None,
            "syn_dist_std": float(np.std(syn_dists)) if syn_dists else None,
            "sent_leak_mean": float(np.mean(sent_leaks)),
            "fw_overlap_mean": float(np.mean(fw_overlaps)),
            "n_hallu_records": n_hallu,
            "n_exemplar_echo": n_echo,
        }

    print()
    print("=" * 80)
    print("【Phase 34 Rerank 评估汇总】")
    print("=" * 80)
    print(f"{'condition':<10} {'full_cov':>10} {'syn_dist':>12} {'sent_leak':>12} {'fw_overlap':>12} {'hallu':>6} {'echo':>6}")
    for cond, s in summary.items():
        sd = f"{s['syn_dist_mean']:.3f}" if s['syn_dist_mean'] is not None else "N/A"
        print(f"{cond:<10} {s['n_full_cov']:>4}/{s['n_records']:<3} ({s['full_cov_pct']:>4.1f}%) "
              f"{sd:>12} {s['sent_leak_mean']:>12.3f} {s['fw_overlap_mean']:>12.3f} "
              f"{s['n_hallu_records']:>6} {s['n_exemplar_echo']:>6}")
    print()

    # verdict
    base = summary["baseline"]
    print("【verdict (vs baseline)】")
    for cond in ["syntax", "hybrid"]:
        s = summary[cond]
        cov_delta = s["n_full_cov"] - base["n_full_cov"]
        if s["syn_dist_mean"] is not None and base["syn_dist_mean"] is not None:
            sd_delta = s["syn_dist_mean"] - base["syn_dist_mean"]
            sd_v = "✓ BETTER" if sd_delta < -0.1 else ("≈ SAME" if abs(sd_delta) < 0.1 else "✗ WORSE")
        else:
            sd_delta, sd_v = None, "N/A"
        sl_delta = s["sent_leak_mean"] - base["sent_leak_mean"]
        sl_v = "✓ BETTER" if sl_delta < -0.05 else ("≈ SAME" if abs(sl_delta) < 0.05 else "✗ WORSE")
        fw_delta = s["fw_overlap_mean"] - base["fw_overlap_mean"]
        fw_v = "✓ BETTER" if fw_delta > 0.05 else ("≈ SAME" if abs(fw_delta) < 0.05 else "✗ WORSE")
        print(f"  {cond}: cov {cov_delta:+d}, syn_dist {sd_delta:+.3f} ({sd_v}), "
              f"sent_leak {sl_delta:+.3f} ({sl_v}), fw_overlap {fw_delta:+.3f} ({fw_v})")

    out = {
        "summary": summary,
        "records_by_condition": cond_results,
    }
    json.dump(out, open(OUT_EVAL, "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] → {OUT_EVAL}")


if __name__ == "__main__":
    main()
