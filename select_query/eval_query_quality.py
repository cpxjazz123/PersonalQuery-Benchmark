#!/usr/bin/env python3
"""Query quality analysis for 10K pipeline.

Per record:
- n_attrs_covered / n_attrs (cov rate)
- n_hallucinations
- query length (chars + words)
- K=4 candidate diversity (pairwise Jaccard)
- prefix pattern (most common openers)
- query style coherence (no broken output)

Aggregate stats across 9982 records.
"""
from __future__ import annotations
import json
import re
import statistics
from collections import Counter
from pathlib import Path

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
INPUT = REPO / "result/query_records_with_query_inject_strict_10k.json"


def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    data = json.load(open(INPUT))
    print(f"Total records: {len(data)}")

    # Coverage
    cov_rates = []
    cov_full = 0  # 5/5 or all attrs covered
    hallu_zero = 0
    n_cands_dist = Counter()

    # Length (best query)
    char_lens = []
    word_lens = []
    short_queries = 0  # < 5 words
    long_queries = 0  # > 50 words

    # Diversity (pairwise Jaccard within K=4)
    diversities = []
    diversity_per_slot = [[], [], []]  # 3 pairs per record (0-1, 0-2, 0-3)

    # Prefix patterns (first 5 words)
    prefix_counts = Counter()

    # Quality flags
    broken_outputs = 0  # queries with broken text artifacts
    no_attr_covered = 0
    high_hallu = 0  # > 1 hallucination

    for d in data:
        attrs = d.get("attrs_used", {})
        n_attrs = len(attrs)
        cov = d.get("n_attrs_covered", 0)
        hallu = d.get("n_hallucinations", 0)
        cands = d.get("all_candidates", [])
        scores = d.get("all_candidates_scores", [])
        best = d.get("y_plus_query_inject", "")

        # Coverage
        if n_attrs > 0:
            cov_rates.append(cov / n_attrs)
        if cov == n_attrs and n_attrs > 0:
            cov_full += 1
        if hallu == 0:
            hallu_zero += 1
        if hallu > 1:
            high_hallu += 1
        if cov == 0:
            no_attr_covered += 1

        n_cands_dist[len(cands)] += 1

        # Length of best query
        n_words = len(best.split())
        n_chars = len(best)
        word_lens.append(n_words)
        char_lens.append(n_chars)
        if n_words < 5:
            short_queries += 1
        if n_words > 50:
            long_queries += 1

        # Diversity (pairwise within K=4)
        if len(cands) >= 2:
            pairs = [(cands[0], cands[i]) for i in range(1, len(cands))]
            pair_jac = []
            for i, (a, b) in enumerate(pairs):
                if i < 3:
                    j = jaccard(a, b)
                    diversity_per_slot[i].append(j)
                    pair_jac.append(j)
            diversities.append(statistics.mean(pair_jac) if pair_jac else 0.0)

        # Prefix pattern (first 5 words)
        prefix = " ".join(best.split()[:5]).lower().strip()
        if prefix:
            prefix_counts[prefix] += 1

        # Broken artifacts (heuristics)
        if "###" in best or "<|endoftext|>" in best or "USER:" in best or "ASSISTANT:" in best:
            broken_outputs += 1

    # === REPORT ===
    n = len(data)
    print()
    print("=" * 60)
    print(f"ATTR COVERAGE ({n} records)")
    print("=" * 60)
    print(f"Mean cov rate:    {statistics.mean(cov_rates):.3f}")
    print(f"Full cov (100%):  {cov_full} ({cov_full/n*100:.2f}%)")
    print(f"Zero cov (0%):    {no_attr_covered} ({no_attr_covered/n*100:.2f}%)")
    print(f"Zero hallucination: {hallu_zero} ({hallu_zero/n*100:.2f}%)")
    print(f"High hallucination (>1): {high_hallu} ({high_hallu/n*100:.2f}%)")

    print()
    print("=" * 60)
    print(f"QUERY LENGTH (best_query)")
    print("=" * 60)
    print(f"Chars: mean={statistics.mean(char_lens):.1f}, median={statistics.median(char_lens):.0f}, "
          f"min={min(char_lens)}, max={max(char_lens)}, p90={sorted(char_lens)[int(n*0.9)]}")
    print(f"Words: mean={statistics.mean(word_lens):.1f}, median={statistics.median(word_lens):.0f}, "
          f"min={min(word_lens)}, max={max(word_lens)}, p90={sorted(word_lens)[int(n*0.9)]}")
    print(f"Too short (<5 words): {short_queries} ({short_queries/n*100:.2f}%)")
    print(f"Too long  (>50 words): {long_queries} ({long_queries/n*100:.2f}%)")

    print()
    print("=" * 60)
    print(f"CANDIDATE DIVERSITY (K=4 intra-record Jaccard)")
    print("=" * 60)
    print(f"Pairwise Jaccard mean:  {statistics.mean(diversities):.3f}")
    print(f"  cand0 vs cand1: {statistics.mean(diversity_per_slot[0]):.3f}")
    print(f"  cand0 vs cand2: {statistics.mean(diversity_per_slot[1]):.3f}")
    print(f"  cand0 vs cand3: {statistics.mean(diversity_per_slot[2]):.3f}")
    # Higher = more redundant, lower = more diverse

    print()
    print("=" * 60)
    print(f"TOP 20 QUERY PREFIXES (first 5 words)")
    print("=" * 60)
    for prefix, cnt in prefix_counts.most_common(20):
        print(f"  {cnt:5d} ({cnt/n*100:5.2f}%)  '{prefix}'")

    print()
    print("=" * 60)
    print(f"QUALITY FLAGS")
    print("=" * 60)
    print(f"Broken output artifacts: {broken_outputs} ({broken_outputs/n*100:.2f}%)")

    # Best query score pattern (cov, hallu)
    cand_score_dist = Counter()
    for d in data:
        for s in d.get("all_candidates_scores", []):
            # s = [cov, hallu, broken]
            cand_score_dist[tuple(s)] += 1
    print()
    print(f"CANDIDATE SCORE DISTRIBUTION [cov, hallu, broken] (per-cand, total ~{sum(cand_score_dist.values())}):")
    for s, cnt in cand_score_dist.most_common(10):
        print(f"  {list(s)}: {cnt}")

    # Per-attr coverage (which attrs are most/least often included)
    print()
    print("=" * 60)
    print("PER-ATTR COVERAGE (how often each attr type is in best query)")
    print("=" * 60)
    attr_cov = Counter()  # attr_name -> count
    attr_total = Counter()
    for d in data:
        attrs = d.get("attrs_used", {})
        for k in attrs.keys():
            attr_total[k] += 1
        # Use attr_check by re-matching best against attrs values
        best = d.get("y_plus_query_inject", "").lower()
        for k, v in attrs.items():
            v_lower = str(v).lower().strip()
            if v_lower and v_lower in best:
                attr_cov[k] += 1
    for attr, total in attr_total.most_common():
        cov = attr_cov.get(attr, 0)
        print(f"  {attr:30s}  {cov}/{total}  ({cov/total*100:.1f}%)")


if __name__ == "__main__":
    main()
