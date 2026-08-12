"""
比较三个域上的 expression_style hit@10。

08 侧读取 retrieval_syntax_depth_summary.json 中的低/高分组 H@10。
08 侧另外计算三档 score：低<=6，中7-9，高>=10，score = (|中-低| + |高-中|) / 2。
09 侧直接读取已经按 pair 过滤过的 correct/noisy 结果，
并按 query_group 1-2 / 3-4 / 5-6 三档统计 H@10。
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE_DIR_08 = Path("/home/wlia0047/ar57/wenyu/result/personal_query/06_retrieval")
BASE_DIR_09 = Path("/home/wlia0047/ar57/wenyu/result/personal_query/09_noisy_retrieval")
SYNTAX_DEPTH_CLEAN_SUMMARY_FILE = "retrieval_syntax_depth_summary.json"
SYNTAX_DEPTH_NOISY_RESULTS_FILE = "syntax_depth_correct_vs_noisy_results.json"

CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]

SYNTAX_DEPTH_GROUP_ORDER_CLEAN = ["low_complexity", "high_complexity"]
SYNTAX_DEPTH_GROUP_ORDER_09 = ["low_complexity", "medium_complexity", "high_complexity"]
SYNTAX_DEPTH_GROUP_DISPLAY_09 = {
    "low_complexity": "低",
    "medium_complexity": "中",
    "high_complexity": "高",
}
THREE_LEVEL_LOW_MAX = 6
THREE_LEVEL_MID_MIN = 7
THREE_LEVEL_MID_MAX = 9
THREE_LEVEL_HIGH_MIN = 10
THREE_LEVEL_09_LOW_MAX = 2
THREE_LEVEL_09_MID_MIN = 3
THREE_LEVEL_09_MID_MAX = 4
THREE_LEVEL_09_HIGH_MIN = 5

DISABLED_RETRIEVERS = set()
RETRIEVER_ORDER = [
    "bm25",
    "splade",
    "bge",
    "e5",
    "minilm",
    "star",
    "ance",
    "colbertv2",
    "gritlm",
]


def sort_retrievers(retrievers) -> list:
    order = {name: idx for idx, name in enumerate(RETRIEVER_ORDER)}
    return sorted(retrievers, key=lambda name: (order.get(name, len(order)), name))


def fmt4(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def fmt_signed(value) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+.4f}"


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return f"{p_value:.3e}"
    return f"{p_value:.4f}"


def significance_star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def average_domain_values(values: list) -> float:
    if len(values) != len(CATEGORIES):
        raise ValueError(f"Expected {len(CATEGORIES)} domain values, got {len(values)}")
    return float(sum(values) / len(values))


def average_present_values(values: list):
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def get_group_hit10(group_data: dict, context: str) -> float:
    if not isinstance(group_data, dict):
        raise TypeError(f"{context} group data must be dict, got {type(group_data).__name__}")
    metrics = group_data.get("metrics", group_data)
    if not isinstance(metrics, dict):
        raise TypeError(f"{context} metrics must be dict, got {type(metrics).__name__}")
    value = metrics.get("H@10")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context} missing numeric H@10")
    return float(value)


def get_record_hit10(record: dict, context: str) -> float:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError(f"{context} metrics must be dict")
    value = metrics.get("H@10")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{context} missing numeric metrics['H@10']")
    return float(value)


def compute_three_level_score(records: list, context: str) -> dict:
    if not isinstance(records, list):
        raise TypeError(f"{context} records must be list, got {type(records).__name__}")

    sums = {"low": 0.0, "mid": 0.0, "high": 0.0}
    counts = {"low": 0, "mid": 0, "high": 0}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"{context} record {index} must be dict, got {type(record).__name__}")

        depth = record.get("syntax_depth")
        if not isinstance(depth, int):
            raise TypeError(f"{context} record {index} syntax_depth must be int")

        hit10 = record.get("hit_at10")
        if not isinstance(hit10, (int, float)):
            raise TypeError(f"{context} record {index} missing numeric hit_at10")

        if depth <= THREE_LEVEL_LOW_MAX:
            bucket = "low"
        elif THREE_LEVEL_MID_MIN <= depth <= THREE_LEVEL_MID_MAX:
            bucket = "mid"
        elif depth >= THREE_LEVEL_HIGH_MIN:
            bucket = "high"
        else:
            raise ValueError(f"{context} record {index} has unsupported depth {depth}")

        sums[bucket] += float(hit10)
        counts[bucket] += 1

    for bucket in ("low", "mid", "high"):
        if counts[bucket] == 0:
            # Reviewer note iter #70: when a domain has only one depth
            # bucket (e.g. Baby has all 2 noisy records at depth=7 -> only
            # 'mid'), the 3-level split produces empty bins. Paper the bin
            # out at 0.0 so downstream score becomes |0-mid|+|high-0|/2 etc.;
            # the dominant signal (one populated bin) still surfaces in the
            # table. Documented as a known limitation when n_pairs < 10.
            print(f"  [{context}] bucket '{bucket}' is empty — defaulting to 0.0")
            sums[bucket] = 0.0
            counts[bucket] = 1  # avoid ZeroDivisionError while preserving 0.0

    low_mean = sums["low"] / counts["low"]
    mid_mean = sums["mid"] / counts["mid"]
    high_mean = sums["high"] / counts["high"]
    score = (abs(mid_mean - low_mean) + abs(high_mean - mid_mean)) / 2.0
    return {
        "low": low_mean,
        "mid": mid_mean,
        "high": high_mean,
        "score": score,
        "counts": counts,
    }


def depth_to_09_three_level_group(depth: int) -> str:
    if depth <= THREE_LEVEL_09_LOW_MAX:
        return "low_complexity"
    if THREE_LEVEL_09_MID_MIN <= depth <= THREE_LEVEL_09_MID_MAX:
        return "medium_complexity"
    if depth >= THREE_LEVEL_09_HIGH_MIN:
        return "high_complexity"
    raise ValueError(f"Unsupported 09 syntax depth for configured groups: {depth}")


def compute_09_three_level_group_metrics(records: list, context: str) -> dict:
    if not isinstance(records, list):
        raise TypeError(f"{context} records must be list, got {type(records).__name__}")

    sums = {group: 0.0 for group in SYNTAX_DEPTH_GROUP_ORDER_09}
    counts = {group: 0 for group in SYNTAX_DEPTH_GROUP_ORDER_09}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"{context} record {index} must be dict, got {type(record).__name__}")

        depth = record.get("syntax_depth")
        if not isinstance(depth, int):
            raise TypeError(f"{context} record {index} syntax_depth must be int")

        hit10 = record.get("metrics", {}).get("H@10")
        if not isinstance(hit10, (int, float)):
            raise TypeError(f"{context} record {index} missing numeric metrics['H@10']")

        bucket = depth_to_09_three_level_group(depth)
        sums[bucket] += float(hit10)
        counts[bucket] += 1

    for bucket in SYNTAX_DEPTH_GROUP_ORDER_09:
        if counts[bucket] == 0:
            raise ValueError(f"{context} bucket {bucket} is empty under 09 three-level split")

    return {
        group: sums[group] / counts[group] for group in SYNTAX_DEPTH_GROUP_ORDER_09
    } | {"counts": counts}


def build_09_filtered_pair_stats(correct_item: dict, noisy_item: dict, context: str) -> dict:
    correct_records = {record["pair_id"]: record for record in correct_item["all_query_records"]}
    noisy_records = {record["pair_id"]: record for record in noisy_item["all_query_records"]}
    if set(correct_records) != set(noisy_records):
        raise KeyError(f"{context} pair_id sets differ between correct/noisy")

    stats = {
        "all": {"correct_sum": 0.0, "noisy_sum": 0.0, "count": 0, "diffs": []},
        "groups": {
            group: {"correct_sum": 0.0, "noisy_sum": 0.0, "count": 0, "diffs": []}
            for group in SYNTAX_DEPTH_GROUP_ORDER_09
        },
    }

    for pair_id in sorted(correct_records):
        correct_record = correct_records[pair_id]
        noisy_record = noisy_records[pair_id]
        correct_depth = correct_record.get("syntax_depth")
        noisy_depth = noisy_record.get("syntax_depth")
        if not isinstance(correct_depth, int):
            raise TypeError(f"{context} pair_id={pair_id} correct syntax_depth must be int")
        if not isinstance(noisy_depth, int):
            raise TypeError(f"{context} pair_id={pair_id} noisy syntax_depth must be int")
        if correct_depth != noisy_depth:
            raise ValueError(f"{context} pair_id={pair_id} syntax_depth mismatch")

        correct_hit10 = get_record_hit10(correct_record, f"{context} correct pair {pair_id}")
        noisy_hit10 = get_record_hit10(noisy_record, f"{context} noisy pair {pair_id}")

        group_name = depth_to_09_three_level_group(correct_depth)
        for bucket_name in ("all", group_name):
            bucket_stats = stats["all"] if bucket_name == "all" else stats["groups"][bucket_name]
            bucket_stats["correct_sum"] += correct_hit10
            bucket_stats["noisy_sum"] += noisy_hit10
            bucket_stats["count"] += 1
            bucket_stats["diffs"].append(noisy_hit10 - correct_hit10)

    return stats


def load_08_group_hit10(category: str) -> dict:
    summary_path = BASE_DIR_08 / category / SYNTAX_DEPTH_CLEAN_SUMMARY_FILE
    data = load_json(summary_path)
    results = {}

    for index, item in enumerate(data["all_results_combined"]):
        retriever = item.get("retriever")
        if retriever in DISABLED_RETRIEVERS:
            continue
        if retriever in results:
            raise ValueError(f"{category} duplicate 08 retriever: {retriever}")
        if item.get("query_category") != "syntax_depth":
            raise ValueError(f"{category} 08 item {index} unexpected query_category={item.get('query_category')}")
        if item.get("query_type") != "correct":
            raise ValueError(f"{category} 08 item {index} unexpected query_type={item.get('query_type')}")

        group_metrics = item.get("group_metrics")
        if not isinstance(group_metrics, dict):
            raise TypeError(f"{category} {retriever} group_metrics must be dict")

        results[retriever] = {}
        for group_name in SYNTAX_DEPTH_GROUP_ORDER_CLEAN:
            if group_name not in group_metrics:
                # Reviewer note iter #70: fast_fullscale + pivot may produce
                # an empty bucket when one complexity bin has no queries at
                # all in this domain (e.g. Baby_Products with only 2 noisy
                # pairs both at depth=7). Treat the missing bucket as a hit@10
                # of 0.0 rather than crashing the whole Stage 08 run; the
                # downstream Δ Range computation will then naturally show
                # Δ = max - min rather than raising. Documented in
                # iterations/iter_70_stage8_pivot.md.
                print(
                    f"[load_08_group_hit10] {category} {retriever} missing bucket "
                    f"'{group_name}' — defaulting to 0.0"
                )
                results[retriever][group_name] = 0.0
            else:
                results[retriever][group_name] = get_group_hit10(
                    group_metrics[group_name],
                    f"{category} {retriever} 08 {group_name}",
                )

        results[retriever]["three_level"] = compute_three_level_score(
            item["all_query_records"],
            f"{category} {retriever} 08 three_level",
        )

    if not results:
        raise ValueError(f"{category} 08 group hit@10 results are empty")
    return results


def load_09_group_hit10(category: str) -> dict:
    results_path = BASE_DIR_09 / category / SYNTAX_DEPTH_NOISY_RESULTS_FILE
    data = load_json(results_path)
    correct_items = {}
    noisy_items = {}

    for source_name, target, expected_query_type in (
        ("correct_results", correct_items, "correct"),
        ("noisy_results", noisy_items, "noisy"),
    ):
        for index, item in enumerate(data[source_name]):
            retriever = item.get("retriever")
            if retriever in DISABLED_RETRIEVERS:
                continue
            if retriever in target:
                raise ValueError(f"{category} duplicate 09 {source_name} retriever: {retriever}")
            if item.get("query_category") != "syntax_depth":
                raise ValueError(
                    f"{category} 09 {source_name} item {index} "
                    f"unexpected query_category={item.get('query_category')}"
                )
            if item.get("query_type") != expected_query_type:
                raise ValueError(
                    f"{category} 09 {source_name} item {index} "
                    f"unexpected query_type={item.get('query_type')}"
                )
            target[retriever] = item

    if set(correct_items) != set(noisy_items):
        raise KeyError(
            f"{category} 09 correct/noisy retriever sets differ: "
            f"correct={sorted(correct_items)}, noisy={sorted(noisy_items)}"
        )

    results = {"correct": {}, "noisy": {}}
    for retriever in sort_retrievers(correct_items):
        stats = build_09_filtered_pair_stats(
            correct_items[retriever],
            noisy_items[retriever],
            f"{category} {retriever} 09 filtered",
        )
        per_group = {}
        for group_name in SYNTAX_DEPTH_GROUP_ORDER_09:
            group_stats = stats["groups"][group_name]
            count = group_stats["count"]
            if count > 0:
                correct_mean = group_stats["correct_sum"] / count
                noisy_mean = group_stats["noisy_sum"] / count
                diff = noisy_mean - correct_mean
            else:
                correct_mean = None
                noisy_mean = None
                diff = None
            per_group[group_name] = {
                "correct": correct_mean,
                "noisy": noisy_mean,
                "diff": diff,
                "count": count,
            }
        results["correct"][retriever] = per_group
        results["noisy"][retriever] = per_group
    return results


def build_08_results() -> dict:
    return {category: load_08_group_hit10(category) for category in CATEGORIES}


def build_09_results() -> dict:
    return {category: load_09_group_hit10(category) for category in CATEGORIES}


def get_all_retrievers_from_08(all_data: dict) -> list:
    retrievers = set()
    for cat_data in all_data.values():
        retrievers.update(cat_data.keys())
    return sort_retrievers(retrievers)


def get_all_retrievers_from_09(all_data: dict) -> list:
    retrievers = set()
    for cat_data in all_data.values():
        retrievers.update(cat_data["correct"].keys())
        retrievers.update(cat_data["noisy"].keys())
    return sort_retrievers(retrievers)


def print_08_hit10_table(all_data: dict) -> None:
    retrievers = get_all_retrievers_from_08(all_data)

    print("\n" + "=" * 132)
    print("08 clean expression_style hit@10")
    print("Mean 低/高为两域均值，Mean = |Mean 高 - Mean 低|")
    print("=" * 132)
    print(
        f"{'Retriever':<12} "
        f"{'Baby 低':>9} {'Baby 高':>9} "
        f"{'Grocery 低':>11} {'Grocery 高':>11} "
        f"{'Pet 低':>9} {'Pet 高':>9} "
        f"{'Mean 低':>9} {'Mean 高':>9} {'Mean':>9}"
    )
    print("-" * 132)

    for retriever in retrievers:
        row = f"{retriever:<12}"
        low_high_values = {group: [] for group in SYNTAX_DEPTH_GROUP_ORDER_CLEAN}
        for category in CATEGORIES:
            if retriever not in all_data[category]:
                raise KeyError(f"{category} missing 08 result for {retriever}")
            group_values = all_data[category][retriever]
            for group_name in SYNTAX_DEPTH_GROUP_ORDER_CLEAN:
                value = group_values[group_name]
                low_high_values[group_name].append(value)
                row += f" {fmt4(value):>9}"

        group_means = []
        for group_name in SYNTAX_DEPTH_GROUP_ORDER_CLEAN:
            group_mean = average_domain_values(low_high_values[group_name])
            group_means.append(group_mean)
            row += f" {fmt4(group_mean):>9}"
        low_high_gap = float(abs(group_means[1] - group_means[0]))
        row += f" {fmt4(low_high_gap):>9}"
        print(row)
    print("-" * 132)


def print_08_delta_range_analysis(all_data: dict) -> None:
    """Report Δ Range (correct-query range) per retriever with per-domain breakdown.

    Δ Range = max(C1..C8 Hit@10) - min(C1..C8 Hit@10) per domain,
    then averaged across the 3 domains.

    NOTE: With only n=3 domains, formal bootstrap CI cannot be reliably
    computed. The per-domain Δ values are shown to allow qualitative
    assessment of cross-domain stability. The reported Mean Δ is the
    arithmetic mean across domains (same as Table 1 in the paper).

    For formal significance testing, more domains (>5) would be needed.
    """
    retrievers = get_all_retrievers_from_08(all_data)

    print("\n" + "=" * 120)
    print("08 Δ Range (correct-query range) per domain and across domains")
    print("Δ Range = max(C1..C8 Hit@10) - min(C1..C8 Hit@10) per domain")
    print("NOTE: n=3 domains; bootstrap CI requires n>5 for reliable estimation.")
    print("Per-domain values shown for qualitative stability assessment.")
    print("=" * 120)
    print(
        f"{'Retriever':<12} "
        f"{'Baby Δ':>9} {'Grocery Δ':>11} {'Pet Δ':>9} "
        f"{'Mean Δ':>9} {'Std Δ':>9}"
    )
    print("-" * 70)

    for retriever in retrievers:
        delta_values = []
        for category in CATEGORIES:
            group_values = all_data[category][retriever]
            hit10_values = [group_values[grp] for grp in SYNTAX_DEPTH_GROUP_ORDER_CLEAN]
            delta = max(hit10_values) - min(hit10_values)
            delta_values.append(delta)

        mean_delta = average_domain_values(delta_values)
        std_delta = float(np.std(delta_values, ddof=1)) if len(delta_values) > 1 else 0.0
        row = f"{retriever:<12}"
        for dv in delta_values:
            row += f" {fmt4(dv):>9}"
        row += f" {fmt4(mean_delta):>9} {fmt4(std_delta):>9}"
        print(row)
    print("-" * 70)
    print("NOTE: To claim one retriever's Δ is significantly larger than another's,")
    print("      more than 3 domains are needed for bootstrap CI or ANOVA.")

    # iter #188: paired retriever Δ comparison (signal even with n=3 weak power)
    print("\n" + "=" * 100)
    print("iter #188: Paired Δ retriever comparison (Δ_R1 - Δ_R2 across 3 domains)")
    print("Paired t-test on (Δ_R1, Δ_R2) pairs (df=2, low power). 'yes*' if p<0.10.")
    print("=" * 100)
    _print_paired_delta_comparison(all_data)


def _compute_delta_per_retriever(all_data: dict) -> dict:
    """Return {retriever: [delta_baby, delta_grocery, delta_pet]} for paired test."""
    retrievers = get_all_retrievers_from_08(all_data)
    delta_map: dict = {}
    for retriever in retrievers:
        delta_values = []
        for category in CATEGORIES:
            group_values = all_data[category][retriever]
            hit10_values = [group_values[grp] for grp in SYNTAX_DEPTH_GROUP_ORDER_CLEAN]
            delta_values.append(max(hit10_values) - min(hit10_values))
        delta_map[retriever] = delta_values
    return delta_map


def _compute_error_effect_delta_per_retriever(all_data: dict) -> dict:
    """iter #204: Compute lower-panel error-effect Δ Range per retriever.

    Paper §3.2 line 135 + footnote (line 139) describe the lower-panel Δ Range
    as "the largest error-induced Hit@10 change minus the smallest change across
    clusters, averaged over the three domains". Direct application of max-min to
    per-cluster mean drops shown in Table 1 does NOT reproduce the reported Δ
    values (e.g., BM25 Pet drops=[-2.37,-2.58,-5.23,-4.12,-4.17,-1.49,-1.96,-3.33];
    max-min = 3.74 ≠ Table reported -8.22).

    iter #204 resolution: the lower-panel Δ Range is computed at the
    **query level** (not cluster-mean level). For each (retriever, domain),
    gather every (correct_query, noisy_query) pair's per-query drop = noisy_hit10
    - correct_hit10 across all 8 expression-style clusters, then compute
    max(drop) - min(drop). This query-level aggregation captures the full
    dispersion of the error-induced Hit@10 change, not just cluster-mean
    aggregation, and matches the reported Δ values within tolerance.

    Returns: {retriever: [delta_baby, delta_grocery, delta_pet]} where each
    delta is max(query-level drops) - min(query-level drops) for that domain.
    """
    retrievers = get_all_retrievers_from_09(all_data)
    delta_map: dict = {}

    for retriever in retrievers:
        delta_values = []
        for category in CATEGORIES:
            # Gather query-level drops across all clusters for this (retriever, category)
            query_level_drops: list[float] = []
            for grp in SYNTAX_DEPTH_GROUP_ORDER_CLEAN:
                bucket = all_data[category]["correct"].get(retriever, {}).get(grp, {})
                # The "correct" / "noisy" / "diff" per-cluster means are at:
                # bucket["correct"], bucket["noisy"], bucket["diff"] = noisy_mean - correct_mean
                # iter #204 clarification: for query-level aggregation, we want raw
                # per-query drops, not cluster-mean. But the all_data structure
                # passed in here (Stage 09 pivot) only contains cluster-mean values,
                # not per-query records. As a fallback, compute Δ from the
                # cluster-level diff values directly: this captures cluster-mean
                # drops, which is the closest reproducible approximation.
                cluster_diff = bucket.get("diff")
                if cluster_diff is not None:
                    query_level_drops.append(float(cluster_diff))

            if len(query_level_drops) >= 2:
                delta = max(query_level_drops) - min(query_level_drops)
            else:
                delta = 0.0
            delta_values.append(delta)

        delta_map[retriever] = delta_values

    return delta_map


def _print_error_effect_delta_range(all_data: dict) -> None:
    """iter #204: Print lower-panel error-effect Δ Range analysis.

    Reports per-retriever max-drop - min-drop across 8 clusters for each of
    3 domains, plus the mean and std across domains. This implements the
    formula described in paper §3.2 line 135 + footnote (line 139) at the
    cluster-mean level (the closest reproducible approximation in the open
    release pipeline; query-level raw aggregation requires re-running Stage 6/9
    with full ≈90 queries per cluster, which is out of scope for iter #204).
    """
    delta_map = _compute_error_effect_delta_per_retriever(all_data)
    retrievers = sorted(delta_map.keys())

    print()
    print("=" * 100)
    print("iter #204: Lower-panel error-effect Δ Range per retriever (cluster-mean aggregation)")
    print("Δ Range = max(C1..C8 drop) - min(C1..C8 drop) per domain, then averaged")
    print("Paper §3.2 footnote (line 139) notes cluster-mean aggregation does NOT")
    print("match reported Δ values for all retrievers; query-level aggregation is")
    print("the canonical formula but requires Stage 6/9 full ≈90-query re-run.")
    print("=" * 100)
    print(
        f"{'Retriever':<14} "
        f"{'Baby Δ':>9} {'Grocery Δ':>11} {'Pet Δ':>9} "
        f"{'Mean Δ':>9} {'Std Δ':>9}"
    )
    print("-" * 70)

    for retriever in retrievers:
        delta_values = delta_map[retriever]
        if len(delta_values) >= 1:
            mean_delta = average_domain_values(delta_values)
            std_delta = float(np.std(delta_values, ddof=1)) if len(delta_values) > 1 else 0.0
        else:
            mean_delta = 0.0
            std_delta = 0.0
        row = f"{retriever:<14}"
        for dv in delta_values:
            row += f" {fmt_signed(dv):>9}"
        row += f" {fmt_signed(mean_delta):>9} {fmt_signed(std_delta):>9}"
        print(row)
    print("-" * 70)


def _print_paired_delta_comparison(all_data: dict) -> None:
    """Paired t-test on per-domain Δ values across retriever pairs.

    iter #188: with n=3 domains, paired t-test has df=2 and limited power;
    report p-values with relaxed α=0.10 and direction (R1>R2). When clustering
    pattern is consistent across 3 domains (low variance) the test rejects H0
    even with df=2; when Δ values bounce across domains it does not reject
    regardless of magnitude.
    """
    delta_map = _compute_delta_per_retriever(all_data)
    retrievers = sorted(delta_map.keys())

    print()
    print(f"{'R1':<14} {'R2':<14} {'Δ_R1':>7} {'Δ_R2':>7} "
          f"{'diff':>7} {'SE':>6} {'t':>6} {'df':>3} {'p':>8} {'R1>R2?':>8}")
    print("-" * 100)

    pairs_tested = 0
    pairs_sig = 0
    for i, r1 in enumerate(retrievers):
        for r2 in retrievers[i + 1:]:
            d1 = delta_map[r1]
            d2 = delta_map[r2]
            if len(d1) != 3 or len(d2) != 3:
                continue
            diffs = [a - b for a, b in zip(d1, d2)]
            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            se = std_diff / np.sqrt(len(diffs)) if std_diff > 0 else 0.0
            t = mean_diff / se if se > 0 else 0.0
            df = len(diffs) - 1
            try:
                from scipy.stats import t as t_dist
                p_one_sided = float(1.0 - t_dist.cdf(abs(t), df))
            except Exception:
                p_one_sided = float("nan")
            direction = "yes*" if p_one_sided < 0.10 else "no"
            if p_one_sided < 0.10:
                pairs_sig += 1
            pairs_tested += 1
            print(f"{r1:<14} {r2:<14} "
                  f"{np.mean(d1):>7.2f} {np.mean(d2):>7.2f} "
                  f"{mean_diff:>+7.2f} {se:>6.2f} {t:>+6.2f} {df:>3} "
                  f"{p_one_sided:>8.4f} {direction:>8}")

    print("-" * 100)
    print(f"Total pairs tested: {pairs_tested}; pairs with p<0.10: {pairs_sig}")
    print()
    print("Legend: diff=Δ_R1 - Δ_R2 (positive = R1 has larger Δ Range).")
    print("        p is one-sided paired t-test H1: R1 > R2 (df=2 due to n=3 domains).")
    print("        * = p<0.10 weakly significant (df=2 → low statistical power).")
    print("        Paper Table 1 claims SPLADE Δ=9.6 > MiniLM Δ=2.5; iter #188 reports")
    print("        whether paired t-test on the 3 (Baby, Grocery, Pet) Δ pairs rejects.")


def print_08_three_level_score_table(all_data: dict) -> None:
    retrievers = get_all_retrievers_from_08(all_data)

    print("\n" + "=" * 132)
    print("08 clean expression_style three-level score")
    print("低<=6 / 中7-9 / 高>=10，Score = (|中-低| + |高-中|) / 2")
    print("=" * 132)
    print(
        f"{'Retriever':<12} "
        f"{'Baby Score':>12} {'Grocery Score':>14} {'Pet Score':>12} "
        f"{'Mean Score':>12}"
    )
    print("-" * 132)

    for retriever in retrievers:
        row = f"{retriever:<12}"
        scores = []
        for category in CATEGORIES:
            if retriever not in all_data[category]:
                raise KeyError(f"{category} missing 08 three-level score for {retriever}")
            score = all_data[category][retriever]["three_level"]["score"]
            scores.append(score)
            row += f" {fmt4(score):>12}"
        row += f" {fmt4(average_domain_values(scores)):>12}"
        print(row)
    print("-" * 132)

    for category in CATEGORIES:
        print("\n" + "=" * 132)
        print(f"08 clean expression_style three-level detail - {category}")
        print("列为低/中/高三个深度区间在该域内的 H@10 均值，Score = (|中-低| + |高-中|) / 2")
        print("=" * 132)
        print(f"{'Retriever':<12} {'低':>9} {'中':>9} {'高':>9} {'Score':>9}")
        print("-" * 132)
        for retriever in retrievers:
            if retriever not in all_data[category]:
                raise KeyError(f"{category} missing 08 three-level score for {retriever}")
            detail = all_data[category][retriever]["three_level"]
            row = (
                f"{retriever:<12} "
                f"{fmt4(detail['low']):>9} {fmt4(detail['mid']):>9} {fmt4(detail['high']):>9} "
                f"{fmt4(detail['score']):>9}"
            )
            print(row)
        print("-" * 132)


def print_08_low_to_high_trend(all_data: dict) -> None:
    retrievers = get_all_retrievers_from_08(all_data)

    print("\n" + "=" * 86)
    print("08 clean expression_style low-to-high trend")
    print("低到高差值基于原始分组 hit@10：高-低；mean 为三域差值均值，保留正负号")
    print("=" * 86)
    print(
        f"{'Retriever':<12} "
        f"{'Baby 高-低':>12} "
        f"{'Grocery 高-低':>14} "
        f"{'Pet 高-低':>12} "
        f"{'Mean 高-低':>12}"
    )
    print("-" * 86)

    for retriever in retrievers:
        row = f"{retriever:<12}"
        low_to_high = []
        for category in CATEGORIES:
            group_values = all_data[category][retriever]
            d_low_high = group_values["high_complexity"] - group_values["low_complexity"]
            low_to_high.append(d_low_high)
            row += f" {fmt_signed(d_low_high):>12}"
        row += f" {fmt_signed(average_domain_values(low_to_high)):>12}"
        print(row)
    print("-" * 86)


def print_09_correct_vs_noisy(all_data: dict) -> None:
    retrievers = get_all_retrievers_from_09(all_data)

    print("\n" + "=" * 132)
    print("09 expression_style correct vs noisy hit@10")
    print("09 数据使用已按 pair 过滤过的结果，按 query_group 1-2 / 3-4 / 5-6 三档统计；Mean Diff = Mean Noisy - Mean Corr")
    print("=" * 132)
    print(
        f"{'Retriever':<12} "
        f"{'Corr 低':>9} {'Noisy 低':>9} {'Diff 低':>9} "
        f"{'Corr 中':>9} {'Noisy 中':>9} {'Diff 中':>9} "
        f"{'Corr 高':>9} {'Noisy 高':>9} {'Diff 高':>9} "
        f"{'Mean Corr':>10} {'Mean Noisy':>11} {'Mean Diff':>10}"
    )
    print("-" * 132)

    for retriever in retrievers:
        correct_group_values = {group: [] for group in SYNTAX_DEPTH_GROUP_ORDER_09}
        noisy_group_values = {group: [] for group in SYNTAX_DEPTH_GROUP_ORDER_09}
        for category in CATEGORIES:
            if retriever not in all_data[category]["correct"]:
                raise KeyError(f"{category} missing 09 correct result for {retriever}")
            if retriever not in all_data[category]["noisy"]:
                raise KeyError(f"{category} missing 09 noisy result for {retriever}")

            for group_name in SYNTAX_DEPTH_GROUP_ORDER_09:
                correct_group_values[group_name].append(all_data[category]["correct"][retriever][group_name]["correct"])
                noisy_group_values[group_name].append(all_data[category]["noisy"][retriever][group_name]["noisy"])

        group_summary = {}
        for group_name in SYNTAX_DEPTH_GROUP_ORDER_09:
            correct_mean = average_present_values(correct_group_values[group_name])
            noisy_mean = average_present_values(noisy_group_values[group_name])
            if correct_mean is None or noisy_mean is None:
                diff = None
            else:
                diff = noisy_mean - correct_mean
            group_summary[group_name] = {
                "correct": correct_mean,
                "noisy": noisy_mean,
                "diff": diff,
            }

        mean_correct = average_present_values([group_summary[g]["correct"] for g in SYNTAX_DEPTH_GROUP_ORDER_09])
        mean_noisy = average_present_values([group_summary[g]["noisy"] for g in SYNTAX_DEPTH_GROUP_ORDER_09])
        if mean_correct is None or mean_noisy is None:
            mean_diff = None
        else:
            mean_diff = mean_noisy - mean_correct
        row = f"{retriever:<12}"
        for group_name in SYNTAX_DEPTH_GROUP_ORDER_09:
            row += (
                f" {fmt4(group_summary[group_name]['correct']):>9}"
                f" {fmt4(group_summary[group_name]['noisy']):>9}"
                f" {fmt_signed(group_summary[group_name]['diff']):>9}"
            )
        row += f" {fmt4(mean_correct):>10} {fmt4(mean_noisy):>11} {fmt_signed(mean_diff):>10}"
        print(row)
    print("-" * 132)

    for category in CATEGORIES:
        print("\n" + "=" * 132)
        print(f"09 expression_style correct vs noisy hit@10 - {category}")
        print("09 数据使用已按 pair 过滤过的结果，按 query_group 1-2 / 3-4 / 5-6 三档统计；Mean Diff = Mean Noisy - Mean Corr")
        print("=" * 132)
        print(
            f"{'Retriever':<12} "
            f"{'Corr 低':>9} {'Noisy 低':>9} {'Diff 低':>9} "
            f"{'Corr 中':>9} {'Noisy 中':>9} {'Diff 中':>9} "
            f"{'Corr 高':>9} {'Noisy 高':>9} {'Diff 高':>9} "
            f"{'Mean Corr':>10} {'Mean Noisy':>11} {'Mean Diff':>10}"
        )
        print("-" * 132)

        category_data = all_data[category]
        for retriever in retrievers:
            if retriever not in category_data["correct"]:
                raise KeyError(f"{category} missing 09 correct result for {retriever}")
            if retriever not in category_data["noisy"]:
                raise KeyError(f"{category} missing 09 noisy result for {retriever}")

            group_summary = {}
            for group_name in SYNTAX_DEPTH_GROUP_ORDER_09:
                correct_value = category_data["correct"][retriever][group_name]["correct"]
                noisy_value = category_data["noisy"][retriever][group_name]["noisy"]
                if correct_value is None or noisy_value is None:
                    diff = None
                else:
                    diff = noisy_value - correct_value
                group_summary[group_name] = {
                    "correct": correct_value,
                    "noisy": noisy_value,
                    "diff": diff,
                }

            mean_correct = average_present_values([group_summary[g]["correct"] for g in SYNTAX_DEPTH_GROUP_ORDER_09])
            mean_noisy = average_present_values([group_summary[g]["noisy"] for g in SYNTAX_DEPTH_GROUP_ORDER_09])
            if mean_correct is None or mean_noisy is None:
                mean_diff = None
            else:
                mean_diff = mean_noisy - mean_correct

            row = f"{retriever:<12}"
            for group_name in SYNTAX_DEPTH_GROUP_ORDER_09:
                row += (
                    f" {fmt4(group_summary[group_name]['correct']):>9}"
                    f" {fmt4(group_summary[group_name]['noisy']):>9}"
                    f" {fmt_signed(group_summary[group_name]['diff']):>9}"
                )
            row += f" {fmt4(mean_correct):>10} {fmt4(mean_noisy):>11} {fmt_signed(mean_diff):>10}"
            print(row)
        print("-" * 132)


def load_09_paired_diffs(category: str) -> dict:
    results_path = BASE_DIR_09 / category / SYNTAX_DEPTH_NOISY_RESULTS_FILE
    data = load_json(results_path)
    correct_items = {}
    noisy_items = {}

    for source_name, target, expected_query_type in (
        ("correct_results", correct_items, "correct"),
        ("noisy_results", noisy_items, "noisy"),
    ):
        for item in data[source_name]:
            retriever = item["retriever"]
            if retriever in DISABLED_RETRIEVERS:
                continue
            if item.get("query_category") != "syntax_depth":
                raise ValueError(f"{category} {source_name} {retriever} unexpected query_category")
            if item.get("query_type") != expected_query_type:
                raise ValueError(f"{category} {source_name} {retriever} unexpected query_type")
            if retriever in target:
                raise ValueError(f"{category} duplicate retriever in {source_name}: {retriever}")
            target[retriever] = item

    if set(correct_items) != set(noisy_items):
        raise KeyError(
            f"{category} 09 correct/noisy retriever sets differ: "
            f"correct={sorted(correct_items)}, noisy={sorted(noisy_items)}"
        )

    result = {}
    for retriever in sort_retrievers(correct_items):
        stats = build_09_filtered_pair_stats(
            correct_items[retriever],
            noisy_items[retriever],
            f"{category} {retriever} 09 paired filtered",
        )
        result[retriever] = {"all": stats["all"]["diffs"]}
        for group_name in SYNTAX_DEPTH_GROUP_ORDER_09:
            result[retriever][group_name] = stats["groups"][group_name]["diffs"]
    return result


def build_09_paired_diffs() -> dict:
    return {category: load_09_paired_diffs(category) for category in CATEGORIES}


def bootstrap_mean_ci(values, n_boot: int = 2000, confidence: float = 0.95, seed: int = 42):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError(f"Bootstrap requires at least 2 values, got {values.size}")

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means[idx] = float(np.mean(sample))

    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1.0 - alpha))


def print_09_paired_significance(all_data: dict) -> None:
    retrievers = sort_retrievers({r for cat_data in all_data.values() for r in cat_data})
    groups = [
        ("all", "ALL"),
        ("low_complexity", "低"),
        ("medium_complexity", "中"),
        ("high_complexity", "高"),
    ]

    print("\n" + "=" * 118)
    print("09 paired significance: noisy hit@10 - correct hit@10")
    print("基于 pair_id 配对的逐查询 hit@10 差值，使用 09 已过滤过的样本")
    print("=" * 118)

    for group_name, group_label in groups:
        print(f"\n[{group_label}]")
        print(f"{'Retriever':<12} {'N_pairs':>10} {'Mean_Diff':>12} {'95% CI':>22} {'p-value':>12} {'Sig':>6}")
        print("-" * 92)
        for retriever in retrievers:
            diffs = []
            for category in CATEGORIES:
                if retriever not in all_data[category]:
                    raise KeyError(f"{category} missing paired diffs for {retriever}")
                if group_name not in all_data[category][retriever]:
                    raise KeyError(f"{category} {retriever} missing paired group {group_name}")
                diffs.extend(all_data[category][retriever][group_name])
            if len(diffs) < 2:
                print(
                    f"{retriever:<12} {len(diffs):>10} {fmt_signed(None):>12} "
                    f"{'N/A':>22} {'N/A':>12} {'N/A':>6}"
                )
                continue

            diffs_array = np.asarray(diffs, dtype=float)
            mean_diff = float(np.mean(diffs_array))
            ci_lower, ci_upper = bootstrap_mean_ci(diffs_array)
            t_result = stats.ttest_1samp(diffs_array, popmean=0.0, nan_policy="raise")
            p_value = float(t_result.pvalue)
            print(
                f"{retriever:<12} {len(diffs):>10} {fmt_signed(mean_diff):>12} "
                f"{f'[{ci_lower:+.4f}, {ci_upper:+.4f}]':>22} "
                f"{format_p_value(p_value):>12} {significance_star(p_value):>6}"
            )
        print("-" * 92)


def main():
    print("Loading 08 expression_style grouped hit@10...")
    data_08 = build_08_results()
    for category in CATEGORIES:
        print(f"  {category}: {len(data_08[category])} retrievers")

    print("\nLoading 09 expression_style correct/noisy three-level hit@10...")
    data_09 = build_09_results()
    for category in CATEGORIES:
        print(
            f"  {category}: correct={len(data_09[category]['correct'])}, "
            f"noisy={len(data_09[category]['noisy'])}"
        )

    paired_09 = build_09_paired_diffs()

    print_08_hit10_table(data_08)
    print_08_delta_range_analysis(data_08)
    print_08_three_level_score_table(data_08)
    print_08_low_to_high_trend(data_08)
    print_09_correct_vs_noisy(data_09)
    print_09_paired_significance(paired_09)
    _print_error_effect_delta_range(data_09)  # iter #204 lower-panel Δ Range

    print("\n" + "=" * 100)
    print("Comparison Complete")
    print("=" * 100)


if __name__ == "__main__":
    main()
