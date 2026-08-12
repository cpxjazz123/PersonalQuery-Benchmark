#!/usr/bin/env python3
"""分析 clean query 在不同 cluster 的检索表现（绝对 H@10）

数据源：
- 06_retrieval/{category}/retrieval_by_strict5550_query_gmm_summary.json
  提供 retriever × cluster 的 clean H@10（5550 user 全量评估）
- 10_complexity_analysis_clause_features/{category}/strict5550_query_gmm_user_profiles.jsonl
  提供 user → cluster 映射，用于统计每 cluster 的 N (user 数)
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu")


def load_cluster_counts(category: str) -> dict[int, int]:
    """从 cluster user profiles 统计每个 cluster 的 user 数作为 N"""
    user_profile_file = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / category / "strict5550_query_gmm_user_profiles.jsonl"
    counts = defaultdict(int)
    with open(user_profile_file, encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["cluster_index"]] += 1
    return dict(counts)


def load_cluster_summary(category: str) -> dict:
    """加载 08 生成的 cluster-level clean H@10 汇总"""
    summary_file = REPO_ROOT / "result" / "personal_query" / "06_retrieval" / category / "retrieval_by_strict5550_query_gmm_summary.json"
    with open(summary_file, encoding="utf-8") as f:
        return json.load(f)


def load_user_cluster_map(category: str) -> dict[str, int]:
    """从 user_profiles.jsonl 加载 user_id → cluster_index 映射"""
    user_profile_file = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis_clause_features" / category / "strict5550_query_gmm_user_profiles.jsonl"
    user_to_cluster: dict[str, int] = {}
    with open(user_profile_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user_to_cluster[r["user_id"]] = r["cluster_index"]
    return user_to_cluster


def load_llm_rerank_h10_by_cluster(category: str, user_to_cluster: dict[str, int]) -> dict:
    """从 14_llm_rerank jsonl 计算 bge+llm_rerank 检索器在每个 cluster 的 clean H@10。

    评估公式：H@10 = 1 if target asin in llm_final_ranked_asins[:10] else 0
    评估范围：rerank jsonl 中所有 status=ok 的 records
    """
    rerank_file = REPO_ROOT / "result" / "personal_query" / "14_llm_rerank" / category / "bge__correct_top100_rerank.jsonl"
    if not rerank_file.exists():
        raise FileNotFoundError(f"rerank jsonl not found: {rerank_file}")

    cluster_hits: dict[int, list[int]] = defaultdict(list)
    n_total = 0
    n_unmapped = 0
    with open(rerank_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            n_total += 1
            user_id = r["user_id"]
            target_asin = r["asin"]
            top10 = r.get("llm_final_ranked_asins", [])[:10]
            cluster_idx = user_to_cluster.get(user_id)
            if cluster_idx is None:
                n_unmapped += 1
                continue
            hit = 1 if target_asin in top10 else 0
            cluster_hits[cluster_idx].append(hit)

    cluster_h10 = {cid: sum(hits) / len(hits) for cid, hits in cluster_hits.items() if hits}
    return {
        "cluster_h10": cluster_h10,
        "n_total": n_total,
        "n_unmapped": n_unmapped,
    }


def analyze_category(category: str) -> dict:
    """分析单个类别的 cluster 检索表现"""
    summary = load_cluster_summary(category)
    cluster_counts = load_cluster_counts(category)
    total_users = sum(cluster_counts.values())

    retrievers = {}
    for item in summary.get("retriever_group_results", []):
        retriever = item["retriever"]
        cluster_hit_at10 = item.get("cluster_hit_at10", {})
        cluster_stats = {}
        for cid_key, h10 in cluster_hit_at10.items():
            cid = int(cid_key.split("_")[1])
            cluster_stats[f"cluster_{cid}"] = {
                "N": cluster_counts.get(cid, 0),
                "clean_H@10": h10,
            }

        # 找最强/最弱 cluster
        h10_vals = {k: v["clean_H@10"] for k, v in cluster_stats.items()}
        if h10_vals:
            best = max(h10_vals, key=h10_vals.get)
            worst = min(h10_vals, key=h10_vals.get)
            gap_pct = (h10_vals[best] - h10_vals[worst]) * 100
        else:
            best = worst = None
            gap_pct = 0

        retrievers[retriever] = {
            "cluster_stats": cluster_stats,
            "best_cluster": best,
            "best_h10": h10_vals.get(best, 0) if best else 0,
            "worst_cluster": worst,
            "worst_h10": h10_vals.get(worst, 0) if worst else 0,
            "gap": gap_pct,
            "hit_at10_kruskal_statistic": item.get("hit_at10_kruskal_statistic"),
            "hit_at10_kruskal_pvalue": item.get("hit_at10_kruskal_pvalue"),
            "total_users": total_users,
        }

    # 加入 bge+llm_rerank 检索器（来自 14_llm_rerank jsonl）
    user_to_cluster = load_user_cluster_map(category)
    try:
        rerank_data = load_llm_rerank_h10_by_cluster(category, user_to_cluster)
        cluster_h10_rerank = rerank_data["cluster_h10"]
        cluster_stats_rerank = {}
        for cid, h10 in cluster_h10_rerank.items():
            cluster_stats_rerank[f"cluster_{cid}"] = {
                "N": cluster_counts.get(cid, 0),
                "clean_H@10": h10,
            }
        h10_vals = {k: v["clean_H@10"] for k, v in cluster_stats_rerank.items()}
        if h10_vals:
            best = max(h10_vals, key=h10_vals.get)
            worst = min(h10_vals, key=h10_vals.get)
            gap_pct = (h10_vals[best] - h10_vals[worst]) * 100
        else:
            best = worst = None
            gap_pct = 0
        retrievers["bge+llm_rerank"] = {
            "cluster_stats": cluster_stats_rerank,
            "best_cluster": best,
            "best_h10": h10_vals.get(best, 0) if best else 0,
            "worst_cluster": worst,
            "worst_h10": h10_vals.get(worst, 0) if worst else 0,
            "gap": gap_pct,
            "hit_at10_kruskal_statistic": None,
            "hit_at10_kruskal_pvalue": None,
            "total_users": total_users,
            "n_rerank_records": rerank_data["n_total"],
            "n_unmapped": rerank_data["n_unmapped"],
        }
    except FileNotFoundError as e:
        print(f"  [bge+llm_rerank] 跳过：{e}")

    return retrievers


def main():
    categories = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]

    print("=" * 80)
    print("Clean Query Cluster Analysis (H@10, from 06_retrieval full eval)")
    print("=" * 80)

    all_results = {}
    for cat in categories:
        print(f"\n处理 {cat}...")
        try:
            results = analyze_category(cat)
            all_results[cat] = results
            print(f"  完成，共 {len(results)} 个检索器")
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 打印汇总表
    print("\n" + "=" * 80)
    print("各检索器 Cluster 表现汇总 (H@10 %)")
    print("=" * 80)

    for cat, retriever_results in all_results.items():
        print(f"\n=== {cat} ===")
        print(f"{'Retriever':<12} {'N_users':>7} {'Worst':>8} {'Best':>8} {'Gap':>8} | {'Worst C':>10} {'Best C':>10}")
        print("-" * 80)
        for retriever, data in sorted(retriever_results.items()):
            n = data["total_users"]
            worst = f"{data['worst_h10'] * 100:.2f}%"
            best = f"{data['best_h10'] * 100:.2f}%"
            gap = f"{data['gap']:.2f}%"
            wc = f"{data['worst_cluster']}"
            bc = f"{data['best_cluster']}"
            print(f"{retriever:<12} {n:>7} {worst:>8} {best:>8} {gap:>8} | {wc:<10} {bc:<10}")

    # 打印详细 per-cluster 数据
    print("\n" + "=" * 80)
    print("详细 Cluster H@10 (%)")
    print("=" * 80)

    for cat, retriever_results in all_results.items():
        print(f"\n=== {cat} ===")
        if not retriever_results:
            continue
        sample = next(iter(retriever_results.values()))
        cluster_ids = sorted(sample["cluster_stats"].keys(), key=lambda x: int(x.split("_")[1]))

        header1 = f"{'Retriever':<12}"
        for cid in cluster_ids:
            header1 += f" {cid:>10}"
        header1 += f" {'Gap':>8}"
        header2 = f"{'':12}"
        for cid in cluster_ids:
            n = sample["cluster_stats"][cid]["N"]
            header2 += f" {'N='+str(n):>10}"
        header2 += f" {'':>8}"
        print(header1)
        print(header2)
        print("-" * len(header1))

        for retriever, data in sorted(retriever_results.items()):
            row = f"{retriever:<12}"
            for cid in cluster_ids:
                h10 = data["cluster_stats"][cid]["clean_H@10"]
                row += f" {h10 * 100:>10.2f}%"
            row += f" {data['gap']:>8.2f}%"
            print(row)

    # 汇总：各检索器在三个域的平均 Gap + Kruskal 检验
    print("\n" + "=" * 80)
    print("各检索器跨域平均 Gap (max - min H@10 %)")
    print("=" * 80)
    retrievers = set()
    for cat_results in all_results.values():
        retrievers.update(cat_results.keys())
    retrievers = sorted(retrievers)

    print(f"{'Retriever':<12} {'Baby':>10} {'Grocery':>10} {'Pet':>10} {'Avg Gap':>10}")
    print("-" * 55)
    for retriever in retrievers:
        gaps = []
        for cat in categories:
            if retriever in all_results[cat]:
                gaps.append(all_results[cat][retriever]["gap"])
            else:
                gaps.append(None)
        gap_strs = []
        for g in gaps:
            if g is not None:
                gap_strs.append(f"{g:>10.2f}%")
            else:
                gap_strs.append(f"{'N/A':>10}")
        valid_gaps = [g for g in gaps if g is not None]
        avg_gap = sum(valid_gaps) / len(valid_gaps) if valid_gaps else 0
        print(f"{retriever:<12} {''.join(gap_strs)} {avg_gap:>10.2f}%")

    # 打印 Kruskal-Wallis 检验结果
    print("\n" + "=" * 80)
    print("Kruskal-Wallis 检验 (cluster 间 H@10 分布差异)")
    print("=" * 80)
    print(f"{'Retriever':<12} | {'Baby stat/p':>20} | {'Grocery stat/p':>20} | {'Pet stat/p':>20}")
    print("-" * 90)
    for retriever in retrievers:
        cells = []
        for cat in categories:
            if retriever in all_results[cat]:
                d = all_results[cat][retriever]
                stat = d.get("hit_at10_kruskal_statistic")
                pval = d.get("hit_at10_kruskal_pvalue")
                if stat is not None and pval is not None:
                    cells.append(f"{stat:>9.3f} / {pval:.3e}")
                else:
                    cells.append(f"{'N/A':>20}")
            else:
                cells.append(f"{'N/A':>20}")
        print(f"{retriever:<12} | {cells[0]:>20} | {cells[1]:>20} | {cells[2]:>20}")

    # 保存结果
    output_file = REPO_ROOT / "result" / "personal_query" / "10_complexity_analysis" / "clean_cluster_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
