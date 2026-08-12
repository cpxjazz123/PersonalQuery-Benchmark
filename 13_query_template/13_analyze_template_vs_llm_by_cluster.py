#!/usr/bin/env python3
"""分析 14-stage template query vs 08-stage LLM syntax_depth query 的 H@10 (按 cluster 分组)

数据源：
- 13_query_template/{category}/retrieval_template_summary.json
  14 阶段全量 summary (含 all_query_records 逐 user 命中)
- 12_complexity_analysis_clause_features/{category}/strict5550_query_gmm_user_profiles.jsonl
  user → cluster_index 映射
- 08_retrieval/{category}/retrieval_by_strict5550_query_gmm_summary.json
  08 阶段 cluster-level H@10 (用于对比)
"""

import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
RETRIEVERS = ["bge", "e5", "minilm", "star", "ance", "splade", "colbertv2", "bm25"]


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_user_to_cluster(category: str) -> dict[str, int]:
    fp = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / category / "strict5550_query_gmm_user_profiles.jsonl"
    if not fp.exists():
        raise FileNotFoundError(fp)
    out = {}
    with open(fp, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["user_id"]] = rec["cluster_index"]
    return out


def compute_14_cluster_h10(retriever: str, summary: dict | None, user_to_cluster: dict[str, int]) -> dict[int, dict]:
    if not summary:
        return {}
    for r in summary.get("results", []):
        if r.get("retriever") == retriever:
            records = r.get("all_query_records", [])
            cluster_hits: dict[int, list[int]] = defaultdict(list)
            for rec in records:
                uid = rec.get("user_id")
                cid = user_to_cluster.get(uid)
                if cid is None:
                    continue
                cluster_hits[cid].append(rec.get("hit_at10", 0))
            return {
                cid: {"N": len(v), "H@10": sum(v) / len(v) if v else 0.0}
                for cid, v in cluster_hits.items()
            }
    return {}


def get_08_cluster_h10(retriever: str, summary: dict | None) -> dict[int, dict]:
    if not summary:
        return {}
    for r in summary.get("retriever_group_results", []):
        if r.get("retriever") == retriever:
            cluster_data = r.get("cluster_hit_at10", {})
            cluster_n = r.get("cluster_counts", {})
            out = {}
            for cid_key, h10 in cluster_data.items():
                cid = int(cid_key.split("_")[1])
                out[cid] = {"N": cluster_n.get(cid_key, 0), "H@10": h10}
            return out
    return {}


def main() -> None:
    print("=" * 110)
    print("14-stage (template) vs 08-stage (LLM syntax_depth) H@10 按 Cluster 分组对比")
    print("=" * 110)

    user_cluster_map = {}
    for cat in CATEGORIES:
        user_cluster_map[cat] = load_user_to_cluster(cat)
        print(f"  {cat}: {len(user_cluster_map[cat])} users with cluster labels")

    s14_map, s08_map = {}, {}
    for cat in CATEGORIES:
        s14_map[cat] = load(REPO_ROOT / "result" / "personal_query" / "13_query_template" / cat / "retrieval_template_summary.json")
        s08_map[cat] = load(REPO_ROOT / "result" / "personal_query" / "08_retrieval" / cat / "retrieval_by_strict5550_query_gmm_summary.json")

    all_cluster_ids = set()
    for cat in CATEGORIES:
        for cid in user_cluster_map[cat].values():
            all_cluster_ids.add(cid)
    sorted_cluster_ids = sorted(all_cluster_ids)
    print(f"  全部 cluster id: {sorted_cluster_ids}")
    print()

    for cat in CATEGORIES:
        print("=" * 110)
        print(f"Category: {cat}")
        print("=" * 110)
        cids = sorted(set(user_cluster_map[cat].values()))

        cluster_sizes = {cid: sum(1 for v in user_cluster_map[cat].values() if v == cid) for cid in cids}

        for r in RETRIEVERS:
            h14 = compute_14_cluster_h10(r, s14_map[cat], user_cluster_map[cat])
            h08 = get_08_cluster_h10(r, s08_map[cat])

            if not h14 and not h08:
                continue

            print(f"\n--- {r} ---")
            print(f"  {'Cluster':<10} {'N':>6} {'14-tmpl':>10} {'08-llm':>10} {'diff':>10}   direction")
            cat_diffs = []
            for cid in cids:
                n = cluster_sizes.get(cid, 0)
                v14 = h14.get(cid, {}).get("H@10")
                v08 = h08.get(cid, {}).get("H@10")
                diff = (v14 - v08) if (v14 is not None and v08 is not None) else None
                cat_diffs.append(diff)
                sign = ""
                if diff is not None:
                    sign = "📈 template" if diff > 0 else ("📉 LLM" if diff < 0 else "➖ tie")
                v14s = f"{v14 * 100:.2f}%" if v14 is not None else "    N/A"
                v08s = f"{v08 * 100:.2f}%" if v08 is not None else "    N/A"
                diffs = f"{'+' if diff >= 0 else ''}{diff * 100:.2f}%" if diff is not None else "    N/A"
                print(f"  cluster_{cid:<3} {n:>6} {v14s:>10} {v08s:>10} {diffs:>10}   {sign}")
            valid = [d for d in cat_diffs if d is not None]
            if valid:
                avg = sum(valid) / len(valid)
                direction = "📈 template 胜" if avg > 0 else ("📉 LLM 胜" if avg < 0 else "➖ 持平")
                print(f"  {'avg':<10} {'':>6} {'':>10} {'':>10} {'+' if avg >= 0 else ''}{avg * 100:.2f}%   {direction}")

    print("\n" + "=" * 110)
    print("跨 Category × Cluster 紧凑表 (H@10 %)")
    print("=" * 110)

    for r in RETRIEVERS:
        print(f"\n--- {r} ---")
        for cat in CATEGORIES:
            cids = sorted(set(user_cluster_map[cat].values()))
            cluster_sizes = {cid: sum(1 for v in user_cluster_map[cat].values() if v == cid) for cid in cids}
            h14 = compute_14_cluster_h10(r, s14_map[cat], user_cluster_map[cat])
            h08 = get_08_cluster_h10(r, s08_map[cat])
            if not h14 and not h08:
                continue
            print(f"  {cat}:")
            print(f"    {'cluster':<10}", end="")
            for cid in cids:
                print(f" {f'c{cid}(N={cluster_sizes[cid]})':>14}", end="")
            print()
            for mode_label, data_fn in [("14-tmpl", lambda cid: h14.get(cid, {}).get("H@10")),
                                        ("08-llm ", lambda cid: h08.get(cid, {}).get("H@10"))]:
                print(f"    {mode_label:<10}", end="")
                for cid in cids:
                    v = data_fn(cid)
                    s = f"{v * 100:.2f}%" if v is not None else "    N/A"
                    print(f" {s:>14}", end="")
                print()
            print(f"    {'diff':<10}", end="")
            for cid in cids:
                v14 = h14.get(cid, {}).get("H@10")
                v08 = h08.get(cid, {}).get("H@10")
                if v14 is not None and v08 is not None:
                    d = v14 - v08
                    s = f"{'+' if d >= 0 else ''}{d * 100:.2f}%"
                else:
                    s = "    N/A"
                print(f" {s:>14}", end="")
            print()

    output_file = REPO_ROOT / "result" / "personal_query" / "13_query_template" / "template_vs_llm_h10_by_cluster.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for cat in CATEGORIES:
        serializable[cat] = {}
        for r in RETRIEVERS:
            h14 = compute_14_cluster_h10(r, s14_map[cat], user_cluster_map[cat])
            h08 = get_08_cluster_h10(r, s08_map[cat])
            if not h14 and not h08:
                continue
            serializable[cat][r] = {
                "14_cluster_H@10": {f"cluster_{cid}": v["H@10"] for cid, v in h14.items()},
                "14_cluster_N": {f"cluster_{cid}": v["N"] for cid, v in h14.items()},
                "08_cluster_H@10": {f"cluster_{cid}": v["H@10"] for cid, v in h08.items()},
                "08_cluster_N": {f"cluster_{cid}": v["N"] for cid, v in h08.items()},
            }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
