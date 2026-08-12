#!/usr/bin/env python3
"""分析 14-stage template query 与 08-stage LLM expression_style query 的检索值差异。

数据源：
- 13_query_template/{category}/retrieval_template_summary.json
  提供 template query (clean) 在各 retriever 上的 H@K / P@K / NDCG@K
- 13_query_template/{category}/retrieval_template_summary_noisy.json
  提供 template query (noisy, 07 lambdamart_userbased 注入) 在各 retriever 上的指标
- 08_retrieval/{category}/retrieval_expression_style_summary.json
  提供 LLM 改写 query (clean) 在同样 retriever 上的指标
- 13_preprocessed_retrieval/{category}/preprocessed_vs_noisy_results.json
  raw_noisy_results 字段提供 LLM 改写 query (noisy) 的指标 (541-1171 query 样本)
"""

import json
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
RETRIEVERS = ["bge", "e5", "minilm", "star", "ance", "splade", "colbertv2", "bm25"]
LLM_RERANK_RETRIEVER = "bge+llm_rerank"


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get_result_14(retriever: str, summary: dict | None) -> dict | None:
    if not summary:
        return None
    for r in summary.get("results", []):
        if r.get("retriever") == retriever:
            return r
    return None


def get_result_08(retriever: str, summary: dict | None) -> dict | None:
    if not summary:
        return None
    for r in summary.get("all_results_combined", []):
        if r.get("retriever") == retriever:
            return r
    return None


def get_result_08_noisy(retriever: str, summary: dict | None) -> dict | None:
    """Pull 08-noisy H@10 from 13-stage `raw_noisy_results`."""
    if not summary:
        return None
    for r in summary.get("raw_noisy_results", []):
        if r.get("retriever") == retriever:
            return r
    return None


def get_metrics_14(retriever: str, summary: dict | None) -> dict | None:
    r = get_result_14(retriever, summary)
    return r.get("metrics", {}) if r else None


def get_metrics_08(retriever: str, summary: dict | None) -> dict | None:
    r = get_result_08(retriever, summary)
    return r.get("metrics", {}) if r else None


def get_metrics_08_noisy(retriever: str, summary: dict | None) -> dict | None:
    r = get_result_08_noisy(retriever, summary)
    return r.get("metrics", {}) if r else None


# ============================================================================
# Stage 15 LLM rerank 数据加载（bge+llm_rerank）
# ============================================================================
# H@10 = 1 if target asin in llm_final_ranked_asins[:10] else 0
# 数据源：15_llm_rerank/{category}/{retriever}__{query_type}_top100_rerank.jsonl
#   - 14-template + rerank: bge__template_top100_rerank.jsonl (clean)
#                            bge__template_noisy_top100_rerank.jsonl (noisy)
#   - 08-LLM + rerank:      bge__correct_top100_rerank.jsonl (clean)
#                            bge__noisy_top100_rerank.jsonl (noisy)


def _compute_h10_from_rerank_jsonl(jsonl_path: Path) -> dict | None:
    """从 15 阶段 rerank jsonl 计算 H@10 (status=ok 记录) 和 n_records。"""
    if not jsonl_path.exists():
        return None
    n_total = 0
    n_hit = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            n_total += 1
            target = r.get("asin")
            top10 = r.get("llm_final_ranked_asins", [])[:10]
            if target in top10:
                n_hit += 1
    if n_total == 0:
        return None
    return {"H@10": n_hit / n_total, "n_records": n_total}


def get_llm_rerank_metrics(category: str, query_setting: str) -> dict | None:
    """加载 15 阶段 bge+llm_rerank 的 H@10。

    query_setting:
      - "14_template_clean"  → bge__template_top100_rerank.jsonl
      - "14_template_noisy"  → bge__template_noisy_top100_rerank.jsonl
      - "08_llm_clean"       → bge__correct_top100_rerank.jsonl
      - "08_llm_noisy"       → bge__noisy_top100_rerank.jsonl
    """
    filename_map = {
        "14_template_clean": "bge__template_top100_rerank.jsonl",
        "14_template_noisy": "bge__template_noisy_top100_rerank.jsonl",
        "08_llm_clean": "bge__correct_top100_rerank.jsonl",
        "08_llm_noisy": "bge__noisy_top100_rerank.jsonl",
    }
    filename = filename_map.get(query_setting)
    if not filename:
        raise ValueError(f"Unsupported query_setting: {query_setting}")
    jsonl_path = REPO_ROOT / "result" / "personal_query" / "15_llm_rerank" / category / filename
    return _compute_h10_from_rerank_jsonl(jsonl_path)


def get_with_template_h10(retriever: str, category: str, all_data: dict, noisy: bool = False) -> float | None:
    """统一接口：返回 (retriever, category) 在 14-template (clean or noisy) 上的 H@10。

    retriever=LLM_RERANK_RETRIEVER 时走 stage 15 jsonl，其它走 stage 14 summary。
    """
    if retriever == LLM_RERANK_RETRIEVER:
        data = get_llm_rerank_metrics(category, "14_template_noisy" if noisy else "14_template_clean")
        return data["H@10"] if data else None
    summary = all_data[category]["s14n"] if noisy else all_data[category]["s14"]
    m = get_metrics_14(retriever, summary)
    return m.get("H@10") if m else None


def get_with_llm_h10(retriever: str, category: str, all_data: dict, noisy: bool = False) -> float | None:
    """统一接口：返回 (retriever, category) 在 08-LLM (clean or noisy) 上的 H@10。

    retriever=LLM_RERANK_RETRIEVER 时走 stage 15 jsonl，其它走 stage 08/13 summary。
    """
    if retriever == LLM_RERANK_RETRIEVER:
        data = get_llm_rerank_metrics(category, "08_llm_noisy" if noisy else "08_llm_clean")
        return data["H@10"] if data else None
    if noisy:
        m = get_metrics_08_noisy(retriever, all_data[category]["s13_noisy"])
    else:
        m = get_metrics_08(retriever, all_data[category]["s08"])
    return m.get("H@10") if m else None


def fmt_pct(v):
    return f"{v * 100:.2f}%" if v is not None else "  N/A "


def fmt_signed(v):
    if v is None:
        return "    N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"


def main() -> None:
    print("=" * 90)
    print("14-stage (template) vs 08-stage (LLM expression_style) 检索值对比")
    print("=" * 90)

    all_data = {}
    for cat in CATEGORIES:
        print(f"\n处理 {cat}...")
        s14 = load(REPO_ROOT / "result" / "personal_query" / "13_query_template" / cat / "retrieval_template_summary.json")
        s14n = load(REPO_ROOT / "result" / "personal_query" / "13_query_template" / cat / "retrieval_template_summary_noisy.json")
        s08 = load(REPO_ROOT / "result" / "personal_query" / "08_retrieval" / cat / "retrieval_expression_style_summary.json")
        s13_noisy = load(REPO_ROOT / "result" / "personal_query" / "13_preprocessed_retrieval" / cat / "preprocessed_vs_noisy_results.json")
        all_data[cat] = {"s14": s14, "s14n": s14n, "s08": s08, "s13_noisy": s13_noisy}
        print(f"  14-stage (clean):   {'OK' if s14 else 'MISSING'}")
        print(f"  14-stage (noisy):   {'OK' if s14n else 'MISSING'}")
        print(f"  08-stage (clean):   {'OK' if s08 else 'MISSING'}")
        print(f"  13-stage (08-noisy):{'OK' if s13_noisy else 'MISSING'}")

    print("\n" + "=" * 90)
    print("表 1: 各 Retriever 跨 3 个 Category 的 H@10 详细对比")
    print("=" * 90)

    for r in RETRIEVERS:
        print(f"\n--- {r} ---")
        print(f"  {'Category':<22} {'14-tmpl':>10} {'08-llm':>10} {'diff':>10}   {'N_users':>8} {'N_queries':>10}")
        cat_diffs = []
        for cat in CATEGORIES:
            m14 = get_metrics_14(r, all_data[cat]["s14"])
            m08 = get_metrics_08(r, all_data[cat]["s08"])
            v14 = m14.get("H@10") if m14 else None
            v08 = m08.get("H@10") if m08 else None
            n_users = (get_result_14(r, all_data[cat]["s14"]) or {}).get("num_users") \
                or (get_result_08(r, all_data[cat]["s08"]) or {}).get("num_users")
            n_queries = (get_result_14(r, all_data[cat]["s14"]) or {}).get("num_queries") \
                or (get_result_08(r, all_data[cat]["s08"]) or {}).get("num_queries")
            diff = (v14 - v08) if (v14 is not None and v08 is not None) else None
            cat_diffs.append(diff)
            print(f"  {cat:<22} {fmt_pct(v14):>10} {fmt_pct(v08):>10} {fmt_signed(diff):>10}   "
                  f"{str(n_users) if n_users is not None else 'N/A':>8} {str(n_queries) if n_queries is not None else 'N/A':>10}")
        valid = [d for d in cat_diffs if d is not None]
        avg = sum(valid) / len(valid) if valid else None
        direction = "📈 template 胜" if avg and avg > 0 else ("📉 LLM 胜" if avg and avg < 0 else "➖ 持平")
        print(f"  {'平均':<22} {'':>10} {'':>10} {fmt_signed(avg):>10}   {direction}")

    print("\n" + "=" * 90)
    print("表 2: 完整指标对比 (H@1, P@1, P@5, H@5, P@10, H@10, N@10)")
    print("=" * 90)

    metric_keys = ["H@1", "P@1", "H@5", "P@5", "H@10", "P@10", "N@10"]
    print(f"\n{'Category':<22} {'Retriever':<12} {'Mode':<10}", end="")
    for k in metric_keys:
        print(f" {k:>9}", end="")
    print()

    for cat in CATEGORIES:
        print(f"\n=== {cat} ===")
        for r in RETRIEVERS:
            m14 = get_metrics_14(r, all_data[cat]["s14"])
            m08 = get_metrics_08(r, all_data[cat]["s08"])
            for mode_label, metrics in [("14-tmpl", m14), ("08-llm", m08)]:
                print(f"  {cat[:18]:<22} {r:<12} {mode_label:<10}", end="")
                for k in metric_keys:
                    v = metrics.get(k) if metrics else None
                    print(f" {fmt_pct(v):>9}", end="")
                print()

    print("\n" + "=" * 90)
    print("表 3: Retriever 维度汇总 (3-category 平均)")
    print("=" * 90)

    avg_metrics = {}
    for r in RETRIEVERS:
        per_metric = {k: [] for k in metric_keys}
        for cat in CATEGORIES:
            m14 = get_metrics_14(r, all_data[cat]["s14"])
            m08 = get_metrics_08(r, all_data[cat]["s08"])
            for k in metric_keys:
                v14 = m14.get(k) if m14 else None
                v08 = m08.get(k) if m08 else None
                if v14 is not None and v08 is not None:
                    per_metric[k].append(v14 - v08)
        avg_metrics[r] = {k: (sum(v) / len(v) if v else None) for k, v in per_metric.items()}

    print(f"\n{'Retriever':<12}", end="")
    for k in metric_keys:
        print(f" {k:>9}", end="")
    print()
    print("-" * (12 + 10 * len(metric_keys)))
    for r in RETRIEVERS:
        print(f"{r:<12}", end="")
        for k in metric_keys:
            print(f" {fmt_signed(avg_metrics[r][k]):>9}", end="")
        print()

    print("\n" + "=" * 90)
    print("表 4: Category 维度汇总 (8-retriever 平均)")
    print("=" * 90)

    print(f"\n{'Category':<22} {'14-avg':>9} {'08-avg':>9} {'diff':>9}   {'N_users':>8} {'N_queries':>10}")
    for cat in CATEGORIES:
        m14_list, m08_list = [], []
        for r in RETRIEVERS:
            m14 = get_metrics_14(r, all_data[cat]["s14"])
            m08 = get_metrics_08(r, all_data[cat]["s08"])
            if m14:
                m14_list.append(m14.get("H@10"))
            if m08:
                m08_list.append(m08.get("H@10"))
        avg14 = sum(m14_list) / len(m14_list) if m14_list else None
        avg08 = sum(m08_list) / len(m08_list) if m08_list else None
        diff = (avg14 - avg08) if (avg14 is not None and avg08 is not None) else None
        s14 = all_data[cat]["s14"]
        s08 = all_data[cat]["s08"]
        n_users = s14.get("user_count") if s14 else None
        n_queries = s14.get("query_count") if s14 else None
        print(f"{cat:<22} {fmt_pct(avg14):>9} {fmt_pct(avg08):>9} {fmt_signed(diff):>9}   "
              f"{str(n_users) if n_users else 'N/A':>8} {str(n_queries) if n_queries else 'N/A':>10}")

    print("\n" + "=" * 90)
    print("表 5: 紧凑 H@10 矩阵 (Retriever × {With Template, Without Template, Diff})")
    print("=" * 90)

    header = f"  {'Retriever':<12}"
    cats_header = "  ".join(f"{c[:8]:>8}" for c in CATEGORIES)
    print(f"                       With Template (H@10 %)              Without Template (H@10 %)              Diff")
    print(f"  {'Retriever':<12}  {cats_header} {'Avg':>8}   {cats_header} {'Avg':>8}   {'Avg_T − Avg_P':>13}")
    print("  " + "-" * (12 + 11 * 8 + 13))

    summary_table5 = {}
    table5_retrievers = RETRIEVERS + [LLM_RERANK_RETRIEVER]
    for r in table5_retrievers:
        v14_by_cat, v08_by_cat = [], []
        v14_vals, v08_vals = [], []
        for cat in CATEGORIES:
            v14 = get_with_template_h10(r, cat, all_data, noisy=False)
            v08 = get_with_llm_h10(r, cat, all_data, noisy=False)
            v14_by_cat.append(fmt_pct(v14))
            v08_by_cat.append(fmt_pct(v08))
            if v14 is not None:
                v14_vals.append(v14)
            if v08 is not None:
                v08_vals.append(v08)
        avg14 = sum(v14_vals) / len(v14_vals) if v14_vals else None
        avg08 = sum(v08_vals) / len(v08_vals) if v08_vals else None
        diff = (avg14 - avg08) if (avg14 is not None and avg08 is not None) else None
        print(f"  {r:<14}  "
              f"{v14_by_cat[0]:>8} {v14_by_cat[1]:>8} {v14_by_cat[2]:>8} {fmt_pct(avg14):>8}   "
              f"{v08_by_cat[0]:>8} {v08_by_cat[1]:>8} {v08_by_cat[2]:>8} {fmt_pct(avg08):>8}   "
              f"{fmt_signed(diff):>13}")
        summary_table5[r] = {
            "with_template_H@10": {cat: v14_vals[i] for i, cat in enumerate(CATEGORIES) if i < len(v14_vals)},
            "with_template_avg": avg14,
            "without_template_H@10": {cat: v08_vals[i] for i, cat in enumerate(CATEGORIES) if i < len(v08_vals)},
            "without_template_avg": avg08,
            "diff_avg": diff,
            "n_categories_template": len(v14_vals),
            "n_categories_llm": len(v08_vals),
        }

    print("\n" + "=" * 110)
    print("表 6: 2×2 H@10 设计 (clean×noisy × template×LLM), 列含 Avg + Diff (template-vs-LLM) + Δ_robustness")
    print("=" * 110)
    print("  说明: 14-clean/14-noisy 是全量 (Baby 6961 / Grocery 5753 / Pet 14108 queries);")
    print("        08-noisy 来自 13 阶段 raw_noisy_results, 样本量较小 (Baby 541 / Grocery 536 / Pet 1171 queries)。")
    print("        Δ_robustness = (14-noisy − 14-clean) − (08-noisy − 08-clean), 正值 = 模板对噪声更鲁棒。")

    summary_table6 = {}
    print()
    cats_short = " ".join(f"{c[:8]:>8}" for c in CATEGORIES)
    header_top = (
        f"  {'Retriever':<12}  "
        f"  --- 14-template (H@10) ---                  "
        f"  --- 08-LLM (H@10) ---"
    )
    print(header_top)
    print(f"  {'Retriever':<12}  "
          f"{'clean':>8} {'noisy':>8} {'Δ_T':>8} {'Avg':>8}   "
          f"{'clean':>8} {'noisy':>8} {'Δ_P':>8} {'Avg':>8}   "
          f"{'Δ_robust':>10}  (Avg = 3-cat avg)")
    print("  " + "-" * 108)

    for r in table5_retrievers:
        v14c_vals, v14n_vals, v08c_vals, v08n_vals = [], [], [], []
        n14c, n14n, n08c, n08n = None, None, None, None
        per_cat = {}
        for cat in CATEGORIES:
            v14c = get_with_template_h10(r, cat, all_data, noisy=False)
            v14n = get_with_template_h10(r, cat, all_data, noisy=True)
            v08c = get_with_llm_h10(r, cat, all_data, noisy=False)
            v08n = get_with_llm_h10(r, cat, all_data, noisy=True)
            # 样本量信息（仅 LLM_RERANK 时用到，其它走原 num_queries）
            if r == LLM_RERANK_RETRIEVER:
                t14c = get_llm_rerank_metrics(cat, "14_template_clean")
                t14n = get_llm_rerank_metrics(cat, "14_template_noisy")
                t08c = get_llm_rerank_metrics(cat, "08_llm_clean")
                t08n = get_llm_rerank_metrics(cat, "08_llm_noisy")
                n14c = t14c["n_records"] if t14c else None
                n14n = t14n["n_records"] if t14n else None
                n08c = t08c["n_records"] if t08c else None
                n08n = t08n["n_records"] if t08n else None
            else:
                r14c = get_result_14(r, all_data[cat]["s14"])
                r14n = get_result_14(r, all_data[cat]["s14n"])
                r08c = get_result_08(r, all_data[cat]["s08"])
                r08n = get_result_08_noisy(r, all_data[cat]["s13_noisy"])
                if r14c: n14c = r14c.get("num_queries")
                if r14n: n14n = r14n.get("num_queries")
                if r08c: n08c = r08c.get("num_queries")
                if r08n: n08n = r08n.get("num_queries")
            if v14c is not None: v14c_vals.append(v14c)
            if v14n is not None: v14n_vals.append(v14n)
            if v08c is not None: v08c_vals.append(v08c)
            if v08n is not None: v08n_vals.append(v08n)
            per_cat[cat] = {"14c": v14c, "14n": v14n, "08c": v08c, "08n": v08n}

        avg14c = sum(v14c_vals) / len(v14c_vals) if v14c_vals else None
        avg14n = sum(v14n_vals) / len(v14n_vals) if v14n_vals else None
        avg08c = sum(v08c_vals) / len(v08c_vals) if v08c_vals else None
        avg08n = sum(v08n_vals) / len(v08n_vals) if v08n_vals else None

        delta_t = (avg14n - avg14c) if (avg14n is not None and avg14c is not None) else None
        delta_p = (avg08n - avg08c) if (avg08n is not None and avg08c is not None) else None
        delta_robust = (
            (delta_t - delta_p)
            if (delta_t is not None and delta_p is not None)
            else None
        )
        if delta_robust is not None and delta_robust > 0:
            verdict = "📈 模板更鲁棒"
        elif delta_robust is not None and delta_robust < 0:
            verdict = "📉 LLM 更鲁棒"
        elif delta_robust is not None:
            verdict = "➖ 持平"
        else:
            verdict = "N/A"

        print(f"  {r:<12}  "
              f"{fmt_pct(avg14c):>8} {fmt_pct(avg14n):>8} {fmt_signed(delta_t):>8} {fmt_pct(avg14c if avg14n is None else ((avg14c+avg14n)/2 if avg14c is not None and avg14n is not None else None)):>8}   "
              f"{fmt_pct(avg08c):>8} {fmt_pct(avg08n):>8} {fmt_signed(delta_p):>8} {fmt_pct(avg08c if avg08n is None else ((avg08c+avg08n)/2 if avg08c is not None and avg08n is not None else None)):>8}   "
              f"{fmt_signed(delta_robust):>10}  {verdict}")
        summary_table6[r] = {
            "per_category": per_cat,
            "avg_14_clean": avg14c,
            "avg_14_noisy": avg14n,
            "avg_08_clean": avg08c,
            "avg_08_noisy": avg08n,
            "delta_template": delta_t,
            "delta_llm": delta_p,
            "delta_robustness": delta_robust,
            "n_14_clean": n14c,
            "n_14_noisy": n14n,
            "n_08_clean": n08c,
            "n_08_noisy": n08n,
        }

    print()
    print("  Sample-size note: 14 stages run on full population (Baby 6961 / Grocery 5753 / Pet 14108 queries).")
    print("  08-noisy values come from 13-stage raw_noisy_results, which is a smaller sample")
    print("  (Baby 541 / Grocery 536 / Pet 1171 queries). Trend direction is reliable, magnitude is approximate.")

    # ========================================================================
    # Table 7: bge + LLM Rerank (Stage 15) — 3-domain avg
    # ========================================================================
    print("\n" + "=" * 110)
    print("表 7: bge + LLM Rerank (Stage 15) — per-category H@10 + 3-domain avg")
    print("=" * 110)
    print("  说明: first-stage = bge 检索 top-100, second-stage = MiniMax M2.5 LLM rerank")
    print("  14-template + rerank 数据目前只有 Baby_Products 一类（其他两类未跑）")

    table7 = {LLM_RERANK_RETRIEVER: {}}
    for setting in ("14_template_clean", "14_template_noisy", "08_llm_clean", "08_llm_noisy"):
        per_cat = {}
        for cat in CATEGORIES:
            data = get_llm_rerank_metrics(cat, setting)
            per_cat[cat] = data["H@10"] if data else None
        available = [v for v in per_cat.values() if v is not None]
        avg = sum(available) / len(available) if available else None
        table7[LLM_RERANK_RETRIEVER][setting] = {
            "per_category": per_cat,
            "avg": avg,
            "n_categories": len(available),
        }

    # 打印 per-category 矩阵
    print(f"\n  {'Setting':<22}", end="")
    for cat in CATEGORIES:
        print(f" {cat[:8]:>14}", end="")
    print(f" {'Avg':>10} {'N_dom':>6}")
    print("  " + "-" * 80)
    for setting in ("14_template_clean", "14_template_noisy", "08_llm_clean", "08_llm_noisy"):
        d = table7[LLM_RERANK_RETRIEVER][setting]
        per_cat = d["per_category"]
        avg = d["avg"]
        n_dom = d["n_categories"]
        print(f"  {setting:<22}", end="")
        for cat in CATEGORIES:
            v = per_cat.get(cat)
            s = f"{v * 100:.2f}%" if v is not None else "    N/A"
            print(f" {s:>14}", end="")
        s_avg = f"{avg * 100:.2f}%" if avg is not None else "    N/A"
        print(f" {s_avg:>10} {n_dom:>6}")

    # Δ 鲁棒性分析（template rerank vs LLM rerank 在噪声下的对比）
    print()
    print("  Δ(Error−Clean) (bge + LLM rerank):")
    d14c = table7[LLM_RERANK_RETRIEVER]["14_template_clean"]
    d14n = table7[LLM_RERANK_RETRIEVER]["14_template_noisy"]
    d08c = table7[LLM_RERANK_RETRIEVER]["08_llm_clean"]
    d08n = table7[LLM_RERANK_RETRIEVER]["08_llm_noisy"]

    delta_template = (d14n["avg"] - d14c["avg"]) if (d14c["avg"] is not None and d14n["avg"] is not None) else None
    delta_llm = (d08n["avg"] - d08c["avg"]) if (d08c["avg"] is not None and d08n["avg"] is not None) else None
    delta_robust = (
        (delta_template - delta_llm)
        if (delta_template is not None and delta_llm is not None)
        else None
    )

    table7[LLM_RERANK_RETRIEVER]["delta_template"] = delta_template
    table7[LLM_RERANK_RETRIEVER]["delta_llm"] = delta_llm
    table7[LLM_RERANK_RETRIEVER]["delta_robustness"] = delta_robust

    print(f"    With Template (14):   Δ(Error-Clean) = {fmt_signed(delta_template)}  (样本: {d14c['n_categories']} domains)")
    print(f"    With LLM (08):        Δ(Error-Clean) = {fmt_signed(delta_llm)}  (样本: {d08c['n_categories']} domains)")
    print(f"    Δ_robustness = Δ_template − Δ_LLM   = {fmt_signed(delta_robust)}")
    if delta_robust is not None and delta_robust > 0:
        print("    解读: 模板查询对噪声更鲁棒（+ Δ_robustness > 0）")
    elif delta_robust is not None and delta_robust < 0:
        print("    解读: LLM 改写查询对噪声更鲁棒（− Δ_robustness < 0）")

    print()
    print("  Sample-size note: 14-template + rerank 当前仅 Baby_Products 一类 250 条 smoke-test 样本；")
    print("                    08-LLM + rerank 是 3 域全量（Baby 1170 / Grocery 5753 / Pet 14108 records 量级）。")

    output_file = REPO_ROOT / "result" / "personal_query" / "13_query_template" / "template_vs_llm_analysis.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for cat in CATEGORIES:
        serializable[cat] = {
            "retrievers": {
                r: {
                    "14_metrics": get_metrics_14(r, all_data[cat]["s14"]),
                    "14_noisy_metrics": get_metrics_14(r, all_data[cat]["s14n"]),
                    "08_metrics": get_metrics_08(r, all_data[cat]["s08"]),
                    "08_noisy_metrics": get_metrics_08_noisy(r, all_data[cat]["s13_noisy"]),
                }
                for r in RETRIEVERS
            }
        }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_file}")

    table5_file = REPO_ROOT / "result" / "personal_query" / "13_query_template" / "template_vs_llm_h10_table5.json"
    with open(table5_file, "w", encoding="utf-8") as f:
        json.dump(summary_table5, f, ensure_ascii=False, indent=2, default=str)
    print(f"表 5 数据已保存到: {table5_file}")

    table6_file = REPO_ROOT / "result" / "personal_query" / "13_query_template" / "template_vs_llm_h10_table6_2x2.json"
    with open(table6_file, "w", encoding="utf-8") as f:
        json.dump(summary_table6, f, ensure_ascii=False, indent=2, default=str)
    print(f"表 6 (2×2) 数据已保存到: {table6_file}")

    table7_file = REPO_ROOT / "result" / "personal_query" / "13_query_template" / "template_vs_llm_h10_table7_llm_rerank.json"
    with open(table7_file, "w", encoding="utf-8") as f:
        json.dump(table7, f, ensure_ascii=False, indent=2, default=str)
    print(f"表 7 (bge+llm_rerank) 数据已保存到: {table7_file}")


if __name__ == "__main__":
    main()
