#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""根据当前 PQB 三域真实结果绘制风格分析图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
CLAUSE_BASE = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features"
RETRIEVAL_BASE = REPO_ROOT / "result" / "personal_query" / "08_retrieval"


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} 为空")
    return rows


def load_domain_inputs(domain: str) -> dict:
    clause_dir = CLAUSE_BASE / domain
    retrieval_dir = RETRIEVAL_BASE / domain
    files = {
        "feature_file": clause_dir / "strict5550_query_gmm_features.jsonl",
        "cluster_user_file": clause_dir / "strict5550_query_gmm_user_profiles.jsonl",
        "review_query_alignment_file": clause_dir / "review_query_user_style_alignment_summary.json",
        "retrieval_summary_file": retrieval_dir / "retrieval_by_strict5550_query_gmm_summary.json",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"{domain} 缺少必需文件 {name}: {path}")
    return files


def build_feature_dataframe(feature_rows: list[dict], cluster_rows: list[dict]) -> pd.DataFrame:
    cluster_index_by_user = {}
    for row in cluster_rows:
        user_id = row.get("user_id")
        cluster_index = row.get("cluster_index")
        if user_id is None or cluster_index is None:
            raise ValueError("strict5550_query_gmm_user_profiles.jsonl 中存在缺失字段")
        if user_id in cluster_index_by_user:
            raise ValueError(f"user {user_id} 在 cluster user profiles 中重复")
        cluster_index_by_user[user_id] = int(cluster_index)

    normalized_rows = []
    for row in feature_rows:
        user_id = row["user_id"]
        if user_id not in cluster_index_by_user:
            raise ValueError(f"user {user_id} 不存在于 cluster user profiles")
        record = {
            "user_id": user_id,
            "asin": row["asin"],
            "query": row["query"],
            "cluster_index": int(cluster_index_by_user[user_id]),
        }
        for feature_name, feature_value in row["features"].items():
            record[feature_name] = float(feature_value)
        normalized_rows.append(record)
    return pd.DataFrame(normalized_rows)


def draw_query_style_map(domain: str, files: dict, out_dir: Path) -> None:
    feature_rows = read_jsonl(files["feature_file"])
    cluster_rows = read_jsonl(files["cluster_user_file"])
    df = build_feature_dataframe(feature_rows, cluster_rows)

    feature_cols = [c for c in df.columns if c not in {"user_id", "asin", "query", "cluster_index"}]
    if len(feature_cols) == 0:
        raise ValueError(f"{domain} 没有可用句法特征列")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    X = StandardScaler().fit_transform(X)
    if len(df) < 10:
        raise ValueError(f"{domain} 数据过少，无法绘制风格映射图")

    perplexity = min(30, max(5, len(df) // 50))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(X)

    df_plot = df.copy()
    df_plot["x"] = embedding[:, 0]
    df_plot["y"] = embedding[:, 1]

    fig, ax = plt.subplots(figsize=(7.4, 5.4), dpi=220)
    for cluster_index in sorted(df_plot["cluster_index"].unique()):
        subset = df_plot[df_plot["cluster_index"] == cluster_index]
        ax.scatter(
            subset["x"],
            subset["y"],
            s=10,
            alpha=0.65,
            label=f"cluster_{cluster_index}",
        )

    ax.set_title(f"{domain}: Query Syntactic Style Map")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title="GMM cluster", fontsize=7, title_fontsize=8, markerscale=2, frameon=True)
    fig.tight_layout()

    out_path = out_dir / f"query_style_map_{domain}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def draw_retriever_cluster_heatmap(domain: str, files: dict, out_dir: Path) -> None:
    pivot, _, _ = load_retrieval_hit10_pivot(files)

    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 0.5 * len(pivot.index))), dpi=220)
    image = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto")
    ax.set_title(f"{domain}: Hit@10 by Retriever and GMM Cluster")
    ax.set_xlabel("GMM cluster")
    ax.set_ylabel("Retriever")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Hit@10")
    fig.tight_layout()

    out_path = out_dir / f"retriever_cluster_heatmap_{domain}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def load_retrieval_hit10_pivot(files: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    summary = json.loads(files["retrieval_summary_file"].read_text(encoding="utf-8"))
    if "retriever_results" in summary:
        retriever_rows = summary["retriever_results"]
    elif "retriever_group_results" in summary:
        retriever_rows = summary["retriever_group_results"]
    else:
        raise ValueError(
            f"{files['retrieval_summary_file']} 缺少 retriever_results 或 retriever_group_results"
        )
    if not isinstance(retriever_rows, list) or not retriever_rows:
        raise ValueError(f"{files['retrieval_summary_file']} 中检索结果列表为空")

    table_rows = []
    meta_rows = []
    for row in retriever_rows:
        retriever = row["retriever"]
        mean_metrics = row["group_mean_metrics"]
        if "hit_at10_gap" in row:
            hit_at10_gap = float(row["hit_at10_gap"])
        elif "hit_at10_max_minus_min" in row:
            hit_at10_gap = float(row["hit_at10_max_minus_min"])
        else:
            raise ValueError(f"{retriever} 缺少 hit_at10_gap / hit_at10_max_minus_min")
        if "hit_at10_kruskal_pvalue" in row:
            hit_at10_pvalue = float(row["hit_at10_kruskal_pvalue"])
        elif "hit_at10_kruskal" in row and "p_value" in row["hit_at10_kruskal"]:
            hit_at10_pvalue = float(row["hit_at10_kruskal"]["p_value"])
        else:
            raise ValueError(f"{retriever} 缺少 hit_at10 p-value")
        meta_rows.append(
            {
                "retriever": retriever,
                "gap": hit_at10_gap,
                "p_value": hit_at10_pvalue,
            }
        )
        for cluster_name, metrics in mean_metrics.items():
            if "hit_at10" not in metrics:
                raise ValueError(f"{retriever} / {cluster_name} 缺少 hit_at10")
            table_rows.append(
                {
                    "retriever": retriever,
                    "cluster": cluster_name,
                    "hit_at10": float(metrics["hit_at10"]),
                }
            )

    table = pd.DataFrame(table_rows)
    pivot = table.pivot(index="retriever", columns="cluster", values="hit_at10")
    ordered_cols = sorted(pivot.columns, key=lambda name: int(str(name).split("_")[-1]))
    pivot = pivot[ordered_cols]
    meta = pd.DataFrame(meta_rows).set_index("retriever").loc[pivot.index]
    cluster_counts = summary.get("cluster_counts")
    if not isinstance(cluster_counts, dict):
        raise ValueError(f"{files['retrieval_summary_file']} 缺少 cluster_counts")
    cluster_count_series = pd.Series(
        {cluster_name: int(cluster_counts[cluster_name]) for cluster_name in ordered_cols},
        index=ordered_cols,
        dtype=np.int64,
    )
    return pivot, meta, cluster_count_series


def draw_review_query_alignment_boxplot(domain: str, files: dict, out_dir: Path) -> None:
    records = read_jsonl(files["review_query_alignment_file"].parent / "review_query_user_style_alignment_user_records.jsonl")
    df = pd.DataFrame(records)
    required_cols = {"cluster_index", "review_user_shared_pca_score", "query_shared_pca_score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"review_query_user_style_alignment_user_records.jsonl 缺少字段 {required_cols - set(df.columns)}")

    ordered_clusters = sorted(df["cluster_index"].unique())
    review_groups = [df.loc[df["cluster_index"] == idx, "review_user_shared_pca_score"].to_numpy(dtype=np.float64) for idx in ordered_clusters]
    query_groups = [df.loc[df["cluster_index"] == idx, "query_shared_pca_score"].to_numpy(dtype=np.float64) for idx in ordered_clusters]
    labels = [f"cluster_{idx}" for idx in ordered_clusters]

    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=220)
    review_box = ax.boxplot(
        review_groups,
        positions=positions - 0.18,
        widths=0.32,
        patch_artist=True,
        showfliers=False,
    )
    query_box = ax.boxplot(
        query_groups,
        positions=positions + 0.18,
        widths=0.32,
        patch_artist=True,
        showfliers=False,
    )

    for patch in review_box["boxes"]:
        patch.set_facecolor("#8ecae6")
    for patch in query_box["boxes"]:
        patch.set_facecolor("#ffb703")

    ax.set_title(f"{domain}: Review vs Query Style by GMM Cluster")
    ax.set_xlabel("GMM cluster")
    ax.set_ylabel("Shared PCA style score")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(
        [review_box["boxes"][0], query_box["boxes"][0]],
        ["Review style", "Query style"],
        loc="upper left",
        frameon=True,
    )
    fig.tight_layout()

    out_path = out_dir / f"review_query_alignment_boxplot_{domain}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def build_cluster_palette(cluster_names: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10")
    return {cluster_name: cmap(i % 10) for i, cluster_name in enumerate(cluster_names)}


def display_cluster_label(cluster_name: str) -> str:
    if not cluster_name.startswith("cluster_"):
        raise ValueError(f"非法 cluster 名称: {cluster_name}")
    return f"cluster{cluster_name.split('_', 1)[1]}"


def draw_retriever_cluster_heatmap_gap_variant(
    domain: str,
    files: dict,
    out_dir: Path,
    cmap_name: str,
    output_name: str,
) -> None:
    pivot, meta, cluster_sizes = load_retrieval_hit10_pivot(files)
    ordered_retrievers = meta.sort_values("gap", ascending=True).index
    pivot = pivot.loc[ordered_retrievers]
    meta = meta.loc[ordered_retrievers]
    display_cluster_labels = [display_cluster_label(name) for name in pivot.columns]

    fig = plt.figure(figsize=(11.8, 5.4), dpi=240)
    gs = fig.add_gridspec(1, 3, width_ratios=[8.8, 2.4, 0.22], wspace=0.08)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[0, 1], sharey=ax_heat)
    ax_cbar = fig.add_subplot(gs[0, 2])

    image = ax_heat.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap=cmap_name, vmin=0.05, vmax=0.60)
    ax_heat.set_xlabel("GMM cluster")
    ax_heat.set_ylabel("Retriever")
    ax_heat.set_xticks(np.arange(len(pivot.columns)))
    ax_heat.set_xticklabels(display_cluster_labels, rotation=0, ha="center", fontstyle="normal")
    ax_heat.set_yticks(np.arange(len(pivot.index)))
    ax_heat.set_yticklabels(pivot.index)
    ax_heat.set_title(f"{domain}: Hit@10 by GMM Cluster with Max-Min Gap", pad=16)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.iloc[i, j])
            text_color = "white" if value >= 0.38 else "black"
            ax_heat.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color=text_color)

    y = np.arange(len(meta.index))
    ax_gap.barh(y, meta["gap"].to_numpy(dtype=np.float64), color="#f59e0b", edgecolor="black", linewidth=0.6)
    for row_idx, gap_value in enumerate(meta["gap"].to_numpy(dtype=np.float64)):
        ax_gap.text(gap_value + 0.003, row_idx, f"{gap_value:.3f}", va="center", fontsize=8)
    ax_gap.set_xlabel("Max-Min gap")
    ax_gap.tick_params(axis="y", left=False, labelleft=False)
    ax_gap.set_xlim(0.0, max(0.20, float(meta["gap"].max()) + 0.01))
    ax_gap.spines["top"].set_visible(False)
    ax_gap.spines["right"].set_visible(False)
    ax_gap.spines["left"].set_visible(True)
    ax_gap.spines["left"].set_linewidth(1.0)

    cbar = fig.colorbar(image, cax=ax_cbar)
    cbar.set_label("Hit@10")
    cbar.set_ticks(np.arange(0.1, 0.61, 0.1))
    fig.tight_layout()

    out_path = out_dir / output_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def draw_baby_design_v1_heatmap_gap(domain: str, files: dict, out_dir: Path) -> None:
    draw_retriever_cluster_heatmap_gap_variant(
        domain=domain,
        files=files,
        out_dir=out_dir,
        cmap_name="YlOrRd",
        output_name="baby_retriever_cluster_design_v1_heatmap_gap.png",
    )


def draw_baby_design_v2_range_dot(domain: str, files: dict, out_dir: Path) -> None:
    pivot, meta, cluster_sizes = load_retrieval_hit10_pivot(files)
    ordered_retrievers = meta.sort_values("gap", ascending=False).index
    pivot = pivot.loc[ordered_retrievers]
    meta = meta.loc[ordered_retrievers]
    palette = build_cluster_palette(list(pivot.columns))
    display_cluster_labels = [display_cluster_label(name) for name in pivot.columns]

    fig = plt.figure(figsize=(11.2, 5.8), dpi=240)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 8.5], hspace=0.18)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])

    cluster_x = np.arange(len(cluster_sizes.index))
    ax_top.bar(cluster_x, cluster_sizes.to_numpy(dtype=np.float64), color=[palette[name] for name in cluster_sizes.index], edgecolor="black", linewidth=0.5)
    for idx, value in enumerate(cluster_sizes.to_numpy(dtype=np.int64)):
        ax_top.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    ax_top.set_xticks(cluster_x)
    ax_top.set_xticklabels([display_cluster_label(name) for name in cluster_sizes.index], rotation=0, ha="center", fontsize=8)
    ax_top.set_ylabel("Size")
    ax_top.set_title(f"{domain}: Range-Dot View of Retriever Variation Across Clusters")

    y_positions = np.arange(len(pivot.index))
    x_max = float(np.max(pivot.to_numpy(dtype=np.float64)))
    x_min = float(np.min(pivot.to_numpy(dtype=np.float64)))
    padding = 0.08
    for row_idx, retriever in enumerate(pivot.index):
        values = pivot.loc[retriever]
        ax_main.hlines(row_idx, float(values.min()), float(values.max()), color="#9ca3af", linewidth=2.2, zorder=1)
        for cluster_name, value in values.items():
            ax_main.scatter(
                float(value),
                row_idx,
                s=75,
                color=palette[cluster_name],
                edgecolor="white",
                linewidth=0.7,
                zorder=2,
            )
        ax_main.text(float(values.max()) + 0.006, row_idx, f"gap={meta.loc[retriever, 'gap']:.3f}", va="center", fontsize=8)

    ax_main.set_xlim(x_min - padding, x_max + 0.18)
    ax_main.set_yticks(y_positions)
    ax_main.set_yticklabels(pivot.index)
    ax_main.invert_yaxis()
    ax_main.set_xlabel("Hit@10")
    ax_main.set_ylabel("Retriever")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[name], markeredgecolor="white", markersize=7, label=name)
        for name in pivot.columns
    ]
    ax_main.legend(handles=legend_handles, title="GMM cluster", ncol=4, fontsize=7, title_fontsize=8, frameon=True, loc="lower right")
    fig.tight_layout()

    out_path = out_dir / f"baby_retriever_cluster_design_v2_range_dot.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def draw_baby_design_v3_small_multiples(domain: str, files: dict, out_dir: Path) -> None:
    pivot, meta, cluster_sizes = load_retrieval_hit10_pivot(files)
    ordered_retrievers = meta.sort_values("gap", ascending=False).index
    pivot = pivot.loc[ordered_retrievers]
    meta = meta.loc[ordered_retrievers]
    display_cluster_labels = [display_cluster_label(name) for name in pivot.columns]

    n_retrievers = len(pivot.index)
    fig = plt.figure(figsize=(11.0, 1.0 + 1.2 * n_retrievers), dpi=240)
    gs = fig.add_gridspec(n_retrievers + 1, 1, height_ratios=[1.0] + [1.0] * n_retrievers, hspace=0.15)
    ax_top = fig.add_subplot(gs[0, 0])

    x = np.arange(len(cluster_sizes.index))
    ax_top.bar(x, cluster_sizes.to_numpy(dtype=np.float64), color="#9ca3af", edgecolor="black", linewidth=0.5)
    for idx, value in enumerate(cluster_sizes.to_numpy(dtype=np.int64)):
        ax_top.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([display_cluster_label(name) for name in cluster_sizes.index], rotation=0, ha="center", fontsize=8)
    ax_top.set_ylabel("Size")
    ax_top.set_title(f"{domain}: Small Multiples of Hit@10 Across GMM Clusters")

    axes = []
    y_min = float(np.min(pivot.to_numpy(dtype=np.float64))) - 0.03
    y_max = float(np.max(pivot.to_numpy(dtype=np.float64))) + 0.03
    for idx, retriever in enumerate(pivot.index, start=1):
        share_ax = axes[0] if axes else None
        ax = fig.add_subplot(gs[idx, 0], sharex=share_ax)
        values = pivot.loc[retriever].to_numpy(dtype=np.float64)
        ax.plot(x, values, color="#374151", linewidth=1.4, marker="o", markersize=4)
        max_pos = int(np.argmax(values))
        min_pos = int(np.argmin(values))
        ax.scatter([max_pos], [values[max_pos]], color="black", s=28, zorder=3)
        ax.scatter([min_pos], [values[min_pos]], facecolors="white", edgecolors="#1d4ed8", s=28, linewidths=1.4, zorder=3)
        ax.text(len(x) - 0.15, values[-1], f"gap={meta.loc[retriever, 'gap']:.3f}", ha="right", va="bottom", fontsize=8)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(retriever, rotation=0, ha="right", va="center", labelpad=34)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        if idx != n_retrievers:
            ax.tick_params(axis="x", labelbottom=False)
        axes.append(ax)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(display_cluster_labels, rotation=0, ha="center")
    axes[-1].set_xlabel("GMM cluster")
    fig.tight_layout()

    out_path = out_dir / f"baby_retriever_cluster_design_v3_small_multiples.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def draw_design_prototypes(domain: str, out_dir: Path) -> None:
    files = load_domain_inputs(domain)
    draw_baby_design_v1_heatmap_gap(domain, files, out_dir)
    draw_baby_design_v2_range_dot(domain, files, out_dir)
    draw_baby_design_v3_small_multiples(domain, files, out_dir)


def draw_green_blue_variants(domain: str, out_dir: Path) -> None:
    files = load_domain_inputs(domain)
    domain_prefix = domain.lower()
    draw_retriever_cluster_heatmap_gap_variant(
        domain=domain,
        files=files,
        out_dir=out_dir,
        cmap_name="viridis",
        output_name=f"{domain_prefix}_retriever_cluster_heatmap_gap_green.png",
    )
    draw_retriever_cluster_heatmap_gap_variant(
        domain=domain,
        files=files,
        out_dir=out_dir,
        cmap_name="Blues",
        output_name=f"{domain_prefix}_retriever_cluster_heatmap_gap_blue.png",
    )


def draw_three_domain_panel(out_dir: Path) -> None:
    domain_specs = [
        ("Baby_Products", "YlOrRd", "A"),
        ("Grocery_and_Gourmet_Food", "viridis", "B"),
        ("Pet_Supplies", "Blues", "C"),
    ]

    prepared = []
    for domain, cmap_name, panel_label in domain_specs:
        files = load_domain_inputs(domain)
        pivot, meta, _ = load_retrieval_hit10_pivot(files)
        ordered_retrievers = meta.sort_values("gap", ascending=True).index
        pivot = pivot.loc[ordered_retrievers]
        meta = meta.loc[ordered_retrievers]
        prepared.append(
            {
                "domain": domain,
                "cmap_name": cmap_name,
                "panel_label": panel_label,
                "pivot": pivot,
                "meta": meta,
                "display_cluster_labels": [display_cluster_label(name) for name in pivot.columns],
            }
        )

    fig = plt.figure(figsize=(12.8, 11.2), dpi=240)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=3,
        width_ratios=[8.8, 2.2, 0.22],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.42,
        wspace=0.08,
    )

    for row_idx, spec in enumerate(prepared):
        ax_heat = fig.add_subplot(gs[row_idx, 0])
        ax_gap = fig.add_subplot(gs[row_idx, 1], sharey=ax_heat)
        ax_cbar = fig.add_subplot(gs[row_idx, 2])

        pivot = spec["pivot"]
        meta = spec["meta"]
        image = ax_heat.imshow(
            pivot.to_numpy(dtype=np.float64),
            aspect="auto",
            cmap=spec["cmap_name"],
            vmin=0.05,
            vmax=0.60,
        )
        ax_heat.set_xticks(np.arange(len(pivot.columns)))
        ax_heat.set_xticklabels(spec["display_cluster_labels"], rotation=0, ha="center", fontsize=8)
        ax_heat.set_yticks(np.arange(len(pivot.index)))
        ax_heat.set_yticklabels(pivot.index, fontsize=8)
        ax_heat.set_xlabel("GMM cluster", fontsize=9)
        ax_heat.set_ylabel("Retriever", fontsize=9)
        ax_heat.set_title(
            f"{spec['panel_label']}  {spec['domain']}: Hit@10 by GMM Cluster with Max-Min Gap",
            loc="left",
            pad=12,
            fontsize=12,
            fontweight="bold",
        )
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = float(pivot.iloc[i, j])
                text_color = "white" if value >= 0.38 else "black"
                ax_heat.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color=text_color)

        y = np.arange(len(meta.index))
        ax_gap.barh(y, meta["gap"].to_numpy(dtype=np.float64), color="#f59e0b", edgecolor="black", linewidth=0.6)
        for row_value_idx, gap_value in enumerate(meta["gap"].to_numpy(dtype=np.float64)):
            ax_gap.text(gap_value + 0.003, row_value_idx, f"{gap_value:.3f}", va="center", fontsize=7)
        ax_gap.set_xlabel("Max-Min gap", fontsize=9)
        ax_gap.tick_params(axis="y", left=False, labelleft=False)
        ax_gap.set_xlim(0.0, max(0.20, float(meta["gap"].max()) + 0.01))
        ax_gap.spines["top"].set_visible(False)
        ax_gap.spines["right"].set_visible(False)
        ax_gap.spines["left"].set_visible(True)
        ax_gap.spines["left"].set_linewidth(1.0)
        ax_gap.set_title("Max-Min Gap", fontsize=9, pad=10)

        cbar = fig.colorbar(image, cax=ax_cbar)
        cbar.set_label("Hit@10", fontsize=9)
        cbar.set_ticks(np.arange(0.1, 0.61, 0.1))

    fig.tight_layout()
    out_path = out_dir / "three_domain_retriever_cluster_heatmap_gap_panel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def draw_cluster_centroid_panel(out_dir: Path) -> None:
    domain_specs = [
        ("Baby_Products", "YlOrRd", "A"),
        ("Grocery_and_Gourmet_Food", "viridis", "B"),
        ("Pet_Supplies", "Blues", "C"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), dpi=240, sharex=False, sharey=False)
    plt.subplots_adjust(top=0.78, bottom=0.18, wspace=0.22)

    global_max_size = 0
    prepared = []
    for domain, cmap_name, panel_label in domain_specs:
        files = load_domain_inputs(domain)
        feature_rows = read_jsonl(files["feature_file"])
        cluster_rows = read_jsonl(files["cluster_user_file"])
        df = build_feature_dataframe(feature_rows, cluster_rows)
        feature_cols = [c for c in df.columns if c not in {"user_id", "asin", "query", "cluster_index"}]
        X = df[feature_cols].to_numpy(dtype=np.float64)
        X_std = StandardScaler().fit_transform(X)
        X_pca = PCA(n_components=2, random_state=42).fit_transform(X_std)
        df["pc1"] = X_pca[:, 0]
        df["pc2"] = X_pca[:, 1]
        grouped = (
            df.groupby("cluster_index", as_index=False)
            .agg(
                pc1_mean=("pc1", "mean"),
                pc2_mean=("pc2", "mean"),
                cluster_size=("cluster_index", "size"),
            )
            .sort_values("cluster_index")
        )
        global_max_size = max(global_max_size, int(grouped["cluster_size"].max()))
        prepared.append((domain, cmap_name, panel_label, grouped))

    for ax, (domain, cmap_name, panel_label, grouped) in zip(axes, prepared, strict=True):
        cmap = plt.get_cmap(cmap_name)
        color_positions = np.linspace(0.35, 0.85, len(grouped))
        colors = [cmap(pos) for pos in color_positions]
        sizes = 80 + 1700 * (grouped["cluster_size"].to_numpy(dtype=np.float64) / float(global_max_size))

        ax.scatter(
            grouped["pc1_mean"],
            grouped["pc2_mean"],
            s=sizes,
            c=colors,
            alpha=0.55,
            edgecolors="#2f2f2f",
            linewidths=0.7,
        )
        for _, row in grouped.iterrows():
            ax.text(
                float(row["pc1_mean"]),
                float(row["pc2_mean"]) + 0.08,
                f"c{int(row['cluster_index'])}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_title(f"{panel_label}  {domain}", loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle("Cluster Distributions in PCA Feature Space Across Domains", fontsize=18, fontweight="bold", y=0.95)
    fig.text(0.5, 0.88, "Bubble size represents cluster size; each point is a GMM cluster centroid", ha="center", fontsize=11)

    legend_sizes = [250, 500, 1000, 2000, 4000]
    handles = []
    labels = []
    for size_value in legend_sizes:
        marker_size = 80 + 1700 * (size_value / float(global_max_size))
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#7f8ff4",
                markeredgecolor="#4b5563",
                markersize=np.sqrt(marker_size) / 1.6,
                alpha=0.6,
            )
        )
        labels.append(f"{size_value:,}")
    fig.legend(
        handles,
        labels,
        title="Cluster Size (number of queries)",
        loc="lower center",
        ncol=len(legend_sizes),
        frameon=False,
        bbox_to_anchor=(0.5, 0.04),
    )

    out_path = out_dir / "three_domain_cluster_centroid_pca_panel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def draw_cluster_centroid_tsne_panel(out_dir: Path) -> None:
    domain_specs = [
        ("Baby_Products", "YlOrRd", "A"),
        ("Grocery_and_Gourmet_Food", "viridis", "B"),
        ("Pet_Supplies", "Blues", "C"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), dpi=240, sharex=False, sharey=False)
    plt.subplots_adjust(top=0.78, bottom=0.18, wspace=0.22)

    global_max_size = 0
    prepared = []
    for domain, cmap_name, panel_label in domain_specs:
        files = load_domain_inputs(domain)
        feature_rows = read_jsonl(files["feature_file"])
        cluster_rows = read_jsonl(files["cluster_user_file"])
        df = build_feature_dataframe(feature_rows, cluster_rows)
        feature_cols = [c for c in df.columns if c not in {"user_id", "asin", "query", "cluster_index"}]
        X = df[feature_cols].to_numpy(dtype=np.float64)
        X_std = StandardScaler().fit_transform(X)
        perplexity = min(30, max(5, len(df) // 50))
        X_tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        ).fit_transform(X_std)
        df["tsne1"] = X_tsne[:, 0]
        df["tsne2"] = X_tsne[:, 1]
        grouped = (
            df.groupby("cluster_index", as_index=False)
            .agg(
                tsne1_mean=("tsne1", "mean"),
                tsne2_mean=("tsne2", "mean"),
                cluster_size=("cluster_index", "size"),
            )
            .sort_values("cluster_index")
        )
        global_max_size = max(global_max_size, int(grouped["cluster_size"].max()))
        prepared.append((domain, cmap_name, panel_label, grouped))

    for ax, (domain, cmap_name, panel_label, grouped) in zip(axes, prepared, strict=True):
        cmap = plt.get_cmap(cmap_name)
        color_positions = np.linspace(0.35, 0.85, len(grouped))
        colors = [cmap(pos) for pos in color_positions]
        sizes = 80 + 1700 * (grouped["cluster_size"].to_numpy(dtype=np.float64) / float(global_max_size))

        ax.scatter(
            grouped["tsne1_mean"],
            grouped["tsne2_mean"],
            s=sizes,
            c=colors,
            alpha=0.55,
            edgecolors="#2f2f2f",
            linewidths=0.7,
        )
        for _, row in grouped.iterrows():
            ax.text(
                float(row["tsne1_mean"]),
                float(row["tsne2_mean"]) + 0.08,
                f"c{int(row['cluster_index'])}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_title(f"{panel_label}  {domain}", loc="left", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    fig.suptitle("Cluster Distributions in t-SNE Feature Space Across Domains", fontsize=18, fontweight="bold", y=0.95)
    fig.text(0.5, 0.88, "Bubble size represents cluster size; each point is the mean t-SNE position of a GMM cluster", ha="center", fontsize=11)

    legend_sizes = [250, 500, 1000, 2000, 4000]
    handles = []
    labels = []
    for size_value in legend_sizes:
        marker_size = 80 + 1700 * (size_value / float(global_max_size))
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#7f8ff4",
                markeredgecolor="#4b5563",
                markersize=np.sqrt(marker_size) / 1.6,
                alpha=0.6,
            )
        )
        labels.append(f"{size_value:,}")
    fig.legend(
        handles,
        labels,
        title="Cluster Size (number of queries)",
        loc="lower center",
        ncol=len(legend_sizes),
        frameon=False,
        bbox_to_anchor=(0.5, 0.04),
    )

    out_path = out_dir / "three_domain_cluster_centroid_tsne_panel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def draw_all(domain: str, out_dir: Path) -> None:
    files = load_domain_inputs(domain)
    draw_query_style_map(domain, files, out_dir)
    draw_retriever_cluster_heatmap(domain, files, out_dir)
    draw_review_query_alignment_boxplot(domain, files, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies", "all"], required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "figures" / "pqb_style"))
    parser.add_argument("--design-prototypes", action="store_true")
    parser.add_argument("--green-blue-variants", action="store_true")
    parser.add_argument("--three-domain-panel", action="store_true")
    parser.add_argument("--three-domain-centroid-panel", action="store_true")
    parser.add_argument("--three-domain-centroid-tsne-panel", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.design_prototypes:
        if args.domain == "all":
            raise ValueError("--design-prototypes 只支持单个域")
        draw_design_prototypes(args.domain, out_dir)
        return

    if args.green_blue_variants:
        if args.domain == "all":
            raise ValueError("--green-blue-variants 只支持单个域")
        draw_green_blue_variants(args.domain, out_dir)
        return

    if args.three_domain_panel:
        if args.domain != "all":
            raise ValueError("--three-domain-panel 需要 --domain all")
        draw_three_domain_panel(out_dir)
        return

    if args.three_domain_centroid_panel:
        if args.domain != "all":
            raise ValueError("--three-domain-centroid-panel 需要 --domain all")
        draw_cluster_centroid_panel(out_dir)
        return

    if args.three_domain_centroid_tsne_panel:
        if args.domain != "all":
            raise ValueError("--three-domain-centroid-tsne-panel 需要 --domain all")
        draw_cluster_centroid_tsne_panel(out_dir)
        return

    domains = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"] if args.domain == "all" else [args.domain]
    for domain in domains:
        draw_all(domain, out_dir)


if __name__ == "__main__":
    main()
