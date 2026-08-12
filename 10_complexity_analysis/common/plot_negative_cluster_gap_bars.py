#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path("/fs04/ar57/wenyu")
OUTPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis"
OUTPUT_PNG = OUTPUT_DIR / "retriever_negative_cluster_gap_domain_mean_bar.png"
OUTPUT_PNG_HORIZONTAL = OUTPUT_DIR / "retriever_negative_cluster_gap_domain_mean_bar_horizontal.png"
OUTPUT_JSON = OUTPUT_DIR / "retriever_negative_cluster_gap_domain_mean_bar.json"

CATEGORY_FILES = {
    "Baby_Products": {
        "eval": REPO_ROOT
        / "result"
        / "personal_query"
        / "09_noisy_retrieval"
        / "Baby_Products"
        / "syntax_depth_correct_vs_noisy_results.json",
        "user_profiles": REPO_ROOT
        / "result"
        / "personal_query"
        / "12_complexity_analysis_clause_features"
        / "Baby_Products"
        / "strict5550_query_gmm_user_profiles.jsonl",
    },
    "Grocery_and_Gourmet_Food": {
        "eval": REPO_ROOT
        / "result"
        / "personal_query"
        / "09_noisy_retrieval"
        / "Grocery_and_Gourmet_Food"
        / "syntax_depth_correct_vs_noisy_results.json",
        "user_profiles": REPO_ROOT
        / "result"
        / "personal_query"
        / "12_complexity_analysis_clause_features"
        / "Grocery_and_Gourmet_Food"
        / "strict5550_query_gmm_user_profiles.jsonl",
    },
    "Pet_Supplies": {
        "eval": REPO_ROOT
        / "result"
        / "personal_query"
        / "09_noisy_retrieval"
        / "Pet_Supplies"
        / "syntax_depth_correct_vs_noisy_results.json",
        "user_profiles": REPO_ROOT
        / "result"
        / "personal_query"
        / "12_complexity_analysis_clause_features"
        / "Pet_Supplies"
        / "strict5550_query_gmm_user_profiles.jsonl",
    },
}

RETRIEVER_ORDER = [
    "ance",
    "bge",
    "bm25",
    "colbertv2",
    "e5",
    "minilm",
    "splade",
    "star",
]

CATEGORY_ORDER = [
    "Baby_Products",
    "Grocery_and_Gourmet_Food",
    "Pet_Supplies",
]

CATEGORY_LABELS = {
    "Baby_Products": "Baby",
    "Grocery_and_Gourmet_Food": "Grocery",
    "Pet_Supplies": "Pet",
}

CATEGORY_COLORS = {
    "Baby_Products": "#c4573a",
    "Grocery_and_Gourmet_Food": "#2d6a4f",
    "Pet_Supplies": "#3a6ea5",
}


def load_cluster_by_user(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user_id = row.get("user_id")
            cluster_index = row.get("cluster_index")
            if user_id is None or cluster_index is None:
                raise ValueError(f"{path} 缺少 user_id 或 cluster_index")
            mapping[user_id] = int(cluster_index)
    if not mapping:
        raise ValueError(f"{path} 未读取到 cluster 映射")
    return mapping


def compute_negative_cluster_gaps(
    eval_path: Path,
    cluster_by_user: dict[str, int],
) -> dict[str, dict]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    correct_blocks = {block["retriever"]: block for block in data["raw_correct_results"]}
    noisy_blocks = {block["retriever"]: block for block in data["raw_noisy_results"]}
    results: dict[str, dict] = {}

    for retriever in RETRIEVER_ORDER:
        correct_block = correct_blocks.get(retriever)
        noisy_block = noisy_blocks.get(retriever)
        if correct_block is None or noisy_block is None:
            raise ValueError(f"{eval_path} 缺少 retriever={retriever} 的 raw 结果")

        correct_records = {row["pair_id"]: row for row in correct_block["all_query_records"]}
        noisy_records = {row["pair_id"]: row for row in noisy_block["all_query_records"]}
        shared_pair_ids = sorted(set(correct_records.keys()) & set(noisy_records.keys()))

        cluster_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"clean": [], "noisy": []})
        dropped_noisy_gt_clean = 0

        for pair_id in shared_pair_ids:
            correct_row = correct_records[pair_id]
            noisy_row = noisy_records[pair_id]
            clean_hit = float(correct_row["metrics"]["H@10"])
            noisy_hit = float(noisy_row["metrics"]["H@10"])
            if noisy_hit > clean_hit:
                dropped_noisy_gt_clean += 1
                continue

            user_id = correct_row.get("user_id")
            if user_id is None:
                raise ValueError(f"{eval_path} pair_id={pair_id} 缺少 user_id")
            cluster_index = cluster_by_user.get(user_id)
            if cluster_index is None:
                raise ValueError(f"{eval_path} user_id={user_id} 缺少 cluster 映射")
            cluster_name = f"cluster_{cluster_index}"
            cluster_metrics[cluster_name]["clean"].append(clean_hit)
            cluster_metrics[cluster_name]["noisy"].append(noisy_hit)

        cluster_deltas: dict[str, float] = {}
        for cluster_name, values in cluster_metrics.items():
            if not values["clean"]:
                raise ValueError(f"{eval_path} retriever={retriever} cluster={cluster_name} 没有 clean 样本")
            clean_mean = float(np.mean(values["clean"]))
            noisy_mean = float(np.mean(values["noisy"]))
            delta = noisy_mean - clean_mean
            if delta < 0:
                cluster_deltas[cluster_name] = delta

        if not cluster_deltas:
            raise ValueError(f"{eval_path} retriever={retriever} 没有下降 cluster")

        lightest_cluster = max(cluster_deltas, key=cluster_deltas.get)
        worst_cluster = min(cluster_deltas, key=cluster_deltas.get)
        gap = cluster_deltas[lightest_cluster] - cluster_deltas[worst_cluster]
        results[retriever] = {
            "negative_cluster_count": len(cluster_deltas),
            "dropped_noisy_gt_clean": dropped_noisy_gt_clean,
            "lightest_cluster": lightest_cluster,
            "lightest_delta": cluster_deltas[lightest_cluster],
            "worst_cluster": worst_cluster,
            "worst_delta": cluster_deltas[worst_cluster],
            "gap": gap,
        }

    return results


def build_summary() -> dict[str, dict[str, dict]]:
    summary: dict[str, dict[str, dict]] = {}
    for category in CATEGORY_ORDER:
        files = CATEGORY_FILES[category]
        cluster_by_user = load_cluster_by_user(files["user_profiles"])
        summary[category] = compute_negative_cluster_gaps(files["eval"], cluster_by_user)
    return summary


def compute_domain_mean_gaps(summary: dict[str, dict[str, dict]]) -> dict[str, float]:
    mean_gaps: dict[str, float] = {}
    for retriever in RETRIEVER_ORDER:
        values = [summary[category][retriever]["gap"] for category in CATEGORY_ORDER]
        mean_gaps[retriever] = float(np.mean(values))
    return mean_gaps


def compute_display_decline_values(summary: dict[str, dict[str, dict]]) -> dict[str, float]:
    mean_gaps = compute_domain_mean_gaps(summary)
    return {retriever: -value for retriever, value in mean_gaps.items()}


def apply_axis_background(ax: plt.Axes) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = 320
    height = 320
    x = np.linspace(0.0, 1.0, width)
    y = np.linspace(0.0, 1.0, height)
    xx, yy = np.meshgrid(x, y)

    top_left = np.array([0.86, 0.90, 0.96])
    bottom_right = np.array([0.97, 0.97, 0.95])
    blend = 0.58 * yy + 0.42 * xx
    gradient = (1.0 - blend)[..., None] * top_left + blend[..., None] * bottom_right

    highlight_center = np.array([0.38, 0.28])
    distance = np.sqrt((xx - highlight_center[0]) ** 2 + (yy - highlight_center[1]) ** 2)
    highlight = np.clip(1.0 - distance / 0.9, 0.0, 1.0)[..., None]
    gradient = np.clip(gradient + 0.035 * highlight, 0.0, 1.0)

    ax.imshow(
        gradient,
        extent=[x0, x1, y0, y1],
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
    )
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color="white", linewidth=1.0, alpha=0.72)


def save_plot(summary: dict[str, dict[str, dict]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    display_values = compute_display_decline_values(summary)
    x = np.arange(len(RETRIEVER_ORDER))

    plt.rcParams["font.family"] = "STIXGeneral"
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    fig.patch.set_facecolor("#f6f2eb")
    values = [display_values[retriever] for retriever in RETRIEVER_ORDER]
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color="#3d6b99",
        edgecolor="#1f1f1f",
        linewidth=0.8,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() - 0.0015,
            f"{value * 100:.1f}%",
            ha="center",
            va="top",
            fontsize=9,
        )

    ax.set_title("Mean Negative-Cluster Gap Across Three Domains", fontsize=16, pad=14)
    ax.set_xlabel("Retriever", fontsize=12)
    ax.set_ylabel("Mean decline shown from cluster-gap (H@10, %)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(RETRIEVER_ORDER, rotation=0, fontsize=11)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.6, len(RETRIEVER_ORDER) - 0.4)
    ax.set_ylim(min(values) - 0.02, 0.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f}%"))
    apply_axis_background(ax)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_horizontal_plot(summary: dict[str, dict[str, dict]]) -> None:
    display_values = compute_display_decline_values(summary)
    y = np.arange(len(RETRIEVER_ORDER))
    values = [abs(display_values[retriever]) for retriever in RETRIEVER_ORDER]

    plt.rcParams["font.family"] = "STIXGeneral"
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    fig.patch.set_facecolor("#f6f2eb")
    bars = ax.barh(
        y,
        values,
        height=0.62,
        color="#3d6b99",
        edgecolor="#1f1f1f",
        linewidth=0.8,
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + 0.0015,
            bar.get_y() + bar.get_height() / 2.0,
            f"-{value * 100:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_title("Mean Negative-Cluster Gap Across Three Domains", fontsize=16, pad=14)
    ax.set_xlabel("Mean decline shown from cluster-gap (H@10, %)", fontsize=12)
    ax.set_ylabel("Retriever", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(RETRIEVER_ORDER, fontsize=11)
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis="y", which="both", labelright=False, labelleft=True, right=False, left=False, pad=10)
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, max(values) + 0.02)
    ax.set_ylim(-0.7, len(RETRIEVER_ORDER) - 0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"-{x * 100:.1f}%"))
    apply_axis_background(ax)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG_HORIZONTAL, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_json(summary: dict[str, dict[str, dict]]) -> None:
    payload = {
        "per_domain": summary,
        "domain_mean_gap": compute_domain_mean_gaps(summary),
        "domain_mean_gap_display_negative": compute_display_decline_values(summary),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    summary = build_summary()
    save_json(summary)
    save_plot(summary)
    save_horizontal_plot(summary)
    print(f"已写入: {OUTPUT_JSON}")
    print(f"已写入: {OUTPUT_PNG}")
    print(f"已写入: {OUTPUT_PNG_HORIZONTAL}")


if __name__ == "__main__":
    main()
