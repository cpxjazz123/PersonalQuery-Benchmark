#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Node:
    x: float
    level: float
    text: str
    fontsize: float = 13.0
    color: str = "black"
    weight: str = "normal"
    ha: str = "center"


@dataclass(frozen=True)
class Edge:
    x1: float
    l1: float
    x2: float
    l2: float
    color: str = "black"
    rad: float = 0.0


@dataclass(frozen=True)
class RichText:
    x: float
    level: float
    parts: tuple[tuple[str, str, str], ...]
    fontsize: float = 12.4
    ha: str = "left"


FIG_XMAX = 270.0
LEVEL_BOTTOM = 30.0
WIDE_SENTENCE = (
    "I want Chobani Greek yogurt, which costs 5.99, which is lactose-free, "
    "and which has a strawberry flavour."
)
DEEP_SENTENCE = (
    "I want Chobani Greek yogurt that has a strawberry flavour that comes in a "
    "lactose-free option that costs 5.99."
)


@lru_cache(maxsize=1)
def _load_spacy_model():
    """加载句法分析模型；只做一次缓存。"""
    import spacy

    return spacy.load("en_core_web_sm")


def _token_dependency_depth(token) -> int:
    """计算单个 token 到依存树根节点的深度，根节点深度记为 1。"""
    if token.head == token:
        return 1
    return _token_dependency_depth(token.head) + 1


def compute_sentence_syntax_tree_depth(sentence: str) -> int:
    """
    计算一句话的依存句法树深度。

    定义：
    - 使用 spaCy 的依存句法分析结果；
    - 根节点深度记为 1；
    - 句子深度等于所有非空白、非标点 token 的最大依存深度。
    """
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string")

    sentence = sentence.strip()
    if not sentence:
        raise ValueError("sentence must be a non-empty string")

    nlp = _load_spacy_model()
    doc = nlp(sentence)
    if len(doc) == 0:
        raise ValueError("sentence produced an empty document")

    depths = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        depths.append(_token_dependency_depth(token))

    if not depths:
        raise ValueError("sentence contains no valid tokens for depth computation")

    return max(depths)


def y(level: float) -> float:
    return LEVEL_BOTTOM - level


def _iter_visible_tokens(doc):
    return [token for token in doc if not token.is_space and not token.is_punct]


def _token_label(token) -> str:
    return f"({token.text})\n{token.dep_}"


def _token_color(token) -> str:
    blue = "#2457ff"
    green = "#14a83c"
    orange = "#ff7b19"
    purple = "#7b30c5"

    if token.text in {"Chobani", "Greek", "yogurt"}:
        return blue
    if token.text == "5.99":
        return green
    if token.text in {"lactose", "free", "option"}:
        return orange
    if token.text in {"strawberry", "flavour"}:
        return purple
    return "black"


def _token_fontsize(token) -> float:
    if token.dep_ == "ROOT":
        return 11.8
    if len(token.text) >= 10 or len(token.dep_) >= 8:
        return 8.1
    return 8.8


def _edge_color(token) -> str:
    if token.dep_ == "relcl":
        return "#b01919"
    return "black"


def _edge_rad(parent_x: float, child_x: float) -> float:
    delta = child_x - parent_x
    if delta == 0:
        return 0.0
    return max(min(delta / 220.0, 0.22), -0.22)


def _build_sentence_panel(sentence: str, x_start: float, x_end: float) -> tuple[list[Node], list[Edge], int]:
    nlp = _load_spacy_model()
    doc = nlp(sentence)
    tokens = _iter_visible_tokens(doc)
    if not tokens:
        raise ValueError("sentence contains no visible tokens")

    if len(tokens) == 1:
        x_positions = {tokens[0].i: (x_start + x_end) / 2.0}
    else:
        step = (x_end - x_start) / (len(tokens) - 1)
        x_positions = {token.i: x_start + idx * step for idx, token in enumerate(tokens)}

    visible_ids = {token.i for token in tokens}
    nodes: list[Node] = []
    edges: list[Edge] = []

    for token in tokens:
        depth = _token_dependency_depth(token)
        nodes.append(
            Node(
                x=x_positions[token.i],
                level=depth,
                text=_token_label(token),
                fontsize=_token_fontsize(token),
                color=_token_color(token),
                weight="bold",
            )
        )

        if token.head == token:
            continue

        if token.head.is_space or token.head.is_punct:
            raise ValueError(f"Unexpected head for token {token.text}: {token.head.text}")
        if token.head.i not in visible_ids:
            raise ValueError(f"Head token not visible for token {token.text}: {token.head.text}")

        parent_depth = _token_dependency_depth(token.head)
        parent_x = x_positions[token.head.i]
        child_x = x_positions[token.i]
        edges.append(
            Edge(
                x1=parent_x,
                l1=parent_depth,
                x2=child_x,
                l2=depth,
                color=_edge_color(token),
                rad=_edge_rad(parent_x, child_x),
            )
        )

    panel_depth = max(node.level for node in nodes)
    if panel_depth != compute_sentence_syntax_tree_depth(sentence):
        raise ValueError(
            f"Panel depth mismatch for sentence: computed={compute_sentence_syntax_tree_depth(sentence)}, "
            f"panel={panel_depth}"
        )

    return nodes, edges, panel_depth


def draw_guides(ax, max_level: int) -> None:
    for level in range(1, max_level + 1):
        ax.plot(
            [8.0, FIG_XMAX - 2.0],
            [y(level), y(level)],
            color="#a8a8a8",
            lw=0.85,
            linestyle=(0, (4, 3)),
            zorder=0,
        )

    ax.annotate(
        "",
        xy=(10.0, y(0.25)),
        xytext=(10.0, y(max_level + 0.25)),
        arrowprops=dict(arrowstyle="-|>,head_width=0.65,head_length=1.0", lw=2.0, color="black"),
    )
    ax.text(0.8, LEVEL_BOTTOM + 1.2, "Level", fontsize=20, family="serif", fontweight="bold", ha="left")

    for level in range(1, max_level + 1):
        ax.text(5.6, y(level), str(level), fontsize=14.5, family="serif", fontweight="bold", ha="center")


def draw_panel_title(ax, x: float, title: str) -> None:
    ax.text(
        x,
        y(-1.6),
        title,
        ha="center",
        va="center",
        fontsize=18.0,
        family="serif",
        fontweight="bold",
        zorder=5,
    )


def draw_node(ax, node: Node) -> None:
    ax.text(
        node.x,
        y(node.level),
        node.text,
        ha=node.ha,
        va="center",
        fontsize=node.fontsize,
        family="serif",
        color=node.color,
        fontweight=node.weight,
        linespacing=0.88,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.18, "alpha": 0.96},
        zorder=5,
    )


def draw_edge(ax, edge: Edge) -> None:
    ax.annotate(
        "",
        xy=(edge.x2, y(edge.l2) + 0.36),
        xytext=(edge.x1, y(edge.l1) - 0.12),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.45,
            color=edge.color,
            shrinkA=5,
            shrinkB=5,
            connectionstyle=f"arc3,rad={edge.rad}",
        ),
        zorder=2,
    )


def draw_rich_text(ax, item: RichText) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_disp, y_disp = ax.transData.transform((item.x, y(item.level)))
    current_x = x_disp

    for text, color, weight in item.parts:
        artist = ax.text(
            0,
            0,
            text,
            ha="left",
            va="center",
            fontsize=item.fontsize,
            family="serif",
            color=color,
            fontweight=weight,
            zorder=5,
        )
        bbox = artist.get_window_extent(renderer=renderer)
        data_x, data_y = ax.transData.inverted().transform((current_x, y_disp))
        artist.set_position((data_x, data_y))
        current_x += bbox.width


def wide_content() -> tuple[list[Node], list[Edge], list[RichText]]:
    blue = "#2457ff"
    green = "#14a83c"
    orange = "#ff7b19"
    purple = "#7b30c5"
    nodes, edges, _ = _build_sentence_panel(WIDE_SENTENCE, 24.0, 122.0)

    texts = [
        RichText(
            25,
            -0.75,
            (
                ("I want ", "black", "normal"),
                ("Chobani Greek yogurt", blue, "bold"),
                (", which costs ", "black", "normal"),
                ("5.99", green, "bold"),
                (",", "black", "normal"),
            ),
        ),
        RichText(
            25,
            -0.10,
            (
                ("which is ", "black", "normal"),
                ("lactose-free", orange, "bold"),
                (", and which has a ", "black", "normal"),
                ("strawberry flavour", purple, "bold"),
                (".", "black", "normal"),
            ),
        ),
    ]

    return nodes, edges, texts


def deep_content() -> tuple[list[Node], list[Edge], list[RichText]]:
    blue = "#2457ff"
    green = "#14a83c"
    orange = "#ff7b19"
    purple = "#7b30c5"
    nodes, edges, _ = _build_sentence_panel(DEEP_SENTENCE, 154.0, 262.0)

    texts = [
        RichText(
            158,
            -0.75,
            (
                ("I want ", "black", "normal"),
                ("Chobani Greek yogurt", blue, "bold"),
                (" that has a ", "black", "normal"),
                ("strawberry", purple, "bold"),
            ),
        ),
        RichText(
            158,
            -0.10,
            (
                ("flavour", purple, "bold"),
                (" that comes in a ", "black", "normal"),
                ("lactose-free", orange, "bold"),
                (" option that costs ", "black", "normal"),
                ("5.99", green, "bold"),
                (".", "black", "normal"),
            ),
        ),
    ]

    return nodes, edges, texts


def main() -> None:
    global LEVEL_BOTTOM
    repo_root = Path(__file__).resolve().parent
    output_path = repo_root / "generated_figures" / "wide_deep_expression_shared_axis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wide_nodes, wide_edges, wide_texts = wide_content()
    deep_nodes, deep_edges, deep_texts = deep_content()
    max_depth = int(max([node.level for node in wide_nodes + deep_nodes]))
    LEVEL_BOTTOM = float(max_depth + 1)

    fig, ax = plt.subplots(figsize=(19.5, 14.5), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, FIG_XMAX)
    ax.set_ylim(0.0, LEVEL_BOTTOM + 3.5)
    ax.axis("off")

    draw_guides(ax, max_depth)
    draw_panel_title(ax, 73, "(1) Wide expression")
    draw_panel_title(ax, 208, "(2) Deep expression")

    for edge in wide_edges + deep_edges:
        draw_edge(ax, edge)

    for item in wide_texts + deep_texts:
        draw_rich_text(ax, item)

    for node in wide_nodes + deep_nodes:
        draw_node(ax, node)

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
