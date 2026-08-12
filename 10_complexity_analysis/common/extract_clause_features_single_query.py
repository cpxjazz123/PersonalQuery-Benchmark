#!/usr/bin/env python3
"""Extract clause-related syntactic features for a single query."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import spacy


REPO_ROOT = Path("/fs04/ar57/wenyu")

RELATIVE_MARKERS = {
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
}

SUBORDINATE_DEP_TYPES = {
    "acl",
    "relcl",
    "ccomp",
    "xcomp",
    "advcl",
}

NOUNISH_POS = {
    "NOUN",
    "PROPN",
    "PRON",
}


@lru_cache(maxsize=1)
def load_spacy_model():
    return spacy.load("en_core_web_sm")


def build_child_map(doc):
    child_map = {token.i: [] for token in doc}
    for token in doc:
        if token.head.i != token.i:
            child_map[token.head.i].append(token.i)
    return child_map


def compute_depths(doc, child_map):
    roots = [token for token in doc if token.head.i == token.i]
    if not roots:
        raise ValueError("spaCy parse produced no root")

    depths = {}
    stack = [(root.i, 1) for root in roots]
    while stack:
        node_i, depth = stack.pop()
        depths[node_i] = depth
        for child_i in child_map[node_i]:
            stack.append((child_i, depth + 1))
    return depths


def compute_tree_height(node_i, child_map):
    if not child_map[node_i]:
        return 1
    return 1 + max(compute_tree_height(child_i, child_map) for child_i in child_map[node_i])


def compute_clause_nesting(doc, child_map):
    roots = [token for token in doc if token.head.i == token.i]
    max_nesting = 0
    stack = [(root.i, 0) for root in roots]

    while stack:
        node_i, current_nesting = stack.pop()
        token = doc[node_i]
        next_nesting = current_nesting + 1 if token.dep_ in SUBORDINATE_DEP_TYPES else current_nesting
        max_nesting = max(max_nesting, next_nesting)
        for child_i in child_map[node_i]:
            stack.append((child_i, next_nesting))
    return max_nesting


def compute_dependency_distances(doc):
    distances = []
    long_distance_count = 0
    for token in doc:
        if token.head.i == token.i:
            continue
        distance = abs(token.i - token.head.i)
        distances.append(distance)
        if distance >= 5:
            long_distance_count += 1
    if not distances:
        return 0.0, 0, 0.0
    return (
        sum(distances) / len(distances),
        max(distances),
        long_distance_count / len(distances),
    )


def count_dep_types(doc, dep_type):
    return sum(1 for token in doc if token.dep_ == dep_type)


def count_compounds(doc):
    return sum(1 for token in doc if token.dep_ == "compound")


def count_modifiers(doc):
    modifier_deps = {"amod", "advmod", "nmod", "compound"}
    return sum(1 for token in doc if token.dep_ in modifier_deps)


def count_coordination(doc):
    return sum(1 for token in doc if token.dep_ == "conj" or token.dep_ == "cc")


def compute_max_branching(child_map):
    if not child_map:
        return 0
    return max(len(children) for children in child_map.values())


def extract_clause_features_from_doc(doc, query_text: str) -> dict:
    if not query_text.strip():
        raise ValueError("query text is empty")

    tokens = [token for token in doc if not token.is_space]
    if not tokens:
        raise ValueError("query has no non-space tokens")

    child_map = build_child_map(doc)
    depths = compute_depths(doc, child_map)
    depth_values = [depths[token.i] for token in doc]

    tree_heights = [
        compute_tree_height(root.i, child_map)
        for root in doc
        if root.head.i == root.i
    ]
    dependency_tree_height = max(tree_heights)

    max_dependency_depth = max(depth_values)
    mean_dependency_depth = sum(depth_values) / len(depth_values)
    depth_variance = sum((depth - mean_dependency_depth) ** 2 for depth in depth_values) / len(depth_values)

    clause_nesting_depth = compute_clause_nesting(doc, child_map)
    mean_dependency_distance, max_dependency_distance, long_dependency_ratio = compute_dependency_distances(doc)

    amod_count = count_dep_types(doc, "amod")
    advmod_count = count_dep_types(doc, "advmod")
    nmod_count = count_dep_types(doc, "nmod")
    compound_count = count_compounds(doc)
    modifier_density = count_modifiers(doc) / len(tokens)
    coordination_count = count_coordination(doc)
    max_branching_factor = compute_max_branching(child_map)

    return {
        "max_dependency_depth": max_dependency_depth,
        "mean_dependency_depth": mean_dependency_depth,
        "dependency_tree_height": dependency_tree_height,
        "depth_variance": depth_variance,
        "acl_count": count_dep_types(doc, "acl"),
        "relcl_count": count_dep_types(doc, "relcl"),
        "ccomp_count": count_dep_types(doc, "ccomp"),
        "xcomp_count": count_dep_types(doc, "xcomp"),
        "advcl_count": count_dep_types(doc, "advcl"),
        "clause_nesting_depth": clause_nesting_depth,
        "mean_dependency_distance": mean_dependency_distance,
        "max_dependency_distance": max_dependency_distance,
        "long_dependency_ratio": long_dependency_ratio,
        "amod_count": amod_count,
        "advmod_count": advmod_count,
        "nmod_count": nmod_count,
        "compound_count": compound_count,
        "modifier_density": modifier_density,
        "coordination_count": coordination_count,
        "max_branching_factor": max_branching_factor,
    }


def extract_clause_features(query: str) -> dict:
    nlp = load_spacy_model()
    doc = nlp(query)
    return extract_clause_features_from_doc(doc, query)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = extract_clause_features(args.query)
    text = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(text, encoding="utf-8")
        print(f"Wrote features to {output_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
