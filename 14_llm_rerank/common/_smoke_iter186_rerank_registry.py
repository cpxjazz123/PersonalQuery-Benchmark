#!/usr/bin/env python3
"""Smoke test for iter #186 DeepSeek-v4 rerank registry plumbing.

Tests:
  1. Registry membership: DENSE/SPARSE/COLBERTV2 + paper coverage (via AST parse)
  2. AVAILABLE_RERANKERS + DEFAULT_RERANKER
  3. validate_rerank_config happy path
  4. validate_rerank_config failure: unregistered first-stage retriever
  5. validate_rerank_config warning: single first-stage retriever
  6. rerank_config.json loaded correctly (matches JSON parse)

Uses AST/source-level parsing to avoid heavy deps (torch, transformers).
"""

import ast
import json
import sys
from pathlib import Path

_SCRIPT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/14_llm_rerank/common/llm_rerank_common.py")
_CONFIG = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/14_llm_rerank/common/rerank_config.json")


def _eval_constant(node, env: dict):
    """Evaluate set/binop/literal at AST level. BinOp leaves may be Names (variables)."""
    if isinstance(node, ast.Set):
        out = set()
        for e in node.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                out.add(e.value)
            else:
                return None
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        l = _eval_constant(node.left, env)
        r = _eval_constant(node.right, env)
        if l is None or r is None:
            return None
        return l | r
    if isinstance(node, ast.List):
        out = []
        for e in node.elts:
            if isinstance(e, ast.Constant):
                out.append(e.value)
            else:
                return None
        return out
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    return None


def _extract_constants(source: str) -> dict:
    """Extract module-level constants, resolving DENSE | SPARSE | COLBERT chain via env."""
    tree = ast.parse(source)
    env: dict = {}
    targets_set = {
        "DENSE_RETRIEVERS", "SPARSE_RETRIEVERS", "COLBERTV2_RETRIEVERS",
        "ALL_PAPER_FIRST_STAGE_RETRIEVERS", "AVAILABLE_RERANKERS",
        "DEFAULT_RERANKER", "DEFAULT_FIRST_STAGE_RETRIEVERS",
    }
    found: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in targets_set:
                    val = _eval_constant(node.value, env)
                    if val is not None:
                        env[tgt.id] = val
                        found[tgt.id] = val
        elif isinstance(node, ast.FunctionDef) and node.name == "validate_rerank_config":
            found["_validate_rerank_config_src"] = ast.unparse(node)
    return found


def main():
    print("=" * 60)
    print("iter #186 rerank registry + validate_rerank_config smoke test")
    print("=" * 60)

    source = _SCRIPT.read_text(encoding="utf-8")
    constants = _extract_constants(source)
    assert "DENSE_RETRIEVERS" in constants
    assert "AVAILABLE_RERANKERS" in constants

    # Case 1: registry membership
    print(f"\n[Case 1] Registry membership")
    print(f"  DENSE_RETRIEVERS = {sorted(constants['DENSE_RETRIEVERS'])}")
    print(f"  SPARSE_RETRIEVERS = {sorted(constants['SPARSE_RETRIEVERS'])}")
    print(f"  COLBERTV2_RETRIEVERS = {sorted(constants['COLBERTV2_RETRIEVERS'])}")
    print(f"  ALL_PAPER_FIRST_STAGE_RETRIEVERS = {sorted(constants['ALL_PAPER_FIRST_STAGE_RETRIEVERS'])}")
    print(f"  AVAILABLE_RERANKERS = {sorted(constants['AVAILABLE_RERANKERS'])}")
    print(f"  DEFAULT_RERANKER = {constants['DEFAULT_RERANKER']}")
    print(f"  DEFAULT_FIRST_STAGE_RETRIEVERS = {constants['DEFAULT_FIRST_STAGE_RETRIEVERS']}")
    paper_first_stage = {"bm25", "splade", "bge", "e5", "minilm", "star", "ance", "colbertv2"}
    actual = set(constants["ALL_PAPER_FIRST_STAGE_RETRIEVERS"])
    missing = paper_first_stage - actual
    extra = actual - paper_first_stage
    print(f"  Paper-claimed first-stage coverage: {sorted(paper_first_stage)}")
    print(f"  Missing from registry: {missing}")
    print(f"  Extra in registry: {extra}")
    assert not missing, f"registry missing paper-claimed retrievers: {missing}"
    assert "M2.5" in constants["AVAILABLE_RERANKERS"]
    assert "DeepSeek-v4" in constants["AVAILABLE_RERANKERS"]
    assert constants["DEFAULT_RERANKER"] == "M2.5"
    assert "bge" in constants["DEFAULT_FIRST_STAGE_RETRIEVERS"]
    assert "e5" in constants["DEFAULT_FIRST_STAGE_RETRIEVERS"]

    # Case 2-4: validate_rerank_config — exec the standalone function
    # Use deepcopy-like approach: extract the function source as a local definition
    fn_src = constants["_validate_rerank_config_src"]
    # Build a local namespace with the constants
    ns: dict = {
        "DENSE_RETRIEVERS": constants["DENSE_RETRIEVERS"],
        "SPARSE_RETRIEVERS": constants["SPARSE_RETRIEVERS"],
        "COLBERTV2_RETRIEVERS": constants["COLBERTV2_RETRIEVERS"],
        "ALL_PAPER_FIRST_STAGE_RETRIEVERS": constants["ALL_PAPER_FIRST_STAGE_RETRIEVERS"],
        "AVAILABLE_RERANKERS": constants["AVAILABLE_RERANKERS"],
        "DEFAULT_RERANKER": constants["DEFAULT_RERANKER"],
        "Dict": dict,
        "List": list,
    }
    # exec the function
    exec(fn_src, ns)
    validate_rerank_config = ns["validate_rerank_config"]

    # Case 2: validate_rerank_config happy path
    cfg_happy = {
        "rerank": {"first_stage_retrievers": ["bge", "e5"]},
        "llm": {"model": "M2.5"},
    }
    diag = validate_rerank_config(cfg_happy)
    print(f"\n[Case 2] happy path (bge+e5, M2.5)")
    print(f"  diagnostics = {diag}")
    assert diag["first_stage_retrievers"] == ["bge", "e5"]
    assert diag["reranker"] == "M2.5"
    assert diag["all_first_stage_registered"] is True
    assert diag["missing_paper_coverage"] == sorted(
        {"bm25", "splade", "minilm", "star", "ance", "colbertv2"}
    )

    # Case 3: failure — unregistered first-stage
    print(f"\n[Case 3] unregistered first-stage must raise ValueError")
    cfg_bad = {"rerank": {"first_stage_retrievers": ["deepseek_v4_v2_retriever"]}, "llm": {}}
    try:
        validate_rerank_config(cfg_bad)
        print("  FAILED: expected ValueError")
        assert False
    except ValueError as e:
        msg = str(e)
        print(f"  ✓ ValueError raised: {msg[:100]}...")
        assert "unregistered" in msg
        assert "DeepSeek-v4" in msg, "diagnostic should redirect DeepSeek-v4 → llm.model"

    # Case 4: warning — single first-stage
    print(f"\n[Case 4] single first-stage should emit warning")
    cfg_single = {"rerank": {"first_stage_retrievers": ["bge"]}, "llm": {"model": "M2.5"}}
    diag = validate_rerank_config(cfg_single)
    print(f"  warnings = {diag['warnings']}")
    assert any("multi-retriever" in w for w in diag["warnings"]), \
        "single first-stage should warn about sparse comparison"

    # Case 5: rerank_config.json validity
    cfg_json = json.loads(_CONFIG.read_text(encoding="utf-8"))
    print(f"\n[Case 5] rerank_config.json parses + validates")
    print(f"  first_stage_retrievers = {cfg_json['rerank']['first_stage_retrievers']}")
    print(f"  available_rerankers = {cfg_json['llm'].get('available_rerankers')}")
    diag = validate_rerank_config(cfg_json)
    assert diag["all_first_stage_registered"]
    assert diag["reranker"] == "M2.5"
    assert "DeepSeek-v4" in cfg_json["llm"]["available_rerankers"]
    assert len(cfg_json["rerank"]["first_stage_retrievers"]) == 2

    # Case 6: DeepSeek-v4 as reranker in cfg
    print(f"\n[Case 6] DeepSeek-v4 as second-stage reranker")
    cfg_ds = {
        "rerank": {"first_stage_retrievers": ["bge"]},
        "llm": {"model": "DeepSeek-v4"},
    }
    diag = validate_rerank_config(cfg_ds)
    print(f"  diagnostics.reranker = {diag['reranker']}")
    assert diag["reranker"] == "DeepSeek-v4"

    print("\n" + "=" * 60)
    print("ALL CASES PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()