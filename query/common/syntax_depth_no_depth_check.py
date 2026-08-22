"""Generate 10 expression_style queries per user with attribute validation only.

This is the shared body of the three
`06_generate_by_syntax_depth_no_depth_check_10_<Cat>.py` scripts. The
per-category wrappers are thin entry points; all real logic lives here.

The script keeps the expression_style prompt context but does NOT validate the
generated query's dependency-tree depth. Every retained query must still
use exactly the five provided product attributes.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/04_query")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")

from .attribute_helpers import (
    REQUIRED_ATTR_COUNT,
    _attrs_used_from_source,
    _extract_attrs_from_product,
    _format_attrs_for_prompt,
    count_words,
    log,
    validate_query_uses_exactly_five_attrs,
)
from .llm_runner import call_llm_no_empty_retry, load_minimax_client
from .config import get_category_config


_QUERY_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_config.json")
_SYNTAX_DEPTH_ROOT = Path("/home/wlia0047/ar57/wenyu/result/05_syntactic_analysis")  # iter #199 fix: iter #172 deleted 03_syntactic_analysis/ but data persisted under 05_syntactic_analysis/ (115MB saved); schema (top-level users list + summary + timestamp) matches load_user_syntax_depths contract exactly
_OUTPUT_ROOT = Path("/home/wlia0047/ar57/wenyu/result/04_query")
_OUTPUT_FILE_NAME = "query_by_syntax_depth_no_depth_check_10.json"

NUM_CANDIDATES_PER_USER = 10
REGENERATION_MAX_ROUNDS = 10  # iter #85: paper §2.2 10 candidates × 10 rounds
CACHE_PREWARM_MARKER = "CACHE_PREWARM_ONLY"


def _load_global_query_config() -> dict:
    with open(_QUERY_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_GLOBAL_CONFIG = _load_global_query_config()
NUM_USERS_TO_TEST = _GLOBAL_CONFIG["num_users_to_test"]
MAX_WORKERS = _GLOBAL_CONFIG["max_workers"]
USE_MINIMAXIO = _GLOBAL_CONFIG.get("use_minimaxio", False)


def _syntax_depth_file(category: str) -> Path:
    return _SYNTAX_DEPTH_ROOT / category / "user_average_syntax_depth.json"


def _output_file(category: str) -> Path:
    return _OUTPUT_ROOT / category / _OUTPUT_FILE_NAME


def _system_base(category: str) -> str:
    return f"""You generate e-commerce product search queries for {category}.

Construction advice:
- First include the core product type and brand.
- Then place the remaining attributes into a natural first-person shopping request.
- Make candidates structurally different from each other.
- Do not generate near-duplicate surface rewrites of the same syntax shape.
- Do not repeat any attribute value.
- Keep every attribute value in its original surface form.
- Do not singularize, pluralize, stem, paraphrase, or partially rewrite any attribute value.
- If two attribute values are similar, such as singular/plural variants, you must still preserve each one exactly as given.
- Never collapse the query into a keyword list or telegraphic phrase. Every candidate must be a full natural sentence.

Critical lexical constraint:
- The user prompt provides exactly five Product attributes.
- Every candidate query must include all five provided attribute values.
- In each candidate query, each provided attribute value must appear exactly once.
- Copy each attribute value exactly as written.
- Do not change singular/plural form.
- Do not replace an attribute value with a shorter variant.
- Do not merge two similar attribute values into one mention.
- If an attribute value is awkward, still preserve it exactly and build grammar around it.

Candidate clause-count plan:
- Generate exactly 10 candidates.
- Candidates 1-2 must have no dependent clauses.
- Candidates 3-4 must have exactly one dependent clause.
- Candidates 5-6 must have exactly two dependent clauses.
- Candidates 7-8 must have exactly three dependent clauses.
- Candidates 9-10 must have exactly four dependent clauses.
- A dependent clause means a finite relative, complement, or subordinate clause, usually introduced by words such as "that", "which", "who", "when", "where", "because", "while", "if", or "as".
- Do not count the main clause as a dependent clause.
- Use the assigned clause-count group to create structurally diverse candidates.
- Within each pair, use different attachment patterns.
- Across all 10 candidates, vary clause type and attachment point.

Output schema:
- Return JSON only in this exact schema:
  {{"candidates": [{{"query": "..."}}]}}

Hard rules:
1. Output JSON only.
2. The query must be a natural first-person e-commerce search query.
3. Include exactly the five Product attribute values provided in the user prompt.
4. Each provided Product attribute value must appear exactly once in each candidate query.
5. Do not invent extra product facts.
6. Generate exactly the number of candidates requested by the user prompt.
7. If you cannot satisfy every rule, output {{"status":"IMPOSSIBLE","reason":"..."}}.
8. If the user prompt contains {CACHE_PREWARM_MARKER}, this is cache warmup only. In that case, return the requested number of placeholder candidates with "query": "none"."""


def _round_target_depth(avg_depth: float) -> int:
    if not isinstance(avg_depth, (int, float)):
        raise TypeError(f"avg_depth must be numeric, got {type(avg_depth).__name__}")
    if not math.isfinite(avg_depth):
        raise ValueError(f"avg_depth must be finite, got {avg_depth}")
    target_depth = int(round(avg_depth))
    if target_depth < 1:
        raise ValueError(f"target syntax depth must be >= 1, got {target_depth}")
    return target_depth


def load_user_syntax_depths(category: str) -> dict[str, dict]:
    path = _syntax_depth_file(category)
    if not path.exists():
        raise FileNotFoundError(f"syntax depth file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    users = payload.get("users")
    if not isinstance(users, list):
        raise TypeError("syntax depth file must contain a top-level users list")

    depth_map: dict[str, dict] = {}
    for idx, user in enumerate(users):
        if not isinstance(user, dict):
            raise TypeError(f"users[{idx}] must be an object")
        user_id = user.get("user_id")
        avg_depth = user.get("avg_depth")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"users[{idx}].user_id must be a non-empty string")
        if avg_depth is None:
            raise KeyError(f"users[{idx}] missing avg_depth")
        depth_map[user_id] = {
            "avg_depth": float(avg_depth),
            "target_depth": _round_target_depth(float(avg_depth)),
            "review_count": user.get("review_count"),
            "min_depth": user.get("min_depth"),
            "max_depth": user.get("max_depth"),
        }
    return depth_map


def build_syntax_depth_prompt(
    category: str,
    attrs: dict,
    candidate_count: int = NUM_CANDIDATES_PER_USER,
) -> tuple[str, str]:
    user_content = f"""Product attributes:
{_format_attrs_for_prompt(attrs)}

Return exactly {candidate_count} candidates.
"""
    return _system_base(category), user_content


def build_syntax_depth_prewarm_prompt(category: str) -> tuple[str, str]:
    user_content = f"""{CACHE_PREWARM_MARKER}

This request is only for prompt-cache construction. Do not solve the real task.

Return JSON only in this exact schema:
{{"candidates": [{{"query": "none"}}]}}

Return exactly {NUM_CANDIDATES_PER_USER} candidates.
Every candidate must use:
- "query": "none"
"""
    return _system_base(category), user_content


def prewarm_syntax_depth_cache(category: str) -> None:
    system_base, user_content = build_syntax_depth_prewarm_prompt(category)
    call_llm_no_empty_retry(user_content, system_base=system_base, step_name="SyntaxDepthCachePrewarm")


def parse_syntax_depth_response(text_content: str, target_depth: int) -> list[dict]:
    try:
        json_match = re.search(r"\{[\s\S]*\}", text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)
    except Exception as exc:
        log(f"    [DEBUG] Syntax-depth JSON parse failed: {exc}")
        return []

    if not isinstance(data, dict):
        log("    [DEBUG] Syntax-depth response is not an object")
        return []
    if data.get("status") == "IMPOSSIBLE":
        log(f"    [DEBUG] Model returned IMPOSSIBLE: {data.get('reason')}")
        return []

    raw_candidates = data.get("candidates")
    if raw_candidates is None and "query" in data:
        raw_candidates = [data]
    if not isinstance(raw_candidates, list):
        log("    [DEBUG] Syntax-depth response missing candidates list")
        return []

    candidates = []
    for idx, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            log(f"    [DEBUG] candidates[{idx}] is not an object")
            continue
        query = candidate.get("query")
        if not isinstance(query, str) or not query.strip():
            log(f"    [DEBUG] candidates[{idx}] missing non-empty query")
            continue
        candidates.append(
            {
                "target_depth": target_depth,
                "query": query.strip(),
                "word_count": count_words(query.strip()),
            }
        )
    return candidates


def _load_json_file(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_user_tasks(category: str) -> tuple[list, list[dict]]:
    cat_config = get_category_config(category)
    attr_density_file = cat_config["attr_density_profiles_file"]
    attr_values_file = cat_config["attr_values_file"]

    syntax_depth_map = load_user_syntax_depths(category)
    attr_density_profiles = _load_json_file(attr_density_file)
    attr_values_data = _load_json_file(attr_values_file)

    user_wpa_map: dict[str, float] = {}
    for profile in attr_density_profiles:
        uid = profile.get("user_id")
        wpa = profile.get("words_per_attribute")
        if isinstance(uid, str) and uid and wpa is not None:
            user_wpa_map[uid] = float(wpa)

    if not isinstance(attr_values_data, dict) or "products" not in attr_values_data:
        raise TypeError("attributes file must contain top-level products list")

    user_prod_map: dict[str, list[dict]] = {}
    for product in attr_values_data["products"]:
        if not isinstance(product, dict):
            raise TypeError("every product entry must be an object")
        uid = product.get("user_id")
        if isinstance(uid, str) and uid:
            user_prod_map.setdefault(uid, []).append(product)

    candidate_user_ids = sorted(set(syntax_depth_map) & set(user_prod_map) & set(user_wpa_map))
    log(f"syntax depth users: {len(syntax_depth_map)}")
    log(f"users with product attrs: {len(user_prod_map)}")
    log(f"users with attr density: {len(user_wpa_map)}")
    log(f"candidate users: {len(candidate_user_ids)}")

    out_path = _output_file(category)
    completed_user_ids = set()
    existing_results = []
    if out_path.exists():
        existing_results = _load_json_file(out_path)
        if not isinstance(existing_results, list):
            raise TypeError(f"existing output must be a list: {out_path}")
        completed_user_ids = {item["user_id"] for item in existing_results if "user_id" in item}
        log(f"existing completed users: {len(completed_user_ids)}")

    tasks = []
    skipped_attr_count = 0
    for uid in candidate_user_ids:
        if uid in completed_user_ids:
            continue
        product = user_prod_map[uid][0]
        attrs = _extract_attrs_from_product(product)
        if len(attrs) < REQUIRED_ATTR_COUNT:
            skipped_attr_count += 1
            continue
        depth_info = syntax_depth_map[uid]
        tasks.append(
            {
                "user_id": uid,
                "asin": product.get("asin", ""),
                "attrs": attrs,
                "syntax_depth": depth_info,
                "words_per_attribute": user_wpa_map[uid],
            }
        )
        if len(tasks) >= NUM_USERS_TO_TEST:
            break

    log(f"skipped by attr count: {skipped_attr_count}")
    log(f"new tasks: {len(tasks)}")
    return existing_results, tasks


def process_one_user(category: str, task: dict) -> dict | None:
    """Generate 10 valid expression_style queries per user with regeneration loop.

    Per paper §2.2: "generating 10 candidate sentences per round for a
    maximum of 10 iterations; samples that still fail after 10 rounds are
    excluded." This impl:
      - Calls LLM up to REGENERATION_MAX_ROUNDS=10 times per user.
      - Each round generates NUM_CANDIDATES_PER_USER=10 candidates.
      - Stops early once we have ≥10 valid candidates.
      - Emits regeneration_history (per-round audit trail) regardless of
        whether regeneration was triggered or not (round 1 is always
        recorded), so iter #82's Pipeline_Regeneration_10x10 claim has a
        code-level reproduce path.

    Returns None if 10 valid candidates cannot be collected within
    REGENERATION_MAX_ROUNDS (no fallback / no default — caller drops user).
    """
    uid = task["user_id"]
    attrs = task["attrs"]
    source_attrs_used = _attrs_used_from_source(attrs)
    avg_depth = task["syntax_depth"]["avg_depth"]
    target_depth = task["syntax_depth"]["target_depth"]

    all_collected: list[dict] = []
    regeneration_history: list[dict] = []
    regeneration_rounds_used = 0

    for round_idx in range(REGENERATION_MAX_ROUNDS):
        regeneration_rounds_used = round_idx + 1
        round_record: dict = {
            "round": regeneration_rounds_used,
            "candidates_requested": NUM_CANDIDATES_PER_USER,
            "candidates_returned": 0,
            "candidates_accepted_this_round": 0,
            "accepted_candidate_indices": [],
            "rejected_reasons": [],
            "empty_response": False,
            "parse_failed": False,
            "llm_step_name": f"SyntaxDepthQuery_round{regeneration_rounds_used}",
        }

        system_base, user_content = build_syntax_depth_prompt(category, attrs)
        response = call_llm_no_empty_retry(
            user_content,
            system_base=system_base,
            step_name=round_record["llm_step_name"],
        )
        if not response:
            log(f"    [WARN] Round {regeneration_rounds_used}: Empty response, user={uid}")
            round_record["empty_response"] = True
            regeneration_history.append(round_record)
            continue

        candidates = parse_syntax_depth_response(response, target_depth)
        if not candidates:
            log(f"    [WARN] Round {regeneration_rounds_used}: Parse failed, user={uid}")
            round_record["parse_failed"] = True
            regeneration_history.append(round_record)
            continue

        round_record["candidates_returned"] = len(candidates)
        for idx, parsed in enumerate(candidates, start=1):
            query = parsed["query"]
            attr_ok, attr_error = validate_query_uses_exactly_five_attrs(query, source_attrs_used)
            if not attr_ok:
                round_record["rejected_reasons"].append(f"candidate={idx}: attr failed: {attr_error}")
                continue

            round_record["accepted_candidate_indices"].append(idx)
            round_record["candidates_accepted_this_round"] += 1
            all_collected.append(
                {
                    "target_depth": target_depth,
                    "actual_depth": None,
                    "depth_validation_skipped": True,
                    "user_avg_depth": avg_depth,
                    "query": query,
                    "word_count": count_words(query),
                    "attrs_used": dict(source_attrs_used),
                    "accepted_candidate_index": idx,
                    "candidate_count": len(candidates),
                    "regeneration_round": regeneration_rounds_used,
                }
            )

        regeneration_history.append(round_record)
        if len(all_collected) >= NUM_CANDIDATES_PER_USER:
            break

    if len(all_collected) < NUM_CANDIDATES_PER_USER:
        log(
            f"    [ERROR] Regeneration exhausted: {len(all_collected)}/{NUM_CANDIDATES_PER_USER} "
            f"valid candidates after {regeneration_rounds_used} rounds, user={uid}"
        )
        return None

    accepted_queries = all_collected[:NUM_CANDIDATES_PER_USER]
    return {
        "user_id": uid,
        "asin": task["asin"],
        "syntax_depth_queries": accepted_queries,
        "syntax_depth_query": accepted_queries[0],
        "query_count": len(accepted_queries),
        "depth_validation_skipped": True,
        "regeneration_rounds_used": regeneration_rounds_used,
        "regeneration_triggered": regeneration_rounds_used > 1,
        "regeneration_history": regeneration_history,
    }


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def main(category: str) -> None:
    existing_results, tasks = build_user_tasks(category)
    if not tasks:
        raise ValueError("No expression_style query tasks to process")

    out_path = _output_file(category)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    load_minimax_client(use_minimaxio=USE_MINIMAXIO)
    prewarm_syntax_depth_cache(category)

    results = existing_results.copy()
    existing_completed_before_run = len(existing_results)
    run_success_count = 0
    failed_users = []
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_one_user, category, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            if result:
                results.append(result)
                run_success_count += 1
                write_json_atomic(out_path, results)
                log(
                    f"  [run_success={run_success_count}/{len(tasks)}] "
                    f"total_records={len(results)} user={result['user_id'][:20]}"
                )
            else:
                failed_users.append(task["user_id"])

    elapsed = time.time() - total_start
    log("=" * 60)
    log(f"existing completed users before run: {existing_completed_before_run}")
    log(f"successful users in this run: {run_success_count}")
    log(f"failed users in this run: {len(failed_users)}")
    log(f"processed users in this run: {len(tasks)}")
    log(f"total records after merge: {len(results)}")
    log(f"elapsed: {elapsed:.1f}s")
    log(f"output={out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="Baby_Products | Grocery_and_Gourmet_Food | Pet_Supplies")
    args = parser.parse_args()
    main(args.category)
