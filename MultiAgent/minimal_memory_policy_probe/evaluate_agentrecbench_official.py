#!/usr/bin/env python3
"""Evaluate a recommendation model with official AgentRecBench candidate sets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files
import requests

from supervised_memory_policy_training import LLAMA_FACTORY_DIR


PROBE_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_PATH = (
    LLAMA_FACTORY_DIR / "saves" / "real_multiagent_memory_policy" / "qwen2.5-0.5b" / "lora" / "sft"
)
DEFAULT_OUTPUT_JSON = PROBE_DIR / "real_multiagent_teacher" / "agentrecbench_official_eval_result.json"
DEFAULT_CONTEXT_CACHE_DIR = PROBE_DIR / "real_multiagent_teacher" / "agentrecbench_context_cache"
VALID_SCENARIOS = ("classic", "user_cold_start", "item_cold_start", "long_term", "short_term")
VALID_DOMAINS = ("amazon", "goodreads", "yelp")
SCENARIO_TABLE_GROUP = {
    "classic": "output_data_all",
    "user_cold_start": "output_data_all",
    "item_cold_start": "output_data_all",
    "long_term": "output_data_long",
    "short_term": "output_data_short",
}


@dataclass(frozen=True)
class AgentRecBenchCase:
    scenario: str
    domain: str
    index: int
    user_id: str
    candidate_category: str
    candidate_items: tuple[str, ...]
    ground_truth: str
    loc: tuple[int, int]
    data_type: str | None


@dataclass(frozen=True)
class AgentRecBenchPromptContext:
    table_group: str
    user_record: dict[str, Any]
    item_records: dict[str, dict[str, Any]]
    user_reviews: list[dict[str, Any]]


@dataclass(frozen=True)
class ParsedRanking:
    ranking: tuple[str, ...] | None
    parsed_json: Any | None
    raw_text: str
    parse_error: str | None


@dataclass(frozen=True)
class AgentRecBenchEvalConfig:
    repo_id: str
    scenario: str
    domain: str
    index_start: int
    sample_count: int
    model_name_or_path: str
    adapter_path: Path
    output_json: Path
    max_new_tokens: int
    context_cache_dir: Path
    max_user_reviews: int
    dry_run: bool


def require_key(payload: dict[str, Any], key: str, context: str) -> Any:
    if key not in payload:
        raise ValueError(f"{context} missing required key: {key}")
    return payload[key]


def read_hf_json(repo_id: str, filename: str) -> dict[str, Any]:
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with open(local_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"AgentRecBench file is not a JSON object: {filename}")
    return payload


def safe_cache_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def compact_value(value: Any, max_string_chars: int = 400, max_list_items: int = 6, max_dict_items: int = 16) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_string_chars else value[: max_string_chars - 3] + "..."
    if isinstance(value, list):
        return [compact_value(item, max_string_chars, max_list_items, max_dict_items) for item in value[:max_list_items]]
    if isinstance(value, dict):
        items = list(value.items())[:max_dict_items]
        return {
            str(key): compact_value(item, max_string_chars, max_list_items, max_dict_items)
            for key, item in items
        }
    return value


def compact_user_record(row: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in ("user_id", "source", "type", "name", "average_rating", "rating_number"):
        if key in row:
            selected[key] = compact_value(row[key], max_string_chars=160)
    return selected


def compact_item_record(row: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in ("item_id", "source", "type", "title", "name", "main_category", "categories", "average_rating", "stars", "price", "store"):
        if key in row:
            selected[key] = compact_value(row[key], max_string_chars=180, max_list_items=4, max_dict_items=8)
    if "features" in row:
        selected["features"] = compact_value(row["features"], max_string_chars=180, max_list_items=2, max_dict_items=8)
    if "description" in row:
        selected["description"] = compact_value(row["description"], max_string_chars=220, max_list_items=1, max_dict_items=8)
    return selected


def compact_review_record(row: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in ("review_id", "user_id", "item_id", "stars", "rating", "date", "source", "type"):
        if key in row:
            selected[key] = compact_value(row[key], max_string_chars=120)
    if "text" in row:
        selected["text"] = compact_value(row["text"], max_string_chars=220)
    return selected


def stream_remote_jsonl(repo_id: str, filename: str):
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="dataset")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"remote JSONL row is not a JSON object in {filename}")
            yield payload


def cache_path(config: AgentRecBenchEvalConfig, table_name: str, record_id: str) -> Path:
    return config.context_cache_dir / table_name / f"{safe_cache_key(record_id)}.json"


def load_cached_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cached_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def table_filename(scenario: str, table_name: str) -> str:
    if scenario not in SCENARIO_TABLE_GROUP:
        raise ValueError(f"invalid AgentRecBench scenario for table lookup: {scenario}")
    if table_name not in ("user", "item", "review"):
        raise ValueError(f"invalid AgentRecBench table name: {table_name}")
    return f"{SCENARIO_TABLE_GROUP[scenario]}/{table_name}.json"


def collect_user_records(config: AgentRecBenchEvalConfig, cases: list[AgentRecBenchCase]) -> dict[str, dict[str, Any]]:
    user_records: dict[str, dict[str, Any]] = {}
    missing_user_ids: set[str] = set()
    for case in cases:
        cached = load_cached_json(cache_path(config, "user", case.user_id))
        if cached is None:
            missing_user_ids.add(case.user_id)
        else:
            if not isinstance(cached, dict):
                raise ValueError(f"cached user row is not a JSON object for user_id={case.user_id}")
            user_records[case.user_id] = cached

    if missing_user_ids:
        for row in stream_remote_jsonl(config.repo_id, table_filename(config.scenario, "user")):
            row_user_id = row.get("user_id")
            if not isinstance(row_user_id, str):
                continue
            if row_user_id in missing_user_ids:
                compact_row = compact_user_record(row)
                user_records[row_user_id] = compact_row
                write_cached_json(cache_path(config, "user", row_user_id), compact_row)
                missing_user_ids.remove(row_user_id)
                if not missing_user_ids:
                    break

    if missing_user_ids:
        raise ValueError(f"missing AgentRecBench user rows for user_ids={sorted(missing_user_ids)}")
    return user_records


def collect_item_records(config: AgentRecBenchEvalConfig, cases: list[AgentRecBenchCase]) -> dict[str, dict[str, Any]]:
    item_records: dict[str, dict[str, Any]] = {}
    missing_item_ids: set[str] = set()
    for case in cases:
        for item_id in case.candidate_items:
            cached = load_cached_json(cache_path(config, "item", item_id))
            if cached is None:
                missing_item_ids.add(item_id)
            else:
                if not isinstance(cached, dict):
                    raise ValueError(f"cached item row is not a JSON object for item_id={item_id}")
                item_records[item_id] = cached

    if missing_item_ids:
        for row in stream_remote_jsonl(config.repo_id, table_filename(config.scenario, "item")):
            row_item_id = row.get("item_id")
            if not isinstance(row_item_id, str):
                continue
            if row_item_id in missing_item_ids:
                compact_row = compact_item_record(row)
                item_records[row_item_id] = compact_row
                write_cached_json(cache_path(config, "item", row_item_id), compact_row)
                missing_item_ids.remove(row_item_id)
                if not missing_item_ids:
                    break

    if missing_item_ids:
        raise ValueError(f"missing AgentRecBench item rows for item_ids={sorted(missing_item_ids)}")
    return item_records


def collect_user_reviews(config: AgentRecBenchEvalConfig, cases: list[AgentRecBenchCase]) -> dict[str, list[dict[str, Any]]]:
    reviews_by_user: dict[str, list[dict[str, Any]]] = {}
    missing_user_ids: set[str] = set()
    for case in cases:
        cached = load_cached_json(cache_path(config, "review", case.user_id))
        if cached is None:
            missing_user_ids.add(case.user_id)
        else:
            if not isinstance(cached, list):
                raise ValueError(f"cached review rows are not a JSON list for user_id={case.user_id}")
            reviews_by_user[case.user_id] = cached

    if missing_user_ids:
        live_reviews: dict[str, list[dict[str, Any]]] = {user_id: [] for user_id in missing_user_ids}
        target_count = config.max_user_reviews
        for row in stream_remote_jsonl(config.repo_id, table_filename(config.scenario, "review")):
            row_user_id = row.get("user_id")
            if not isinstance(row_user_id, str):
                continue
            if row_user_id in live_reviews and len(live_reviews[row_user_id]) < target_count:
                compact_row = compact_review_record(row)
                live_reviews[row_user_id].append(compact_row)
                if all(len(rows) >= target_count for rows in live_reviews.values()):
                    break

        unresolved = [user_id for user_id, rows in live_reviews.items() if not rows]
        if unresolved:
            raise ValueError(f"missing AgentRecBench review rows for user_ids={sorted(unresolved)}")
        for user_id, rows in live_reviews.items():
            reviews_by_user[user_id] = rows
            write_cached_json(cache_path(config, "review", user_id), rows)

    return reviews_by_user


def build_prompt_contexts(
    config: AgentRecBenchEvalConfig,
    cases: list[AgentRecBenchCase],
) -> dict[int, AgentRecBenchPromptContext]:
    user_records = collect_user_records(config, cases)
    item_records = collect_item_records(config, cases)
    reviews_by_user = collect_user_reviews(config, cases)
    return {
        case.index: AgentRecBenchPromptContext(
            table_group=SCENARIO_TABLE_GROUP[case.scenario],
            user_record=user_records[case.user_id],
            item_records={item_id: item_records[item_id] for item_id in case.candidate_items},
            user_reviews=reviews_by_user[case.user_id],
        )
        for case in cases
    }


def available_case_indices(repo_id: str, scenario: str, domain: str, group: str) -> tuple[int, ...]:
    if group not in ("tasks", "groundtruth"):
        raise ValueError(f"invalid AgentRecBench group: {group}")
    pattern = re.compile(rf"^task/{re.escape(scenario)}/{re.escape(domain)}/{group}/(?:task|groundtruth)_(\d+)\.json$")
    indices: list[int] = []
    for repo_file in list_repo_files(repo_id, repo_type="dataset"):
        match = pattern.match(repo_file)
        if match is not None:
            indices.append(int(match.group(1)))
    if not indices:
        raise ValueError(f"no AgentRecBench {group} files found for scenario={scenario}, domain={domain}")
    return tuple(sorted(indices))


def load_agentrecbench_cases(config: AgentRecBenchEvalConfig) -> list[AgentRecBenchCase]:
    if config.scenario not in VALID_SCENARIOS:
        raise ValueError(f"invalid AgentRecBench scenario: {config.scenario}")
    if config.domain not in VALID_DOMAINS:
        raise ValueError(f"invalid AgentRecBench domain: {config.domain}")
    if config.index_start < 0:
        raise ValueError("index_start must be non-negative")
    if config.sample_count <= 0:
        raise ValueError("sample_count must be positive")

    available_task_indices = set(available_case_indices(config.repo_id, config.scenario, config.domain, "tasks"))
    available_groundtruth_indices = set(
        available_case_indices(config.repo_id, config.scenario, config.domain, "groundtruth")
    )
    available_indices = sorted(available_task_indices.intersection(available_groundtruth_indices))
    if not available_indices:
        raise ValueError(
            f"no overlapping AgentRecBench task and groundtruth files for scenario={config.scenario}, domain={config.domain}"
        )

    cases: list[AgentRecBenchCase] = []
    for index in range(config.index_start, config.index_start + config.sample_count):
        if index not in available_task_indices or index not in available_groundtruth_indices:
            raise ValueError(
                "requested AgentRecBench index is unavailable: "
                f"scenario={config.scenario}, domain={config.domain}, index={index}, "
                f"available_count={len(available_indices)}, max_index={available_indices[-1]}"
            )
        task_filename = f"task/{config.scenario}/{config.domain}/tasks/task_{index}.json"
        groundtruth_filename = f"task/{config.scenario}/{config.domain}/groundtruth/groundtruth_{index}.json"
        task_payload = read_hf_json(config.repo_id, task_filename)
        groundtruth_payload = read_hf_json(config.repo_id, groundtruth_filename)

        task_type = require_key(task_payload, "type", task_filename)
        if task_type != "recommendation":
            raise ValueError(f"unsupported AgentRecBench task type in {task_filename}: {task_type}")
        user_id = require_key(task_payload, "user_id", task_filename)
        candidate_category = require_key(task_payload, "candidate_category", task_filename)
        candidate_list = require_key(task_payload, "candidate_list", task_filename)
        loc = require_key(task_payload, "loc", task_filename)
        ground_truth = require_key(groundtruth_payload, "ground truth", groundtruth_filename)

        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"user_id must be a non-empty string in {task_filename}")
        if not isinstance(candidate_list, list):
            raise ValueError(f"candidate_list must be a list in {task_filename}")
        if not isinstance(candidate_category, str) or not candidate_category:
            raise ValueError(f"candidate_category must be a non-empty string in {task_filename}")
        if not isinstance(loc, list) or len(loc) != 2:
            raise ValueError(f"loc must be a two-element list in {task_filename}")
        candidate_items = tuple(str(item) for item in candidate_list)
        if len(candidate_items) != 20:
            raise ValueError(f"candidate_list must contain exactly 20 items in {task_filename}: {len(candidate_items)}")
        if not isinstance(ground_truth, str) or not ground_truth:
            raise ValueError(f"ground truth must be a non-empty string in {groundtruth_filename}")
        if ground_truth not in candidate_items:
            raise ValueError(f"ground truth is not in candidate_list for index {index}: {ground_truth}")

        cases.append(
            AgentRecBenchCase(
                scenario=config.scenario,
                domain=config.domain,
                index=index,
                user_id=user_id,
                candidate_category=candidate_category,
                candidate_items=candidate_items,
                ground_truth=ground_truth,
                loc=(int(loc[0]), int(loc[1])),
                data_type=str(task_payload["data type"]) if "data type" in task_payload else None,
            )
        )

    return cases


def build_messages(case: AgentRecBenchCase, context: AgentRecBenchPromptContext) -> list[dict[str, str]]:
    candidate_json = json.dumps(list(case.candidate_items), ensure_ascii=False, indent=2)
    item_context = [context.item_records[item_id] for item_id in case.candidate_items]
    system = (
        "You are an AgentRecBench recommendation ranker. Use the official user, item, and review table rows "
        "provided below to rank the official 20 candidate_items from most suitable to least suitable. "
        "Return JSON only as a single array."
    )
    user = (
        "Return only one JSON array containing all 20 candidate_items exactly once, with no extra items.\n"
        f"user_id: {case.user_id}\n"
        f"scenario: {case.scenario}\n"
        f"domain: {case.domain}\n"
        f"candidate_category: {case.candidate_category}\n"
        f"loc: {list(case.loc)}\n"
        f"data_type: {case.data_type}\n"
        f"candidate_items:\n{candidate_json}"
        "\n\nofficial_user_table_row:\n"
        f"{json.dumps(context.user_record, ensure_ascii=False, indent=2)}"
        "\n\nofficial_item_table_rows:\n"
        f"{json.dumps(item_context, ensure_ascii=False, indent=2)}"
        "\n\nofficial_review_table_rows_for_user:\n"
        f"{json.dumps(context.user_reviews, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_value(text: str) -> Any:
    if not text.strip():
        raise ValueError("model returned an empty response")
    stripped = text.strip()
    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start == -1 or array_end == -1 or array_end <= array_start:
        raise ValueError(f"response does not contain a JSON array: {text[:300]}")
    payload = json.loads(stripped[array_start : array_end + 1])
    return payload


def parse_ranking(text: str, case: AgentRecBenchCase) -> ParsedRanking:
    try:
        parsed = extract_json_value(text)
        if not isinstance(parsed, list):
            raise ValueError("model response JSON must be an array")
        ranking = tuple(str(item) for item in parsed)
        if len(ranking) != 20:
            raise ValueError(f"ranking must contain exactly 20 items: {len(ranking)}")
        candidate_counter = Counter(case.candidate_items)
        ranking_counter = Counter(ranking)
        invalid_items = [item for item in ranking if item not in candidate_counter]
        if invalid_items:
            raise ValueError(f"ranking contains items outside candidate_list: {invalid_items}")
        if ranking_counter != candidate_counter:
            raise ValueError(
                "ranking item counts do not match candidate_list: "
                f"expected={dict(candidate_counter)}, actual={dict(ranking_counter)}"
            )
        return ParsedRanking(ranking=ranking, parsed_json=parsed, raw_text=text, parse_error=None)
    except (json.JSONDecodeError, ValueError) as exc:
        return ParsedRanking(ranking=None, parsed_json=None, raw_text=text, parse_error=str(exc))


def load_qwen_model(model_name_or_path: str, adapter_path: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AgentRecBench Qwen evaluation")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter path does not exist: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_ranking(
    model,
    tokenizer,
    case: AgentRecBenchCase,
    context: AgentRecBenchPromptContext,
    max_new_tokens: int,
) -> ParsedRanking:
    import torch

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    messages = build_messages(case, context)
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, input_ids.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_ranking(text, case)


def hit_at_k(case: AgentRecBenchCase, prediction: ParsedRanking, k: int) -> int:
    if k <= 0:
        raise ValueError("k must be positive")
    if prediction.ranking is None:
        return 0
    return 1 if case.ground_truth in prediction.ranking[:k] else 0


def summarize(cases: list[AgentRecBenchCase], predictions: list[ParsedRanking]) -> dict[str, Any]:
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions length mismatch")
    if not cases:
        raise ValueError("cases cannot be empty")
    hr_at_1 = sum(hit_at_k(case, prediction, 1) for case, prediction in zip(cases, predictions)) / len(cases)
    hr_at_3 = sum(hit_at_k(case, prediction, 3) for case, prediction in zip(cases, predictions)) / len(cases)
    hr_at_5 = sum(hit_at_k(case, prediction, 5) for case, prediction in zip(cases, predictions)) / len(cases)
    return {
        "hr_at_1": hr_at_1,
        "hr_at_3": hr_at_3,
        "hr_at_5": hr_at_5,
        "hr_avg": (hr_at_1 + hr_at_3 + hr_at_5) / 3,
        "parse_success_rate": sum(prediction.parse_error is None for prediction in predictions) / len(predictions),
        "parse_errors": {
            error: sum(prediction.parse_error == error for prediction in predictions)
            for error in sorted({prediction.parse_error for prediction in predictions if prediction.parse_error is not None})
        },
    }


def dry_run_predictions(cases: list[AgentRecBenchCase]) -> list[ParsedRanking]:
    predictions: list[ParsedRanking] = []
    for case in cases:
        remaining = list(case.candidate_items)
        remaining.remove(case.ground_truth)
        ordered = (case.ground_truth,) + tuple(remaining)
        parsed_json = list(ordered)
        predictions.append(
            ParsedRanking(
                ranking=ordered,
                parsed_json=parsed_json,
                raw_text=json.dumps(parsed_json, ensure_ascii=False),
                parse_error=None,
            )
        )
    return predictions


def run_eval(config: AgentRecBenchEvalConfig) -> dict[str, Any]:
    cases = load_agentrecbench_cases(config)
    contexts = build_prompt_contexts(config, cases)
    prompt_previews = {
        case.index: build_messages(case, contexts[case.index])[1]["content"][:2000]
        for case in cases[:3]
    }
    if config.dry_run:
        predictions = dry_run_predictions(cases)
    else:
        tokenizer, model = load_qwen_model(config.model_name_or_path, config.adapter_path)
        predictions = [
            generate_ranking(model, tokenizer, case, contexts[case.index], config.max_new_tokens)
            for case in cases
        ]

    result = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "sizes": {
            "cases": len(cases),
            "candidate_items_per_case": 20,
        },
        "metrics": summarize(cases, predictions),
        "examples": [
            {
                "case": asdict(case),
                "prediction": {
                    "ranking": list(prediction.ranking) if prediction.ranking is not None else None,
                    "parse_error": prediction.parse_error,
                    "parsed_json": prediction.parsed_json,
                    "raw_text": prediction.raw_text[:1200],
                },
                "prompt_preview": prompt_previews.get(case.index),
            }
            for case, prediction in zip(cases[:3], predictions[:3])
        ],
    }
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> AgentRecBenchEvalConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="SGJQovo/AgentRecBench")
    parser.add_argument("--scenario", choices=VALID_SCENARIOS, default="classic")
    parser.add_argument("--domain", choices=VALID_DOMAINS, default="amazon")
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--context-cache-dir", type=Path, default=DEFAULT_CONTEXT_CACHE_DIR)
    parser.add_argument("--max-user-reviews", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return AgentRecBenchEvalConfig(
        repo_id=args.repo_id,
        scenario=args.scenario,
        domain=args.domain,
        index_start=args.index_start,
        sample_count=args.sample_count,
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        output_json=args.output_json,
        max_new_tokens=args.max_new_tokens,
        context_cache_dir=args.context_cache_dir,
        max_user_reviews=args.max_user_reviews,
        dry_run=args.dry_run,
    )


def main() -> None:
    config = parse_args()
    result = run_eval(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
