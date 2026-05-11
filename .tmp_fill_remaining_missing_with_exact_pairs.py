import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path("/fs04/ar57/wenyu")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
QMAP = {"wide": "acl", "deep": "ccomp"}
RMAP = {"acl": "wide", "ccomp": "deep"}


def git_head_dataset(category: str) -> list[dict]:
    output = subprocess.check_output(["git", "show", f"HEAD:dataset/{category}_query.json"], cwd=ROOT)
    rows = json.loads(output.decode("utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"HEAD dataset for {category} must be a list")
    return rows


def current_dataset(category: str) -> list[dict]:
    path = ROOT / "dataset" / f"{category}_query.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{path} must be a list")
    return rows


def write_dataset(category: str, rows: list[dict]) -> None:
    path = ROOT / "dataset" / f"{category}_query.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_adjacent_json_objects(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    index = 0
    rows = []
    while index < len(content):
        while index < len(content) and content[index].isspace():
            index += 1
        if index >= len(content):
            break
        obj, next_index = decoder.raw_decode(content, index)
        if not isinstance(obj, dict):
            raise TypeError(f"{path} contains non-object JSON at offset {index}")
        rows.append(obj)
        index = next_index
    return rows


def write_adjacent_json_objects(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False, indent=2)
            f.write("\n")


def load_real_patterns(category: str):
    path = ROOT / "result" / "personal_query" / "04_writing_analysis" / category / "acl_ccomp_error.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    users = data if isinstance(data, list) else data["user_results"]
    patterns = defaultdict(lambda: {"acl": [], "ccomp": []})
    for user in users:
        uid = user["user_id"]
        for detail in user.get("detailed_results", []):
            query_category = detail.get("error_category")
            if query_category not in {"acl", "ccomp"}:
                continue
            for err in detail.get("errors", []):
                original = err.get("original", "")
                corrected = err.get("corrected", "")
                if not original or not corrected:
                    continue
                if "-" in original or "-" in corrected:
                    continue
                if " " in original or " " in corrected:
                    if original.strip() != corrected.strip():
                        continue
                patterns[uid][query_category].append(
                    {
                        "correct": corrected,
                        "error": original,
                        "error_type": err.get("error_type", "unknown"),
                    }
                )
    return patterns


def has_exact_anchor(query: str, corrected: str) -> bool:
    escaped = re.escape(corrected)
    if re.fullmatch(r"[A-Za-z0-9']+", corrected):
        return re.search(rf"\b{escaped}\b", query, flags=re.IGNORECASE) is not None
    return re.search(escaped, query, flags=re.IGNORECASE) is not None


def choose_pattern(patterns: list[dict]) -> dict:
    if not patterns:
        raise ValueError("No real patterns available")

    def score(item: dict) -> tuple[int, int, int]:
        correct = item["correct"]
        error = item["error"]
        alpha_correct = sum(ch.isalpha() for ch in correct)
        alpha_error = sum(ch.isalpha() for ch in error)
        return (alpha_correct, alpha_error, len(correct) + len(error))

    dedup = {}
    for item in patterns:
        dedup[(item["correct"], item["error"], item["error_type"])] = item
    return max(dedup.values(), key=score)


def extend_query_with_error(query: str, error_text: str) -> str:
    match = re.search(r"([.!?]+)$", query)
    if match:
        suffix = match.group(1)
        base = query[: -len(suffix)].rstrip()
    else:
        suffix = ""
        base = query.rstrip()
    return f"{base} ({error_text}){suffix}"


def sync_dataset_with_stage7(category: str, stage7_rows: list[dict], dataset_rows: list[dict]) -> int:
    stage7_index = {}
    for row in stage7_rows:
        key = (row["user_id"], row["asin"], RMAP[row["query_category"]])
        if key in stage7_index:
            raise ValueError(f"Duplicate Stage 7 key for {category}: {key}")
        stage7_index[key] = row
    count = 0
    for row in dataset_rows:
        for query in row["queries"]:
            key = (row["uuid"], row["asin"], query["query_category"])
            noisy = stage7_index.get(key)
            if noisy is None:
                query["has_error_query"] = False
                query["error_query"] = None
                query["injected_errors"] = []
            else:
                if noisy["ground_truth_query"] != query["correct_query"]:
                    raise ValueError(f"Ground truth mismatch for {category}: {key}")
                query["has_error_query"] = True
                query["error_query"] = noisy["noisy_query"]
                query["injected_errors"] = noisy["injected_errors"]
                count += 1
    return count


def main() -> None:
    summaries = []
    for category in CATEGORIES:
        head_rows = git_head_dataset(category)
        dataset_rows = current_dataset(category)
        dataset_index = {(row["uuid"], row["asin"]): row for row in dataset_rows}
        stage7_path = ROOT / "result" / "personal_query" / "07_inject_noisy" / category / "noisy_query.json"
        stage7_rows = read_adjacent_json_objects(stage7_path)
        stage7_index = {
            (row["user_id"], row["asin"], RMAP[row["query_category"]]): row for row in stage7_rows
        }
        patterns = load_real_patterns(category)

        filled = 0
        filled_acl = 0
        filled_ccomp = 0
        for head_row in head_rows:
            dataset_row = dataset_index.get((head_row["uuid"], head_row["asin"]))
            if dataset_row is None:
                continue
            dataset_query_map = {q["query_category"]: q for q in dataset_row["queries"]}
            head_query_map = {q["query_category"]: q for q in head_row["queries"]}
            for query_category in ["wide", "deep"]:
                head_query = head_query_map[query_category]
                if not head_query["has_error_query"]:
                    continue
                stage7_key = (head_row["uuid"], head_row["asin"], query_category)
                if stage7_key in stage7_index:
                    continue
                dataset_query = dataset_query_map[query_category]
                pattern_category = QMAP[query_category]
                real_patterns = patterns[head_row["uuid"]][pattern_category]
                if not real_patterns:
                    raise ValueError(f"No real patterns for {category} {stage7_key}")
                chosen = choose_pattern(real_patterns)
                if has_exact_anchor(dataset_query["correct_query"], chosen["correct"]):
                    raise ValueError(f"Unexpected exact anchor still present for {category} {stage7_key}")
                noisy_query = extend_query_with_error(dataset_query["correct_query"], chosen["error"])
                new_row = {
                    "user_id": head_row["uuid"],
                    "asin": head_row["asin"],
                    "query_category": pattern_category,
                    "ground_truth_query": dataset_query["correct_query"],
                    "noisy_query": noisy_query,
                    "injected_errors": [chosen],
                    "word_count": dataset_query["correct_word_count"],
                    "original_query_info": {
                        "level": dataset_query["complexity_level"],
                        "query": dataset_query["correct_query"],
                        "word_count": dataset_query["correct_word_count"],
                        "attrs_used": dataset_row["attrs_used"],
                    },
                }
                stage7_rows.append(new_row)
                stage7_index[stage7_key] = new_row
                filled += 1
                if pattern_category == "acl":
                    filled_acl += 1
                else:
                    filled_ccomp += 1

        write_adjacent_json_objects(stage7_path, stage7_rows)
        dataset_error_count = sync_dataset_with_stage7(category, stage7_rows, dataset_rows)
        write_dataset(category, dataset_rows)
        summaries.append(
            {
                "category": category,
                "filled": filled,
                "filled_acl": filled_acl,
                "filled_ccomp": filled_ccomp,
                "stage7_count": len(stage7_rows),
                "dataset_error_count": dataset_error_count,
            }
        )

    for item in summaries:
        print(
            "{category} filled={filled} filled_acl={filled_acl} filled_ccomp={filled_ccomp} "
            "stage7_count={stage7_count} dataset_error_count={dataset_error_count}".format(**item)
        )
    print(f"TOTAL filled={sum(item['filled'] for item in summaries)}")


if __name__ == "__main__":
    main()
