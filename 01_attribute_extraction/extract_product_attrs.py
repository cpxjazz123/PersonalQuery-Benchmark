#!/usr/bin/env python3
"""Product attribute extraction from meta_Baby_Products_2023.jsonl.gz.

Reads product metadata, extracts structured attributes per ASIN (Brand,
Main Category, Color, Material, Size, ..., no numeric values), writes to
result/product_attributes.json.

Also exports select_top_attrs() utility used by gaussian/build_user.py
to pick top-N attrs per ASIN with numeric value exclusion.

参数全部硬编码 (Rule 3), 不接受 CLI 参数.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")

# === Inputs ===
DATA = Path("/fs04/ar57/wenyu/PersoanlQuery/data")
META_GZ = DATA / "meta_Baby_Products_2023.jsonl"

# 用户指令 2026-09-04: multiprocessing 加速 (13 cores, 留 3 给 OS/IO).
N_WORKERS = min(10, max(1, (os.cpu_count() or 4) - 3))
# chunk size: 每 worker 一次拿 N 条, 减少 IPC 开销
CHUNK_SIZE = 5_000

# === Outputs ===
PRODUCT_ATTRS_JSON = REPO_ROOT / "result" / "01_attribute_extraction" / "product_attributes.json"

# === Step 1 — extract_attrs 参数 ===
MAX_STR_LEN = 200
TOP_LEVEL_NUMERIC_FIELDS = ("average_rating", "rating_number", "price")

# 用户指令 2026-09-04: 源头过滤含数字的 attr (单 attr 维度).
EXCLUDE_NUMERIC_VALUES = True

# 用户指令 2026-09-04: 过滤数字后, attrs 数 < MIN_ATTRS 的 ASIN 整条跳过.
MIN_ATTRS_PER_ASIN = 5

# 用户指令 2026-09-04: 短语 attr 过滤 — value word 数 > MAX_VALUE_WORDS
# 视为短语跳过 (>3 words 的 value 多为 Feature 整句描述, 难被 LLM 自然注入 query).
MAX_VALUE_WORDS = 3

# 用户指令 2026-09-04: yes/no 单值 attr 过滤 — value 是 yes/no/true/false
# 的 attr 是二元 metadata (e.g. 'Is Discontinued By Manufacturer: Yes'),
# 对 LLM 生成自然 query 无信息量, 跳过.
_YES_NO_VALUES = {"yes", "no", "true", "false"}


def _has_ascii_alpha(s: str) -> bool:
    """value/key 是否至少含一个 ASCII 英文字母."""
    return any(c.isascii() and c.isalpha() for c in str(s))


def _is_non_cjk(s: str) -> bool:
    """value/key 不含 CJK (中文/日文/韩文) 字符. accented (ü/é/ñ) 允许."""
    return not any(
        0x4E00 <= ord(c) <= 0x9FFF  # CJK Unified Ideographs
        or 0x3040 <= ord(c) <= 0x309F  # Hiragana
        or 0x30A0 <= ord(c) <= 0x30FF  # Katakana
        or 0xAC00 <= ord(c) <= 0xD7AF  # Hangul
        for c in str(s)
    )

# === select_top_attrs 参数 (供下游 cohort construction 共用) ===
MAX_ATTRS = 5
MAX_ATTR_VALUE_LEN = 100
EXCLUDE_NUMERIC_ATTRS = True
_NUMERIC_KEYWORDS = {"price", "average rating", "rating number", "item weight",
                     "item model number", "date first available",
                     "package dimensions", "product dimensions",
                     "minimum weight recommendation",
                     "maximum weight recommendation",
                     "batteries required", "is discontinued by manufacturer"}
# 用户指令 2026-08-29: 非语义属性类型一并过滤 (和数值属性同等处理)。
# 这些 key 描述的是商品 metadata (路由/分类/产地/计数/日期/排名),
# 而非商品本身的语义特征 (品牌/颜色/材质/尺寸等)。LLM 用这些 attrs 生成
# query 时无意义 (例如 "Main Category: Baby Products" 对 query 无信息量)。
_NON_SEMANTIC_KEYWORDS = {
    # 路由/分类 (非特征)
    "main category", "department",
    # 产地 (metadata)
    "country", "country of origin", "country/region of origin",
    # 计数 (非特征)
    "number of items", "number of pieces", "unit count",
    # 日期/排名 (metadata)
    "date", "date listed", "best sellers rank",
}
ATTR_PRIORITY = [
    "Brand", "Main Category", "Item model number", "Manufacturer",
    "Color", "Material", "Material Type", "Fabric Type", "Frame Material",
    "Item Weight", "Product Dimensions", "Size", "Style", "Pattern", "Theme",
    "Age Range (Description)", "Special Feature", "Target gender",
    "Batteries required", "Number Of Items", "Is Discontinued By Manufacturer",
    "Date First Available", "Country/Region of origin", "Country of Origin",
    "Price", "Average Rating", "Rating Number",
]


def log(msg: str) -> None:
    print(f"[extract_product_attrs] {msg}", flush=True)


def has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in s)


def extract_attrs(d: dict) -> dict:
    """Extract structured attributes from a single product meta dict.

    Supports two meta formats:
    - New (2023): features (list) + details (dict)
    - Old (2018): top-level fields only

    Returns dict {field_name: string_value}. No numeric value filtering
    here — that is select_top_attrs()'s job (called downstream).
    """
    out: dict = {}

    # New 2023 format: details dict
    details = d.get("details")
    if isinstance(details, dict):
        for k, v in details.items():
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s or len(s) > MAX_STR_LEN:
                continue
            out[k] = s

    # New 2023 format: features list
    features = d.get("features")
    if isinstance(features, list):
        for i, v in enumerate(features):
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s or len(s) > MAX_STR_LEN:
                continue
            # features are anonymous; name them by index
            out[f"Feature {i+1}"] = s

    # Old 2018 format: top-level fields (also used as fallback)
    for field in ("brand", "main_category"):
        v = d.get(field)
        if isinstance(v, str) and v.strip():
            name = "Brand" if field == "brand" else "Main Category"
            if name not in out:
                out[name] = v.strip()

    for field in TOP_LEVEL_NUMERIC_FIELDS:
        v = d.get(field)
        if v is None or v == "":
            continue
        out[field.replace("_", " ").title().replace(" ", " ")] = v

    # 用户指令 2026-09-04: 源头过滤含数字的 (k, v) (单 attr 维度)
    if EXCLUDE_NUMERIC_VALUES:
        out = {k: v for k, v in out.items() if not has_digit(str(v))}

    # 用户指令 2026-09-04: value 去重 (大小写不敏感), 同一 value 已出现过则
    # 跳过新的 (k, v) — 保留首次出现的 key
    seen_values: set[str] = set()
    deduped: dict = {}
    for k, v in out.items():
        v_key = str(v).strip().lower()
        if v_key in seen_values:
            continue
        seen_values.add(v_key)
        deduped[k] = v
    out = deduped

    # 用户指令 2026-09-04: 短语过滤 — value word 数 > MAX_VALUE_WORDS 视为短语跳过
    out = {
        k: v
        for k, v in out.items()
        if len(re.findall(r"\S+", str(v))) <= MAX_VALUE_WORDS
    }

    # 用户指令 2026-09-04: yes/no 二元值过滤
    out = {
        k: v
        for k, v in out.items()
        if str(v).strip().lower() not in _YES_NO_VALUES
    }

    # 用户指令 2026-09-04: 非英文 key/value 过滤 — key 和 value 都至少
    # 需含一个 ASCII 英文字母 (CJK / 纯数字 / 纯符号全部跳过)
    out = {
        k: v
        for k, v in out.items()
        if _has_ascii_alpha(k) and _has_ascii_alpha(v) and _is_non_cjk(v)
    }

    # 用户指令 2026-09-04: 多值取首段 — value 含逗号时只保留第一段 (e.g.
    # 'Lightweight, Breathable' → 'Lightweight'), 取首段后重新过 yes/no /
    # 短语 / 英文过滤
    cleaned: dict = {}
    for k, v in out.items():
        if "," in str(v):
            first = str(v).split(",", 1)[0].strip()
            if not first:
                continue
            # 重新过严过滤
            if (
                _has_ascii_alpha(k)
                and _has_ascii_alpha(first)
                and _is_non_cjk(first)
                and first.lower() not in _YES_NO_VALUES
                and len(re.findall(r"\S+", first)) <= MAX_VALUE_WORDS
            ):
                cleaned[k] = first
        else:
            cleaned[k] = v
    out = cleaned

    return out


def select_top_attrs(asin_attrs: dict, max_n: int = MAX_ATTRS) -> dict:
    """Select top-N attributes for an ASIN, excluding numeric values.

    Used by both Stage 1 vLLM prompt construction and downstream cohort
    selection. Returns dict {field_name: string_value}, with priority
    order ATTR_PRIORITY first, then any remaining fields.

    Numeric exclusion: any value with digits, or any key in _NUMERIC_KEYWORDS,
    is dropped. Implements 用户指令 2026-08-28: "属性数值过滤".

    Args:
        asin_attrs: dict of field_name -> string_value from extract_attrs()
        max_n: maximum number of attrs to return (Stage 1 = 5, Stage 4 cohort = 4)
    """
    def _skip(k: str, s: str) -> bool:
        if not s or len(s) > MAX_ATTR_VALUE_LEN:
            return True
        k_low = k.lower()
        # 用户指令 2026-08-29: 非语义属性类型过滤 (与数值属性同等处理)
        if any(nk in k_low for nk in _NON_SEMANTIC_KEYWORDS):
            return True
        if EXCLUDE_NUMERIC_ATTRS:
            if any(nk in k_low for nk in _NUMERIC_KEYWORDS):
                return True
            if has_digit(s):
                return True
        return False

    out: dict = {}
    used: set[str] = set()
    for k in ATTR_PRIORITY:
        v = asin_attrs.get(k)
        if not v:
            continue
        s = str(v).strip()
        if _skip(k, s):
            continue
        out[k] = s
        used.add(k)
        if len(out) >= max_n:
            return out
    for k, v in asin_attrs.items():
        if k in used:
            continue
        if v is None:
            continue
        s = str(v).strip()
        if _skip(k, s):
            continue
        out[k] = s
        if len(out) >= max_n:
            break
    return out


def _process_one(raw_line: str):
    """Worker 入口: 1 行 metadata → (asin, attrs) 或 None.

    必须 module-level + 纯函数, 让 multiprocessing.Pool 能 pickle.
    返回值用 None 标记 skip (无 asin / attrs < MIN),主进程 reduce 时忽略。
    """
    line = raw_line.strip()
    if not line:
        return None
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        return None
    asin = d.get("parent_asin") or d.get("asin")
    if not asin:
        return None
    attrs = extract_attrs(d)
    if len(attrs) < MIN_ATTRS_PER_ASIN:
        return None
    return (asin, attrs)


def _process_chunk(chunk: list[str]):
    """Worker 入口 (chunked): N 行 metadata → list of (asin, attrs).

    比 _process_one 减少 IPC 次数, 主进程 imap_unordered 按 chunk 喂。
    """
    out = []
    for line in chunk:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        asin = d.get("parent_asin") or d.get("asin")
        if not asin:
            continue
        attrs = extract_attrs(d)
        if len(attrs) < MIN_ATTRS_PER_ASIN:
            continue
        out.append((asin, attrs))
    return out


def step1_extract_attrs() -> dict[str, dict]:
    log("=== Step 1: extract_attrs (multiprocessing) ===")
    log(f"  workers={N_WORKERS}, chunk_size={CHUNK_SIZE}")

    product_attrs: dict[str, dict] = {}
    n_total = 0
    n_with_attrs = 0
    n_skipped_few_attrs = 0
    n_with_details = 0
    n_top_level_only = 0
    field_counter: dict[str, int] = {}

    # 主进程先一次性读取所有行到内存 (1.5 GB), 避免 worker fork 后
    # 多进程争抢同一个文件描述符 (POSIX 行为).
    _open = gzip.open if str(META_GZ).endswith('.gz') else open
    mode = 'rt' if _open is gzip.open else 'r'
    with _open(META_GZ, mode) as f:
        all_lines = f.readlines()
    n_total = len(all_lines)
    log(f"  read {n_total} lines from metadata; dispatching to {N_WORKERS} workers")

    # chunked dispatch
    chunks = [
        all_lines[i : i + CHUNK_SIZE]
        for i in range(0, len(all_lines), CHUNK_SIZE)
    ]
    n_skipped_few_attrs = 0
    with Pool(processes=N_WORKERS) as pool:
        for sub in pool.imap_unordered(_process_chunk, chunks):
            for asin, attrs in sub:
                if attrs:
                    product_attrs[asin] = attrs
                    n_with_attrs += 1
                    has_details = any(
                        k not in ("Brand", "Main Category", "Average Rating",
                                  "Rating Number", "Price")
                        for k in attrs
                    )
                    if has_details:
                        n_with_details += 1
                    else:
                        n_top_level_only += 1
                    for k in attrs:
                        field_counter[k] = field_counter.get(k, 0) + 1

    PRODUCT_ATTRS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PRODUCT_ATTRS_JSON.write_text(
        json.dumps(product_attrs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"  total metadata products={n_total}, with attrs={n_with_attrs}")
    log(f"    details-based={n_with_details}, top-level only={n_top_level_only}")
    log(f"    distinct attr fields={len(field_counter)}")
    log(f"    skipped ASINs (filtered attrs <{MIN_ATTRS_PER_ASIN})={n_total - n_with_attrs}")
    log(f"  wrote → {PRODUCT_ATTRS_JSON}")
    return product_attrs


def main() -> None:
    log("=== extract_product_attrs.py — Step 1 only ===")
    step1_extract_attrs()
    log("=== DONE ===")


if __name__ == "__main__":
    main()
