"""构建并清洗 `result/11_syntactic_evaluation/asin_to_doc.json`。

本脚本从 `syntax_subspace_retrieval_unified.py` 中分离出来（2026-09-22），
承担两件事：
  1. 把 `data/meta_Baby_Products_2023.jsonl` 的每条记录组装成结构化文档
     （product / brand / category / dimensions / weight / description / features）
  2. 对组装好的 doc 字符串施加 4 层清洗：
       Tier 1：营销话术剥离（Brand Story / From the Manufacturer / See more /
               Add to Cart / Worry-free / Customer Service 等纯模板噪声）
       Tier 2：内容去重（dimensions/weight 在 description 中重复、product 标题
               在 description / features 中重复、features 首条 == product 全名）
       Tier 3：格式规范化（features 编号重排、ALL CAPS 行转 title case、URL /
               hashtag 清理）
       Tier 5：编码修复（mojibake、混合非英文字符）

清洗规则的设计目标：
  - 不引入 fallback（Rule 7）：规则不命中 → 保留原文，不替换
  - 不修改源数据：META_FILE 只读
  - 缓存签名包含 META_FILE mtime+size + n_asins，源文件变化自动失效
  - 输出覆盖 `result/11_syntactic_evaluation/asin_to_doc.json`，下游
    BM25/SPLADE/MiniLM/MPNet/BGE/GTE/ColBERTv2 7 个检索器自动消费清洗后版本

被引用方：
  - 11_syntactic_evaluation/syntax_subspace_retrieval_unified.py（main + dense_retrieve）
  - 12_typo_evaluation / 13_syntactic_rerank / 14_typo_rerank 均消费同一份 json

运行方式（无 CLI 参数，按 Rule 4）：
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python common/build_asin_to_doc.py
"""
from __future__ import annotations

import collections
import hashlib
import html
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterable

# ===========================================================================
# PATHS (硬编码，按 Rule 4)
# ===========================================================================
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
# 用户指令 2026-09-23: data 目录从 REPO_ROOT/data 迁移到 hj82 同名 data 目录.
DATA_DIR = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data")
META_FILE = DATA_DIR / "meta_Baby_Products_2023.jsonl"
ASIN_TO_DOC_CACHE = REPO_ROOT / "result/11_syntactic_evaluation/asin_to_doc.json"
# 用户指令 2026-09-23: 3 个 category 各自一份 (Baby / Musical / Video_Games),
# main() 改为串行跑 3 个 domain, 产物写到 result/11_syntactic_evaluation/<subdir>/.
CATEGORY_INPUTS = [
    # (category_key, subdir)
    ("Baby",                "baby"),
    ("Musical_Instruments", "musical"),
    ("Video_Games",         "video_games"),
]

# ===========================================================================
# LOG
# ===========================================================================
def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ===========================================================================
# 基础字符清理（与原文件 _clean_doc_text 等价）
# ===========================================================================
_ALLOWED_PUNCTUATION = set("-_/.,:%&+()[]'\"#")


def _clean_doc_text(value) -> str:
    """基础字符清理：HTML escape、emoji、非字母数字符号 → 空格。

    保留 # / & / + 等产品文档常见符号（features 用 '#N:' 编号、category 用 '/'
    分隔、dimensions 用 'x' 分隔）。
    """
    if isinstance(value, list):
        value = " ".join(str(x) for x in value if x is not None)
    elif not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = re.sub(r"<[^>]*>", " ", value)
    value = value.replace("\r", " ").replace("\n", " ")
    cleaned = []
    for char in value:
        category = unicodedata.category(char)
        if category.startswith("C") or category.startswith("So"):
            cleaned.append(" ")
        elif char.isalnum() or char.isspace() or char in _ALLOWED_PUNCTUATION:
            cleaned.append(char)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split()).strip()


# ===========================================================================
# Tier 5：编码修复（轻量 ftfy 替代，不引第三方库）
# ===========================================================================
# 常见 mojibake 字节残留（UTF-8/Latin-1/CP1252 互转错误）
_MOJIBAKE_MAP = {
    "\u00c2\u00a0": " ",       # NBSP via UTF-8 of Latin-1 0xA0
    "\u00c2\u2019": "'",       # curly quote
    "\u00c2\u201c": '"',
    "\u00c2\u201d": '"',
    "\u00c2\u00ae": "(R)",     # ®
    "\u00c2\u2122": "(TM)",    # ™
    "\u00c2\u00b0": "deg",     # °
    "\u00c2\u00bd": "1/2",
    "\u00c2\u00bc": "1/4",
    "\u00c2\u00be": "3/4",
    "\u00e2\u0080\u0099": "'",  # ’
    "\u00e2\u0080\u009c": '"',  # “
    "\u00e2\u0080\u009d": '"',  # ”
    "\u00e2\u0080\u0093": "-",  # –
    "\u00e2\u0080\u0094": "-",  # —
    "\u00e2\u0080\u00a6": "...",# …
    "\u00c3\u00a9": "e",       # é
    "\u00c3\u00a8": "e",       # è
    "\u00c3\u00a0": "a",       # à
    "\u00c3\u00ae": "i",       # î
    "\u00c3\u00b1": "n",       # ñ → n
    "\u00ef\u00bf\u00bd": "",  # U+FFFD replacement
}
_MOJIBAKE_PATTERN = re.compile("|".join(re.escape(k) for k in _MOJIBAKE_MAP.keys()))

# 孤立 mojibake 字符（Latin-1/CP1252 字节被错误解码为 UTF-8）
_ISOLATED_MOJIBAKE_RE = re.compile(r"[\u00c2\u00e2](?!\S)")
# 在单词内部的孤立 â（剩余 mojibake 残留，非源数据）；保留为 ''

# 非英文字符（CJK + 西文重音字母 + 西里尔 + 希腊 + 阿拉伯 + 希伯来）
_NON_EN_RE = re.compile(
    r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af"
    r"\u0400-\u04ff\u0370-\u03ff\u0590-\u05ff"
    r"\u0600-\u06ff]"
)


def _fix_encoding(text: str) -> str:
    """修复 mojibake + 剥离混合的非英文字符。

    mojibake 是确定性的字符替换；混合 CJK 等字符通常是数据采集错误，整段剥离。
    """
    if not text:
        return text
    fixed = _MOJIBAKE_PATTERN.sub(lambda m: _MOJIBAKE_MAP[m.group(0)], text)
    fixed = _ISOLATED_MOJIBAKE_RE.sub("", fixed)
    fixed = _NON_EN_RE.sub(" ", fixed)
    return re.sub(r"\s+", " ", fixed).strip()


# ===========================================================================
# Tier 1：营销话术剥离
# ===========================================================================
# 这些模式描述的子句/片段在产品描述中无信息价值。
# 匹配方式：每个模式是一个 regex，匹配 description / features 中的子句。
# 设计原则：仅匹配"成句"或"独立短语"，避免误删真实产品信息。
_MARKETING_SENTENCE_PATTERNS = [
    # UI / 模板 header
    r"^\s*Product Description\s*[:.\-]?\s*$",
    r"^\s*Product Details\s*[:.\-]?\s*$",
    r"^\s*From the Manufacturer\s*[:.\-]?\s*$",
    r"^\s*Brand Story\s*(?:By\s+[\w\s&\-'.]+)?\s*[:.\-]?\s*$",
    r"^\s*See more\s*\.?\s*$",
    r"^\s*Read more\s*\.?\s*$",
    # CTA / 销售话术
    r"^\s*Click(?:\s+[\"'])?Add to Cart(?:\s+[\"'])?\s+now\.?\s*$",
    r"^\s*Add to Cart(?:\s+now)?\.?\s*$",
    r"^\s*Order\s+Now(?:\s+Without\s+Risk)?\.?\s*$",
    r"^\s*Buy it now\.?\s*$",
    r"^\s*Don'?t hesitate(?:\s+any\s+more)?(?:,)?\s*buy it now\.?\s*$",
    r"^\s*Make these?\s+[\w\s]+part of your\s+[\w\s]+(?:daily\s+life|essentials?)\.?\s*$",
    # 售后承诺
    r"^\s*24[\- ]?hour\s+friendly\s+customer\s+service\.?\s*$",
    r"^\s*24[\- ]?h\s+(?:friendly\s+)?customer\s+service\.?\s*$",
    r"^\s*We offer\s+24[\- ]?h(?:our)?\s+(?:friendly\s+)?(?:customer\s+)?service\.?\s*$",
    r"^\s*Worry[\- ]?free\s+after[\- ]?sales\.?\s*$",
    r"^\s*We (?:will )?(?:try|are committed) to (?:provide|solve|help).*?(?:customer service|issue|problem).*?\.?\s*$",
    r"^\s*For any reason you are not 100%\s+satisfied.*?\.?\s*$",
    r"^\s*If you (?:have any|are not).*?(?:question|issue|problem).*?please.*?\.?\s*$",
    r"^\s*Please do not hesitate to contact us.*?\.?\s*$",
    r"^\s*(?:Contact|Reach out to) us (?:at once|right away|now).*?\.?\s*$",
    r"^\s*We (?:offer|provide) 24h? service.*?\.?\s*$",
    # 营销泛化
    r"^\s*It(?:'s| is) (?:really |so )?(?:a )?(?:great|good|nice|perfect|amazing) (?:gift|choice|option)(?:\s+for\s+(?:the\s+)?[\w\s]+)?\.?\s*$",
    r"^\s*The (?:perfect|ideal|best) (?:gift|choice|option) for\s+.*?\.?\s*$",
    r"^\s*Makes? (?:a )?(?:great|perfect|ideal) (?:gift|present)\.?\s*$",
    # 全大写标题行（marketing slogan）：以全大写词开头
    r"^\s*([A-Z][A-Z&\-]{2,}\s+){3,}[A-Z][A-Z&\-:]*\s*\.?\s*$",
    # 子串级（必须整段匹配才删）
]
_MARKETING_SUBSTRING_PATTERNS = [
    (r"\bBrand Story\s+By\s+\S+(?:\s+\S+){0,3}\.?\s*", " "),
    (r"\bProduct Description\s*[:.\-]?\s*", " "),
    (r"\bFrom the Manufacturer\s*[:.\-]?\s*", " "),
    (r"\bSee more\.?\s*", " "),
    (r"\bRead more\.?\s*", " "),
    (r"Click(?:\s+[\"'])?Add to Cart(?:\s+[\"'])?\s+now\.?", " "),
    (r"Order Now Without Risk\.?", " "),
    (r"\bAdd to Cart\s+now\.?", " "),
    (r"\b24[\- ]?hour friendly customer service\.?", " "),
    (r"\bWorry[\- ]?free after[\- ]?sales\.?", " "),
]


def _strip_marketing_noise(text: str) -> str:
    """Tier 1：删除纯营销话术子句/片段。"""
    if not text:
        return text
    # Step 1：按子句级删除整句（sentence-level）
    # 句子切分：以 . ! ? 加 换行/大写字母起首 作为边界
    sentences = re.split(r"(?<=[.!?])\s+|(?<=\.)\s*\n", text)
    kept = []
    for s in sentences:
        s_strip = s.strip()
        if not s_strip:
            continue
        is_marketing = False
        for pat in _MARKETING_SENTENCE_PATTERNS:
            if re.match(pat, s_strip, re.IGNORECASE if pat.startswith(r"^\s*[A-Z]") else 0):
                is_marketing = True
                break
        if not is_marketing:
            kept.append(s)
    text = " ".join(kept)
    # Step 2：子串级删除（部分话术不在句首）
    for pat, repl in _MARKETING_SUBSTRING_PATTERNS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    # Step 3：URL 清理
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bwww\.\S+", " ", text)
    # 收尾空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===========================================================================
# Tier 2：内容去重
# ===========================================================================
def _extract_numeric_tokens(text: str) -> list[str]:
    """提取文本中的数字/单位 token（用于 dimensions/weight 重复检测）。"""
    return re.findall(r"\d+(?:\.\d+)?", text)


def _dedup_repeated_content(desc: str, dims: str, wt: str, product_title: str) -> str:
    """Tier 2：从 description 中删除与 dimensions/weight 数值重复的子句，
    以及首句如果是 product 标题全文。
    """
    if not desc:
        return desc
    sentences = re.split(r"(?<=[.!?])\s+", desc)
    kept = []
    dim_nums = _extract_numeric_tokens(dims) if dims else []
    wt_nums = _extract_numeric_tokens(wt) if wt else []
    for s in sentences:
        s_strip = s.strip()
        if not s_strip:
            continue
        # 检查：是否本句主要在复述 dimensions 数字？
        nums_in_s = _extract_numeric_tokens(s_strip)
        if dim_nums and len(nums_in_s) >= 2:
            overlap = sum(1 for n in nums_in_s if n in dim_nums)
            # 本句数字多数(≥2且覆盖超 80%)与 dimensions 数字重叠 → 复述尺寸
            if overlap >= max(2, int(len(nums_in_s) * 0.8)):
                continue
        # 检查：是否本句主要在复述 weight 数字？
        if wt_nums and len(nums_in_s) == 1:
            if nums_in_s[0] in wt_nums and len(s_strip) < 60:
                continue
        # 检查：是否本句 == product 标题全文（重复）
        if product_title and len(product_title) >= 20:
            pt_lower = product_title.lower()
            s_lower = s_strip.lower()
            # 句子包含 product_title 且长度不超过 title 的 1.5倍 → 复述
            if pt_lower in s_lower and len(s_strip) <= int(len(product_title) * 1.5):
                continue
        kept.append(s)
    desc = " ".join(kept)
    return re.sub(r"\s+", " ", desc).strip()


def _dedup_features_title(features: list[str], product_title: str) -> list[str]:
    """Tier 2：若 features[0] 与 product 标题相同或高度相似，删除该条。"""
    if not features or not product_title or len(product_title) < 15:
        return features
    pt_lower = product_title.lower()
    first = features[0].strip()
    first_lower = first.lower()
    # features[0] 包含 product_title 且长度不超过 1.5x → 复述
    if pt_lower in first_lower and len(first) <= int(len(product_title) * 1.5):
        return features[1:]
    # features[0] 与 product_title 完全相同
    if first_lower == pt_lower:
        return features[1:]
    return features


# ===========================================================================
# Tier 3：格式规范化
# ===========================================================================
# 已知产品缩写 / 商标 / 材质缩写，转 title case 时应保留全大写
_KEEP_UPPER = {
    "BPA", "CPSIA", "FDA", "LFGB", "PVC", "PEVA", "PE", "PU", "PP",
    "USA", "UK", "EU", "USD", "TOG", "NICU", "PVC", "TPU", "TPE",
    "DIY", "OEM", "ODM", "SGS", "CE", "ROHS", "ROHS",
    "USB", "AC", "DC", "LED", "LCD", "HD", "GPS",
    "XL", "XXL", "XS", "SM", "MD", "LG",
    "DIY", "BPA", "SIDS", "HALO", "ECR", "ECR4KIDS",
    "DCDC", "MICROFIBER", "MICROFIBRE", "POLYESTER",
    "NYLON", "COTTON", "SILICONE", "ALUMINUM", "ALUMINIUM",
    "STAINLESS", "STEEL", "WOOD", "METAL", "PLASTIC",
    "OZ", "ML", "LB", "LBS", "KG", "G", "CM", "MM", "IN", "INCH", "INCHES",
    "FT", "FEET", "YARD", "YDS",
}


def _normalize_caps_line(line: str) -> str:
    """Tier 3.1：检测行首前 N 个词是否全大写（marketing 标题型），若是 → 转 title case。

    阈值：前 5 个字母词中 uppercase >= 4 个 且长度 > 30 字符。
    """
    if len(line) < 30:
        return line
    words = line.split()
    if len(words) < 3:
        return line
    # 取行首前 5 个有字母的词
    head_words = []
    for w in words:
        if any(c.isalpha() for c in w):
            head_words.append(w)
        if len(head_words) >= 5:
            break
    if len(head_words) < 3:
        return line
    # 检查 head_words 中是否大多数 uppercase
    upper_cnt = sum(1 for w in head_words if w.isupper() or sum(1 for c in w if c.isupper()) >= len([c for c in w if c.isalpha()]) - 1)
    if upper_cnt < 3:
        return line
    # 转为 title case：保留缩写词大写
    out = []
    for w in words:
        stripped = w.strip(".,:;()[]'\"")
        if stripped.upper() in _KEEP_UPPER:
            out.append(w.upper())
        elif any(c.isdigit() for c in w):
            out.append(w)
        elif w.isupper() and len(w) <= 4:
            out.append(w)
        else:
            out.append(w.capitalize() if w.isupper() else w)
    return " ".join(out)


def _renumber_features(features: list[str]) -> list[str]:
    """Tier 3.2：features 重编号（按当前顺序 1..N），原 '#N:' 前缀会被替换。"""
    cleaned = [f.strip() for f in features if f and f.strip()]
    return [f"#{i}: {feat}" for i, feat in enumerate(cleaned, 1)]


# ===========================================================================
# 字段级清洗管道
# ===========================================================================
def _clean_description(desc: str, dims: str, weight: str, product_title: str) -> str:
    """对 description 字段施加 全部 4 层清洗。"""
    if not desc:
        return desc
    desc = _strip_marketing_noise(desc)
    # Tier 5
    desc = _fix_encoding(desc)
    # Tier 2：与 dimensions/weight 数字重复 + product 标题重复 → 子句级删除
    desc = _dedup_repeated_content(desc, dims, weight, product_title)
    # Tier 3.1：ALL CAPS 行规范化
    # description 已经被规范化成单行（build_meta_corpus 时已 \n → space）
    desc = _normalize_caps_line(desc)
    return desc


def _clean_features(features: list[str], product_title: str) -> list[str]:
    """对 features 列表施加清洗：Tier 1 (marketing) + Tier 2 (title dup) + Tier 3 (renumber)。"""
    cleaned = []
    for f in features:
        if not f:
            continue
        c = _strip_marketing_noise(f)
        c = _fix_encoding(c)
        if c:
            cleaned.append(c)
    # Tier 2：features[0] 是否与 product 标题重复
    cleaned = _dedup_features_title(cleaned, product_title)
    # Tier 3.2：重编号
    return _renumber_features(cleaned)


def _clean_simple_field(value: str) -> str:
    """对 brand / category / dimensions / weight 等简单字段做 Tier 5 + Tier 1 子串清理。"""
    if not value:
        return value
    value = _strip_marketing_noise(value)
    value = _fix_encoding(value)
    return re.sub(r"\s+", " ", value).strip()


# ===========================================================================
# 缓存签名（与 syntax_subspace_retrieval_unified.py 兼容）
# ===========================================================================
def _asin_content_sha1(asin_to_doc: dict[str, str]) -> str:
    """Compute sha1 over asin_to_doc sorted contents (40 hex chars)."""
    content_h = hashlib.sha1()
    keys = sorted(asin_to_doc.keys())
    log(f"  hashing {len(keys)} ASIN documents")
    for index, k in enumerate(keys, start=1):
        content_h.update(k.encode())
        content_h.update(b"\x00")
        content_h.update(asin_to_doc[k].encode("utf-8", errors="ignore"))
        content_h.update(b"\x00")
        if index % 100_000 == 0 or index == len(keys):
            log(f"  hash progress: {index}/{len(keys)} ASINs")
    return content_h.hexdigest()


def _corpus_signature(asin_to_doc: dict[str, str] | None = None,
                     meta_file: Path | None = None) -> str:
    """Fast corpus fingerprint。

    公式：`sha1(meta_mtime+size + "|" + content_sha1)[:16]`

    - meta_mtime+size：META_FILE 状态变化失效
    - content_sha1：清洗后内容变化失效
    - 16 hex 截断：紧凑指纹
    - 2026-09-23：修复 self-reference bug——旧版读 .sig 文件内容作 ad_sig
      输入指纹，但 .sig 内容本身是上一次的指纹（含 meta mtime+size），导致
      cache check 与 build 末尾的两次算法输入不一致，永远 mismatch 死循环。
      新版要求 .sig 第一行是 content_sha1（40 hex），cache check 直接读第一行。
    """
    mf = Path(meta_file) if meta_file is not None else META_FILE
    st = mf.stat()
    if asin_to_doc is not None:
        content_sha1 = _asin_content_sha1(asin_to_doc)
    elif mf == META_FILE:
        sig_path = ASIN_TO_DOC_CACHE.with_suffix(ASIN_TO_DOC_CACHE.suffix + ".sig")
        if sig_path.exists():
            # .sig 第一行 = content_sha1 (40 hex)，其余为兼容历史注释行
            content_sha1 = sig_path.read_text(encoding="utf-8").splitlines()[0].strip()
        else:
            content_sha1 = "0" * 40
    else:
        content_sha1 = "0" * 40
    h = hashlib.sha1()
    h.update(f"{mf.name}|mtime={int(st.st_mtime)}|size={st.st_size}".encode())
    h.update(b"|content_sha1=")
    h.update(content_sha1.encode())
    return h.hexdigest()[:16]


def _sig_path_for(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".sig")


# ===========================================================================
# 主入口：build_meta_corpus
# ===========================================================================
def build_meta_corpus(force: bool = False) -> dict[str, str]:
    """构建并清洗结构化产品文档，缓存到 ASIN_TO_DOC_CACHE。

    Args:
        force: 强制重建（忽略 sig 命中）。默认 False → sig 命中时直接读缓存。

    Returns:
        dict[asin, doc_text]。

    Raises:
        FileNotFoundError: META_FILE 不存在（Rule 7：不静默 fallback）
        ValueError: META_FILE 中记录缺少必填字段（Rule 7：不静默 fallback）
    """
    sig_path = _sig_path_for(ASIN_TO_DOC_CACHE)
    if not force and ASIN_TO_DOC_CACHE.exists() and sig_path.exists():
        current_sig = _corpus_signature(meta_file=META_FILE)
        # .sig 文件格式：第一行 content_sha1 (40 hex)，第二行 fingerprint (16 hex)。
        # cache check 比的是 fingerprint（第二行），与 _corpus_signature 输出对齐。
        cached_sig = sig_path.read_text(encoding="utf-8").splitlines()
        cached_fingerprint = cached_sig[1].strip() if len(cached_sig) >= 2 else cached_sig[0].strip()
        if cached_fingerprint == current_sig:
            t0 = time.time()
            asin_to_doc = json.load(open(ASIN_TO_DOC_CACHE, encoding="utf-8"))
            log(f"  ✓ asin_to_doc cache hit ({len(asin_to_doc)} ASINs, "
                f"sig={current_sig}, {time.time() - t0:.2f}s)")
            return asin_to_doc
        log(f"  ⚠ asin_to_doc sig mismatch (cached={cached_fingerprint}, "
            f"current={current_sig}), rebuilding...")
    elif not force and ASIN_TO_DOC_CACHE.exists() and not sig_path.exists():
        log(f"  ⚠ asin_to_doc sig missing, rebuilding...")
    else:
        log(f"  ⚠ asin_to_doc cache missing or force=True, building...")

    if not META_FILE.exists():
        raise FileNotFoundError(
            f"META_FILE not found: {META_FILE}. "
            f"Required for Stage 11/12/13/14 retrieval (Rule 7: no fallback)."
        )

    asin_to_doc: dict[str, str] = {}
    n_records = 0
    n_skip_no_title = 0
    n_cleaned_features = 0
    n_cleaned_desc = 0
    n_dropped_title_dup = 0
    log(f"  loading metadata from {META_FILE}")
    t0 = time.time()
    with open(META_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_records += 1
            if n_records % 50_000 == 0:
                log(f"  metadata progress: {n_records:,} records, "
                    f"{len(asin_to_doc):,} ASIN docs ({time.time() - t0:.1f}s)")
            asin = _clean_doc_text(r.get("parent_asin"))
            if not asin:
                raise ValueError(
                    f"Record #{n_records} missing required field 'parent_asin'"
                )
            details = r.get("details")
            if not isinstance(details, dict):
                raise ValueError(
                    f"Required details field is not an object for ASIN {asin}"
                )
            raw_title = r.get("title")
            # title 缺失为预期内（数据源 217k 条中有 14 条，占 0.006%）：跳过记录
            if not raw_title:
                n_skip_no_title += 1
                continue
            categories = r.get("categories")
            if not isinstance(categories, list):
                raise ValueError(
                    f"Required categories field is not a list for ASIN {asin}"
                )
            raw_features = r.get("features")
            if not isinstance(raw_features, list):
                raise ValueError(
                    f"Required features field is not a list for ASIN {asin}"
                )

            # 字段级清洗
            title = _clean_doc_text(raw_title)
            brand = _clean_simple_field(_clean_doc_text(details.get("Brand")))
            category = " / ".join(
                _clean_simple_field(_clean_doc_text(x))
                for x in categories if x is not None
            )
            dimensions = _clean_simple_field(_clean_doc_text(details.get("Product Dimensions")))
            weight = _clean_simple_field(_clean_doc_text(details.get("Item Weight")))
            description = _clean_doc_text(r.get("description"))
            if description:
                new_desc = _clean_description(description, dimensions, weight, title)
                if new_desc != description:
                    n_cleaned_desc += 1
                description = new_desc
            raw_feature_values = [_clean_doc_text(x) for x in raw_features]
            raw_feature_values = [x for x in raw_feature_values if x]
            features = _clean_features(raw_feature_values, title)
            if len(features) < len(raw_feature_values):
                # features 被清洗掉至少一条
                if features and features[0] != f"#1: {raw_feature_values[0]}":
                    n_dropped_title_dup += 1

            # 组装
            lines = [f"product: {title}"]
            if brand:
                lines.append(f"brand: {brand}")
            if category:
                lines.append(f"category: {category}")
            if dimensions:
                lines.append(f"dimensions: {dimensions}")
            if weight:
                lines.append(f"weight: {weight}")
            if description:
                lines.append(f"description: {description}")
            if features:
                lines.append("features:")
                lines.extend(features)
            asin_to_doc[asin] = "\n".join(lines)
    log(f"  loaded {len(asin_to_doc)} ASIN docs from {n_records} records "
        f"({time.time() - t0:.1f}s, skipped_no_title={n_skip_no_title})")
    log(f"  cleaning stats: description_cleaned={n_cleaned_desc}, "
        f"features_title_dedup={n_dropped_title_dup}")
    log(f"  writing {len(asin_to_doc):,} ASIN documents to {ASIN_TO_DOC_CACHE}")

    ASIN_TO_DOC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(ASIN_TO_DOC_CACHE, "w", encoding="utf-8") as f:
        json.dump(asin_to_doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    # 2026-09-23: .sig 第一行写 content_sha1（40 hex），第二行写 fingerprint（16 hex）。
    # 下次 cache check 读 .sig 第一行 → inline 算 fingerprint → 必 cache hit。
    content_sha1 = _asin_content_sha1(asin_to_doc)
    current_sig = _corpus_signature(asin_to_doc=asin_to_doc, meta_file=META_FILE)
    sig_path = _sig_path_for(ASIN_TO_DOC_CACHE)
    sig_path.write_text(f"{content_sha1}\n{current_sig}\n")
    log(f"  cached → {ASIN_TO_DOC_CACHE} "
        f"({ASIN_TO_DOC_CACHE.stat().st_size / 1e6:.1f} MB, "
        f"{time.time() - t0:.1f}s, sig={current_sig})")
    return asin_to_doc


# ===========================================================================
# CLI 入口
# ===========================================================================
def main_task_body(force: bool = False) -> None:
    """独立运行入口：生成并清洗 asin_to_doc.json。"""
    log("=== build_asin_to_doc.py ===")
    log(f"  META_FILE = {META_FILE}")
    log(f"  OUTPUT    = {ASIN_TO_DOC_CACHE}")
    asin_to_doc = build_meta_corpus(force=force)
    log(f"  generated {len(asin_to_doc)} ASIN documents")
    log(f"  output → {ASIN_TO_DOC_CACHE}")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """用户指令 2026-09-23: 串行运行 3 个 category.

    每个 category 重新绑定该脚本使用的路径常量为 category-specific 路径,
    然后调原 main_task_body() (保持原有逻辑不动). 产物写到
    result/<stage>/<baby|musical|video_games>/ 子目录.
    """
    global SENT_CACHE, UID_TO_SENTS, ASIN_USERS_PATH, ATTRIBUTES_PATH, META_FILE, OUT_DIR, OUT_PATH, ASIN_TO_DOC_CACHE  # noqa
    # backup current (Baby) defaults
    saved = {
        k: v for k, v in globals().items()
        if k in {"SENT_CACHE", "UID_TO_SENTS", "ASIN_USERS_PATH", "ATTRIBUTES_PATH",
                 "META_FILE", "OUT_DIR", "OUT_PATH", "ASIN_TO_DOC_CACHE"}
        and isinstance(v, Path)
    }
    base_out = REPO_ROOT / "result" / Path(__file__).parent.name
    for category, subdir in CATEGORY_INPUTS:
        log(f"\n========== [{category}] (subdir={subdir}) ==========")
        # Reset all known category-dependent paths to point at the per-category subdir.
        if "SENT_CACHE" in saved:
            SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "UID_TO_SENTS" in saved:
            UID_TO_SENTS = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "ASIN_USERS_PATH" in saved:
            ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract" / f"asin_to_users_{subdir}.pkl"
        if "ATTRIBUTES_PATH" in saved:
            ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction" / f"product_attributes_{subdir}.pkl"
        if "META_FILE" in saved:
            META_FILE = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data") / {
                "baby": "meta_Baby_Products_2023.jsonl",
                "musical": "meta_Musical_Instruments.jsonl",
                "video_games": "meta_Video_Games.jsonl",
            }[subdir]
        if "OUT_DIR" in saved:
            OUT_DIR = base_out / subdir
        if "OUT_PATH" in saved:
            OUT_PATH = base_out / subdir / saved["OUT_PATH"].name
        if "ASIN_TO_DOC_CACHE" in saved:
            ASIN_TO_DOC_CACHE = base_out / subdir / saved["ASIN_TO_DOC_CACHE"].name
        OUT_DIR.mkdir(parents=True, exist_ok=True) if "OUT_DIR" in saved else None
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True) if "OUT_PATH" in saved else None
        ASIN_TO_DOC_CACHE.parent.mkdir(parents=True, exist_ok=True) if "ASIN_TO_DOC_CACHE" in saved else None
        try:
            main_task_body()
        except Exception as e:
            log(f"[{category}] FAILED: {e!r}")
            raise
    # Restore Baby defaults (for import compatibility with downstream).
    for k, v in saved.items():
        globals()[k] = v


if __name__ == "__main__":
    main()