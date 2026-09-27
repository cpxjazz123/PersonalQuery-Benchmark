"""
Baby / Musical_Instruments / Video_Games 原始数据 → 用户评论句子提取

从 <DATA_DIR>/<category>_*.jsonl 提取每个用户的全部句子，
按 user_id 聚合，保存为 pickle（dict[user_id] -> [sentences]）。

输出 (用户指令 2026-09-23, 与 01_attribute_extraction 一致的不合并策略):
  - Baby              → result/02_user_review_sentence_extract/uid_to_sentences_baby.pkl
                       + asin_to_users_baby.pkl
                       + cohort_manifest_baby.json  (供 Stage 03/04/05/09 cross-validation)
  - Musical_Instruments → ..._musical.pkl × 2 / cohort_manifest_musical.json
  - Video_Games       → ..._video_games.pkl × 2 / cohort_manifest_video_games.json

2026-09-12: 新增 _clean_html() 在写文件前清洗 raw sentences (HTML entity decode +
HTML tag strip + space collapse)。原因为 GECToR 把 HTML 残留当作文本错误,产出污染
的 edit pairs → Stage 10 transformation_history 全空 → 99.2% fallback 到 generic。
2026-09-12: 新增 MIN_USER_SENTS=10 过滤,丢弃 <10 句子的用户(asin_to_users 同步移除)。
2026-09-23: 三个 category 各自一份输出, 不合并 (用户指令)。
2026-09-23: 只保留 .pkl (下游全是 Python 消费, 删 .json 省 ~2GB 空间).
"""
import gzip, hashlib, html, json, pickle, re
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path('/home/wlia0047/hj82_scratch2/wenyu')
DATA_DIR = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data")
OUT_DIR = REPO_ROOT / "result" / "02_user_review_sentence_extract"

# 用户指令 2026-09-23: 三个 category 各自一份输出 (不合并), 文件名按 category
# 后缀区分 (semantic suffix, 非版本号 — Rule 17 合规).
# 用户指令 2026-09-23: 只写 .pkl (下游 Python-only, 删 .json 节省磁盘).
CATEGORY_INPUTS = [
    # (category_key, raw_data_path, uid_to_sents_pkl, asin_to_users_pkl, manifest_out)
    # 用户指令 2026-09-23: Baby / Musical_Instruments / Video_Games.
    ("Baby",
     DATA_DIR / "Baby_Products_2023.jsonl",
     OUT_DIR / "uid_to_sentences_baby.pkl",
     OUT_DIR / "asin_to_users_baby.pkl",
     SCRATCH / "cohort_manifest_baby.json"),
    ("Musical_Instruments",
     DATA_DIR / "Musical_Instruments.jsonl",
     OUT_DIR / "uid_to_sentences_musical.pkl",
     OUT_DIR / "asin_to_users_musical.pkl",
     SCRATCH / "cohort_manifest_musical.json"),
    ("Video_Games",
     DATA_DIR / "Video_Games.jsonl",
     OUT_DIR / "uid_to_sentences_video_games.pkl",
     OUT_DIR / "asin_to_users_video_games.pkl",
     SCRATCH / "cohort_manifest_video_games.json"),
]

# === Tuning constants ===
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

# 用户最少句子数过滤(<10 句子的用户丢弃;Stage 09 GECToR 跑不动 + Stage 10 transformation_history 为空)
MIN_USER_SENTS = 10

# HTML artifact cleaning (Stage 02 出口,2026-09-12)
_HTML_TAG_RE = re.compile(r"<br\s*/?>")
_HTML_OTHER_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_MULTI_SPACE_RE = re.compile(r" {2,}")


def _clean_html(text: str) -> str:
    """Remove HTML artifacts from raw review sentences.

    Steps:
      1. html.unescape(): &#34; → ", < → <, > → >, & → &
      2. Strip <br/> / <br>: keep readability (space, not empty)
      3. Strip other HTML tags (<p>, </p>, <div>, etc.)
      4. Collapse multiple spaces
    """
    if not text:
        return text
    # Decode repeatedly because some source reviews contain double-encoded
    # entities such as &quot; → " → ".
    previous = None
    while text != previous:
        previous = text
        text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_OTHER_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def run_for_category(category: str, raw_path: Path,
                     uid_sents_pkl: Path,
                     asin_users_pkl: Path,
                     manifest_out: Path) -> None:
    """提取单个 category 的 raw review → uid_to_sentences_<cat>.pkl + asin_to_users_<cat>.pkl + cohort_manifest_<cat>.json."""
    print(f"\n========== [{category}] ==========")
    print(f"Stage 0: 加载原始数据 from {raw_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw data not found: {raw_path}")
    uid_to_sents: dict[str, list[str]] = {}
    asin_to_users: dict[str, set[str]] = {}

    # Auto-detect: .gz → gzip.open, else → open (plain JSONL)
    _open = gzip.open if str(raw_path).endswith('.gz') else open
    mode = 'rt' if _open is gzip.open else 'r'
    n_cleaned = 0
    n_total = 0
    with _open(raw_path, mode) as f:
        for line in f:
            e = json.loads(line)
            uid = e['user_id']
            asin = e.get('parent_asin')
            if not asin:
                raise ValueError("required field 'parent_asin' is missing")
            text = e.get('text', '')
            cleaned = _clean_html(text)
            if cleaned != text:
                n_cleaned += 1
            n_total += 1
            sents = [s.strip() for s in SENT_SPLIT.split(cleaned) if s.strip()]
            if uid not in uid_to_sents:
                uid_to_sents[uid] = []
            uid_to_sents[uid].extend(sents)
            asin_to_users.setdefault(asin, set()).add(uid)

    asin_to_users = {asin: sorted(users) for asin, users in sorted(asin_to_users.items())}
    print(f"  用户数: {len(uid_to_sents)}")
    print(f"  parent_asin 数: {len(asin_to_users)}")
    print(f"  HTML-cleaned reviews: {n_cleaned} / {n_total} ({100*n_cleaned/max(n_total,1):.2f}%)")
    counts = sorted([len(v) for v in uid_to_sents.values()], reverse=True)
    print(f"  句子数分布: min={min(counts)} median={counts[len(counts)//2]} max={max(counts)}")
    print(f"  用户≥100句: {sum(1 for c in counts if c >= 100)}")
    print(f"  用户≥50句: {sum(1 for c in counts if c >= 50)}")
    print(f"  用户≥20句: {sum(1 for c in counts if c >= 20)}")

    # 过滤 < MIN_USER_SENTS 句子的用户
    n_before_users = len(uid_to_sents)
    n_before_asins = len(asin_to_users)
    uid_to_sents = {uid: sents for uid, sents in uid_to_sents.items()
                    if len(sents) >= MIN_USER_SENTS}
    # 反向重建 asin_to_users (使用过滤后的 uid_to_sents)
    new_asin_to_users: dict[str, list[str]] = {}
    for asin, uids in asin_to_users.items():
        kept = [u for u in uids if u in uid_to_sents]
        if kept:
            new_asin_to_users[asin] = kept
    asin_to_users = new_asin_to_users
    n_after_users = len(uid_to_sents)
    n_after_asins = len(asin_to_users)
    print(f"  MIN_USER_SENTS={MIN_USER_SENTS} filter:")
    print(f"    users: {n_before_users} → {n_after_users} (dropped {n_before_users - n_after_users})")
    print(f"    asins: {n_before_asins} → {n_after_asins} (dropped {n_before_asins - n_after_asins} all-empty)")

    # 用户指令 2026-09-23: 只写 .pkl (下游全是 Python 消费, 删 .json 省 ~2GB 空间).
    uid_sents_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(uid_sents_pkl, 'wb') as f:
        pickle.dump(uid_to_sents, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  → uid_to_sentences PKL: {uid_sents_pkl}")

    with open(asin_users_pkl, 'wb') as f:
        pickle.dump(asin_to_users, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  → asin_to_users PKL: {asin_users_pkl}")

    # Cohort manifest: hash-based fingerprints for downstream Stage 03/04/05/09
    # cross-validation. Re-running Stage 02 with any source change will produce
    # different fingerprints, forcing the downstream cache & artifacts to rebuild
    # rather than silently reusing a stale cohort.
    _uids_sorted = sorted(uid_to_sents.keys())
    _n_sents_sorted = [len(uid_to_sents[u]) for u in _uids_sorted]
    _sentences_concat = "\n".join(s for u in _uids_sorted for s in uid_to_sents[u])
    _sentence_source_fingerprint = hashlib.sha256(
        _sentences_concat.encode("utf-8")).hexdigest()
    _uid_layout_hasher = hashlib.sha256()
    _uid_layout_hasher.update(
        f"uids={len(_uids_sorted)}|min_sents={MIN_USER_SENTS}".encode("utf-8"))
    for uid, n_sents in zip(_uids_sorted, _n_sents_sorted):
        _uid_layout_hasher.update(uid.encode("utf-8"))
        _uid_layout_hasher.update(n_sents.to_bytes(4, "big"))
    _uid_layout_fingerprint = _uid_layout_hasher.hexdigest()
    cohort_manifest = {
        "schema_version": 2,
        "category": category,
        "min_user_sents": int(MIN_USER_SENTS),
        "n_users": len(_uids_sorted),
        "n_total_sents": int(sum(_n_sents_sorted)),
        "n_asins": len(asin_to_users),
        "raw_data_source": str(raw_path),
        "sentence_source_fingerprint": _sentence_source_fingerprint,
        "uid_layout_fingerprint": _uid_layout_fingerprint,
        "html_clean": True,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump(cohort_manifest, f, indent=2, ensure_ascii=False)
    print(f"  → manifest: {manifest_out} "
          f"(sentence_source={_sentence_source_fingerprint[:12]}, "
          f"uid_layout={_uid_layout_fingerprint[:12]})")
    print(f"[{category}] Done.")


def main() -> None:
    print("=== extract_user_sentences.py (multi-category, pkl-only) ===")
    for category, raw_path, uid_sents_pkl, asin_users_pkl, manifest_out in CATEGORY_INPUTS:
        run_for_category(category, raw_path, uid_sents_pkl,
                         asin_users_pkl, manifest_out)
    print("=== ALL DONE ===")


if __name__ == "__main__":
    main()
