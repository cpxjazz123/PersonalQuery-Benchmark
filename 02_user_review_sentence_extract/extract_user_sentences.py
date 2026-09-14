"""
Baby Products 原始数据 → 用户评论句子提取

从 Baby_Products_2023.jsonl.gz 提取每个用户的全部句子，
按 user_id 聚合，保存为 JSON（dict[user_id] -> [sentences]）。

输出：scratch2/wenyu/gaussian_vades/uid_to_sentences.json

2026-09-12: 新增 _clean_html() 在写文件前清洗 raw sentences (HTML entity decode +
HTML tag strip + space collapse)。原因为 GECToR 把 HTML 残留当作文本错误,产出污染
的 edit pairs → Stage 10 transformation_history 全空 → 99.2% fallback 到 generic。
2026-09-12: 新增 MIN_USER_SENTS=10 过滤,丢弃 <10 句子的用户(asin_to_users 同步移除)。
"""
import gzip, html, json, re, pickle
from pathlib import Path

SCRATCH = Path('/home/wlia0047/hj82_scratch2/wenyu')
RAW_DATA = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/data/Baby_Products_2023.jsonl')
OUT_PATH = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/02_user_review_sentence_extract/uid_to_sentences.json')
OUT_PKL  = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/02_user_review_sentence_extract/uid_to_sentences.pkl')
ASIN_USERS_OUT = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/02_user_review_sentence_extract/asin_to_users.json')

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
      1. html.unescape(): &#34; → ", &lt; → <, &gt; → >, &amp; → &
      2. Strip <br/> / <br>: keep readability (space, not empty)
      3. Strip other HTML tags (<p>, </p>, <div>, etc.)
      4. Collapse multiple spaces
    """
    if not text:
        return text
    # Decode repeatedly because some source reviews contain double-encoded
    # entities such as &amp;quot; → &quot; → ".
    previous = None
    while text != previous:
        previous = text
        text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_OTHER_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


print("Stage 0: 加载原始数据...")
uid_to_sents = {}
asin_to_users = {}

# Auto-detect: .gz → gzip.open, else → open (plain JSONL)
_open = gzip.open if str(RAW_DATA).endswith('.gz') else open
mode = 'rt' if _open is gzip.open else 'r'
n_cleaned = 0
n_total = 0
with _open(RAW_DATA, mode) as f:
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
removed_uids = set()
# 反向重建 asin_to_users (使用过滤后的 uid_to_sents)
new_asin_to_users = {}
for asin, uids in asin_to_users.items():
    kept = [u for u in uids if u in uid_to_sents]
    if kept:
        new_asin_to_users[asin] = sorted(kept)
    else:
        new_asin_to_users[asin] = []
        removed_uids.add(asin)
new_asin_to_users = {a: u for a, u in new_asin_to_users.items() if u}
removed_asins = set(asin_to_users.keys()) - set(new_asin_to_users.keys())
n_after_users = len(uid_to_sents)
n_after_asins = len(new_asin_to_users)
print(f"  MIN_USER_SENTS={MIN_USER_SENTS} filter:")
print(f"    users: {n_before_users} → {n_after_users} (dropped {n_before_users - n_after_users})")
print(f"    asins: {n_before_asins} → {n_after_asins} (dropped {len(removed_asins)} all-empty)")
asin_to_users = new_asin_to_users

SCRATCH.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(uid_to_sents, f, indent=2, ensure_ascii=False)
print(f"  → JSON: {OUT_PATH}")

with open(OUT_PKL, 'wb') as f:
    pickle.dump(uid_to_sents, f)
print(f"  → PKL: {OUT_PKL}")

with open(ASIN_USERS_OUT, 'w') as f:
    json.dump(asin_to_users, f, ensure_ascii=False)
print(f"  → ASIN users: {ASIN_USERS_OUT}")

# Cohort manifest: hash-based fingerprints for downstream Stage 03/04/05/09
# cross-validation. Re-running Stage 02 with any source change will produce
# different fingerprints, forcing the downstream cache & artifacts to rebuild
# rather than silently reusing a stale cohort.
import hashlib
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
    "min_user_sents": int(MIN_USER_SENTS),
    "n_users": len(_uids_sorted),
    "n_total_sents": int(sum(_n_sents_sorted)),
    "n_asins": len(asin_to_users),
    "raw_data_source": str(RAW_DATA),
    "sentence_source_fingerprint": _sentence_source_fingerprint,
    "uid_layout_fingerprint": _uid_layout_fingerprint,
    "html_clean": True,
}
with open(SCRATCH / "cohort_manifest.json", "w") as f:
    json.dump(cohort_manifest, f, indent=2, ensure_ascii=False)
print(f"  → manifest: {SCRATCH / 'cohort_manifest.json'} "
      f"(sentence_source={_sentence_source_fingerprint[:12]}, "
      f"uid_layout={_uid_layout_fingerprint[:12]})")
print("Done.")
