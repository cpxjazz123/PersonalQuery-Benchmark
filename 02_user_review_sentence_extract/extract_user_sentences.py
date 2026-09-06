"""
Baby Products 原始数据 → 用户评论句子提取

从 Baby_Products_2023.jsonl.gz 提取每个用户的全部句子，
按 user_id 聚合，保存为 JSON（dict[user_id] -> [sentences]）。

输出：scratch2/wenyu/gaussian_vades/uid_to_sentences.json
"""
import gzip, json, re, pickle
from pathlib import Path

SCRATCH = Path('/home/wlia0047/hj82_scratch2/wenyu')
RAW_DATA = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/data/Baby_Products_2023.jsonl')
OUT_PATH = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/02_user_review_sentence_extract/uid_to_sentences.json')
OUT_PKL  = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/02_user_review_sentence_extract/uid_to_sentences.pkl')
ASIN_USERS_OUT = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/02_user_review_sentence_extract/asin_to_users.json')

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

print("Stage 0: 加载原始数据...")
uid_to_sents = {}
asin_to_users = {}

# Auto-detect: .gz → gzip.open, else → open (plain JSONL)
_open = gzip.open if str(RAW_DATA).endswith('.gz') else open
mode = 'rt' if _open is gzip.open else 'r'
with _open(RAW_DATA, mode) as f:
    for line in f:
        e = json.loads(line)
        uid = e['user_id']
        asin = e.get('parent_asin')
        if not asin:
            raise ValueError("required field 'parent_asin' is missing")
        text = e.get('text', '')
        sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
        if uid not in uid_to_sents:
            uid_to_sents[uid] = []
        uid_to_sents[uid].extend(sents)
        asin_to_users.setdefault(asin, set()).add(uid)

asin_to_users = {asin: sorted(users) for asin, users in sorted(asin_to_users.items())}
print(f"  用户数: {len(uid_to_sents)}")
print(f"  parent_asin 数: {len(asin_to_users)}")
counts = sorted([len(v) for v in uid_to_sents.values()], reverse=True)
print(f"  句子数分布: min={min(counts)} median={counts[len(counts)//2]} max={max(counts)}")
print(f"  用户≥100句: {sum(1 for c in counts if c >= 100)}")
print(f"  用户≥50句: {sum(1 for c in counts if c >= 50)}")
print(f"  用户≥20句: {sum(1 for c in counts if c >= 20)}")

SCRATCH.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(uid_to_sents, f)
print(f"  → JSON: {OUT_PATH}")

with open(OUT_PKL, 'wb') as f:
    pickle.dump(uid_to_sents, f)
print(f"  → PKL: {OUT_PKL}")

with open(ASIN_USERS_OUT, 'w') as f:
    json.dump(asin_to_users, f, ensure_ascii=False)
print(f"  → ASIN users: {ASIN_USERS_OUT}")
print("Done.")
