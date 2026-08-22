#!/usr/bin/env python3
"""spaCy sentence extraction for neutral rewrite generation.

与 Pipeline A (gaussian_vades.extract_first_twenty_sentences_for_users) 同源,
保证 sentence_rows 与 rewrites.jsonl 的句子文本完全一致, 消除 72% 覆盖率问题。

工作流:
  1. extract_sentences_spacy.py (本脚本)  -> sentences_for_rewrite.jsonl (spaCy 切)
  2. generate_neutral_rewrites.py          -> rewrites.jsonl
  3. extract_residual_hidden.py            -> residual_hidden.npz
  4. main_train(diagonal_residual_llm)     -> 训练 + 推理

注意: 本脚本只 import spacy, 不 import torch/vllm, 不与 vLLM 在同一进程。
"""
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
SCRATCH.mkdir(parents=True, exist_ok=True)

# === 配置 (硬编码, Rule 3) ===
DATA_SOURCE = REPO_ROOT / "result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews_3000u.json"
OUTPUT_FILE = SCRATCH / "sentences_for_rewrite.jsonl"
MAX_USERS = int(os.environ.get("VADES_NEUTRAL_MAX_USERS", "2918"))  # 默认全量
SENTS_PER_USER = int(os.environ.get("VADES_NEUTRAL_SENTS_PER_USER", "15"))
MIN_WORDS = 5
MAX_WORDS = 60


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def main() -> int:
    print(f"[main] starting at {time.strftime('%H:%M:%S')}")
    print(f"[main] loading spacy...")
    sys.path.insert(0, str(REPO_ROOT))
    # === 直接 import spacy, 不再走 syntactic_analysis/extract_clause_features_single_query
    # (318d features 路径已废, 2026-08-22 cleanup; 本脚本只做 sentence segmentation) ===
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print(f"[main] loading {DATA_SOURCE}")
    with open(DATA_SOURCE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[main] 总用户数: {len(data)}, 限制: {MAX_USERS}")
    # === 增量: 已抽取的 user_id 集合 ===
    done_users: set[str] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_users.add(d["user_id"])
                except Exception:
                    continue
        print(f"[main] 已抽取 {len(done_users)} 用户, 增量跳过")

    # === 收集所有需要 nlp.pipe() 的文本 ===
    # 摊平所有用户的 review 文本到一个 list, 记录 (user_id, review_index, source_text)
    pending: list[tuple[str, str, int, str]] = []
    user_meta: dict[str, dict] = {}
    for entry in data:
        user_id = entry.get("user_id", "?")
        if user_id in done_users:
            continue
        if len(user_meta) >= MAX_USERS:
            break
        asin = entry.get("asin", "?")
        review_texts: list[str] = []
        for rev in entry.get("reviews", []):
            trg = rev.get("target_reviews", [])
            if isinstance(trg, list):
                review_texts.extend(trg)
            elif isinstance(trg, str):
                review_texts.append(trg)
        if not review_texts:
            continue
        user_meta[user_id] = {"asin": asin, "review_count": len(review_texts)}
        for review_index, t in enumerate(review_texts):
            pending.append((user_id, asin, review_index, t))

    print(f"[main] 待处理用户 {len(user_meta)}, review 文本 {len(pending)}")
    if not pending:
        print(f"[main] ✓ 无待抽取文本, 退出")
        return 0

    # === nlp.pipe() 批量解析 (108x 提速 vs 逐句) ===
    BATCH = 500
    user_collected: dict[str, list[dict]] = {uid: [] for uid in user_meta}
    seen_sentences: dict[str, set[str]] = {uid: set() for uid in user_meta}
    print(f"[main] 开始 nlp.pipe 批量解析 (batch={BATCH})...")
    t0 = time.time()
    last_user_id = None
    processed = 0
    for batch_start in range(0, len(pending), BATCH):
        batch = pending[batch_start:batch_start + BATCH]
        texts = [t for _, _, _, t in batch]
        docs = list(nlp.pipe(texts, batch_size=BATCH))
        for (uid, asin, review_index, _), doc in zip(batch, docs):
            for sent in doc.sents:
                if len(user_collected[uid]) >= SENTS_PER_USER:
                    break
                normalized = normalize_text(sent.text)
                if not normalized:
                    continue
                n = len(normalized.split())
                if MIN_WORDS <= n <= MAX_WORDS:
                    if normalized in seen_sentences[uid]:
                        continue
                    seen_sentences[uid].add(normalized)
                    user_collected[uid].append({
                        "user_id": uid,
                        "asin": asin,
                        "review_index": review_index,
                        "sentence_text": normalized,  # 与 Pipeline A (gaussian_vades) 字段名一致
                        "word_count": n,
                    })
        processed += len(batch)
        # 进度报告: 每完成 500 texts
        if processed % 5000 < BATCH:
            elapsed = time.time() - t0
            rate = processed / max(elapsed, 1e-6)
            eta = (len(pending) - processed) / max(rate, 1e-6)
            users_done = sum(1 for uid in user_collected if len(user_collected[uid]) >= SENTS_PER_USER or len(seen_sentences[uid]) >= SENTS_PER_USER)
            print(f"  parsed {processed}/{len(pending)} ({rate:.0f}/s, ETA {eta:.0f}s, users_done={users_done})")

    # === 写盘 ===
    rows: list[dict] = []
    for uid, sents in user_collected.items():
        if not sents:
            continue
        rows.extend(sents[:SENTS_PER_USER])
    print(f"[main] 实际抽取 {len(user_collected)} 用户, {len(rows)} 句")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[main] ✓ 写入 {OUTPUT_FILE}")
    print(f"[main] done at {time.strftime('%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
