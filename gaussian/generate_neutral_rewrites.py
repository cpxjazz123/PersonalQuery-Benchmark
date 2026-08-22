#!/usr/bin/env python3
"""Generate neutral rewrites for VADES training sentences (Qwen hidden-space residual).

为 VADES diagonal_residual_llm 模式做前置: 把每个用户评论句子改写成中立、plain 风格。
残差 = user_hidden - neutral_hidden 将在 Qwen hidden 空间计算,不是 318d 句法空间。

输入: result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews_3000u.json
输出: hj82_scratch2/wenyu/gaussian_vades/rewrites.jsonl
"""
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
SCRATCH.mkdir(parents=True, exist_ok=True)

# === 让 llm_client 可被 import ===
sys.path.insert(0, str(REPO_ROOT))

# === 配置 (硬编码, 不接受 CLI 参数, Rule 3) ===
DATA_SOURCE = REPO_ROOT / "result/personal_query/01_preference_extraction/Baby_Products/stage1_filtered_users_reviews_3000u.json"
OUTPUT_FILE = SCRATCH / "rewrites.jsonl"
MAX_USERS = int(os.environ.get("VADES_NEUTRAL_MAX_USERS", "20"))  # smoke: 20 用户, ~300 句
SENTS_PER_USER = int(os.environ.get("VADES_NEUTRAL_SENTS_PER_USER", "15"))
MIN_WORDS = 5
MAX_WORDS = 60
REWRITE_BATCH = int(os.environ.get("VADES_NEUTRAL_REWRITE_BATCH", "16"))
MAX_NEW = 128
REWRITE_TEMPERATURE = 0.3
REWRITE_SYSTEM = (
    "You are a careful editor. Rewrite the following review sentence in a "
    "plain, neutral, matter-of-fact style. Keep the exact same meaning and "
    "all facts, but remove personal tone, slang, exclamations, and emotional "
    "words. Output ONLY the rewritten sentence, nothing else."
)

# === 加载现有 rewrites (增量, 不重复生成) ===
def load_existing_rewrites() -> dict[str, str]:
    cache: dict[str, str] = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    s = d.get("sentence")
                    r = d.get("rewrite")
                    if s and r:
                        cache[s] = r
                except Exception:
                    continue
    return cache


def append_rewrites(rows: list[dict]) -> None:
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# === 提取用户句子 ===
def extract_user_sentences(limit_users: int = MAX_USERS, sents_per_user: int = SENTS_PER_USER) -> list[tuple[str, str, str]]:
    """(user_id, sentence, asin) 三元组,过滤长度 5..60 words。

    注意: 不能 import spacy / thinc, 因为它们会触发 torch 初始化,导致后续 vLLM fork 失败。
    这里只用 regex 切句, 牺牲一些边界情况(缩写/小数点)换取 vLLM 可启动。
    """
    print(f"[extract] 加载 {DATA_SOURCE}")
    with open(DATA_SOURCE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[extract] 总用户数: {len(data)}, 限制: {limit_users}")
    out: list[tuple[str, str, str]] = []
    # 简单 regex 切句: 在 . ! ? 后跟空格处切;不处理 Mr./Dr./1.5 等边界情况(neutral rewrite 对此容忍)
    sent_split = re.compile(r"(?<=[.!?])\s+")
    kept_users = 0
    for entry in data:
        if kept_users >= limit_users:
            break
        user_id = entry.get("user_id", "?")
        asin = entry.get("asin", "?")
        # 把 reviews 列表(每个含 target_reviews)摊平
        review_texts: list[str] = []
        for rev in entry.get("reviews", []):
            trg = rev.get("target_reviews", [])
            if isinstance(trg, list):
                review_texts.extend(trg)
            elif isinstance(trg, str):
                review_texts.append(trg)
        if not review_texts:
            continue
        sents: list[str] = []
        for t in review_texts:
            for chunk in sent_split.split(t):
                txt = chunk.strip()
                if not txt:
                    continue
                n = len(txt.split())
                if MIN_WORDS <= n <= MAX_WORDS:
                    sents.append(txt)
        if not sents:
            continue
        # 保留每个用户前 sents_per_user 句
        for sent in sents[:sents_per_user]:
            out.append((user_id, sent, asin))
        kept_users += 1
    print(f"[extract] 实际抽取 {kept_users} 用户, 共 {len(out)} 句")
    return out


# === 批量 vLLM 生成 (Rule 4/5g/h: 必须 batched) ===
def batch_generate_neutral(prompts: list[str]) -> list[str]:
    """直接调 vLLM backend.model.generate([prompts], sampling_params) 一次传所有 prompt。"""
    from vllm import SamplingParams
    from llm_client import _LocalBackend  # noqa: F401  确认 backend 已加载

    # === 走与 e23 一致的 chat_template ===
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client()  # 用 default Qwen2-7B
    backend = client._backend
    assert backend is not None, "vLLM backend 未初始化"

    sampling = SamplingParams(
        max_tokens=MAX_NEW,
        temperature=REWRITE_TEMPERATURE,
        top_p=0.95,
    )
    full_prompts = [
        backend.tokenizer.apply_chat_template(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": s}],
            tokenize=False, add_generation_prompt=True,
        )
        for s in prompts
    ]
    outputs = backend.model.generate(full_prompts, sampling)
    texts = []
    for o in outputs:
        if o.outputs:
            t = o.outputs[0].text.strip()
            t = t.split("<|im_end|>")[0].strip()
            texts.append(t)
        else:
            texts.append("")
    return texts


def main() -> int:
    print(f"[main] starting at {time.strftime('%H:%M:%S')}")
    sents = extract_user_sentences()
    if not sents:
        print("[main] ✗ 无句子, 退出")
        return 1
    # 增量: 已生成的跳过
    cached = load_existing_rewrites()
    todo = [(uid, s, asin) for uid, s, asin in sents if s not in cached]
    print(f"[main] sentences total={len(sents)}, cached={len(cached)}, todo={len(todo)}")
    if not todo:
        print("[main] ✓ 全部已缓存, 无需重跑")
        return 0
    # 分批生成
    new_rows: list[dict] = []
    start = time.time()
    for i in range(0, len(todo), REWRITE_BATCH):
        chunk = todo[i:i + REWRITE_BATCH]
        prompts = [s for _, s, _ in chunk]
        try:
            rewrites = batch_generate_neutral(prompts)
        except Exception as exc:
            import traceback
            print(f"[main] ✗ batch_generate failed: {exc!r}")
            traceback.print_exc()
            raise
        for (uid, s, asin), r in zip(chunk, rewrites):
            new_rows.append({
                "user_id": uid,
                "asin": asin,
                "sentence": s,
                "rewrite": r,
            })
        # 增量写盘 (防中断丢失)
        append_rewrites(new_rows)
        new_rows.clear()
        done = min(i + REWRITE_BATCH, len(todo))
        elapsed = time.time() - start
        rate = done / max(elapsed, 1e-6)
        eta = (len(todo) - done) / max(rate, 1e-6)
        print(f"  rewrite {done}/{len(todo)} ({rate:.1f}/s, ETA {eta:.0f}s)")
    print(f"[main] ✓ 完成, 写入 {OUTPUT_FILE}")
    print(f"[main] done at {time.strftime('%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())