#!/usr/bin/env python3
"""E24 Phase B-fix — 重建 per-user tagged dev construction pool。

背景：
  e23 scale sweep 把 100 dev users 的 construction pool 写到进程内存（dict），
  但隐藏状态 cache（scale_hidden_L{24,26,27}.npz）只存了 text→vec 映射，
  没有 user_id tag。
  → Phase B 把所有用户共享一个 14k 句 construction pool，方向信号被洗掉。
  → 所有 cell 都得到 top1 ≈ 1/100, own-other ≈ 0。

修复：
  1. 重新扫描 reviews 收集 100 dev users 的 construction sentences（5-60 words）
  2. 输出 dev_pool_100u.jsonl：
       每行: {user_id, construction_sents: [...], heldout_sents: [...]}
     注意 sentence 文本与 e23 scale_hidden cache 完全兼容（同一过滤规则）。
  3. Phase B 重跑时按 per-user construction list 取前 Y 句构造 mean_diff。

输出:
  /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/dev_pool_100u.jsonl
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))
sys.path.insert(0, str(REPO_ROOT))

from query_gen_main import REVIEWS


def _load_spacy_model():
    from extract_clause_features_single_query import load_spacy_model
    return load_spacy_model()

OUT_E23 = Path("/fs04/ar57/wenyu/PersoanlQuery/result/e23_style_vector")
OUT_E24 = Path("/home/wlia0047/hj82_scratch2/wenyu/e24_style_vector")

VEC_NPZ = OUT_E23 / "style_vectors_100u.npz"
HO_JSON = OUT_E23 / "style_vectors_heldout_validity.json"
OUT_POOL = OUT_E24 / "dev_pool_100u.jsonl"

REV_CAP = 500                # per-user review cap (matching e23)
MIN_SENT_WORDS = 5
MAX_SENT_WORDS = 60


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t0 = time.time()
    OUT_E24.mkdir(parents=True, exist_ok=True)

    # dev user list (100) from style_vectors_100u.npz
    v = np.load(VEC_NPZ, allow_pickle=True)
    dev_users = [str(u) for u in v["users"]]
    log(f"loaded {len(dev_users)} dev users from {VEC_NPZ}")

    # exclude set: the 1000-sent review-level sentences + per-user held-out
    construct_1000 = set(str(s) for s in v["sentences"])
    ho_valid = json.load(open(HO_JSON))
    ho_per_user = {u: ho_valid["heldout_sentences_per_user"].get(u, [])
                   for u in dev_users}
    exclude = set(construct_1000)
    for u in dev_users:
        exclude.update(ho_per_user[u])
    log(f"exclude set: {len(exclude)} unique sentences "
        f"(construct_1000={len(construct_1000)}, held-out={len(exclude)-len(construct_1000)})")

    # collect reviews per user
    reviews: dict[str, list[str]] = {u: [] for u in dev_users}
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            if u not in reviews or len(reviews[u]) >= REV_CAP:
                continue
            t = (d.get("text") or "").strip()
            if t:
                reviews[u].append(t)
    log(f"reviews collected for "
        f"{sum(1 for u in dev_users if reviews[u])}/{len(dev_users)} users")

    # spaCy pipe batched sentence segmentation
    nlp = _load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    K_MAX = 200  # collect up to 200 sents per user to match e23 K_GRID max
    out_lines: list[str] = []
    n_ok = 0
    for u in dev_users:
        sents: list[str] = []
        if reviews[u]:
            for doc in nlp.pipe(reviews[u], batch_size=64):
                for s in doc.sents:
                    txt = s.text.strip()
                    words = [t for t in s if not t.is_space]
                    nw = len(words)
                    if txt and txt not in exclude \
                            and MIN_SENT_WORDS <= nw <= MAX_SENT_WORDS:
                        sents.append(txt)
        cs = sents[:K_MAX]
        ho = ho_per_user[u]
        out_lines.append(json.dumps({
            "user_id": u,
            "construction_sents": cs,
            "heldout_sents": ho,
        }))
        if len(cs) >= K_MAX:
            n_ok += 1
    log(f"users with >= {K_MAX} construction sents: {n_ok}/{len(dev_users)}")

    OUT_POOL.write_text("\n".join(out_lines) + "\n")
    log(f"wrote {OUT_POOL} with {len(out_lines)} users "
        f"({OUT_POOL.stat().st_size / 1e6:.1f} MB)")
    log(f"DONE — total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()