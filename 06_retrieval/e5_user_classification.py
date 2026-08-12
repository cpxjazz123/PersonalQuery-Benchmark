#!/usr/bin/env python3
"""E5 — Sentence-level user classification: top-1 + Recall@5.

For each user in stage1_filtered_users_reviews:
- Encode each review sentence via BGE
- Per-user centroid (mean of training sentences)
- Test: for each held-out sentence, classify to nearest user by cosine

Reports top-1 and Recall@5 across multiple batch sizes (16 / 32 / 64).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PREF_DIR = Path("/fs04/ar57/wenyu/PersoanlQuery/result/personal_query/01_preference_extraction")
OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/RAG/E5_user_classification")
CATEGORIES = ["Baby_Products", "Grocery_and_Gourmet_Food", "Pet_Supplies"]
TRAIN_RATIO = 0.8
SEED = 42


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_users_reviews(category: str) -> Dict[str, List[str]]:
    """Returns {user_id: [review_text, ...]} — flatten all reviews per user."""
    p = PREF_DIR / category / "stage1_filtered_users_reviews.json"
    with open(p) as f:
        d = json.load(f)
    user_reviews: Dict[str, List[str]] = defaultdict(list)
    for rec in d["users"]:
        uid = rec.get("user_id")
        for res in rec.get("results", []):
            for r in res.get("target_reviews", []):
                if r and isinstance(r, str) and r.strip():
                    user_reviews[uid].append(r[:1000])  # truncate
    return dict(user_reviews)


def split_train_test(user_reviews: Dict[str, List[str]], ratio: float, rng: random.Random):
    train: Dict[str, List[str]] = {}
    test: Dict[str, List[str]] = {}
    for uid, rs in user_reviews.items():
        rs = list(rs)
        rng.shuffle(rs)
        cut = max(1, int(len(rs) * ratio))
        if cut >= len(rs):
            cut = max(1, len(rs) - 1)
        train[uid] = rs[:cut]
        test[uid] = rs[cut:]
    return train, test


def classify_topk(test_embs: np.ndarray, train_embs_by_user: Dict[str, np.ndarray],
                  user_list: List[str], k: int = 5) -> List[List[str]]:
    """For each test_emb, return top-k nearest user ids by cosine sim."""
    from sentence_transformers import util
    import torch
    user_embs = np.stack([train_embs_by_user[u] for u in user_list], axis=0)
    user_embs_t = torch.from_numpy(user_embs).float()
    test_t = torch.from_numpy(test_embs).float()
    scores = util.cos_sim(test_t, user_embs_t).numpy()  # (N_test, N_users)
    topk_idx = np.argsort(-scores, axis=1)[:, :k]
    return [[user_list[i] for i in row] for row in topk_idx]


def evaluate(predictions: List[List[str]], ground_truth: List[str], k_values=(1, 5)):
    n = len(ground_truth)
    out = {}
    for k in k_values:
        correct = sum(1 for gt, preds in zip(ground_truth, predictions) if gt in preds[:k])
        out[f"top_{k}"] = correct / n if n else 0.0
    return out


def process_category(category: str, batch_size: int = 32, seed: int = SEED) -> Dict:
    log(f"\n=== {category} (batch={batch_size}) ===")
    rng = random.Random(seed)
    user_reviews = load_users_reviews(category)
    log(f"  users={len(user_reviews)}")
    counts = [len(v) for v in user_reviews.values()]
    log(f"  reviews per user: min={min(counts)} max={max(counts)} mean={np.mean(counts):.1f}")
    train, test = split_train_test(user_reviews, TRAIN_RATIO, rng)
    log(f"  train_users={len(train)} test_users={len(test)} "
        f"train_reviews={sum(len(v) for v in train.values())} "
        f"test_reviews={sum(len(v) for v in test.values())}")

    # Encode all train reviews in ONE batch, then average per user
    import torch
    sys.path.insert(0, "/fs04/ar57/wenyu/PersoanlQuery/06_retrieval")
    from utils.retrievers import BGERetriever
    os.environ.setdefault("HF_HOME", "/home/wlia0047/.cache/huggingface/hub")
    bge = BGERetriever(model_name="BAAI/bge-large-en-v1.5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bge.device = device

    user_ids = sorted(train.keys())
    # Flatten all train sentences with their user_ids
    flat_sents: List[str] = []
    sent_user_idx: List[int] = []
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    for uid in user_ids:
        for s in train[uid]:
            flat_sents.append(s)
            sent_user_idx.append(user_to_idx[uid])
    log(f"  encoding {len(flat_sents)} train sentences in one batch...")
    with torch.no_grad():
        train_all_embs = bge.encode_queries(flat_sents, batch_size=batch_size)
    log(f"  train encoded, shape={train_all_embs.shape}")

    # Per-user centroid = mean of its sentence embeddings
    n_users = len(user_ids)
    dim = train_all_embs.shape[1]
    user_sums = np.zeros((n_users, dim), dtype=np.float32)
    user_counts = np.zeros(n_users, dtype=np.int64)
    for i, uid_idx in enumerate(sent_user_idx):
        user_sums[uid_idx] += train_all_embs[i]
        user_counts[uid_idx] += 1
    user_means = user_sums / np.maximum(user_counts[:, None], 1)
    # L2-normalize
    norms = np.linalg.norm(user_means, axis=1, keepdims=True) + 1e-9
    user_means = user_means / norms
    train_embs_by_user = {uid: user_means[user_to_idx[uid]] for uid in user_ids}
    log(f"  computed {len(train_embs_by_user)} user centroids")

    # Encode test sentences
    test_sentences: List[str] = []
    test_users: List[str] = []
    for uid, sents in test.items():
        for s in sents:
            test_sentences.append(s)
            test_users.append(uid)
    with torch.no_grad():
        test_embs = bge.encode_queries(test_sentences, batch_size=batch_size)
    log(f"  encoded {len(test_sentences)} test sentences")

    # Classify
    preds_top5 = classify_topk(test_embs, train_embs_by_user, user_ids, k=5)
    metrics = evaluate(preds_top5, test_users, k_values=(1, 5))
    log(f"  top_1={metrics['top_1']:.4f}  Recall@5={metrics['top_5']:.4f}")

    # Baseline: random
    rand_preds = []
    for _ in test_users:
        rand_preds.append(rng.sample(user_ids, 5))
    rand_metrics = evaluate(rand_preds, test_users, k_values=(1, 5))
    log(f"  random baseline: top_1={rand_metrics['top_1']:.4f} Recall@5={rand_metrics['top_5']:.4f}")

    # Majority baseline: always pick most common user
    from collections import Counter
    majority_user = Counter(train.keys()).most_common(1)[0][0]
    maj_preds = [[majority_user] * 5 for _ in test_users]
    maj_metrics = evaluate(maj_preds, test_users, k_values=(1, 5))
    log(f"  majority baseline: top_1={maj_metrics['top_1']:.4f} Recall@5={maj_metrics['top_5']:.4f}")

    del bge
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "category": category,
        "batch_size": batch_size,
        "n_users": len(train),
        "n_train_sentences": sum(len(v) for v in train.values()),
        "n_test_sentences": len(test_sentences),
        "user_clf": metrics,
        "random_baseline": rand_metrics,
        "majority_baseline": maj_metrics,
    }


def write_summary_md(sums: List[Dict]) -> str:
    rows = ["# E5 — Sentence-level user classification (top-1 / Recall@5)\n",
            "**Method**: train user centroids from 80% of reviews per user; classify held-out 20% sentences.\n",
            "| Category | n_users | n_test | top_1 | Recall@5 | random top_1 | majority top_1 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for s in sums:
        rows.append(
            f"| {s['category']} | {s['n_users']} | {s['n_test_sentences']} | "
            f"{s['user_clf']['top_1']:.4f} | {s['user_clf']['top_5']:.4f} | "
            f"{s['random_baseline']['top_1']:.4f} | {s['majority_baseline']['top_1']:.4f} |"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "summary_full.md"
    with open(p, "w") as f:
        f.write("\n".join(rows))
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sums = []
    for cat in args.categories:
        try:
            s = process_category(cat, batch_size=args.batch_size)
            with open(OUT_DIR / f"{cat}_user_clf.json", "w") as f:
                json.dump(s, f, indent=2, default=str)
            sums.append(s)
        except Exception as e:
            log(f"ERROR {cat}: {e}")
            import traceback
            traceback.print_exc()
    if sums:
        p = write_summary_md(sums)
        log(f"wrote {p}")
    log("=== done ===")


if __name__ == "__main__":
    main()