#!/usr/bin/env python3
"""
句法鲁棒性评估：同一商品（ASIN），多用户 query 检索性能。

框架 (user-specified):
  Hit@10 / MRR / AvgRankDrop / ρ(D_syn, RR)

D_syn 计算：dependency tree depth + POS pattern distance（spaCy）

检索：对每条 query，在同 ASIN 用户池内 rank 目标用户。
"""
from __future__ import annotations
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy
from scipy.stats import spearmanr

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
BUCKET_FRESH = SCRATCH / "bucket_fresh"
RAW_DATA = REPO / "data/Baby_Products_2023.jsonl.gz"

PCA_D = 32
LAYER_KEY = 26

BUCKETS = [
    ("1",      1,   1),
    ("2-5",    2,   5),
    ("6-10",   6,  10),
    ("11-20", 11,  20),
    ("21-50", 21,  50),
    ("51+",   51, 9999),
]

MAX_SENTS_PER_USER = 20


# ──────────────────────────────────────────────────────────────────────────
# Step 1: Load data
# ──────────────────────────────────────────────────────────────────────────
def load_data():
    print("[load] Loading bucket_fresh data...")
    # rewrites: original → neutral rewrite
    rewrite_map = json.load(open(BUCKET_FRESH / "rewrites.json"))
    uid_meta = json.load(open(BUCKET_FRESH / "uid_meta.json"))
    all_sents = list(rewrite_map.keys())
    print(f"  {len(all_sents)} sentence pairs")

    # residuals: (sentences, residual_layer_26)
    data = np.load(BUCKET_FRESH / "residuals.npy", allow_pickle=True).item()
    resid_arr = data["residual_layer_26"].astype(np.float32)   # (N, 3584)
    resid_sents = data["sentences"].tolist()
    sent_to_idx = {s: i for i, s in enumerate(resid_sents)}
    print(f"  {len(resid_sents)} residuals, shape={resid_arr.shape}")
    return rewrite_map, uid_meta, resid_arr, sent_to_idx


# ──────────────────────────────────────────────────────────────────────────
# Step 2: spaCy syntactic features
# ──────────────────────────────────────────────────────────────────────────
def extract_syntactic_features(texts: list[str], nlp) -> list[dict]:
    """Extract dependency depth profile + POS tag distribution per sentence."""
    feats = []
    for doc in nlp.pipe(texts, batch_size=128):
        # Depath depth stats
        depths = []
        for tok in doc:
            d = len([a for a in list(tok.ancestors)])
            depths.append(d)
        depth_hist = np.zeros(8)
        for d in depths:
            depth_hist[min(d, 7)] += 1
        depth_hist /= (sum(depths) + 1e-8)

        # POS tag distribution (top 15 tags)
        pos_counts = defaultdict(float)
        for tok in doc:
            pos_counts[tok.pos_] += 1
        total_pos = sum(pos_counts.values()) + 1e-8
        pos_vec = np.zeros(15)
        pos_tags = ["NOUN", "VERB", "ADJ", "DET", "ADP", "PUNCT", "CCONJ",
                    "AUX", "PRON", "PART", "ADV", "NUM", "PROPN", "INTJ", "SCONJ"]
        for i, tag in enumerate(pos_tags):
            pos_vec[i] = pos_counts.get(tag, 0) / total_pos

        feats.append({
            "depth_hist": depth_hist,
            "pos_vec": pos_vec,
            "n_tokens": len(doc),
            "n_clauses": sum(1 for tok in doc if tok.dep_ in ("ROOT", "ccomp", "xcomp", "advcl", "relcl")),
        })
    return feats


def syntactic_distance(f1: dict, f2: dict) -> float:
    """Cosine distance between syntactic feature vectors."""
    v1 = np.concatenate([f1["depth_hist"], f1["pos_vec"], [f1["n_tokens"] / 100, f1["n_clauses"] / 10]])
    v2 = np.concatenate([f2["depth_hist"], f2["pos_vec"], [f2["n_tokens"] / 100, f2["n_clauses"] / 10]])
    v1 /= (np.linalg.norm(v1) + 1e-8)
    v2 /= (np.linalg.norm(v2) + 1e-8)
    return float(1 - np.dot(v1, v2))


# ──────────────────────────────────────────────────────────────────────────
# Step 3: Build per-user, per-ASIN profiles
# ──────────────────────────────────────────────────────────────────────────
def build_profiles(uid_meta: dict, sent_to_idx: dict, resid_arr: np.ndarray,
                   rewrite_map: dict):
    """Build uid → {asin: [(orig, rewrite, resid_vec)]} mapping from raw data."""
    print("\n[build] Scanning raw data for ASIN info...")
    uid_asin_sents = defaultdict(lambda: defaultdict(list))

    with gzip.open(RAW_DATA, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("user_id")
            if uid not in uid_meta:
                continue
            asin = r.get("asin", "")
            text = r.get("text", "").strip()
            if not text or not asin:
                continue
            if len(uid_asin_sents[uid][asin]) < MAX_SENTS_PER_USER:
                uid_asin_sents[uid][asin].append(text)

    print("[build] spaCy sentence splitting...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 200000

    uid_asin_sent_info = defaultdict(lambda: defaultdict(list))
    total = sum(len(sents) for d in uid_asin_sents.values() for sents in d.values())
    done = 0

    for uid, asin_dict in uid_asin_sents.items():
        for asin, raws in asin_dict.items():
            all_sents_raw = []
            for raw in raws:
                doc = nlp(raw)
                for sent in doc.sents:
                    txt = sent.text.strip()
                    if txt and len(txt) > 10:
                        all_sents_raw.append(txt)

            for orig in all_sents_raw:
                if orig not in sent_to_idx:
                    continue
                rewrite = rewrite_map.get(orig, orig)
                idx = sent_to_idx[orig]
                uid_asin_sent_info[uid][asin].append({
                    "orig": orig,
                    "rewrite": rewrite,
                    "idx": idx,
                    "resid": resid_arr[idx],
                })
                done += 1
            if done % 5000 == 0:
                print(f"  [{done}/{total}]", flush=True)

    print(f"  built {done} (uid,asin,sent) records")
    return uid_asin_sent_info


# ──────────────────────────────────────────────────────────────────────────
# Step 4: Per-ASIN retrieval evaluation
# ──────────────────────────────────────────────────────────────────────────
def evaluate(uid_asin_sent_info: dict, uid_meta: dict, resid_arr: np.ndarray,
              sent_to_idx: dict):
    print("\n[eval] Computing retrieval metrics per ASIN...")

    results_by_bucket = defaultdict(lambda: {
        "ranks_q0": [], "ranks_q1": [], "d_syns": [],
        "hit10_q0": [], "hit10_q1": [],
        "rr_q0": [], "rr_q1": [],
    })

    from sklearn.decomposition import PCA

    # Global PCA on all residuals
    all_resids = []
    for uid_asin_dict in uid_asin_sent_info.values():
        for asin_dict in uid_asin_dict.values():
            for rec in asin_dict:
                all_resids.append(rec["resid"])
    all_resids = np.stack(all_resids).astype(np.float32)
    print(f"  fitting PCA on {len(all_resids)} residuals...")
    pca = PCA(n_components=PCA_D, random_state=42)
    pca_resids = pca.fit_transform(all_resids).astype(np.float32)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Build per-user mean profile (in PCA space)
    uid_mean_prof = defaultdict(lambda: np.zeros(PCA_D, dtype=np.float32))
    uid_count_prof = defaultdict(int)
    uid_bucket = {}

    for uid, asin_dict in uid_asin_sent_info.items():
        for asin, recs in asin_dict.items():
            idxs = [r["idx"] for r in recs]
            if not idxs:
                continue
            vec = pca_resids[idxs].mean(axis=0)
            uid_mean_prof[uid] = vec
            uid_count_prof[uid] += 1
            uid_bucket[uid] = uid_meta.get(uid, {}).get("bucket", "?")

    # Extract syntactic features for all sentences
    print("  extracting syntactic features...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 200000

    # Collect all sentences needing features
    all_texts = []
    sent_text_to_feat_idx = defaultdict(list)
    for uid_asin_dict in uid_asin_sent_info.values():
        for asin_dict in uid_asin_dict.values():
            for rec in asin_dict:
                orig = rec["orig"]
                rew = rec["rewrite"]
                if orig not in sent_text_to_feat_idx:
                    sent_text_to_feat_idx[orig].append(("orig", len(all_texts)))
                    all_texts.append(orig)
                if rew not in sent_text_to_feat_idx:
                    sent_text_to_feat_idx[rew].append(("rew", len(all_texts)))
                    all_texts.append(rew)

    syn_feats = extract_syntactic_features(all_texts, nlp)
    sent_to_feat = {}
    for sent, indices in sent_text_to_feat_idx.items():
        feat_list = [syn_feats[i] for tag, i in indices]
        sent_to_feat[sent] = feat_list[0] if feat_list else None

    # Cosine similarity in PCA space
    uid_ids = sorted(uid_mean_prof.keys())
    uid_mat = np.stack([uid_mean_prof[u] for u in uid_ids], axis=0)
    norms = np.linalg.norm(uid_mat, axis=1, keepdims=True) + 1e-8
    uid_mat_norm = uid_mat / norms

    uid_to_idx = {u: i for i, u in enumerate(uid_ids)}

    # Build ASIN → all uids mapping
    asin_to_uids = defaultdict(list)
    for uid, asin_dict in uid_asin_sent_info.items():
        for asin in asin_dict:
            asin_to_uids[asin].append(uid)

    # Retrieval: for each (uid, asin, sent_rec), rank target uid in ASIN pool
    print("  running retrieval evaluation...")
    retrieval_records = []

    for uid, asin_dict in uid_asin_sent_info.items():
        bucket = uid_bucket.get(uid, "?")
        for asin, recs in asin_dict.items():
            if len(recs) < 1:
                continue

            # ASIN pool: ALL uids who reviewed this ASIN (across the full sampled data)
            pool_uids = asin_to_uids[asin]
            pool_indices = [uid_to_idx[u] for u in pool_uids if u in uid_to_idx]
            if len(pool_indices) < 2:
                continue
            pool_mat = uid_mat_norm[pool_indices]

            for rec in recs:
                orig = rec["orig"]
                rew = rec["rewrite"]
                feat_o = sent_to_feat.get(orig)
                feat_r = sent_to_feat.get(rew)
                if feat_o is None or feat_r is None:
                    continue

                # D_syn between orig and rewrite
                d_syn = syntactic_distance(feat_o, feat_r)

                # Project query residual to PCA space
                q_resid = rec["resid"][None, :]  # (1, 3584)
                q_pca = pca.transform(q_resid).astype(np.float32)  # (1, 32)
                q_norm = q_pca / (np.linalg.norm(q_pca) + 1e-8)

                # Scores vs all pool users
                scores = (pool_mat * q_norm).sum(axis=1)  # (pool_size,)
                sorted_idx = np.argsort(-scores)  # descending
                sorted_uids = [pool_uids[i] for i in sorted_idx]

                target_rank_q0 = sorted_uids.index(uid) + 1  # 1-indexed

                # Same for rewrite query
                rw_resid = resid_arr[sent_to_idx.get(rew, 0)][None, :]
                rw_pca = pca.transform(rw_resid).astype(np.float32)
                rw_norm = rw_pca / (np.linalg.norm(rw_pca) + 1e-8)
                rw_scores = (pool_mat * rw_norm).sum(axis=1)
                rw_sorted_idx = np.argsort(-rw_scores)
                rw_sorted_uids = [pool_uids[i] for i in rw_sorted_idx]
                target_rank_q1 = rw_sorted_uids.index(uid) + 1

                retrieval_records.append({
                    "bucket": bucket,
                    "asin": asin,
                    "uid": uid,
                    "d_syn": d_syn,
                    "rank_q0": target_rank_q0,
                    "rank_q1": target_rank_q1,
                    "rr_q0": 1.0 / target_rank_q0,
                    "rr_q1": 1.0 / target_rank_q1,
                    "hit10_q0": 1 if target_rank_q0 <= 10 else 0,
                    "hit10_q1": 1 if target_rank_q1 <= 10 else 0,
                })

    return retrieval_records, pca


def print_results(retrieval_records: list, pca):
    """Aggregate and print final metrics."""
    from collections import defaultdict

    records_by_bucket = defaultdict(list)
    for r in retrieval_records:
        records_by_bucket[r["bucket"]].append(r)

    print("\n" + "=" * 80)
    print(f"{'Bucket':<10} {'N':>5} {'Hit@10_q0':>10} {'Hit@10_q1':>10} "
          f"{'MRR_q0':>8} {'MRR_q1':>8} {'AvgDrop':>8} {'ρ(D_syn,RR)':>12}")
    print("=" * 80)

    all_records = retrieval_records  # global pool

    bucket_results = {}
    for bname, _, _ in BUCKETS:
        recs = records_by_bucket.get(bname, [])
        if not recs:
            continue
        n = len(recs)
        hit10_q0 = np.mean([r["hit10_q0"] for r in recs])
        hit10_q1 = np.mean([r["hit10_q1"] for r in recs])
        mrr_q0 = np.mean([r["rr_q0"] for r in recs])
        mrr_q1 = np.mean([r["rr_q1"] for r in recs])
        drops = [r["rank_q1"] - r["rank_q0"] for r in recs]
        avg_drop = np.mean(drops)

        d_syns = [r["d_syn"] for r in recs]
        rrs = [r["rr_q1"] for r in recs]
        if len(recs) > 3:
            rho, pval = spearmanr(d_syns, rrs)
        else:
            rho, pval = np.nan, np.nan

        bucket_results[bname] = {
            "n": n, "hit10_q0": hit10_q0, "hit10_q1": hit10_q1,
            "mrr_q0": mrr_q0, "mrr_q1": mrr_q1,
            "avg_drop": avg_drop, "rho": rho, "pval": pval,
        }

        print(f"{bname:<10} {n:>5} {hit10_q0:>10.3f} {hit10_q1:>10.3f} "
              f"{mrr_q0:>8.3f} {mrr_q1:>8.3f} {avg_drop:>8.2f} {rho:>10.3f}")

    # Global
    if all_records:
        n = len(all_records)
        hit10_q0 = np.mean([r["hit10_q0"] for r in all_records])
        hit10_q1 = np.mean([r["hit10_q1"] for r in all_records])
        mrr_q0 = np.mean([r["rr_q0"] for r in all_records])
        mrr_q1 = np.mean([r["rr_q1"] for r in all_records])
        drops = [r["rank_q1"] - r["rank_q0"] for r in all_records]
        avg_drop = np.mean(drops)
        d_syns = [r["d_syn"] for r in all_records]
        rrs = [r["rr_q1"] for r in all_records]
        rho, pval = spearmanr(d_syns, rrs)
        print("-" * 80)
        print(f"{'ALL':<10} {n:>5} {hit10_q0:>10.3f} {hit10_q1:>10.3f} "
              f"{mrr_q0:>8.3f} {mrr_q1:>8.3f} {avg_drop:>8.2f} {rho:>10.3f} (p={pval:.3f})")

    print("=" * 80)

    # Length-controlled: split by query length
    short = [r for r in all_records if r["d_syn"] < np.median([x["d_syn"] for x in all_records])]
    long = [r for r in all_records if r["d_syn"] >= np.median([x["d_syn"] for x in all_records])]
    if short and long:
        print("\n[Length-controlled]")
        for label, recs in [("Low D_syn", short), ("High D_syn", long)]:
            n = len(recs)
            hit10_q1 = np.mean([r["hit10_q1"] for r in recs])
            mrr_q1 = np.mean([r["rr_q1"] for r in recs])
            drops = np.mean([r["rank_q1"] - r["rank_q0"] for r in recs])
            print(f"  {label}: N={n}, Hit@10={hit10_q1:.3f}, MRR={mrr_q1:.3f}, AvgDrop={drops:.2f}")

    return bucket_results


def main():
    t0 = time.time()
    rewrite_map, uid_meta, resid_arr, sent_to_idx = load_data()

    uid_asin_sent_info = build_profiles(uid_meta, sent_to_idx, resid_arr, rewrite_map)

    retrieval_records, pca = evaluate(uid_asin_sent_info, uid_meta, resid_arr, sent_to_idx)

    print(f"\n[total retrieval records: {len(retrieval_records)}]")
    results = print_results(retrieval_records, pca)

    # Save results
    out_path = SCRATCH / "syntactic_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"records": retrieval_records, "summary": results}, f)
    print(f"\n[saved] → {out_path}")
    print(f"[total time] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
