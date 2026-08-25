"""Stage 10H-B — Personalized Retrieval Volatility (mean_t50).

Stage 10G's mean_t50 selection produces genuinely different personalized queries
per user (unique 6.46/10 vs absolute 1.97). Stage 10H-B measures how this
affects retrieval volatility — does the **expression-induced** variation in
personalized queries cause retrieval results to flip?

RQ: In the case where users actually get different queries, how much does
retrieval (Hit@1 / Hit@5 / RR) flip across users for the same target product?

Pipeline:
1. mean_t50 selection per (asin, user) — get personalized query.
2. Run BM25 (bm25s) + MiniLM (GPU) retrieval against Baby_Products corpus.
3. Compute per-ASIN volatility (hit@1 flip, hit@5 flip, RR std) — same
   metrics as Stage 9D.
4. Aggregate across ASINs; paired bootstrap CI for BM25 vs MiniLM.

Output:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10h_b_per_query.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10h_b_volatility.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10h_b_summary.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \\
        gaussian/syntax_subspace_stage10h_b_volatility.py \\
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage10h_b_run.log 2>&1 &
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
USER_GAUSSIANS = SCRATCH / "stage9g_user_gaussians_3levels.json"
POOL_8_5 = SCRATCH / "stage8_5_pool.json"
STAGE8_5_SELECTION = SCRATCH / "stage8_5_selection.json"
STAGE8_5_ASINS = SCRATCH / "stage8_5_asins.json"
STAGE7_CORPUS_DOCS = SCRATCH / "stage7_corpus_docs.jsonl.gz"

OUTPUT_PER_Q = SCRATCH / "stage10h_b_per_query.json"
OUTPUT_VOLATILITY = SCRATCH / "stage10h_b_volatility.json"
OUTPUT_SUMMARY = SCRATCH / "stage10h_b_summary.json"

PCA_DIM = 48
PCA_SEED = 2024
SEED = 2024
N_BOOT = 10000
THRESHOLD_PCT = 50


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=== Stage 10H-B — Personalized Retrieval Volatility (mean_t50) ===")

    # === 1. Load scaler + PCA48 ===
    log("\n=== 1. Loading scaler + PCA48 ===")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")
    from gaussian_vades import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    feature_names = P["feature_names_ordered"]
    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(P["X_scaled"]), size=min(5000, len(P["X_scaled"])), replace=False)
    pca = PCA(n_components=PCA_DIM, random_state=PCA_SEED)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 EV={pca.explained_variance_ratio_.sum():.4f}")

    # === 2. Load user Gaussians (base level) ===
    log("\n=== 2. Loading user Gaussians ===")
    g_data = json.load(open(USER_GAUSSIANS))
    base_users = g_data["users"]["base"]
    user_mu = {uid: np.array(u["mu"], dtype=np.float64) for uid, u in base_users.items()}
    user_sigma = {uid: np.array(u["sigma_diag"], dtype=np.float64) for uid, u in base_users.items()}
    log(f"  users: {len(user_mu)}")

    # === 3. Load Stage 8.5 pool + features ===
    log("\n=== 3. Loading pool + features ===")
    pool_data = json.load(open(POOL_8_5))
    pool_by_asin = pool_data["pools"]

    feat_cache = SCRATCH / "stage7b_query_features.jsonl.gz"
    feat_map = {}
    with gzip.open(feat_cache, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  pool: {sum(len(v) for v in pool_by_asin.values())} queries across {len(pool_by_asin)} ASINs")

    # === 4. Project pool queries to 48d ===
    log("\n=== 4. Projecting pool queries ===")
    import hashlib
    def project_query(text):
        k = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()
        feats = feat_map.get(k)
        if not feats:
            return None
        numeric = {n: float(v) for n, v in feats.items() if isinstance(v, (int, float))}
        if not numeric or len(numeric) < len(feature_names) * 0.5:
            return None
        vec = np.array([numeric.get(n, 0.0) for n in feature_names], dtype=np.float64)
        return pca.transform(scaler.transform(vec[None, :]))[0]

    pool_z_by_asin = {}
    n_skipped = 0
    for asin, queries in pool_by_asin.items():
        zs = []
        for q in queries:
            z = project_query(q["query"])
            if z is None:
                zs.append(None)
                n_skipped += 1
            else:
                zs.append(z)
        pool_z_by_asin[asin] = zs
    log(f"  projected; skipped: {n_skipped}")

    # === 5. Load (asin, user) pairs ===
    sel_data = json.load(open(STAGE8_5_SELECTION))
    asin_users = defaultdict(list)
    for e in sel_data["entries"]:
        asin = e["asin"]
        uid = e["user_id"]
        if uid not in user_mu:
            continue
        if uid not in asin_users[asin]:
            asin_users[asin].append(uid)
    log(f"  asins with users: {len(asin_users)}")

    # === 6. mean_t50 selection per (asin, user) ===
    log("\n=== 6. mean_t50 selection per (asin, user) ===")
    selected_queries = []  # list of {asin, user_id, query}
    selection_distances = []

    for asin, users in asin_users.items():
        zs = pool_z_by_asin.get(asin, [])
        valid_idx = [i for i, z in enumerate(zs) if z is not None]
        if len(valid_idx) < 5 or len(users) < 2:
            continue
        z_arr = np.stack([zs[i] for i in valid_idx])
        n_q = len(valid_idx)
        valid_users = [u for u in users if u in user_mu]
        if len(valid_users) < 2:
            continue

        mu_arr = np.stack([user_mu[uid] for uid in valid_users])
        sigma_arr = np.stack([user_sigma[uid] for uid in valid_users])
        diff = z_arr[:, None, :] - mu_arr[None, :, :]
        D = np.sum(diff ** 2 / sigma_arr[None, :, :], axis=2)

        for ui, uid in enumerate(valid_users):
            d_self = D[:, ui]
            d_others = np.delete(D[:, np.arange(len(valid_users))], ui, axis=1)
            margin = np.mean(d_others, axis=1) - d_self
            threshold = float(np.percentile(d_self, THRESHOLD_PCT))
            valid_mask = d_self <= threshold
            m = np.where(valid_mask, margin, -np.inf)
            sel_idx = int(np.argmax(m))
            # Original pool index
            orig_idx = valid_idx[sel_idx]
            sel_query_text = pool_by_asin[asin][orig_idx]["query"]
            selected_queries.append({
                "asin": asin,
                "user_id": uid,
                "query": sel_query_text,
                "selection_distance": float(d_self[sel_idx]),
            })
            selection_distances.append(float(d_self[sel_idx]))

    log(f"  total selected: {len(selected_queries)}")
    log(f"  mean selection_distance: {np.mean(selection_distances):.2f}")

    # === 7. Build corpus ===
    log("\n=== 7. Building corpus ===")
    asin_to_doc = {}
    with gzip.open(STAGE7_CORPUS_DOCS, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("asin")
            doc = rec.get("doc", "") or rec.get("text", "")
            if asin and doc:
                asin_to_doc[asin] = doc[:1000]
    asins = sorted(asin_to_doc.keys())
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    log(f"  corpus: {len(asins)} ASINs")

    queries = [r["query"] for r in selected_queries]
    target_indices = np.array([asin_to_idx.get(r["asin"], -1) for r in selected_queries])
    n_missing = int((target_indices < 0).sum())
    log(f"  queries: {len(queries)}, missing target: {n_missing}")

    # === 8. BM25 retrieval ===
    log("\n=== 8. BM25 retrieval (bm25s) ===")
    import bm25s
    corpus_texts = [asin_to_doc[a] for a in asins]
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    retriever.index(corpus_tokens, show_progress=False)
    query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
    bm25_results_raw = retriever.retrieve(query_tokens, k=len(corpus_texts), show_progress=False)
    log(f"  BM25 done in {time.time() - t0:.1f}s")

    bm25_results = []
    for i in range(len(queries)):
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            bm25_results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sorted_docs = bm25_results_raw.documents[i]
        positions = np.where(sorted_docs == tgt_idx)[0]
        if len(positions) == 0:
            bm25_results.append({"rank": None, "RR": 0.0, "hit10": 0})
        else:
            rank = int(positions[0]) + 1
            bm25_results.append({"rank": rank, "RR": 1.0 / rank, "hit10": 1 if rank <= 10 else 0})

    # === 9. MiniLM retrieval (GPU) ===
    log("\n=== 9. MiniLM retrieval (GPU) ===")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  device: {device}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    t0 = time.time()
    doc_embeds = model.encode(corpus_texts, batch_size=512, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)
    q_embeds = model.encode(queries, batch_size=512, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
    log(f"  MiniLM encoded in {time.time() - t0:.1f}s")
    doc_embeds_gpu = torch.from_numpy(doc_embeds).cuda()
    q_embeds_gpu = torch.from_numpy(q_embeds).cuda()
    sims = q_embeds_gpu @ doc_embeds_gpu.T
    minilm_results = []
    for i in range(sims.shape[0]):
        tgt_idx = target_indices[i]
        if tgt_idx < 0:
            minilm_results.append({"rank": None, "RR": 0.0, "hit10": 0})
            continue
        sc = sims[i]
        rank = int((sc > sc[tgt_idx]).sum().item()) + 1
        minilm_results.append({"rank": rank, "RR": 1.0 / rank, "hit10": 1 if rank <= 10 else 0})

    del doc_embeds_gpu, q_embeds_gpu, sims, model
    torch.cuda.empty_cache()

    # === 10. Save per-query ===
    log("\n=== 10. Saving per-query retrieval ===")
    per_query = []
    for r, bm25_r, minilm_r in zip(selected_queries, bm25_results, minilm_results):
        per_query.append({
            "asin": r["asin"],
            "user_id": r["user_id"],
            "query": r["query"],
            "selection_distance": r["selection_distance"],
            "variant": "selected_mean_t50",
            "bm25_rank": bm25_r["rank"],
            "bm25_RR": bm25_r["RR"],
            "bm25_hit10": bm25_r["hit10"],
            "minilm_rank": minilm_r["rank"],
            "minilm_RR": minilm_r["RR"],
            "minilm_hit10": minilm_r["hit10"],
        })
    with open(OUTPUT_PER_Q, "w") as f:
        json.dump(per_query, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_PER_Q}")

    # === 11. Compute volatility per ASIN ===
    log("\n=== 11. Per-ASIN volatility ===")
    asin_to_recs = defaultdict(list)
    for r in per_query:
        asin_to_recs[r["asin"]].append(r)

    def compute_flip_rate(hits):
        pairs = list(combinations(range(len(hits)), 2))
        if not pairs:
            return 0.0
        return sum(1 for i, j in pairs if hits[i] != hits[j]) / len(pairs)

    def compute_rr_std(rrs):
        if len(rrs) < 2:
            return 0.0
        return float(np.std(rrs, ddof=1))

    per_asin = {}
    for asin, recs in asin_to_recs.items():
        n = len(recs)
        bm25_ranks = [r["bm25_rank"] for r in recs]
        bm25_rrs = [r["bm25_RR"] for r in recs]
        bm25_hit1 = [1 if r["bm25_rank"] is not None and r["bm25_rank"] == 1 else 0 for r in recs]
        bm25_hit5 = [1 if r["bm25_rank"] is not None and r["bm25_rank"] <= 5 else 0 for r in recs]

        minilm_ranks = [r["minilm_rank"] for r in recs]
        minilm_rrs = [r["minilm_RR"] for r in recs]
        minilm_hit1 = [1 if r["minilm_rank"] is not None and r["minilm_rank"] == 1 else 0 for r in recs]
        minilm_hit5 = [1 if r["minilm_rank"] is not None and r["minilm_rank"] <= 5 else 0 for r in recs]

        per_asin[asin] = {
            "n_users": n,
            "n_pairs": n * (n - 1) // 2,
            "bm25_hit1_flip_rate": compute_flip_rate(bm25_hit1),
            "bm25_hit5_flip_rate": compute_flip_rate(bm25_hit5),
            "bm25_rr_std": compute_rr_std(bm25_rrs),
            "bm25_hit1_count": sum(bm25_hit1),
            "bm25_hit5_count": sum(bm25_hit5),
            "bm25_rr_mean": float(np.mean(bm25_rrs)),
            "bm25_rank_median": float(np.median([r for r in bm25_ranks if r is not None])),
            "minilm_hit1_flip_rate": compute_flip_rate(minilm_hit1),
            "minilm_hit5_flip_rate": compute_flip_rate(minilm_hit5),
            "minilm_rr_std": compute_rr_std(minilm_rrs),
            "minilm_hit1_count": sum(minilm_hit1),
            "minilm_hit5_count": sum(minilm_hit5),
            "minilm_rr_mean": float(np.mean(minilm_rrs)),
            "minilm_rank_median": float(np.median([r for r in minilm_ranks if r is not None])),
        }

    asins_list = sorted(per_asin.keys())
    log(f"  ASINs: {len(asins_list)}")

    with open(OUTPUT_VOLATILITY, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 10H-B: mean_t50 retrieval volatility",
                "n_asins": len(asins_list),
                "SEED": SEED,
                "N_BOOT": N_BOOT,
            },
            "per_asin": per_asin,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_VOLATILITY}")

    # === 12. BM25 vs MiniLM comparison ===
    log("\n=== 12. BM25 vs MiniLM comparison ===")

    def bootstrap_paired_diff(a, b, n_boot=N_BOOT, seed=SEED):
        rng_b = np.random.RandomState(seed)
        a = np.array(a)
        b = np.array(b)
        diff = a - b
        mean_diff = float(diff.mean())
        n = len(diff)
        if n < 2:
            return mean_diff, mean_diff, mean_diff, 1.0
        boot_diffs = []
        for _ in range(n_boot):
            idx = rng_b.randint(0, n, size=n)
            boot_diffs.append(diff[idx].mean())
        boot_diffs = np.array(boot_diffs)
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))
        p_ge0 = float((boot_diffs >= 0).mean())
        return mean_diff, ci_low, ci_high, p_ge0

    summary = {}
    for metric in ["hit1_flip_rate", "hit5_flip_rate", "rr_std"]:
        bm25_vals = np.array([per_asin[a][f"bm25_{metric}"] for a in asins_list])
        minilm_vals = np.array([per_asin[a][f"minilm_{metric}"] for a in asins_list])

        log(f"\n  {metric}:")
        log(f"    BM25:   mean={bm25_vals.mean()*100:.3f}%, median={np.median(bm25_vals)*100:.3f}%")
        log(f"    MiniLM: mean={minilm_vals.mean()*100:.3f}%, median={np.median(minilm_vals)*100:.3f}%")

        diff = bm25_vals - minilm_vals
        if np.all(diff == 0):
            w_stat, w_p = 0.0, 1.0
        else:
            try:
                w = wilcoxon(bm25_vals, minilm_vals, alternative="two-sided")
                w_stat = float(w.statistic)
                w_p = float(w.pvalue)
            except ValueError:
                w_stat, w_p = 0.0, 1.0
        log(f"    Wilcoxon: stat={w_stat:.1f}, p={w_p:.4f}")

        m_diff, ci_low, ci_high, p_ge0 = bootstrap_paired_diff(bm25_vals.tolist(), minilm_vals.tolist())
        log(f"    Bootstrap mean(B-M)={m_diff*100:.3f}%, CI=[{ci_low*100:.3f}%, {ci_high*100:.3f}%], p(B>M)={p_ge0:.3f}")

        summary[metric] = {
            "bm25_mean": float(bm25_vals.mean()),
            "minilm_mean": float(minilm_vals.mean()),
            "mean_diff_bm25_minus_minilm": m_diff,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "p_bm25_gt_minilm_bootstrap": p_ge0,
            "wilcoxon_p": w_p,
        }

    for metric in ["hit1_flip_rate", "hit5_flip_rate", "rr_std"]:
        bm25_better = sum(1 for a in asins_list if per_asin[a][f"bm25_{metric}"] < per_asin[a][f"minilm_{metric}"])
        equal = sum(1 for a in asins_list if per_asin[a][f"bm25_{metric}"] == per_asin[a][f"minilm_{metric}"])
        minilm_better = len(asins_list) - bm25_better - equal
        log(f"  {metric}: BM25 better in {bm25_better}/{len(asins_list)}, equal in {equal}, MiniLM better in {minilm_better}")
        summary[metric]["per_asin_bm25_better"] = bm25_better
        summary[metric]["per_asin_minilm_better"] = minilm_better

    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump({
            "config": {
                "description": "Stage 10H-B: mean_t50 BM25 vs MiniLM comparison",
                "metrics": ["hit1_flip_rate", "hit5_flip_rate", "rr_std"],
                "n_asins": len(asins_list),
            },
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUTPUT_SUMMARY}")

    log("\n=== Stage 10H-B complete ===")


if __name__ == "__main__":
    main()