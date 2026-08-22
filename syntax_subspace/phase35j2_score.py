#!/usr/bin/env python3
"""Phase 35.J-2 scoring: Forced N=7 + L=36-50 candidates.

Use Phase 35.G softmax_g32_τ0.5 to score the 544 forced-generated candidates.
Compare to:
  - V2 N=7 K=4 raw: 50.7% Rank-1 (size>=2 filter)
  - V2 N=7 L=36-50 length-matched (n=15): 100% (post-hoc artifact)
  - V2 N=7 L=36-50 length-controlled (n=23): 69.6% / 2.49x lift
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
PHASE35E_DIR = REPO_ROOT / "result/phase35e"
OUT_DIR = REPO_ROOT / "result/phase35j2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace")
N_FIXED = 7
TAU = 0.5
PCA_D = 32


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def count_attrs_covered(text, attrs):
    if not attrs: return 0
    return sum(1 for v in attrs.values() if v and str(v).strip().lower() in text.lower())


def main():
    cache = __import__("torch").load(SCRATCH / "phase35e_user_qwen_residuals.pt", weights_only=False)
    all_resids = cache["residuals"]
    flat_meta = cache["meta"]
    user_resids = {}
    for i, m in enumerate(flat_meta):
        if m[0] == "user":
            user_resids.setdefault(m[1], []).append(all_resids[i])

    pca_data = np.load(PHASE35E_DIR / "pca_components.npz", allow_pickle=True)
    def _maybe_int(k):
        try: return int(k)
        except (ValueError, TypeError): return k
    pca_components = {_maybe_int(k): v for k, v in pca_data["components"].item().items()}
    pca_means = {_maybe_int(k): v for k, v in pca_data["means"].item().items()}

    user_z_per_d = {}
    for d in [PCA_D]:
        for uid, vecs in user_resids.items():
            V = np.stack(vecs, axis=0).astype(np.float32)
            Z = (V - pca_means[d]) @ pca_components[d].T
            user_z_per_d.setdefault(d, {})[uid] = Z

    user_pca_mu = {}
    pca_global_inv_sigma = {}
    for d in [PCA_D]:
        var_global = np.stack([user_z_per_d[d][u].var(axis=0) for u in user_z_per_d[d]]).mean(axis=0)
        pca_global_inv_sigma[d] = 1.0 / np.maximum(var_global, 1e-6)
        for uid in user_z_per_d[d]:
            user_pca_mu[(uid, d)] = user_z_per_d[d][uid].mean(axis=0)

    def maha_global(cand_z, uid, d):
        mu = user_pca_mu[(uid, d)]
        inv_sigma = pca_global_inv_sigma[d]
        diffs = cand_z - mu
        return float(np.sqrt(np.maximum((diffs * diffs * inv_sigma).sum(), 1e-8)))

    # Load forced cands
    cands = json.load(open(SCRATCH / "phase35j2_forced_cands.json"))
    print(f"[load] {len(cands)} forced cands (N={N_FIXED})")

    # Compute cand residuals (no cache, need fresh Qwen forward... actually reuse Qwen via batch encode)
    print("[main] loading Qwen for cand residuals...")
    sys.path.insert(0, str(REPO_ROOT))
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)
    LAYERS = [26]
    BATCH = 16

    # Filter to only those with target asin in Phase 35.B records pool
    # Build per-asin pool
    asin_to_recs = {}
    for i, c in enumerate(cands):
        asin_to_recs.setdefault(c["asin"], []).append(i)
    print(f"[pool] {len(asin_to_recs)} asins from forced cands")

    # Encode all cand texts in batch
    cand_residuals = np.zeros((len(cands), 3584), dtype=np.float32)
    for i in range(0, len(cands), BATCH):
        chunk = [c["query"] for c in cands[i:i+BATCH]]
        hidden_dict = client.get_hidden_states(chunk, layers=LAYERS, max_length=256, batch_size=BATCH)
        cand_residuals[i:i+BATCH] = hidden_dict[LAYERS[0]]
        done = min(i + BATCH, len(cands))
        if done % 64 == 0 or done == len(cands):
            print(f"  [resid] {done}/{len(cands)}")

    # Project to PCA32
    cand_z = (cand_residuals - pca_means[PCA_D]) @ pca_components[PCA_D].T  # [n, 32]

    # === Eval: per (asin, target_uid) Rank-1 ===
    print("\n[eval] building per-asin score matrix...")
    per_record_hits = []
    per_record_pool = []
    per_record_mean_rank = []
    per_record_L = []
    per_record_cov = []
    skipped_no_user_pool = 0
    for asin, idxs in asin_to_recs.items():
        pool = []
        for i in idxs:
            c = cands[i]
            cov = count_attrs_covered(c["query"], c["attrs_used"])
            L = len(c["query"].split())
            pool.append({"uid": c["user_id"], "z32": cand_z[i], "L": L, "cov": cov})
        size = len(pool)
        if size < 2: continue
        uids_in_pool = list({p["uid"] for p in pool})
        if len(uids_in_pool) < 2:
            skipped_no_user_pool += 1
            continue
        n_users = len(uids_in_pool)
        uid_to_idx = {u: i for i, u in enumerate(uids_in_pool)}
        D_g32 = np.zeros((size, n_users))
        for j, v in enumerate(uids_in_pool):
            D_g32[:, j] = [maha_global(p["z32"], v, PCA_D) for p in pool]
        for uid in uids_in_pool:
            target_idx = uid_to_idx[uid]
            logits = -D_g32 / TAU
            probs = softmax(logits, axis=1)
            scores = probs[:, target_idx]
            order = np.argsort(-scores)
            rank1_uid = pool[int(order[0])]["uid"]
            hit = rank1_uid == uid
            own_indices = [i for i, p in enumerate(pool) if p["uid"] == uid]
            if not own_indices: continue
            own_ranks = [int(np.where(order == i)[0][0]) for i in own_indices]
            per_record_hits.append(1 if hit else 0)
            per_record_pool.append(n_users)
            per_record_mean_rank.append(float(np.mean(own_ranks)))
            per_record_L.extend(p["L"] for p in pool if p["uid"] == uid)
            per_record_cov.extend(p["cov"] / N_FIXED for p in pool if p["uid"] == uid)

    n_records = len(per_record_hits)
    rank1 = np.mean(per_record_hits) * 100
    mean_pool = np.mean(per_record_pool)
    mean_rank = np.mean(per_record_mean_rank)
    random_acc = np.mean([1.0 / s for s in per_record_pool]) * 100
    lift = rank1 / random_acc if random_acc > 0 else 0
    avg_L = np.mean(per_record_L) if per_record_L else 0
    avg_cov = np.mean(per_record_cov) * 100 if per_record_cov else 0

    print(f"\n{'='*70}")
    print(f"Phase 35.J-2 forced gen scoring (N=7 L=36-50, K={len(cands)//len(set(c['asin'] for c in cands))} per record)")
    print(f"{'='*70}")
    print(f"  n_records: {n_records} (skipped no-user-pool asins: {skipped_no_user_pool})")
    print(f"  pool avg:  {mean_pool:.2f}")
    print(f"  Rank-1:    {rank1:.1f}%   (random = {random_acc:.2f}%, lift = {lift:.2f}x)")
    print(f"  mean_rank: {mean_rank:.2f}")
    print(f"  avg L:     {avg_L:.1f} (target: 36-50)")
    print(f"  avg cov:   {avg_cov:.1f}%")
    print(f"\n  Reference comparisons:")
    print(f"    Phase 35.G/I-v2 SOTA  (V2 N=7 K=4 raw):        50.7% (lift 2.27x)")
    print(f"    V2 N=7 L=36-50 length-matched (n=15):         100.0% (pool artifact)")
    print(f"    V2 N=7 L=36-50 length-controlled (n=23, full pool): 69.6% / 2.49x lift")

    json.dump({
        "config": {"N": N_FIXED, "tau": TAU, "pca_d": PCA_D, "target_L": "36-50", "n_cands": len(cands)},
        "forced": {
            "n_records": n_records,
            "pool_avg": mean_pool,
            "rank1_pct": rank1,
            "random_acc": random_acc,
            "lift_vs_random": lift,
            "mean_rank": mean_rank,
            "avg_L": avg_L,
            "avg_cov_pct": avg_cov,
        },
        "comparison": {
            "v2_n7_k4_raw": 50.7,
            "v2_n7_L36_50_length_matched_n15": 100.0,
            "v2_n7_L36_50_length_controlled_n23": 69.6,
        },
    }, open(OUT_DIR / "j2_forced_summary.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n[save] → {OUT_DIR}/j2_forced_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
