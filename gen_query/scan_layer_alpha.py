#!/usr/bin/env python3
"""Scan injection layer × alpha × user review-count bucket.

Design:
  - 5 users × 4 buckets = 20 users (deterministic seed)
  - Each user: generate K=2 queries × 2 conditions (baseline=alpha=0, injected=alpha)
  - Pre-encode all unique queries via Qwen server once
  - Score via PCA32 Mahalanobis distance to user_mu
  - Report lift = D_baseline - D_injected (positive = injected is better)

Grid: 5 layers × 3 alphas = 15 conditions × 20 users × 2 queries = 600 generations
Expected time: ~600 × 1s = ~10 min

Outputs:
  /home/wlia0047/hj82_scratch2/wenyu/user_style_steering/layer_alpha_scan.json
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
SCRATCH_G = SCRATCH / "gaussian_vades"
SCRATCH_U = SCRATCH / "user_style_steering"
SCRATCH_U.mkdir(parents=True, exist_ok=True)

REWRITES = SCRATCH_G / "rewrites_10k.jsonl"
NPZ_FILE = SCRATCH_G / "residual_hidden_10k.npz"
RECORDS_IN = REPO / "result/query_records_10k.json"
PCA_OUT = SCRATCH_G / "pca_components_10k.npz"
STYLE_VEC_OUT = SCRATCH_U / "user_style_vectors_real_10k.jsonl"
OUTPUT_JSON = SCRATCH_U / "layer_alpha_scan.json"

# Scan grid
SCAN_LAYERS = [14, 16, 20, 24, 26]
SCAN_ALPHAS = [0.5, 1.0, 2.0]
GEN_K = 2
N_PER_BUCKET = 5
PCA_D = 32
TAU = 0.5
LAYERS_KEY = [16, 20, 24, 26]

BUCKETS = [
    ("1-5",   1,   5),
    ("5-10",  5,  10),
    ("10-15", 10,  15),
]


def load_user_vectors() -> tuple[dict[str, dict[int, np.ndarray]], dict[str, int]]:
    """Returns (user_vecs, user_review_count). Always rebuilds to get accurate review counts."""
    print("[build] style vectors...")
    sent_to_uid = {}
    with open(REWRITES, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sent = r.get("sentence_text", "")
            uid = r.get("user_id", "")
            if sent and sent not in sent_to_uid:
                sent_to_uid[sent] = uid

    npz = np.load(NPZ_FILE, allow_pickle=True)
    sentences = npz["sentences"].tolist()
    layer_resids = {L: npz[f"residual_layer_{L}"] for L in LAYERS_KEY}

    user_vecs = defaultdict(lambda: {L: [] for L in LAYERS_KEY})
    user_review_count: dict[str, int] = defaultdict(int)
    for sent_idx, sent in enumerate(sentences):
        uid = sent_to_uid.get(sent)
        if uid is None:
            continue
        user_review_count[uid] += 1
        for L in LAYERS_KEY:
            user_vecs[uid][L].append(layer_resids[L][sent_idx])

    print(f"[write] {len(user_vecs)} users → {STYLE_VEC_OUT}")
    with open(STYLE_VEC_OUT, "w", encoding="utf-8") as fh:
        for uid in sorted(user_vecs):
            row = {"user_id": uid}
            for L in LAYERS_KEY:
                vecs = user_vecs[uid][L]
                mean_vec = (np.stack(vecs, axis=0).mean(axis=0) if vecs
                            else np.zeros(3584, dtype=np.float32))
                row[f"residual_mean_layer_{L}"] = mean_vec.tolist()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return (
        {uid: {L: (np.stack(user_vecs[uid][L], axis=0).mean(axis=0) if user_vecs[uid][L]
                   else np.zeros(3584, dtype=np.float32)).astype(np.float32)
                for L in LAYERS_KEY}
         for uid in user_vecs},
        dict(user_review_count),
    )


def load_pca():
    if PCA_OUT.exists():
        print(f"[PCA] loading: {PCA_OUT}")
        pca_data = np.load(PCA_OUT, allow_pickle=True)
        components = {int(k): v for k, v in pca_data["components"].item().items()}
        means = {int(k): v for k, v in pca_data["means"].item().items()}
        return components, means

    print(f"[PCA] computing PCA{PCA_D}...")
    npz = np.load(NPZ_FILE, allow_pickle=True)
    all_resids = npz["residual_layer_26"].astype(np.float32)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_D, random_state=42)
    pca.fit(all_resids)
    comp = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    np.savez_compressed(PCA_OUT,
                        components=np.array([{PCA_D: comp}], dtype=object),
                        means=np.array([{PCA_D: mean}], dtype=object))
    print(f"[PCA] explained={pca.explained_variance_ratio_.sum():.3f}")
    return {PCA_D: comp}, {PCA_D: mean}


def main() -> None:
    t0 = time.time()

    # ── Load user vectors ──────────────────────────────────────────
    user_vecs, user_residual_count = load_user_vectors()
    print(f"[users] {len(user_vecs)}")

    # ── Bucket users by residual review count ───────────────────────
    bucket_users = defaultdict(list)
    uid_to_bucket_all = {}
    for uid in user_vecs:
        cnt = user_residual_count.get(uid, 0)
        for bname, lo, hi in BUCKETS:
            if lo <= cnt < hi:
                bucket_users[bname].append(uid)
                uid_to_bucket_all[uid] = bname
                break

    print("\n[Bucket sizes]")
    for bname, _, _ in BUCKETS:
        print(f"  {bname}: {len(bucket_users[bname])}")

    # Deterministic sample
    np.random.seed(42)
    sampled_users = []
    uid_to_bucket = uid_to_bucket_all  # share the bucket mapping
    for bname, _, _ in BUCKETS:
        cand = bucket_users[bname]
        chosen = list(np.random.choice(cand, min(N_PER_BUCKET, len(cand)), replace=False))
        sampled_users.extend(chosen)
    print(f"\n[Sampled] {len(sampled_users)} users")

    # ── User records ──────────────────────────────────────────────
    records = json.load(open(RECORDS_IN, encoding="utf-8"))
    rec_by_uid = defaultdict(list)
    for r in records:
        if r["user_id"] in set(sampled_users):
            rec_by_uid[r["user_id"]].append(r)
    user_records = {uid: rec_by_uid[uid][0] for uid in sampled_users if uid in rec_by_uid}
    print(f"[Records] {len(user_records)} sampled users with records")

    # ── PCA ──────────────────────────────────────────────────────
    components, pca_means = load_pca()
    comp = components[PCA_D]
    mu_all = pca_means[PCA_D]

    # ── User PCA projections ───────────────────────────────────────
    user_z = {}
    for uid in sampled_users:
        vec = user_vecs[uid][26]  # project using layer 26
        user_z[uid] = (vec - mu_all) @ comp.T

    all_z = np.stack([user_z[u] for u in sampled_users])
    var_global = all_z.var(axis=0)
    inv_sigma = 1.0 / np.maximum(var_global, 1e-6)
    print(f"[sigma] mean_var={var_global.mean():.4f}")

    # ── Pre-generate queries via Qwen server ───────────────────────
    print("\n[gen] Generating queries via Qwen hidden server...")
    sys.path.insert(0, str(REPO))
    from llm_client import QwenLocalClient
    from gen_query.generate_strict_nohallu import make_prompt

    # Build all (uid, layer, alpha, k) tasks
    tasks = []
    for uid in sampled_users:
        rec = user_records[uid]
        attrs = rec.get("attrs_used", {})
        prompt = make_prompt(attrs, variant_idx=0)
        for layer in SCAN_LAYERS:
            bias = user_vecs[uid].get(layer, np.zeros(3584, dtype=np.float32))
            for alpha in SCAN_ALPHAS:
                for k in range(GEN_K):
                    tasks.append({
                        "uid": uid,
                        "layer": layer,
                        "alpha": alpha,
                        "k": k,
                        "bias": bias,
                        "prompt": prompt,
                        "bucket": uid_to_bucket[uid],
                    })
    print(f"  Total tasks: {len(tasks)} (15 conds × {len(sampled_users)} users × {GEN_K} k)")

    # ── Batched generation: one call per (layer, alpha, k) condition ─
    HIDDEN_DIM = 3584
    results_by_cond = {f"L{l}_A{a}": [] for l in SCAN_LAYERS for a in SCAN_ALPHAS}
    results_by_cond["baseline"] = []

    t_gen = time.time()
    done_conds = 0
    total_conds = len(SCAN_LAYERS) * len(SCAN_ALPHAS) * GEN_K

    client = QwenLocalClient(with_vllm=False)

    for layer in SCAN_LAYERS:
        for alpha in SCAN_ALPHAS:
            for k in range(GEN_K):
                # Collect all user prompts/biases for this condition
                prompts = []
                biases_t = []
                task_meta = []  # (uid, bucket)
                for uid in sampled_users:
                    rec = user_records[uid]
                    attrs = rec.get("attrs_used", {})
                    prompt = make_prompt(attrs, variant_idx=k)
                    bias = user_vecs[uid].get(layer, np.zeros(HIDDEN_DIM, dtype=np.float32))
                    prompts.append(prompt)
                    biases_t.append(torch.as_tensor(bias, dtype=torch.float32))
                    task_meta.append((uid, uid_to_bucket[uid]))

                # One call: injected
                try:
                    inj_results = client.generate_with_hidden_injection(
                        system_text="You are a helpful assistant.",
                        user_texts=prompts,
                        injection_per_row=biases_t,
                        injection_layers=[layer],
                        injection_alpha=alpha,
                        max_new_tokens=64,
                        temperature=0.7,
                        top_p=0.95,
                        repetition_penalty=1.1,
                        batch_size=len(prompts),
                        max_input_length=256,
                        mask_cjk=True,
                    )
                except Exception as e:
                    print(f"  ERROR L{layer}_A{alpha}_k{k}: {e}")
                    inj_results = [None] * len(prompts)

                for (uid, bucket), q in zip(task_meta, inj_results):
                    results_by_cond[f"L{layer}_A{alpha}"].append({
                        "uid": uid, "bucket": bucket, "query": q, "layer": layer, "alpha": alpha, "k": k,
                    })

                # One call: baseline (alpha=0)
                zero_biases = [torch.zeros(HIDDEN_DIM, dtype=torch.float32)] * len(prompts)
                try:
                    base_results = client.generate_with_hidden_injection(
                        system_text="You are a helpful assistant.",
                        user_texts=prompts,
                        injection_per_row=zero_biases,
                        injection_layers=[layer],
                        injection_alpha=0.0,
                        max_new_tokens=64,
                        temperature=0.7,
                        top_p=0.95,
                        repetition_penalty=1.1,
                        batch_size=len(prompts),
                        max_input_length=256,
                        mask_cjk=True,
                    )
                except Exception as e:
                    print(f"  ERROR baseline L{layer}_A{alpha}_k{k}: {e}")
                    base_results = [None] * len(prompts)

                for (uid, bucket), q in zip(task_meta, base_results):
                    results_by_cond["baseline"].append({
                        "uid": uid, "bucket": bucket, "query": q, "layer": layer, "k": k,
                    })

                done_conds += 1
                elapsed = time.time() - t_gen
                rate = done_conds / elapsed if elapsed > 0 else 0
                eta = (total_conds - done_conds) / rate if rate > 0 else 0
                print(f"  [{done_conds}/{total_conds}] L{layer}_A{alpha}_k{k} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    print(f"\n[gen] Done in {time.time()-t_gen:.0f}s")

    # ── Deduplicate queries for encoding ───────────────────────────
    all_queries = {}
    for cond, items in results_by_cond.items():
        for item in items:
            q = item["query"]
            if q not in all_queries:
                all_queries[q] = {"uid": item["uid"], "bucket": item["bucket"], "layer": item.get("layer"), "cond": cond}

    print(f"\n[encode] {len(all_queries)} unique queries...")
    from llm_client import QwenLocalClient
    q_hiddens = {}
    q_list = list(all_queries.keys())
    batch_size = 16
    enc_client = QwenLocalClient(with_vllm=False)
    for i in range(0, len(q_list), batch_size):
        chunk = q_list[i:i + batch_size]
        try:
            hidden_dict = enc_client.get_hidden_states(
                chunk, layers=[26], max_length=256, batch_size=batch_size
            )
        except Exception as e:
            print(f"  encode ERROR: {e}")
            continue
        h = hidden_dict[26]
        if hasattr(h, "numpy"):
            h = h.numpy()
        for q, h_vec in zip(chunk, h):
            q_hiddens[q] = h_vec.astype(np.float32)
        print(f"  [{min(i+batch_size, len(q_list))}/{len(q_list)}]", flush=True)

    # ── Score all ─────────────────────────────────────────────────
    print("\n[score] Computing Mahalanobis distances...")
    scores = {cond: defaultdict(list) for cond in results_by_cond}

    for cond, items in results_by_cond.items():
        for item in items:
            q = item["query"]
            uid = item["uid"]
            if q not in q_hiddens:
                continue
            z_q = (q_hiddens[q] - mu_all) @ comp.T
            mu_u = user_z[uid]
            d = float(np.sqrt(np.maximum(((z_q - mu_u) ** 2 * inv_sigma).sum(), 1e-8)))
            scores[cond][item["bucket"]].append(d)
            scores[cond]["__all__"].append(d)

    # ── Aggregate ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Condition':<12} {'Bucket':<8} {'N':>4} {'Mean_D':>8} {'Std_D':>8}")
    print("="*70)

    best = None
    best_lift = -999

    for layer in SCAN_LAYERS:
        for alpha in SCAN_ALPHAS:
            cond = f"L{layer}_A{alpha}"
            baseline_scores = scores.get("baseline", {}).get("__all__", [])
            injected_scores = scores.get(cond, {}).get("__all__", [])

            if not baseline_scores or not injected_scores:
                continue

            mean_base = np.mean(baseline_scores)
            mean_inj = np.mean(injected_scores)
            lift = mean_base - mean_inj  # positive = injected is closer

            print(f"\n{cond}")
            for bname, _, _ in BUCKETS:
                bs = scores[cond].get(bname, [])
                os = scores["baseline"].get(bname, [])
                if bs and os:
                    print(f"  {bname:<8} N={len(bs):>3}  D_inj={np.mean(bs):.3f}  lift={np.mean(os)-np.mean(bs):+.3f}")

            print(f"  OVERALL   N={len(injected_scores):>3}  D_base={mean_base:.3f}  D_inj={mean_inj:.3f}  lift={lift:+.3f}")

            if lift > best_lift:
                best_lift = lift
                best = cond

    print(f"\n{'='*70}")
    print(f"BEST: {best} with lift={best_lift:+.3f}")

    # ── Save ──────────────────────────────────────────────────────
    output = {
        "scan_grid": {
            "layers": SCAN_LAYERS,
            "alphas": SCAN_ALPHAS,
            "k": GEN_K,
            "n_per_bucket": N_PER_BUCKET,
            "pca_d": PCA_D,
        },
        "sampled_users": {u: uid_to_bucket[u] for u in sampled_users},
        "best_condition": best,
        "best_lift": best_lift,
        "scores": {
            cond: {b: [float(x) for x in vals] for b, vals in bucket_scores.items()}
            for cond, bucket_scores in scores.items()
        },
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    print(f"\n[saved] → {OUTPUT_JSON}")
    print(f"[total time] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
