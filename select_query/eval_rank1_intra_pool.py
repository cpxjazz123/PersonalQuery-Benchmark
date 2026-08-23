#!/usr/bin/env python3
"""Eval: 对每条 best_query, 算它在 9982 用户池里 Mahalanobis 距自身的 rank。

答案用户: rank=1 表示该 query 距自身 user 最近, 距其他 9981 users 都远
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")

sys.path.insert(0, str(REPO))

PROFILES = SCRATCH / "vades_10k_user_profiles.jsonl"
PCA_FILE = REPO / "result/pca_components_10k.npz"
BEST_CANDS = SCRATCH / "best_cands_10k.json"


def main() -> None:
    # 1. Load PCA components + mean
    pca_data = np.load(PCA_FILE, allow_pickle=True)
    components = pca_data["components"].item()[32]
    means = pca_data["means"].item()[32]
    print(f"[load] PCA32 components shape={components.shape}, mean shape={means.shape}")

    # 2. Load VADES user_mu (20-dim latent, but we need layer 26 hidden, not latent)
    # Actually pick_best_cand_10k uses Qwen residual → PCA32, NOT VADES latent
    # So we need user residuals from residual_hidden_10k.npz instead
    print("[load] residuals from residual_hidden_10k.npz ...")
    npz = np.load(SCRATCH / "residual_hidden_10k.npz", allow_pickle=True)
    sentences_arr = npz["sentences"]
    layer_key = "residual_layer_26"
    resids_arr = npz[layer_key]  # (N_sents, 3584)
    print(f"  residuals shape={resids_arr.shape}")

    # 3. Load sentence -> uid mapping from rewrites_10k.jsonl (dedup by sentence_text)
    # residual extraction deduped via dict{sentence_text: rewrite}, losing 512 dup user_ids
    # We replicate that same dedup (keep first uid per sentence_text) so order matches
    print("[load] sentence→uid mapping from rewrites_10k.jsonl (dedup) ...")
    sent_to_uid: dict[str, str] = {}
    with open(SCRATCH / "rewrites_10k.jsonl", "r") as f:
        for line in f:
            r = json.loads(line)
            sent = r.get("sentence_text", "")
            uid = r.get("user_id", "")
            if sent and sent not in sent_to_uid:
                sent_to_uid[sent] = uid
    # residuals npz saves sentences in encounter order; reconstruct the same order
    sent_uid = [sent_to_uid.get(s, "") for s in sentences_arr.tolist()]
    print(f"  {len(sent_uid)} sentences, unique uids={len(set(sent_uid) - {''})}")

    if len(sent_uid) != resids_arr.shape[0]:
        raise ValueError(
            f"Cannot align: rewrites has {len(sent_uid)} but residuals has {resids_arr.shape[0]}"
        )

    # 4. Group residuals by user_id (mean pool per user)
    print("[group] per-user mean residual ...")
    user_resids: dict[str, np.ndarray] = {}
    for uid, r in zip(sent_uid, resids_arr):
        if uid not in user_resids:
            user_resids[uid] = []
        user_resids[uid].append(r)
    user_mean = {
        uid: np.mean(np.stack(v, axis=0).astype(np.float32), axis=0)
        for uid, v in user_resids.items()
    }
    print(f"  {len(user_mean)} users with mean residual")

    # 5. Project user means to PCA32
    user_ids = sorted(user_mean.keys())
    user_mat = np.stack([user_mean[u] for u in user_ids], axis=0).astype(np.float32)
    user_pca32 = (user_mat - means[None, :]) @ components.T  # (N, 32)
    user_pca32_inv_sigma = user_pca32 / (np.std(user_pca32, axis=0) + 1e-9)  # for Maha
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    print(f"  user_pca32 shape={user_pca32.shape}")

    # 6. Load best_cands and encode via Qwen
    best_cands = json.load(open(BEST_CANDS, "r"))
    print(f"[encode] {len(best_cands)} best queries via Qwen layer 26 ...")
    from llm_client import create_qwen_local_client
    client = create_qwen_local_client(with_vllm=False)

    queries = [d["best_query"] for d in best_cands]
    q_hidden = {}
    BATCH = 64  # hidden extraction is memory-bound; bump from 16 → 64 for ~3x speedup
    t0 = time.time()
    for i in range(0, len(queries), BATCH):
        chunk = queries[i:i + BATCH]
        h_dict = client.get_hidden_states(chunk, layers=[26], max_length=256, batch_size=BATCH)
        h_layer = h_dict[26]
        if hasattr(h_layer, "cpu"):
            h_layer = h_layer.cpu().numpy()
        for q, h in zip(chunk, h_layer):
            q_hidden[q] = h
        if (i // BATCH) % 50 == 0:
            print(f"  [{i + len(chunk)}/{len(queries)}] {time.time() - t0:.1f}s", flush=True)
    print(f"  encoded in {time.time() - t0:.1f}s")

    # 7. Project best queries to PCA32
    q_mat = np.stack([q_hidden[d["best_query"]] for d in best_cands], axis=0).astype(np.float32)
    q_pca32 = (q_mat - means[None, :]) @ components.T  # (N, 32)

    # 8. For each best_query, compute Mahalanobis distance to ALL users and rank
    # Maha(q, u) = ||(q_pca32 - u_pca32) / sigma||^2 (smaller = closer)
    # To avoid 9982×9982×32 ≈ 13 GB tensor, compute per-query chunked
    sigma = np.std(user_pca32, axis=0) + 1e-9  # (32,)
    sigma_inv = 1.0 / sigma
    user_pca32_normed = user_pca32 * sigma_inv[None, :]  # (Nu, 32) — pre-normalize users
    q_pca32_normed = q_pca32 * sigma_inv[None, :]  # (Nq, 32)

    ranks = np.zeros(len(best_cands), dtype=np.int32)
    BATCH_Q = 256  # chunk queries; d_maha block = 256×9982×32×4 = 32 MB
    for q_start in range(0, len(best_cands), BATCH_Q):
        q_end = min(q_start + BATCH_Q, len(best_cands))
        q_chunk = q_pca32_normed[q_start:q_end]  # (Bq, 32)
        # d_maha_block[i,j] = ||q_i - u_j||^2 (with sigma-normalized)
        # = ||q_i||^2 + ||u_j||^2 - 2 q_i · u_j
        q_normsq = (q_chunk ** 2).sum(axis=1, keepdims=True)  # (Bq, 1)
        u_normsq = (user_pca32_normed ** 2).sum(axis=1)  # (Nu,)
        dot = q_chunk @ user_pca32_normed.T  # (Bq, Nu)
        d_block = q_normsq + u_normsq[None, :] - 2 * dot  # (Bq, Nu)
        # Rank of own user (column uidx is smallest distance?)
        for i in range(q_end - q_start):
            d = best_cands[q_start + i]
            uidx = user_id_to_idx.get(d["user_id"], -1)
            if uidx < 0:
                ranks[q_start + i] = -1
                continue
            d_to_own = d_block[i, uidx]
            n_closer = (d_block[i, :] < d_to_own).sum()
            ranks[q_start + i] = n_closer + 1
    print()
    print("=" * 60)
    print("RESULT: best_query intra-pool Mahalanobis rank against all 9982 users")
    print("=" * 60)
    print(f"Total queries: {len(ranks)}")
    print(f"Rank-1 (own user nearest): {(ranks == 1).sum()} ({(ranks == 1).mean() * 100:.2f}%)")
    print(f"Rank-5:                    {(ranks <= 5).sum()} ({(ranks <= 5).mean() * 100:.2f}%)")
    print(f"Rank-10:                   {(ranks <= 10).sum()} ({(ranks <= 10).mean() * 100:.2f}%)")
    print(f"Rank-50:                   {(ranks <= 50).sum()} ({(ranks <= 50).mean() * 100:.2f}%)")
    print(f"Rank-100:                  {(ranks <= 100).sum()} ({(ranks <= 100).mean() * 100:.2f}%)")
    print(f"Mean rank:                 {ranks.mean():.1f} / {len(user_ids)}")
    print(f"Median rank:               {np.median(ranks):.1f}")
    print(f"Min rank:                  {ranks.min()}, Max rank: {ranks.max()}")


if __name__ == "__main__":
    main()
