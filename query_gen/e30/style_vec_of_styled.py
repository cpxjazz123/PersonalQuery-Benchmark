"""E30.12 — extract style vectors from styled queries and check spread.

Take the 10 styled queries from e30_styled_queries.jsonl (TinyStyler K=8/α=2),
encode with AnnaWegmann → 10 768-dim vectors.  Measure:

1. **Pairwise cos sim** among styled-query vectors — should be LOW (different users
   produce different styles) → diversity check.
2. **Self cos sim** — cos(styled_emb(user_i), user_style_emb_i) — should be HIGH
   if style transfer actually transferred.
3. **Cross cos sim** — cos(styled_emb(user_i), user_style_emb_j) for j ≠ i — should
   be LOWER than self (style matching).
4. Compare against:
   - baseline query embeddings (zero-style, should be ~uniform)
   - source query embedding (single point)

Output: spread summary + heatmap-like pairwise matrix.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = '/fs04/ar57/wenyu/.cache/huggingface'

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'
ANNAWEGMANN_SNAP = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--AnnaWegmann--Style-Embedding/snapshots/d7d0f5ca829316a8f5695e49dfce80b86db5e76c'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cos_sim_matrix(X):
    """X: [N, D] → [N, N] cos sim matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    return Xn @ Xn.T


def main():
    log("loading AnnaWegmann...")
    aw = SentenceTransformer(ANNAWEGMANN_SNAP).to('cuda:0').eval()

    # load e30_styled_queries.jsonl (10 records, all same source_query)
    with open(f'{PAPER}/e30_styled_queries.jsonl') as f:
        pairs = [json.loads(l) for l in f]
    log(f"  {len(pairs)} records")

    # unique users
    users = []
    seen = set()
    for p in pairs:
        if p['user_id'] not in seen:
            users.append(p['user_id'])
            seen.add(p['user_id'])
    log(f"  {len(users)} unique users")

    # collect texts
    source_query = pairs[0]['source_query']
    styled_texts = []
    baseline_texts = []
    user_idx_in_pairs = []
    for p in pairs:
        styled_texts.append(p['styled_query'])
        baseline_texts.append(p['baseline_query'])
        user_idx_in_pairs.append(users.index(p['user_id']))

    log(f"\n  source_query: {source_query[:100]!r}...")

    # encode all texts with AnnaWegmann
    log("encoding styled + baseline + source...")
    with torch.no_grad():
        styled_embs = aw.encode(styled_texts, convert_to_tensor=True,
                                 normalize_embeddings=True, show_progress_bar=False).cpu().numpy()
        baseline_embs = aw.encode(baseline_texts, convert_to_tensor=True,
                                   normalize_embeddings=True, show_progress_bar=False).cpu().numpy()
        src_emb = aw.encode([source_query], convert_to_tensor=True,
                              normalize_embeddings=True, show_progress_bar=False)[0].cpu().numpy()

    # load user style embeddings (AnnaWegmann 768-dim)
    npz = np.load(f'{PAPER}/e30_style_embs.npz', allow_pickle=True)
    uids_arr = npz['user_ids']
    embs = npz['embs']
    uid_to_emb = {str(u): embs[i] for i, u in enumerate(uids_arr)}
    user_style = np.stack([uid_to_emb[u] for u in users])
    # normalize
    user_style_n = user_style / (np.linalg.norm(user_style, axis=1, keepdims=True) + 1e-12)

    # ---- 1. pairwise cos sim within styled ----
    log("\n=== 1. pairwise cos sim within STYLED queries (diversity) ===")
    m = cos_sim_matrix(styled_embs)
    pairs_sim = m[np.triu_indices(len(users), k=1)]
    log(f"  mean={pairs_sim.mean():.4f}  std={pairs_sim.std():.4f}  "
        f"min={pairs_sim.min():.4f}  max={pairs_sim.max():.4f}")
    log(f"  range: {pairs_sim.max()-pairs_sim.min():.4f}")
    log(f"  → LOW mean & HIGH range = good style diversity")

    # ---- 2. pairwise cos sim within BASELINE (control) ----
    log("\n=== 2. pairwise cos sim within BASELINE queries (zero-style control) ===")
    m_b = cos_sim_matrix(baseline_embs)
    pairs_b = m_b[np.triu_indices(len(users), k=1)]
    log(f"  mean={pairs_b.mean():.4f}  std={pairs_b.std():.4f}  "
        f"min={pairs_b.min():.4f}  max={pairs_b.max():.4f}")
    log(f"  → HIGH mean & LOW range = all baseline similar (expected)")

    # ---- 3. self vs cross: cos(styled_i, user_i_style) vs cos(styled_i, user_j_style) ----
    log("\n=== 3. SELF vs CROSS: cos(styled_emb_i, user_style_emb_i vs j) ===")
    styled_embs_n = styled_embs / (np.linalg.norm(styled_embs, axis=1, keepdims=True) + 1e-12)
    sim_matrix = styled_embs_n @ user_style_n.T  # [N, N]
    self_sim = np.diag(sim_matrix)
    cross_sim = sim_matrix[~np.eye(len(users), dtype=bool)]
    log(f"  SELF (cos(styled_i, user_i_emb)):  mean={self_sim.mean():.4f}  std={self_sim.std():.4f}  min={self_sim.min():.4f}  max={self_sim.max():.4f}")
    log(f"  CROSS(cos(styled_i, user_j_emb)): mean={cross_sim.mean():.4f}  std={cross_sim.std():.4f}  min={cross_sim.min():.4f}  max={cross_sim.max():.4f}")
    gap = self_sim.mean() - cross_sim.mean()
    log(f"  SELF − CROSS gap = {gap:+.4f}  → positive = style transfer effective")

    # also: cos(baseline_i, user_i_style) vs cos(baseline_i, user_j_style)
    log("\n=== 4. SELF vs CROSS for BASELINE (control) ===")
    baseline_embs_n = baseline_embs / (np.linalg.norm(baseline_embs, axis=1, keepdims=True) + 1e-12)
    sim_matrix_b = baseline_embs_n @ user_style_n.T
    self_b = np.diag(sim_matrix_b)
    cross_b = sim_matrix_b[~np.eye(len(users), dtype=bool)]
    log(f"  SELF:  mean={self_b.mean():.4f}")
    log(f"  CROSS: mean={cross_b.mean():.4f}")
    log(f"  gap = {self_b.mean()-cross_b.mean():+.4f}")

    # ---- 5. cos to source query ----
    log("\n=== 5. cos(styled_emb, source_emb) ===")
    src_emb_n = src_emb / (np.linalg.norm(src_emb) + 1e-12)
    cos_src = styled_embs_n @ src_emb_n
    log(f"  mean={cos_src.mean():.4f}  std={cos_src.std():.4f}  min={cos_src.min():.4f}  max={cos_src.max():.4f}")

    # ---- 6. Cos between user style embs themselves (baseline reference) ----
    log("\n=== 6. pairwise cos sim among USER STYLE EMBEDDINGS (AnnaWegmann reference) ===")
    m_u = cos_sim_matrix(user_style_n)
    pairs_u = m_u[np.triu_indices(len(users), k=1)]
    log(f"  mean={pairs_u.mean():.4f}  std={pairs_u.std():.4f}  "
        f"min={pairs_u.min():.4f}  max={pairs_u.max():.4f}")
    log(f"  range: {pairs_u.max()-pairs_u.min():.4f}")
    log(f"  → This is the upper bound of achievable spread in style space")

    # ---- 7. cos(src_emb, each user_style) ----
    log("\n=== 7. cos(source_emb, user_style_emb) ===")
    cos_src_users = user_style_n @ src_emb_n
    log(f"  mean={cos_src_users.mean():.4f}  std={cos_src_users.std():.4f}")

    # save full sim matrix
    out = {
        'users': users,
        'styled_embs': styled_embs.tolist(),
        'baseline_embs': baseline_embs.tolist(),
        'user_style_embs': user_style.tolist(),
        'source_emb': src_emb.tolist(),
        'styled_pairwise_cos': m.tolist(),
        'baseline_pairwise_cos': m_b.tolist(),
        'user_style_pairwise_cos': m_u.tolist(),
        'styled_to_user_style_sim': sim_matrix.tolist(),
        'baseline_to_user_style_sim': sim_matrix_b.tolist(),
    }
    out_path = f'{PAPER}/e30_style_vec_of_styled.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"\n  wrote {out_path}")

    # final summary
    log("\n" + "="*70)
    log("SUMMARY")
    log("="*70)
    log(f"  spread(styled_queries) = {pairs_sim.max()-pairs_sim.min():.4f}")
    log(f"  spread(baseline_queries) = {pairs_b.max()-pairs_b.min():.4f}")
    log(f"  spread(user_style_embs) = {pairs_u.max()-pairs_u.min():.4f}")
    log(f"  style_transfer_gap (self-cross) = {gap:+.4f}")
    log(f"  → interpret: if styled spread >> baseline spread AND")
    log(f"    style_transfer_gap >> 0, TinyStyler actually moved style")


if __name__ == '__main__':
    main()