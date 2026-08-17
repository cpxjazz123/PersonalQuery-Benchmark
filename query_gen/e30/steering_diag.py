"""E30 steering diagnostic — check if user embeddings are sufficiently
discriminative for TinyStyler style injection to work.

Reads e30_style_embs.npz, computes pairwise cos sim among 10 users used in the
1-product smoke run, and reports min/median/max cos sim.
"""
import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np

PAPER = '/home/wlia0047/hj82_scratch2/wenyu/e29_paper'

# load user embs
npz = np.load(f'{PAPER}/e30_style_embs.npz', allow_pickle=True)
uids_arr = npz['user_ids']
embs = npz['embs']  # (n_users, 768)
uid_to_idx = {str(u): i for i, u in enumerate(uids_arr)}

# load the 10 users from the 1-product run
with open(f'{PAPER}/e30_styled_queries.jsonl') as f:
    pairs = [json.loads(l) for l in f]
print(f'Pairs: {len(pairs)}')

# gather user ids from records
sample_users = []
seen = set()
for p in pairs:
    if p['user_id'] not in seen:
        sample_users.append(p['user_id'])
        seen.add(p['user_id'])

print(f'Unique users in styled run: {len(sample_users)}')
for u in sample_users:
    print(f'  {u}')

# also pull 100 random users for broader stats
rng = np.random.default_rng(42)
random_users = [str(uids_arr[i]) for i in rng.choice(len(uids_arr), 100, replace=False)]
sample_users_strs = [str(u) for u in sample_users]

# extract embeddings
def get_emb(uid_str):
    idx = uid_to_idx[uid_str]
    return embs[idx]

sample_embs = np.stack([get_emb(u) for u in sample_users_strs])  # (10, 768)
random_embs = np.stack([get_emb(u) for u in random_users])  # (100, 768)

# pairwise cos sim for the 10 users (same source query)
def cos_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + 1e-12)
    return Xn @ Xn.T

print('\n=== 10-user pairwise cos sim (the ones in styled run) ===')
m = cos_sim_matrix(sample_embs)
print('matrix shape:', m.shape)
for i in range(len(sample_users_strs)):
    for j in range(i + 1, len(sample_users_strs)):
        print(f'  ({sample_users_strs[i][:8]}..,{sample_users_strs[j][:8]}..) = {m[i, j]:.4f}')

# stats: mean cos, min cos (most diverse pair), max cos
all_pairs = []
for i in range(len(sample_users_strs)):
    for j in range(i + 1, len(sample_users_strs)):
        all_pairs.append(m[i, j])
all_pairs = np.array(all_pairs)
print(f'\n10-user stats: mean={all_pairs.mean():.4f} std={all_pairs.std():.4f} '
      f'min={all_pairs.min():.4f} (most diverse) max={all_pairs.max():.4f} (most similar)')

# broader check: 100 random users
m100 = cos_sim_matrix(random_embs)
pairs100 = m100[np.triu_indices(100, k=1)]
print(f'\n100-user stats: mean={pairs100.mean():.4f} std={pairs100.std():.4f} '
      f'min={pairs100.min():.4f} max={pairs100.max():.4f}')

# norm stats
norms = np.linalg.norm(sample_embs, axis=1)
print(f'\n10-user norm stats: mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}')

# distance stats
dists = 1 - all_pairs
print(f'\n10-user euclidean-cos distance: mean={dists.mean():.4f} max={dists.max():.4f}')
print(f'  → if max_dist < 0.05, vectors are essentially identical and TinyStyler cannot distinguish users')