"""
构建用户 embedding 缓存（MPNet 768d → PCA 128d）

从 result/03_gaussian/uid_to_sentences.pkl 读取用户句子，
MPNet 编码 → PCA 降维，保存为 uid_embed_cache.pkl。

输出：result/03_gaussian/uid_embed_cache.pkl
"""
import pickle, sys, torch, time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np

SENT_CACHE   = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/user_review_sentence_extract/uid_to_sentences.pkl')
EMBED_CACHE  = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/user_review_sentence_extract/uid_embed_cache.pkl')

MAX_SENTS = 100   # 每用户最多句子数
MIN_SENTS = 50    # 筛选用户：至少句子数

print("Loading sentence cache...")
with open(SENT_CACHE, 'rb') as f:
    uid_to_sents = pickle.load(f)
print(f"  Total users: {len(uid_to_sents)}")

# 筛选 + 截断
candidates = [(uid, sents[:MAX_SENTS]) for uid, sents in uid_to_sents.items() if len(sents) >= MIN_SENTS]
candidates.sort(key=lambda x: len(x[1]), reverse=True)
selected = candidates  # 全部 5145 个用户
uid_to_sents_sel = {uid: sents for uid, sents in selected}
uid_list = list(uid_to_sents_sel.keys())
print(f"  Selected users (>= {MIN_SENTS} sents): {len(uid_list)}")
print(f"  Max sents/user: {max(len(s) for s in uid_to_sents_sel.values())}")

# 收集句子
all_texts, uid_slice = [], []
for uid in uid_list:
    sents = uid_to_sents_sel[uid]
    start = len(all_texts)
    all_texts.extend(sents)
    uid_slice.append((uid, start, len(all_texts)))
print(f"  Total sentences: {len(all_texts)}")

print("Loading AnnaWegmann/Style-Embedding...")
wegmann = SentenceTransformer('AnnaWegmann/Style-Embedding', device='cuda')
wegmann.eval()

print("Encoding 768d...")
t0 = time.time()
batch_size = 256
all_768d = []
for i in range(0, len(all_texts), batch_size):
    batch = all_texts[i:i+batch_size]
    with torch.no_grad():
        emb = wegmann.encode(batch, batch_size=len(batch), normalize_embeddings=False)
    all_768d.append(emb.astype(np.float32))
    if i % 10240 == 0 and i > 0:
        print(f"  {i}/{len(all_texts)}")
all_768d = np.concatenate(all_768d, axis=0)
print(f"  Done in {time.time()-t0:.1f}s, shape: {all_768d.shape}")

print("Fitting PCA(128)...")
pca = PCA(n_components=128, random_state=42)
all_128d = pca.fit_transform(all_768d).astype(np.float32)
print(f"  PCA var explained: {pca.explained_variance_ratio_.sum():.4f}")

uid_to_768d, uid_to_128d = {}, {}
for uid, start, end in uid_slice:
    uid_to_768d[uid] = list(all_768d[start:end])
    uid_to_128d[uid] = list(all_128d[start:end])

EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
with open(EMBED_CACHE, 'wb') as f:
    pickle.dump({
        'uid_to_embed_768': uid_to_768d,
        'uid_to_embed_128': uid_to_128d,
        'all_uids': uid_list,
        'pca': pca,
    }, f)
print(f"Saved: {EMBED_CACHE} ({len(uid_list)} users)")
print("Done.")
