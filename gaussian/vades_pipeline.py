"""
VADES Pipeline — 完整四阶段流水线

Stage 1: 训练 VADESSigma（μ_u, σ_u）→ 保存 phase8h_vades.pt
Stage 2: 冻结 μ_u，训练 σ_u 回归目标 → 保存 phase8h_vades_sigma.pt
Stage 3: 训练 StyleMapper（64d→768d）→ 保存 phase8h_style_mapper.pt
Stage 4: TinyStyler 个性化生成 + 三指标验证

依赖：
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/phase8f_style_embed_original.pkl
  - /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage1_filtered_users_reviews_10k_dense_ge50.json
  - result/product_attributes.json
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import hashlib, pickle, json, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════════
SCRATCH      = Path('/home/wlia0047/hj82_scratch2/wenyu')
EMBED_CACHE  = SCRATCH / 'gaussian_vades/phase8f_style_embed_original.pkl'
REVIEW_CACHE = SCRATCH / 'gaussian_vades/stage1_filtered_users_reviews_10k_dense_ge50.json'
ATTR_CACHE   = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/product_attributes.json')
VADES_PT     = SCRATCH / 'gaussian_vades/phase8h_vades.pt'
SIGMA_PT     = SCRATCH / 'gaussian_vades/phase8h_vades_sigma.pt'
MAPPER_PT    = SCRATCH / 'gaussian_vades/phase8h_style_mapper.pt'
QWEN_URL     = 'http://localhost:8800/v1/completions'
QWEN_MODEL   = '/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct'
TS_DIR       = '/fs04/ar57/wenyu/.cache/huggingface/hub/models--tinystyler--tinystyler/snapshots/2a879107b2ec342e57170b82cdc344d5179fa32b'

PCA_DIM    = 128
LATENT_DIM = 64
HIDDEN     = 256
WEGMANN_DIM = 768
N_EPOCHS   = 80
LR         = 1e-3
BATCH_SIZE = 256
L_SAMPLES  = 5
SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS        = 1e-6
N_SAMPLES_GEN = 30
NEUTRAL_SOURCE = "Looking for a lightweight stroller that is easy to fold and suitable for travel."

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SCRATCH / 'gaussian_vades', exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════════
# Stage 0: 加载数据
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Stage 0: 加载数据")
print("="*70)

with open(EMBED_CACHE, 'rb') as f:
    cache = pickle.load(f)
uid_to_embed = cache['uid_to_embed']

with open(REVIEW_CACHE) as f:
    review_data = json.load(f)

MIN_SENTS = 10
valid_uids = [u for u in cache['all_uids'] if u in uid_to_embed and len(uid_to_embed[u]) >= MIN_SENTS]
uid_to_idx = {u: i for i, u in enumerate(valid_uids)}
n_users = len(valid_uids)
print(f"  Users: {n_users}")

# PCA降维
all_embs = np.vstack([e for el in uid_to_embed.values() for e in el])
pca = PCA(n_components=PCA_DIM, random_state=SEED)
pca.fit(all_embs)
uid_to_pca = {uid: pca.transform(np.array(el)).astype(np.float32)
               for uid, el in uid_to_embed.items() if len(el) > 0}

# 加载 sentences 用于 stylometric dispersion
uid_to_sentences = {}
for entry in review_data:
    uid = entry['user_id']
    if uid not in uid_to_idx:
        continue
    sents = []
    for rev in entry.get('reviews', []):
        for s in rev.get('target_reviews', []):
            sents.append(s)
    uid_to_sentences[uid] = sents

# 80/20 split for dispersion supervision
SEED_PREFIX = "vades_pipe|"
train_items, val_items = [], []
for uid in valid_uids:
    idx = uid_to_idx[uid]
    pl  = uid_to_pca[uid]
    for i in range(len(pl)):
        h = int(hashlib.sha1((SEED_PREFIX + uid + f"_{i}").encode()).hexdigest(), 16)
        if (h % 100) < 80:
            train_items.append((pl[i].astype(np.float32), idx))
        else:
            val_items.append((pl[i].astype(np.float32), idx))

trX = torch.from_numpy(np.array([x[0] for x in train_items])).float()
trY = torch.from_numpy(np.array([x[1] for x in train_items])).long()
vaX = torch.from_numpy(np.array([x[0] for x in val_items])).float()
vaY = torch.from_numpy(np.array([x[1] for x in val_items])).long()
print(f"  Train: {len(train_items)}, Val: {len(val_items)}")

# 计算 per-user stylometric dispersion (train/val independent)
FW = {'i','me','my','mine','we','our','ours','you','your','yours',
      'he','him','his','she','her','hers','it','its','they','them','their','theirs',
      'a','an','the','and','or','but','if','then','because','as','of','at','by',
      'for','with','about','against','between','into','through','during','before',
      'after','above','below','to','from','up','down','in','out','on','off','over',
      'under','again','further','then','once','here','there','when','where','why',
      'how','all','each','few','more','most','other','some','such','no','nor','not',
      'only','own','same','so','than','too','very','can','will','just','should','could',
      'do','does','did','have','has','had','this','that','these','those','what','which',
      'who','whom','whose','is','are','was','were','be','been','being'}

def stylometric(text: str) -> np.ndarray:
    words = text.lower().split()
    n = len(words); nc = max(len(text), 1)
    n_sent = max(1, text.count('.')+text.count('!')+text.count('?'))
    feats = [
        sum(len(w) for w in words)/max(n,1)/10.0,
        sum(1 for c in text if c in '.,!?;:-"\'()[]{}')/nc,
        sum(1 for w in words if w in FW)/max(n,1),
        n/max(n_sent,1)/50.0,
        sum(1 for c in text if c.isupper())/nc,
        sum(1 for c in text if c.isdigit())/nc,
        (text.count('!')+text.count('?'))/max(n_sent,1),
        (sum(len(w) for w in words)/max(n,1))/(n/max(n_sent,1)/50.0+0.01),
        len(set(words))/max(n,1),
        sum(1 for w in words if len(w)>6)/max(n,1),
        sum(1 for w in words if len(w)<=3)/max(n,1),
        text.count('?')/max(n_sent,1),
    ]
    return np.clip(feats, -5, 5).astype(np.float32)

uid_disp_train, uid_disp_val = {}, {}
for uid in valid_uids:
    sents = uid_to_sentences.get(uid, [""] * len(uid_to_pca[uid]))
    feats_train, feats_val = [], []
    for i in range(len(uid_to_pca[uid])):
        h = int(hashlib.sha1((SEED_PREFIX + uid + f"_{i}").encode()).hexdigest(), 16)
        text = sents[i] if i < len(sents) else ""
        feat = stylometric(text)
        if (h % 100) < 80:
            feats_train.append(feat)
        else:
            feats_val.append(feat)
    feats_train = np.array(feats_train) if feats_train else np.zeros((1,12),np.float32)
    feats_val   = np.array(feats_val)   if feats_val   else np.zeros((1,12),np.float32)
    uid_disp_train[uid] = float(feats_train.std(axis=0).mean()) if len(feats_train)>1 else 0.0
    uid_disp_val[uid]   = float(feats_val.std(axis=0).mean())   if len(feats_val)>1   else 0.0

disp_val_arr = np.array([uid_disp_val[u] for u in valid_uids])
disp_raw_tensor = torch.zeros(n_users, device=DEVICE, dtype=torch.float32)
for uid, idx in uid_to_idx.items():
    disp_raw_tensor[idx] = uid_disp_train.get(uid, 0.0)

# ════════════════════════════════════════════════════════════════════════════════
# Stage 1: 训练 VADESSigma（μ_u, σ_u）
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Stage 1: 训练 VADESSigma（μ_u, σ_u）")
print("="*70)

class VADESSigma(nn.Module):
    def __init__(self, n_users, in_dim=PCA_DIM, hidden=HIDDEN, latent=LATENT_DIM):
        super().__init__()
        self.doc_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.doc_mu = nn.Linear(hidden, latent)
        self.doc_lv = nn.Linear(hidden, latent)
        self.user_mu = nn.Embedding(n_users, latent)
        self.user_lv = nn.Embedding(n_users, latent)
        nn.init.normal_(self.user_mu.weight, 0.0, 0.05)
        nn.init.constant_(self.user_lv.weight, np.log(np.exp(0.15**2)-1)+EPS)

    def doc_gaussian(self, x):
        h = self.doc_net(x)
        mu = self.doc_mu(h); lv = self.doc_lv(h)
        return mu, F.softplus(lv).clamp(max=6.0) + EPS

    def sample_z(self, mu, lv, n=1):
        bs, ld = mu.size(0), mu.size(1)
        eps = torch.randn(bs, n, ld, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(1) + lv.sqrt().unsqueeze(1) * eps

    def forward(self, doc_emb, author_idx, lambda_disp=1.0):
        mu_x, lv_x = self.doc_gaussian(doc_emb)
        z_x = self.sample_z(mu_x, lv_x, n=L_SAMPLES).mean(dim=1)
        dists = torch.cdist(z_x, self.user_mu.weight, p=2)
        L_id = F.nll_loss(F.log_softmax(-dists, dim=1), author_idx)
        sigma_u = F.softplus(self.user_lv(author_idx)).clamp(max=6.0).sqrt()
        sigma_u_mean = sigma_u.mean(dim=1)
        L_disp = F.mse_loss(sigma_u_mean, disp_raw_tensor[author_idx])
        L_tot = L_id + lambda_disp * L_disp
        return L_tot, {'id': L_id.item(), 'disp': L_disp.item()}

def eval_accuracy(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(vaX), 512):
            batch = vaX[i:i+512].to(DEVICE)
            idxs  = vaY[i:i+512].to(DEVICE)
            mu_x, _ = model.doc_gaussian(batch)
            dists = torch.cdist(mu_x, model.user_mu.weight, p=2)
            pred = dists.argmin(dim=1)
            correct += (pred == idxs).sum().item()
    return 100 * correct / len(vaX)

def calibration_diagnostics(sigma_per_user, disp_arr, tag):
    mae   = np.abs(sigma_per_user - disp_arr).mean()
    bias  = (sigma_per_user - disp_arr).mean()
    sharp = sigma_per_user.std() / (disp_arr.std() + 1e-8)
    corr  = np.corrcoef(sigma_per_user, disp_arr)[0,1]
    print(f"    {tag}: Corr={corr:.4f} MAE={mae:.4f} Bias={bias:+.4f} Sharp={sharp:.4f}")
    return {'corr': corr, 'mae': mae, 'bias': bias, 'sharpness': sharp}

# 训练
model = VADESSigma(n_users).to(DEVICE)
VADES_PT = SCRATCH / 'gaussian_vades/phase8h_vades.pt'
if VADES_PT.exists():
    print(f"  [CACHE] Loading Stage 1 from {VADES_PT}")
    ckpt = torch.load(VADES_PT, map_location=DEVICE)
    model.user_mu.weight.data = ckpt['user_mu'].to(DEVICE)
    model.user_lv.weight.data = ckpt['user_lv'].to(DEVICE)
    model.doc_net.load_state_dict(ckpt['doc_net'])
    model.doc_mu.load_state_dict(ckpt['doc_mu'])
    model.doc_lv.load_state_dict(ckpt['doc_lv'])
    print(f"  [CACHE] Stage 1 loaded (calib_corr={ckpt.get('calib_corr','?')})")
else:
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=N_EPOCHS)
    for epoch in range(N_EPOCHS):
        model.train()
        rng = np.random.RandomState(SEED + epoch)
        order = list(range(len(train_items)))
        rng.shuffle(order)
        ep = {k: 0.0 for k in ['tot','id','disp']}; nb = 0
        for i in range(0, len(order), BATCH_SIZE):
            bi = order[i:i+BATCH_SIZE]
            loss, stats = model(trX[bi].to(DEVICE), trY[bi].to(DEVICE), lambda_disp=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ep['tot'] += loss.item(); ep['id'] += stats['id']; ep['disp'] += stats['disp']; nb += 1
        sched.step()
        if epoch % 20 == 0 or epoch == N_EPOCHS-1:
            av = {k: v/max(nb,1) for k,v in ep.items()}
            print(f"  ep {epoch:3d}: tot={av['tot']:.4f} id={av['id']:.4f} disp={av['disp']:.6f}")

# 保存 Stage 1 checkpoint
user_lv_np = model.user_lv.weight.detach().cpu().numpy()
user_sigma = np.sqrt(np.maximum(np.exp(user_lv_np), 1e-6))
sigma_mean_per_user = user_sigma.mean(axis=1)
calib = calibration_diagnostics(sigma_mean_per_user, disp_val_arr, "val")
print(f"  val: Corr={calib['corr']:.4f} MAE={calib['mae']:.4f} Bias={calib['bias']:+.4f} Sharp={calib['sharpness']:.4f}")

torch.save({
    'user_mu': model.user_mu.weight.detach().cpu(),
    'user_lv': model.user_lv.weight.detach().cpu(),
    'doc_net': model.doc_net.state_dict(),
    'doc_mu': model.doc_mu.state_dict(),
    'doc_lv': model.doc_lv.state_dict(),
    'lambda_disp': 1.0,
    'calib_corr': float(calib['corr']),
}, VADES_PT)
print(f"  → {VADES_PT}")

# ════════════════════════════════════════════════════════════════════════════════
# Stage 2: 冻结 μ_u，训练 σ_u 回归（基于 embedding 统计量）
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Stage 2: 冻结 μ_u，训练 σ_u 回归")
print("="*70)

# 计算每个用户的真实 embedding 方差（两个分支都用到，先提前计算）
uid_to_target_var = {}
for uid in valid_uids:
    pl = uid_to_pca[uid]
    std_per_dim = pl.std(axis=0)
    uid_to_target_var[uid] = std_per_dim
all_vars = np.stack([uid_to_target_var[u].mean() for u in valid_uids])
print(f"  Target variance: mean={all_vars.mean():.4f}, std={all_vars.std():.4f}")

SIGMA_PT = SCRATCH / 'gaussian_vades/phase8h_vades_sigma.pt'
if SIGMA_PT.exists():
    print(f"  [CACHE] Loading Stage 2 from {SIGMA_PT}")
    ckpt2 = torch.load(SIGMA_PT, map_location='cpu', weights_only=True)
    model2 = VADESSigma(n_users).to(DEVICE)
    with torch.no_grad():
        model2.user_mu.weight.copy_(ckpt2['user_mu'].to(DEVICE))
        model2.user_lv.weight.copy_(ckpt2['user_lv'].to(DEVICE))
        model2.doc_net.load_state_dict(ckpt2['doc_net'])
        model2.doc_mu.load_state_dict(ckpt2['doc_mu'])
        model2.doc_lv.load_state_dict(ckpt2['doc_lv'])
    corr_sigma = float(np.corrcoef(
        np.array([uid_to_target_var[uid].mean() for uid in valid_uids]),
        model2.user_lv.weight.detach().cpu().numpy().mean(axis=1))[0,1])
    print(f"  [CACHE] Stage 2 loaded, corr_sigma={corr_sigma:.4f}")
else:
    # 重新初始化模型，冻结 user_mu
    model2 = VADESSigma(n_users).to(DEVICE)
    ckpt = torch.load(VADES_PT, map_location='cpu', weights_only=True)
    with torch.no_grad():
        model2.user_mu.weight.copy_(ckpt['user_mu'])
        model2.user_mu.weight.requires_grad = False
    model2.train()

    opt2 = AdamW([
        {"params": model2.user_lv.parameters()},
        {"params": model2.doc_net.parameters()},
        {"params": model2.doc_mu.parameters()},
        {"params": model2.doc_lv.parameters()},
    ], lr=5e-4)

    for epoch in range(N_EPOCHS):
        rng = np.random.default_rng(SEED + epoch)
        order = list(range(len(train_items)))
        rng.shuffle(order)
        ep_nce = 0.0; ep_var = 0.0; nb = 0
        for i in range(0, len(order), BATCH_SIZE):
            bi = order[i:i+BATCH_SIZE]
            doc_emb  = trX[bi].to(DEVICE)
            user_idx = trY[bi].to(DEVICE)
            mu_x, lv_x = model2.doc_gaussian(doc_emb)
            z_x = model2.sample_z(mu_x, lv_x, n=L_SAMPLES).mean(dim=1)
            dists = torch.cdist(z_x, model2.user_mu.weight, p=2)
            nce_loss = F.nll_loss(F.log_softmax(-dists, dim=1), user_idx)
            target_var_t = torch.tensor(
                np.array([uid_to_target_var[valid_uids[u]].mean() for u in user_idx.cpu().numpy()]),
                dtype=torch.float32).to(DEVICE)
            pred_var = lv_x.mean(dim=1)
            var_loss = F.mse_loss(pred_var, target_var_t)
            loss = nce_loss + 0.5 * var_loss
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0); opt2.step()
            ep_nce += nce_loss.item(); ep_var += var_loss.item(); nb += 1
        if epoch % 20 == 0 or epoch == N_EPOCHS-1:
            print(f"  ep {epoch:3d}: nce={ep_nce/max(nb,1):.4f} var_loss={ep_var/max(nb,1):.4f}")

    # 分析
    with torch.no_grad():
        user_lv2 = F.softplus(model2.user_lv.weight).cpu().numpy()
    user_mean_var = user_lv2.mean(axis=1)
    target_vars = np.array([uid_to_target_var[uid].mean() for uid in valid_uids])
    corr_sigma = np.corrcoef(target_vars, user_mean_var)[0,1]
    print(f"  Target vs σ_u corr: {corr_sigma:.4f}")

    torch.save({
        'user_mu': model2.user_mu.weight.detach().cpu(),
        'user_lv': model2.user_lv.weight.detach().cpu(),
        'doc_net': model2.doc_net.state_dict(),
        'doc_mu': model2.doc_mu.state_dict(),
        'doc_lv': model2.doc_lv.state_dict(),
    }, SIGMA_PT)
    print(f"  → {SIGMA_PT}")

# ════════════════════════════════════════════════════════════════════════════════
# Stage 3: 训练 StyleMapper（64d→768d）
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Stage 3: 训练 StyleMapper（64d→768d）")
print("="*70)

# StyleMapper 类定义（Stage 3 和 Stage 4 共用）
class StyleMapper(nn.Module):
    def __init__(self, in_dim=LATENT_DIM, hidden=HIDDEN, out_dim=WEGMANN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)
    def get_style_vector(self, x):
        return F.normalize(self.forward(x), dim=-1)

if MAPPER_PT.exists():
    print(f"  [CACHE] Loading Stage 3 from {MAPPER_PT}")
    mp3 = torch.load(MAPPER_PT, map_location='cpu')
    mapper = StyleMapper().to(DEVICE)
    mapper.load_state_dict(mp3['state_dict'])
    mapper.eval()
    best_val_cos = float(mp3['val_cos'])
    print(f"  [CACHE] StyleMapper loaded, val_cos={best_val_cos:.4f}")
    # Stage 4 需要 user_mu_np / user_lv_np，从 VADES checkpoint 加载
    ckpt_vades = torch.load(VADES_PT, map_location='cpu', weights_only=True)
    user_mu_np = ckpt_vades['user_mu'].numpy()
    user_lv_np = ckpt_vades['user_lv'].numpy()
else:
    ckpt_vades = torch.load(VADES_PT, map_location='cpu', weights_only=True)
    user_mu_np = ckpt_vades['user_mu'].numpy()
    user_lv_np = ckpt_vades['user_lv'].numpy()
    N_SAMPLES_MAPPER = 5

    uid_to_wegmann_centroid = {}
    for uid in valid_uids:
        embeds = uid_to_embed.get(uid, [])
        if len(embeds) < 5:
            continue
        uid_to_wegmann_centroid[uid] = np.stack(embeds).mean(axis=0)

    X_list, Y_list = [], []
    for uid in valid_uids:
        idx = uid_to_idx[uid]
        mu = user_mu_np[idx]
        lv = np.exp(user_lv_np[idx])
        centroid = uid_to_wegmann_centroid.get(uid)
        if centroid is None:
            continue
        for _ in range(N_SAMPLES_MAPPER):
            z_sampled = mu + np.sqrt(lv) * np.random.randn(LATENT_DIM)
            X_list.append(z_sampled)
            Y_list.append(centroid)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    X_tr, X_va, Y_tr, Y_va = train_test_split(X, Y, test_size=0.2, random_state=SEED)
    print(f"  Train: {X_tr.shape[0]}, Val: {X_va.shape[0]}")

    mapper = StyleMapper().to(DEVICE)
    opt_m = torch.optim.AdamW(mapper.parameters(), lr=1e-3)

    best_val_cos = -1; best_state = None
    for epoch in range(200):
        mapper.train()
        rng = np.random.RandomState(SEED + epoch)
        order = list(range(len(X_tr)))
        rng.shuffle(order)
        for i in range(0, len(order), 16):
            bi = order[i:i+16]
            y_pred = mapper.get_style_vector(torch.from_numpy(X_tr[bi]).to(DEVICE))
            y_true = F.normalize(torch.from_numpy(Y_tr[bi]).to(DEVICE), dim=-1)
            cos = (y_pred * y_true).sum(dim=-1).mean()
            opt_m.zero_grad(); (-cos).backward(); opt_m.step()
        mapper.eval()
        with torch.no_grad():
            val_pred = mapper.get_style_vector(torch.from_numpy(X_va).to(DEVICE))
            val_true = F.normalize(torch.from_numpy(Y_va).to(DEVICE), dim=-1)
            val_cos = (val_pred * val_true).sum(dim=-1).mean().item()
        if val_cos > best_val_cos:
            best_val_cos = val_cos
            best_state = {k: v.cpu().clone() for k, v in mapper.state_dict().items()}
        if epoch % 40 == 0 or epoch == 199:
            print(f"  ep {epoch:3d}: val_cos={val_cos:.4f} best={best_val_cos:.4f}")

    mapper.load_state_dict(best_state)
    mapper.eval()
    print(f"  Best val_cos: {best_val_cos:.4f}")

    torch.save({
        'state_dict': best_state,
        'config': {'latent_dim': LATENT_DIM, 'hidden': HIDDEN, 'wegmann_dim': WEGMANN_DIM},
        'val_cos': best_val_cos,
    }, MAPPER_PT)
    print(f"  → {MAPPER_PT}")

# ════════════════════════════════════════════════════════════════════════════════
# Stage 4: TinyStyler 生成 + 三指标验证（纯 VADES + StyleMapper 链）
#
# 编码链（无任何 T5+PCA）:
#   text → TinyStyler T5 encode → 768d → VADES doc_net → 64d → StyleMapper → 768d Wegmann
#
# 三指标均在 Wegmann 768d 空间计算（与 TinyStyler injection 同一空间）:
#   (a) Style adherence:  cos(z_injected, z_reencoded) in 768d
#   (b) σ→Dispersion:    Corr(σ_u, Dispersion_gen) in 768d
#   (c) User attribution: nearest μ_u^768d in 768d
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Stage 4: TinyStyler 生成 + 三指标验证（VADES+StyleMapper 链）")
print("="*70)

# 加载 TinyStyler
sys.path.insert(0, TS_DIR)
from tinystyler import get_tinystyler_model
tinystyler_tok, tinystyler_model = get_tinystyler_model(DEVICE)
tinystyler_model.eval()
ts_t5 = tinystyler_model.model.encoder   # TinyStyler T5 encoder
print("  TinyStyler loaded")

# 加载 VADES doc_net（用于 768d→64d 投影）
# 从 VADES checkpoint 取出 doc_net 结构并加载权重
class VADESEncoder(nn.Module):
    """128d embed → 64d latent (VADES doc_encoder)"""
    def __init__(self, hidden=HIDDEN, latent=LATENT_DIM, in_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.mu_proj = nn.Linear(hidden, latent)
        self.lv_proj = nn.Linear(hidden, latent)
    def forward(self, x):
        h = self.net(x)
        return self.mu_proj(h), self.lv_proj(h)

# T5-Large outputs 1024d; VADES doc_net expects 128d — add projection
t5_to_doc = nn.Linear(1024, 128).to(DEVICE)
t5_to_doc.eval()
for p in t5_to_doc.parameters():
    p.requires_grad = False

vades_encoder = VADESEncoder().to(DEVICE)
doc_net = ckpt['doc_net']
vades_encoder.load_state_dict({
    'net.0.weight': doc_net['0.weight'],
    'net.0.bias': doc_net['0.bias'],
    'net.1.weight': doc_net['1.weight'],
    'net.1.bias': doc_net['1.bias'],
    'net.4.weight': doc_net['4.weight'],
    'net.4.bias': doc_net['4.bias'],
    'net.5.weight': doc_net['5.weight'],
    'net.5.bias': doc_net['5.bias'],
    'mu_proj.weight': ckpt['doc_mu']['weight'],
    'mu_proj.bias': ckpt['doc_mu']['bias'],
    'lv_proj.weight': ckpt['doc_lv']['weight'],
    'lv_proj.bias': ckpt['doc_lv']['bias'],
}, strict=True)
vades_encoder.eval()
print("  VADES encoder (768d→64d) loaded")

# ── Stage 4a: 批量编码所有用户句子 → 768d → 64d → 768d Wegmann
print("\n  Stage 4a: 批量编码用户句子（纯 VADES+StyleMapper 链）...")

def encode_sentences_to_wegmann(texts, batch_size=512):
    """text → TinyStyler T5 → 768d → VADES doc_net → 64d → StyleMapper → 768d"""
    all_768d = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        inputs = tinystyler_tok(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            enc = ts_t5(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            emb_768 = enc.last_hidden_state.mean(dim=1)  # (batch, 768)
            emb_128 = t5_to_doc(emb_768)  # (batch, 128) → VADES doc_net
        with torch.no_grad():
            mu_64, _ = vades_encoder(emb_128)  # (batch, 64)
            mu_64_norm = F.normalize(mu_64, dim=-1)
            wegmann_768 = mapper.get_style_vector(mu_64_norm)  # (batch, 768)
        all_768d.append(wegmann_768.cpu().numpy())
    return np.concatenate(all_768d, axis=0).astype(np.float32)  # (N, 768)

# 收集所有用户的句子
uid_set_review = {e['user_id'] for e in review_data}
common_uids = [u for u in valid_uids if u in uid_set_review][:999]
uid_gen = [u for u in common_uids if u in uid_to_sentences]
print(f"  Encoding {len(uid_gen)} users' sentences...")

uid_to_sentence_list = {}
all_sent_texts = []
all_sent_uidx = []
for gi, uid in enumerate(uid_gen):
    sents = uid_to_sentences[uid][:50]  # 最多50句
    uid_to_sentence_list[uid] = sents
    for s in sents[:20]:  # 最多20句，加速
        all_sent_texts.append(s)
        all_sent_uidx.append(gi)

print(f"  Total sentences to encode: {len(all_sent_texts)}")
t_enc = time.time()
all_sent_wegmann = encode_sentences_to_wegmann(all_sent_texts, batch_size=256)
print(f"  Encoded in {time.time()-t_enc:.1f}s, shape: {all_sent_wegmann.shape}")

# ── 同时编码真实句子到 VADES 64d 空间（用于在 VADES 原空间测 dispersion）
def encode_sentences_to_vades_64(texts, batch_size=512):
    """text → TinyStyler T5 → 768d → VADES doc_net → 64d"""
    all_64d = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        inputs = tinystyler_tok(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            enc = ts_t5(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            emb_768 = enc.last_hidden_state.mean(dim=1)  # (batch, 768)
            emb_128 = t5_to_doc(emb_768)  # 1024→128
        with torch.no_grad():
            mu_64, _ = vades_encoder(emb_128)  # (batch, 64)
            all_64d.append(mu_64.detach().cpu().numpy())
    return np.concatenate(all_64d, axis=0).astype(np.float32)  # (N, 64)

t_enc64 = time.time()
all_sent_vades_64 = encode_sentences_to_vades_64(all_sent_texts, batch_size=256)
print(f"  VADES 64d encoding: {len(all_sent_texts)} sents in {time.time()-t_enc64:.1f}s, shape: {all_sent_vades_64.shape}")

# 用户在 VADES 64d 空间的 σ_VADES（来自模型预测）
# 以及真实文本在 VADES 64d 空间的 dispersion
sigma_vades = np.zeros(len(uid_gen), dtype=np.float32)
uid_vades_disp = {}  # gi -> {eucl, ang}
for gi, uid in enumerate(uid_gen):
    idx = uid_to_idx[uid]
    lv_t = np.exp(user_lv_np[idx].astype(np.float32))
    sigma_vades[gi] = float(np.sqrt(np.maximum(lv_t.mean(), 1e-6)))
    mask = [i for i, u in enumerate(all_sent_uidx) if u == gi]
    if len(mask) >= 3:
        emb_v = all_sent_vades_64[mask]
        cen_v = emb_v.mean(axis=0)
        d_e = float(np.linalg.norm(emb_v - cen_v, axis=1).mean())
        # VADES 64d 角距离
        emb_vn = emb_v / (np.linalg.norm(emb_v, axis=1, keepdims=True) + 1e-8)
        cen_vn = cen_v / (np.linalg.norm(cen_v) + 1e-8)
        cos_sim = np.clip(np.dot(emb_vn, cen_vn), -1, 1)
        d_a = float(np.arccos(cos_sim).mean())
        uid_vades_disp[gi] = {'eucl': d_e, 'ang': d_a}
    else:
        uid_vades_disp[gi] = {'eucl': np.nan, 'ang': np.nan}
print(f"  σ_VADES range: {sigma_vades.min():.4f}–{sigma_vades.max():.4f}")

# 计算每个用户在 768d Wegmann 空间的真实 centroid
uid_to_wegmann_centroid_768 = {}
for gi, uid in enumerate(uid_gen):
    mask = [i for i, u in enumerate(all_sent_uidx) if u == gi]
    if len(mask) >= 3:
        uid_to_wegmann_centroid_768[uid] = all_sent_wegmann[mask].mean(axis=0)

# 用户在 768d Wegmann 空间的 μ_u 和 σ_u
mu_w = np.zeros((len(uid_gen), WEGMANN_DIM), dtype=np.float32)
sigma_w = np.zeros(len(uid_gen), dtype=np.float32)
for gi, uid in enumerate(uid_gen):
    mask = [i for i, u in enumerate(all_sent_uidx) if u == gi]
    if len(mask) >= 3:
        embs = all_sent_wegmann[mask]
        mu_w[gi] = embs.mean(axis=0)
        sigma_w[gi] = np.linalg.norm(embs - mu_w[gi], axis=1).mean()
print(f"  User centroids: {len(uid_to_wegmann_centroid_768)} users computed")

# ── Stage 4b: TinyStyler 批量生成（全量收集后一次性 batch generate）
# 策略：不通过 injection_strength 控制 dispersion（已被证伪，TinyStyler 归一化抹掉了幅度差异）
# 改为：采样 z ~ N(μ_u, σ_u²)，用多个随机种子重复生成，测量生成文本 dispersion
print(f"\n  Stage 4b: TinyStyler 生成（{len(uid_gen)} users × {N_SAMPLES_GEN} samples per user）...")

torch.backends.cudnn.benchmark = True

all_z_64 = []
all_uid_idx = []
rng_global = np.random.default_rng(SEED)

for gi, uid in enumerate(uid_gen):
    idx = uid_to_idx[uid]
    mu_t = user_mu_np[idx].astype(np.float32)
    lv_t = np.exp(user_lv_np[idx].astype(np.float32))
    sigma_t = float(np.sqrt(np.maximum(lv_t.mean(), 1e-6)))
    rng_s = np.random.default_rng(SEED + gi)
    for k in range(N_SAMPLES_GEN):
        eps = rng_s.normal(0, 1, LATENT_DIM).astype(np.float32)
        z = mu_t + eps * sigma_t  # 直接用 σ_u 作为 scale（不用 ALPHA_SCALE 放大）
        z = z / (np.linalg.norm(z) + 1e-8)
        all_z_64.append(z)
        all_uid_idx.append(gi)

all_z_64 = np.array(all_z_64, dtype=np.float32)
print(f"  Total z vectors: {len(all_z_64)}")

# Batched generation: 分批，每批 64 个 style 向量一次性 generate
BATCH_G = 64
base_inputs = tinystyler_tok(NEUTRAL_SOURCE, return_tensors="pt", padding=True,
                               truncation=True, max_length=128)
base_ids = base_inputs['input_ids'].repeat(BATCH_G, 1).to(DEVICE)  # (BATCH_G, seq)
base_att = base_inputs['attention_mask'].repeat(BATCH_G, 1).to(DEVICE)  # (BATCH_G, seq)

all_gen_texts = [None] * len(all_z_64)
t_gen = time.time()

for start in range(0, len(all_z_64), BATCH_G):
    end = min(start + BATCH_G, len(all_z_64))
    batch_n = end - start
    z_batch = torch.from_numpy(all_z_64[start:end]).float().to(DEVICE)  # (B, 64)
    # z → StyleMapper → 768d
    with torch.no_grad():
        style_768 = mapper.get_style_vector(z_batch)  # (B, 768)
    # 调整 base inputs 到实际 batch size
    ids_batch = base_ids[:batch_n]
    att_batch = base_att[:batch_n]
    with torch.no_grad():
        out_ids = tinystyler_model.generate(
            input_ids=ids_batch,
            attention_mask=att_batch,
            style=style_768,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            max_new_tokens=64,
        )
    texts = tinystyler_tok.batch_decode(out_ids, skip_special_tokens=True)
    for j, text in enumerate(texts):
        all_gen_texts[start + j] = text.strip()
    if (start + BATCH_G) % 500 == 0 or end == len(all_z_64):
        print(f"  Generated {end}/{len(all_z_64)} texts ({end/len(all_z_64)*100:.0f}%)")

print(f"  Generation done in {time.time()-t_gen:.1f}s")
print(f"  Sample:")
for i in range(3):
    print(f"    [{uid_gen[all_uid_idx[i]][:8]}] {all_gen_texts[i][:80]}")

# ── Stage 4c: 重新编码生成文本（纯 VADES+StyleMapper 链）
print("\n  Stage 4c: 重新编码生成文本（VADES+StyleMapper 链）...")

valid_texts = [(i, t) for i, t in enumerate(all_gen_texts) if t and len(t.strip()) > 5]
valid_idx = [i for i, t in valid_texts]
valid_gen = [t for i, t in valid_texts]
print(f"  Valid texts: {len(valid_gen)}/{len(all_gen_texts)}")

t_enc2 = time.time()
gen_emb_768 = encode_sentences_to_wegmann(valid_gen, batch_size=256)
print(f"  Re-encoded in {time.time()-t_enc2:.1f}s")

gen_emb_full = np.zeros((len(all_gen_texts), WEGMANN_DIM), dtype=np.float32)
for j, orig_idx in enumerate(valid_idx):
    gen_emb_full[orig_idx] = gen_emb_768[j]

# (a) Style adherence: cos(z_injected, z_reencoded)
#     检验生成质量：TinyStyler 是否忠实将 z 注入到文本
cos_inject = []
for orig_idx in valid_idx:
    gi = all_uid_idx[orig_idx]
    uid = uid_gen[gi]
    idx = uid_to_idx[uid]
    mu_t = torch.from_numpy(user_mu_np[idx]).float().to(DEVICE)
    mu_norm = F.normalize(mu_t.unsqueeze(0), dim=-1)
    z_inj = mapper.get_style_vector(mu_norm.squeeze(0)).detach().cpu().numpy()
    z_hat = gen_emb_full[orig_idx]
    cos_inject.append(np.dot(z_inj, z_hat))
cos_inject = np.array(cos_inject)
print(f"\n  (a) Style adherence: mean={cos_inject.mean():.4f} median={np.median(cos_inject):.4f}")
print(f"      → TinyStyler 是否忠实注入了风格向量？mean≥0.98 为佳")

# ── Stage 4d: 在多个空间测 dispersion
# 关键：σ_u 来自 VADES 64d 空间，测量 dispersion 的空间应与 σ_u 对齐
print("\n  Stage 4d: 多空间 dispersion 测量...")

# 1. T5 768d 空间（直接编码生成文本，不过 VADES/StyleMapper）
def encode_t5_768(texts, batch_size=256):
    all_768 = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        inputs = tinystyler_tok(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            enc = ts_t5(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            emb_1024 = enc.last_hidden_state.mean(dim=1)  # (batch, 1024) T5-Large
            emb_768 = t5_proj_1024_to_768(emb_1024)  # (batch, 768)
        all_768.append(emb_768.cpu().numpy())
    return np.concatenate(all_768, axis=0).astype(np.float32)

# T5-Large 1024d → 768d 投影（用于 T5 直接编码空间）
t5_proj_1024_to_768 = nn.Linear(1024, 768).to(DEVICE)
t5_proj_1024_to_768.eval()
for p in t5_proj_1024_to_768.parameters():
    p.requires_grad = False

gen_emb_t5 = encode_t5_768(valid_gen, batch_size=256)
gen_emb_t5_full = np.zeros((len(all_gen_texts), 768), dtype=np.float32)
for j, orig_idx in enumerate(valid_idx):
    gen_emb_t5_full[orig_idx] = gen_emb_t5[j]

# 2. VADES 64d 空间（生成文本的 64d 表征）
gen_emb_64_full = np.zeros((len(all_gen_texts), LATENT_DIM), dtype=np.float32)
gen_emb_64 = []
for start in range(0, len(valid_gen), 256):
    end = min(start + 256, len(valid_gen))
    batch = valid_gen[start:end]
    inputs = tinystyler_tok(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        enc = ts_t5(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        emb_768 = enc.last_hidden_state.mean(dim=1)  # (B, 768)
        emb_128 = t5_to_doc(emb_768)
        mu_64, _ = vades_encoder(emb_128)
        gen_emb_64.append(mu_64.detach().cpu().numpy())
gen_emb_64 = np.concatenate(gen_emb_64, axis=0).astype(np.float32)
for j, orig_idx in enumerate(valid_idx):
    gen_emb_64_full[orig_idx] = gen_emb_64[j]

# 3. 对每个用户计算 dispersion（欧氏 + 角距离）
# 角距离更适合单位球面向量：arccos(clamp(cos_sim, -1, 1))
def angular_disp(embs):
    """单位向量集的平均角距离（弧度）"""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / (norms + 1e-8)
    centroid = embs_n.mean(axis=0)
    cn = np.linalg.norm(centroid) + 1e-8
    centroid /= cn
    cos_sims = np.clip(np.sum(embs_n * centroid, axis=1), -1, 1)
    angles = np.arccos(cos_sims)
    return float(np.nanmean(angles))

# ════════════════════════════════════════════════════════════════════════════════
# Stage 4d: 核心验证 — σ_Wegmann 能否预测生成文本的 dispersion
#
# 实验设计（对齐之前成功的 999-user 实验）：
#   - σ_Wegmann = 从真实文本 embedding 算出的用户内 dispersion
#   - dispersion_real = 真实文本在 Wegmann 空间的 dispersion
#   - dispersion_gen = TinyStyler 生成文本在 Wegmann 空间的 dispersion
#
# 成功基准：σ_Wegmann → dispersion_real ρ≈1.0（定义性关系）
# 成功基准：σ_Wegmann → dispersion_real ρ≈0.77（999-user 历史最佳）
# 当前测试：σ_Wegmann → dispersion_gen ρ>?（我们想验证的）
# ════════════════════════════════════════════════════════════════════════════════
print("\n  Stage 4d: σ_Wegmann → dispersion 核心验证...")

# Step 1: 计算每个用户真实文本的 Wegmann 768d dispersion
# 真实文本的 embedding 已经保存在 all_sent_wegmann 里
uid_real_disp = {}  # gi -> {eucl, ang}
for gi in range(len(uid_gen)):
    mask = [i for i, u in enumerate(all_sent_uidx) if u == gi]
    if len(mask) < 3:
        uid_real_disp[gi] = {'eucl': np.nan, 'ang': np.nan}
        continue
    emb_real = all_sent_wegmann[mask]
    cen_real = emb_real.mean(axis=0)
    d_eucl = float(np.linalg.norm(emb_real - cen_real, axis=1).mean())
    d_ang = angular_disp(emb_real)
    uid_real_disp[gi] = {'eucl': d_eucl, 'ang': d_ang}

# Step 2: 计算每个用户 TinyStyler 生成文本的 Wegmann 768d dispersion
uid_gen_disp = {}  # gi -> {eucl, ang}
for gi in range(len(uid_gen)):
    mask = [i for i in valid_idx if all_uid_idx[i] == gi]
    if len(mask) < 2:
        uid_gen_disp[gi] = {'eucl': np.nan, 'ang': np.nan}
        continue
    emb_gen = gen_emb_full[mask]
    cen_gen = emb_gen.mean(axis=0)
    d_eucl = float(np.linalg.norm(emb_gen - cen_gen, axis=1).mean())
    d_ang = angular_disp(emb_gen)
    uid_gen_disp[gi] = {'eucl': d_eucl, 'ang': d_ang}

# Step 2b: VADES 64d space dispersion for generated texts
uid_gen_disp_vades = {}
for gi in range(len(uid_gen)):
    mask = [i for i in valid_idx if all_uid_idx[i] == gi]
    if len(mask) < 2:
        uid_gen_disp_vades[gi] = {'eucl': np.nan, 'ang': np.nan}
        continue
    emb_v = gen_emb_64_full[mask]
    cen_v = emb_v.mean(axis=0)
    d_e = float(np.linalg.norm(emb_v - cen_v, axis=1).mean())
    emb_vn = emb_v / (np.linalg.norm(emb_v, axis=1, keepdims=True) + 1e-8)
    cen_vn = cen_v / (np.linalg.norm(cen_v) + 1e-8)
    cos_sim = np.clip(np.sum(emb_vn * cen_vn), -1, 1)
    d_a = float(np.arccos(cos_sim).mean())
    uid_gen_disp_vades[gi] = {'eucl': d_e, 'ang': d_a}

# Step 3: 计算 σ_Wegmann → dispersion_real（定义性 sanity check，期望 ≈1.0）
print("\n  [Sanity Check] σ_Wegmann 定义性关系：")
for metric in ['eucl', 'ang']:
    sigma_vals = []
    disp_vals = []
    for gi in range(len(uid_gen)):
        if not np.isnan(uid_real_disp[gi][metric]):
            sigma_vals.append(sigma_w[gi])
            disp_vals.append(uid_real_disp[gi][metric])
    if len(sigma_vals) >= 5:
        rho, p = spearmanr(sigma_vals, disp_vals)
        label = '欧氏' if metric == 'eucl' else '角距'
        print(f"    σ_Wegmann → dispersion_real ({label}): n={len(sigma_vals)} ρ={rho:.4f} p={p:.2e}")

# Step 4: 计算 σ_Wegmann → dispersion_gen（核心测试）
print("\n  [Core Test] σ_Wegmann → TinyStyler 生成文本 dispersion：")
summary_rhos = {}
for metric in ['eucl', 'ang']:
    sigma_vals = []
    disp_vals = []
    for gi in range(len(uid_gen)):
        if not np.isnan(uid_gen_disp[gi][metric]) and not np.isnan(uid_real_disp[gi][metric]):
            sigma_vals.append(sigma_w[gi])
            disp_vals.append(uid_gen_disp[gi][metric])
    if len(sigma_vals) >= 5:
        rho, p = spearmanr(sigma_vals, disp_vals)
        label = '欧氏' if metric == 'eucl' else '角距'
        key = f'wegmann_gen_{metric}'
        sig = "✓" if abs(rho) > 0.3 and p < 0.05 else "✗"
        print(f"    σ_Wegmann → dispersion_gen_Wegmann ({label}): n={len(sigma_vals)} ρ={rho:.4f} p={p:.2e} {sig}")
        summary_rhos[key] = (rho, p)

# Step 4b: 在 VADES 64d 空间测量（σ 是 VADES 空间学的，测量也应该在 VADES 空间）
print("\n  [Core Test VADES] σ_VADES → VADES 64d dispersion：")
for space_key, sigma_arr, uid_disp_dict, space_name in [
    ('σ_VADES', sigma_vades, uid_gen_disp_vades, 'VADES 64d'),
    ('σ_VADES', sigma_vades, uid_gen_disp, 'Wegmann 768d'),
]:
    for metric in ['eucl', 'ang']:
        sigma_vals = []
        disp_vals = []
        for gi in range(len(uid_gen)):
            if not np.isnan(uid_disp_dict[gi][metric]):
                sigma_vals.append(sigma_arr[gi])
                disp_vals.append(uid_disp_dict[gi][metric])
        if len(sigma_vals) >= 5:
            rho, p = spearmanr(sigma_vals, disp_vals)
            label = '欧氏' if metric == 'eucl' else '角距'
            key = f'vades_gen_{space_name.replace(" ", "_").lower()}_{metric}'
            sig = "✓" if abs(rho) > 0.3 and p < 0.05 else "✗"
            print(f"    {space_key} → dispersion_gen_{space_name} ({label}): n={len(sigma_vals)} ρ={rho:.4f} p={p:.2e} {sig}")
            summary_rhos[key] = (rho, p)

# Step 5: 直接比较同一用户的真实 dispersion vs 生成 dispersion（对齐度检验）
print("\n  [Alignment Check] 真实文本 dispersion vs TinyStyler 生成文本 dispersion：")
for metric in ['eucl', 'ang']:
    real_vals = []
    gen_vals = []
    for gi in range(len(uid_gen)):
        if not np.isnan(uid_real_disp[gi][metric]) and not np.isnan(uid_gen_disp[gi][metric]):
            real_vals.append(uid_real_disp[gi][metric])
            gen_vals.append(uid_gen_disp[gi][metric])
    if len(real_vals) >= 5:
        rho, p = spearmanr(real_vals, gen_vals)
        label = '欧氏' if metric == 'eucl' else '角距'
        key = f'real_vs_gen_{metric}'
        print(f"    dispersion_real → dispersion_gen ({label}): n={len(real_vals)} ρ={rho:.4f} p={p:.2e}")
        summary_rhos[key] = (rho, p)
        print(f"      (真实文本 dispersion 范围: mean={np.mean(real_vals):.4f}, 生成文本: mean={np.mean(gen_vals):.4f})")

# 同时打印 σ_Wegmann 在 T5 空间的 dispersion 预测能力作为对比
print("\n  [Auxiliary] σ_Wegmann → dispersion_gen_T5 欧氏：")
sigma_t5_vals = []
disp_t5_vals = []
for gi in range(len(uid_gen)):
    mask = [i for i in valid_idx if all_uid_idx[i] == gi]
    if len(mask) < 2:
        continue
    emb_t5 = gen_emb_t5_full[mask]
    cen_t5 = emb_t5.mean(axis=0)
    d_t5 = float(np.linalg.norm(emb_t5 - cen_t5, axis=1).mean())
    sigma_t5_vals.append(sigma_w[gi])
    disp_t5_vals.append(d_t5)
if len(sigma_t5_vals) >= 5:
    rho, p = spearmanr(sigma_t5_vals, disp_t5_vals)
    print(f"    σ_Wegmann → dispersion_gen_T5 (欧氏): n={len(sigma_t5_vals)} ρ={rho:.4f} p={p:.2e}")

# ════════════════════════════════════════════════════════════════════════════════
# Stage 4e: 补充分析 — 高 σ 用户生成的文本，centroid 是否离真实 centroid 更远？
#
# 新假设：TinyStyler 归一化后 dispersion 被压缩，但 σ 大的用户生成的文本
#         平均风格向量（gen_centroid）可能离真实 centroid 更远
#         因为 σ 大 → 采样方向更分散 → 归一化后方向差异累积 → centroid 偏移
# ════════════════════════════════════════════════════════════════════════════════
print("\n  Stage 4e: σ_Wegmann → centroid_offset 补充分析...")

# 计算每个用户生成文本的 centroid 与真实文本 centroid 的偏移
offset_eucl_list = []
offset_ang_list = []
sigma_list = []
disp_gen_list = []

for gi in range(len(uid_gen)):
    # 真实文本 centroid
    mask_real = [i for i, u in enumerate(all_sent_uidx) if u == gi]
    if len(mask_real) < 3:
        continue
    cen_real = all_sent_wegmann[mask_real].mean(axis=0)
    # 生成文本 centroid
    mask_gen = [i for i in valid_idx if all_uid_idx[i] == gi]
    if len(mask_gen) < 2:
        continue
    cen_gen = gen_emb_full[mask_gen].mean(axis=0)
    # 归一化后计算偏移
    cen_real_n = cen_real / (np.linalg.norm(cen_real) + 1e-8)
    cen_gen_n = cen_gen / (np.linalg.norm(cen_gen) + 1e-8)
    offset_eucl = float(np.linalg.norm(cen_gen_n - cen_real_n))
    cos_sim = np.dot(cen_real_n, cen_gen_n)
    cos_sim = np.clip(cos_sim, -1, 1)
    offset_ang = float(np.arccos(cos_sim))
    offset_eucl_list.append(offset_eucl)
    offset_ang_list.append(offset_ang)
    sigma_list.append(sigma_w[gi])
    disp_gen_list.append(uid_gen_disp[gi]['eucl'])

if len(sigma_list) >= 5:
    print(f"    Gen centroid 偏移（欧氏）: mean={np.mean(offset_eucl_list):.4f} std={np.std(offset_eucl_list):.4f}")
    print(f"    Gen centroid 偏移（角距）: mean={np.mean(offset_ang_list):.4f} std={np.std(offset_ang_list):.4f}")
    rho_off_e, p_off_e = spearmanr(sigma_list, offset_eucl_list)
    rho_off_a, p_off_a = spearmanr(sigma_list, offset_ang_list)
    print(f"    σ_Wegmann → centroid_offset (欧氏): n={len(sigma_list)} ρ={rho_off_e:.4f} p={p_off_e:.2e}")
    print(f"    σ_Wegmann → centroid_offset (角距): n={len(sigma_list)} ρ={rho_off_a:.4f} p={p_off_a:.2e}")

    # 同时测 dispersion_gen 和 centroid_offset 的关系
    rho_disp_off, p_disp_off = spearmanr(disp_gen_list, offset_eucl_list)
    print(f"    dispersion_gen → centroid_offset (欧氏): ρ={rho_disp_off:.4f} p={p_disp_off:.2e}")
    print(f"      (若 high dispersion + small offset → TinyStyler 均匀压缩风格)")
    print(f"      (若 high dispersion + large offset → TinyStyler 方向偏移)")

    summary_rhos['sigma_wegmann_centroid_offset_eucl'] = (rho_off_e, p_off_e)
    summary_rhos['sigma_wegmann_centroid_offset_ang'] = (rho_off_a, p_off_a)


# ════════════════════════════════════════════════════════════════════════════════
# 保存结果
# ════════════════════════════════════════════════════════════════════════════════
results = {
    'stage1_calib_corr': float(calib['corr']),
    'stage2_sigma_target_corr': float(corr_sigma),
    'stage3_mapper_val_cos': float(best_val_cos),
    'n_users': len(uid_gen),
    'n_samples_per_user': N_SAMPLES_GEN,
    'n_generated': len(valid_gen),
    'style_adherence_mean_cos': float(cos_inject.mean()),
    'dispersion_results': {k: {'rho': float(r), 'p': float(p)} for k, (r, p) in summary_rhos.items()},
    'best_dispersion_key': max(summary_rhos, key=lambda k: abs(summary_rhos[k][0])) if summary_rhos else 'none',
    'best_rho': float(max(abs(r) for r, p in summary_rhos.values())) if summary_rhos else 0.0,
}

RESULT_OUT = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian')
RESULT_OUT.mkdir(exist_ok=True, parents=True)
out_path = RESULT_OUT / 'vades_pipeline_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  → {out_path}")

best_key = max(summary_rhos, key=lambda k: abs(summary_rhos[k][0])) if summary_rhos else 'none'
best_rho, best_p = summary_rhos.get(best_key, (0.0, 1.0))
print(f"\n{'='*70}")
print("VADES PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"  Stage 1 (VADESSigma): Corr={calib['corr']:.4f}")
print(f"  Stage 2 (σ regression): TargetCorr={corr_sigma:.4f}")
print(f"  Stage 3 (StyleMapper): val_cos={best_val_cos:.4f}")
print(f"  Stage 4 (Verification):")
print(f"    (a) Style adherence: {cos_inject.mean():.4f}")
print(f"    最佳 dispersion: {best_key}  ρ={best_rho:.4f} {'✓' if abs(best_rho)>0.3 and best_p<0.05 else '✗'}")
print("当前任务已完成，请做下一个任务的指示。")
