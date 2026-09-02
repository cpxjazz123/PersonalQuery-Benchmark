"""
VADES Pipeline — Gaussian Predictive Validity 验证

不经过 TinyStyler 生成，直接测 Gaussian 的 predictive validity：
  80% 真实文本 → 拟合 N(μ_u, σ_u²)
  20% held-out 真实文本 → 计算 dispersion → 与 σ_u 相关性

Stage 1: 训练 VADESSigma（μ_u, σ_u）→ 保存 phase8h_vades.pt
Stage 2: 冻结 μ_u，训练 σ_u 回归目标 → 保存 phase8h_vades_sigma.pt
Stage 3: 训练 StyleMapper（64d→768d）→ 保存 phase8h_style_mapper.pt
Stage 4: Gaussian Predictive Validity（纯净测试，无 TinyStyler 生成）

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
from scipy.stats import spearmanr, f_oneway
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════════
SCRATCH      = Path('/home/wlia0047/hj82_scratch2/wenyu')
EMBED_CACHE  = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/user_review_sentence_extract/uid_embed_cache.pkl')
SENT_CACHE   = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/user_review_sentence_extract/uid_to_sentences.pkl')
RAW_DATA    = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/data/Baby_Products_2023.jsonl.gz')
ATTR_CACHE   = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/product_attributes.json')
SKIP_STAGES_1_3 = False   # False = 训练 VADES/StyleMapper，Stage 4 测 predictive validity
# 新路径（旧 gaussian_vades/ 已删除，改到这里）
GAUSSIAN_CACHE = SCRATCH / 'gaussian_vades_cache.pkl'
VADES_PT     = SCRATCH / 'gaussian_vades_vades.pt'
SIGMA_PT     = SCRATCH / 'gaussian_vades_sigma.pt'
MAPPER_PT    = SCRATCH / 'gaussian_vades_style_mapper.pt'
QWEN_URL     = 'http://localhost:8800/v1/completions'
QWEN_MODEL   = '/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct'

PCA_DIM    = 128
LATENT_DIM = 64
HIDDEN     = 256
WEGMANN_DIM = 768
N_EPOCHS   = 40
LR         = 1e-3
BATCH_SIZE = 512
L_SAMPLES  = 5
SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS        = 1e-6

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SCRATCH / 'gaussian_vades', exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════════
# Stage 0: 加载数据
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Stage 0: 加载数据")
print("="*70)

# 加载 embed 缓存（MPNet 768d → PCA 128d，1000用户×50句）
with open(EMBED_CACHE, 'rb') as f:
    cache = pickle.load(f)
uid_to_embed_768 = cache['uid_to_embed_768']   # 768d 原始 MPNet（Stage 3 StyleMapper 用）
uid_to_embed_128 = cache['uid_to_embed_128']   # 128d PCA（Stage 1 VADES 用）
pca = cache.get('pca', None)

# 加载句子缓存（原始句子文本，用于 stylometric dispersion 和 Stage 4）
with open(SENT_CACHE, 'rb') as f:
    uid_to_sentences_raw = pickle.load(f)

# 取恰好有 100 句的用户（embed cache 前 5145 个）
MAX_USERS = 5145
all_cache_uids = list(uid_to_embed_128.keys())[:MAX_USERS]

# Stage 1 VADESSigma 用 128d PCA embeddings
MIN_SENTS = 10
valid_uids = [u for u in all_cache_uids if u in uid_to_embed_128 and len(uid_to_embed_128[u]) >= MIN_SENTS]
uid_to_idx = {u: i for i, u in enumerate(valid_uids)}
n_users = len(valid_uids)
print(f"  Users: {n_users}")

# uid_to_pca = 128d（Stage 1 VADESSigma 直接用）
uid_to_pca = {uid: np.array(uid_to_embed_128[uid], dtype=np.float32) for uid in all_cache_uids if uid in uid_to_embed_128}

# uid_to_sentences 原始句子缓存（最多50句）
uid_to_sentences = {}
for uid in valid_uids:
    uid_to_sentences[uid] = uid_to_sentences_raw.get(uid, [])[:100]

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
if SKIP_STAGES_1_3:
    print("\n  [SKIP] Stage 1/2/3 — 直接使用经验 Gaussian")
else:
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
_cache_loaded = False
if VADES_PT.exists():
    print(f"  [CACHE] Loading Stage 1 from {VADES_PT}")
    ckpt = torch.load(VADES_PT, map_location=DEVICE)
    model.user_mu.weight.data = ckpt['user_mu'].to(DEVICE)
    model.user_lv.weight.data = ckpt['user_lv'].to(DEVICE)
    model.doc_net.load_state_dict(ckpt['doc_net'])
    model.doc_mu.load_state_dict(ckpt['doc_mu'])
    model.doc_lv.load_state_dict(ckpt['doc_lv'])
    print(f"  [CACHE] Stage 1 loaded (calib_corr={ckpt.get('calib_corr','?')})")
    _cache_loaded = True
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

# 保存 Stage 1 checkpoint（cache 加载时跳过 calibration，直接保存当前模型）
if not _cache_loaded:
    user_lv_np = model.user_lv.weight.detach().cpu().numpy()
    user_sigma = np.sqrt(np.maximum(np.exp(user_lv_np), 1e-6))
    sigma_mean_per_user = user_sigma.mean(axis=1)
    calib = calibration_diagnostics(sigma_mean_per_user, disp_val_arr, "val")
    print(f"  val: Corr={calib['corr']:.4f} MAE={calib['mae']:.4f} Bias={calib['bias']:+.4f} Sharp={calib['sharpness']:.4f}")
else:
    calib = {'corr': float(ckpt.get('calib_corr', 0))}

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
    pl = np.array(uid_to_pca[uid], dtype=np.float32)
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
        embeds = uid_to_embed_768.get(uid, [])
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
# Stage 4: Gaussian Predictive Validity（纯净测试，不经过 TinyStyler）
#
# 用 MPNet (768d) 对真实句子编码，在 Wegmann 空间测 Gaussian predictive validity
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Stage 4: Gaussian Predictive Validity（纯净测试）")
print("="*70)

# 加载 AnnaWegmann/Style-Embedding（ACL 2022 原生 Wegmann 模型，768d）
from sentence_transformers import SentenceTransformer
wegmann = SentenceTransformer('AnnaWegmann/Style-Embedding', device=DEVICE)
wegmann.eval()
print("  Wegmann (AnnaWegmann/Style-Embedding, 768d) loaded")

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

# ── Stage 4a: 批量编码所有用户句子 → MPNet 768d
print("\n  Stage 4a: 批量编码用户句子（Wegmann 768d）...")

def encode_sentences_to_wegmann(texts, batch_size=256):
    """text → MPNet → 768d Wegmann space"""
    all_768d = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        with torch.no_grad():
            emb = wegmann.encode(batch, batch_size=len(batch), normalize_embeddings=True)  # (batch, 768)
        all_768d.append(emb.astype(np.float32))
    return np.concatenate(all_768d, axis=0).astype(np.float32)  # (N, 768)

# SKIP: 直接从 embed 缓存加载 768d（不需要重新编码）
if SKIP_STAGES_1_3:
    uid_gen = list(valid_uids)
    uid_to_sentence_list = {}
    all_sent_wegmann_list = []
    all_sent_uidx = []
    for gi, uid in enumerate(uid_gen):
        sents = uid_to_sentences_raw.get(uid, [])[:100]
        uid_to_sentence_list[uid] = sents
        emb_768 = np.array(uid_to_embed_768[uid], dtype=np.float32)
        for emb in emb_768:
            all_sent_wegmann_list.append(emb)
            all_sent_uidx.append(gi)
    all_sent_wegmann = np.array(all_sent_wegmann_list)
    print(f"  [SKIP] Loaded from cache: {len(uid_gen)} users, {len(all_sent_wegmann)} sents, shape: {all_sent_wegmann.shape}")
else:
    uid_gen = list(valid_uids)
    uid_to_sentence_list = {}
    all_sent_wegmann_list = []
    all_sent_uidx = []
    for gi, uid in enumerate(uid_gen):
        sents = uid_to_sentences_raw.get(uid, [])[:100]
        uid_to_sentence_list[uid] = sents
        emb_768 = np.array(uid_to_embed_768[uid], dtype=np.float32)
        for emb in emb_768:
            all_sent_wegmann_list.append(emb)
            all_sent_uidx.append(gi)
    all_sent_wegmann = np.array(all_sent_wegmann_list)
    print(f"  [CACHE] Loaded from embed cache: {len(uid_gen)} users, {len(all_sent_wegmann)} sents, shape: {all_sent_wegmann.shape}")

# ── O(n log n) 索引：argsort + searchsorted，替代三处 O(n²) 的 enumerate+if ──
_sent_uidx_arr = np.array(all_sent_uidx, dtype=np.int32)
_sent_sort_idx = np.argsort(_sent_uidx_arr, kind='quicksort')
_uid_counts = np.bincount(_sent_uidx_arr, minlength=len(uid_gen))
_uid_offsets = np.concatenate([[0], np.cumsum(_uid_counts[:-1])])
def _mask_for(gi):
    o, c = _uid_offsets[gi], _uid_counts[gi]
    return _sent_sort_idx[o:o+c]

# ── 768d Wegmann 空间 centroid ──
uid_to_wegmann_centroid_768 = {}
for gi, uid in enumerate(uid_gen):
    mask = _mask_for(gi)
    if len(mask) >= 3:
        uid_to_wegmann_centroid_768[uid] = all_sent_wegmann[mask].mean(axis=0)

# 用户在 768d Wegmann 空间的 μ_u 和 σ_u
mu_w = np.zeros((len(uid_gen), WEGMANN_DIM), dtype=np.float32)
sigma_w = np.zeros(len(uid_gen), dtype=np.float32)
for gi, uid in enumerate(uid_gen):
    mask = _mask_for(gi)
    if len(mask) >= 3:
        embs = all_sent_wegmann[mask]
        mu_w[gi] = embs.mean(axis=0)
        sigma_w[gi] = np.linalg.norm(embs - mu_w[gi], axis=1).mean()
print(f"  User centroids: {len(uid_to_wegmann_centroid_768)} users computed")

# ════════════════════════════════════════════════════════════════════════════════
# Stage 4 — Gaussian Predictive Validity（纯净测试，不经过 TinyStyler）
#
# 核心问题：Gaussian 本身是否能预测用户未来文本的风格分布？
#
# 实验设计：
#   80% 真实文本 → 拟合 N(μ_u, σ_u²)
#   20% held-out 真实文本 → 计算真实 dispersion D_u^heldout
#
# 两个核心指标：
#   (a) cos(μ_train, μ_heldout)  — 中心预测能力
#   (b) Corr(σ_train, D_u^heldout) — 范围预测能力
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Stage 4: Gaussian Predictive Validity（纯净测试）")
print("="*70)

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

# ── 从 VADES checkpoint 加载 σ_u（始终需要）─
ckpt_vades = torch.load(VADES_PT, map_location='cpu', weights_only=True)
user_lv_np = ckpt_vades['user_lv'].numpy()  # (n_users, 64)
sigma_vades_per_user = np.sqrt(np.maximum(np.exp(user_lv_np), 1e-6)).mean(axis=1)  # (n_users,)
print(f"  VADES σ_u loaded: {len(sigma_vades_per_user)} users, mean={sigma_vades_per_user.mean():.4f}")

# ── Per-user Gaussian 缓存（基于 embed cache + MAX_USERS + SEED 签名）─
_embed_mtime = EMBED_CACHE.stat().st_mtime if EMBED_CACHE.exists() else 0
_cache_sig = {
    'embed_mtime': _embed_mtime,
    'max_users': MAX_USERS,
    'seed': SEED,
    'min_sents': MIN_SENTS,
}
_cache_valid = False
if GAUSSIAN_CACHE.exists():
    try:
        with open(GAUSSIAN_CACHE, 'rb') as f:
            _cached = pickle.load(f)
        if _cached.get('signature') == _cache_sig:
            cos_train_heldout           = _cached['cos_train_heldout']
            sigma_train_list            = _cached['sigma_train_list']
            sigma_maha_train_list       = _cached['sigma_maha_train_list']
            disp_heldout_eucl           = _cached['disp_heldout_eucl']
            disp_heldout_ang            = _cached['disp_heldout_ang']
            disp_heldout_ang_from_train_cen = _cached['disp_heldout_ang_from_train_cen']
            sigma_maha_heldout_list     = _cached['sigma_maha_heldout_list']
            N_u_list   = _cached['N_u_list']
            L_u_list   = _cached['L_u_list']
            W_u_list   = _cached['W_u_list']
            Q_mu_list  = _cached['Q_mu_list']
            Q_sigma_list = _cached['Q_sigma_list']
            n_users_used = len(cos_train_heldout)
            _cache_valid = True
            print(f"  [CACHE] Loaded per-user Gaussian from {GAUSSIAN_CACHE} ({n_users_used} users)")
        else:
            print(f"  [CACHE] Signature mismatch, recomputing...")
    except Exception as e:
        print(f"  [CACHE] Load failed ({e}), recomputing...")

if not _cache_valid:
    cos_train_heldout = []
    sigma_train_list = []
    sigma_maha_train_list = []
    disp_heldout_eucl = []
    disp_heldout_ang = []
    disp_heldout_ang_from_train_cen = []
    sigma_maha_heldout_list = []
    N_u_list = []
    L_u_list = []
    W_u_list = []
    Q_mu_list = []
    Q_sigma_list = []
    n_users_used = 0

    for gi, uid in enumerate(uid_gen):
        mask_all = _mask_for(gi)
        n_total = len(mask_all)
        if n_total < 10:
            continue

        rng_split = np.random.default_rng(SEED + gi)
        idx_shuffled = list(range(n_total))
        rng_split.shuffle(idx_shuffled)
        n_train = int(np.ceil(n_total * 0.8))
        idx_train = idx_shuffled[:n_train]
        idx_held = idx_shuffled[n_train:]

        emb_train = all_sent_wegmann[mask_all][idx_train]
        mu_train = emb_train.mean(axis=0)
        mu_train_n = mu_train / (np.linalg.norm(mu_train) + 1e-8)

        diffs = emb_train - mu_train
        var_diag = diffs.var(axis=0) + 1e-4
        # 经验 σ（原始）vs VADES σ_u（训练好的）
        sigma_tr = float(sigma_vades_per_user[gi])   # VADES-learned σ_u
        D_maha_train = float(np.sqrt((diffs ** 2 / var_diag).sum(axis=1)).mean())

        emb_held = all_sent_wegmann[mask_all][idx_held]
        n_held = len(emb_held)
        if n_held < 2:
            continue

        mu_held = emb_held.mean(axis=0)
        mu_held_n = mu_held / (np.linalg.norm(mu_held) + 1e-8)
        cos_sim = float(np.dot(mu_train_n, mu_held_n))
        cos_train_heldout.append(cos_sim)

        emb_held_diff = emb_held - mu_held
        d_eucl = float(np.linalg.norm(emb_held_diff, axis=1).mean())
        d_ang = angular_disp(emb_held)
        disp_heldout_eucl.append(d_eucl)
        disp_heldout_ang.append(d_ang)

        emb_held_n = emb_held / (np.linalg.norm(emb_held, axis=1, keepdims=True) + 1e-8)
        cos_sims = np.clip(np.sum(emb_held_n * mu_train_n, axis=1), -1, 1)
        angles = np.arccos(cos_sims)
        d_ang_from_tr = float(np.nanmean(angles))
        disp_heldout_ang_from_train_cen.append(d_ang_from_tr)

        D_maha_heldout = float(np.sqrt((emb_held_diff ** 2 / var_diag).sum(axis=1)).mean())

        sigma_train_list.append(sigma_tr)
        sigma_maha_train_list.append(D_maha_train)
        sigma_maha_heldout_list.append(D_maha_heldout)

        N_u_list.append(n_total)
        L_u_list.append(float(np.mean([len(s.split()) for s in uid_to_sentence_list[uid]])))
        W_u_list.append(float(np.sum([len(s.split()) for s in uid_to_sentence_list[uid]])))
        Q_mu_list.append(cos_sim)
        Q_sigma_list.append(abs(sigma_tr - d_eucl))
        n_users_used += 1

    _cached_data = {
        'signature': _cache_sig,
        'cos_train_heldout': cos_train_heldout,
        'sigma_train_list': sigma_train_list,
        'sigma_maha_train_list': sigma_maha_train_list,
        'disp_heldout_eucl': disp_heldout_eucl,
        'disp_heldout_ang': disp_heldout_ang,
        'disp_heldout_ang_from_train_cen': disp_heldout_ang_from_train_cen,
        'sigma_maha_heldout_list': sigma_maha_heldout_list,
        'N_u_list': N_u_list,
        'L_u_list': L_u_list,
        'W_u_list': W_u_list,
        'Q_mu_list': Q_mu_list,
        'Q_sigma_list': Q_sigma_list,
    }
    with open(GAUSSIAN_CACHE, 'wb') as f:
        pickle.dump(_cached_data, f)
    print(f"  [CACHE] Saved to {GAUSSIAN_CACHE} ({n_users_used} users)")

cos_arr = np.array(cos_train_heldout)
sigma_vades_arr = np.array(sigma_train_list)
sigma_maha_tr_arr = np.array(sigma_maha_train_list)
disp_eucl_arr = np.array(disp_heldout_eucl)
disp_ang_arr = np.array(disp_heldout_ang)
disp_ang_ft_arr = np.array(disp_heldout_ang_from_train_cen)
sigma_maha_ho_arr = np.array(sigma_maha_heldout_list)

print(f"  n_users: {n_users_used}")
print(f"\n  (a) 中心预测 — cos(μ_train, μ_heldout):")
print(f"      mean={cos_arr.mean():.4f}  median={np.median(cos_arr):.4f}  min={cos_arr.min():.4f}")
print(f"      → 期望 > 0.9（用户风格中心稳定）")

print(f"\n  (b) 范围预测:")
rho_sigma_tr_ang, p_sta = spearmanr(sigma_vades_arr, disp_ang_ft_arr)
rho_sigma_tr_eucl, p_ste = spearmanr(sigma_vades_arr, disp_eucl_arr)
rho_maha_tr_ho, p_mho = spearmanr(sigma_maha_tr_arr, sigma_maha_ho_arr)
print(f"      σ_VADES → Dispersion_heldout (角距): ρ={rho_sigma_tr_ang:.4f}  p={p_sta:.2e}")
print(f"      σ_VADES → Dispersion_heldout (欧氏): ρ={rho_sigma_tr_eucl:.4f}  p={p_ste:.2e}")
print(f"      D_maha_train → D_maha_heldout: ρ={rho_maha_tr_ho:.4f}  p={p_mho:.2e}")
print(f"      → 期望 ρ > 0.3 且 p < 0.05")

# ════════════════════════════════════════════════════════════════════════════════
# Part C: 长度分解分析（W_u / N_u / L_u 对 Q_mu / Q_sigma 的影响）
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part C: 长度分解分析")
print("="*70)

N_arr = np.array(N_u_list, dtype=float)
L_arr = np.array(L_u_list, dtype=float)
W_arr = np.array(W_u_list, dtype=float)
logN_arr = np.log(N_arr + 1)
Q_mu_arr = np.array(Q_mu_list, dtype=float)
Q_sigma_arr = np.array(Q_sigma_list, dtype=float)

print(f"\n  基础统计:")
print(f"    N_u: mean={N_arr.mean():.1f} median={np.median(N_arr):.0f} min={N_arr.min():.0f} max={N_arr.max():.0f}")
print(f"    L_u: mean={L_arr.mean():.1f} median={np.median(L_arr):.1f} min={L_arr.min():.1f} max={L_arr.max():.1f}")
print(f"    W_u: mean={W_arr.mean():.0f} median={np.median(W_arr):.0f} min={W_arr.min():.0f} max={W_arr.max():.0f}")

print(f"\n  相关性分析:")
for label, predictor, target, name in [
    ("W_u → Q_mu",   W_arr,       Q_mu_arr,    "W_u"),
    ("logN_u → Q_mu", logN_arr,   Q_mu_arr,    "logN_u"),
    ("L_u → Q_mu",   L_arr,       Q_mu_arr,    "L_u"),
    ("W_u → Q_sigma", W_arr,      Q_sigma_arr, "W_u"),
    ("logN_u → Q_sigma", logN_arr, Q_sigma_arr, "logN_u"),
    ("L_u → Q_sigma", L_arr,      Q_sigma_arr, "L_u"),
]:
    r, p = spearmanr(predictor, target)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"    {label}: ρ={r:+.4f} p={p:.2e} {sig}")

print(f"\n  多元回归 — Q_mu = β0 + β1*logN_u + β2*L_u:")
X = np.column_stack([logN_arr, L_arr])
X = sm.add_constant(X)
try:
    m_qmu = sm.OLS(Q_mu_arr, X).fit()
    print(f"    β0(const)={m_qmu.params[0]:+.4f} p={m_qmu.pvalues[0]:.2e}")
    print(f"    β1(logN_u)={m_qmu.params[1]:+.4f} p={m_qmu.pvalues[1]:.2e} {'*' if m_qmu.pvalues[1]<0.05 else ''}")
    print(f"    β2(L_u)={m_qmu.params[2]:+.4f} p={m_qmu.pvalues[2]:.2e} {'*' if m_qmu.pvalues[2]<0.05 else ''}")
    print(f"    R²={m_qmu.rsquared:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

print(f"\n  多元回归 — Q_sigma = β0 + β1*logN_u + β2*L_u:")
try:
    m_qsig = sm.OLS(Q_sigma_arr, X).fit()
    print(f"    β0(const)={m_qsig.params[0]:+.4f} p={m_qsig.pvalues[0]:.2e}")
    print(f"    β1(logN_u)={m_qsig.params[1]:+.4f} p={m_qsig.pvalues[1]:.2e} {'*' if m_qsig.pvalues[1]<0.05 else ''}")
    print(f"    β2(L_u)={m_qsig.params[2]:+.4f} p={m_qsig.pvalues[2]:.2e} {'*' if m_qsig.pvalues[2]<0.05 else ''}")
    print(f"    R²={m_qsig.rsquared:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# Part D: Matched Experiment（N=20 固定句子数，按 L_u 分 short/medium/long）
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part D: Matched Experiment (N=20 sentences)")
print("="*70)

# 取恰好有≥20句的用户，取前20句
N_FIXED = 100
idx_ge20 = [i for i, n in enumerate(N_u_list) if n >= N_FIXED]
print(f"  有≥{N_FIXED}句的用户: {len(idx_ge20)}/{len(N_u_list)}")

N20_N = np.array([N_u_list[i] for i in idx_ge20])
N20_L = np.array([L_u_list[i] for i in idx_ge20])
N20_Qmu = np.array([Q_mu_list[i] for i in idx_ge20])
N20_Qsigma = np.array([Q_sigma_list[i] for i in idx_ge20])

# 按 L_u 三分位分 short / medium / long
p33, p67 = float(np.percentile(N20_L, 33)), float(np.percentile(N20_L, 67))
groups = np.where(N20_L <= p33, 'short', np.where(N20_L <= p67, 'medium', 'long'))
print(f"  L_u 分位点: short≤{p33:.1f} medium≤{p67:.1f} long>{p67:.1f}")

print(f"\n  Q_mu = cos(μ_train, μ_heldout) 对比:")
for g in ['short', 'medium', 'long']:
    mask = groups == g
    vals = N20_Qmu[mask]
    print(f"    {g:8s}: n={mask.sum():3d}  mean={vals.mean():.4f}  std={vals.std():.4f}")

print(f"\n  Q_sigma = |σ_train - Disp_heldout| 对比:")
for g in ['short', 'medium', 'long']:
    mask = groups == g
    vals = N20_Qsigma[mask]
    print(f"    {g:8s}: n={mask.sum():3d}  mean={vals.mean():.4f}  std={vals.std():.4f}")

# ANOVA 检验
for label, vals_by_group in [("Q_mu", [N20_Qmu[groups==g] for g in ['short','medium','long']]),
                               ("Q_sigma", [N20_Qsigma[groups==g] for g in ['short','medium','long']])]:
    try:
        f, pval = f_oneway(*vals_by_group)
        print(f"\n  ANOVA {label}: F={f:.3f} p={pval:.4f} {'*' if pval<0.05 else ''}")
    except Exception:
        pass

# 保存结果
summary_rhos_gaussian = {
    'cos_center_prediction': {'mean': float(cos_arr.mean()), 'median': float(np.median(cos_arr))},
    'sigma_train_euclidean_vs_disp_heldout_ang': {'rho': float(rho_sigma_tr_ang), 'p': float(p_sta)},
    'sigma_train_euclidean_vs_disp_heldout_eucl': {'rho': float(rho_sigma_tr_eucl), 'p': float(p_ste)},
    'D_maha_train_vs_D_maha_heldout': {'rho': float(rho_maha_tr_ho), 'p': float(p_mho)},
    'n_users': n_users_used,
}

# ════════════════════════════════════════════════════════════════════════════════
# 保存结果
# ════════════════════════════════════════════════════════════════════════════════
RESULT_OUT = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian')
RESULT_OUT.mkdir(exist_ok=True, parents=True)

# ── 保存 per-user Gaussian 分布（μ_u 64d + σ_u 向量）──
gauss_out = RESULT_OUT / 'vades_user_gaussians.npz'
np.savez_compressed(
    gauss_out,
    user_ids=np.array([str(u) for u in valid_uids]),
    mu_64d=user_mu_np.astype(np.float32),
    lv_64d=user_lv_np.astype(np.float32),
    sigma_vades=sigma_vades_per_user.astype(np.float32),
)
print(f"  → {gauss_out}")

results = {
    'stage1_calib_corr': float(calib['corr']),
    'stage2_sigma_target_corr': float(corr_sigma),
    'stage3_mapper_val_cos': float(best_val_cos),
    'n_users': n_users_used,
    'gaussian_predictive_validity': summary_rhos_gaussian,
}

out_path = RESULT_OUT / 'vades_pipeline_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  → {out_path}")

print(f"\n{'='*70}")
print("VADES PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"  Stage 1 (VADESSigma): Corr={calib['corr']:.4f}")
print(f"  Stage 2 (σ regression): TargetCorr={corr_sigma:.4f}")
print(f"  Stage 3 (StyleMapper): val_cos={best_val_cos:.4f}")
print(f"  Stage 4 (Gaussian Predictive Validity — 经验 Gaussian):")
print(f"    (a) cos(μ_train, μ_heldout): {cos_arr.mean():.4f}")
print(f"    (b) σ_train(Euclid) → Disp_heldout (ang): ρ={rho_sigma_tr_ang:.4f} p={p_sta:.2e}")
print(f"    (c) σ_train(Euclid) → Disp_heldout (eucl): ρ={rho_sigma_tr_eucl:.4f} p={p_ste:.2e}")
print(f"    (d) D_maha_train → D_maha_heldout: ρ={rho_maha_tr_ho:.4f} p={p_mho:.2e}")
rho_keys = [k for k in summary_rhos_gaussian
            if isinstance(summary_rhos_gaussian[k], dict) and 'rho' in summary_rhos_gaussian[k]]
if rho_keys:
    best_key = max(rho_keys, key=lambda k: abs(summary_rhos_gaussian[k]['rho']))
    best_rho = summary_rhos_gaussian[best_key]['rho']
    best_p = summary_rhos_gaussian[best_key]['p']
else:
    best_key = 'none'; best_rho = 0.0; best_p = 1.0
print(f"    最佳: {best_key} ρ={best_rho:.4f} {'✓' if abs(best_rho)>0.3 and best_p<0.05 else '✗'}")
print("当前任务已完成，请做下一个任务的指示。")

