"""
经验 Gaussian 验证 — 直接用真实文本在风格空间里的经验均值和协方差

不依赖 VADES / StyleMapper，直接测 Gaussian predictive validity：
  80% 真实文本 → 拟合经验 N(μ_u, Σ_u)
  20% held-out 真实文本 → 计算 dispersion → 评估预测能力

Part A: σ_train(Euclidean/Mahalanobis) → Dispersion_heldout 相关性
Part B: 长度分解分析（W_u / N_u / L_u 对 Q_mu / Q_sigma 的影响）
Part C: Matched Experiment（N=20 固定句子数，按 L_u 分 short/medium/long）
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'
import pickle, json, sys, time, hashlib
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, f_oneway
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

SCRATCH          = Path('/home/wlia0047/hj82_scratch2/wenyu')
EMBED_CACHE      = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/user_review_sentence_extract/uid_embed_cache.pkl')
SENT_CACHE       = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/user_review_sentence_extract/uid_to_sentences.pkl')
GAUSSIAN_CACHE   = SCRATCH / 'gaussian_empirical_cache.pkl'
WEGMANN_DIM      = 768
PCA_DIM          = 128
SEED             = 42
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULT_OUT       = Path('/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian')
RESULT_OUT.mkdir(exist_ok=True, parents=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

# ════════════════════════════════════════════════════════════════════════════════
# Stage 0: 加载数据
# ════════════════════════════════════════════════════════════════════════════════
print("="*70)
print("Stage 0: 加载数据")
print("="*70)

with open(EMBED_CACHE, 'rb') as f:
    cache = pickle.load(f)
uid_to_embed_768 = cache['uid_to_embed_768']
uid_to_embed_128 = cache['uid_to_embed_128']

with open(SENT_CACHE, 'rb') as f:
    uid_to_sentences_raw = pickle.load(f)

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
    uid_to_sentences[uid] = uid_to_sentences_raw.get(uid, [])[:50]

print(f"  Loaded {n_users} users from embed cache")

# 直接从 embed 缓存加载 768d embeddings（已由 AnnaWegmann 编码）
print("Loading 768d embeddings from cache...")
uid_gen = list(valid_uids)
uid_to_sentence_list = {}
all_sent_wegmann_list = []
all_sent_uidx = []
for gi, uid in enumerate(uid_gen):
    sents = uid_to_sentences_raw.get(uid, [])[:50]
    uid_to_sentence_list[uid] = sents
    emb_768 = np.array(uid_to_embed_768[uid], dtype=np.float32)
    for emb in emb_768:
        all_sent_wegmann_list.append(emb)
        all_sent_uidx.append(gi)

all_sent_wegmann = np.array(all_sent_wegmann_list)
print(f"  Loaded {len(uid_gen)} users, {len(all_sent_wegmann)} sentences, shape: {all_sent_wegmann.shape}")

# ════════════════════════════════════════════════════════════════════════════════
# Stage 4 — 经验 Gaussian Predictive Validity
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Stage 4: 经验 Gaussian Predictive Validity")
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

# ── Per-user Gaussian 缓存（基于 embed cache + MAX_USERS + SEED 签名）─
_embed_mtime = EMBED_CACHE.stat().st_mtime if EMBED_CACHE.exists() else 0
_cache_signature = {
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
        if _cached.get('signature') == _cache_signature:
            cos_train_heldout   = _cached['cos_train_heldout']
            sigma_train_list    = _cached['sigma_train_list']
            sigma_maha_train_list = _cached['sigma_maha_train_list']
            disp_heldout_eucl   = _cached['disp_heldout_eucl']
            disp_heldout_ang    = _cached['disp_heldout_ang']
            disp_heldout_ang_from_train_cen = _cached['disp_heldout_ang_from_train_cen']
            sigma_maha_heldout_list  = _cached['sigma_maha_heldout_list']
            N_u_list   = _cached['N_u_list']
            L_u_list   = _cached['L_u_list']
            W_u_list   = _cached['W_u_list']
            Q_mu_list  = _cached['Q_mu_list']
            Q_sigma_list = _cached['Q_sigma_list']
            mu_768_list = _cached.get('mu_768_list', [])
            n_users_used = len(cos_train_heldout)
            _cache_valid = True
            print(f"  [CACHE] Loaded per-user Gaussian from {GAUSSIAN_CACHE} ({n_users_used} users)")
        else:
            print(f"  [CACHE] Signature mismatch, recomputing per-user Gaussian...")
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
    mu_768_list = []

    for gi, uid in enumerate(uid_gen):
        mask_all = [i for i, u in enumerate(all_sent_uidx) if u == gi]
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
        mu_768_list.append(mu_train.copy())
        mu_train_n = mu_train / (np.linalg.norm(mu_train) + 1e-8)

        diffs = emb_train - mu_train
        var_diag = diffs.var(axis=0) + 1e-4
        sigma_tr = float(np.linalg.norm(diffs, axis=1).mean())
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

    # 保存缓存
    _cached_data = {
        'signature': _cache_signature,
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
        'mu_768_list': mu_768_list,
    }
    with open(GAUSSIAN_CACHE, 'wb') as f:
        pickle.dump(_cached_data, f)
    print(f"  [CACHE] Saved to {GAUSSIAN_CACHE} ({n_users_used} users)")

disp_heldout_ang_ft_arr = np.array(disp_heldout_ang_from_train_cen)
sigma_maha_ho_arr = np.array(sigma_maha_heldout_list)

cos_arr = np.array(cos_train_heldout)
sigma_tr_arr = np.array(sigma_train_list)
sigma_maha_tr_arr = np.array(sigma_maha_train_list)
disp_eucl_arr = np.array(disp_heldout_eucl)
disp_ang_arr = np.array(disp_heldout_ang)
disp_ang_ft_arr = np.array(disp_heldout_ang_from_train_cen)
sigma_maha_ho_arr = np.array(sigma_maha_heldout_list)

# ── 保存 per-user 经验 Gaussian 分布（768d μ + 标量 σ）──
mu_768_arr = np.array(mu_768_list, dtype=np.float32)  # (n_users, 768)
sigma_arr = np.array(sigma_train_list, dtype=np.float32)  # (n_users,)
emp_out = RESULT_OUT / 'empirical_user_gaussians.npz'
np.savez_compressed(
    emp_out,
    user_ids=np.array([str(u) for u in uid_gen[:n_users_used]]),
    mu_768d=mu_768_arr,
    sigma_euclidean=sigma_arr,
)
print(f"  → {emp_out}")

print(f"  n_users: {n_users_used}")
print(f"\n  (A) 中心预测 — cos(μ_train, μ_heldout):")
print(f"      mean={cos_arr.mean():.4f}  median={np.median(cos_arr):.4f}  min={cos_arr.min():.4f}")

print(f"\n  (B) 范围预测:")
rho_sigma_tr_ang, p_sta = spearmanr(sigma_tr_arr, disp_ang_ft_arr)
rho_sigma_tr_eucl, p_ste = spearmanr(sigma_tr_arr, disp_eucl_arr)
rho_maha_tr_ho, p_mho = spearmanr(sigma_maha_tr_arr, sigma_maha_ho_arr)
print(f"      σ_train(Euclidean) → Disp_heldout (角距): ρ={rho_sigma_tr_ang:+.4f}  p={p_sta:.2e}")
print(f"      σ_train(Euclidean) → Disp_heldout (欧氏): ρ={rho_sigma_tr_eucl:+.4f}  p={p_ste:.2e}")
print(f"      D_maha_train → D_maha_heldout: ρ={rho_maha_tr_ho:+.4f}  p={p_mho:.2e}")

# ════════════════════════════════════════════════════════════════════════════════
# Part C: 长度分解分析
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part C: 长度分解分析")
print("="*70)

import statsmodels.api as sm

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
for label, predictor, target in [
    ("W_u → Q_mu",         W_arr,    Q_mu_arr),
    ("logN_u → Q_mu",      logN_arr, Q_mu_arr),
    ("L_u → Q_mu",         L_arr,    Q_mu_arr),
    ("W_u → Q_sigma",      W_arr,    Q_sigma_arr),
    ("logN_u → Q_sigma",   logN_arr, Q_sigma_arr),
    ("L_u → Q_sigma",      L_arr,    Q_sigma_arr),
]:
    r, p = spearmanr(predictor, target)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"    {label}: ρ={r:+.4f} p={p:.2e} {sig}")

print(f"\n  多元回归 — Q_mu = β0 + β1*logN_u + β2*L_u:")
X = np.column_stack([logN_arr, L_arr])
X = sm.add_constant(X)
try:
    m_qmu = sm.OLS(Q_mu_arr, X).fit()
    print(f"    β0(const)={m_qmu.params[0]:+.4f}")
    print(f"    β1(logN_u)={m_qmu.params[1]:+.4f} p={m_qmu.pvalues[1]:.2e}{'*' if m_qmu.pvalues[1]<0.05 else ''}")
    print(f"    β2(L_u)={m_qmu.params[2]:+.4f} p={m_qmu.pvalues[2]:.2e}{'*' if m_qmu.pvalues[2]<0.05 else ''}")
    print(f"    R²={m_qmu.rsquared:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

print(f"\n  多元回归 — Q_sigma = β0 + β1*logN_u + β2*L_u:")
try:
    m_qsig = sm.OLS(Q_sigma_arr, X).fit()
    print(f"    β0(const)={m_qsig.params[0]:+.4f}")
    print(f"    β1(logN_u)={m_qsig.params[1]:+.4f} p={m_qsig.pvalues[1]:.2e}{'*' if m_qsig.pvalues[1]<0.05 else ''}")
    print(f"    β2(L_u)={m_qsig.params[2]:+.4f} p={m_qsig.pvalues[2]:.2e}{'*' if m_qsig.pvalues[2]<0.05 else ''}")
    print(f"    R²={m_qsig.rsquared:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# Part D: Matched Experiment（N=20 固定句子数，按 L_u 分 short/medium/long）
# ════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part D: Matched Experiment (N=20 sentences)")
print("="*70)

N_FIXED = 100
idx_ge20 = [i for i, n in enumerate(N_u_list) if n >= N_FIXED]
print(f"  有≥{N_FIXED}句的用户: {len(idx_ge20)}/{len(N_u_list)}")

N20_L = np.array([L_u_list[i] for i in idx_ge20])
N20_Qmu = np.array([Q_mu_list[i] for i in idx_ge20])
N20_Qsigma = np.array([Q_sigma_list[i] for i in idx_ge20])

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

# ANOVA
for label, vals_by_group in [
    ("Q_mu",    [N20_Qmu[groups==g]    for g in ['short','medium','long']]),
    ("Q_sigma", [N20_Qsigma[groups==g] for g in ['short','medium','long']]),
]:
    try:
        f, pval = f_oneway(*vals_by_group)
        print(f"\n  ANOVA {label}: F={f:.3f} p={pval:.4f}{'*' if pval<0.05 else ''}")
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════════════════════
# 保存结果
# ════════════════════════════════════════════════════════════════════════════════
summary = {
    'n_users': n_users_used,
    'cos_center_prediction': {'mean': float(cos_arr.mean()), 'median': float(np.median(cos_arr))},
    'sigma_euclidean_vs_disp_heldout_ang': {'rho': float(rho_sigma_tr_ang), 'p': float(p_sta)},
    'sigma_euclidean_vs_disp_heldout_eucl': {'rho': float(rho_sigma_tr_eucl), 'p': float(p_ste)},
    'D_maha_train_vs_D_maha_heldout': {'rho': float(rho_maha_tr_ho), 'p': float(p_mho)},
    'length_decomposition': {
        'N_mean': float(N_arr.mean()), 'N_median': float(np.median(N_arr)),
        'L_mean': float(L_arr.mean()), 'L_median': float(np.median(L_arr)),
        'W_mean': float(W_arr.mean()), 'W_median': float(np.median(W_arr)),
    },
}

out_path = RESULT_OUT / 'empirical_gaussian_results.json'
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n  → {out_path}")
print(f"\n{'='*70}")
print("EXPERIMENT COMPLETE")
print(f"{'='*70}")
