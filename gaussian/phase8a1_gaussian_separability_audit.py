"""Phase 8.A.1 — Gaussian Separability Audit.

Question: Does the trained Learned Syntax Gaussian actually separate users in
its 32d subspace? Don't just look at overall Rank@1=29.9%. Inspect geometry directly.

5 audits on B00A4B34IA, 12-user cohort, 80/20 train/test:

  (1) μ separation:
      ||μ_u - μ_v|| for learned 32d vs PCA48 48d centroids

  (2) Bhattacharyya distance D_B between every (u, v) Gaussian pair
      (diagonal closed form)

  (3) Within-user vs between-user distance for held-out real sentences:
      M_G = D_nearest_other - D_self
      → overall P(M_G > 0) and median M_G

  (4) Per-user P_u(M_G > 0) for all 12 users

  (5) σ_bias: ρ(σ_mean, rank1 frequency) — recheck

GOAL: confirm that learned model separated users geometrically, not just
boosted classification accuracy by accident.
"""
import collections, hashlib, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
sys.path.insert(0, f"{REPO}/common")
from syntactic_features import per_sentence_features_v2 as psf
import spacy, gzip, re

# ────────── Config ──────────
COHORT = [
    "AHM56WA4FAB2KZITXPRW2WUPZIMQ", "AHLG5OEROOZJVHMD6TNKUKOL6GFA",
    "AHACBVNWG7USY3FSLG2TPVWOQZZA", "AEVYSHLGGB64TL6SSODHFRP67VGQ",
    "AGAORBCX76OT4GQU3SJZF3TNICWA", "AFIROMS23ORMKGET6XC6ZEDCTEXQ",
    "AGKSBZKWZHGJKCNMBMM6AYTX3WHQ", "AEPMPLRZWHU7VAPDGCFF2JPQDJYQ",
    "AECSNCOIZWGHJCONYOWQE5IIARCQ", "AG2ADGIK63GFBE4DLKJXCJ2DS3MA",
    "AH2EEKGHPEZTYJLJX7DESQ3VY4GQ", "AERM7FIHHMIQH5FSEATT4WBBKF3Q",
]
ASIN = "B00A4B34IA"
SEED_PREFIX = "phase8a_v1|"
TRAIN_FRAC = 0.80

# F3 features
gdoc = np.load("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_shared_scaler.npz", allow_pickle=True)
ALL_FNAMES = list(gdoc["feature_names_ordered"])
FNAMES_F3 = list(gdoc["fnames_f3"])
assert len(FNAMES_F3) == 103
SCALER_MEAN = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
SCALER_SCALE = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
PCA_COMP = np.asarray(gdoc["pca_components"], dtype=np.float64)
PCA_MEAN = np.asarray(gdoc["pca_mean"], dtype=np.float64)
gdoc.close()
F3_COL = [ALL_FNAMES.index(nm) for nm in FNAMES_F3]

# Model
N_USERS = len(COHORT)
D_IN = 103
D_HID = [128, 64]
D_OUT = 32
SIGMA_MIN = 0.1
SIGMA_MAX = 2.0
TAU = 1.0
LAMBDA_SIGMA = 0.01
LR = 1e-3
EPOCHS = 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

NLP = spacy.load("en_core_web_sm", disable=["ner","lemmatizer","attribute_ruler"])


def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def is_train_sentence(text):
    h = int(hashlib.sha1((SEED_PREFIX + text.strip().lower()).encode()).hexdigest(), 16)
    return (h % 100) < int(TRAIN_FRAC * 100)


def featurize_103(text):
    sents = [s for s in split_sents(text) if 8 <= len(s.split()) <= 60]
    if not sents: return None
    doc = next(NLP.pipe([sents[0]], batch_size=1, n_process=1))
    sds = list(doc.sents)
    if not sds: return None
    feats = []
    for s in sds:
        f = psf(s)
        if f is not None:
            feats.append(f)
    if not feats: return None
    all_keys = set()
    for f in feats:
        all_keys.update(f.keys())
    mean_feats = {}
    for k in all_keys:
        vals = [f.get(k, 0.0) for f in feats]
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
            try:
                mean_feats[k] = float(np.mean(vals))
            except Exception:
                pass
    vec_full = np.array([mean_feats.get(nm, 0.0) for nm in ALL_FNAMES], dtype=np.float64)
    return vec_full[F3_COL]


def project_48(text):
    v = featurize_103(text)
    if v is None: return None
    v_std = (v - SCALER_MEAN) / SCALER_SCALE
    return (v_std - PCA_MEAN) @ PCA_COMP.T


# ────────── Load data ──────────
print(f"[1] scanning reviews for {N_USERS}-user cohort...")
t0 = time.time()
user_sents = collections.defaultdict(list)
n_scanned = 0
with gzip.open(f"{REPO}/data/Baby_Products_2023.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        uid = r.get("reviewerID") or r.get("user_id")
        if uid in set(COHORT) and r.get("text"):
            user_sents[uid].extend(split_sents(r["text"]))
        n_scanned += 1
        if n_scanned % 5_000_000 == 0:
            print(f"  {n_scanned/1e6:.1f}M scanned, t={time.time()-t0:.0f}s")
print(f"  done: scanned {n_scanned}, t={time.time()-t0:.0f}s")

for uid in user_sents:
    seen = set(); uniq = []
    for s in user_sents[uid]:
        k = s.strip().lower()
        if k not in seen:
            seen.add(k); uniq.append(s)
    user_sents[uid] = uniq


# ────────── Build train + test sets + 103d, 48d projections ──────────
print(f"\n[2] featurize train/test to 103d...")
train_data = []
test_data = []
for ui, uid in enumerate(COHORT):
    for s in user_sents.get(uid, []):
        v = featurize_103(s)
        if v is None: continue
        if is_train_sentence(s):
            train_data.append((ui, v))
        else:
            test_data.append((ui, v))
print(f"  n_train={len(train_data)}, n_test={len(test_data)}")

X_train = torch.tensor(np.array([d[1] for d in train_data]), dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(np.array([d[0] for d in train_data]), dtype=torch.long, device=DEVICE)
X_test = torch.tensor(np.array([d[1] for d in test_data]), dtype=torch.float32, device=DEVICE)
y_test = torch.tensor(np.array([d[0] for d in test_data]), dtype=torch.long, device=DEVICE)

# Project train + test to PCA48 (for baseline audit)
print("[3] projecting train/test to PCA48...")
Z_train_48 = (X_train.cpu().numpy() - SCALER_MEAN) / SCALER_SCALE
Z_train_48 = (Z_train_48 - PCA_MEAN) @ PCA_COMP.T
Z_test_48 = (X_test.cpu().numpy() - SCALER_MEAN) / SCALER_SCALE
Z_test_48 = (Z_test_48 - PCA_MEAN) @ PCA_COMP.T
y_train_np = y_train.cpu().numpy()
y_test_np = y_test.cpu().numpy()
print(f"  Z_train_48: {Z_train_48.shape}, Z_test_48: {Z_test_48.shape}")


# ────────── Model ──────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, D_HID[0]),
            nn.ReLU(),
            nn.Linear(D_HID[0], D_HID[1]),
            nn.ReLU(),
            nn.Linear(D_HID[1], D_OUT),
        )

    def forward(self, x):
        return self.net(x)


class UserGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(N_USERS, D_OUT))
        self.log_sigma = nn.Parameter(torch.zeros(N_USERS, D_OUT))

    @property
    def sigma(self):
        s = torch.exp(self.log_sigma)
        return torch.clamp(s, SIGMA_MIN, SIGMA_MAX)


encoder = Encoder().to(DEVICE)
gaussian = UserGaussian().to(DEVICE)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(gaussian.parameters()), lr=LR)


def train_epoch():
    encoder.train(); gaussian.train()
    optimizer.zero_grad()
    h = encoder(X_train)
    diff = h.unsqueeze(1) - gaussian.mu.unsqueeze(0)
    sig = gaussian.sigma.unsqueeze(0)
    energy = 0.5 * ((diff ** 2) / (sig ** 2 + 1e-6)).sum(dim=-1)
    log_p = -energy / TAU
    loss_id = F.cross_entropy(log_p, y_train)
    loss_sigma = (gaussian.log_sigma ** 2).sum()
    loss = loss_id + LAMBDA_SIGMA * loss_sigma
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        gaussian.log_sigma.data.clamp_(min=np.log(SIGMA_MIN), max=np.log(SIGMA_MAX))


print(f"\n[4] training ({EPOCHS} epochs)...")
t_train = time.time()
for ep in range(EPOCHS):
    train_epoch()
print(f"  trained in {time.time()-t_train:.0f}s")


# ────────── Extract learned (μ, σ) and embeddings ──────────
encoder.eval(); gaussian.eval()
with torch.no_grad():
    learned_mu = gaussian.mu.detach().cpu().numpy()                    # (12, 32)
    learned_sigma = gaussian.sigma.detach().cpu().numpy()              # (12, 32)
    H_train = encoder(X_train).detach().cpu().numpy()                  # (n_train, 32)
    H_test = encoder(X_test).detach().cpu().numpy()                    # (n_test, 32)


# ────────── PCA48 baseline (μ, σ) from train ──────────
print(f"\n[5] building PCA48 baseline Gaussians from train...")
pca48_mu = np.zeros((N_USERS, 48))
pca48_sigma_diag = np.zeros((N_USERS, 48))
for ui in range(N_USERS):
    mask = (y_train_np == ui)
    if mask.sum() > 0:
        pca48_mu[ui] = Z_train_48[mask].mean(axis=0)
        pca48_sigma_diag[ui] = Z_train_48[mask].var(axis=0) + 1e-3


# ════════════════════════════════════════════════════════════════════
# Audit 1: μ separation
# ════════════════════════════════════════════════════════════════════
print(f"\n" + "="*80)
print(f"AUDIT 1: μ separation (||μ_u - μ_v|| pairwise)")
print("="*80)


def pairwise_mu_dist(mu):
    """Return upper-triangular pairwise distances."""
    n = mu.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(mu[i] - mu[j])))
    return np.array(dists)


learned_mu_dists = pairwise_mu_dist(learned_mu)
pca48_mu_dists = pairwise_mu_dist(pca48_mu)

print(f"  Learned (32d)  : n={len(learned_mu_dists)}  "
      f"min={learned_mu_dists.min():.3f}  med={np.median(learned_mu_dists):.3f}  "
      f"mean={learned_mu_dists.mean():.3f}  max={learned_mu_dists.max():.3f}")
print(f"  PCA48  (48d)   : n={len(pca48_mu_dists)}  "
      f"min={pca48_mu_dists.min():.3f}  med={np.median(pca48_mu_dists):.3f}  "
      f"mean={pca48_mu_dists.mean():.3f}  max={pca48_mu_dists.max():.3f}")
print(f"  → mean Δ: {learned_mu_dists.mean() - pca48_mu_dists.mean():+.3f}  "
      f"(learned vs PCA48)")


# ════════════════════════════════════════════════════════════════════
# Audit 2: Bhattacharyya distance (closed-form diagonal)
# ════════════════════════════════════════════════════════════════════
print(f"\n" + "="*80)
print(f"AUDIT 2: Bhattacharyya distance (diagonal Gaussian pair)")
print("="*80)


def bhatt_diag(mu1, sig1, mu2, sig2):
    """Diagonal closed-form Bhattacharyya distance.
    sig1, sig2: variance vectors (σ²)
    """
    var1 = sig1 ** 2
    var2 = sig2 ** 2
    var_avg = (var1 + var2) / 2
    mu_diff = mu1 - mu2
    # First term: (1/8) μ^T Σ^-1 μ = (1/4) Σ μ_i² / (var1_i + var2_i)
    term1 = 0.25 * np.sum(mu_diff ** 2 / var_avg)
    # Second term: (1/2) log(det Σ_avg / sqrt(det Σ1 det Σ2))
    #   = (1/2) Σ [log((var1+var2)/2) - 0.5 log(var1) - 0.5 log(var2)]
    #   = (1/4) Σ [log((var1+var2)/2) - log(sqrt(var1 var2))]
    #   = (1/4) Σ log((var1+var2) / (2 sqrt(var1 var2)))
    ratio = var_avg / np.sqrt(var1 * var2 + 1e-12)
    term2 = 0.25 * np.sum(np.log(ratio + 1e-12))
    return term1 + term2


def pairwise_bhatt(mu, sig):
    n = mu.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(bhatt_diag(mu[i], sig[i], mu[j], sig[j]))
    return np.array(out)


learned_bd = pairwise_bhatt(learned_mu, learned_sigma)
pca48_bd = pairwise_bhatt(pca48_mu, np.sqrt(pca48_sigma_diag))

print(f"  Learned (32d): n={len(learned_bd)}  "
      f"min={learned_bd.min():.3f}  med={np.median(learned_bd):.3f}  "
      f"mean={learned_bd.mean():.3f}  max={learned_bd.max():.3f}")
print(f"  PCA48  (48d): n={len(pca48_bd)}  "
      f"min={pca48_bd.min():.3f}  med={np.median(pca48_bd):.3f}  "
      f"mean={pca48_bd.mean():.3f}  max={pca48_bd.max():.3f}")
print(f"  → mean Δ: {learned_bd.mean() - pca48_bd.mean():+.3f}  "
      f"(learned vs PCA48; positive = learned more separated)")


# ════════════════════════════════════════════════════════════════════
# Audit 3: Within-user vs between-user distance on held-out real
# ════════════════════════════════════════════════════════════════════
print(f"\n" + "="*80)
print(f"AUDIT 3: M_G = D_nearest_other - D_self on held-out real sentences")
print("="*80)


def energy_diag(h, mu, sig):
    """Mahalanobis² energy (diagonal)."""
    diff = h - mu
    return 0.5 * np.sum((diff ** 2) / (sig ** 2 + 1e-6), axis=-1)


# learned energies on test set
learned_energy_test = np.zeros((len(y_test_np), N_USERS))
for ui in range(N_USERS):
    learned_energy_test[:, ui] = energy_diag(H_test, learned_mu[ui], learned_sigma[ui])

# PCA48 energies on test set
pca48_energy_test = np.zeros((len(y_test_np), N_USERS))
for ui in range(N_USERS):
    pca48_energy_test[:, ui] = energy_diag(Z_test_48, pca48_mu[ui], np.sqrt(pca48_sigma_diag[ui]))

# For each test sentence, compute D_self and D_nearest_other
def audit_M_G(energies, y_true):
    n = len(y_true)
    d_self = np.zeros(n)
    d_other = np.zeros(n)
    other_uid = np.zeros(n, dtype=int)
    for i in range(n):
        ui = y_true[i]
        d_self[i] = energies[i, ui]
        # mask out self
        e = energies[i].copy()
        e[ui] = np.inf
        d_other[i] = e.min()
        other_uid[i] = int(np.argmin(e))
    M_G = d_other - d_self  # >0 means self closer than any other
    return M_G, d_self, d_other, other_uid


M_G_learned, d_self_l, d_other_l, other_l = audit_M_G(learned_energy_test, y_test_np)
M_G_pca48, d_self_p, d_other_p, other_p = audit_M_G(pca48_energy_test, y_test_np)

p_pos_learned = float((M_G_learned > 0).mean())
p_pos_pca48 = float((M_G_pca48 > 0).mean())
med_learned = float(np.median(M_G_learned))
med_pca48 = float(np.median(M_G_pca48))

print(f"  Learned 32d: P(M_G>0) = {p_pos_learned*100:.1f}%  median(M_G) = {med_learned:+.3f}")
print(f"  PCA48  48d: P(M_G>0) = {p_pos_pca48*100:.1f}%  median(M_G) = {med_pca48:+.3f}")
print(f"  → learned Δ P(M_G>0): {(p_pos_learned - p_pos_pca48)*100:+.1f}pp  "
      f"Δ median: {med_learned - med_pca48:+.3f}")
print(f"\n  Within-user mean D_self   : learned={d_self_l.mean():.3f}  PCA48={d_self_p.mean():.3f}")
print(f"  Between-user mean D_other : learned={d_other_l.mean():.3f}  PCA48={d_other_p.mean():.3f}")


# ════════════════════════════════════════════════════════════════════
# Audit 4: per-user P(M_G > 0) for all 12 users
# ════════════════════════════════════════════════════════════════════
print(f"\n" + "="*80)
print(f"AUDIT 4: per-user P_u(M_G > 0) and median(M_G_u)")
print("="*80)
print(f"{'uid':<32} {'n_test':>7} {'learned_P':>10} {'learned_med':>13}  {'pca48_P':>8} {'pca48_med':>11}")
print("-"*100)
per_user_audit = {}
for ui, uid in enumerate(COHORT):
    mask = (y_test_np == ui)
    n_test = int(mask.sum())
    if n_test == 0:
        print(f"{uid[:30]:<32} {0:>7}     —           —               —           —")
        per_user_audit[uid] = {"n_test": 0, "learned_P": None, "learned_med": None,
                               "pca48_P": None, "pca48_med": None}
        continue
    p_l = float((M_G_learned[mask] > 0).mean())
    p_p = float((M_G_pca48[mask] > 0).mean())
    m_l = float(np.median(M_G_learned[mask]))
    m_p = float(np.median(M_G_pca48[mask]))
    print(f"{uid[:30]:<32} {n_test:>7} {p_l*100:>9.1f}%  {m_l:>+12.3f}  "
          f"{p_p*100:>7.1f}%  {m_p:>+10.3f}")
    per_user_audit[uid] = {
        "n_test": n_test,
        "learned_P_M_G_pos": p_l,
        "learned_med_M_G": m_l,
        "pca48_P_M_G_pos": p_p,
        "pca48_med_M_G": m_p,
    }


# ════════════════════════════════════════════════════════════════════
# Audit 5: σ_bias recheck
# ════════════════════════════════════════════════════════════════════
print(f"\n" + "="*80)
print(f"AUDIT 5: σ_bias recheck")
print("="*80)
from scipy.stats import pearsonr as pr

sigma_means = learned_sigma.mean(axis=1)
# rank1 frequency = (M_G > 0) per user on test (which user got "rank1 = own" by margin test)
# Actually, M_G > 0 means nearest_other > self, i.e., user is rank1 for their own sentences.
# So per-user frequency = per-user P(M_G > 0).
rank1_per_user = np.array([per_user_audit[uid]["learned_P_M_G_pos"] or 0.0
                           for uid in COHORT])
rho, pval = pr(sigma_means, rank1_per_user)
print(f"  ρ(σ_mean, per-user P(M_G>0)): {rho:+.4f}  (p={pval:.4f}, target |ρ|<0.2)")
print(f"  σ_mean range: [{sigma_means.min():.3f}, {sigma_means.max():.3f}]  "
      f"median={np.median(sigma_means):.3f}")
# Also raw σ_mean per user
print(f"\n  σ_mean per user (32d, avg over 32 dims):")
for ui, uid in enumerate(COHORT):
    print(f"    {uid[:30]}: σ_mean={sigma_means[ui]:.4f}  P(M_G>0)={rank1_per_user[ui]*100:.1f}%")


# ════════════════════════════════════════════════════════════════════
# Verdict: did learned model actually separate users?
# ════════════════════════════════════════════════════════════════════
print(f"\n" + "="*80)
print(f"VERDICT")
print("="*80)
verdict_lines = []
# (1) μ more separated?
mu_check = learned_mu_dists.mean() > pca48_mu_dists.mean()
verdict_lines.append(f"(1) μ separation (mean ‖μ_u-μ_v‖): "
                     f"learned={learned_mu_dists.mean():.3f} vs PCA48={pca48_mu_dists.mean():.3f}  "
                     f"→ {'✓ learned MORE separated' if mu_check else '✗ learned NOT more separated'}")
# (2) BD more separated?
bd_check = learned_bd.mean() > pca48_bd.mean()
verdict_lines.append(f"(2) Bhattacharyya distance: "
                     f"learned={learned_bd.mean():.3f} vs PCA48={pca48_bd.mean():.3f}  "
                     f"→ {'✓ learned MORE separated' if bd_check else '✗ learned NOT more separated'}")
# (3) within-user lower?
ws_check = d_self_l.mean() < d_self_p.mean()
verdict_lines.append(f"(3) within-user distance: "
                     f"learned={d_self_l.mean():.3f} vs PCA48={d_self_p.mean():.3f}  "
                     f"→ {'✓ learned within-user CLOSER' if ws_check else '✗ learned within-user NOT closer'}")
# (4) between-user higher?
bs_check = d_other_l.mean() > d_other_p.mean()
verdict_lines.append(f"(4) between-user distance: "
                     f"learned={d_other_l.mean():.3f} vs PCA48={d_other_p.mean():.3f}  "
                     f"→ {'✓ learned between-user FARTHER' if bs_check else '✗ learned between-user NOT farther'}")
# (5) σ_bias
sb_check = abs(rho) < 0.2
verdict_lines.append(f"(5) σ_bias |ρ|: {abs(rho):.4f}  "
                     f"→ {'✓' if sb_check else '✗'} (target < 0.20)")

for line in verdict_lines:
    print(f"  {line}")
n_pass = sum(1 for line in verdict_lines if "✓" in line)
print(f"\n  → {n_pass}/5 audit checks passed")
if n_pass >= 4:
    print(f"  → VERDICT: Learned Syntax Gaussian DOES separate users geometrically.")
elif n_pass >= 3:
    print(f"  → VERDICT: Learned model partially separates; main gain from one direction.")
else:
    print(f"  → VERDICT: Learned model did NOT separate users; Rank@1 lift was statistical.")


# ════════════════════════════════════════════════════════════════════
# Save JSON
# ════════════════════════════════════════════════════════════════════
out_path = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/phase8a1_gaussian_separability_audit.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result = {
    "config": {
        "asin": ASIN,
        "n_users": N_USERS,
        "epochs": EPOCHS,
        "learned_dim": D_OUT,
        "pca48_dim": 48,
    },
    "data": {
        "n_train": len(train_data),
        "n_test": len(test_data),
    },
    "audit_1_mu_separation": {
        "learned_mean": float(learned_mu_dists.mean()),
        "learned_median": float(np.median(learned_mu_dists)),
        "pca48_mean": float(pca48_mu_dists.mean()),
        "pca48_median": float(np.median(pca48_mu_dists)),
        "delta_mean": float(learned_mu_dists.mean() - pca48_mu_dists.mean()),
        "check_passed": bool(mu_check),
    },
    "audit_2_bhattacharyya": {
        "learned_mean": float(learned_bd.mean()),
        "learned_median": float(np.median(learned_bd)),
        "pca48_mean": float(pca48_bd.mean()),
        "pca48_median": float(np.median(pca48_bd)),
        "delta_mean": float(learned_bd.mean() - pca48_bd.mean()),
        "check_passed": bool(bd_check),
    },
    "audit_3_within_between_distance": {
        "learned_D_self_mean": float(d_self_l.mean()),
        "learned_D_other_mean": float(d_other_l.mean()),
        "pca48_D_self_mean": float(d_self_p.mean()),
        "pca48_D_other_mean": float(d_other_p.mean()),
        "learned_P_M_G_pos": p_pos_learned,
        "pca48_P_M_G_pos": p_pos_pca48,
        "learned_med_M_G": med_learned,
        "pca48_med_M_G": med_pca48,
        "check_within_passed": bool(ws_check),
        "check_between_passed": bool(bs_check),
    },
    "audit_4_per_user": per_user_audit,
    "audit_5_sigma_bias": {
        "rho_sigma_rank1_freq": float(rho),
        "p_value": float(pval),
        "abs_rho": float(abs(rho)),
        "check_passed": bool(sb_check),
        "sigma_mean_per_user": {COHORT[ui]: float(sigma_means[ui]) for ui in range(N_USERS)},
    },
    "verdict": {
        "n_passed": n_pass,
        "verdict_text": "Learned Syntax Gaussian DOES separate users geometrically" if n_pass >= 4
                        else ("Learned model partially separates" if n_pass >= 3
                              else "Learned model did NOT separate users"),
        "checks": verdict_lines,
    },
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nsaved → {out_path}")
