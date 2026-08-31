"""Phase 8.A.2 — Gaussian-geometry-aware training.

Adds two new terms to the training loss:
  L_compact  = mean per-user Mahalanobis² distance to own μ
               (pull same-user sentences close to their μ)
  L_separate = mean over user pairs of max(0, margin - BD(G_u, G_v))
               (push different-user Gaussians apart)

Total: L = L_id + λ_1·L_compact + λ_2·L_separate

Reuses 8.A's:
  - F3 103d → encoder (103 → 128 → 64 → 32)
  - per-user trainable Gaussian (μ, σ ∈ ℝ³²)
  - 80/20 hash-based train/test split on B00A4B34IA, 12-user cohort
  - σ clamp [0.1, 2]
  - 80 epochs

Evaluates on 5-audit geometry + Rank@1, compares to 8.A baseline.
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
LR = 1e-3

# NEW: geometry-aware loss weights
LAMBDA_SIGMA = 0.01
LAMBDA_COMPACT = float(os.environ.get("PHASE8A2_LAMBDA_COMPACT", "0.001"))
LAMBDA_SEPARATE = float(os.environ.get("PHASE8A2_LAMBDA_SEPARATE", "0.05"))
BD_MARGIN = float(os.environ.get("PHASE8A2_BD_MARGIN", "1.0"))

EPOCHS = int(os.environ.get("PHASE8A2_EPOCHS", "80"))
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


def bhatt_diag_torch(mu1, sig1, mu2, sig2):
    """Diagonal Bhattacharyya distance (torch)."""
    var1 = sig1 ** 2 + 1e-6
    var2 = sig2 ** 2 + 1e-6
    var_avg = (var1 + var2) / 2
    mu_diff = mu1 - mu2
    term1 = 0.25 * torch.sum(mu_diff ** 2 / var_avg)
    ratio = var_avg / torch.sqrt(var1 * var2 + 1e-12)
    term2 = 0.25 * torch.sum(torch.log(ratio + 1e-12))
    return term1 + term2


def compact_loss_torch(H, mu, sig, y, n_users):
    """Mean per-user Mahalanobis² distance to own μ."""
    total = H.new_zeros(())
    n_with_data = 0
    for u in range(n_users):
        mask = (y == u)
        if not mask.any(): continue
        diff = H[mask] - mu[u]                  # (n_u, D)
        energy = 0.5 * ((diff ** 2) / (sig[u] ** 2 + 1e-6)).sum(dim=-1)  # (n_u,)
        total = total + energy.mean()
        n_with_data += 1
    if n_with_data == 0:
        return total
    return total / n_with_data


def separation_loss_torch(mu, sig, n_users, margin):
    """Mean hinge max(0, margin - BD(G_u, G_v)) over all pairs."""
    total = mu.new_zeros(())
    n_pairs = 0
    for i in range(n_users):
        for j in range(i + 1, n_users):
            bd = bhatt_diag_torch(mu[i], sig[i], mu[j], sig[j])
            total = total + torch.clamp(margin - bd, min=0.0)
            n_pairs += 1
    if n_pairs == 0:
        return total
    return total / n_pairs


def train_epoch():
    encoder.train(); gaussian.train()
    optimizer.zero_grad()
    h = encoder(X_train)
    # L_id: contrastive softmax
    diff = h.unsqueeze(1) - gaussian.mu.unsqueeze(0)
    sig = gaussian.sigma.unsqueeze(0)
    energy = 0.5 * ((diff ** 2) / (sig ** 2 + 1e-6)).sum(dim=-1)
    log_p = -energy / TAU
    loss_id = F.cross_entropy(log_p, y_train)
    # L_σ regularization
    loss_sigma = (gaussian.log_sigma ** 2).sum()
    # L_compact
    loss_compact = compact_loss_torch(h, gaussian.mu, gaussian.sigma, y_train, N_USERS)
    # L_separate
    loss_separate = separation_loss_torch(gaussian.mu, gaussian.sigma, N_USERS, BD_MARGIN)
    # Total
    loss = loss_id + LAMBDA_SIGMA * loss_sigma \
                + LAMBDA_COMPACT * loss_compact \
                + LAMBDA_SEPARATE * loss_separate
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        gaussian.log_sigma.data.clamp_(min=np.log(SIGMA_MIN), max=np.log(SIGMA_MAX))
    return (float(loss_id), float(loss_sigma), float(loss_compact),
            float(loss_separate), float(loss))


print(f"\n[3] training ({EPOCHS} epochs, device={DEVICE})...")
print(f"  λ_compact={LAMBDA_COMPACT}  λ_separate={LAMBDA_SEPARATE}  margin={BD_MARGIN}")
t_train = time.time()
hist = []
for ep in range(EPOCHS):
    losses = train_epoch()
    hist.append({"epoch": ep + 1, **{k: v for k, v in zip(
        ["loss_id", "loss_sigma", "loss_compact", "loss_separate", "loss_total"], losses)}})
    if (ep + 1) % 10 == 0 or ep == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep+1:>3}/{EPOCHS}: id={losses[0]:.3f} σ={losses[1]:.3f} "
              f"compact={losses[2]:.3f} sep={losses[3]:.3f} total={losses[4]:.3f}")
print(f"  trained in {time.time()-t_train:.0f}s")


# ────────── Evaluate on held-out ──────────
print(f"\n[4] evaluating on held-out test set ({len(test_data)} sentences)...")
encoder.eval(); gaussian.eval()
with torch.no_grad():
    learned_mu = gaussian.mu.detach().cpu().numpy()
    learned_sigma = gaussian.sigma.detach().cpu().numpy()
    H_train = encoder(X_train).detach().cpu().numpy()
    H_test = encoder(X_test).detach().cpu().numpy()
y_train_np = y_train.cpu().numpy()
y_test_np = y_test.cpu().numpy()


# ────────── 5-audit geometry metrics ──────────
def pairwise_mu_dist(mu):
    n = mu.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(mu[i] - mu[j])))
    return np.array(dists)


def bhatt_diag_np(mu1, sig1, mu2, sig2):
    var1 = sig1 ** 2
    var2 = sig2 ** 2
    var_avg = (var1 + var2) / 2
    mu_diff = mu1 - mu2
    term1 = 0.25 * np.sum(mu_diff ** 2 / var_avg)
    ratio = var_avg / np.sqrt(var1 * var2 + 1e-12)
    term2 = 0.25 * np.sum(np.log(ratio + 1e-12))
    return term1 + term2


def pairwise_bhatt(mu, sig):
    n = mu.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(bhatt_diag_np(mu[i], sig[i], mu[j], sig[j]))
    return np.array(out)


def energy_diag(h, mu, sig):
    diff = h - mu
    return 0.5 * np.sum((diff ** 2) / (sig ** 2 + 1e-6), axis=-1)


def compute_test_metrics(energies, y_true):
    """Per-user and overall P(M_G>0) and median M_G."""
    n = len(y_true)
    d_self = np.zeros(n)
    d_other = np.zeros(n)
    for i in range(n):
        ui = y_true[i]
        d_self[i] = energies[i, ui]
        e = energies[i].copy()
        e[ui] = np.inf
        d_other[i] = e.min()
    M_G = d_other - d_self
    return M_G, d_self, d_other


# Learned energies on test
learned_energy_test = np.zeros((len(y_test_np), N_USERS))
for ui in range(N_USERS):
    learned_energy_test[:, ui] = energy_diag(H_test, learned_mu[ui], learned_sigma[ui])

# Train energies for compact loss evaluation
learned_energy_train = np.zeros((len(y_train_np), N_USERS))
for ui in range(N_USERS):
    learned_energy_train[:, ui] = energy_diag(H_train, learned_mu[ui], learned_sigma[ui])

# ────────── Audit 1: μ separation ──────────
learned_mu_dists = pairwise_mu_dist(learned_mu)

# ────────── Audit 2: Bhattacharyya distance ──────────
learned_bd = pairwise_bhatt(learned_mu, learned_sigma)

# ────────── Audit 3: within-user vs between-user ──────────
M_G_learned, d_self_l, d_other_l = compute_test_metrics(learned_energy_test, y_test_np)
M_G_train, d_self_train, d_other_train = compute_test_metrics(learned_energy_train, y_train_np)

# ────────── Audit 4: per-user P(M_G > 0) ──────────
per_user_audit = {}
for ui, uid in enumerate(COHORT):
    mask = (y_test_np == ui)
    n_test = int(mask.sum())
    if n_test == 0:
        per_user_audit[uid] = {"n_test": 0, "P_M_G_pos": None, "med_M_G": None}
        continue
    p = float((M_G_learned[mask] > 0).mean())
    m = float(np.median(M_G_learned[mask]))
    per_user_audit[uid] = {"n_test": n_test, "P_M_G_pos": p, "med_M_G": m}

# ────────── Audit 5: σ_bias ──────────
from scipy.stats import pearsonr as pr
sigma_means = learned_sigma.mean(axis=1)
rank1_per_user = np.array([per_user_audit[uid]["P_M_G_pos"] or 0.0 for uid in COHORT])
rho, pval = pr(sigma_means, rank1_per_user)

# ────────── Rank@1 on test ──────────
pred = learned_energy_test.argmin(axis=1)
rank1_overall = float((pred == y_test_np).mean())
per_user_rank1 = {}
for ui, uid in enumerate(COHORT):
    mask = (y_test_np == ui)
    if mask.sum() == 0:
        per_user_rank1[uid] = None
        continue
    per_user_rank1[uid] = float((pred[mask] == ui).mean())

# ────────── Print ──────────
print(f"\n=== 8.A.2 RESULTS ===")
print(f"\n[Rank@1]")
print(f"  Held-out overall Rank@1 : {rank1_overall*100:.1f}%")
print(f"  8.A baseline: 29.9%")
print(f"\n[Audit 1] μ pairwise distance")
print(f"  mean={learned_mu_dists.mean():.4f}  median={np.median(learned_mu_dists):.4f}  "
      f"min={learned_mu_dists.min():.4f}  max={learned_mu_dists.max():.4f}")
print(f"  8.A baseline mean=0.188")
print(f"\n[Audit 2] Bhattacharyya distance (within learned 32d space)")
print(f"  mean={learned_bd.mean():.4f}  median={np.median(learned_bd):.4f}  "
      f"min={learned_bd.min():.4f}  max={learned_bd.max():.4f}")
print(f"  8.A baseline mean=0.017  (NB: within-space comparison, not cross-space)")
print(f"\n[Audit 3] Within-user vs Between-user")
print(f"  D_self mean: {d_self_l.mean():.4f}    D_other mean: {d_other_l.mean():.4f}    "
      f"margin: {(d_other_l.mean() - d_self_l.mean()):+.4f}")
print(f"  8.A baseline: D_self=162.186, D_other=161.431, margin=-0.755")
print(f"  Median M_G: {np.median(M_G_learned):+.4f}  P(M_G>0): {(M_G_learned>0).mean()*100:.1f}%")
print(f"  8.A baseline: median=-0.777, P(M_G>0)=29.9%")
print(f"\n[Audit 4] Per-user P(M_G>0)")
print(f"{'uid':<32} {'n_test':>7} {'P(M_G>0)':>10} {'med(M_G)':>10}")
for ui, uid in enumerate(COHORT):
    a = per_user_audit[uid]
    if a["n_test"] == 0:
        print(f"{uid[:30]:<32} {0:>7}     —          —")
        continue
    print(f"{uid[:30]:<32} {a['n_test']:>7} {a['P_M_G_pos']*100:>9.1f}% {a['med_M_G']:>+9.4f}")
print(f"\n[Audit 5] σ_bias")
print(f"  ρ(σ_mean, P(M_G>0)): {rho:+.4f}  (p={pval:.4f}, target |ρ|<0.2)")
print(f"  σ_mean range: [{sigma_means.min():.4f}, {sigma_means.max():.4f}]  "
      f"median={np.median(sigma_means):.4f}")

# ────────── Save ──────────
out_path = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/phase8a2_geometry_aware_gaussian.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result = {
    "config": {
        "asin": ASIN,
        "n_users": N_USERS,
        "epochs": EPOCHS,
        "encoder_dims": [D_IN] + D_HID + [D_OUT],
        "lambda_compact": LAMBDA_COMPACT,
        "lambda_separate": LAMBDA_SEPARATE,
        "bd_margin": BD_MARGIN,
    },
    "data": {"n_train": len(train_data), "n_test": len(test_data)},
    "rank1_overall": rank1_overall,
    "rank1_per_user": per_user_rank1,
    "audit_1_mu_separation": {
        "mean": float(learned_mu_dists.mean()),
        "median": float(np.median(learned_mu_dists)),
        "min": float(learned_mu_dists.min()),
        "max": float(learned_mu_dists.max()),
    },
    "audit_2_bhattacharyya": {
        "mean": float(learned_bd.mean()),
        "median": float(np.median(learned_bd)),
        "min": float(learned_bd.min()),
        "max": float(learned_bd.max()),
    },
    "audit_3_within_between": {
        "test_D_self_mean": float(d_self_l.mean()),
        "test_D_other_mean": float(d_other_l.mean()),
        "test_margin": float(d_other_l.mean() - d_self_l.mean()),
        "test_P_M_G_pos": float((M_G_learned > 0).mean()),
        "test_median_M_G": float(np.median(M_G_learned)),
        "train_D_self_mean": float(d_self_train.mean()),
        "train_D_other_mean": float(d_other_train.mean()),
    },
    "audit_4_per_user": per_user_audit,
    "audit_5_sigma_bias": {
        "rho": float(rho),
        "p_value": float(pval),
        "sigma_mean_per_user": {COHORT[ui]: float(sigma_means[ui]) for ui in range(N_USERS)},
    },
    "training_history": hist,
    "comparison_to_8a": {
        "rank1_8a2": rank1_overall,
        "rank1_8a": 0.299,
        "delta_rank1_pp": (rank1_overall - 0.299) * 100,
        "mu_dist_8a2_mean": float(learned_mu_dists.mean()),
        "mu_dist_8a_mean": 0.188,
        "bd_8a2_mean": float(learned_bd.mean()),
        "bd_8a_mean": 0.017,
        "within_user_8a2": float(d_self_l.mean()),
        "within_user_8a": 162.186,
        "margin_8a2": float(d_other_l.mean() - d_self_l.mean()),
        "margin_8a": -0.755,
        "rho_sigma_8a2": float(abs(rho)),
        "rho_sigma_8a": 0.043,
    },
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nsaved → {out_path}")

# ────────── Verdict ──────────
print(f"\n=== COMPARISON: 8.A.2 vs 8.A ===")
print(f"  Rank@1:        8.A=29.9%   8.A.2={rank1_overall*100:.1f}%   "
      f"Δ={(rank1_overall-0.299)*100:+.1f}pp")
print(f"  μ dist mean:   8.A=0.188   8.A.2={learned_mu_dists.mean():.4f}   "
      f"Δ={(learned_mu_dists.mean()-0.188):+.4f}")
print(f"  BD mean:       8.A=0.017   8.A.2={learned_bd.mean():.4f}   "
      f"Δ={(learned_bd.mean()-0.017):+.4f}")
print(f"  within-user:   8.A=162.186 8.A.2={d_self_l.mean():.4f}  "
      f"Δ={(d_self_l.mean()-162.186):+.4f}")
print(f"  margin:        8.A=-0.755  8.A.2={d_other_l.mean()-d_self_l.mean():+.4f}  "
      f"Δ={(d_other_l.mean()-d_self_l.mean()+0.755):+.4f}")
print(f"  |ρ(σ,rank1)|:  8.A=0.043   8.A.2={abs(rho):.4f}  "
      f"{'✓' if abs(rho) < 0.2 else '✗'}")
