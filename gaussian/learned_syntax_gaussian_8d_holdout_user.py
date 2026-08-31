"""Phase 8.D — Held-out user zero-shot generalization test.

Pipeline:
  1. Load 80/20 user split (74 train users / 26 held-out users).
  2. Train encoder + per-user Gaussian on 74 train users only.
     Each train user: 80% sentences train / 20% test (within-user hash).
  3. Freeze encoder. For each held-out user:
       a. Project all their 8-60 word sentences through frozen encoder → 32d
       b. Split 80/20 within-user (deterministic hash)
       c. Fit Gaussian (μ=mean, σ=var+ε) from 80% sentences
       d. Compute Rank@1 on 20% sentences against ALL 100 user Gaussians
          (74 train learned + 26 held-out zero-shot fit)

GO criterion: held-out user Rank@1 ≥ 29.9% (8.A 12-user baseline).

This is the TRUE zero-shot test:
  - Encoder has NEVER seen any sentence from the 26 held-out users
  - Their Gaussian is fit fresh from their own 80% sentences
  - The model must generalize the 32d representation to brand-new users
"""
import collections, hashlib, json, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
sys.path.insert(0, f"{REPO}/common")
from syntactic_features import per_sentence_features_v2 as psf
import spacy

from _user_sentence_cache import load_user_sents

# ────────── Config ──────────
SPLIT_PATH = f"{REPO}/result/gaussian/phase8d_split_80_20.json"
COHORT_PATH = f"{REPO}/result/gaussian/phase8c_cohort_100user.json"
SEED_PREFIX = "phase8d_v1|"
TRAIN_FRAC = 0.80  # within-user sentence split (same for train/test users)

# Same hyperparams as 8.A.3 / 8.C.3
D_IN = 103
D_HID = [128, 64]
D_OUT = 32
SIGMA_MIN = 0.1
SIGMA_MAX = 2.0
TAU = 1.0
LR = 1e-3
LAMBDA_SIGMA = 0.01
LAMBDA_COMPACT = float(os.environ.get("PHASE8A2_LAMBDA_COMPACT", "0.0005"))
LAMBDA_SEPARATE = float(os.environ.get("PHASE8A2_LAMBDA_SEPARATE", "0.5"))
BD_MARGIN = float(os.environ.get("PHASE8A2_BD_MARGIN", "1.0"))
LAMBDA_RANK = float(os.environ.get("PHASE8A3_LAMBDA_RANK", "0.1"))
RANK_MARGIN = float(os.environ.get("PHASE8A3_RANK_MARGIN", "1.0"))
EPOCHS = int(os.environ.get("PHASE8A2_EPOCHS", "80"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])


def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def is_train_sentence(text):
    h = int(hashlib.sha1((SEED_PREFIX + text.strip().lower()).encode()).hexdigest(), 16)
    return (h % 100) < int(TRAIN_FRAC * 100)


# ────────── Load F3 features (same scaler/PCA as 8.A/8.C) ──────────
gdoc = np.load("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_shared_scaler.npz",
               allow_pickle=True)
ALL_FNAMES = list(gdoc["feature_names_ordered"])
FNAMES_F3 = list(gdoc["fnames_f3"])
assert len(FNAMES_F3) == 103
SCALER_MEAN = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
SCALER_SCALE = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
PCA_COMP = np.asarray(gdoc["pca_components"], dtype=np.float64)
PCA_MEAN = np.asarray(gdoc["pca_mean"], dtype=np.float64)
gdoc.close()
F3_COL = [ALL_FNAMES.index(nm) for nm in FNAMES_F3]


def featurize_103(text):
    sents = [s for s in split_sents(text) if 8 <= len(s.split()) <= 60]
    if not sents:
        return None
    doc = next(NLP.pipe([sents[0]], batch_size=1, n_process=1))
    sds = list(doc.sents)
    if not sds:
        return None
    feats = []
    for s in sds:
        f = psf(s)
        if f is not None:
            feats.append(f)
    if not feats:
        return None
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


# ────────── Load split ──────────
print("[1] loading 80/20 user split...")
with open(SPLIT_PATH) as f:
    split_data = json.load(f)
TRAIN_USERS = [u["uid"] for u in split_data["train_users"]]
TEST_USERS = [u["uid"] for u in split_data["test_users"]]
ALL_USERS = TRAIN_USERS + TEST_USERS
print(f"  train={len(TRAIN_USERS)}, test(held-out)={len(TEST_USERS)}")

# Load cached sentences for ALL users
print(f"\n[2] loading cached sentences for {len(ALL_USERS)} users...")
t0 = time.time()
sents_all = load_user_sents(ALL_USERS, REPO)
print(f"  loaded in {time.time()-t0:.1f}s ({sum(len(v) for v in sents_all.values())} sents)")


# ────────── Featurize ──────────
print(f"\n[3] featurize train/test sentences to 103d...")
train_data = []   # (uid_idx, 103d vector)
test_data_train_users = []    # within-user test split, for 80% metric
heldout_full = {}  # all held-out sents, for zero-shot fit + test

for ui, uid in enumerate(ALL_USERS):
    for s in sents_all.get(uid, []):
        v = featurize_103(s)
        if v is None:
            continue
        if uid in TRAIN_USERS:
            if is_train_sentence(s):
                train_data.append((ui, v))
            else:
                test_data_train_users.append((ui, v))
        else:  # held-out user — keep ALL sentences
            heldout_full.setdefault(uid, []).append((s, v))

print(f"  n_train (80 train users, 80% sents): {len(train_data)}")
print(f"  n_test (80 train users, 20% sents): {len(test_data_train_users)}")
print(f"  n_heldout users: {len(heldout_full)}, total sents: "
      f"{sum(len(v) for v in heldout_full.values())}")


# ────────── Convert to tensors ──────────
def to_tensors(rows):
    if not rows:
        return None, None
    X = torch.tensor(np.array([r[1] for r in rows]), dtype=torch.float32, device=DEVICE)
    y = torch.tensor(np.array([r[0] for r in rows]), dtype=torch.long, device=DEVICE)
    return X, y

X_train, y_train = to_tensors(train_data)
X_test_train_users, y_test_train_users = to_tensors(test_data_train_users)


# ────────── Model ──────────
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, D_HID[0]), nn.ReLU(),
            nn.Linear(D_HID[0], D_HID[1]), nn.ReLU(),
            nn.Linear(D_HID[1], D_OUT),
        )

    def forward(self, x):
        return self.net(x)


class UserGaussian(nn.Module):
    def __init__(self, n_users):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(n_users, D_OUT))
        self.log_sigma = nn.Parameter(torch.zeros(n_users, D_OUT))

    @property
    def sigma(self):
        s = torch.exp(self.log_sigma)
        return torch.clamp(s, SIGMA_MIN, SIGMA_MAX)


N_TRAIN = len(TRAIN_USERS)
encoder = Encoder().to(DEVICE)
gaussian = UserGaussian(N_TRAIN).to(DEVICE)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(gaussian.parameters()), lr=LR)


def bhatt_diag_torch(mu1, sig1, mu2, sig2):
    var1 = sig1 ** 2 + 1e-6
    var2 = sig2 ** 2 + 1e-6
    var_avg = (var1 + var2) / 2
    mu_diff = mu1 - mu2
    term1 = 0.25 * torch.sum(mu_diff ** 2 / var_avg)
    ratio = var_avg / torch.sqrt(var1 * var2 + 1e-12)
    term2 = 0.25 * torch.sum(torch.log(ratio + 1e-12))
    return term1 + term2


def compact_loss_torch(H, mu, sig, y, n_users):
    total = H.new_zeros(())
    n_with = 0
    for u in range(n_users):
        mask = (y == u)
        if not mask.any(): continue
        diff = H[mask] - mu[u]
        energy = 0.5 * ((diff ** 2) / (sig[u] ** 2 + 1e-6)).sum(dim=-1)
        total = total + energy.mean()
        n_with += 1
    return total / max(n_with, 1)


def separation_loss_torch(mu, sig, n_users, margin):
    total = mu.new_zeros(())
    n_pairs = 0
    for i in range(n_users):
        for j in range(i + 1, n_users):
            bd = bhatt_diag_torch(mu[i], sig[i], mu[j], sig[j])
            total = total + torch.clamp(margin - bd, min=0.0)
            n_pairs += 1
    return total / max(n_pairs, 1)


def ranking_loss_torch(H, mu, sig, y, n_users, margin):
    diff = H.unsqueeze(1) - mu.unsqueeze(0)
    energy = 0.5 * ((diff ** 2) / (sig.unsqueeze(0) ** 2 + 1e-6)).sum(dim=-1)
    diag_self = energy.gather(1, y.unsqueeze(1)).squeeze(1)
    energy_masked = energy.clone()
    energy_masked.scatter_(1, y.unsqueeze(1), float("inf"))
    nearest_other, _ = energy_masked.min(dim=1)
    return torch.clamp(diag_self + margin - nearest_other, min=0.0).mean()


# ────────── Train encoder + Gaussian on 80 train users ──────────
print(f"\n[4] training on {N_TRAIN} train users ({EPOCHS} epochs)...")
t_train = time.time()
hist = []
for ep in range(EPOCHS):
    encoder.train(); gaussian.train()
    optimizer.zero_grad()
    h = encoder(X_train)
    diff = h.unsqueeze(1) - gaussian.mu.unsqueeze(0)
    sig = gaussian.sigma.unsqueeze(0)
    energy = 0.5 * ((diff ** 2) / (sig ** 2 + 1e-6)).sum(dim=-1)
    log_p = -energy / TAU
    loss_id = F.cross_entropy(log_p, y_train)
    loss_sigma = (gaussian.log_sigma ** 2).sum()
    loss_compact = compact_loss_torch(h, gaussian.mu, gaussian.sigma, y_train, N_TRAIN)
    loss_separate = separation_loss_torch(gaussian.mu, gaussian.sigma, N_TRAIN, BD_MARGIN)
    loss_rank = ranking_loss_torch(h, gaussian.mu, gaussian.sigma, y_train, N_TRAIN, RANK_MARGIN)
    loss = loss_id + LAMBDA_SIGMA * loss_sigma \
                + LAMBDA_COMPACT * loss_compact \
                + LAMBDA_SEPARATE * loss_separate \
                + LAMBDA_RANK * loss_rank
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        gaussian.log_sigma.data.clamp_(min=np.log(SIGMA_MIN), max=np.log(SIGMA_MAX))
    hist.append({"epoch": ep + 1, "loss_total": float(loss)})
    if (ep + 1) % 10 == 0 or ep == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep+1:>3}/{EPOCHS}: total={float(loss):.3f}")
print(f"  trained in {time.time()-t_train:.0f}s")


# ────────── Evaluate on train users (sanity check) ──────────
print(f"\n[5] eval on 80 train users (within-user test split)...")
encoder.eval(); gaussian.eval()
with torch.no_grad():
    learned_mu_train = gaussian.mu.detach().cpu().numpy()
    learned_sigma_train = gaussian.sigma.detach().cpu().numpy()

def energy(h, mu, sig):
    diff = h - mu
    return 0.5 * ((diff ** 2) / (sig ** 2 + 1e-6)).sum(axis=-1)

if X_test_train_users is not None:
    with torch.no_grad():
        H_test = encoder(X_test_train_users).detach().cpu().numpy()
    energies = np.zeros((H_test.shape[0], N_TRAIN))
    for ui in range(N_TRAIN):
        energies[:, ui] = energy(H_test, learned_mu_train[ui], learned_sigma_train[ui])
    pred = energies.argmin(axis=1)
    y_test_np = y_test_train_users.cpu().numpy()
    train_rank1 = float((pred == y_test_np).mean())
    print(f"  Train users held-out Rank@1: {train_rank1*100:.1f}%")
else:
    train_rank1 = None
    print("  No within-user test sentences (small cohort).")


# ────────── Zero-shot: fit Gaussian for each held-out user ──────────
print(f"\n[6] zero-shot Gaussian fit for {len(TEST_USERS)} held-out users...")
encoder.eval()
heldout_results = []
heldout_mu = np.zeros((len(TEST_USERS), D_OUT), dtype=np.float32)
heldout_sigma = np.zeros((len(TEST_USERS), D_OUT), dtype=np.float32)
heldout_test_sents = []   # list of (uid_idx, H_test_32d)
heldout_test_y = []

for hi, uid in enumerate(TEST_USERS):
    sents = heldout_full.get(uid, [])
    if not sents:
        heldout_mu[hi] = 0.0
        heldout_sigma[hi] = 1.0
        continue
    fit_h, test_h, fit_idx = [], [], []
    for k, (raw_s, v_103) in enumerate(sents):
        if is_train_sentence(raw_s):
            fit_idx.append(k)
        else:
            pass  # test
    # Project all through frozen encoder
    V = torch.tensor(np.array([v for _, v in sents]), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        H_all = encoder(V).detach().cpu().numpy()
    # Fit Gaussian from "train" sentences (those whose hash → train bucket)
    fit_mask = np.array([is_train_sentence(s) for s, _ in sents])
    test_mask = ~fit_mask
    H_fit = H_all[fit_mask]
    H_test = H_all[test_mask]
    if len(H_fit) < 2:
        heldout_mu[hi] = H_all.mean(axis=0)
        heldout_sigma[hi] = np.ones(D_OUT, dtype=np.float32)
    else:
        heldout_mu[hi] = H_fit.mean(axis=0)
        var = H_fit.var(axis=0)
        heldout_sigma[hi] = np.clip(np.sqrt(var + 1e-6), SIGMA_MIN, SIGMA_MAX)
    n_test = int(test_mask.sum())
    if n_test > 0:
        for v in H_test:
            heldout_test_sents.append((hi, v))
            heldout_test_y.append(N_TRAIN + hi)  # global Gaussian index

heldout_test_sents = np.array(heldout_test_sents, dtype=object)
heldout_test_y = np.array(heldout_test_y)
print(f"  total held-out test sentences: {len(heldout_test_sents)}")


# ────────── Rank@1 on held-out users against ALL 100 user Gaussians ──────────
print(f"\n[7] held-out user Rank@1 against all 100 user Gaussians "
      f"({N_TRAIN} learned + {len(TEST_USERS)} zero-shot)...")
N_ALL = N_TRAIN + len(TEST_USERS)
all_mu = np.concatenate([learned_mu_train, heldout_mu], axis=0)
all_sigma = np.concatenate([learned_sigma_train, heldout_sigma], axis=0)

if len(heldout_test_sents) > 0:
    n = len(heldout_test_sents)
    energies = np.zeros((n, N_ALL))
    H_test_arr = np.stack([v for _, v in heldout_test_sents])
    for j in range(N_ALL):
        energies[:, j] = energy(H_test_arr, all_mu[j], all_sigma[j])
    pred = energies.argmin(axis=1)
    heldout_rank1 = float((pred == heldout_test_y).mean())
    print(f"  Held-out users Rank@1: {heldout_rank1*100:.1f}%")
else:
    heldout_rank1 = None
    print("  No held-out test sentences.")


# ────────── Per-user Rank@1 breakdown ──────────
per_user = {}
if len(heldout_test_sents) > 0:
    for hi, uid in enumerate(TEST_USERS):
        gi = N_TRAIN + hi  # global Gaussian index
        mask = (heldout_test_y == gi)
        n_test = int(mask.sum())
        if n_test == 0:
            per_user[uid] = {"n_test": 0, "P(M_G>0)": None, "rank1": None}
            continue
        e_self = energies[mask, gi]
        e_other = energies[mask].copy()
        e_other[:, gi] = np.inf
        d_self = e_self
        d_other = e_other.min(axis=1)
        M = d_other - d_self
        rank1_user = float((pred[mask] == gi).mean())
        per_user[uid] = {
            "n_test": n_test,
            "P(M_G>0)": float((M > 0).mean()),
            "med_M_G": float(np.median(M)),
            "rank1": rank1_user,
        }

# ────────── Save ──────────
out_path = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/phase8d_holdout_user_zero_shot.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result = {
    "config": {
        "n_train_users": N_TRAIN,
        "n_holdout_users": len(TEST_USERS),
        "n_total_users": N_ALL,
        "lambda_compact": LAMBDA_COMPACT,
        "lambda_separate": LAMBDA_SEPARATE,
        "lambda_rank": LAMBDA_RANK,
        "bd_margin": BD_MARGIN,
        "rank_margin": RANK_MARGIN,
        "epochs": EPOCHS,
    },
    "data": {
        "n_train_sents": len(train_data),
        "n_test_train_user_sents": len(test_data_train_users),
        "n_holdout_test_sents": len(heldout_test_sents),
    },
    "rank1_train_users_within_test": train_rank1,
    "rank1_holdout_users_zero_shot": heldout_rank1,
    "per_holdout_user": per_user,
    "training_history_tail": hist[-5:],
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nsaved → {out_path}")

# ────────── Verdict ──────────
print(f"\n=== 8.D VERDICT ===")
print(f"  Train users within-user Rank@1 : {train_rank1*100:.1f}%" if train_rank1 else "  Train: N/A")
print(f"  Held-out users zero-shot Rank@1: {heldout_rank1*100:.1f}%" if heldout_rank1 is not None else "  Held-out: N/A")
print(f"  8.A 12u baseline Rank@1        : 29.9%")
if heldout_rank1 is not None:
    go = heldout_rank1 >= 0.299
    print(f"  → {'GO' if go else 'NO-GO'}: held-out Rank@1 ≥ 29.9% baseline? {go}")
