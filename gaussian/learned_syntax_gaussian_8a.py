"""Phase 8.A — Learned Syntax Gaussian.

Pipeline:
  F3 CoreStruct 103d
    → small MLP encoder (103 → 128 → 64 → 32)
    → per-user trainable Gaussian (μ, σ ∈ ℝ³²)
    → contrastive softmax over 12-user same-ASIN cohort
    → L = L_id + λ_σ · L_σ
    → hard σ clamp to [0.1, 2]

Train/test split: 80/20 hash-based, deterministic per sentence.
Test on held-out 20% only.

Goal: compare learned model against PCA48 + Centroid and PCA48 + Mahalanobis
on the SAME 12-user B00A4B34IA cohort + 20% held-out sentences.

GO criteria:
  Rank@1_learned > Rank@1_PCA48 + 10pp
  User Coverage ↑
  |ρ(σ_mean, rank1_freq)| < 0.2
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

# Hash-based 80/20 split (deterministic per sentence text)
SEED_PREFIX = "phase8a_v1|"
TRAIN_FRAC = 0.80

# F3 features
gdoc = np.load("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_shared_scaler.npz", allow_pickle=True)
ALL_FNAMES = list(gdoc["feature_names_ordered"])
FNAMES_F3 = list(gdoc["fnames_f3"])
assert len(FNAMES_F3) == 103, f"expected 103 F3 features, got {len(FNAMES_F3)}"
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

# Training
SMOKE_EPOCHS = int(os.environ.get("PHASE8A_SMOKE_EPOCHS", "20"))
FULL_EPOCHS = int(os.environ.get("PHASE8A_FULL_EPOCHS", "80"))
IS_SMOKE = os.environ.get("PHASE8A_SMOKE", "0") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

NLP = spacy.load("en_core_web_sm", disable=["ner","lemmatizer","attribute_ruler"])

def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]

def is_train_sentence(text: str) -> bool:
    """Hash-based 80/20 split (deterministic)."""
    h = int(hashlib.sha1((SEED_PREFIX + text.strip().lower()).encode()).hexdigest(), 16)
    return (h % 100) < int(TRAIN_FRAC * 100)

def featurize_103(text: str) -> np.ndarray | None:
    """Extract 103d F3 CoreStruct feature for a sentence."""
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
    # mean over sentences (per-feature)
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
    return vec_full[F3_COL]  # (103,)

def project_48(text: str) -> np.ndarray | None:
    """For baseline: extract 103d F3, scale, PCA48."""
    v = featurize_103(text)
    if v is None:
        return None
    v_std = (v - SCALER_MEAN) / SCALER_SCALE
    return (v_std - PCA_MEAN) @ PCA_COMP.T  # (48,)


# ────────── Load data ──────────
print(f"[1] scanning reviews for {len(COHORT)} cohort users...")
t0 = time.time()
user_sents = collections.defaultdict(list)  # uid -> list of raw sentences
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
print(f"  scanned {n_scanned}, found {sum(len(v) for v in user_sents.values())} total sents")

# dedupe
for uid in user_sents:
    seen = set()
    uniq = []
    for s in user_sents[uid]:
        k = s.strip().lower()
        if k not in seen:
            seen.add(k); uniq.append(s)
    user_sents[uid] = uniq

print(f"\n[2] 80/20 hash-split + featurize to 103d...")
train_data = []  # (uid_idx, X_103)
test_data = []
for ui, uid in enumerate(COHORT):
    sents = user_sents.get(uid, [])
    n_train = n_test = 0
    for s in sents:
        v = featurize_103(s)
        if v is None:
            continue
        if is_train_sentence(s):
            train_data.append((ui, v))
            n_train += 1
        else:
            test_data.append((ui, v))
            n_test += 1
    print(f"  {uid[:30]}: n_train={n_train:>3}  n_test={n_test:>3}")
print(f"  total: n_train={len(train_data)}  n_test={len(test_data)}")

# Convert to tensors
X_train = torch.tensor(np.array([d[1] for d in train_data]), dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(np.array([d[0] for d in train_data]), dtype=torch.long, device=DEVICE)
X_test = torch.tensor(np.array([d[1] for d in test_data]), dtype=torch.float32, device=DEVICE)
y_test = torch.tensor(np.array([d[0] for d in test_data]), dtype=torch.long, device=DEVICE)
print(f"  X_train: {tuple(X_train.shape)}  X_test: {tuple(X_test.shape)}")


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
        # initialize mu at 0, log_sigma at 0 (sigma=1)
        self.mu = nn.Parameter(torch.zeros(N_USERS, D_OUT))
        self.log_sigma = nn.Parameter(torch.zeros(N_USERS, D_OUT))

    @property
    def sigma(self):
        s = torch.exp(self.log_sigma)
        return torch.clamp(s, SIGMA_MIN, SIGMA_MAX)

    def energy(self, h, user_idx):
        """Per-user Mahalanobis² energy (no log|Σ|)."""
        mu_u = self.mu[user_idx]                      # (B, D)
        sig_u = self.sigma[user_idx]                  # (B, D)
        diff = h - mu_u                               # (B, D)
        e = 0.5 * ((diff ** 2) / (sig_u ** 2 + 1e-6)).sum(dim=-1)  # (B,)
        return e

    def all_energies(self, h):
        """Compute energy of h against every user. Returns (B, N_USERS)."""
        # h: (B, D); mu: (N, D); sigma: (N, D)
        diff = h.unsqueeze(1) - self.mu.unsqueeze(0)   # (B, N, D)
        sig = self.sigma.unsqueeze(0)                  # (1, N, D)
        e = 0.5 * ((diff ** 2) / (sig ** 2 + 1e-6)).sum(dim=-1)  # (B, N)
        return e


encoder = Encoder().to(DEVICE)
gaussian = UserGaussian().to(DEVICE)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(gaussian.parameters()), lr=LR)


# ────────── Training loop ──────────
def train_epoch(epoch):
    encoder.train(); gaussian.train()
    optimizer.zero_grad()
    h = encoder(X_train)                              # (B, D)
    energies = gaussian.all_energies(h)              # (B, N)
    log_p = -energies / TAU                          # (B, N)
    loss_id = F.cross_entropy(log_p, y_train)
    loss_sigma = (gaussian.log_sigma ** 2).sum()
    loss = loss_id + LAMBDA_SIGMA * loss_sigma
    loss.backward()
    optimizer.step()
    # hard clamp sigma
    with torch.no_grad():
        gaussian.log_sigma.data.clamp_(min=np.log(SIGMA_MIN), max=np.log(SIGMA_MAX))
    return float(loss_id), float(loss_sigma), float(loss)


# ────────── Evaluation ──────────
@torch.no_grad()
def eval_test():
    encoder.eval(); gaussian.eval()
    h = encoder(X_test)                               # (B, D)
    energies = gaussian.all_energies(h)               # (B, N)
    # Rank@1: argmin energy → predicted user
    pred = energies.argmin(dim=1)                     # (B,)
    correct = (pred == y_test).float()
    overall_rank1 = float(correct.mean())
    # per-user P(Rank1)
    per_user_freq = {}
    for ui in range(N_USERS):
        mask = (y_test == ui)
        if mask.sum() == 0:
            per_user_freq[ui] = None
            continue
        per_user_freq[ui] = float(correct[mask].mean())
    return overall_rank1, per_user_freq, energies.cpu().numpy(), pred.cpu().numpy()


@torch.no_grad()
def eval_baseline_pca48_centroid():
    """Build PCA48 vectors for train+test, fit centroid per user from train, evaluate on test."""
    # Project train + test to PCA48
    train_z = []
    for (_, v) in train_data:
        v_std = (v - SCALER_MEAN) / SCALER_SCALE
        train_z.append((v_std - PCA_MEAN) @ PCA_COMP.T)
    test_z = []
    for (_, v) in test_data:
        v_std = (v - SCALER_MEAN) / SCALER_SCALE
        test_z.append((v_std - PCA_MEAN) @ PCA_COMP.T)
    train_z = np.stack(train_z)
    test_z = np.stack(test_z)
    # centroid per user
    centroids = np.zeros((N_USERS, 48))
    for ui in range(N_USERS):
        mask = np.array([d[0] == ui for d in train_data])
        if mask.sum() > 0:
            centroids[ui] = train_z[mask].mean(axis=0)
    # distance to each centroid
    dists = np.zeros((len(test_data), N_USERS))
    for ui in range(N_USERS):
        dists[:, ui] = np.linalg.norm(test_z - centroids[ui], axis=1)
    pred = dists.argmin(axis=1)
    correct = (pred == np.array([d[0] for d in test_data]))
    overall = float(correct.mean())
    per_user = {}
    for ui in range(N_USERS):
        y = np.array([d[0] for d in test_data])
        mask = (y == ui)
        if mask.sum() == 0:
            per_user[ui] = None
            continue
        per_user[ui] = float(correct[mask].mean())
    return overall, per_user


@torch.no_grad()
def eval_baseline_pca48_mahal():
    """Build PCA48 vectors, fit per-user diagonal Mahalanobis from train, evaluate on test."""
    train_z = []
    for (_, v) in train_data:
        v_std = (v - SCALER_MEAN) / SCALER_SCALE
        train_z.append((v_std - PCA_MEAN) @ PCA_COMP.T)
    test_z = []
    for (_, v) in test_data:
        v_std = (v - SCALER_MEAN) / SCALER_SCALE
        test_z.append((v_std - PCA_MEAN) @ PCA_COMP.T)
    train_z = np.stack(train_z)
    test_z = np.stack(test_z)
    y_train_arr = np.array([d[0] for d in train_data])
    y_test_arr = np.array([d[0] for d in test_data])
    mu = np.zeros((N_USERS, 48))
    sig = np.zeros((N_USERS, 48))
    for ui in range(N_USERS):
        mask = (y_train_arr == ui)
        if mask.sum() > 0:
            mu[ui] = train_z[mask].mean(axis=0)
            sig[ui] = train_z[mask].var(axis=0) + 1e-3
    dists = np.zeros((len(test_data), N_USERS))
    for ui in range(N_USERS):
        diff = test_z - mu[ui]
        dists[:, ui] = ((diff ** 2) / sig[ui]).sum(axis=1)
    pred = dists.argmin(axis=1)
    correct = (pred == y_test_arr)
    overall = float(correct.mean())
    per_user = {}
    for ui in range(N_USERS):
        mask = (y_test_arr == ui)
        if mask.sum() == 0:
            per_user[ui] = None
            continue
        per_user[ui] = float(correct[mask].mean())
    return overall, per_user


# ────────── Run ──────────
n_epochs = SMOKE_EPOCHS if IS_SMOKE else FULL_EPOCHS
mode = "SMOKE" if IS_SMOKE else "FULL"
print(f"\n[3] training ({mode}, {n_epochs} epochs, device={DEVICE})...")
t_train = time.time()
hist = []
for ep in range(n_epochs):
    li, ls, lt = train_epoch(ep)
    if (ep + 1) % max(1, n_epochs // 10) == 0 or ep == 0:
        rank1, per_user, _, _ = eval_test()
        hist.append({"epoch": ep + 1, "loss_id": li, "loss_sigma": ls, "loss": lt, "test_rank1": rank1})
        print(f"  ep {ep+1:>3}/{n_epochs}: loss_id={li:.3f}  loss_σ={ls:.3f}  "
              f"loss={lt:.3f}  test_rank1={rank1*100:.1f}%")
print(f"  trained in {time.time()-t_train:.0f}s")


# ────────── Final evaluation ──────────
print(f"\n[4] final evaluation on held-out test set ({len(test_data)} sentences)...")
rank1_learned, per_user_learned, energies, pred = eval_test()
rank1_pca_centroid, per_user_pca_centroid = eval_baseline_pca48_centroid()
rank1_pca_mahal, per_user_pca_mahal = eval_baseline_pca48_mahal()

print(f"\n=== Overall Rank@1 on held-out test ===")
print(f"  PCA48 + Centroid      : {rank1_pca_centroid*100:>5.1f}%")
print(f"  PCA48 + diag Mahal    : {rank1_pca_mahal*100:>5.1f}%")
print(f"  Learned Syntax Gauss. : {rank1_learned*100:>5.1f}%")

# Per-user P(Rank1)
print(f"\n=== Per-user P(Rank1) ===")
print(f"{'uid':<32} {'PCA-Cent':>9} {'PCA-Mahal':>10} {'Learned':>8}  {'σ_mean':>7}")
sigma_mean = gaussian.sigma.mean(dim=1).detach().cpu().numpy()
for ui, uid in enumerate(COHORT):
    pc = per_user_pca_centroid[ui]
    pm = per_user_pca_mahal[ui]
    pl = per_user_learned[ui]
    pc_s = f"{pc*100:>5.1f}%" if pc is not None else "  —  "
    pm_s = f"{pm*100:>5.1f}%" if pm is not None else "  —  "
    pl_s = f"{pl*100:>5.1f}%" if pl is not None else "  —  "
    print(f"{uid[:30]:<32} {pc_s:>9} {pm_s:>10} {pl_s:>8}  "
          f"{sigma_mean[ui]:>6.3f}")

# User coverage
def coverage(per_user):
    return sum(1 for v in per_user.values() if v is not None and v > 0) / N_USERS
print(f"\n=== User Coverage (# users with P(Rank1)>0) ===")
print(f"  PCA48 + Centroid      : {coverage(per_user_pca_centroid)*100:.1f}% "
      f"({int(coverage(per_user_pca_centroid)*N_USERS)}/{N_USERS})")
print(f"  PCA48 + diag Mahal    : {coverage(per_user_pca_mahal)*100:.1f}% "
      f"({int(coverage(per_user_pca_mahal)*N_USERS)}/{N_USERS})")
print(f"  Learned Syntax Gauss. : {coverage(per_user_learned)*100:.1f}% "
      f"({int(coverage(per_user_learned)*N_USERS)}/{N_USERS})")

# ρ(σ_mean, rank1 frequency)
from scipy.stats import pearsonr as pr
# rank1 freq = count of correct rank1 / total test count per user
test_user_count = np.array([(y_test.cpu().numpy() == ui).sum() for ui in range(N_USERS)], dtype=float)
rank1_count = np.array([
    ((pred == ui) & (y_test.cpu().numpy() == ui)).sum()
    for ui in range(N_USERS)
], dtype=float)
rank1_freq = np.where(test_user_count > 0, rank1_count / np.maximum(test_user_count, 1), 0.0)
# σ_mean is in 32d
sigma_mean_arr = gaussian.sigma.mean(dim=1).detach().cpu().numpy()
rho, pval = pr(sigma_mean_arr, rank1_freq)
print(f"\n=== σ confounder check ===")
print(f"  ρ(σ_mean, Rank1 frequency): {rho:+.4f}  (p={pval:.4f}, target |ρ|<0.2)")
print(f"  σ_mean range: [{sigma_mean_arr.min():.3f}, {sigma_mean_arr.max():.3f}]  "
      f"median={np.median(sigma_mean_arr):.3f}")


# ────────── Save result ──────────
out_path = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/phase8a_learned_gaussian.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result = {
    "mode": mode,
    "n_epochs": n_epochs,
    "config": {
        "asin": ASIN,
        "cohort_size": N_USERS,
        "encoder_dims": [D_IN] + D_HID + [D_OUT],
        "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX,
        "tau": TAU, "lambda_sigma": LAMBDA_SIGMA, "lr": LR,
    },
    "data": {
        "n_train": len(train_data),
        "n_test": len(test_data),
        "per_user_n_train": {COHORT[ui]: int(sum(1 for d in train_data if d[0] == ui))
                             for ui in range(N_USERS)},
        "per_user_n_test":  {COHORT[ui]: int(sum(1 for d in test_data  if d[0] == ui))
                             for ui in range(N_USERS)},
    },
    "metrics": {
        "overall_rank1": {
            "pca48_centroid": rank1_pca_centroid,
            "pca48_mahal":    rank1_pca_mahal,
            "learned":        rank1_learned,
        },
        "per_user_rank1": {
            "pca48_centroid": {COHORT[ui]: per_user_pca_centroid[ui] for ui in range(N_USERS)},
            "pca48_mahal":    {COHORT[ui]: per_user_pca_mahal[ui]    for ui in range(N_USERS)},
            "learned":        {COHORT[ui]: per_user_learned[ui]        for ui in range(N_USERS)},
        },
        "user_coverage": {
            "pca48_centroid": coverage(per_user_pca_centroid),
            "pca48_mahal":    coverage(per_user_pca_mahal),
            "learned":        coverage(per_user_learned),
        },
        "sigma_bias": {
            "rho_sigma_rank1_freq": float(rho),
            "p_value": float(pval),
            "sigma_mean_per_user": {COHORT[ui]: float(sigma_mean_arr[ui]) for ui in range(N_USERS)},
            "rank1_freq_per_user": {COHORT[ui]: float(rank1_freq[ui]) for ui in range(N_USERS)},
        },
    },
    "training_history": hist,
    "go_no_go": {
        "rank1_improvement_pp": float((rank1_learned - max(rank1_pca_centroid, rank1_pca_mahal)) * 100),
        "user_coverage_learned": float(coverage(per_user_learned)),
        "user_coverage_best_baseline": float(max(coverage(per_user_pca_centroid),
                                                  coverage(per_user_pca_mahal))),
        "abs_rho_sigma_rank1": float(abs(rho)),
    },
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nsaved → {out_path}")


# ────────── Print GO/NO-GO verdict ──────────
print(f"\n=== GO/NO-GO CHECK ===")
g1 = (rank1_learned > max(rank1_pca_centroid, rank1_pca_mahal) + 0.10)
g2 = (coverage(per_user_learned) > max(coverage(per_user_pca_centroid),
                                       coverage(per_user_pca_mahal)) + 1e-6)
g3 = (abs(rho) < 0.20)
print(f"  (1) Rank@1_learned > best_baseline + 10pp: "
      f"{'✓' if g1 else '✗'} ({rank1_learned*100:.1f}% vs "
      f"{max(rank1_pca_centroid, rank1_pca_mahal)*100:.1f}%, "
      f"Δ={(rank1_learned-max(rank1_pca_centroid, rank1_pca_mahal))*100:+.1f}pp)")
print(f"  (2) User Coverage increase: "
      f"{'✓' if g2 else '✗'} (learned={coverage(per_user_learned)*100:.1f}%, "
      f"best_baseline={max(coverage(per_user_pca_centroid), coverage(per_user_pca_mahal))*100:.1f}%)")
print(f"  (3) |ρ(σ_mean, rank1_freq)| < 0.20: "
      f"{'✓' if g3 else '✗'} (|ρ|={abs(rho):.4f})")
print(f"  → Verdict: {'GO' if (g1 and g2 and g3) else 'NO-GO'}")
