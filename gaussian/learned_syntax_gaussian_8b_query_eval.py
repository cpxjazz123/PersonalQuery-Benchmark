"""Phase 8.B — Apply trained Learned Syntax Gaussian to LLM-generated pool queries.

For each of 173 strict B00A4B34IA pool queries:
  - Project to 103d F3
  - Forward through trained encoder (103 → 32)
  - Compute Mahalanobis² energy to all 12 users
  - Rank1 = argmin energy

Compare 4 methods on the same 173 queries:
  1. PCA48 + Centroid (raw)
  2. PCA48 + diagonal Mahalanobis (raw)
  3. Self-CDF calibrated (7.J.12/7.J.13 standard)
  4. Learned Syntax Gaussian (this phase)

Output metrics:
  - Rank@1 distribution per user (each method)
  - User Coverage (each method)
  - Top-3 concentration
  - M_CDF exclusivity per user (learned only)
  - σ_bias ρ(σ_mean, rank1_freq)
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
EPOCHS = int(os.environ.get("PHASE8B_EPOCHS", "80"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

NLP = spacy.load("en_core_web_sm", disable=["ner","lemmatizer","attribute_ruler"])


def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def is_train_sentence(text: str) -> bool:
    h = int(hashlib.sha1((SEED_PREFIX + text.strip().lower()).encode()).hexdigest(), 16)
    return (h % 100) < int(TRAIN_FRAC * 100)


def featurize_103(text: str):
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


# ────────── Load training data ──────────
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

print(f"\n[2] 80/20 split + featurize...")
train_data = []
for ui, uid in enumerate(COHORT):
    for s in user_sents.get(uid, []):
        if not is_train_sentence(s):
            continue
        v = featurize_103(s)
        if v is not None:
            train_data.append((ui, v))
print(f"  n_train={len(train_data)}")

X_train = torch.tensor(np.array([d[1] for d in train_data]), dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(np.array([d[0] for d in train_data]), dtype=torch.long, device=DEVICE)

# Also build self-history sents for CDF calibration
print("\n[3] project self-history for self-CDF calibration...")
user_z_self = {}
self_dist = {}
for ui, uid in enumerate(COHORT):
    sents = user_sents.get(uid, [])
    zs = []
    for s in sents:
        v = featurize_103(s)
        if v is not None:
            zs.append(v)
    user_z_self[uid] = np.asarray(zs, dtype=np.float64)
print(f"  per-user n_self_z: " + ", ".join(f"{COHORT[i][:8]}={len(user_z_self[COHORT[i]])}" for i in range(N_USERS)))


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

    def all_energies(self, h):
        diff = h.unsqueeze(1) - self.mu.unsqueeze(0)
        sig = self.sigma.unsqueeze(0)
        return 0.5 * ((diff ** 2) / (sig ** 2 + 1e-6)).sum(dim=-1)


encoder = Encoder().to(DEVICE)
gaussian = UserGaussian().to(DEVICE)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(gaussian.parameters()), lr=LR)


def train_epoch():
    encoder.train(); gaussian.train()
    optimizer.zero_grad()
    h = encoder(X_train)
    energies = gaussian.all_energies(h)
    log_p = -energies / TAU
    loss_id = F.cross_entropy(log_p, y_train)
    loss_sigma = (gaussian.log_sigma ** 2).sum()
    loss = loss_id + LAMBDA_SIGMA * loss_sigma
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        gaussian.log_sigma.data.clamp_(min=np.log(SIGMA_MIN), max=np.log(SIGMA_MAX))


print(f"\n[4] training ({EPOCHS} epochs, device={DEVICE})...")
t_train = time.time()
for ep in range(EPOCHS):
    train_epoch()
    if (ep + 1) % 20 == 0 or ep == 0:
        with torch.no_grad():
            h_tr = encoder(X_train)
            e_tr = gaussian.all_energies(h_tr)
            acc_tr = float((e_tr.argmin(dim=1) == y_train).float().mean())
        print(f"  ep {ep+1:>3}/{EPOCHS}: in-sample acc={acc_tr*100:.1f}%")
print(f"  trained in {time.time()-t_train:.0f}s")


# ────────── Load pool queries ──────────
print(f"\n[5] loading {ASIN} strict pool queries...")
pool_full = json.load(open("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_K200_F3pca48_full.json"))
queries = pool_full["pools"][ASIN]
strict_q = [q for q in queries if q.get("strict", False)]
print(f"  pool total={len(queries)}, strict={len(strict_q)}")

# project queries to 103d
print(f"\n[6] projecting {len(strict_q)} queries to 103d + 48d...")
q_keys = sorted([q["k"] for q in strict_q])
query_z103 = {}
query_z48 = {}
for q in strict_q:
    k = q["k"]
    v103 = featurize_103(q["query"])
    if v103 is not None:
        query_z103[k] = v103
        v48 = project_48(q["query"])
        query_z48[k] = v48
print(f"  projected {len(query_z103)}/{len(strict_q)} queries")
q_keys = sorted(query_z103.keys())

X_q = torch.tensor(np.array([query_z103[k] for k in q_keys]), dtype=torch.float32, device=DEVICE)
Z_q48 = np.stack([query_z48[k] for k in q_keys])  # (n_q, 48)


# ────────── Method 1: Learned Syntax Gaussian ──────────
print(f"\n[7] evaluating learned model on {len(q_keys)} queries...")
encoder.eval(); gaussian.eval()
with torch.no_grad():
    h_q = encoder(X_q)                              # (n_q, 32)
    learned_energy = gaussian.all_energies(h_q)     # (n_q, 12)
    learned_rank1_col = learned_energy.argmin(dim=1).cpu().numpy()
    learned_energy_np = learned_energy.cpu().numpy()

# Also need raw Mahalanobis on 32d (within learned space)
learned_raw_m2 = (learned_energy_np * 2)  # energy = 0.5 * m², so m² = 2 * energy

learned_count = collections.Counter(learned_rank1_col.tolist())


# ────────── Method 2: PCA48 + Centroid (in-sample) ──────────
print(f"\n[8] evaluating PCA48 + Centroid baseline (in-sample centroid from self-history)...")
# Use user_z_self (per-user sentence z's) to fit centroids in 48d
mu_centroid = {}
for ui, uid in enumerate(COHORT):
    Z = user_z_self[uid]
    if len(Z) > 0:
        # z from self-history, project to 48d
        Z48 = (Z - SCALER_MEAN) / SCALER_SCALE
        Z48 = (Z48 - PCA_MEAN) @ PCA_COMP.T
        mu_centroid[uid] = Z48.mean(axis=0)
    else:
        mu_centroid[uid] = np.zeros(48)

dist_centroid = np.zeros((len(q_keys), N_USERS))
for ui, uid in enumerate(COHORT):
    dist_centroid[:, ui] = np.linalg.norm(Z_q48 - mu_centroid[uid], axis=1)
centroid_rank1 = dist_centroid.argmin(axis=1)
centroid_count = collections.Counter(centroid_rank1.tolist())


# ────────── Method 3: PCA48 + diagonal Mahalanobis ──────────
print(f"[9] evaluating PCA48 + Mahalanobis (in-sample)...")
sigma_diag = {}
for ui, uid in enumerate(COHORT):
    Z = user_z_self[uid]
    if len(Z) > 0:
        Z48 = (Z - SCALER_MEAN) / SCALER_SCALE
        Z48 = (Z48 - PCA_MEAN) @ PCA_COMP.T
        sigma_diag[uid] = Z48.var(axis=0) + 1e-3
    else:
        sigma_diag[uid] = np.ones(48) * 1e-3

dist_mahal = np.zeros((len(q_keys), N_USERS))
for ui, uid in enumerate(COHORT):
    diff = Z_q48 - mu_centroid[uid]
    dist_mahal[:, ui] = ((diff ** 2) / sigma_diag[uid]).sum(axis=1)
mahal_rank1 = dist_mahal.argmin(axis=1)
mahal_count = collections.Counter(mahal_rank1.tolist())


# ────────── Method 4: Self-CDF calibrated (7.J.12 standard) ──────────
print(f"[10] evaluating self-CDF calibrated (using per-user Mahalanobis on PCA48)...")
# Build per-user self CDF of d²_self for each user (from 103d path)
self_dist_103 = {}
for ui, uid in enumerate(COHORT):
    Z = user_z_self[uid]
    if len(Z) == 0:
        self_dist_103[uid] = np.array([]); continue
    mu_u, sd_u = mu_centroid[uid], sigma_diag[uid]
    dists = []
    for z in Z:
        z48 = (z - SCALER_MEAN) / SCALER_SCALE
        z48 = (z48 - PCA_MEAN) @ PCA_COMP.T
        d = ((z48 - mu_u)**2 / sd_u).sum()
        dists.append(float(d))
    self_dist_103[uid] = np.sort(np.asarray(dists))

def percentile_self(d, sorted_self):
    if len(sorted_self) == 0: return 1.0
    return float(np.searchsorted(sorted_self, d, side='right') / len(sorted_self))

S_cdf = np.zeros((len(q_keys), N_USERS))
for i in range(len(q_keys)):
    for ui, uid in enumerate(COHORT):
        S_cdf[i, ui] = percentile_self(dist_mahal[i, ui], self_dist_103[uid])
self_cdf_rank1 = S_cdf.argmin(axis=1)
self_cdf_count = collections.Counter(self_cdf_rank1.tolist())


# ────────── Print comparison ──────────
print(f"\n=== 4-method Rank@1 distribution on {len(q_keys)} strict pool queries ===")
print(f"{'uid':<32} {'Centroid':>9} {'Mahal':>8} {'Self-CDF':>9} {'Learned':>9}  {'σ_mean':>7}")
print("-"*90)
sigma_arr = gaussian.sigma.mean(dim=1).detach().cpu().numpy()
for ui, uid in enumerate(COHORT):
    c = centroid_count.get(ui, 0)
    m = mahal_count.get(ui, 0)
    s = self_cdf_count.get(ui, 0)
    L = learned_count.get(ui, 0)
    print(f"{uid[:30]:<32} {c:>4} ({c/len(q_keys)*100:>4.1f}%) "
          f"{m:>3} ({m/len(q_keys)*100:>4.1f}%) "
          f"{s:>4} ({s/len(q_keys)*100:>4.1f}%) "
          f"{L:>4} ({L/len(q_keys)*100:>4.1f}%) "
          f"  {sigma_arr[ui]:>6.3f}")

# Top-3 concentration
def top3(counter, n):
    return sum(v for _, v in counter.most_common(3)) / n
print(f"\n=== Top-3 concentration ===")
print(f"  PCA48 Centroid : {top3(centroid_count, len(q_keys))*100:.1f}%")
print(f"  PCA48 Mahal    : {top3(mahal_count, len(q_keys))*100:.1f}%")
print(f"  Self-CDF       : {top3(self_cdf_count, len(q_keys))*100:.1f}%")
print(f"  Learned        : {top3(learned_count, len(q_keys))*100:.1f}%")

# User coverage
def cov(counter):
    return sum(1 for v in counter.values() if v > 0) / N_USERS
print(f"\n=== User Coverage ===")
print(f"  PCA48 Centroid : {cov(centroid_count)*100:.1f}% ({int(cov(centroid_count)*N_USERS)}/{N_USERS})")
print(f"  PCA48 Mahal    : {cov(mahal_count)*100:.1f}% ({int(cov(mahal_count)*N_USERS)}/{N_USERS})")
print(f"  Self-CDF       : {cov(self_cdf_count)*100:.1f}% ({int(cov(self_cdf_count)*N_USERS)}/{N_USERS})")
print(f"  Learned        : {cov(learned_count)*100:.1f}% ({int(cov(learned_count)*N_USERS)}/{N_USERS})")

# σ_bias check on learned
from scipy.stats import pearsonr as pr
test_user_count = np.array([(learned_rank1_col == ui).sum() for ui in range(N_USERS)], dtype=float)
rho, pval = pr(sigma_arr, test_user_count)
print(f"\n=== Learned σ confounder check ===")
print(f"  ρ(σ_mean, rank1 count): {rho:+.4f}  (p={pval:.4f}, target |ρ|<0.2)")
print(f"  σ_mean range: [{sigma_arr.min():.3f}, {sigma_arr.max():.3f}]  median={np.median(sigma_arr):.3f}")


# ────────── M_CDF exclusivity on Learned ──────────
print(f"\n[11] M_CDF exclusivity (Learned only)...")
per_user_r1 = collections.defaultdict(list)
for i in range(len(q_keys)):
    col = learned_rank1_col[i]
    uid = COHORT[col]
    s_u = learned_energy_np[i, col]
    s_sorted = np.sort(learned_energy_np[i])
    s_2nd = s_sorted[1] if len(s_sorted) > 1 else 1.0
    second_uid = COHORT[int(np.where(learned_energy_np[i] == s_2nd)[0][0])]
    # M_CDF in self-CDF space: use S_cdf (CDF calibrated scores)
    s_u_cdf = S_cdf[i, col]
    s_2nd_cdf = S_cdf[i, self_cdf_rank1[i] if self_cdf_rank1[i] != col else
                       int(np.argsort(S_cdf[i])[1])]
    m_cdf = s_2nd_cdf - s_u_cdf
    per_user_r1[uid].append({
        "k": q_keys[i],
        "S_u_cdf": float(s_u_cdf),
        "S_2nd_cdf": float(s_2nd_cdf),
        "second_uid": second_uid,
        "learned_energy_u": float(s_u),
        "learned_energy_2nd": float(s_2nd),
        "M_CDF": float(m_cdf),
    })

print(f"\n{'uid':<32} {'n_r1':>5} {'best_M_CDF':>11}")
print("-"*60)
for uid in COHORT:
    cands = per_user_r1[uid]
    if not cands:
        print(f"{uid[:30]:<32} {0:>5}     —")
        continue
    best = max(cands, key=lambda c: c["M_CDF"])
    print(f"{uid[:30]:<32} {len(cands):>5} {best['M_CDF']:>+10.4f}")
    # also print query text
    qtext = next(q["query"] for q in strict_q if q["k"] == best["k"])
    print(f"    q*: \"{qtext[:120]}\"")


# ────────── Save result ──────────
out_path = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/phase8b_learned_query_eval.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result = {
    "config": {
        "asin": ASIN,
        "cohort_size": N_USERS,
        "epochs": EPOCHS,
        "encoder_dims": [D_IN] + D_HID + [D_OUT],
        "sigma_min": SIGMA_MIN, "sigma_max": SIGMA_MAX,
        "tau": TAU, "lambda_sigma": LAMBDA_SIGMA, "lr": LR,
    },
    "data": {
        "n_pool_total": len(queries),
        "n_pool_strict": len(strict_q),
        "n_pool_projected": len(q_keys),
        "n_train_sentences": len(train_data),
    },
    "metrics": {
        "rank1_count_per_user": {
            "pca48_centroid": {COHORT[ui]: int(centroid_count.get(ui, 0)) for ui in range(N_USERS)},
            "pca48_mahal":    {COHORT[ui]: int(mahal_count.get(ui, 0))    for ui in range(N_USERS)},
            "self_cdf":       {COHORT[ui]: int(self_cdf_count.get(ui, 0)) for ui in range(N_USERS)},
            "learned":        {COHORT[ui]: int(learned_count.get(ui, 0))  for ui in range(N_USERS)},
        },
        "user_coverage": {
            "pca48_centroid": cov(centroid_count),
            "pca48_mahal":    cov(mahal_count),
            "self_cdf":       cov(self_cdf_count),
            "learned":        cov(learned_count),
        },
        "top3_concentration": {
            "pca48_centroid": top3(centroid_count, len(q_keys)),
            "pca48_mahal":    top3(mahal_count, len(q_keys)),
            "self_cdf":       top3(self_cdf_count, len(q_keys)),
            "learned":        top3(learned_count, len(q_keys)),
        },
        "learned_sigma": {
            "sigma_mean_per_user": {COHORT[ui]: float(sigma_arr[ui]) for ui in range(N_USERS)},
            "rho_sigma_rank1_count": float(rho),
            "p_value": float(pval),
        },
    },
    "M_CDF_per_user_learned": {
        uid: sorted([c for c in per_user_r1[uid]],
                    key=lambda c: -c["M_CDF"])
        for uid in COHORT
    },
    "go_no_go": {
        "learned_top3_concentration": float(top3(learned_count, len(q_keys))),
        "self_cdf_top3_concentration": float(top3(self_cdf_count, len(q_keys))),
        "learned_coverage": float(cov(learned_count)),
        "self_cdf_coverage": float(cov(self_cdf_count)),
        "abs_rho_sigma_rank1": float(abs(rho)),
    },
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nsaved → {out_path}")

# ────────── Verdict ──────────
print(f"\n=== GO/NO-GO ===")
print(f"  Learned top-3 concentration: {top3(learned_count, len(q_keys))*100:.1f}% "
      f"(vs Self-CDF {top3(self_cdf_count, len(q_keys))*100:.1f}%)")
print(f"  Learned user coverage: {cov(learned_count)*100:.1f}% "
      f"(vs Self-CDF {cov(self_cdf_count)*100:.1f}%)")
print(f"  σ_bias |ρ|: {abs(rho):.4f}")
