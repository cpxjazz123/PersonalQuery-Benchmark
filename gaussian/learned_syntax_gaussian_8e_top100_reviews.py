"""Phase 8.E — Ranking-margin loss on the global top-100 most-prolific reviewers.

Cohort = 100 users with the most REVIEWS in the whole Baby_Products_2023
corpus (134-554 reviews each, median 177; 60-3725 sents/user, median 698).
Built by build_cohort_top100.py → result/gaussian/phase8e_cohort_top100.json.

Per-user hash sampling caps data size (train ≤200 sents, test ≤60 sents),
deterministic per (uid, sentence). Same loss stack as 8.A.3/8.C.3:
  L = L_id + λ_σ·L_σ + λ_compact·L_compact + λ_separate·L_separate + λ_rank·L_rank

GO criterion (vs 8.C.3 100-user cross-ASIN):
  - Rank@1 ≥ 30.2%
  - σ_bias |ρ| < 0.20
"""
import collections, hashlib, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
sys.path.insert(0, f"{REPO}/common")
from syntactic_features import per_sentence_features_v2 as psf
import spacy, re

from _user_sentence_cache import load_user_sents

# ────────── Config ──────────
COHORT_PATH = f"{REPO}/result/gaussian/phase8e_cohort_top100.json"
with open(COHORT_PATH) as f:
    _cohort_data = json.load(f)
COHORT = [u["uid"] for u in _cohort_data["users"]]
ASIN_LABEL = f"top100_reviews({len(COHORT)}u)"
SEED_PREFIX = "phase8e_v1|"
TRAIN_FRAC = 0.80

MAX_TRAIN_SENTS = 200   # per user, hash-sampled
MAX_TEST_SENTS = 60     # per user, hash-sampled

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


def sample_sents(sents, uid, max_n):
    """Deterministic per-(uid, sentence) hash sample."""
    if len(sents) <= max_n:
        return sents
    def _h(s):
        return int(hashlib.sha1((uid + "|" + s).encode()).hexdigest(), 16)
    return sorted(sents, key=_h)[:max_n]


def _featurize_one_sentence(doc):
    """per_sentence_features_v2 over spaCy doc → 103d F3 vector."""
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


def featurize_batch(texts, batch_size=512):
    """Batch spaCy (n_process=8) + F3 features over many single sentences.

    ~5x faster than per-sentence NLP.pipe; keeps identical feature semantics:
    each input sentence is first split_sents'd and only sents[0] is used,
    matching the historical featurize_103.
    """
    prims = []
    for t in texts:
        ss = [s for s in split_sents(t) if 8 <= len(s.split()) <= 60]
        prims.append(ss[0] if ss else None)
    out = [None] * len(prims)
    idx = [i for i, p in enumerate(prims) if p is not None]
    if not idx:
        return out
    docs = list(NLP.pipe([prims[i] for i in idx], batch_size=batch_size, n_process=8))
    for i, d in zip(idx, docs):
        out[i] = _featurize_one_sentence(d)
    return out


# ────────── Feature cache (scratch2) ──────────
CACHE_KEY = hashlib.sha1((
    os.path.basename(COHORT_PATH) + "|" + SEED_PREFIX + "|" + str(TRAIN_FRAC) + "|"
    + str(MAX_TRAIN_SENTS) + "|" + str(MAX_TEST_SENTS)
).encode()).hexdigest()[:12]
FEAT_CACHE = (f"/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/"
              f"phase8e_feats_{CACHE_KEY}.npz")

# Full-feature 182d cache (built by gaussian/build_features_top100_full.py).
# cohort_hash must match COHORT order — build_features_top100_full.py uses
# SHA1("-".join(sorted(COHORT)))[:12] which is cohort-order independent, so
# this stays stable as long as the cohort membership doesn't change.
_cohort_hash = hashlib.sha1("-".join(sorted(COHORT)).encode()).hexdigest()[:12]
FULL_FEAT_CACHE = (f"/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/"
                   f"phase8e_full_feats_{_cohort_hash}.npz")


def load_from_full_cache(COHORT):
    """Read (train_feats, train_uid, test_feats, test_uid) from the 182d
    full-feature npz (built by build_features_top100_full.py) and apply
    the SAME hash sampling as the original featurize_batch path:
      for each user, take ≤MAX_TRAIN_SENTS train sents (split=0) and
      ≤MAX_TEST_SENTS test sents (split=1), deterministically by
      sample_sents(uid, sentence).
    Then project to F3 columns. Semantically identical to the old
    featurize_batch output (verified 1e-6 on smoke).
    """
    t_f = time.time()
    print(f"  [full-cache] loading {FULL_FEAT_CACHE}")
    z = np.load(FULL_FEAT_CACHE)
    feats_all = z["feats"]            # (N, 182) float32
    uid_idx_all = z["uid_idx"]        # (N,) int32
    split_all = z["split"]            # (N,) int8
    print(f"    loaded in {time.time()-t_f:.1f}s, shape={feats_all.shape}, "
          f"split: train={int((split_all==0).sum())} test={int((split_all==1).sum())}")

    # Group row indices by (uid_idx, split)
    print(f"  [full-cache] hash-sampling (≤{MAX_TRAIN_SENTS}/user train, "
          f"≤{MAX_TEST_SENTS}/user test)...")
    t_s = time.time()
    # We need the original sentence TEXT to apply sample_sents (deterministic
    # per-(uid, sentence)). Re-derive text from the sentence cache (which is
    # cohort-hash-keyed and already built).
    user_sents_cache = load_user_sents(COHORT, REPO)
    train_rows, test_rows = [], []
    for ui, uid in enumerate(COHORT):
        sents_u = list(user_sents_cache.get(uid, []))
        tr_sents = sample_sents([s for s in sents_u if is_train_sentence(s)], uid, MAX_TRAIN_SENTS)
        te_sents = sample_sents([s for s in sents_u if not is_train_sentence(s)], uid, MAX_TEST_SENTS)
        train_rows.append((ui, tr_sents))
        test_rows.append((ui, te_sents))
    print(f"    sampled in {time.time()-t_s:.1f}s")

    # Now: for each (ui, s) selected, find the npz row by (uid_idx == ui) AND
    # sentence text match (npz's `order` field doesn't help here because it
    # indexes into the cache load order, not the npz order). The npz is
    # already in cohort order, so within each uid block rows are sequential.
    # Build (sentence → f3 row) for each user once.
    t_p = time.time()
    print(f"  [full-cache] projecting to F3 (103d) and aligning rows...")
    f3_feats = feats_all[:, F3_COL]   # (N, 103) — slice ~1ms
    print(f"    f3 slice shape={f3_feats.shape} in {time.time()-t_p:.1f}s")

    # Per-user sentence→row map. Since npz stores one row per (uid, sentence)
    # and sentences are loaded in cohort order (uid then cache append order),
    # we can index by `order` within each uid block.
    train_feats, train_uid = [], []
    test_feats, test_uid = [], []
    # Build offsets: row index in npz for (uid, order=j) is sum of |sents| for
    # all preceding uids in cohort order plus j.
    print(f"  [full-cache] building per-user (sentence→row) index...")
    row_start = {}
    cumulative = 0
    for ui, uid in enumerate(COHORT):
        row_start[uid] = cumulative
        cumulative += int((uid_idx_all == ui).sum())
    print(f"    cumulative rows built in {time.time()-t_p:.1f}s")

    # Cache-load order matches the npz order (both walk uid then sentence),
    # so `order` is the within-uid offset. But we need a sentence→order map.
    # Since the sentence cache preserves insertion order (uid then per-uid
    # append), and the npz was built iterating uid in COHORT order, the
    # (uid, order) → row index mapping is: row_start[uid] + order.
    # We map sentence text → order via load_user_sents cache.
    sentence_to_order = {}
    for uid in COHORT:
        sents_u = list(user_sents_cache.get(uid, []))
        for j, s in enumerate(sents_u):
            sentence_to_order[(uid, s)] = j

    n_miss = 0
    for ui, tr_sents in train_rows:
        base = row_start[COHORT[ui]]
        for s in tr_sents:
            j = sentence_to_order.get((COHORT[ui], s))
            if j is None:
                n_miss += 1; continue
            train_feats.append(f3_feats[base + j])
            train_uid.append(ui)
    for ui, te_sents in test_rows:
        base = row_start[COHORT[ui]]
        for s in te_sents:
            j = sentence_to_order.get((COHORT[ui], s))
            if j is None:
                n_miss += 1; continue
            test_feats.append(f3_feats[base + j])
            test_uid.append(ui)
    train_feats = np.asarray(train_feats, dtype=np.float32)
    train_uid = np.asarray(train_uid, dtype=np.int32)
    test_feats = np.asarray(test_feats, dtype=np.float32)
    test_uid = np.asarray(test_uid, dtype=np.int32)
    print(f"  [full-cache] aligned: n_train={len(train_feats)}, n_test={len(test_feats)}, "
          f"miss={n_miss}, total {time.time()-t_f:.1f}s")
    return train_feats, train_uid, test_feats, test_uid


def load_or_build_feats(user_sents, COHORT):
    """Build (train_feats, train_uid, test_feats, test_uid) with caching.

    Priority:
      1. Try the F3-only npz cache (FEAT_CACHE) — written by earlier runs.
      2. Else try the 182d full cache (FULL_FEAT_CACHE) — fast path.
      3. Else fall back to on-the-fly featurize_batch.
    """
    if os.path.exists(FEAT_CACHE):
        print(f"  [feat-cache] hit {FEAT_CACHE}")
        z = np.load(FEAT_CACHE)
        return (z["train_feats"], z["train_uid"], z["test_feats"], z["test_uid"])
    if os.path.exists(FULL_FEAT_CACHE):
        return load_from_full_cache(COHORT)
    print(f"  [feat-cache] miss (no full cache either), batch-featurizing...")
    t_f = time.time()
    train_list = []  # (uid_idx, feat)
    test_list = []
    n_sampled_train = n_sampled_test = 0
    for ui, uid in enumerate(COHORT):
        sents_u = user_sents.get(uid, [])
        tr = sample_sents([s for s in sents_u if is_train_sentence(s)], uid, MAX_TRAIN_SENTS)
        te = sample_sents([s for s in sents_u if not is_train_sentence(s)], uid, MAX_TEST_SENTS)
        n_sampled_train += len(tr)
        n_sampled_test += len(te)
        for s in tr:
            train_list.append((ui, s))
        for s in te:
            test_list.append((ui, s))
    all_rows = [(ui, s, 0) for ui, s in train_list] + [(ui, s, 1) for ui, s in test_list]
    # featurize in one batched pass (train first, then test)
    feats = featurize_batch([s for _, s, _ in all_rows])
    train_feats, train_uid, test_feats, test_uid = [], [], [], []
    for (ui, s, bucket), fv in zip(all_rows, feats):
        if fv is None:
            continue
        if bucket == 0:
            train_feats.append(fv); train_uid.append(ui)
        else:
            test_feats.append(fv); test_uid.append(ui)
    train_feats = np.array(train_feats); train_uid = np.array(train_uid)
    test_feats = np.array(test_feats); test_uid = np.array(test_uid)
    np.savez(FEAT_CACHE, train_feats=train_feats, train_uid=train_uid,
             test_feats=test_feats, test_uid=test_uid)
    print(f"  featurized {len(train_feats)+len(test_feats)} sents in "
          f"{time.time()-t_f:.0f}s → {FEAT_CACHE}")
    print(f"  sampled: train={n_sampled_train}, test={n_sampled_test}")
    return train_feats, train_uid, test_feats, test_uid


# ────────── Load data ──────────
print(f"[1] loading cohort sentences (cached)...", flush=True)
t0 = time.time()
user_sents_cached = load_user_sents(COHORT, REPO)
print(f"  loaded in {time.time()-t0:.1f}s ({sum(len(v) for v in user_sents_cached.values())} sents)")
user_sents = {uid: list(s) for uid, s in user_sents_cached.items()}

print(f"\n[2] featurize train/test to 103d "
      f"(train ≤{MAX_TRAIN_SENTS}/user, test ≤{MAX_TEST_SENTS}/user, hash-sampled)...")
train_feats_np, train_uid_np, test_feats_np, test_uid_np = load_or_build_feats(user_sents, COHORT)
print(f"  featurized: n_train={len(train_feats_np)}, n_test={len(test_feats_np)}")

X_train = torch.tensor(train_feats_np, dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(train_uid_np, dtype=torch.long, device=DEVICE)
X_test = torch.tensor(test_feats_np, dtype=torch.float32, device=DEVICE)
y_test = torch.tensor(test_uid_np, dtype=torch.long, device=DEVICE)
train_data = list(zip(train_uid_np, train_feats_np))
test_data = list(zip(test_uid_np, test_feats_np))


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
    n_with_data = 0
    for u in range(n_users):
        mask = (y == u)
        if not mask.any(): continue
        diff = H[mask] - mu[u]
        energy = 0.5 * ((diff ** 2) / (sig[u] ** 2 + 1e-6)).sum(dim=-1)
        total = total + energy.mean()
        n_with_data += 1
    if n_with_data == 0:
        return total
    return total / n_with_data


def separation_loss_torch(mu, sig, n_users, margin):
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


def ranking_loss_torch(H, mu, sig, y, n_users, margin):
    diff = H.unsqueeze(1) - mu.unsqueeze(0)              # (N, n_users, D)
    energy = 0.5 * ((diff ** 2) / (sig.unsqueeze(0) ** 2 + 1e-6)).sum(dim=-1)
    diag_self = energy.gather(1, y.unsqueeze(1)).squeeze(1)  # (N,)
    energy_masked = energy.clone()
    energy_masked.scatter_(1, y.unsqueeze(1), float("inf"))
    nearest_other, _ = energy_masked.min(dim=1)              # (N,)
    per_sentence = torch.clamp(diag_self + margin - nearest_other, min=0.0)
    return per_sentence.mean()


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
    loss_compact = compact_loss_torch(h, gaussian.mu, gaussian.sigma, y_train, N_USERS)
    loss_separate = separation_loss_torch(gaussian.mu, gaussian.sigma, N_USERS, BD_MARGIN)
    loss_rank = ranking_loss_torch(h, gaussian.mu, gaussian.sigma, y_train, N_USERS, RANK_MARGIN)
    loss = loss_id + LAMBDA_SIGMA * loss_sigma \
                + LAMBDA_COMPACT * loss_compact \
                + LAMBDA_SEPARATE * loss_separate \
                + LAMBDA_RANK * loss_rank
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        gaussian.log_sigma.data.clamp_(min=np.log(SIGMA_MIN), max=np.log(SIGMA_MAX))
    return (float(loss_id), float(loss_sigma), float(loss_compact),
            float(loss_separate), float(loss_rank), float(loss))


print(f"\n[3] training ({EPOCHS} epochs, device={DEVICE})...")
print(f"  λ_compact={LAMBDA_COMPACT}  λ_separate={LAMBDA_SEPARATE}  "
      f"λ_rank={LAMBDA_RANK}  bd_margin={BD_MARGIN}  rank_margin={RANK_MARGIN}")
t_train = time.time()
hist = []
for ep in range(EPOCHS):
    losses = train_epoch()
    hist.append({"epoch": ep + 1, **{k: v for k, v in zip(
        ["loss_id", "loss_sigma", "loss_compact", "loss_separate",
         "loss_rank", "loss_total"], losses)}})
    if (ep + 1) % 10 == 0 or ep == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep+1:>3}/{EPOCHS}: id={losses[0]:.3f} σ={losses[1]:.3f} "
              f"compact={losses[2]:.3f} sep={losses[3]:.3f} rank={losses[4]:.3f} "
              f"total={losses[5]:.3f}", flush=True)
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


learned_energy_test = np.zeros((len(y_test_np), N_USERS))
for ui in range(N_USERS):
    learned_energy_test[:, ui] = energy_diag(H_test, learned_mu[ui], learned_sigma[ui])

learned_energy_train = np.zeros((len(y_train_np), N_USERS))
for ui in range(N_USERS):
    learned_energy_train[:, ui] = energy_diag(H_train, learned_mu[ui], learned_sigma[ui])

learned_mu_dists = pairwise_mu_dist(learned_mu)
learned_bd = pairwise_bhatt(learned_mu, learned_sigma)
M_G_learned, d_self_l, d_other_l = compute_test_metrics(learned_energy_test, y_test_np)
M_G_train, d_self_train, d_other_train = compute_test_metrics(learned_energy_train, y_train_np)

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

from scipy.stats import pearsonr as pr
sigma_means = learned_sigma.mean(axis=1)
rank1_per_user = np.array([per_user_audit[uid]["P_M_G_pos"] or 0.0 for uid in COHORT])
rho, pval = pr(sigma_means, rank1_per_user)

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
print(f"\n=== 8.E RESULTS ===")
print(f"\n[Rank@1]")
print(f"  Held-out overall Rank@1 : {rank1_overall*100:.1f}%")
print(f"  8.C.3 100-user baseline : 30.2%")
print(f"\n[Audit 1] μ pairwise distance")
print(f"  mean={learned_mu_dists.mean():.4f}  median={np.median(learned_mu_dists):.4f}")
print(f"\n[Audit 2] Bhattacharyya distance")
print(f"  mean={learned_bd.mean():.4f}  median={np.median(learned_bd):.4f}")
print(f"\n[Audit 3] Within-user vs Between-user")
print(f"  D_self mean: {d_self_l.mean():.4f}    D_other mean: {d_other_l.mean():.4f}    "
      f"margin: {(d_other_l.mean() - d_self_l.mean()):+.4f}")
print(f"  Median M_G: {np.median(M_G_learned):+.4f}  P(M_G>0): {(M_G_learned>0).mean()*100:.1f}%")
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
out_path = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/gaussian/phase8e_ranking_margin_top100.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
result = {
    "config": {
        "cohort": ASIN_LABEL,
        "selection": "global_top100_review_count",
        "n_users": N_USERS,
        "epochs": EPOCHS,
        "max_train_sents_per_user": MAX_TRAIN_SENTS,
        "max_test_sents_per_user": MAX_TEST_SENTS,
        "encoder_dims": [D_IN] + D_HID + [D_OUT],
        "lambda_compact": LAMBDA_COMPACT,
        "lambda_separate": LAMBDA_SEPARATE,
        "bd_margin": BD_MARGIN,
        "lambda_rank": LAMBDA_RANK,
        "rank_margin": RANK_MARGIN,
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
    "comparison": {
        "rank1": {"8c3_100user_cross_asin": 0.302, "8e_top100_reviews": rank1_overall,
                  "delta_8e_vs_8c3_pp": (rank1_overall - 0.302) * 100},
        "sigma_bias_abs_rho": {"8c3": 0.016, "8e": float(abs(rho))},
    },
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nsaved → {out_path}")

# ────────── Verdict ──────────
print(f"\n=== 8.E GO criterion (vs 8.C.3) ===")
rank1_ok = rank1_overall >= 0.302
sigma_ok = abs(rho) < 0.20
print(f"  Rank@1 ≥ 30.2%   : {rank1_ok}  (got {rank1_overall*100:.1f}%)")
print(f"  |ρ(σ)| < 0.20    : {sigma_ok}  (got {abs(rho):.3f})")
if rank1_ok and sigma_ok:
    print(f"  → GO: top-100 评论用户 cohort 保持 ranking 能力")
elif rank1_ok:
    print(f"  → PARTIAL-GO: Rank@1 ≥ 30.2%,但 σ_bias 需检查")
else:
    print(f"  → NO-GO: top-100 评论用户 cohort 未能保持 ranking (详见 metric 表)")
