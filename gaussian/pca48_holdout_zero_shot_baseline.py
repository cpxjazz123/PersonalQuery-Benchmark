"""Phase 8.D diagnostic — PCA48 unsupervised zero-shot baseline.

Same 80/20 user split + within-user 80/20 sentence hash as the 8.D
learned-encoder test, but "encoder" = fixed PCA48 projection (never
trained on any user).

Purpose: decompose the 8.D NO-GO (held-out Rank@1 = 6.7% vs train-user
39.8%). If PCA48 held-out Rank@1 is also ~5-10%, the ceiling is the
task itself (sentence-level identification of brand-new users). If
PCA48 is much higher, the learned encoder overfits its 74 train users.

Saved → result/gaussian/phase8d_pca48_zero_shot_baseline.json
"""
import hashlib, json, os, re, sys, time
import numpy as np

REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
sys.path.insert(0, f"{REPO}/common")
from syntactic_features import per_sentence_features_v2 as psf
import spacy

from _user_sentence_cache import load_user_sents

# ────────── Config (identical to 8.D) ──────────
SPLIT_PATH = f"{REPO}/result/gaussian/phase8d_split_80_20.json"
SEED_PREFIX = "phase8d_v1|"
TRAIN_FRAC = 0.80

NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "attribute_ruler"])


def split_sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]


def is_train_sentence(text):
    h = int(hashlib.sha1((SEED_PREFIX + text.strip().lower()).encode()).hexdigest(), 16)
    return (h % 100) < int(TRAIN_FRAC * 100)


# ────────── Load F3 scaler/PCA ──────────
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
D = PCA_COMP.shape[0]  # 48


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


def project_48(v103):
    v_std = (v103 - SCALER_MEAN) / SCALER_SCALE
    return (v_std - PCA_MEAN) @ PCA_COMP.T


# ────────── Load split ──────────
print("[1] loading 80/20 user split...", flush=True)
with open(SPLIT_PATH) as f:
    split_data = json.load(f)
TRAIN_USERS = [u["uid"] for u in split_data["train_users"]]
TEST_USERS = [u["uid"] for u in split_data["test_users"]]
ALL_USERS = TRAIN_USERS + TEST_USERS
N_TRAIN = len(TRAIN_USERS)
print(f"  train={N_TRAIN}, held-out={len(TEST_USERS)}")

print("\n[2] loading cached sentences...")
t0 = time.time()
sents_all = load_user_sents(ALL_USERS, REPO)
print(f"  loaded in {time.time()-t0:.1f}s ({sum(len(v) for v in sents_all.values())} sents)")

# ────────── Featurize + project to PCA48 ──────────
print("\n[3] featurize + PCA48 project...")
train_z, train_y = [], []       # 74 train users' 80% sents
test_z, test_y = [], []         # 74 train users' 20% sents (within-user sanity)
holdout_all = {uid: [] for uid in TEST_USERS}  # uid -> list of z48

for ui, uid in enumerate(ALL_USERS):
    for s in sents_all.get(uid, []):
        v = featurize_103(s)
        if v is None:
            continue
        z = project_48(v)
        if uid in TRAIN_USERS:
            if is_train_sentence(s):
                train_z.append(z); train_y.append(ui)
            else:
                test_z.append(z); test_y.append(ui)
        else:
            holdout_all[uid].append((is_train_sentence(s), z))

train_z = np.array(train_z); train_y = np.array(train_y)
test_z = np.array(test_z); test_y = np.array(test_y)
print(f"  train sents={len(train_z)}, train-user test sents={len(test_z)}, "
      f"holdout sents={sum(len(v) for v in holdout_all.values())}")


def energy(z, mu, sig):
    diff = z - mu
    return 0.5 * ((diff ** 2) / (sig ** 2 + 1e-6)).sum(axis=-1)


def fit_gaussian(Z):
    mu = Z.mean(axis=0)
    var = Z.var(axis=0)
    sig = np.clip(np.sqrt(var + 1e-6), 0.1, 2.0)
    return mu, sig


# ────────── Fit Gaussians ──────────
print("\n[4] fit Gaussians...")
mus = np.zeros((len(ALL_USERS), D))
sigs = np.zeros((len(ALL_USERS), D))
for ui in range(N_TRAIN):
    mask = train_y == ui
    if mask.sum() >= 2:
        mu, sig = fit_gaussian(train_z[mask])
    else:
        mu, sig = train_z[mask].mean(axis=0), np.ones(D)
    mus[ui], sigs[ui] = mu, sig
for hi, uid in enumerate(TEST_USERS):
    gi = N_TRAIN + hi
    fit = np.array([z for is_tr, z in holdout_all[uid] if is_tr])
    if len(fit) >= 2:
        mus[gi], sigs[gi] = fit_gaussian(fit)
    else:
        mus[gi], sigs[gi] = np.zeros(D), np.ones(D)

# ────────── Rank@1 ──────────
print("\n[5] Rank@1 eval...")
# Train users (within-user sanity)
E = np.zeros((len(test_z), len(ALL_USERS)))
for j in range(len(ALL_USERS)):
    E[:, j] = energy(test_z, mus[j], sigs[j])
pred = E.argmin(axis=1)
train_rank1 = float((pred == test_y).mean())

# Held-out users
H_z, H_y = [], []
for hi, uid in enumerate(TEST_USERS):
    gi = N_TRAIN + hi
    for is_tr, z in holdout_all[uid]:
        if not is_tr:
            H_z.append(z); H_y.append(gi)
H_z = np.array(H_z); H_y = np.array(H_y)
E2 = np.zeros((len(H_z), len(ALL_USERS)))
for j in range(len(ALL_USERS)):
    E2[:, j] = energy(H_z, mus[j], sigs[j])
pred2 = E2.argmin(axis=1)
holdout_rank1 = float((pred2 == H_y).mean())

# ────────── Save ──────────
out_path = f"{REPO}/result/gaussian/phase8d_pca48_zero_shot_baseline.json"
result = {
    "config": {"n_train_users": N_TRAIN, "n_holdout_users": len(TEST_USERS),
               "dim": D, "space": "PCA48-unsupervised"},
    "rank1_train_users_within_test": train_rank1,
    "rank1_holdout_users_zero_shot": holdout_rank1,
    "n_holdout_test_sents": len(H_z),
    "learned_encoder_reference": {"rank1_train_users_within_test": 0.398,
                                  "rank1_holdout_users_zero_shot": 0.067,
                                  "note": "from phase8d_holdout_user_zero_shot.json (80ep)"},
}
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nsaved → {out_path}")

print(f"\n=== 8.D PCA48 DIAGNOSTIC ===")
print(f"  PCA48 train users within-user Rank@1 : {train_rank1*100:.1f}%")
print(f"  PCA48 held-out users zero-shot Rank@1: {holdout_rank1*100:.1f}%  (n={len(H_z)})")
print(f"  learned-encoder held-out Rank@1      : 6.7%")
