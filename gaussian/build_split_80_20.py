"""Phase 8.D.0 — 80/20 user split for held-out generalization test.

Splits the 100-user cohort (from phase8c_cohort_100user.json) into:
  - 80 train users: encoder + Gaussian trained on these users
  - 20 held-out users: ZERO-shot test — encoder never sees their
                       sentences during training

Split is deterministic by user-id hash (independent of sentence order).

Saved → result/gaussian/phase8d_split_80_20.json
"""
import hashlib, json, os

REPO = "/home/wlia0047/ar57/wenyu/PersoanlQuery"
IN_PATH = f"{REPO}/result/gaussian/phase8c_cohort_100user.json"
OUT_PATH = f"{REPO}/result/gaussian/phase8d_split_80_20.json"

SPLIT_SEED = "phase8d_split_v1|"
TRAIN_FRAC = 0.80

with open(IN_PATH) as f:
    cohort_data = json.load(f)

train_users, test_users = [], []
for entry in cohort_data["users"]:
    uid = entry["uid"]
    h = int(hashlib.sha1((SPLIT_SEED + uid).encode()).hexdigest(), 16) % 100
    rec = dict(entry)
    if h < int(TRAIN_FRAC * 100):
        train_users.append(rec)
    else:
        test_users.append(rec)

print(f"split: {len(train_users)} train / {len(test_users)} test users")
n_asin_train = len({u["asin"] for u in train_users})
n_asin_test = len({u["asin"] for u in test_users})
print(f"  train ASINs: {n_asin_train}")
print(f"  test ASINs: {n_asin_test}")
overlap = set(u["asin"] for u in train_users) & set(u["asin"] for u in test_users)
print(f"  overlap ASINs: {len(overlap)}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump({
        "config": {
            "SPLIT_SEED": SPLIT_SEED,
            "TRAIN_FRAC": TRAIN_FRAC,
            "n_train_users": len(train_users),
            "n_test_users": len(test_users),
        },
        "train_users": train_users,
        "test_users": test_users,
    }, f, indent=2)
print(f"\nsaved → {OUT_PATH}")
