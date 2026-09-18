#!/usr/bin/env python3
"""Train on real (user, ASIN) sentence cohorts.

Protocol:
  - real parent_asin sentence ownership from raw JSONL
  - retain (user, ASIN) pairs with >= 50 real sentences
  - retain ASINs with >= 3 qualified users, then deterministically keep exactly 3 users
  - split each pair's 50 sentences into 30 train / 10 val / 10 test
  - each train anchor uses 10 positives from the same (user, ASIN) train set
    and 10 negatives from each of the other 2 users (20 negatives total)
  - multi-positive InfoNCE; no fallback sampling
"""
from __future__ import annotations

import html
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
RAW_DATA = REPO_ROOT / "data/Baby_Products_2023.jsonl"
SENT_VECTORS = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/sent_vectors.npz")
SVD_COMPONENTS = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/svd_components.npz")
USER_N_SENTS = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/user_n_sents.json")
UID_LIST = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/uid_list.json")
OUT_DIR = REPO_ROOT / "result/03_spacy_encode"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42
N_PAIR_SENTENCES = 30
N_TRAIN = 18
N_VAL = 6
N_TEST = 6
COHORT_USERS = 3
N_POS = 6  # positives per anchor per epoch (1/3 of N_TRAIN, aligned with protocol)
K_PER_NEG_USER = N_POS  # K must equal the positive count
N_EPOCHS = 30
BATCH_SIZE = 128
N_NEG = (COHORT_USERS - 1) * K_PER_NEG_USER
TEMPERATURE = 0.1
LR = 3e-3
SVD_DIM = 256
MLP_HIDDEN = 128
MLP_OUT = 32
SMOKE = False
SMOKE_N_ASINS = 5

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HTML_TAG_RE = re.compile(r"<br\s*/?>")
_HTML_OTHER_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_MULTI_SPACE_RE = re.compile(r" {2,}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_html(text: str) -> str:
    if not text:
        return text
    previous = None
    while text != previous:
        previous = text
        text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _HTML_OTHER_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


class StyleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SVD_DIM, MLP_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(MLP_HIDDEN, MLP_OUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def multi_positive_nce(anchor, positives, negatives):
    pos_logits = torch.einsum("bd,bpd->bp", anchor, positives) / TEMPERATURE
    neg_logits = torch.einsum("bd,bnd->bn", anchor, negatives) / TEMPERATURE
    all_logits = torch.cat([pos_logits, neg_logits], dim=1)
    return -(torch.logsumexp(pos_logits, dim=1) - torch.logsumexp(all_logits, dim=1)).mean()


def load_real_pair_rows(uid_list: list[str], user_n_sents: list[int]):
    """Return true pair -> global sentence rows, aligned to sent_vectors user-major rows."""
    uid_to_idx = {uid: i for i, uid in enumerate(uid_list)}
    row_starts = np.zeros(len(user_n_sents) + 1, dtype=np.int64)
    row_starts[1:] = np.cumsum(user_n_sents)
    pair_local = defaultdict(list)
    observed_counts = np.zeros(len(uid_list), dtype=np.int64)
    n_reviews = 0
    n_sentences = 0
    skipped_external_reviews = 0
    external_users: set[str] = set()
    t0 = time.time()
    with open(RAW_DATA, "r") as f:
        for line in f:
            rec = json.loads(line)
            uid = rec["user_id"]
            asin = rec.get("parent_asin")
            if not asin:
                raise ValueError("Required parent_asin missing")
            u = uid_to_idx.get(uid)
            if u is None:
                skipped_external_reviews += 1
                external_users.add(uid)
                continue
            text = clean_html(rec.get("text", ""))
            sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
            start = int(observed_counts[u])
            rows = np.arange(row_starts[u] + start,
                             row_starts[u] + start + len(sentences), dtype=np.int64)
            pair_local[(u, asin)].extend(rows.tolist())
            observed_counts[u] += len(sentences)
            n_reviews += 1
            n_sentences += len(sentences)
            if n_reviews % 500000 == 0:
                log(f"  raw reviews={n_reviews} pairs={len(pair_local)} elapsed={time.time()-t0:.1f}s")
    expected = np.asarray(user_n_sents, dtype=np.int64)
    if not np.array_equal(observed_counts, expected):
        bad = np.flatnonzero(observed_counts != expected)
        raise ValueError(f"Sentence row alignment failed for {len(bad)} users")
    log(f"  raw aligned: reviews={n_reviews} sentences={n_sentences} pairs={len(pair_local)} "
        f"skipped_external_reviews={skipped_external_reviews} external_users={len(external_users)}")
    return pair_local, row_starts


def build_real_cohorts(pair_local):
    qualified_by_asin = defaultdict(list)
    for (u, asin), rows in pair_local.items():
        if len(rows) >= N_PAIR_SENTENCES:
            qualified_by_asin[asin].append(u)
    qualified_by_asin = {
        asin: sorted(set(users))
        for asin, users in qualified_by_asin.items()
        if len(set(users)) >= COHORT_USERS
    }
    selected = {
        asin: users[:COHORT_USERS]
        for asin, users in sorted(qualified_by_asin.items())
    }
    if not selected:
        raise ValueError("No real ASIN has three qualified users")
    return selected


def make_split(pair_local, cohorts):
    rng = np.random.default_rng(SEED)
    train_rows, val_rows, test_rows = [], [], []
    uid_train, uid_val, uid_test = [], [], []
    asin_train, asin_val, asin_test = [], [], []
    pair_train_rows = {}
    pair_val_rows = {}
    pair_test_rows = {}
    for asin, users in cohorts.items():
        for u in users:
            rows = np.asarray(pair_local[(u, asin)], dtype=np.int64)
            chosen = rng.choice(rows, size=N_PAIR_SENTENCES, replace=False)
            tr = chosen[:N_TRAIN]
            va = chosen[N_TRAIN:N_TRAIN + N_VAL]
            te = chosen[N_TRAIN + N_VAL:]
            pair_train_rows[(u, asin)] = tr
            pair_val_rows[(u, asin)] = va
            pair_test_rows[(u, asin)] = te
            train_rows.extend(tr.tolist()); uid_train.extend([u] * N_TRAIN); asin_train.extend([asin] * N_TRAIN)
            val_rows.extend(va.tolist()); uid_val.extend([u] * N_VAL); asin_val.extend([asin] * N_VAL)
            test_rows.extend(te.tolist()); uid_test.extend([u] * N_TEST); asin_test.extend([asin] * N_TEST)
    return (
        np.asarray(train_rows, dtype=np.int64), np.asarray(val_rows, dtype=np.int64), np.asarray(test_rows, dtype=np.int64),
        np.asarray(uid_train, dtype=np.int64), np.asarray(uid_val, dtype=np.int64), np.asarray(uid_test, dtype=np.int64),
        np.asarray(asin_train, dtype=object), np.asarray(asin_val, dtype=object), np.asarray(asin_test, dtype=object),
        pair_train_rows, pair_val_rows, pair_test_rows,
    )


def make_schedules(train_rows, uid_train, asin_train, pair_train_rows, cohorts, n_epochs):
    row_to_local = {int(r): i for i, r in enumerate(train_rows)}
    pair_local = {(u, a): np.asarray([row_to_local[int(r)] for r in rs], dtype=np.int64)
                  for (u, a), rs in pair_train_rows.items()}
    # Fixed 10 train sentence choices per negative user; exact k per user.
    neg_local = np.empty((len(train_rows), N_NEG), dtype=np.int64)
    pos_local = np.empty((n_epochs, len(train_rows), N_POS), dtype=np.int64)
    rng = np.random.default_rng(SEED + 1)
    asin_to_users = {a: users for a, users in cohorts.items()}
    for i in range(len(train_rows)):
        u = int(uid_train[i]); a = str(asin_train[i])
        own = pair_local[(u, a)]
        other_users = [v for v in asin_to_users[a] if v != u]
        if len(other_users) != COHORT_USERS - 1:
            raise ValueError(f"ASIN {a} cohort is not exactly {COHORT_USERS}")
        neg_parts = []
        for v in other_users:
            rows = pair_local[(v, a)]
            neg_parts.append(rng.choice(rows, size=K_PER_NEG_USER, replace=False))
        neg_local[i] = np.concatenate(neg_parts)
        for ep in range(n_epochs):
            candidates = own[own != i]
            if len(candidates) < N_POS:
                raise ValueError(f"Pair {(u, a)} has insufficient positive candidates")
            pos_local[ep, i] = rng.choice(candidates, size=N_POS, replace=False)
    return pos_local, neg_local


@torch.no_grad()
def eval_metrics(enc, z_all_t, test_rows, uid_test, asin_test, pair_train_rows, cohorts):
    z_test = enc(z_all_t[test_rows]).cpu().numpy()
    train_rows = np.concatenate(list(pair_train_rows.values()))
    z_train = enc(z_all_t[train_rows]).cpu().numpy()
    pair_proto = {}
    for pair, rows in pair_train_rows.items():
        # rows are global; recompute by direct batch map below
        emb = enc(z_all_t[np.asarray(rows, dtype=np.int64)]).cpu().numpy()
        m = emb.mean(axis=0); pair_proto[pair] = m / np.linalg.norm(m)
    ranks = []
    asin_to_users = cohorts
    for i, (u, a_obj) in enumerate(zip(uid_test, asin_test)):
        a = str(a_obj); u = int(u)
        users = asin_to_users[a]
        peers = [v for v in users if v != u]
        score_true = float(z_test[i] @ pair_proto[(u, a)])
        peer_scores = [float(z_test[i] @ pair_proto[(v, a)]) for v in peers]
        ranks.append(sum(s >= score_true for s in peer_scores))
    ranks = np.asarray(ranks)
    return {
        "n_evaluated": int(len(ranks)), "mean_rank": float(ranks.mean()),
        "top1_acc": float(np.mean(ranks == 0)), "top3_acc": float(np.mean(ranks < 3)),
        "random_baseline_top1": 1.0 / COHORT_USERS,
    }


def main():
    log(f"device={DEVICE}, real cohorts exact={COHORT_USERS}, k={K_PER_NEG_USER}, split={N_TRAIN}/{N_VAL}/{N_TEST}")
    with np.load(SVD_COMPONENTS) as npz:
        Vt = np.asarray(npz["Vt"], dtype=np.float32)
    d = np.load(SENT_VECTORS, allow_pickle=True)
    X = sp.csr_matrix((d["data"].astype(np.float32), d["indices"].astype(np.int32), d["indptr"].astype(np.int32)), shape=tuple(d["shape"]))
    with open(USER_N_SENTS) as f: user_n_sents = json.load(f)
    with open(UID_LIST) as f: uid_list = json.load(f)
    pair_local, row_starts = load_real_pair_rows(uid_list, user_n_sents)
    cohorts = build_real_cohorts(pair_local)
    if SMOKE:
        cohorts = dict(list(cohorts.items())[:SMOKE_N_ASINS])
        log(f"SMOKE=True → ASINs={len(cohorts)}")
    log(f"  selected exact-3 cohorts: ASINs={len(cohorts)} users={len(set(u for us in cohorts.values() for u in us))}")
    split = make_split(pair_local, cohorts)
    train_rows, val_rows, test_rows, uid_train, uid_val, uid_test, asin_train, asin_val, asin_test, pair_train, pair_val, pair_test = split
    log(f"  split rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    pos_local, neg_local = make_schedules(train_rows, uid_train, asin_train, pair_train, cohorts, N_EPOCHS)
    z_all = (X @ Vt.T).astype(np.float32); del X
    z_all_t = torch.from_numpy(z_all).to(DEVICE)
    z_train = z_all_t[train_rows]
    enc = StyleMLP().to(DEVICE); enc = torch.compile(enc)
    opt = torch.optim.Adam(enc.parameters(), lr=LR)
    for ep in range(N_EPOCHS):
        perm = np.random.default_rng(SEED + ep).permutation(len(train_rows))
        total = 0.0
        for s in range(0, len(train_rows), BATCH_SIZE):
            idx = perm[s:s+BATCH_SIZE]
            anchor = enc(z_train[idx])
            pos = enc(z_train[pos_local[ep, idx].reshape(-1)]).view(len(idx), N_POS, MLP_OUT)
            neg = enc(z_train[neg_local[idx].reshape(-1)]).view(len(idx), N_NEG, MLP_OUT)
            loss = multi_positive_nce(anchor, pos, neg)
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.item())
        log(f"  epoch {ep+1}/{N_EPOCHS} loss={total / ((len(train_rows)+BATCH_SIZE-1)//BATCH_SIZE):.4f}")
    result = eval_metrics(enc, z_all_t, test_rows, uid_test, asin_test, pair_train, cohorts)
    log(f"  test n={result['n_evaluated']} top1={result['top1_acc']*100:.2f}% mean_rank={result['mean_rank']:.3f}")
    out = OUT_DIR / "real_cohort3_training_summary.json"
    with open(out, "w") as f:
        json.dump({"config": {"cohort_users": COHORT_USERS, "k_per_negative_user": K_PER_NEG_USER, "pair_sentences": N_PAIR_SENTENCES, "split": f"{N_TRAIN}/{N_VAL}/{N_TEST}", "real_parent_asin": True}, "selected_asins": len(cohorts), "selected_users": len(set(u for us in cohorts.values() for u in us)), "result": result}, f, indent=2)
    log(f"DONE wrote {out}")
    # Save encoder weights for downstream Stage 4 reuse (per-user Gaussian on the same encoder)
    try:
        underlying = enc._orig_mod if hasattr(enc, "_orig_mod") else enc
        underlying.eval()
        ckpt = OUT_DIR / f"cohort{COHORT_USERS}_mlp{MLP_OUT}_{N_PAIR_SENTENCES}_{N_EPOCHS}ep.pt"
        torch.save(underlying.state_dict(), ckpt)
        log(f"DONE wrote encoder weights {ckpt}")
    except Exception as e:
        log(f"ERROR saving encoder weights: {type(e).__name__}: {e}")
        raise
    # Dump the cohort3-trained unique user ids (Stage 4 variant restricts
    # per-user Gaussian to this exact set so the MLP forward-pass does not
    # leak into the un-trained rest of strict3's 72k users).
    trained_uids_path = OUT_DIR / "cohort3_trained_uids.json"
    with open(trained_uids_path, "w") as f:
        json.dump(sorted({uid_list[int(u)] for u in uid_train}), f)
    log(f"DONE wrote trained uid whitelist {trained_uids_path} "
        f"({len({uid_list[int(u)] for u in uid_train})} unique uids)")


if __name__ == "__main__":
    main()
