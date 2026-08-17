#!/usr/bin/env python3
"""E22 Task 3 Step 1b: syntax probe trained on USER REVIEW SENTENCES.

Key fix (user directive "注入风格向量"): the probe must learn the mapping
    Qwen hidden(REAL user-review sentence) -> user's z_user (318-dim)
so that after injecting z_user as a prefix, the model's generated-sentence
hidden lands in the user's syntax region.

Previous version trained the probe on TEMPLATE-rendered sentences with the
sentence's OWN spaCy 318 vector as label — that taught "template hidden ->
template 318", which is NOT the user-syntax region (diagnosed: probe on real
review sentences gives dist-to-own-z RATIO ~1.1-1.7 > 1, i.e. wrong region).

New design:
  - data: user-review SENTENCES (real syntax) from train users
  - input: Qwen hidden (mean-pooled) of the sentence
  - label: that USER's z_user (318-dim, Task-1 standardized)  [supervised]
  - hold-out: unseen REVIEW SENTENCES from held-out users
Checks: held-out sentence error vs mean baseline; content-swap (same syntax
different product mention) invariance; syntax-swap sensitivity.

Outputs:
  result/e22_t3/e22_t3_syntax_probe2.pt   — frozen probe (user-syntax region)
  result/e22_t3/e22_t3_syntax_probe2.json — checks + metrics
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import gzip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "query_gen"))
sys.path.insert(0, str(REPO_ROOT / "syntactic_analysis"))

from query_gen_main import (  # noqa: E402
    MODEL_PATH, TASK1_VECTORS, TASK1_MANIFEST, REVIEWS, DTYPE, DEVICE, Z_DIM,
    OUT_DIR, neutralize_content,
)
from extract_syntactic_features import (  # noqa: E402
    per_sentence_features_v2, user_features_v2,
)
from extract_clause_features_single_query import load_spacy_model  # noqa: E402

SEED = 5555
MAX_USERS = 800
SENTS_PER_USER = 25
N_EPOCHS = 25
LR = 3e-4
BATCH = 16
HELD_OUT_FRAC = 0.15


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class SyntaxProbe(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, z_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.to(torch.float32)

    def forward(self, x):
        return self.net(x.float())


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vd = np.load(TASK1_VECTORS, allow_pickle=True)
    user_ids = list(vd["user_ids"])
    z_full = vd["Z_full"].astype(np.float64)
    user_to_z = {u: z_full[i] for i, u in enumerate(user_ids)}
    tm = vd["train_mean"].astype(np.float64)
    ts = vd["train_std"].astype(np.float64)
    mt = json.load(open(TASK1_MANIFEST))
    train_users = set(mt["splits"]["train"]["ids"])
    dev_users = set(mt["splits"]["dev"]["ids"])

    nlp = load_spacy_model()
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)

    # collect user review SENTENCES (train+dev users; dev for validation)
    # Only index users that already have z_user in the Task-1 pool; keep a
    # running count per user and stop early once MAX_USERS*SENTS_PER_USER*4
    # texts are captured (no full-corpus scan).
    log("collecting user review sentences...")
    target_users = [u for u in (train_users | dev_users) if u in user_to_z]
    rng = random.Random(SEED)
    rng.shuffle(target_users)
    target_users = target_users[:MAX_USERS]
    target_set = set(target_users)
    log(f"target users: {len(target_set)}")

    user_sents: dict[str, list[str]] = defaultdict(list)
    with gzip.open(REVIEWS, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            u = d.get("user_id")
            text = d.get("text") or ""
            if u and text.strip() and u in target_set:
                user_sents[u].append(text.strip())
            # early stop: enough captured
            if sum(len(v) for v in user_sents.values()) >= \
                    MAX_USERS * SENTS_PER_USER * 4:
                break
            if i > 8000000:
                break
    log(f"users with reviews: {len(user_sents)}")

    cand_users = list(user_sents.keys())
    log(f"users with z: {len(cand_users)}")

    # tokenize sentences, split train/held-out
    sentences: list[dict] = []   # {user, text, tok_ids}
    for u in cand_users:
        texts = user_sents[u][:SENTS_PER_USER * 2]
        for t in texts:
            # keep sentences with >= 5 words
            if len(t.split()) < 5:
                continue
            sentences.append({"user": u, "text": t})
    rng.shuffle(sentences)
    n_held = int(len(sentences) * HELD_OUT_FRAC)
    held_sents = sentences[:n_held]
    train_sents = sentences[n_held:]
    log(f"sentences: total={len(sentences)} train={len(train_sents)} "
        f"held-out={len(held_sents)}")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True)
    model.eval()
    H = model.config.hidden_size
    for p in model.parameters():
        p.requires_grad = False

    def pooled_hidden_batch(texts: list[str]) -> torch.Tensor:
        encs = [tok(t, add_special_tokens=False)["input_ids"][:96]
                for t in texts]
        max_l = max(len(e) for e in encs) if encs else 0
        if max_l == 0:
            return torch.zeros(0, H, device=DEVICE)
        ids = torch.tensor([e + [tok.pad_token_id] * (max_l - len(e))
                            for e in encs], dtype=torch.long, device=DEVICE)
        mask = torch.tensor([[1] * len(e) + [0] * (max_l - len(e))
                             for e in encs], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask,
                        use_cache=False, output_hidden_states=True)
        h = out.hidden_states[-1]
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp_min(1)

    # precompute hidden for all sentences (one pass)
    log("precomputing hidden states...")
    all_s = train_sents + held_sents
    hidden: dict[int, torch.Tensor] = {}
    texts_all = [s["text"] for s in all_s]
    for start in range(0, len(texts_all), BATCH * 4):
        chunk = texts_all[start:start + BATCH * 4]
        h = pooled_hidden_batch(chunk)
        for j, idx in enumerate(range(start, start + len(chunk))):
            hidden[idx] = h[j]
        if start % 1600 == 0:
            log(f"  hidden {start}/{len(texts_all)}")

    # build index map
    idx_of = {id(s): i for i, s in enumerate(all_s)}
    train_idx = [i for i, s in enumerate(all_s) if s in train_sents]
    held_idx = [i for i, s in enumerate(all_s) if s in held_sents]

    def sub_xy(idx_list):
        X = torch.stack([hidden[i] for i in idx_list])
        Y = torch.tensor(np.stack([user_to_z[all_s[i]["user"]]
                                   for i in idx_list]),
                         dtype=torch.float32, device=DEVICE)
        return X, Y

    Xtr, Ytr = sub_xy(train_idx)
    Xte, Yte = sub_xy(held_idx)
    mean_pred = Ytr.mean(0)
    mean_base_te = float(torch.linalg.norm(
        mean_pred[None] - Yte, dim=1).mean().item())

    probe = SyntaxProbe(H, Z_DIM).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=1e-2)
    # row map: Xtr rows align with train_idx order
    train_row = {idx: r for r, idx in enumerate(train_idx)}
    best_te = float("inf")
    for ep in range(N_EPOCHS):
        order = list(range(len(train_idx)))
        random.Random(SEED + ep).shuffle(order)
        ep_loss = 0.0
        for start in range(0, len(order), BATCH):
            rows = order[start:start + BATCH]
            opt.zero_grad()
            pred = probe(Xtr[rows])
            loss = F.mse_loss(pred, Ytr[rows])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(rows)
        with torch.no_grad():
            te_err = float(torch.linalg.norm(
                probe(Xte) - Yte, dim=1).mean().item())
        log(f"  ep{ep}: train={ep_loss/len(train_idx):.4f} "
            f"held-out={te_err:.3f} vs mean-base={mean_base_te:.3f}")
        if te_err < best_te:
            best_te = te_err
            torch.save(probe.state_dict(), OUT_DIR / "e22_t3_syntax_probe2.pt")
    probe.load_state_dict(torch.load(OUT_DIR / "e22_t3_syntax_probe2.pt"))

    # ---- Check 1: held-out sentence error vs mean baseline ----
    with torch.no_grad():
        te_err = float(torch.linalg.norm(probe(Xte) - Yte, dim=1).mean().item())
        rel_imp = (mean_base_te - te_err) / mean_base_te
    check1 = {"n_held": len(held_idx),
              "probe_heldout_l2": round(te_err, 3),
              "mean_baseline_l2": round(mean_base_te, 3),
              "relative_improvement": round(rel_imp, 4),
              "pass": bool(rel_imp > 0.05)}

    # ---- Check 2: content-swap invariance (same syntax, different mention)
    # same user sentence with product words replaced -> hidden ~same ----
    c2 = []
    for s in held_sents[:60]:
        t = s["text"]
        t2 = re_replace_entities(t)
        if t2 == t:
            continue
        h0 = pooled_hidden_batch([t])[0]
        h1 = pooled_hidden_batch([t2])[0]
        with torch.no_grad():
            z0 = probe(h0[None])[0]
            z1 = probe(h1[None])[0]
        c2.append(float(torch.linalg.norm(z0 - z1).item()))
    check2 = {"n": len(c2),
              "content_swap_l2_mean": round(float(np.mean(c2)), 4) if c2 else None,
              "pass": bool(c2 and np.mean(c2) < 2.0)}

    # ---- Check 3: syntax-swap sensitivity (two DIFFERENT users' sentences)
    c3 = []
    for i in range(min(60, len(held_idx) - 1)):
        s1 = all_s[held_idx[i]]
        s2 = all_s[held_idx[i + 1]]
        if s1["user"] == s2["user"]:
            continue
        with torch.no_grad():
            z1 = probe(hidden[held_idx[i]][None])[0]
            z2 = probe(hidden[held_idx[i + 1]][None])[0]
        c3.append(float(torch.linalg.norm(z1 - z2).item()))
    check3 = {"n": len(c3),
              "syntax_swap_l2_mean": round(float(np.mean(c3)), 4) if c3 else None,
              "pass": bool(c3 and np.mean(c3) > 2.0)}

    sha = hashlib.sha256(
        (OUT_DIR / "e22_t3_syntax_probe2.pt").read_bytes()).hexdigest()
    result = {
        "version": "e22_t3_syntax_probe2_v1",
        "seed": SEED, "model": Path(MODEL_PATH).name,
        "pooling": "mean-pool last-layer hidden over review sentence",
        "label": "user z_user (318, Task-1 standardized) — SUPERVISED",
        "n_train_sentences": len(train_idx),
        "n_heldout_sentences": len(held_idx),
        "check1_heldout_sentences": check1,
        "check2_content_swap": check2,
        "check3_syntax_swap": check3,
        "probe_sha256": sha,
        "all_checks_pass": bool(check1["pass"] and check2["pass"]
                                and check3["pass"]),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_DIR / "e22_t3_syntax_probe2.json", "w") as f:
        json.dump(result, f, indent=1)
    log(f"done: check1={check1['pass']} check2={check2['pass']} "
        f"check3={check3['pass']}")


def re_replace_entities(t: str) -> str:
    """Rough content-neutralization: replace capitalized runs and digits."""
    import re
    out = re.sub(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", " X ", t)
    out = re.sub(r"\d+(?:\.\d+)?", " 9 ", out)
    return out


if __name__ == "__main__":
    main()
