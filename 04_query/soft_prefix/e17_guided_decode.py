#!/usr/bin/env python3
"""Style-target-guided decoding: sample N candidates per (user, product),
score by distance to the user's syntactic-statistics center, pick the best.

This avoids the missing-supervision problem entirely: the selection signal is
the user's own review syntax statistics (deterministic, no training signal
needed). Content fidelity stays with copy-aware generation; the selection
only re-ranks syntax.

Protocol: 30 users x 3 products x 10 samples -> pick per user center;
measure intra-user syntactic distance vs inter-user, and user classification
on the SELECTED outputs.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT / "04_query" / "soft_prefix"))
sys.path.insert(0, str(REPO_ROOT / "10_complexity_analysis" / "common"))

from copy_aware import build_attr_prompt_lines  # noqa: E402
from copy_aware_generate import CopyAwareGenerator  # noqa: E402
from user_stat_vector import FEATURES20, build_user_stat_vectors  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features  # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter

CKPT = "/home/wlia0047/hj82_scratch2/wenyu/RAG/e14p_sv/checkpoint"
DATASET = REPO_ROOT / "result" / "personal_query" / "e14_multiproduct" / "baby_dataset.json"
SEED = 42
N_SAMPLES = 10
N_USERS = 30
PRODUCTS_PER_USER = 3


def main() -> None:
    rng = np.random.default_rng(SEED)
    cfg = json.load(open(CKPT + "/config.json"))
    tok = AutoTokenizer.from_pretrained(CKPT + "/tokenizer", trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model_path"], torch_dtype=torch.bfloat16,
                                                device_map="cuda:0", trust_remote_code=True)
    from peft import PeftModel
    base = PeftModel.from_pretrained(base, CKPT + "/lora_adapter").eval()
    from projector import SoftPrefixProjector
    proj = SoftPrefixProjector(cfg["user_dim"], 128, cfg["num_tokens"], cfg["model_dim"],
                               dtype=torch.bfloat16, gate_init=cfg.get("gate_init", 1e-3)).to("cuda:0")
    proj.load_state_dict(torch.load(CKPT + "/projector.pt", map_location="cuda:0"))
    from copy_aware import CopyAwareHead
    head = CopyAwareHead(cfg["model_dim"], base.get_output_embeddings().weight.size(0),
                         dtype=torch.bfloat16).to("cuda:0")
    head.load_state_dict(torch.load(CKPT + "/copy_head.pt", map_location="cuda:0"))
    gen = CopyAwareGenerator(base, proj, head, "cuda:0")

    ds = json.load(open(DATASET))
    users = sorted({r["user_id"] for r in ds})
    chosen = list(rng.choice(users, size=min(N_USERS, len(users)), replace=False))
    stat_vectors = build_user_stat_vectors("Baby_Products", chosen)
    # user syntax centers: 20-dim feature means (from stat vectors, first 20)
    user_center = {u: v[:20] for u, v in stat_vectors.items()}
    nlp = load_spacy_model()

    # sample products per user
    user_products = defaultdict(list)
    for r in ds:
        if r["user_id"] in chosen and len(user_products[r["user_id"]]) < PRODUCTS_PER_USER:
            user_products[r["user_id"]].append(r)

    STRONG = ["compound_count", "amod_count", "advmod_count", "relcl_count", "coordination_count"]
    # z-score: standardize each space (user centers vs candidate distribution)
    centers = np.array([user_center[u] for u in chosen])           # [U, 20]
    mu_c = centers.mean(axis=0); sd_c = centers.std(axis=0) + 1e-9
    cands_buf: dict = {}
    for u in chosen:
        center = (user_center[u] - mu_c) / sd_c
        for rec in user_products[u]:
            attrs = rec["attrs"]
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": "You are a shopping query writer. Write one short natural shopping query that mentions every listed attribute."},
                 {"role": "user", "content": build_attr_prompt_lines(attrs)}],
                tokenize=False, add_generation_prompt=True)
            vec = stat_vectors[u][:30]
            for _ in range(N_SAMPLES):
                with torch.no_grad():
                    q = gen.generate(prompt, vec, attrs, tok, 32, tok.eos_token_id,
                                     do_sample=True, temperature=1.2, top_k=50)
                f = extract_clause_features(q)
                fv = np.asarray([float(f.get(k, 0.0)) for k in FEATURES20], dtype=np.float32)
                cands_buf.setdefault((u, rec["asin"]), []).append((fv, q))
    # per-product z-score over candidates, then strong-dim weighted distance
    selected = {}
    for key, cands in cands_buf.items():
        F = np.array([c[0] for c in cands])
        sd = F.std(axis=0) + 1e-9
        u = key[0]
        center = (user_center[u] - mu_c) / sd_c
        best_q, best_d = None, 1e18
        for fv, q in cands:
            z = (fv - F.mean(axis=0)) / sd
            d = float(np.linalg.norm(z[[FEATURES20.index(k) for k in STRONG]] -
                                     center[[FEATURES20.index(k) for k in STRONG]]))
            if d < best_d:
                best_d, best_q = d, q
        selected[key] = best_q

    # features of selected outputs
    X, y = [], []
    for (u, asin), q in selected.items():
        f = extract_clause_features(q)
        X.append([float(f.get(k, 0.0)) for k in STRONG])
        y.append(u)
    # intra-user vs inter-user distances
    same, diff = [], []
    by_user = defaultdict(list)
    for (u, asin), q in selected.items():
        f = extract_clause_features(q)
        by_user[u].append(np.asarray([float(f.get(k, 0.0)) for k in STRONG]))
    print(f"selection dims: {STRONG}")
    us = list(by_user)
    for u in us:
        for i in range(len(by_user[u])):
            for j in range(i + 1, len(by_user[u])):
                same.append(np.linalg.norm(by_user[u][i] - by_user[u][j]))
    for u in us:
        for o in rng.choice([x for x in us if x != u], min(3, len(us) - 1), replace=False):
            diff.append(np.linalg.norm(by_user[u][0] - by_user[o][0]))
    print(f"intra={np.mean(same):.3f} inter={np.mean(diff):.3f} ratio={np.mean(diff)/max(np.mean(same),1e-9):.2f}")
    cnt = Counter(y)
    sel = [u for u, c in cnt.items() if c >= 2]
    idx = [i for i, u in enumerate(y) if u in set(sel)]
    if len(sel) >= 2:
        le = LabelEncoder(); yl = le.fit_transform(np.array(y)[idx])
        skf = StratifiedKFold(n_splits=min(3, len(sel)), shuffle=True, random_state=0)
        acc = cross_val_score(LogisticRegression(max_iter=2000), np.array(X)[idx], yl, cv=skf)
        print(f"classification: acc={acc.mean():.3f} | random={1/len(sel):.3f} | users={len(sel)}")
    print(f"samples: {len(selected)} (users={len(chosen)}, products={PRODUCTS_PER_USER}/user, N={N_SAMPLES})")


if __name__ == "__main__":
    main()
