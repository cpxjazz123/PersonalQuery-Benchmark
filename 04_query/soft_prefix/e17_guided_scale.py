#!/usr/bin/env python3
"""Scaled significance test of style-target-guided decoding.

100 users x up to 5 products x 8 sampled candidates (batched generation);
selection by z-scored strong-dim distance to the user syntax center.
Significance: permutation test on intra<inter ratio + classification vs
random (permuted labels), N=199 permutations.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = "/fs04/ar57/wenyu/PersoanlQuery"
sys.path.insert(0, REPO_ROOT + "/04_query/soft_prefix")
sys.path.insert(0, REPO_ROOT + "/10_complexity_analysis/common")

from copy_aware import build_attr_prompt_lines  # noqa: E402
from copy_aware_generate import CopyAwareGenerator  # noqa: E402
from user_stat_vector import FEATURES20, build_user_stat_vectors  # noqa: E402
from extract_clause_features_single_query import load_spacy_model, extract_clause_features  # noqa: E402

CKPT = "/home/wlia0047/hj82_scratch2/wenyu/RAG/e14p_sv/checkpoint"
DATASET = REPO_ROOT + "/result/personal_query/e14_multiproduct/baby_dataset.json"
SEED = 42
N_SAMPLES = 8
N_USERS = 100
MAX_PRODUCTS = 5
BATCH = 16
STRONG = ["compound_count", "amod_count", "advmod_count", "relcl_count", "coordination_count"]


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
                               dtype=torch.bfloat16, gate_init=0.05).to("cuda:0")
    proj.load_state_dict(torch.load(CKPT + "/projector.pt", map_location="cuda:0"))
    from copy_aware import CopyAwareHead
    head = CopyAwareHead(cfg["model_dim"], base.get_output_embeddings().weight.size(0),
                         dtype=torch.bfloat16).to("cuda:0")
    head.load_state_dict(torch.load(CKPT + "/copy_head.pt", map_location="cuda:0"))
    gen = CopyAwareGenerator(base, proj, head, "cuda:0")
    nlp = load_spacy_model()

    ds = json.load(open(DATASET))
    users = sorted({r["user_id"] for r in ds})
    chosen = list(rng.choice(users, size=min(N_USERS, len(users)), replace=False))
    stat_vectors = build_user_stat_vectors("Baby_Products", chosen)
    user_center = {u: v[:20] for u, v in stat_vectors.items()}

    by_user = defaultdict(list)
    for r in ds:
        if r["user_id"] in chosen and len(by_user[r["user_id"]]) < MAX_PRODUCTS:
            by_user[r["user_id"]].append(r)

    centers = np.array([user_center[u] for u in chosen])
    mu_c, sd_c = centers.mean(axis=0), centers.std(axis=0) + 1e-9

    # 1) generate candidates (batched, sampled)
    jobs = [(u, rec) for u in chosen for rec in by_user[u] for _ in range(N_SAMPLES)]
    cands = defaultdict(list)
    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i:i + BATCH]
        prompts, attrs_list, vecs = [], [], []
        for u, rec in chunk:
            attrs = rec["attrs"]
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": "You are a shopping query writer. Write one short natural shopping query that mentions every listed attribute."},
                 {"role": "user", "content": build_attr_prompt_lines(attrs)}],
                tokenize=False, add_generation_prompt=True))
            attrs_list.append(attrs)
            vecs.append(stat_vectors[u][:30])
        qs = gen.generate_batch(prompts, attrs_list, vecs, tok, 32, tok.eos_token_id,
                                do_sample=True, temperature=1.2, top_k=50)
        for (u, rec), q in zip(chunk, qs):
            if q:
                cands[(u, rec["asin"])].append(q)
    torch.cuda.empty_cache()

    # 2) strong-dim selection with per-product z-score
    def feats(q):
        f = extract_clause_features(q)
        return np.asarray([float(f.get(k, 0.0)) for k in STRONG], dtype=np.float32)

    selected = {}
    for key, qs in cands.items():
        u = key[0]
        if len(qs) < 2:
            selected[key] = qs[0]
            continue
        F = np.array([feats(q) for q in qs])
        sd = F.std(axis=0) + 1e-9
        center = (user_center[u] - mu_c)[[FEATURES20.index(k) for k in STRONG]] / \
            sd_c[[FEATURES20.index(k) for k in STRONG]]
        z = (F - F.mean(axis=0)) / sd
        best = int(np.argmin(np.linalg.norm(z - center, axis=1)))
        selected[key] = qs[best]

    # 3) intra/inter distances on STRONG dims
    feat_by_user = defaultdict(list)
    for (u, asin), q in selected.items():
        feat_by_user[u].append(feats(q))
    intra, inter = [], []
    for u in feat_by_user:
        f = feat_by_user[u]
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                intra.append(np.linalg.norm(f[i] - f[j]))
    ul = list(feat_by_user)
    for a in ul:
        others = [x for x in ul if x != a]
        for o in rng.choice(others, min(3, len(others)), replace=False):
            inter.append(np.linalg.norm(feat_by_user[a][0] - feat_by_user[o][0]))
    intra = np.array(intra); inter = np.array(inter)
    ratio = inter.mean() / max(intra.mean(), 1e-9)
    print(f"users={len(ul)} pairs intra={len(intra)} inter={len(inter)}")
    print(f"intra={intra.mean():.3f} inter={inter.mean():.3f} ratio={ratio:.3f}")

    # permutation test: intra vs inter (labels permuted)
    stat_obs = inter.mean() - intra.mean()
    all_d = np.concatenate([intra, inter])
    rngp = np.random.default_rng(0)
    cnt = 0
    for _ in range(199):
        perm = rngp.permutation(len(all_d))
        d_intra, d_inter = all_d[perm[:len(intra)]], all_d[perm[len(intra):]]
        if d_inter.mean() - d_intra.mean() >= stat_obs:
            cnt += 1
    p_intra = (cnt + 1) / 200
    print(f"permutation p(intra<inter) = {p_intra:.3f}")

    # 4) classification vs random (permuted labels)
    X = np.array([feats(q) for q in selected.values()])
    y = np.array([k[0] for k in selected])
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder(); yl = le.fit_transform(y)
    n_cls = len(le.classes_)
    skf = StratifiedKFold(n_splits=min(3, n_cls), shuffle=True, random_state=0)
    acc_obs = cross_val_score(LogisticRegression(max_iter=2000), X, yl, cv=skf).mean()
    rngp2 = np.random.default_rng(1)
    cnt2 = 0
    for _ in range(199):
        yp = yl.copy()
        rngp2.shuffle(yp)
        acc_p = cross_val_score(LogisticRegression(max_iter=2000), X, yp, cv=skf).mean()
        if acc_p >= acc_obs:
            cnt2 += 1
    p_acc = (cnt2 + 1) / 200
    print(f"classification: acc={acc_obs:.3f} (random={1/n_cls:.3f}) p={p_acc:.3f}")
    json.dump({"users": len(ul), "intra": float(intra.mean()), "inter": float(inter.mean()),
               "ratio": float(ratio), "p_intra_lt_inter": p_intra,
               "acc": float(acc_obs), "random": 1 / n_cls, "p_acc": p_acc,
               "n_samples": len(selected)},
              open(REPO_ROOT + "/result/personal_query/e17_guided_scale.json", "w"), indent=1)


if __name__ == "__main__":
    main()
