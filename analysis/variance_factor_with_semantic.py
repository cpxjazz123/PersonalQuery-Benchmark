"""
User σ²_u multi-factor regression v2: add length std, semantic diversity, ASIN/category diversity.

Inputs:
  - /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/all_doc_rules_5000.pkl (rules per sentence)
  - /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/sent_vectors.npz (vocab CSR)
  - /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/user_n_sents.json
  - /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/uid_list.json
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/02_user_review_sentence_extract/uid_to_sentences.pkl
      (dict[uid] -> list of raw sentences, full corpus)
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/data/Baby_Products_2023.jsonl (raw reviews)
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/data/meta_Baby_Products_2023.jsonl (product categories)

Per-user features:
  v1 (from cache):
    - n_u, unique_rules, rule_entropy, rules_per_sent_mean, rules_per_sent_std,
      vocab_unique, vocab_entropy
  v2 (new):
    - len_mean, len_std (token-length mean + std per user)
    - semantic_var (per-user variance of MPNet 768d sentence embeddings)
    - asin_unique, category_unique (product diversity)

Output:
  /home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/syntax_pcfg_variance_factors_v2.json
"""

import gzip
import json
import os
import pickle
import re
from collections import Counter

import numpy as np
from scipy import sparse, stats

CACHE = "/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache"
DATA_DIR = "/home/wlia0047/ar57/wenyu/PersoanlQuery/data"
ABL_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/syntax_pcfg_ablation.json"
OUT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/syntax_pcfg_variance_factors_v2.json"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
LEN_FILTER = lambda s: 5 < len(s.split()) < 50


def shannon_entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    abl = json.load(open(ABL_PATH))
    var_per_user = abl["user_variance"]["per_user"]
    uid_list = json.load(open(f"{CACHE}/uid_list.json"))
    n_sents_cache = json.load(open(f"{CACHE}/user_n_sents.json"))
    assert list(var_per_user.keys()) == uid_list

    SMOKE = os.environ.get("SMOKE", "0") == "1"
    if SMOKE:
        uid_list = uid_list[:50]
        n_sents_cache = n_sents_cache[:50]
        var_per_user = {u: var_per_user[u] for u in uid_list}

    # ----- load per-sentence rules (816023 flat) -----
    with open(f"{CACHE}/all_doc_rules_5000.pkl", "rb") as f:
        rules_data = pickle.load(f)
    rules_flat = rules_data["rules"]
    assert len(rules_flat) == 816023

    # ----- load raw sentences (filtered 5<tokens<50) per user, in same order as cache -----
    with open("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/02_user_review_sentence_extract/uid_to_sentences.pkl", "rb") as f:
        uid_to_sents_raw = pickle.load(f)
    filtered_sents = {}  # uid -> list of filtered sentences (matches cache order)
    for uid in uid_list:
        sents = [s for s in uid_to_sents_raw[uid] if LEN_FILTER(s)]
        filtered_sents[uid] = sents
    # verify per-user counts match cache
    for i, uid in enumerate(uid_list):
        if len(filtered_sents[uid]) != n_sents_cache[i]:
            raise ValueError(
                f"count mismatch {uid}: filtered={len(filtered_sents[uid])} cache={n_sents_cache[i]}"
            )

    # ----- per-user features v1 from rules + CSR vocab -----
    sv = sparse.load_npz(f"{CACHE}/sent_vectors.npz")
    offsets = np.concatenate([[0], np.cumsum(n_sents_cache)])
    feats = {}
    for i, uid in enumerate(uid_list):
        s, e = offsets[i], offsets[i + 1]
        rule_counts = Counter()
        per_sent_rule_counts = np.zeros(e - s, dtype=np.float32)
        for j, rs in enumerate(rules_flat[s:e]):
            per_sent_rule_counts[j] = len(rs)
            for r in rs:
                rule_counts[r] += 1
        rule_arr = np.array(list(rule_counts.values()), dtype=np.float64)
        feats[uid] = {
            "n_u": int(e - s),
            "unique_rules": int(rule_arr.size),
            "rule_entropy": shannon_entropy(rule_arr) if rule_arr.size else 0.0,
            "rules_per_sent_mean": float(per_sent_rule_counts.mean()),
            "rules_per_sent_std": float(per_sent_rule_counts.std()),
        }
        sub = sv[s:e].tocoo()
        if sub.nnz:
            vocab_ids = np.unique(sub.col)
            feats[uid]["vocab_unique"] = int(vocab_ids.size)
            vocab_counts = np.asarray(sv[s:e].sum(axis=0)).ravel()
            vocab_counts = vocab_counts[vocab_counts > 0]
            feats[uid]["vocab_entropy"] = shannon_entropy(vocab_counts)
        else:
            feats[uid]["vocab_unique"] = 0
            feats[uid]["vocab_entropy"] = 0.0
        if (i + 1) % 500 == 0:
            print(f"  feat v1 {i+1}/5000", flush=True)

    # ----- per-user features v2: length mean + std -----
    for uid in uid_list:
        sents = filtered_sents[uid]
        lens = np.array([len(s.split()) for s in sents], dtype=np.float64)
        feats[uid]["len_mean"] = float(lens.mean())
        feats[uid]["len_std"] = float(lens.std())
    print("  feat len done", flush=True)

    # ----- per-user asin/category diversity via re-extraction from raw jsonl -----
    print("  scanning raw jsonl for parent_asin...", flush=True)
    uid_to_asins = {uid: [] for uid in uid_list}
    target_uid_set = set(uid_list)
    raw_path = f"{DATA_DIR}/Baby_Products_2023.jsonl"
    n_lines = 0
    with open(raw_path, "r") as f:
        for line in f:
            n_lines += 1
            if n_lines % 500000 == 0:
                print(f"    {n_lines} lines, {sum(len(v) for v in uid_to_asins.values())} asin refs",
                      flush=True)
            e = json.loads(line)
            uid = e["user_id"]
            if uid not in target_uid_set:
                continue
            asin = e.get("parent_asin")
            if not asin:
                raise ValueError(f"asin missing for uid={uid}")
            # number of sentences from this review that pass filter = number of split sents 5<n.tok<50
            text = e.get("text", "")
            sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
            kept = sum(1 for s in sents if LEN_FILTER(s))
            uid_to_asins[uid].extend([asin] * kept)
    print(f"  jsonl scanned: {n_lines} lines", flush=True)
    # verify per-user
    for uid in uid_list:
        if len(uid_to_asins[uid]) != n_sents_cache[uid_list.index(uid)]:
            raise ValueError(
                f"asin count mismatch {uid}: {len(uid_to_asins[uid])} vs {n_sents_cache[uid_list.index(uid)]}"
            )

    # ----- load meta jsonl for parent_asin -> category -----
    print("  loading meta_Baby_Products for parent_asin->category...", flush=True)
    asin_to_cat = {}
    meta_path = f"{DATA_DIR}/meta_Baby_Products_2023.jsonl"
    with open(meta_path, "r") as f:
        for line in f:
            e = json.loads(line)
            asin = e.get("parent_asin")
            cat = e.get("main_category")
            if asin and cat is not None:
                asin_to_cat[asin] = cat
    print(f"  meta entries: {len(asin_to_cat)}", flush=True)

    for uid in uid_list:
        asins = uid_to_asins[uid]
        feats[uid]["asin_unique"] = int(len(set(asins)))
        cats = [asin_to_cat[a] for a in asins if a in asin_to_cat]
        feats[uid]["category_unique"] = int(len(set(cats))) if cats else 0
        if cats:
            cat_counts = np.array(list(Counter(cats).values()), dtype=np.float64)
            feats[uid]["category_entropy"] = shannon_entropy(cat_counts)
        else:
            feats[uid]["category_entropy"] = 0.0
    print("  feat asin/cat done", flush=True)

    # ----- MPNet semantic embeddings per sentence -----
    print("  loading MPNet model + encoding all 816K sentences...", flush=True)
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        cache_folder="/home/wlia0047/hj82_scratch2/wenyu/hf_cache",
        device=device,
    )
    # flatten 816K sentences in cache order
    flat_sents = []
    for uid in uid_list:
        flat_sents.extend(filtered_sents[uid])
    assert len(flat_sents) == sum(n_sents_cache)

    model.max_seq_length = 128
    emb = model.encode(
        flat_sents,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    print(f"  emb shape: {emb.shape}", flush=True)

    # ----- per-user semantic variance -----
    emb_offset = 0
    for i, uid in enumerate(uid_list):
        n = n_sents_cache[i]
        sub = emb[emb_offset:emb_offset + n]
        emb_offset += n
        # trace of covariance
        if n > 1:
            mu = sub.mean(axis=0)
            cov = ((sub - mu).T @ (sub - mu)) / max(n - 1, 1)
            tr = float(np.trace(cov))
            # also normalized to dim
            feats[uid]["semantic_var"] = tr
        else:
            feats[uid]["semantic_var"] = 0.0
    print("  feat semantic done", flush=True)

    # ----- assemble arrays -----
    feat_names = [
        "n_u", "unique_rules", "rule_entropy", "rules_per_sent_mean",
        "rules_per_sent_std", "vocab_unique", "vocab_entropy",
        "len_mean", "len_std", "asin_unique", "category_unique",
        "category_entropy", "semantic_var",
    ]
    X = np.array([[feats[u][k] for k in feat_names] for u in uid_list], dtype=np.float64)
    y = np.array([var_per_user[u] for u in uid_list], dtype=np.float64)

    log_idx = [0, 1, 5, 9, 10, 12]  # n_u, unique_rules, vocab_unique, asin_unique, category_unique, semantic_var
    X_log = X.copy()
    X_log[:, log_idx] = np.log1p(X_log[:, log_idx])

    # ----- univariate correlations -----
    print("\n=== Univariate correlations with σ²_u ===")
    univariate = {}
    for j, name in enumerate(feat_names):
        col = X[:, j]
        col_log = X_log[:, j]
        rp, pp = stats.pearsonr(col, y)
        rs, ps = stats.spearmanr(col, y)
        rpl, ppl = stats.pearsonr(col_log, y)
        univariate[name] = {
            "pearson_r": float(rp), "pearson_p": float(pp),
            "spearman_r": float(rs), "spearman_p": float(ps),
            "log_pearson_r": float(rpl), "log_pearson_p": float(ppl),
        }
        print(f"  {name:<22} Pearson={rp:+.4f} (p={pp:.1e}) "
              f"Spearman={rs:+.4f} (p={ps:.1e}) "
              f"logPearson={rpl:+.4f}")

    # ----- partial correlation controlling for log n_u, log semantic_var, log asin_unique -----
    print("\n=== Partial correlations ===")
    ctrl_idx = [0, 9, 12]  # n_u, asin_unique, semantic_var
    ctrl = X_log[:, ctrl_idx]
    A_ctrl = np.column_stack([np.ones_like(ctrl[:, 0]), ctrl])

    def residualize(target, A):
        b, *_ = np.linalg.lstsq(A, target, rcond=None)
        return target - A @ b

    partial = {}
    for j, name in enumerate(feat_names):
        col = X_log[:, j]
        r_x = residualize(col, A_ctrl)
        r_y = residualize(y, A_ctrl)
        rp, pp = stats.pearsonr(r_x, r_y)
        partial[name] = {"partial_r": float(rp), "partial_p": float(pp)}
        print(f"  {name:<22} partial_r={rp:+.4f} p={pp:.1e}")

    # ----- OLS multiple regression (drop collinear mirrors) -----
    # Drop vocab_unique, vocab_entropy, category_entropy (mirrors of unique_rules / rule_entropy / asin_unique)
    keep = [
        "n_u", "rule_entropy", "unique_rules", "rules_per_sent_mean", "rules_per_sent_std",
        "len_mean", "len_std", "asin_unique", "category_unique",
        "semantic_var",
    ]
    keep_idx = [feat_names.index(k) for k in keep]
    X_keep = X_log[:, keep_idx]

    print("\n=== OLS regression (10 features) ===")
    A = np.column_stack([np.ones(len(y)), X_keep])
    from numpy.linalg import lstsq
    coef, *_ = lstsq(A, y, rcond=None)
    y_hat = A @ coef
    r2 = 1.0 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    n_obs, k = A.shape
    adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - k - 1)
    resid = y - y_hat
    mse = (resid ** 2).mean()
    XtX_inv = np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(XtX_inv) * mse)
    t_stat = coef / se
    p_ols = 2 * stats.t.sf(np.abs(t_stat), df=n_obs - k)
    ols = {}
    print(f"  R²={r2:.4f}  Adj R²={adj_r2:.4f}  n={n_obs}")
    for j, name in enumerate(["intercept"] + keep):
        ols[name] = {
            "coef": float(coef[j]),
            "se": float(se[j]),
            "t": float(t_stat[j]),
            "p": float(p_ols[j]),
            "std_coef": float(coef[j] * X_keep[:, j-1].std() / y.std()) if j > 0 else None,
        }
        print(f"  {name:<22} β={coef[j]:+.4f}  se={se[j]:.4f}  t={t_stat[j]:+.2f}  "
              f"p={p_ols[j]:.1e}  stdβ={ols[name]['std_coef']}")

    # ----- VIF -----
    print("\n=== VIF ===")
    vif = {}
    Z = X_keep - X_keep.mean(axis=0)
    for j, name in enumerate(keep):
        others = np.delete(Z, j, axis=1)
        b, *_ = lstsq(others, Z[:, j], rcond=None)
        r2_j = 1.0 - ((Z[:, j] - others @ b) ** 2).sum() / (Z[:, j] ** 2).sum()
        v = 1.0 / max(1.0 - r2_j, 1e-12)
        vif[name] = float(v)
        print(f"  {name:<22} VIF={v:.2f}")

    # ----- save -----
    out = {
        "config": {
            "n_users": 5000,
            "n_features": len(feat_names),
            "features": feat_names,
            "log_transformed": [feat_names[i] for i in log_idx],
            "encoder": "sentence-transformers/all-mpnet-base-v2 (768d)",
            "len_filter": "5 < tokens < 50",
        },
        "target": {
            "name": "sigma2_u",
            "median": float(np.median(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
        },
        "univariate": univariate,
        "partial_controlling_log_n_u_asin_semantic": partial,
        "ols": {
            "R2": float(r2),
            "adj_R2": float(adj_r2),
            "n_obs": int(n_obs),
            "n_features": int(k - 1),
            "features_kept": keep,
            "coefficients": ols,
            "vif": vif,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nwrote → {OUT_PATH}")


if __name__ == "__main__":
    main()