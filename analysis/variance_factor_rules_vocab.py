"""
User σ²_u multi-factor regression analysis.

Goal: identify what predicts per-user variance of supervised 32d z embeddings.

Inputs (read-only):
  - /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/all_doc_rules_5000.pkl
      keys: 'rules' (list[list[str]] of length 816023)
  - /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/sent_vectors.npz
      CSR (816023, 21737) vocab one-hot
  - /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/user_n_sents.json
      list of per-user sentence counts (5000)
  - /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/uid_list.json
      list of 5000 user IDs
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/syntax_pcfg_ablation.json
      user_variance.per_user (target: σ²_u)

Per-user features (6):
  1. n_u          : profile sentence count
  2. unique_rules : |{rule string}| for user
  3. rule_entropy : Shannon entropy of rule distribution (over unique rules)
  4. rules_per_sent_mean : mean(|rules per sentence|) — structural complexity proxy
  5. rules_per_sent_std  : std(|rules per sentence|) — structural variation
  6. vocab_unique : |{non-zero vocab ids}| for user (lexical diversity)
  7. vocab_entropy: Shannon entropy of vocab distribution

Output:
  - /home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/syntax_pcfg_variance_factors.json
    correlations (Pearson / Spearman) and OLS coefficients + partial correlations.
"""

import json
import os
import pickle
from collections import Counter

import numpy as np
from scipy import sparse, stats

CACHE = "/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache"
ABL_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/syntax_pcfg_ablation.json"
OUT_PATH = "/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/syntax_pcfg_variance_factors.json"


def shannon_entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    # ----- target: per-user σ²_u -----
    abl = json.load(open(ABL_PATH))
    var_per_user = abl["user_variance"]["per_user"]
    uid_list = json.load(open(f"{CACHE}/uid_list.json"))
    n_sents = json.load(open(f"{CACHE}/user_n_sents.json"))
    assert len(uid_list) == 5000 and len(n_sents) == 5000
    assert list(var_per_user.keys()) == uid_list

    # ----- per-sentence rules (list[list[str]] length 816023) -----
    with open(f"{CACHE}/all_doc_rules_5000.pkl", "rb") as f:
        rules_data = pickle.load(f)
    rules_flat = rules_data["rules"]
    assert len(rules_flat) == 816023, f"got {len(rules_flat)}"

    # ----- per-sentence vocab CSR (816023, 21737) -----
    sv = sparse.load_npz(f"{CACHE}/sent_vectors.npz")

    # ----- per-user offsets -----
    offsets = np.concatenate([[0], np.cumsum(n_sents)])
    assert offsets[-1] == 816023, f"offset sum mismatch: {offsets[-1]}"

    # ----- per-user features -----
    feats = {}
    for i, uid in enumerate(uid_list):
        s, e = offsets[i], offsets[i + 1]
        # syntactic: rule features
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
            # lexical: vocab CSR row slice
            "vocab_unique": int((sv[s:e].getnnz(axis=1) > 0).sum()),  # sent with any vocab
            # placeholder, real vocab_unique is total distinct vocab ids in user
        }
        # vocab features: distinct word IDs in user profile
        sub = sv[s:e]
        sub_coo = sub.tocoo()
        if sub_coo.nnz:
            vocab_ids = np.unique(sub_coo.col)
            feats[uid]["vocab_unique"] = int(vocab_ids.size)
            # vocab entropy: weight per word = sum of counts
            vocab_counts = np.asarray(sub.sum(axis=0)).ravel()
            vocab_counts = vocab_counts[vocab_counts > 0]
            feats[uid]["vocab_entropy"] = shannon_entropy(vocab_counts)
        else:
            feats[uid]["vocab_unique"] = 0
            feats[uid]["vocab_entropy"] = 0.0

        if (i + 1) % 500 == 0:
            print(f"  feat {i+1}/5000 uid={uid}", flush=True)

    # ----- arrays -----
    feat_names = ["n_u", "unique_rules", "rule_entropy", "rules_per_sent_mean",
                  "rules_per_sent_std", "vocab_unique", "vocab_entropy"]
    X = np.array([[feats[u][k] for k in feat_names] for u in uid_list], dtype=np.float64)
    y = np.array([var_per_user[u] for u in uid_list], dtype=np.float64)

    # log-transform skewed features (n_u, unique_rules, vocab_unique)
    log_idx = [0, 1, 5]  # n_u, unique_rules, vocab_unique
    X_log = X.copy()
    X_log[:, log_idx] = np.log1p(X_log[:, log_idx])

    # ----- univariate correlations on raw and log scales -----
    print("=== Univariate correlations with σ²_u ===")
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

    # ----- partial correlation: each feature vs y, controlling for n_u (log) -----
    print("\n=== Partial correlations (controlling for log n_u) ===")
    ctrl = X_log[:, 0]  # log n_u
    partial = {}
    for j, name in enumerate(feat_names):
        if name == "n_u":
            continue
        col = X_log[:, j]
        # residualize both on ctrl
        from numpy.linalg import lstsq
        A = np.column_stack([np.ones_like(ctrl), ctrl])
        r_y = y - A @ lstsq(A, y, rcond=None)[0]
        r_x = col - A @ lstsq(A, col, rcond=None)[0]
        rp, pp = stats.pearsonr(r_x, r_y)
        partial[name] = {"partial_r": float(rp), "partial_p": float(pp)}
        print(f"  {name:<22} partial_r={rp:+.4f} p={pp:.1e}")

    # ----- drop collinear mirrors (unique_rules ↔ vocab_unique, rule_entropy ↔ vocab_entropy) -----
    keep = ["n_u", "unique_rules", "rule_entropy", "rules_per_sent_mean", "rules_per_sent_std"]
    keep_idx = [feat_names.index(k) for k in keep]
    X_keep = X_log[:, keep_idx]

    # ----- OLS multiple regression (log scale, 5 features) -----
    print("\n=== OLS regression: σ²_u ~ log features (5 non-collinear) ===")
    A = np.column_stack([np.ones(len(y)), X_keep])
    coef, residuals_vec, rank, sv_vals = lstsq(A, y, rcond=None)
    y_hat = A @ coef
    r2 = 1.0 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    n_obs, k = A.shape
    adj_r2 = 1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - k - 1)
    resid = y - y_hat
    mse = (resid ** 2).mean()
    # standard errors
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

    # ----- VIF for kept features -----
    print("\n=== VIF (Variance Inflation Factor, log scale, 5 features) ===")
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
        },
        "target": {
            "name": "sigma2_u (per-user variance of supervised 32d z)",
            "median": float(np.median(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
        },
        "univariate": univariate,
        "partial_controlling_log_n_u": partial,
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