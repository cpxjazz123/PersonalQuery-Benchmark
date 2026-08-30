"""Phase 7.B.1 — Anchor Identity Audit.

用户 critique (2026-08-31): 同一空间同一批句子下 μ_full 必比 μ_LOO 更靠近 z_s
(d(s,μ_full) = (n-1)/n · d(s,μ_LOO)), 但 7.B 观测 d_full≈5.9-6.7 > d_LOO≈4.0-4.2,
方向反了 → 存在管线错位。本脚本纯计算 (无 LLM) 验证:

  A. 句子级 direct 重算: 对候选句集合重算 μ_full^direct / μ_LOO^direct,
     验证 ratio d(s,μ_full)/d(s,μ_LOO) == (n-1)/n (应几乎精确).
  B. μ_full^direct (句子级) vs cache μ_u^cache: L2 / max-abs / 逐维最大位置.
  C. review 级 repro: 整条 review 文本投影 (per_sentence_features_v2(doc),
     与 FEAT_CACHE 写入粒度一致) → μ_review^repro vs μ_u^cache:
     若两者接近 → cache μ 是 "整条 review 特征" 的均值 → 粒度错位实锤
     (cache 支撑集合 = review 全文, 7.B 候选 = 单句);
     若两者也差很远 → 还有额外错位 (spaCy 版本 / FEAT_CACHE 管线 / 集合过滤).

覆盖: 5 个 7.B 用户 (全部候选句) + 10 个随机用户 (n_reviews≥15, seed=42).

输出: result/gen_query/phase7b1_anchor_identity_audit.json (单一结果文件).
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
REVIEWS_GZ = REPO_ROOT / "data/Baby_Products_2023.jsonl.gz"
GAUSS_PATH = SCRATCH / "stage8_5_user_gaussians.json"
FEAT_CACHE = REPO_ROOT / "select_query" / "stage7b_query_features.jsonl.gz"
OUT_JSON = REPO_ROOT / "result" / "gen_query" / "phase7b1_anchor_identity_audit.json"

# --- 7.B 同款配置 ---
USERS_7B = [
    "AERFSIGUZKWIES3W3FRBVUNX4RUA",
    "AFOTTSAZYNXVVEZ5IV24QOP5QOWQ",
    "AFYB7O3AY4KNFJ466V2KSOQB2ODQ",
    "AHRQVI734AF32IXI37RUF7P56KMQ",
    "AFRGJMSUHGKN6E36WF7NUXIQVXVQ",
]
MIN_TOKENS = 8
MAX_TOKENS = 60
N_RANDOM_USERS = 10
MIN_REVIEWS_RANDOM = 15
RANDOM_SEED = 42


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [audit] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def filter_sentences(reviews, min_tok=MIN_TOKENS, max_tok=MAX_TOKENS):
    """7.B 同款候选句: 全 ASIN (audit 不排除 target, 因为 cache μ 含 target review)."""
    out = []
    for text, asin in reviews:
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        for s in sents:
            s = s.strip()
            if not s:
                continue
            n_tok = len(s.split())
            if n_tok < min_tok or n_tok > max_tok:
                continue
            out.append(s)
    return out


def project_sentence_level(texts, nlp, all_fnames, col_idx, scaler_mean,
                           scaler_scale, pca_components, pca_mean, psf):
    """7.B 同款: 逐句 psf + mean → 投影. 返回 (z48, ok)."""
    n = len(texts)
    z48 = np.zeros((n, 48), dtype=np.float64)
    ok = np.zeros(n, dtype=bool)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=128, n_process=4)):
        sents = list(doc.sents)
        if not sents:
            continue
        feats_per_sent = []
        for s in sents:
            f = psf(s)
            if f is not None:
                feats_per_sent.append(f)
        if not feats_per_sent:
            continue
        all_keys = set()
        for f in feats_per_sent:
            all_keys.update(f.keys())
        mean_feats = {}
        for k in all_keys:
            vals = [f.get(k, 0.0) for f in feats_per_sent]
            if all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                try:
                    mean_feats[k] = float(np.mean(vals))
                except (TypeError, ValueError):
                    pass
        vec_full = np.array([mean_feats.get(nm, 0.0) for nm in all_fnames], dtype=np.float64)
        v = vec_full[col_idx]
        v_scaled = (v - scaler_mean) / scaler_scale
        z48[i] = (v_scaled - pca_mean) @ pca_components.T
        ok[i] = True
    return z48, ok


def project_review_level(texts, nlp, all_fnames, col_idx, scaler_mean,
                         scaler_scale, pca_components, pca_mean, psf):
    """Cache 侧粒度: 整条 review 文本作为一个 doc, psf(doc) 单特征向量 → 投影.

    与 FEAT_CACHE 写入 (stage_features: per_sentence_features_v2(doc))
    及 build_user Welford 累积粒度一致.
    """
    n = len(texts)
    z48 = np.zeros((n, 48), dtype=np.float64)
    ok = np.zeros(n, dtype=bool)
    for i, doc in enumerate(nlp.pipe(texts, batch_size=128, n_process=4)):
        try:
            f = psf(doc)
        except Exception:
            continue
        if f is None:
            continue
        vec_full = np.array([f.get(nm, 0.0) for nm in all_fnames], dtype=np.float64)
        v = vec_full[col_idx]
        v_scaled = (v - scaler_mean) / scaler_scale
        z48[i] = (v_scaled - pca_mean) @ pca_components.T
        ok[i] = True
    return z48, ok


def main() -> None:
    log("=== Phase 7.B.1 — Anchor Identity Audit ===")

    # --- 1. Load gaussian cache (scaler/PCA 与 7.B 同源) ---
    gdoc = json.load(open(GAUSS_PATH))
    scaler_mean = np.asarray(gdoc["scaler_mean"], dtype=np.float64)
    scaler_scale = np.asarray(gdoc["scaler_scale"], dtype=np.float64)
    pca_components = np.asarray(gdoc["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gdoc["pca_mean"], dtype=np.float64)
    all_fnames = gdoc["feature_names_ordered"]
    fnames_f3 = gdoc["fnames_f3"]
    col_idx = [all_fnames.index(nm) for nm in fnames_f3]
    users_cache = gdoc["users"]
    log(f"  F3 {len(fnames_f3)}d → PCA48, scaler/pca from cache")

    # --- 2. Select random users (n_reviews ≥ MIN_REVIEWS_RANDOM) ---
    rng = random.Random(RANDOM_SEED)
    candidates = [u for u, e in users_cache.items()
                  if e.get("n_reviews", 0) >= MIN_REVIEWS_RANDOM]
    random_users = rng.sample(candidates, N_RANDOM_USERS)
    audit_users = USERS_7B + random_users
    user_set = set(audit_users)
    log(f"  audit users: {len(USERS_7B)} (7.B) + {len(random_users)} (random) = {len(audit_users)}")

    # --- 3. Scan reviews for audit users ---
    user_reviews = collections.defaultdict(list)
    n_records = 0
    with gzip.open(REVIEWS_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            uid = r.get("reviewerID") or r.get("user_id")
            if uid in user_set and r.get("text"):
                asin = r.get("asin") or r.get("parent_asin")
                user_reviews[uid].append((r["text"], asin))
            n_records += 1
            if n_records % 2_000_000 == 0:
                log(f"    {n_records/1e6:.1f}M scanned, {len(user_reviews)} users found")
    log(f"  scanned {n_records} records, {len(user_reviews)} users matched")

    # --- 4. Load FEAT_CACHE keys for review-level reproduction (sha1 match) ---
    review_shas = {}
    for uid, revs in user_reviews.items():
        for t, asin in revs:
            review_shas[feat_key(t)] = uid
    log(f"  review sha1 keys: {len(review_shas)} (FEAT_CACHE matching)")

    # --- 5. spaCy ---
    import spacy
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2 as psf
    nlp = spacy.load("en_core_web_sm")
    for comp in ("ner", "lemmatizer", "attribute_ruler"):
        if comp in nlp.pipe_names:
            nlp.disable_pipe(comp)
    log("  spaCy loaded")

    # --- 6. Per-user audit ---
    per_user = []
    all_ratios = []
    all_gaps = {"sent_vs_cache_l2": [], "sent_vs_cache_max": [],
                "review_vs_cache_l2": [], "review_vs_cache_max": [],
                "sent_vs_review_l2": []}
    for uid in audit_users:
        u = uid[:12]
        e = users_cache.get(uid)
        if e is None:
            log(f"  {u}: NOT in cache — skip")
            continue
        mu_cache = np.asarray(e["mu"], dtype=np.float64)
        revs = user_reviews.get(uid, [])
        cache_n = e["n_sentences"]
        cache_n_rev = e["n_reviews"]

        rec = {"user_id": uid, "n_reviews_raw": len(revs),
               "cache_n_sentences": cache_n, "cache_n_reviews": cache_n_rev}

        # --- 6a. sentence-level direct (7.B 口径) ---
        sents = filter_sentences(revs)
        rec["n_sentences_cand"] = len(sents)
        if sents:
            z_s, ok_s = project_sentence_level(
                sents, nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                pca_components, pca_mean, psf)
            z_s = z_s[ok_s]
            n_s = len(z_s)
            rec["n_sentences_proj"] = n_s
            if n_s >= 3:
                mu_full_direct = z_s.mean(axis=0)
                # per-sentence LOO mean
                loo_mu = (z_s.sum(axis=0) - z_s) / (n_s - 1)
                d_full = np.linalg.norm(z_s - mu_full_direct[None, :], axis=1)
                d_loo = np.linalg.norm(z_s - loo_mu, axis=1)
                ratio = d_full / d_loo
                expected = (n_s - 1) / n_s
                rec["sentence_level"] = {
                    "n": n_s,
                    "ratio_expected": expected,
                    "ratio_med": float(np.median(ratio)),
                    "ratio_mean": float(np.mean(ratio)),
                    "ratio_std": float(np.std(ratio)),
                    "d_full_med": float(np.median(d_full)),
                    "d_loo_med": float(np.median(d_loo)),
                    "d_full_vs_loo": float(np.median(d_full) - np.median(d_loo)),
                }
                all_ratios.append(np.abs(ratio - expected).max())
                # sentence-μ vs cache μ
                gap_l2 = float(np.linalg.norm(mu_full_direct - mu_cache))
                gap_max = float(np.max(np.abs(mu_full_direct - mu_cache)))
                rec["sent_vs_cache"] = {"l2": gap_l2, "max_abs": gap_max}
                all_gaps["sent_vs_cache_l2"].append(gap_l2)
                all_gaps["sent_vs_cache_max"].append(gap_max)
                # sentence-μ vs review-μ (computed below)
                mu_sent_direct = mu_full_direct
            else:
                log(f"  {u}: too few sentence projections ({n_s})")
                mu_sent_direct = None
        else:
            rec["n_sentences_proj"] = 0
            mu_sent_direct = None

        # --- 6b. review-level repro (cache 粒度: 整条 review 全文) ---
        rev_texts = [t for t, _ in revs]
        if rev_texts:
            z_r, ok_r = project_review_level(
                rev_texts, nlp, all_fnames, col_idx, scaler_mean, scaler_scale,
                pca_components, pca_mean, psf)
            z_r = z_r[ok_r]
            n_r = len(z_r)
            rec["n_reviews_proj"] = n_r
            if n_r >= 3:
                mu_review = z_r.mean(axis=0)
                gap_l2 = float(np.linalg.norm(mu_review - mu_cache))
                gap_max = float(np.max(np.abs(mu_review - mu_cache)))
                rec["review_vs_cache"] = {"l2": gap_l2, "max_abs": gap_max}
                all_gaps["review_vs_cache_l2"].append(gap_l2)
                all_gaps["review_vs_cache_max"].append(gap_max)
                if mu_sent_direct is not None:
                    gap_l2 = float(np.linalg.norm(mu_sent_direct - mu_review))
                    rec["sent_vs_review"] = {"l2": gap_l2}
                    all_gaps["sent_vs_review_l2"].append(gap_l2)
        else:
            rec["n_reviews_proj"] = 0

        per_user.append(rec)
        log(f"  {u}: cache_n_sent={cache_n}(revs={cache_n_rev}) | "
            f"cand_sents={rec.get('n_sentences_cand',0)} proj={rec.get('n_sentences_proj',0)} | "
            f"review_repro={rec.get('n_reviews_proj',0)} | "
            f"ratio_med={rec.get('sentence_level',{}).get('ratio_med','-')} | "
            f"sent_vs_cache_l2={rec.get('sent_vs_cache',{}).get('l2','-')} | "
            f"review_vs_cache_l2={rec.get('review_vs_cache',{}).get('l2','-')}")

    # --- 7. Aggregate ---
    def _med(xs):
        return float(np.median(xs)) if xs else None

    summary = {
        "n_users_audited": len(per_user),
        "ratio_identity": {
            "expected": None,
            "max_abs_dev_med": _med(all_ratios),
            "n_users_ratio_ok": sum(1 for r in per_user
                                    if abs(r.get("sentence_level", {}).get("ratio_med", 1.0)
                                           - r["sentence_level"]["ratio_expected"]) < 1e-6)
            if any("sentence_level" in r for r in per_user) else None,
        },
        "gap_summary": {k: {"med": _med(v)} for k, v in all_gaps.items()},
    }
    out = {"config": {
        "description": ("Phase 7.B.1 anchor identity audit: sentence-level direct "
                        "ratio check + review-level repro vs cache μ"),
        "n_random_users": N_RANDOM_USERS, "min_reviews_random": MIN_REVIEWS_RANDOM,
        "seed": RANDOM_SEED, "min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS},
        "summary": summary, "per_user": per_user}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"\n  wrote → {OUT_JSON}")

    log("\n=== SUMMARY ===")
    log(f"  sentence-level ratio d_full/d_LOO vs (n-1)/n: "
        f"max_abs_dev_med={summary['ratio_identity']['max_abs_dev_med']}")
    log(f"  sent_vs_cache μ L2 med: {summary['gap_summary']['sent_vs_cache_l2']['med']}")
    log(f"  review_vs_cache μ L2 med: {summary['gap_summary']['review_vs_cache_l2']['med']}")
    log(f"  sent_vs_review μ L2 med: {summary['gap_summary']['sent_vs_review_l2']['med']}")
    log("=== DONE ===")


if __name__ == "__main__":
    main()
