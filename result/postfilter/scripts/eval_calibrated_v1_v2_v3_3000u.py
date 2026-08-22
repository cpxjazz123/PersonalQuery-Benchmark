#!/usr/bin/env python3
"""基于 3000u 校准阈值 τ_u 和 margin m 重评 V1/V2/V3 final queries.

对每个 case:
- load V1/V2/V3 final query
- 提取 20-d 句法特征 → encoder → mu_q
- 计算 D_u(q) (用 disentangle user_mu + user_logvar)
- D_cross(q) = min_{v≠u} D_v(q)
- margin(q) = D_cross(q) - D_u(q)

判定:
- in_range: D_u(q) ≤ τ_u
- is_unique: margin(q) ≥ m_youden
- both: in_range AND is_unique
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/syntactic_analysis")

from gaussian.gaussian_vades import (
    SentenceEncoder, UserDistributionTableDisentangled,
)

OUT_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/postfilter")
VADES_DIR = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/personal_query/12_complexity_analysis_clause_features/Baby_Products")

OUTPUT_TAG = "vades_disentangled_v2_3000u"
ENCODER_CKPT = VADES_DIR / f"vades_encoder_{OUTPUT_TAG}.pt"
USER_TABLE_CKPT = VADES_DIR / f"vades_user_table_{OUTPUT_TAG}.pt"
USER_PROFILE_FILE = VADES_DIR / f"{OUTPUT_TAG}_user_profiles.jsonl"
SENTENCE_FILE = VADES_DIR / f"{OUTPUT_TAG}_sentences.jsonl"

V1_FILE = OUT_DIR / "iterative_refined.jsonl"
V2_FILE = OUT_DIR / "iterative_refined_v2.jsonl"
V3_FILE = OUT_DIR / "iterative_refined_v3.jsonl"
INITIAL_FILE = OUT_DIR / "candidates.jsonl"

PER_USER_TAU = OUT_DIR / "per_user_tau_3000u.json"
GLOBAL_MARGIN = OUT_DIR / "global_margin_3000u.json"

OUT_REPORT = OUT_DIR / "calibrated_v1_v2_v3_report_3000u.json"
OUT_PER_CASE = OUT_DIR / "calibrated_per_case_3000u.jsonl"

INPUT_DIM = 20
HIDDEN_DIM = 64
LATENT_DIM = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    "max_dependency_depth", "mean_dependency_depth", "dependency_tree_height",
    "depth_variance", "acl_count", "relcl_count", "ccomp_count", "xcomp_count",
    "advcl_count", "clause_nesting_depth", "mean_dependency_distance",
    "max_dependency_distance", "long_dependency_ratio", "amod_count",
    "advmod_count", "nmod_count", "compound_count", "modifier_density",
    "coordination_count", "max_branching_factor",
]


def load_models():
    enc_sd = torch.load(ENCODER_CKPT, map_location=DEVICE, weights_only=False)
    ut_sd = torch.load(USER_TABLE_CKPT, map_location=DEVICE, weights_only=False)
    n_clusters = int(ut_sd["style_centers"].shape[0])
    num_users = int(ut_sd["user_offsets"].shape[0])
    latent_dim = int(ut_sd["style_centers"].shape[1])
    placeholder_cluster = torch.zeros(num_users, dtype=torch.long)
    placeholder_cluster[0] = n_clusters - 1
    encoder = SentenceEncoder(INPUT_DIM, HIDDEN_DIM, latent_dim).to(DEVICE)
    user_table = UserDistributionTableDisentangled(
        num_users=num_users, latent_dim=latent_dim,
        user_cluster_ids=placeholder_cluster, style_anchors=None,
    )
    encoder.load_state_dict(enc_sd)
    user_table.load_state_dict(ut_sd)
    encoder.eval()
    user_table.eval()
    return encoder, user_table


def fit_scaler():
    rows = []
    with SENTENCE_FILE.open() as f:
        for line in f:
            rows.append(json.loads(line))
    feat = np.array([[float(r["features"][n]) for n in FEATURE_NAMES] for r in rows])
    scaler = StandardScaler().fit(feat)
    return scaler


def load_user_mu_logvar(user_table):
    with torch.no_grad():
        all_idx = torch.arange(user_table.num_users)
        um, ul = user_table(all_idx)
        return um.cpu().numpy(), ul.cpu().numpy()


def mahalanobis_diag(mu_s, user_mu, user_logvar):
    diff = mu_s - user_mu
    var = np.exp(user_logvar).clip(min=1e-6)
    return float((diff ** 2 / var).sum())


def batch_mahalanobis(mu_s, user_mu, user_logvar):
    diff = mu_s[:, None, :] - user_mu[None, :, :]
    var = np.exp(user_logvar).clip(min=1e-6)
    return (diff ** 2 / var[None, :, :]).sum(axis=2)


def load_queries(version: str) -> dict[tuple[str, str], str]:
    fp = {"V1": V1_FILE, "V2": V2_FILE, "V3": V3_FILE}[version]
    out = {}
    with fp.open() as f:
        for line in f:
            r = json.loads(line)
            out[(r["user_id"], r["asin"])] = r["final_query"]
    return out


def load_initial() -> dict[tuple[str, str], str]:
    out = {}
    with INITIAL_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["user_id"], r["asin"])
            if key not in out and r.get("candidates"):
                out[key] = r["candidates"][0]
    return out


def extract_features_batch(queries: dict[tuple[str, str], str], scaler) -> dict[tuple[str, str], np.ndarray]:
    log = lambda m: print(f"[eval3000u] {m}", flush=True)
    log(f"  批量解析 {len(queries)} queries ...")
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/syntactic_analysis")
    from extract_clause_features_single_query import extract_clause_features_from_doc
    keys = list(queries.keys())
    texts = [queries[k] for k in keys]
    feats = []
    for doc in nlp.pipe(texts, batch_size=64):
        try:
            f = extract_clause_features_from_doc(doc, doc.text)
            feats.append([float(f[name]) for name in FEATURE_NAMES])
        except Exception:
            feats.append([0.0] * INPUT_DIM)
    feat_arr = np.array(feats, dtype=np.float64)
    feat_scaled = scaler.transform(feat_arr)
    return {k: feat_scaled[i] for i, k in enumerate(keys)}


def main():
    log = lambda m: print(f"[eval3000u] {m}", flush=True)
    log("=" * 70)
    log("3000u 模型: 基于校准阈值 τ_u 和 margin m 重评 V1/V2/V3")
    log("=" * 70)

    calib = json.loads(PER_USER_TAU.read_text())
    tau = calib["tau_per_user"]
    log(f"  τ_u: 加载 {len(tau)} 个用户")
    margin_cfg = json.loads(GLOBAL_MARGIN.read_text())
    m_youden = margin_cfg["m_youden"]
    m_f1 = margin_cfg["m_f1"]
    log(f"  m (Youden J) = {m_youden:.3f}, m (F1) = {m_f1:.3f}")

    encoder, user_table = load_models()
    user_mu, user_logvar = load_user_mu_logvar(user_table)
    scaler = fit_scaler()

    uid_to_idx = {}
    with USER_PROFILE_FILE.open() as f:
        for i, line in enumerate(f):
            uid_to_idx[json.loads(line)["user_id"]] = i
    log(f"  user_table: {user_mu.shape[0]} users")

    log("加载 V1/V2/V3/initial queries ...")
    versions = ["initial", "V1", "V2", "V3"]
    queries = {"initial": load_initial()}
    queries["V1"] = load_queries("V1")
    queries["V2"] = load_queries("V2")
    queries["V3"] = load_queries("V3")
    for v in versions:
        log(f"  {v}: {len(queries[v])} cases")

    feats = {v: extract_features_batch(queries[v], scaler) for v in versions}

    log("encoder forward ...")
    mu_q = {}
    for v in versions:
        keys = list(feats[v].keys())
        X = np.array([feats[v][k] for k in keys])
        with torch.no_grad():
            mu, _, _ = encoder(torch.as_tensor(X, dtype=torch.float32, device=DEVICE))
            mu_v = mu.cpu().numpy()
        mu_q[v] = {k: mu_v[i] for i, k in enumerate(keys)}

    log("compute D_u(q) and margin(q) for each (version, case) ...")
    OUT_PER_CASE.unlink(missing_ok=True)
    stats = {v: {"in_range": 0, "unique": 0, "both": 0, "n": 0,
                 "d_self": [], "margin": []} for v in versions}
    per_case_out = []
    for v in versions:
        for k in mu_q[v]:
            uid, asin = k
            u_idx = uid_to_idx[uid]
            mu_s = mu_q[v][k]
            d_self = mahalanobis_diag(mu_s, user_mu[u_idx], user_logvar[u_idx])
            d_all = batch_mahalanobis(mu_s[None, :], user_mu, user_logvar)[0]
            d_all[u_idx] = np.inf
            d_cross = float(d_all.min())
            margin = d_cross - d_self
            tau_u = tau.get(uid, np.inf)
            in_range = d_self <= tau_u
            unique = margin >= m_youden
            both = in_range and unique
            stats[v]["in_range"] += int(in_range)
            stats[v]["unique"] += int(unique)
            stats[v]["both"] += int(both)
            stats[v]["n"] += 1
            stats[v]["d_self"].append(d_self)
            stats[v]["margin"].append(margin)
            per_case_out.append({
                "version": v, "user_id": uid, "asin": asin,
                "tau_u": tau_u, "d_self": d_self, "d_cross_min": d_cross,
                "margin": margin, "in_range": in_range, "unique": unique, "both": both,
            })

    with OUT_PER_CASE.open("w", encoding="utf-8") as f:
        for r in per_case_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log("=" * 70)
    log("3000u 模型: 校准后的对比 (D_u(q) ≤ τ_u AND margin ≥ m_youden)")
    log("=" * 70)
    report = {}
    print(f"\n{'version':<10} {'n':>4} {'%in_range':>10} {'%unique':>10} {'%both':>10} "
          f"{'mean_d':>12} {'mean_margin':>12}")
    for v in versions:
        n = stats[v]["n"]
        pct_in = stats[v]["in_range"] / n * 100
        pct_un = stats[v]["unique"] / n * 100
        pct_both = stats[v]["both"] / n * 100
        mean_d = float(np.mean(stats[v]["d_self"]))
        mean_m = float(np.mean(stats[v]["margin"]))
        log(f"  {v:<10} {n:>4} {pct_in:>9.1f}% {pct_un:>9.1f}% {pct_both:>9.1f}% "
            f"{mean_d:>12.2f} {mean_m:>12.2f}")
        report[v] = {
            "n": n,
            "n_in_range": stats[v]["in_range"],
            "n_unique": stats[v]["unique"],
            "n_both": stats[v]["both"],
            "pct_in_range": pct_in,
            "pct_unique": pct_un,
            "pct_both": pct_both,
            "mean_d_self": mean_d,
            "mean_margin": mean_m,
            "median_d_self": float(np.median(stats[v]["d_self"])),
            "median_margin": float(np.median(stats[v]["margin"])),
        }

    report["m_youden"] = m_youden
    report["m_f1"] = m_f1
    report["tau_quantile"] = calib["tau_quantile"]
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"\n已写入 {OUT_REPORT}")
    log(f"已写入 {OUT_PER_CASE}")


if __name__ == "__main__":
    main()