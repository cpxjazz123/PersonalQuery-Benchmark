"""Stage 4 v6i — Gate Deconfounding Ablation (用户指令 2026-08-28).

σ-confounding 是 Mahal² gate 候选集本身的 bias — v6h 已证明。
继续在 per-user Mahalanobis gate 后换 selector 没有意义,先改 gate。

3 个 gate variant + 统一 L2 margin score (M_L2):

  G1: NO GATE — 直接 argmax_q M_L2(q, u),无任何 gate
  G2: shared-Σ gate — 所有用户用 pooled_σ_diag 算 Mahal²_shared,
                          gate: Mahal²_shared(z, u) ≤ χ²(0.95, 48)
  G3: L2-radius gate — unified L2 半径 (PCA48 z-space),
                          gate: ||z_q − μ_u||_2 ≤ √χ²(0.95, 48) ≈ 8.07
                          (理论: 48 维 z-space ||z|| ~ χ(48), 95th ≈ √65.17)

统一 score: M_L2(q, u) = min_{v≠u} ||z_q − μ_v||_2 − ||z_q − μ_u||_2
            (PCA48 z-space L2, 完全无 per-user σ)

GO 条件:
  1. |ρ(M_L2, log_det(Σ_u))| < 0.3  (deconfounded)
  2. Rank@1 不明显恶化 (>= 25% 即可接受)
  3. M_L2 > 0 比例 不明显恶化 (>= 25%)

如果 G1/G2/G3 之一 deconfound 成功 (|ρ|<0.3) 但 Rank@1 仍 < 25%,
那说明真正瓶颈是 candidate pool coverage,下一步应扩 K。

输入: ASINS_IN / POOL_IN / GAUSSIANS_IN / FEAT_CACHE
输出: result/select_query/v6i_ablation.json
"""

from __future__ import annotations

import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN,
    MAHAL_THRESHOLD_CHI2_PPF, MAHAL_THRESHOLD_DF,
    PCA_DIM, log, feat_key,
)
from scipy.stats import chi2


def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def main():
    log("=== Stage 4 v6i — Gate Deconfounding Ablation ===")

    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][P["train_idx"]])
    log(f"  PCA{PCA_DIM} ready")

    log("\n=== 2. Loading inputs ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    log(f"  ASINs: {len(asin_data)}, pools: {len(pools)}, user Gaussians: {len(users_gauss)}")

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  features: {len(feat_map)}")

    log("\n=== 3. Projecting pool queries ===")
    pool_z = {}
    miss = 0
    for asin, qs in pools.items():
        zs_for_asin = []
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            vec = np.array([feats.get(n, 0.0) for n in fnames], dtype=np.float64)
            z = pca.transform(scaler.transform(vec[None, :]))[0]
            zs_for_asin.append((z, q))
        pool_z[asin] = zs_for_asin
    log(f"  ASINs with pool_z: {len(pool_z)}, missing features: {miss}")

    # pooled_σ_diag
    pooled_sigma_diag = np.zeros(PCA_DIM)
    for uid, g in users_gauss.items():
        pooled_sigma_diag += np.array(g["sigma_diag"])
    pooled_sigma_diag /= len(users_gauss)
    log(f"\n=== Global pooled_σ_diag (mean over {len(users_gauss)} users) ===")
    log(f"  per-dim mean = {pooled_sigma_diag.mean():.4f}, "
        f"range = [{pooled_sigma_diag.min():.4f}, {pooled_sigma_diag.max():.4f}]")

    log("\n=== 4. Gates ===")
    mahal_threshold = chi2.ppf(MAHAL_THRESHOLD_CHI2_PPF, MAHAL_THRESHOLD_DF)
    l2_radius_threshold = float(np.sqrt(mahal_threshold))  # ≈ 8.073 (PCA48 z-space χ(48) 95th)
    log(f"  G2: Mahal²_shared threshold = χ²({MAHAL_THRESHOLD_CHI2_PPF}, "
        f"{MAHAL_THRESHOLD_DF}) = {mahal_threshold:.3f}")
    log(f"  G3: L2 radius threshold = √χ²({MAHAL_THRESHOLD_CHI2_PPF}, "
        f"{MAHAL_THRESHOLD_DF}) = {l2_radius_threshold:.3f}")
    log(f"    (假设 PCA48 z ~ N(0,1), ||z-μ|| ~ χ(48), 95th ≈ {l2_radius_threshold:.3f}; "
        f"实际 pooled 95th percentile of L2 distance 从经验数据会更准,本 ablation 用理论值)")

    by_asin_users = collections.OrderedDict()
    for entry in asin_data:
        by_asin_users[entry["asin"]] = list(entry["users_sampled"])

    variant_results = {
        "G1_no_gate": {"selected_entries": []},
        "G2_shared_cov_gate": {"selected_entries": []},
        "G3_L2_radius_gate": {"selected_entries": []},
    }

    n_asin_processed = 0
    n_asin_skip = {"no_pool": 0, "no_strict": 0, "lt_2_users": 0}

    for asin_idx, (asin, user_ids) in enumerate(by_asin_users.items()):
        if asin_idx % 100 == 0:
            log(f"  ... ASIN {asin_idx}/{len(by_asin_users)}")
        zqs = pool_z.get(asin)
        if zqs is None:
            n_asin_skip["no_pool"] += 1
            continue
        strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
        if not strict_zqs:
            n_asin_skip["no_strict"] += 1
            continue

        asin_users = []
        for uid in user_ids:
            if uid not in users_gauss:
                continue
            mu = np.array(users_gauss[uid]["mu"])
            sigma = np.array(users_gauss[uid]["sigma_diag"])
            asin_users.append((uid, mu, sigma))
        if len(asin_users) < 2:
            n_asin_skip["lt_2_users"] += 1
            continue

        n_users = len(asin_users)
        n_queries = len(strict_zqs)

        # Mahal² per-user (记录用,但不作为 gate)
        mahal_per_user = np.zeros((n_users, n_queries))
        log_det = np.zeros(n_users)
        for ui, (uid, mu, sigma) in enumerate(asin_users):
            log_det[ui] = float(np.sum(np.log(sigma)))
            for qi, (z, _) in enumerate(strict_zqs):
                mahal_per_user[ui, qi] = mahalanobis_sq(z, mu, sigma)

        # Mahal² shared (G2 gate)
        mahal_shared = np.zeros((n_users, n_queries))
        for ui, (uid, mu, _) in enumerate(asin_users):
            for qi, (z, _) in enumerate(strict_zqs):
                mahal_shared[ui, qi] = mahalanobis_sq(z, mu, pooled_sigma_diag)

        # L2 distance (G3 gate + score basis)
        l2 = np.zeros((n_users, n_queries))
        for ui, (uid, mu, _) in enumerate(asin_users):
            for qi, (z, _) in enumerate(strict_zqs):
                l2[ui, qi] = float(np.linalg.norm(z - mu))

        # M_L2 = min_{v≠u} ||z-μ_v|| − ||z-μ_u||
        sorted_l2 = np.sort(l2, axis=0)
        argmin_l2_per_qi = np.argmin(l2, axis=0)
        is_self_argmin = (argmin_l2_per_qi[None, :] == np.arange(n_users)[:, None])
        d_other_l2 = np.where(is_self_argmin, sorted_l2[1], sorted_l2[0])
        M_L2 = d_other_l2 - l2  # (n_users, n_queries)

        n_asin_processed += 1

        for ui, (uid, mu, sigma) in enumerate(asin_users):
            ld = float(log_det[ui])

            # ---- G1: NO gate, argmax M_L2 over all queries ----
            best_qi_g1 = int(np.argmax(M_L2[ui]))
            best_mahal_g1 = float(mahal_per_user[ui][best_qi_g1])
            target_M = float(M_L2[ui][best_qi_g1])
            sorted_M_per_qi = np.sort(M_L2[:, best_qi_g1])[::-1]
            rank_g1 = int((sorted_M_per_qi >= target_M - 1e-12).sum())
            other_M = np.delete(M_L2[:, best_qi_g1], ui)
            margin_g1 = target_M - float(other_M.max())

            variant_results["G1_no_gate"]["selected_entries"].append({
                "asin": asin,
                "user_id": uid,
                "selected_query": strict_zqs[best_qi_g1][1]["query"][:80],
                "selected_mahal_per_user": best_mahal_g1,
                "selected_score": target_M,
                "target_rank": rank_g1,
                "score_margin": margin_g1,
                "log_det": ld,
            })

            # ---- G2: shared-Σ gate ----
            in_dist_g2 = mahal_shared[ui] <= mahal_threshold
            if in_dist_g2.any():
                avail_M = np.where(in_dist_g2, M_L2[ui], -np.inf)
                best_qi_g2 = int(np.argmax(avail_M))
                best_mahal_g2 = float(mahal_per_user[ui][best_qi_g2])
                target_M_g2 = float(M_L2[ui][best_qi_g2])
                sorted_M_g2 = np.sort(M_L2[:, best_qi_g2])[::-1]
                rank_g2 = int((sorted_M_g2 >= target_M_g2 - 1e-12).sum())
                other_M_g2 = np.delete(M_L2[:, best_qi_g2], ui)
                margin_g2 = target_M_g2 - float(other_M_g2.max())
            else:
                best_qi_g2 = None
                best_mahal_g2 = None
                target_M_g2 = None
                rank_g2 = None
                margin_g2 = None

            variant_results["G2_shared_cov_gate"]["selected_entries"].append({
                "asin": asin,
                "user_id": uid,
                "selected_query": (strict_zqs[best_qi_g2][1]["query"][:80]
                                    if best_qi_g2 is not None else None),
                "selected_mahal_per_user": best_mahal_g2,
                "selected_score": target_M_g2,
                "target_rank": rank_g2,
                "score_margin": margin_g2,
                "log_det": ld,
                "gate_passed": best_qi_g2 is not None,
            })

            # ---- G3: L2-radius gate ----
            in_dist_g3 = l2[ui] <= l2_radius_threshold
            if in_dist_g3.any():
                avail_M_g3 = np.where(in_dist_g3, M_L2[ui], -np.inf)
                best_qi_g3 = int(np.argmax(avail_M_g3))
                best_mahal_g3 = float(mahal_per_user[ui][best_qi_g3])
                target_M_g3 = float(M_L2[ui][best_qi_g3])
                sorted_M_g3 = np.sort(M_L2[:, best_qi_g3])[::-1]
                rank_g3 = int((sorted_M_g3 >= target_M_g3 - 1e-12).sum())
                other_M_g3 = np.delete(M_L2[:, best_qi_g3], ui)
                margin_g3 = target_M_g3 - float(other_M_g3.max())
            else:
                best_qi_g3 = None
                best_mahal_g3 = None
                target_M_g3 = None
                rank_g3 = None
                margin_g3 = None

            variant_results["G3_L2_radius_gate"]["selected_entries"].append({
                "asin": asin,
                "user_id": uid,
                "selected_query": (strict_zqs[best_qi_g3][1]["query"][:80]
                                    if best_qi_g3 is not None else None),
                "selected_mahal_per_user": best_mahal_g3,
                "selected_score": target_M_g3,
                "target_rank": rank_g3,
                "score_margin": margin_g3,
                "log_det": ld,
                "gate_passed": best_qi_g3 is not None,
            })

    log(f"\n  ASINs processed: {n_asin_processed}")
    log(f"  ASINs skipped: {n_asin_skip}")

    log("\n=== 5. Per-variant aggregate ===")
    summary = {}
    for var_name in ["G1_no_gate", "G2_shared_cov_gate", "G3_L2_radius_gate"]:
        entries = variant_results[var_name]["selected_entries"]
        # Filter to gate-passed if applicable
        if "gate_passed" in entries[0]:
            gate_passed_entries = [e for e in entries if e["gate_passed"]]
        else:
            gate_passed_entries = entries
        if not gate_passed_entries:
            log(f"\n--- {var_name}: no entries passed gate ---")
            continue

        scores = np.array([e["selected_score"] for e in gate_passed_entries])
        log_dets = np.array([e["log_det"] for e in gate_passed_entries])
        ranks = np.array([e["target_rank"] for e in gate_passed_entries])
        margins = np.array([e["score_margin"] for e in gate_passed_entries])
        rank1 = float((ranks == 1).mean())
        margin_pos = float((margins > 0).mean())

        if np.std(scores) > 0 and np.std(log_dets) > 0:
            rho_p, p_p = pearsonr(scores, log_dets)
            rho_s, p_s = spearmanr(scores, log_dets)
        else:
            rho_p = rho_s = 0.0
            p_p = p_s = 1.0

        by_asin_q = collections.defaultdict(list)
        for e in gate_passed_entries:
            by_asin_q[e["asin"]].append(e["selected_query"])
        n_q_total = sum(len(qs) for qs in by_asin_q.values())
        n_unique_total = sum(len(set(qs)) for qs in by_asin_q.values())
        n_all_same = sum(1 for qs in by_asin_q.values() if len(set(qs)) == 1)
        unique_ratio = n_unique_total / max(n_q_total, 1)

        summary[var_name] = {
            "n_total": len(entries),
            "n_gate_passed": len(gate_passed_entries),
            "gate_pass_rate": len(gate_passed_entries) / len(entries) if entries else 0.0,
            "score_mean": float(scores.mean()),
            "score_median": float(np.median(scores)),
            "score_std": float(scores.std()),
            "target_rank_mean": float(ranks.mean()),
            "target_rank_median": float(np.median(ranks)),
            "rank1_fraction": rank1,
            "rank_le_2_fraction": float((ranks <= 2).mean()),
            "score_margin_mean": float(margins.mean()),
            "score_margin_median": float(np.median(margins)),
            "margin_positive_fraction": margin_pos,
            "rho_pearson_score_vs_logdet": float(rho_p),
            "rho_spearman_score_vs_logdet": float(rho_s),
            "pearson_p": float(p_p),
            "spearman_p": float(p_s),
            "uniqueness": {
                "avg_unique_ratio": float(unique_ratio),
                "n_all_same_asins": n_all_same,
                "n_total_asins": len(by_asin_q),
            },
        }

        log(f"\n--- {var_name} ---")
        log(f"  total entries: {len(entries)}, gate-passed: {len(gate_passed_entries)} "
            f"({len(gate_passed_entries)/len(entries)*100:.1f}%)")
        log(f"  score (M_L2): mean={scores.mean():.3f}, median={np.median(scores):.3f}, "
            f"std={scores.std():.3f}")
        log(f"  target rank: mean={ranks.mean():.2f}, median={np.median(ranks)}, "
            f"Rank@1={rank1*100:.1f}%, Rank≤2={float((ranks<=2).mean())*100:.1f}%")
        log(f"  score margin: mean={margins.mean():.3f}, median={np.median(margins):.3f}, "
            f"M_L2>0={margin_pos*100:.1f}%")
        log(f"  ρ(M_L2, log_det): Pearson={rho_p:+.3f} (p={p_p:.2e}), "
            f"Spearman={rho_s:+.3f} (p={p_s:.2e})")
        log(f"  uniqueness: avg ratio={unique_ratio:.3f}, "
            f"all-same ASINs={n_all_same}/{len(by_asin_q)}")
        if abs(rho_s) > 0.3:
            log(f"  ⚠ 仍 σ-confounded (|ρ_s|>0.3)")
        else:
            log(f"  ✓ deconfounded (|ρ_s|<0.3)")

    log("\n=== 6. GO verdict ===")
    verdict = {}
    for var_name in ["G1_no_gate", "G2_shared_cov_gate", "G3_L2_radius_gate"]:
        if var_name not in summary:
            continue
        s = summary[var_name]
        deconfounded = abs(s["rho_spearman_score_vs_logdet"]) < 0.3
        rank1_acceptable = s["rank1_fraction"] >= 0.25
        margin_acceptable = s["margin_positive_fraction"] >= 0.25
        go = deconfounded and rank1_acceptable and margin_acceptable
        verdict[var_name] = {
            "deconfounded": deconfounded,
            "rank1_acceptable": rank1_acceptable,
            "margin_acceptable": margin_acceptable,
            "go": go,
        }
        log(f"  {var_name}: deconfounded={deconfounded}, "
            f"rank1≥25%={rank1_acceptable}, margin≥25%={margin_acceptable}, "
            f"GO={go}")

    OUT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/select_query/v6i_ablation.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": ("Stage 4 v6i: 3 gate variant ablation with unified L2 margin "
                                "score M_L2 = min_other − self L2 distance. "
                                "G1 = no gate; G2 = shared-Σ Mahal² gate; "
                                "G3 = unified L2 radius gate (PCA48 z-space). "
                                "GO: |ρ(M_L2, log_det)|<0.3 + Rank@1≥25% + M_L2>0≥25%."),
                "MAHAL_THRESHOLD_G2": float(mahal_threshold),
                "L2_RADIUS_THRESHOLD_G3": float(l2_radius_threshold),
                "MAHAL_THRESHOLD_CHI2_PPF": MAHAL_THRESHOLD_CHI2_PPF,
                "MAHAL_THRESHOLD_DF": MAHAL_THRESHOLD_DF,
                "pooled_sigma_diag_mean": float(pooled_sigma_diag.mean()),
            },
            "summary": summary,
            "verdict": verdict,
        }, f, ensure_ascii=False, indent=2)
    log(f"\nwrote → {OUT}")


if __name__ == "__main__":
    main()