"""Syntax Subspace — Stage 4 (Mahalanobis select).

归 select_query/: 基于 per-user Gaussian 从共享候选池选最佳查询。

用户指令 2026-08-28: 只输出 selected (Mahalanobis 最小) per (asin, user),
删掉 random / farthest 3-way contrast — Stage 5 volatility 已经只读 selected,
random / farthest 既不被评估也不被报告,只是浪费 ~3× retrieval + selection 计算。

Stage 4: 对每 (asin, user) 选 Mahalanobis 最小候选 (selected only)

用法:
  python select_query/syntax_subspace_select.py --stage select

I/O 路径:
  输入: stage8_5_asins.json / stage8_5_pool.json / stage8_5_user_gaussians.json
  输出: stage8_5_selection.json + stage8_5_selection_stats.json  (Stage 4)

共享工具 (log, feat_key, paths, hyperparams, _syntax_subspace_prepare) 来自:
  common/syntax_subspace_utils.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from syntax_subspace_utils import (  # noqa: E402
    ASINS_IN, FEAT_CACHE, GAUSSIANS_IN, POOL_IN, SELECTION_IN, SELECTION_OUT,
    SELECTION_STATS_OUT,
    PCA_DIM, PCA_SEED, SEED, log, feat_key,
)


# ===========================================================================
# STAGE 4 — MAHA SELECT
# ===========================================================================

def mahalanobis_sq(z: np.ndarray, mu: np.ndarray, sigma_diag: np.ndarray) -> float:
    diff = z - mu
    return float((diff * diff / sigma_diag).sum())


def stage_select():
    log("=== STAGE 4 — MAHA SELECT ===")

    log("\n=== 1. Loading PCA48 ===")
    from syntax_subspace_utils import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 ready")

    log("\n=== 2. Loading inputs ===")
    asin_data = json.load(open(ASINS_IN))["asins"]
    pool_data = json.load(open(POOL_IN))
    pools = pool_data["pools"]
    gauss_data = json.load(open(GAUSSIANS_IN))
    users_gauss = gauss_data["users"]
    log(f"  ASINs: {len(asin_data)}, pools: {len(pools)}")
    log(f"  user Gaussians: {len(users_gauss)}")

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

    log("\n=== 4. Selection per (asin, user) ===")

    selection_entries = []

    n_skip_no_pool = 0
    for entry in asin_data:
        asin = entry["asin"]
        attrs = entry["attrs_used"]
        n_input = len(attrs)
        zqs = pool_z.get(asin)
        if zqs is None:
            # 用户指令 2026-08-28: Stage 1 N=5 + numeric/metadata filter 主动过滤
            # 掉 <5 非数值 attr 的 ASIN(典型原因:商品页 attrs 多数是 numeric/Country/Department/
            # Dimensions 等)。这些 ASIN 不在 pool 是预期行为,跳过而非 raise。
            n_skip_no_pool += 1
            continue
        n_pool_strict = sum(1 for z, q in zqs if q["strict"])

        for uid in entry["users_sampled"]:
            # 用户指令 2026-08-27: 去掉 asin_centroid_fallback, 上游保证每个 user 都有 Gaussian
            if uid not in users_gauss:
                raise KeyError(
                    f"user {uid} (ASIN={asin}) not in user_gaussians. "
                    f"上游 build_dataset.py 已过滤 MIN_REVIEWS_PER_USER=20, "
                    f"但 Stage 3 没找到该 user 的 Gaussian — 字段提取不一致, "
                    f"需修 build_dataset.py 或 gaussian/syntax_subspace_user_gaussians.py 的 user_id 提取。"
                )
            mu = np.array(users_gauss[uid]["mu"])
            sigma = np.array(users_gauss[uid]["sigma_diag"])
            source = users_gauss[uid]["source"]
            n_reviews = users_gauss[uid]["n_reviews"]

            strict_zqs = [(z, q) for z, q in zqs if q["strict"]]
            if not strict_zqs:
                # 用户指令 2026-08-27: 不再用 all (含 invalid) 作 strict fallback;
                # 上游 Stage 1 generation 失败/属性不匹配时, 该 ASIN 直接 no_pool
                selection_entries.append({
                    "asin": asin,
                    "user_id": uid,
                    "attrs_used": attrs,
                    "selection_method": "no_pool",
                    "selected": None,
                    "selected_distance": None,
                    "n_candidates": 0,
                    "user_source": source,
                    "n_reviews": n_reviews,
                })
                continue

            distances = np.array([mahalanobis_sq(z, mu, sigma) for z, _ in strict_zqs])
            best_idx = int(np.argmin(distances))
            selected_q = strict_zqs[best_idx][1]

            selection_entries.append({
                "asin": asin,
                "user_id": uid,
                "attrs_used": attrs,
                "selection_method": "mahal_min",
                "selected": selected_q,
                "selected_distance": float(distances[best_idx]),
                "n_candidates": len(strict_zqs),
                "user_source": source,
                "n_reviews": n_reviews,
            })

    log(f"  total entries: {len(selection_entries)}")
    log(f"  ASINs skipped (no pool, Stage 1 filtered): {n_skip_no_pool}/{len(asin_data)} "
        f"(Stage 1 N=5 + numeric/metadata filter 主动过滤 <5 非数值 attr 的 ASIN)")

    SELECTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "description": "Stage 8.5: Mahalanobis selection from shared pool (selected only)",
                "SEED": SEED,
            },
            "n_entries": len(selection_entries),
            "entries": selection_entries,
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {SELECTION_OUT}")

    log("\n=== 7. Validation ===")
    mahal_entries = [e for e in selection_entries if e["selected_distance"] is not None]
    selected = np.array([e["selected_distance"] for e in mahal_entries])

    log(f"  N pairs: {len(mahal_entries)}")
    log(f"  selected (mahal_min): mean={selected.mean():.3f}, "
        f"std={selected.std():.3f}, median={np.median(selected):.3f}")
    log(f"  (mahal 越小 = query 越接近 user 个体句法骨架; selected 是全 strict 池中的最小, "
        f"无需 3-way 对比 — 删掉 random/farthest 2026-08-28)")

    by_source = collections.defaultdict(list)
    for e in mahal_entries:
        by_source[e["user_source"]].append(e["selected_distance"])

    log(f"\n=== Per-source breakdown ===")
    for src, ds in by_source.items():
        arr = np.array(ds)
        log(f"  {src} (n={len(ds)}): mean={arr.mean():.3f}, median={np.median(arr):.3f}")

    with open(SELECTION_STATS_OUT, "w", encoding="utf-8") as f:
        json.dump({
            "n_pairs": len(mahal_entries),
            "selected_mean": float(selected.mean()),
            "selected_std": float(selected.std()),
            "selected_median": float(np.median(selected)),
            "per_source": {
                src: {
                    "n": len(ds),
                    "selected_mean": float(np.mean(ds)),
                    "selected_median": float(np.median(ds)),
                }
                for src, ds in by_source.items()
            },
        }, f, ensure_ascii=False, indent=2)
    log(f"wrote → {SELECTION_STATS_OUT}")


# MAIN
# ===========================================================================

STAGE_FUNCTIONS = {
    "select": stage_select,
}


def main():
    parser = argparse.ArgumentParser(description="Syntax Subspace — select_query (Mahalanobis)")
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGE_FUNCTIONS.keys()) + ["all"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    log(f"=== syntax_subspace_select.py — stage={args.stage} ===")

    if args.stage == "all":
        for stage_name in STAGE_FUNCTIONS:
            log(f"\n>>> Running stage: {stage_name}")
            STAGE_FUNCTIONS[stage_name]()
    else:
        STAGE_FUNCTIONS[args.stage]()


if __name__ == "__main__":
    main()