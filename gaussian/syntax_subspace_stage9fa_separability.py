"""Stage 9F-A — 182d vs PCA48 family separability check.

Goal: locate where 8 grammatical families lose diversity.
- Option (a): 182d raw spaCy features are already indistinguishable across families → feature bottleneck
- Option (b): 182d is separable but PCA48 collapses families → PCA bottleneck

For each family in 9C pool:
1. Collect all 182d features (using cached stage7b_query_features.jsonl.gz)
2. Project to PCA48 using frozen scaler + PCA
3. Compute intra/inter family distances:
   - Mean intra-family L2 (mean distance within family)
   - Mean inter-family L2 (mean distance between families)
   - Silhouette score (per-family-cluster separation; 1=perfect, 0=overlap, <0=mislabeled)
4. Compute same metrics on PCA48
5. Per-family PCA48 mean position + vs other families

Decision:
- If 182d silhouette is HIGH (families well-separated) AND PCA48 silhouette LOW → Option (b), PCA bottleneck
- If 182d silhouette is LOW (families already overlap in raw space) → Option (a), feature bottleneck
- If both HIGH → no real diversity loss (selection collapse from selection objective, not representation)
- If both LOW → complete feature design failure

Output:
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fa_182d_separability.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fa_pca48_separability.json
- /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fa_summary.json

Run:
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
        gaussian/syntax_subspace_stage9fa_separability.py \
        > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage9fa_run.log 2>&1 &
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades")
POOL_9C_IN = SCRATCH / "stage9c_pool_pilot.json"
FEAT_CACHE = SCRATCH / "stage7b_query_features.jsonl.gz"

OUT_182D = SCRATCH / "stage9fa_182d_separability.json"
OUT_PCA48 = SCRATCH / "stage9fa_pca48_separability.json"
OUT_SUMMARY = SCRATCH / "stage9fa_summary.json"

SEED = 2024


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def feat_key(t: str) -> str:
    return hashlib.sha1(t.strip().lower().encode("utf-8")).hexdigest()


def compute_separability(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute intra/inter-family distances and silhouette score."""
    families = sorted(set(labels))
    fam_to_idx = {f: [] for f in families}
    for i, f in enumerate(labels):
        fam_to_idx[f].append(i)

    # Intra-family mean L2 distance (mean of mean pairwise within family)
    intra_means = {}
    for f, idxs in fam_to_idx.items():
        if len(idxs) < 2:
            intra_means[f] = 0.0
            continue
        sub = X[idxs]
        dists = []
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                dists.append(np.linalg.norm(sub[i] - sub[j]))
        intra_means[f] = float(np.mean(dists)) if dists else 0.0

    # Inter-family: mean distance between family centroids
    centroids = {}
    for f, idxs in fam_to_idx.items():
        centroids[f] = X[idxs].mean(axis=0)
    inter_centroid_dists = {}
    for i, f1 in enumerate(families):
        for f2 in families[i + 1:]:
            d = float(np.linalg.norm(centroids[f1] - centroids[f2]))
            inter_centroid_dists[f"{f1}__{f2}"] = d

    # Per-query inter-family mean L2 (mean distance from each point to all points NOT in its family)
    inter_pointwise = {}
    for f, idxs in fam_to_idx.items():
        sub = X[idxs]
        others_idx = [i for g, ix in fam_to_idx.items() if g != f for i in ix]
        others = X[others_idx]
        # Compute mean L2 to others
        d_to_others = []
        for p in sub:
            d_to_others.append(float(np.mean(np.linalg.norm(others - p, axis=1))))
        inter_pointwise[f] = float(np.mean(d_to_others)) if d_to_others else 0.0

    # Silhouette score (only meaningful if >= 2 families and > 1 sample per family)
    sil = None
    if len(families) >= 2 and all(len(fam_to_idx[f]) >= 2 for f in families):
        try:
            sil = float(silhouette_score(X, labels, metric="euclidean"))
        except Exception as e:
            log(f"    silhouette failed: {e}")
            sil = None

    return {
        "intra_family_mean_l2": intra_means,
        "inter_family_mean_l2_to_centroid": inter_centroid_dists,
        "inter_family_pointwise_mean_l2": inter_pointwise,
        "silhouette": sil,
        "n_families": len(families),
        "family_sizes": {f: len(idxs) for f, idxs in fam_to_idx.items()},
    }


def main():
    log("=== Stage 9F-A — 182d vs PCA48 family separability check ===")

    # === 1. Load PCA48 setup ===
    log("\n=== 1. Loading PCA48 ===")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
    sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/gaussian")
    from gaussian_vades import _syntax_subspace_prepare
    P = _syntax_subspace_prepare()
    scaler = P["scaler"]
    fnames = P["feature_names_ordered"]
    train_idx = P["train_idx"]

    pca = PCA(n_components=48, random_state=SEED)
    pca.fit(P["X_scaled"][train_idx])
    log(f"  PCA48 ready; scaler mean shape: {scaler.mean_.shape}")

    # === 2. Load 9C pool + features ===
    log("\n=== 2. Loading 9C pool + features ===")
    pool_9C_data = json.load(open(POOL_9C_IN))
    pools_9C = pool_9C_data["pools"]

    feat_map = {}
    with gzip.open(FEAT_CACHE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            feat_map[rec["k"]] = rec["v"]
    log(f"  features loaded: {len(feat_map)}")

    # Build (query, family, features) tuples
    samples = []
    miss = 0
    for asin, qs in pools_9C.items():
        for q in qs:
            k = feat_key(q["query"])
            feats = feat_map.get(k)
            if not feats:
                miss += 1
                continue
            samples.append({
                "asin": asin,
                "family": q.get("family", "?"),
                "query": q["query"],
                "feats": feats,
            })
    log(f"  samples: {len(samples)}, missing features: {miss}")

    family_distribution = collections.Counter(s["family"] for s in samples)
    log(f"  family distribution: {dict(family_distribution)}")

    # Build 182d matrix
    X_182d = np.array([
        np.array([s["feats"].get(n, 0.0) for n in fnames], dtype=np.float64)
        for s in samples
    ])
    labels = np.array([s["family"] for s in samples])
    log(f"  X_182d shape: {X_182d.shape}, families: {len(set(labels))}")

    # === 3. Compute separability in raw 182d space ===
    log("\n=== 3. Separability in RAW 182d space (before PCA) ===")
    sep_182d = compute_separability(X_182d, labels)
    log(f"  silhouette (182d): {sep_182d['silhouette']}")
    log(f"  intra-family mean L2 (per family):")
    for f, d in sep_182d["intra_family_mean_l2"].items():
        log(f"    {f}: {d:.3f}")
    log(f"  inter-family mean L2 (pointwise, per family):")
    for f, d in sep_182d["inter_family_pointwise_mean_l2"].items():
        log(f"    {f}: {d:.3f}")

    with open(OUT_182D, "w", encoding="utf-8") as f:
        json.dump(sep_182d, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUT_182D}")

    # === 4. Project to PCA48 and compute separability ===
    log("\n=== 4. Projecting to PCA48 (scaled + 48 components) ===")
    X_scaled = scaler.transform(X_182d)
    X_pca48 = pca.transform(X_scaled)
    log(f"  X_pca48 shape: {X_pca48.shape}")

    sep_pca48 = compute_separability(X_pca48, labels)
    log(f"  silhouette (PCA48): {sep_pca48['silhouette']}")
    log(f"  intra-family mean L2 (per family):")
    for f, d in sep_pca48["intra_family_mean_l2"].items():
        log(f"    {f}: {d:.3f}")
    log(f"  inter-family mean L2 (pointwise, per family):")
    for f, d in sep_pca48["inter_family_pointwise_mean_l2"].items():
        log(f"    {f}: {d:.3f}")

    with open(OUT_PCA48, "w", encoding="utf-8") as f:
        json.dump(sep_pca48, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUT_PCA48}")

    # === 5. Also test PCA10 / PCA100 / PCA150 to see compression curve ===
    log("\n=== 5. Compression curve: silhouette vs PCA dim ===")
    sil_curve = {}
    for n_dim in [10, 20, 30, 48, 64, 96, 128, 150, 182]:
        if n_dim > 182:
            continue
        if n_dim == 182:
            X_dim = X_scaled  # use full scaled
        else:
            pca_n = PCA(n_components=n_dim, random_state=SEED)
            pca_n.fit(P["X_scaled"][train_idx])
            X_dim = pca_n.transform(X_scaled)
        if len(set(labels)) >= 2:
            try:
                sil = float(silhouette_score(X_dim, labels, metric="euclidean"))
            except Exception as e:
                sil = None
        else:
            sil = None
        sil_curve[n_dim] = sil
        log(f"  PCA{n_dim}: silhouette = {sil}")

    # === 6. Decision ===
    log("\n=== 6. Decision ===")
    sil_182d = sep_182d["silhouette"]
    sil_pca48 = sep_pca48["silhouette"]
    decision = "unknown"
    if sil_182d is not None and sil_pca48 is not None:
        if sil_182d > 0.10 and sil_pca48 < 0.05:
            decision = "PCA bottleneck (Option b): 182d is separable but PCA collapses families"
        elif sil_182d < 0.05:
            decision = "Feature bottleneck (Option a): 182d raw features are already indistinguishable"
        elif sil_182d > 0.10 and sil_pca48 > 0.10:
            decision = "No representation bottleneck (neither); selection collapse is from selection objective"
        else:
            decision = "Mixed: both have low silhouette; feature is the primary bottleneck"
    log(f"  Decision: {decision}")
    log(f"  silhouette 182d = {sil_182d}")
    log(f"  silhouette PCA48 = {sil_pca48}")

    # Also test relative compression ratio
    intra_182d_mean = float(np.mean(list(sep_182d["intra_family_mean_l2"].values())))
    inter_182d_mean = float(np.mean(list(sep_182d["inter_family_pointwise_mean_l2"].values())))
    intra_pca48_mean = float(np.mean(list(sep_pca48["intra_family_mean_l2"].values())))
    inter_pca48_mean = float(np.mean(list(sep_pca48["inter_family_pointwise_mean_l2"].values())))

    sep_ratio_182d = inter_182d_mean / intra_182d_mean if intra_182d_mean > 0 else 0
    sep_ratio_pca48 = inter_pca48_mean / intra_pca48_mean if intra_pca48_mean > 0 else 0
    log(f"  separation ratio (inter/intra):")
    log(f"    182d: {sep_ratio_182d:.3f}")
    log(f"    PCA48: {sep_ratio_pca48:.3f}")
    log(f"    ratio preserved: {(sep_ratio_pca48 / sep_ratio_182d * 100) if sep_ratio_182d else 0:.1f}%")

    summary = {
        "config": {
            "description": "Stage 9F-A — 182d vs PCA48 family separability check",
            "n_samples": len(samples),
            "n_families": len(set(labels)),
            "family_distribution": dict(family_distribution),
            "SEED": SEED,
        },
        "silhouette_182d": sil_182d,
        "silhouette_pca48": sil_pca48,
        "silhouette_curve": sil_curve,
        "intra_family_mean_l2_182d": intra_182d_mean,
        "inter_family_pointwise_mean_l2_182d": inter_182d_mean,
        "intra_family_mean_l2_pca48": intra_pca48_mean,
        "inter_family_pointwise_mean_l2_pca48": inter_pca48_mean,
        "separation_ratio_182d": sep_ratio_182d,
        "separation_ratio_pca48": sep_ratio_pca48,
        "separation_ratio_preserved_pct": (sep_ratio_pca48 / sep_ratio_182d * 100) if sep_ratio_182d else 0,
        "decision": decision,
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  wrote → {OUT_SUMMARY}")

    log("\n=== Stage 9F-A complete ===")


if __name__ == "__main__":
    main()