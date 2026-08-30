#!/usr/bin/env python3
"""Phase 6.D.4 — Pilot cohort (N=10 users, filter μ_norm > 0.5).

对比 Phase 6.D.3 1-user smoke:
- 选 N_USERS=10 个 normal μ_norm 用户 (避免 degenerate near-origin)
- 每个 user 跑同样 18 candidates (3 z_targets × 3 skeletons × 2 cands)
- Cohort-level stats: d_self, margin, accept_rate per user
- Per-user verdict: d_self mean, margin mean, accept rate

Verdict GO 标准:
- mean d_self < 8.0 (明显好于 v6m 11.35 baseline)
- median margin > -2.0 (target user 比多数其他 user 更近)
- accept_rate >= 30% (margin > 0 AND d_self < 12)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
SKELETON_DIR = SCRATCH / "syntax_skeleton"
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"
LOG_DIR = SCRATCH / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Pilot config ===
N_USERS = 10
MU_NORM_MIN = 0.5       # avoid degenerate users
MU_NORM_MAX = 3.0       # avoid extreme outliers
K_Z_TARGETS = 3
TOP_K_SKELETONS = 3
N_CANDIDATES = 2
TEMPERATURE_GEN = 0.8
TOP_P_GEN = 0.95
MAX_NEW_TOKENS = 80
SEED = 42
ACCEPT_MARGIN_THRESHOLD = 0.0
ACCEPT_D_SELF_THRESHOLD = 12.0  # permissive (R_95 typically 8-15)
VLLM_URL = "http://localhost:8800/v1"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [phase6d_pilot] {msg}", flush=True)


def main():
    log(f"=== Phase 6.D.4 — Pilot cohort N={N_USERS}, μ_norm∈[{MU_NORM_MIN},{MU_NORM_MAX}] ===")

    # === Load skeleton pool + embeddings ===
    skeleton_pool = []
    with open(SKELETON_DIR / "syntax_skeleton_pool_n1000.jsonl") as f:
        for line in f:
            skeleton_pool.append(json.loads(line))
    skeleton_z = np.load(SKELETON_DIR / "skeleton_48d_embeddings_n1000.npy")
    log(f"  loaded skeleton pool: {len(skeleton_pool)}, z shape={skeleton_z.shape}")

    # === Load Gaussians ===
    gauss = json.load(open(GAUSS_PATH))
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    user_gauss = gauss["user_gaussians"] if "user_gaussians" in gauss else gauss.get("users", {})
    log(f"  loaded {len(user_gauss)} user Gaussians")

    # === Load spaCy + F3 helper ===
    sys.path.insert(0, str(REPO_ROOT / "common"))
    from syntactic_features import per_sentence_features_v2
    import spacy
    nlp = spacy.load("en_core_web_sm")
    if "textcat" in nlp.pipe_names:
        nlp.remove_pipe("textcat")

    def query_to_z(text: str):
        doc = nlp(text)
        try:
            d = per_sentence_features_v2(doc)
            if d is None:
                return None
            feats = np.zeros(len(fnames_all), dtype=np.float64)
            for j, fn in enumerate(fnames_all):
                feats[j] = float(d.get(fn, 0.0))
            f3 = feats[col_idx_f3]
            f3_scaled = (f3 - scaler_mean) / np.maximum(scaler_scale, 1e-12)
            z = (f3_scaled - pca_mean) @ pca_components.T
            return z
        except Exception:
            return None

    # === Pre-compute all user mus for margin calc ===
    all_user_mus = np.array([np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
                             for ug in user_gauss.values()
                             if np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64).shape == (48,)])
    log(f"  computed {len(all_user_mus)} user mus")

    # === Filter users: μ_norm in range ===
    candidate_users = []
    for uid, ug in user_gauss.items():
        mu = np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
        if mu.shape != (48,):
            continue
        n = float(np.linalg.norm(mu))
        if MU_NORM_MIN <= n <= MU_NORM_MAX:
            candidate_users.append((uid, mu, n))
    log(f"  candidate users (μ_norm∈[{MU_NORM_MIN},{MU_NORM_MAX}]): {len(candidate_users)}")

    rng = np.random.default_rng(SEED)
    rng.shuffle(candidate_users)

    # === Load train.jsonl for attrs ===
    samples = []
    with open(SCRATCH / "pca48_dataset" / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    # Build user_id -> attrs lookup (first attrs seen)
    user_attrs = {}
    for s in samples:
        uid = s.get("user_id") or s.get("user")
        if uid is None:
            continue
        if uid not in user_attrs and s.get("c_i") and len(s["c_i"]) >= 3:
            user_attrs[uid] = s["c_i"]

    # === vLLM API client ===
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)

    # === Main loop: select N_USERS with attrs, then generate ===
    selected_users = []
    for uid, mu, mu_n in candidate_users:
        if uid not in user_attrs:
            continue
        selected_users.append({
            "user_id": uid, "mu": mu, "mu_norm": mu_n, "attrs": user_attrs[uid],
        })
        if len(selected_users) >= N_USERS:
            break
    log(f"  selected {len(selected_users)} test users with attrs")

    all_results = []
    for pi, pd in enumerate(selected_users):
        uid = pd["user_id"]
        attrs = pd["attrs"]
        mu = pd["mu"]
        log(f"\n=== User {pi+1}/{len(selected_users)}: {uid} (μ_norm={pd['mu_norm']:.2f}) ===")

        # Sample K z_targets from user Gaussian
        sigma_diag = np.asarray(user_gauss[uid].get("sigma_diag", np.ones(48)), dtype=np.float64)
        sigma_diag_safe = np.maximum(sigma_diag, 1e-6)
        rng_local = np.random.default_rng(SEED + pi)
        z_targets = rng_local.normal(loc=mu, scale=np.sqrt(sigma_diag_safe),
                                     size=(K_Z_TARGETS, 48))

        user_results = []
        for ki, z_t in enumerate(z_targets):
            dist = np.linalg.norm(skeleton_z - z_t[None, :], axis=1)
            top_idx = np.argsort(dist)[:TOP_K_SKELETONS]

            for si, idx in enumerate(top_idx):
                sk = skeleton_pool[int(idx)]
                skel_str = sk["skeleton_str"]
                attrs_text = ", ".join(attrs)
                prompt = (
                    f"You are helping a real customer write a natural shopping query.\n"
                    f"Use this syntax structure:\n  {skel_str}\n"
                    f"Product attributes: {attrs_text}\n"
                    f"Fill the structure with the attributes in a natural first-person way. "
                    f"Output only the filled query, nothing else."
                )

                responses = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE_GEN,
                    top_p=TOP_P_GEN,
                    n=N_CANDIDATES,
                    seed=SEED + ki * 100 + si,
                )
                for ri, resp in enumerate(responses.choices):
                    query = resp.message.content.strip()
                    doc = nlp(query)
                    try:
                        d = per_sentence_features_v2(doc)
                        if d is None:
                            continue
                        feats = np.zeros(len(fnames_all), dtype=np.float64)
                        for j, fn in enumerate(fnames_all):
                            feats[j] = float(d.get(fn, 0.0))
                        f3 = feats[col_idx_f3]
                        f3_scaled = (f3 - scaler_mean) / np.maximum(scaler_scale, 1e-12)
                        z_q = (f3_scaled - pca_mean) @ pca_components.T
                    except Exception:
                        continue

                    d_self = float(np.linalg.norm(z_q - mu))
                    d_to_all = np.linalg.norm(all_user_mus - z_q[None, :], axis=1)
                    d_nearest_other = float(np.min(d_to_all))
                    margin = d_nearest_other - d_self
                    accepted = margin > ACCEPT_MARGIN_THRESHOLD and d_self < ACCEPT_D_SELF_THRESHOLD

                    user_results.append({
                        "z_target_idx": ki, "skeleton_idx": si, "candidate_idx": ri,
                        "skel_d": float(dist[idx]),
                        "d_self": d_self, "d_nearest_other": d_nearest_other, "margin": margin,
                        "accepted": accepted, "query": query,
                    })

        # Per-user summary
        if user_results:
            d_self_mean = np.mean([r["d_self"] for r in user_results])
            margin_mean = np.mean([r["margin"] for r in user_results])
            accept_rate = np.mean([r["accepted"] for r in user_results]) * 100
            log(f"  d_self mean={d_self_mean:.2f}  margin mean={margin_mean:.2f}  accept={accept_rate:.0f}%")
            all_results.append({
                "user_id": uid, "mu_norm": pd["mu_norm"], "attrs": attrs,
                "d_self_mean": d_self_mean, "margin_mean": margin_mean,
                "accept_rate": accept_rate, "n_candidates": len(user_results),
                "candidates": user_results,
            })

    # === Save ===
    out_path = LOG_DIR / "phase6d_pilot.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "n_users": N_USERS, "mu_norm_range": [MU_NORM_MIN, MU_NORM_MAX],
                "k_z_targets": K_Z_TARGETS, "top_k_skeletons": TOP_K_SKELETONS,
                "n_candidates": N_CANDIDATES, "seed": SEED,
                "accept_margin": ACCEPT_MARGIN_THRESHOLD,
                "accept_d_self": ACCEPT_D_SELF_THRESHOLD,
            },
            "results": all_results,
        }, f, indent=2)
    log(f"\n  wrote → {out_path}")

    # === Cohort summary ===
    if all_results:
        d_self_means = [r["d_self_mean"] for r in all_results]
        margin_means = [r["margin_mean"] for r in all_results]
        accept_rates = [r["accept_rate"] for r in all_results]
        log(f"\n=== COHORT SUMMARY (N={len(all_results)}) ===")
        log(f"  d_self_mean:    median={np.median(d_self_means):.2f}  mean={np.mean(d_self_means):.2f}  "
            f"min={min(d_self_means):.2f}  max={max(d_self_means):.2f}")
        log(f"  margin_mean:    median={np.median(margin_means):.2f}  mean={np.mean(margin_means):.2f}  "
            f"min={min(margin_means):.2f}  max={max(margin_means):.2f}")
        log(f"  accept_rate:    median={np.median(accept_rates):.0f}%  mean={np.mean(accept_rates):.0f}%")
        n_above_thresh = sum(1 for m in margin_means if m > -2.0)
        log(f"  users with margin_mean > -2.0: {n_above_thresh}/{len(margin_means)}")
        log(f"\n  vs v6m baseline (memory):")
        log(f"    v6m min_d_self ≈ 11.35 (much higher than our median {np.median(d_self_means):.2f})")
        log(f"    v6m margin_mean ≈ 0 (close, our median {np.median(margin_means):.2f})")


if __name__ == "__main__":
    main()