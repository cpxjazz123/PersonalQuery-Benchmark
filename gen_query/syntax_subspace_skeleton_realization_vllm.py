#!/usr/bin/env python3
"""Phase 6.D.3 — End-to-end smoke: skeleton retrieval + LLM realization + closed-loop verify.

用户 2026-08-30 22:20 提案:
- z_target (μ_u or sample) → nearest skeletons → LLM 填内容 → 重算 PCA48 → 验收

Smoke: 1 test user, K=3 z_targets, top-3 skeletons each, 2 gens/skeleton.
Total = 3 × 3 × 2 = 18 generations + 18 verifies.
Output: candidate queries + accept/reject + per-step diagnostics.

依赖: vLLM server @ :8800, skeleton pool, user Gaussians, F3 → PCA48 pipeline.
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

# === Smoke config ===
N_USERS = 1
K_Z_TARGETS = 3
TOP_K_SKELETONS = 3
N_CANDIDATES = 2
TEMPERATURE_GEN = 0.8
TOP_P_GEN = 0.95
MAX_NEW_TOKENS = 80
SEED = 42
VLLM_URL = "http://localhost:8800/v1"
MODEL_NAME = "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [phase6d_e2e] {msg}", flush=True)


def main():
    log("=== Phase 6.D.3 — End-to-end smoke: retrieval + LLM fill + verify ===")

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

    # === Pick test user ===
    samples = []
    with open(SCRATCH / "pca48_dataset" / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    rng = np.random.default_rng(SEED)
    pair_data = []
    for s in samples:
        uid = s.get("user_id") or s.get("user")
        if uid is None or uid not in user_gauss:
            continue
        ug = user_gauss[uid]
        mu = np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
        if mu.shape != (48,):
            continue
        attrs = s["c_i"]
        if not attrs or len(attrs) < 3:
            continue
        pair_data.append({"user_id": uid, "attrs": attrs, "mu": mu})
        if len(pair_data) >= N_USERS:
            break
    log(f"  picked {len(pair_data)} test users")

    # === vLLM API client ===
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=VLLM_URL)
    log(f"  vLLM client ready (URL={VLLM_URL})")

    # === Closed-loop verification ===
    def verify_query(query: str, mu_user: np.ndarray, all_user_mus: np.ndarray) -> dict:
        """Compute d_self, margin, and accept/reject decision."""
        z_q = query_to_z(query)
        if z_q is None:
            return {"query": query, "accepted": False, "reason": "query_to_z failed"}
        d_self = float(np.linalg.norm(z_q - mu_user))
        # Margin vs nearest OTHER user (excluding self)
        d_to_all = np.linalg.norm(all_user_mus - z_q[None, :], axis=1)
        d_nearest_other = float(np.min(d_to_all))
        margin = d_nearest_other - d_self
        # R_95 (per-user, but we'll use median for now)
        accepted = margin > 0 and d_self < 12.0  # ~R_95 from Phase 6.D design
        return {
            "query": query, "z_q": z_q.tolist(),
            "d_self": d_self, "d_nearest_other": d_nearest_other, "margin": margin,
            "accepted": accepted,
        }

    # === Pre-compute all user mus for margin calc ===
    all_user_mus = np.array([np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
                             for ug in user_gauss.values()
                             if np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64).shape == (48,)])
    log(f"  computed {len(all_user_mus)} user mus for margin calculation")

    # === Main loop ===
    all_results = []
    for pi, pd in enumerate(pair_data):
        uid = pd["user_id"]
        attrs = pd["attrs"]
        mu = pd["mu"]
        log(f"\n=== User {uid} ===")
        log(f"  attrs: {attrs}")
        log(f"  μ norm: {np.linalg.norm(mu):.3f}")

        # Sample K z_targets from user Gaussian
        sigma_diag = np.asarray(user_gauss[uid].get("sigma_diag", np.ones(48)), dtype=np.float64)
        sigma_diag_safe = np.maximum(sigma_diag, 1e-6)
        rng_local = np.random.default_rng(SEED + pi)
        z_targets = rng_local.normal(loc=mu, scale=np.sqrt(sigma_diag_safe),
                                     size=(K_Z_TARGETS, 48))

        # For each z_target: top-3 skeletons
        for ki, z_t in enumerate(z_targets):
            dist = np.linalg.norm(skeleton_z - z_t[None, :], axis=1)
            top_idx = np.argsort(dist)[:TOP_K_SKELETONS]
            log(f"\n  z_target {ki+1}/{K_Z_TARGETS} (||z_t - μ||={np.linalg.norm(z_t - mu):.3f})")

            for si, idx in enumerate(top_idx):
                sk = skeleton_pool[int(idx)]
                skel_str = sk["skeleton_str"]

                # LLM prompt: fill skeleton with 5 attrs, first-person
                attrs_text = ", ".join(attrs)
                prompt = (
                    f"You are helping a real customer write a natural shopping query.\n"
                    f"Use this syntax structure:\n  {skel_str}\n"
                    f"Product attributes: {attrs_text}\n"
                    f"Fill the structure with the attributes in a natural first-person way. "
                    f"Output only the filled query, nothing else."
                )

                log(f"    skeleton {si+1} (d={dist[idx]:.3f}): {skel_str[:80]}...")

                # Batched generation: N_CANDIDATES
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
                    verify = verify_query(query, mu, all_user_mus)
                    log(f"      cand {ri+1}: d_self={verify.get('d_self', 'NA'):.3f} "
                        f"margin={verify.get('margin', 'NA'):.3f} "
                        f"{'✓' if verify['accepted'] else '✗'}: {query[:80]}...")
                    all_results.append({
                        "user_id": uid,
                        "z_target_idx": ki,
                        "skeleton_idx": si,
                        "skeleton_str": skel_str,
                        "skeleton_d": float(dist[idx]),
                        "candidate_idx": ri,
                        **verify,
                    })

    # === Save + summary ===
    out_path = LOG_DIR / "phase6d_e2e_smoke.json"
    serializable = {
        "config": {
            "n_users": N_USERS,
            "k_z_targets": K_Z_TARGETS,
            "top_k_skeletons": TOP_K_SKELETONS,
            "n_candidates": N_CANDIDATES,
            "seed": SEED,
        },
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log(f"\n  wrote → {out_path}")

    # Summary
    n_total = len(all_results)
    n_accepted = sum(1 for r in all_results if r["accepted"])
    log(f"\n=== SUMMARY ===")
    log(f"  total candidates: {n_total}")
    log(f"  accepted: {n_accepted} ({n_accepted / n_total * 100:.1f}%)")
    if n_total > 0:
        d_selfs = [r["d_self"] for r in all_results if "d_self" in r]
        margins = [r["margin"] for r in all_results if "margin" in r]
        log(f"  mean d_self: {np.mean(d_selfs):.3f}")
        log(f"  mean margin: {np.mean(margins):.3f}")
        if n_accepted > 0:
            accepted_d = [r["d_self"] for r in all_results if r["accepted"]]
            log(f"  accepted mean d_self: {np.mean(accepted_d):.3f}")


if __name__ == "__main__":
    main()