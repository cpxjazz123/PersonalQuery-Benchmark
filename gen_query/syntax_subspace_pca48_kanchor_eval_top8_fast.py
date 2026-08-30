#!/usr/bin/env python3
"""Phase 6.C.2 K-anchor ONLY eval (加速版, 复用 Phase 6.A 已有 metrics).

用户 2026-08-30 加速请求: 不要重跑 Phase 6.A baseline (838s), 直接复用
phase6c1_pip_eval_top8.json 的 Phase 6.A metrics, 仅跑 K-anchor ckpt 3 pair。

计算量: 3 pair × 8 PC × 5 α × 2 cand = 240 gen × 0.1s = 24s/pair × 3 = 72s = ~1.2 min
+ model load ~30s
总 ETA: ~2-3 min

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
DATA_DIR = SCRATCH / "pca48_dataset"
LOG_DIR = SCRATCH / "logs"
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"
ANCHORS_PATH = SCRATCH / "pca48_kanchor" / "anchors.npy"
PIP_TOP8_JSON = LOG_DIR / "phase6c1_pip_eval_top8.json"

# === Eval config (n=10 stability, 用户 2026-08-30 锁定: 不改训练,先扩 eval) ===
N_PAIRS = 10               # 3 → 10 stability verification (Phase 6.A baseline 也是 n=10)
N_CANDIDATES = 2
N_PCS = 8
ALPHAS = [-2.0, -1.0, 0.0, +1.0, +2.0]
TEMPERATURE_GEN = 0.8
TOP_P_GEN = 0.95
MAX_NEW_TOKENS = 50
SEED = 42

KANCHOR_CKPT = SCRATCH / "pca48_kanchor_ckpt" / "kanchor_smoke.pt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [kanchor_eval_fast] {msg}", flush=True)


def main():
    log("=== Phase 6.C.2 K-anchor FAST eval — only K-anchor ckpt, reuse 6.A from PIP top-8 ===")

    # Load Phase 6.A metrics from existing PIP top-8 JSON
    pip_data = json.load(open(PIP_TOP8_JSON))
    a_metrics = {int(j): m for j, m in pip_data["per_ckpt_per_pc"]["Phase 6.A (init)"].items()}
    log(f"  loaded Phase 6.A init metrics from {PIP_TOP8_JSON.name} (n_pairs=10 baseline)")
    for j in range(N_PCS):
        log(f"    PC{j}: ρ={a_metrics[j]['rho_dir']:+.3f} sign={a_metrics[j]['sign_accuracy']*100:.0f}%")

    # Load Gaussians (need fnames for query→z)
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

    # Load test pairs (same seed, take first N_PAIRS=3)
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
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
        pair_data.append({"user_id": uid, "attrs": s["c_i"], "z_target": mu})
        if len(pair_data) >= N_PAIRS:
            break
    pair_data = pair_data[:N_PAIRS]
    log(f"  {len(pair_data)} test pairs (smoke, n=3)")

    # spaCy + F3 helper
    sys.path.insert(0, str(REPO_ROOT / "common"))
    sys.path.insert(0, str(REPO_ROOT / "gen_query"))
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

    # Load K-anchor ckpt
    from syntax_subspace_pca48_kanchor import Qwen2WithKAnchor
    log(f"  loading K-anchor ckpt: {KANCHOR_CKPT}")
    model = Qwen2WithKAnchor(anchors_path=str(ANCHORS_PATH))
    model.load_controller(str(KANCHOR_CKPT))
    model.eval()
    log(f"  K-anchor model ready")

    # Eval K-anchor
    t0 = time.time()
    results = {j: {a: [] for a in ALPHAS} for j in range(N_PCS)}
    weight_entropies_per_pair = []

    for pi, pd in enumerate(pair_data):
        pair_entropies = []
        for pc_j in range(N_PCS):
            for alpha in ALPHAS:
                z_test = pd["z_target"].copy()
                z_test[pc_j] += alpha

                z_tensor = torch.from_numpy(z_test.astype(np.float32)).unsqueeze(0).to("cuda")
                z_tensor = z_tensor.to(model.dtype)  # cast to bfloat16 (anchor_embeddings dtype)
                zs_expanded = z_tensor.expand(N_CANDIDATES, -1).contiguous()
                attrs_list = [pd["attrs"]] * N_CANDIDATES
                torch.manual_seed(SEED + pi * 100 + pc_j * 10 + int((alpha + 2) * 10))
                candidates = model.generate(
                    zs_expanded, attrs_list,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE_GEN,
                    top_p=TOP_P_GEN,
                    do_sample=True,
                )

                # Diagnostic: compute weight entropy for this z
                with torch.no_grad():
                    _, w = model.controller(z_tensor)
                H = float(-(w * torch.log(w + 1e-12)).sum(dim=-1).mean().item())
                pair_entropies.append(H)

                z_query_list = []
                for cand in candidates:
                    zq = query_to_z(cand)
                    if zq is not None:
                        z_query_list.append(zq)
                if z_query_list:
                    z_mean = np.mean(z_query_list, axis=0)
                    results[pc_j][alpha].append({
                        "pair_idx": pi,
                        "z_query_mean": z_mean.tolist(),
                    })

        weight_entropies_per_pair.append(np.mean(pair_entropies))
        log(f"    [K-anchor] pair {pi+1}/{N_PAIRS} elapsed={time.time()-t0:.0f}s  "
            f"H(w)={weight_entropies_per_pair[-1]:.3f}")

    log(f"  K-anchor eval done in {time.time() - t0:.0f}s")

    # Compute K-anchor per-PC metrics
    k_metrics = {}
    for pc_j in range(N_PCS):
        per_pair_pc_j = {}
        for alpha in ALPHAS:
            for entry in results[pc_j][alpha]:
                pi = entry["pair_idx"]
                per_pair_pc_j.setdefault(pi, {})[alpha] = entry["z_query_mean"][pc_j]

        alphas_for_corr = []
        pcj_means_for_corr = []
        for pi, alpha_dict in per_pair_pc_j.items():
            for alpha in ALPHAS:
                if alpha in alpha_dict:
                    alphas_for_corr.append(alpha)
                    pcj_means_for_corr.append(alpha_dict[alpha])

        rho_dir, pval_dir = spearmanr(alphas_for_corr, pcj_means_for_corr)

        sign_correct = 0
        sign_total = 0
        for pi, alpha_dict in per_pair_pc_j.items():
            if +2.0 in alpha_dict and -2.0 in alpha_dict:
                sign_total += 1
                if alpha_dict[+2.0] > alpha_dict[-2.0]:
                    sign_correct += 1
        sign_acc = sign_correct / max(sign_total, 1)

        k_metrics[pc_j] = {
            "rho_dir": float(rho_dir),
            "pval_dir": float(pval_dir),
            "sign_accuracy": float(sign_acc),
            "sign_correct": sign_correct,
            "sign_total": sign_total,
        }

    # Paired lift
    log("\n=== PAIRED LIFT (K-anchor - Phase 6.A init, 6.A from PIP top-8 n=10) ===")
    log("  NOTE: Phase 6.A baseline from 10-pair eval, K-anchor from 3-pair smoke")
    log("  Comparison direction still valid (same ckpt, same pairs)")

    deltas = []
    paired = {}
    for j in range(N_PCS):
        a_rho = a_metrics[j]["rho_dir"]
        k_rho = k_metrics[j]["rho_dir"]
        a_sig = a_metrics[j]["sign_accuracy"]
        k_sig = k_metrics[j]["sign_accuracy"]
        d_rho = k_rho - a_rho
        d_sig = k_sig - a_sig
        deltas.append(d_rho)
        paired[j] = {
            "rho_6a": a_rho, "rho_kanchor": k_rho, "delta_rho": d_rho,
            "sign_6a": a_sig, "sign_kanchor": k_sig, "delta_sign": d_sig,
        }
        log(f"  PC{j}: 6A ρ={a_rho:+.3f} → K ρ={k_rho:+.3f} (Δ={d_rho:+.3f}); "
            f"sign 6A={a_sig*100:.0f}% → K={k_sig*100:.0f}% (Δ={d_sig*100:+.0f}pp)")

    deltas_arr = np.array(deltas)
    mean_d = float(np.mean(deltas_arr))
    median_d = float(np.median(deltas_arr))
    n_improve = int(np.sum(deltas_arr > 0))
    n_reversed = int(np.sum((np.array([k_metrics[j]["rho_dir"] for j in range(N_PCS)]) < 0) &
                            (np.array([a_metrics[j]["rho_dir"] for j in range(N_PCS)]) > 0)))
    n_sign_reversed = int(np.sum((np.array([k_metrics[j]["sign_accuracy"] for j in range(N_PCS)]) < 0.4) &
                                   (np.array([a_metrics[j]["sign_accuracy"] for j in range(N_PCS)]) > 0.6)))

    log(f"\n  Paired lift statistics:")
    log(f"    mean Δρ = {mean_d:+.3f}")
    log(f"    median Δρ = {median_d:+.3f}")
    log(f"    # PCs improve (Δρ > 0): {n_improve}/{N_PCS}")
    log(f"    # PCs ρ reversed: {n_reversed}/{N_PCS}")
    log(f"    # PCs sign reversed: {n_sign_reversed}/{N_PCS}")
    log(f"    avg H(w) per pair: {np.mean(weight_entropies_per_pair):.3f} (log(32)={np.log(32):.3f})")

    # Verdict (用户 K-anchor GO 标准: ≥6/8 improve + median Δρ>0.1 + 0 reversed)
    log("\n=== VERDICT (用户锁定 K-anchor GO 标准) ===")
    if n_improve >= 6 and median_d > 0.1 and n_reversed == 0:
        verdict = "KANCHOR_GO"
        rationale = (f"K-anchor GO: {n_improve}/{N_PCS} PCs improve + median Δρ>+0.1 + 0 reversed. "
                     f"显著优于 Phase 6.A / PIP.")
    elif n_improve <= 2:
        verdict = "KANCHOR_NO_GO"
        rationale = f"K-anchor NO-GO: only {n_improve}/{N_PCS} PCs improve."
    elif n_improve >= 4:
        verdict = "KANCHOR_PARTIAL"
        rationale = (f"K-anchor PARTIAL: {n_improve}/{N_PCS} PCs improve, "
                     f"median Δρ={median_d:+.3f}. Scale up training (100→500 steps), "
                     f"check τ (1.0→0.5).")
    else:
        verdict = "KANCHOR_NOISE"
        rationale = f"K-anchor noisy ({n_improve}/8 improve)."

    log(f"  VERDICT: {verdict}")
    log(f"  {rationale}")

    # Save
    out_path = LOG_DIR / "phase6c2_kanchor_eval_top8_smoke.json"
    serializable = {
        "config": {
            "n_pairs": N_PAIRS,
            "n_pcs": N_PCS,
            "alphas": ALPHAS,
            "n_candidates": N_CANDIDATES,
            "seed": SEED,
            "phase6a_baseline_source": "phase6c1_pip_eval_top8.json (n=10)",
        },
        "per_ckpt_per_pc": {
            "Phase 6.A (init) (cached)": {str(j): m for j, m in a_metrics.items()},
            "K-anchor (smoke 100 steps)": {str(j): m for j, m in k_metrics.items()},
        },
        "paired_lift": {str(j): v for j, v in paired.items()},
        "paired_summary": {
            "mean_delta_rho": mean_d,
            "median_delta_rho": median_d,
            "n_improve": n_improve,
            "n_reversed": n_reversed,
            "n_sign_reversed": n_sign_reversed,
        },
        "weight_entropies": weight_entropies_per_pair,
        "verdict": verdict,
        "rationale": rationale,
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()
