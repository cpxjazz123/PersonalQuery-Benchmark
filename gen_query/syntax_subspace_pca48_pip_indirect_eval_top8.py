#!/usr/bin/env python3
"""Phase 6.C.1-eval-top8 — top-8 PC directional audit for PIP-Indirect ckpt.

用户指令 2026-08-30:
PIP lift 不是显著 (+0.061 PC0, marginal sign lift), 不能仅凭 PC0 判定 GO/NO-GO。
下一步验证 PIP 是否对 PCA48 空间普遍改善:
- 测 top-8 PCs (按 explained variance 排序,固定, 不挑表现好的 PC, 避免 selection bias)
- 每个 PC α ∈ {-2, -1, 0, +1, +2} (5 档)
- 对比 Phase 6.A init vs PIP final, 算 paired Δρ_j = ρ_j^PIP - ρ_j^6A
- 统计: mean/median Δρ + # PCs improve + # PCs reversed

判定:
- ≥6/8 PCs improve AND median Δρ > 0 AND few reversed → PIP_GO_LOCAL (继续调参)
- ≤2/8 PCs improve → PIP_LOCAL_ONLY (停 PIP, 转 CIE)
- 3-5/8 PCs improve → PIP_PARTIAL_TOP8 (scale up 或 specific PC investigation)

Smoke (Rule 18) per ckpt: 3 pair × 8 PC × 5 α × 4 cand = 480 gen, ~9 min
Lite: 20 pair × 8 PC × 5 α × 4 cand = 3200 gen × 2 ckpt = 6400 total, ~60 min

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

LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Eval config (LITE, smoke PASSED: PIP_PARTIAL_TOP8 n=3) ===
# smoke (3 pair × 8 PC × 5 α × 4 cand) PASSED pipeline.
# smoke verdict = PIP_PARTIAL_TOP8 (5/8 improve, median Δρ=+0.076, no sign reverse)
# 升 lite (20 pair) 验证 verdict 稳健性:
# lite: 20 pair × 8 PC × 5 α × 4 cand = 3200 gen × 2 ckpt = 6400 total, ~60 min
N_PAIRS = 20                      # LITE (升自 smoke 3 pair, 用户确认 top-8 扩量)
N_CANDIDATES = 4
N_PCS = 8                         # PC0 ~ PC7 (top-8 by explained variance)
ALPHAS = [-2.0, -1.0, 0.0, +1.0, +2.0]
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 50
SEED = 42

CKPT_PATHS = [
    ("Phase 6.A (init)", SCRATCH / "pca48_ckpt" / "projector_final.pt"),
    ("PIP final", SCRATCH / "pca48_pip_indirect_ckpt" / "pip_indirect_final.pt"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pip_eval_top8] {msg}", flush=True)


def main():
    log("=== Phase 6.C.1-eval-top8 — top-8 PC directional audit ===")

    # Load Gaussians
    gauss = json.load(open(GAUSS_PATH))
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    explained_var = gauss.get("explained_variance", None)
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])

    user_gauss = gauss["user_gaussians"] if "user_gaussians" in gauss else gauss.get("users", {})
    log(f"  loaded {len(user_gauss)} user Gaussians")

    # Show top-8 PCs explained variance
    if explained_var is not None:
        ev = np.asarray(explained_var)
        log(f"  top-{N_PCS} PCs explained variance:")
        for j in range(min(N_PCS, len(ev))):
            log(f"    PC{j}: {ev[j]:.4f} ({ev[j]/ev.sum()*100:.2f}%)")
        cumvar = ev[:N_PCS].sum() / ev.sum()
        log(f"  cumulative: {cumvar*100:.2f}%")

    # Load test pairs
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
    log(f"  {len(pair_data)} test pairs")

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

    # Load model
    from syntax_subspace_pca48_projector import Qwen2WithPrefix
    log("loading Qwen2-7B + LoRA + projector ...")
    model = Qwen2WithPrefix()

    total_gens_per_ckpt = len(pair_data) * N_PCS * len(ALPHAS) * N_CANDIDATES
    log(f"  per ckpt: {len(pair_data)} pairs × {N_PCS} PCs × {len(ALPHAS)} α × {N_CANDIDATES} cands = {total_gens_per_ckpt} generations")

    # Per ckpt results
    results_per_ckpt = {}

    for ckpt_name, ckpt_path in CKPT_PATHS:
        if not Path(ckpt_path).exists():
            log(f"  skip {ckpt_name}: not found at {ckpt_path}")
            continue
        log(f"\n=== Eval ckpt: {ckpt_name} ({ckpt_path}) ===")
        model.load_projector(str(ckpt_path))
        model.eval()

        t0 = time.time()
        results = {j: {a: [] for a in ALPHAS} for j in range(N_PCS)}

        for pi, pd in enumerate(pair_data):
            for pc_j in range(N_PCS):
                for alpha in ALPHAS:
                    z_test = pd["z_target"].copy()
                    z_test[pc_j] += alpha

                    z_tensor = torch.from_numpy(z_test.astype(np.float32)).unsqueeze(0).to("cuda")
                    zs_expanded = z_tensor.expand(N_CANDIDATES, -1).contiguous()
                    attrs_list = [pd["attrs"]] * N_CANDIDATES
                    torch.manual_seed(SEED + pi * 100 + pc_j * 10 + int((alpha + 2) * 10))
                    candidates = model.generate(
                        zs_expanded, attrs_list,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        do_sample=True,
                    )

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

            elapsed = time.time() - t0
            log(f"    [{ckpt_name}] pair {pi+1}/{len(pair_data)} elapsed={elapsed:.0f}s")

        log(f"  ckpt eval done in {time.time() - t0:.0f}s")
        results_per_ckpt[ckpt_name] = results

    # Compute per-PC per-ckpt metrics
    log("\n\n=== METRICS per ckpt per PC ===")
    metrics_per_ckpt = {}

    for ckpt_name, results in results_per_ckpt.items():
        log(f"\n[{ckpt_name}]")
        per_pc_metrics = {}
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

            per_pc_metrics[pc_j] = {
                "rho_dir": float(rho_dir),
                "pval_dir": float(pval_dir),
                "sign_accuracy": float(sign_acc),
                "sign_correct": sign_correct,
                "sign_total": sign_total,
            }
        metrics_per_ckpt[ckpt_name] = per_pc_metrics

    # Paired lift Δρ_j = ρ_j^PIP - ρ_j^6A
    log("\n\n=== PAIRED LIFT (PIP final - Phase 6.A init) ===")
    paired = {}
    if "Phase 6.A (init)" in metrics_per_ckpt and "PIP final" in metrics_per_ckpt:
        a_metrics = metrics_per_ckpt["Phase 6.A (init)"]
        p_metrics = metrics_per_ckpt["PIP final"]
        deltas = []
        for j in range(N_PCS):
            a_rho = a_metrics[j]["rho_dir"]
            p_rho = p_metrics[j]["rho_dir"]
            a_sig = a_metrics[j]["sign_accuracy"]
            p_sig = p_metrics[j]["sign_accuracy"]
            d_rho = p_rho - a_rho
            d_sig = p_sig - a_sig
            deltas.append(d_rho)
            paired[j] = {
                "rho_6a": a_rho,
                "rho_pip": p_rho,
                "delta_rho": d_rho,
                "sign_6a": a_sig,
                "sign_pip": p_sig,
                "delta_sign": d_sig,
            }
            log(f"  PC{j}: 6A ρ={a_rho:+.3f} → PIP ρ={p_rho:+.3f} (Δ={d_rho:+.3f}); "
                f"sign 6A={a_sig*100:.0f}% → PIP={p_sig*100:.0f}% (Δ={d_sig*100:+.0f}pp)")

        deltas_arr = np.array(deltas)
        mean_d = float(np.mean(deltas_arr))
        median_d = float(np.median(deltas_arr))
        n_improve = int(np.sum(deltas_arr > 0))
        n_reversed = int(np.sum((np.array([p_metrics[j]["rho_dir"] for j in range(N_PCS)]) < 0) &
                                (np.array([a_metrics[j]["rho_dir"] for j in range(N_PCS)]) > 0)))
        n_sign_reversed = int(np.sum((np.array([p_metrics[j]["sign_accuracy"] for j in range(N_PCS)]) < 0.4) &
                                       (np.array([a_metrics[j]["sign_accuracy"] for j in range(N_PCS)]) > 0.6)))

        log(f"\n  Paired lift statistics:")
        log(f"    mean Δρ = {mean_d:+.3f}")
        log(f"    median Δρ = {median_d:+.3f}")
        log(f"    # PCs improve (Δρ > 0): {n_improve}/{N_PCS}")
        log(f"    # PCs ρ reversed (6A > 0 → PIP < 0): {n_reversed}/{N_PCS}")
        log(f"    # PCs sign reversed (6A > 60% → PIP < 40%): {n_sign_reversed}/{N_PCS}")

        paired["_summary"] = {
            "mean_delta_rho": mean_d,
            "median_delta_rho": median_d,
            "n_improve": n_improve,
            "n_reversed": n_reversed,
            "n_sign_reversed": n_sign_reversed,
        }

    # Final verdict (用户定义)
    log("\n\n=== VERDICT ===")
    if "_summary" in paired:
        s = paired["_summary"]
        log(f"  mean Δρ={s['mean_delta_rho']:+.3f}, median Δρ={s['median_delta_rho']:+.3f}")
        log(f"  {s['n_improve']}/{N_PCS} PCs improve, {s['n_reversed']} reversed")
        if s["n_improve"] >= N_PCS * 3 // 4 and s["median_delta_rho"] > 0 and s["n_reversed"] <= 1:
            verdict = "PIP_GO_LOCAL"
            rationale = (f"PIP route GO: {s['n_improve']}/{N_PCS} PCs improve + median Δρ>0 + {s['n_reversed']} reversed. "
                         f"Continue tuning (steps/λ).")
        elif s["n_improve"] <= N_PCS // 4:
            verdict = "PIP_LOCAL_ONLY"
            rationale = (f"Only {s['n_improve']}/{N_PCS} PCs improve. PIP improvement is local. "
                         f"Stop PIP route, switch to Phase 6.C.2 CIE K-anchor.")
        elif s["n_improve"] >= N_PCS // 2 and s["median_delta_rho"] >= 0:
            verdict = "PIP_PARTIAL_TOP8"
            rationale = (f"PIP partial: {s['n_improve']}/{N_PCS} improve, median Δρ={s['median_delta_rho']:+.3f}. "
                         f"Marginal evidence, consider scale-up or specific PC investigation.")
        else:
            verdict = "PIP_NOISE"
            rationale = f"PIP effect noisy. Re-run with n=20 (lite) or more pairs."
        log(f"  VERDICT: {verdict}")
        log(f"  {rationale}")
        paired["_verdict"] = verdict
        paired["_rationale"] = rationale

    # Save
    out_path = LOG_DIR / "phase6c1_pip_eval_top8.json"
    serializable = {
        "config": {
            "n_pairs": N_PAIRS,
            "n_pcs": N_PCS,
            "alphas": ALPHAS,
            "n_candidates": N_CANDIDATES,
            "seed": SEED,
        },
        "per_ckpt_per_pc": {
            ckpt: {str(j): m for j, m in mets.items()}
            for ckpt, mets in metrics_per_ckpt.items()
        },
        "paired_lift": {str(j): v for j, v in paired.items() if not str(j).startswith("_")},
        "paired_summary": paired.get("_summary", {}),
        "verdict": paired.get("_verdict", "UNKNOWN"),
        "rationale": paired.get("_rationale", ""),
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()