#!/usr/bin/env python3
"""Phase 6.C.1-eval — 5-alpha sweep for PIP-Indirect ckpt.

用户指令 2026-08-30: 验证 Phase 6.C.1 PIP-Indirect training 是否保留 Phase 6.A
(+0.251 ρ PC0) conditioning 并加 directional lift。

Eval 配置 (SMOKE per Rule 18):
- N_PAIRS = 3 (smoke)
- N_PCS = 2 (PC0, PC1)
- ALPHAS = [-2.0, -1.0, 0.0, +1.0, +2.0] (5 alphas)
- N_CANDIDATES = 4
- 2 ckpts: Phase 6.A init + PIP final

输出:
- /home/wlia0047/hj82_scratch2/wenyu/logs/phase6c1_pip_eval.json
- per PC per ckpt: ρ(α, PC_j(z_query)), sign_acc, leakage

GO 标准 (用户):
- ρ_PC0 > 0.4
- 不退化(> 0.251 baseline)
- 不出现 +0.251 → negative

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

# === Eval config (LITE, after SMOKE passed: PC0 ρ=+0.393 / sign=100%) ===
# smoke (3 pair) PASSED, 升到 lite 验证 verdict 是否稳健
# lite: 20 pair × 2 PC × 5 α × 4 cand = 800 gen × 2 ckpt = 1600 total, ~10 min
N_PAIRS = 20
N_CANDIDATES = 4
N_PCS = 2                       # PC0, PC1
ALPHAS = [-2.0, -1.0, 0.0, +1.0, +2.0]   # 5 alphas
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 50
SEED = 42

CKPT_PATHS = [
    ("Phase 6.A (init)", SCRATCH / "pca48_ckpt" / "projector_final.pt"),
    ("PIP final", SCRATCH / "pca48_pip_indirect_ckpt" / "pip_indirect_final.pt"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [pip_eval] {msg}", flush=True)


def main():
    log("=== Phase 6.C.1-eval — 5-alpha sweep ===")

    # Load Gaussians
    gauss = json.load(open(GAUSS_PATH))
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)  # (48, 103)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])

    user_gauss = gauss["user_gaussians"] if "user_gaussians" in gauss else gauss.get("users", {})
    log(f"  loaded {len(user_gauss)} user Gaussians")

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
            z = (f3_scaled - pca_mean) @ pca_components.T  # (48,)
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
                    torch.manual_seed(SEED + pi)
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
                        z_mean = np.mean(z_query_list, axis=0)  # (48,)
                        results[pc_j][alpha].append({
                            "pair_idx": pi,
                            "z_query_mean": z_mean.tolist(),
                        })

            elapsed = time.time() - t0
            log(f"    [{ckpt_name}] pair {pi+1}/{len(pair_data)} elapsed={elapsed:.0f}s")

        log(f"  ckpt eval done in {time.time() - t0:.0f}s")
        results_per_ckpt[ckpt_name] = results

    # Compute metrics
    log("\n\n=== METRICS per ckpt per PC ===")
    metrics_per_ckpt = {}

    for ckpt_name, results in results_per_ckpt.items():
        log(f"\n[{ckpt_name}]")
        per_pc_metrics = {}
        for pc_j in range(N_PCS):
            log(f"\n  PC{pc_j}:")
            # 1. Direction ρ(α, mean PC_j(z_query))
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
            log(f"    1. Direction ρ(α, PC_j(z_query)) = {rho_dir:+.3f}  p={pval_dir:.4f}")

            # 2. Sign accuracy: P(PC_j at α=+2 > PC_j at α=-2) per pair
            sign_correct = 0
            sign_total = 0
            for pi, alpha_dict in per_pair_pc_j.items():
                if +2.0 in alpha_dict and -2.0 in alpha_dict:
                    sign_total += 1
                    if alpha_dict[+2.0] > alpha_dict[-2.0]:
                        sign_correct += 1
            sign_acc = sign_correct / max(sign_total, 1)
            log(f"    2. Sign accuracy: {sign_correct}/{sign_total} = {sign_acc*100:.1f}%")

            # 3. Cross-PC leakage
            leakage_per_pair = []
            for pi, alpha_dict in per_pair_pc_j.items():
                z_means_per_alpha = []
                for alpha in ALPHAS:
                    if alpha in alpha_dict:
                        for entry in results[pc_j][alpha]:
                            if entry["pair_idx"] == pi:
                                z_means_per_alpha.append(entry["z_query_mean"])
                                break
                if len(z_means_per_alpha) >= 2:
                    z_arr = np.array(z_means_per_alpha)
                    non_j = [k for k in range(48) if k != pc_j]
                    non_j_stds = np.std(z_arr[:, non_j], axis=0)
                    leakage_per_pair.append(np.mean(non_j_stds))
            leakage_mean = float(np.mean(leakage_per_pair)) if leakage_per_pair else None
            log(f"    3. Cross-PC leakage: {leakage_mean:.4f}" if leakage_mean is not None else "    3. Leakage: N/A")

            per_pc_metrics[pc_j] = {
                "rho_dir": float(rho_dir),
                "pval_dir": float(pval_dir),
                "sign_accuracy": float(sign_acc),
                "sign_correct": sign_correct,
                "sign_total": sign_total,
                "cross_pc_leakage": leakage_mean,
            }

        # Aggregate
        n_direction_ok = sum(1 for m in per_pc_metrics.values()
                             if m["rho_dir"] > 0 and m["sign_accuracy"] > 0.6)
        n_reversed = sum(1 for m in per_pc_metrics.values()
                         if m["rho_dir"] < 0 and m["sign_accuracy"] < 0.4)

        log(f"\n  Summary: {n_direction_ok}/{N_PCS} PCs direction-correct, {n_reversed}/{N_PCS} PCs reversed")
        metrics_per_ckpt[ckpt_name] = {
            "per_pc": per_pc_metrics,
            "n_direction_ok": n_direction_ok,
            "n_reversed": n_reversed,
        }

    # Final summary table
    log("\n\n=== FINAL SUMMARY ===")
    log(f"{'Ckpt':22s}  {'PC0 ρ':>7s} {'PC0 sig':>7s}  {'PC1 ρ':>7s} {'PC1 sig':>7s}  {'#OK':>3s} {'#REV':>4s}")
    for ckpt_name, ckpt_metrics in metrics_per_ckpt.items():
        pcs_metrics = ckpt_metrics["per_pc"]
        row = [ckpt_name]
        for pc_j in range(N_PCS):
            m = pcs_metrics[pc_j]
            row.append(f"{m['rho_dir']:+7.3f}")
            row.append(f"{m['sign_accuracy']*100:6.1f}%")
        row.append(f"{ckpt_metrics['n_direction_ok']:3d}")
        row.append(f"{ckpt_metrics['n_reversed']:4d}")
        log("  ".join(f"{x:>7s}" if i == 0 else f"{x}" for i, x in enumerate(row)))

    # Verdict
    log("\n=== VERDICT (用户标准) ===")
    if metrics_per_ckpt:
        pip_metrics = metrics_per_ckpt.get("PIP final", {})
        pip_pcs = pip_metrics.get("per_pc", {})
        n_ok = pip_metrics.get("n_direction_ok", 0)
        n_rev = pip_metrics.get("n_reversed", 0)
        pc0 = pip_pcs.get(0, {})
        rho_pc0 = pc0.get("rho_dir", 0)
        sig_pc0 = pc0.get("sign_accuracy", 0)

        if rho_pc0 > 0.4 and sig_pc0 > 0.6:
            verdict = "DIRECTION_OK"
            rationale = f"PIP final: PC0 ρ={rho_pc0:+.3f} > 0.4 ✓, sign_acc={sig_pc0:.1%} > 60% ✓"
        elif rho_pc0 > 0 and sig_pc0 > 0.6 and rho_pc0 >= 0.251:
            verdict = "DIRECTION_PARTIAL"
            rationale = f"PIP final: PC0 ρ={rho_pc0:+.3f} > 0 但 < 0.4; sign_acc={sig_pc0:.1%}; no regression vs baseline"
        elif rho_pc0 < 0:
            verdict = "DIRECTION_REVERSED"
            rationale = f"PIP final: PC0 ρ={rho_pc0:+.3f} < 0; regression vs baseline +0.251"
        else:
            verdict = "DIRECTION_NOISE"
            rationale = f"PIP final: PC0 ρ={rho_pc0:+.3f}; weak signal"
        log(f"  {verdict}")
        log(f"  {rationale}")

    # Save
    out_path = LOG_DIR / "phase6c1_pip_eval.json"
    serializable = {}
    for ckpt_name, results in results_per_ckpt.items():
        serializable[ckpt_name] = {
            "per_pc": metrics_per_ckpt.get(ckpt_name, {}).get("per_pc", {}),
            "summary": {
                "n_direction_ok": metrics_per_ckpt.get(ckpt_name, {}).get("n_direction_ok", 0),
                "n_reversed": metrics_per_ckpt.get(ckpt_name, {}).get("n_reversed", 0),
            },
        }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()