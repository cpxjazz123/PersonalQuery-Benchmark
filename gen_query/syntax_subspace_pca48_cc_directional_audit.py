#!/usr/bin/env python3
"""Phase 6.B.4b — Full Directional Geometry Audit.

用户指令 2026-08-30:
PC0 ρ=-0.8 在 n=20 × 1 PC 下不稳健。验证"方向反转"是否稳定现象:
- 测试 top-N_PC (默认 4) PCs
- α ∈ {-2, 0, +2} (三档 alpha, 5 太慢)
- N=30 pairs (vs Phase 6.B.4 的 20)
- 4 candidates per (pair, PC, α)
- 2 ckpts: Phase 6.A init + CC final

3 个核心指标(per ckpt per PC):
1. 方向相关性: ρ(α, mean PC_j(z_query))
   真正成功应该 ρ > 0 (target → z_query 沿 PC_j 单调)
   Phase 6.A init PC0 ρ=+0.8 但只测 4 alphas (n=4)
2. Cross-PC leakage: 扰动 PC_j 时,非扰动 PCs 不应乱漂
   metric: 平均 std of mean PC_{≠j}(z_query) across α
3. Sign accuracy: P(mean PC_j at α=+2 > mean PC_j at α=-2)
   50% = random; >70% = good direction

参数全部硬编码 (Rule 3).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
SCRATCH = Path("/home/wlia0047/hj82_scratch2/wenyu")
DATA_DIR = SCRATCH / "pca48_dataset"
LOG_DIR = SCRATCH / "logs"
CKPT_DIR = SCRATCH / "pca48_cc_ckpt"
GAUSS_PATH = SCRATCH / "gaussian_vades" / "stage8_5_user_gaussians.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# === Audit config (LITE, user-approved 2026-08-30) ===
# smoke (3 pair × 1 PC × 2 α) 已通过: PC0 ρ +0.293 vs Phase 6.A -0.098 (lift!)
# smoke verdict = DIRECTION_OK 但 n=3 不可靠, 升到 lite 验证
# lite: 20 pair × 2 PC × 2 α = 320 gen × 2 ckpt = 640 total, ~5 min
# 全量 30 pair × 4 PC × 3 α 仅在 lite 通过 + 用户明确批准后才跑
N_PAIRS = 20
N_CANDIDATES = 4
N_PCS = 2               # lite 测试 PC0, PC1
ALPHAS = [-2.0, +2.0]   # 2 alphas (sign accuracy 直接可算)
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 50
SEED = 42

CKPT_PATHS = [
    ("Phase 6.A (init)", SCRATCH / "pca48_ckpt" / "projector_final.pt"),
    ("CC final", CKPT_DIR / "cc_final.pt"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [dir_audit] {msg}", flush=True)


def main():
    log("=== Phase 6.B.4b — Full Directional Geometry Audit ===")

    # Load Gaussians
    gauss = json.load(open(GAUSS_PATH))
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)  # (48, 103)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])

    user_gauss = gauss["users"] if "users" in gauss else gauss.get("user_gaussians", {})
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

    # Per ckpt results
    # Structure: results[ckpt_name][pc_j][alpha] = list of {pair_idx, z_query_mean (48,)}
    results_per_ckpt = {}
    total_gens_per_ckpt = len(pair_data) * N_PCS * len(ALPHAS) * N_CANDIDATES
    log(f"  per ckpt: {len(pair_data)} pairs × {N_PCS} PCs × {len(ALPHAS)} α × {N_CANDIDATES} cands = {total_gens_per_ckpt} generations")

    for ckpt_name, ckpt_path in CKPT_PATHS:
        if not Path(ckpt_path).exists():
            log(f"  skip {ckpt_name}: not found at {ckpt_path}")
            continue
        log(f"\n=== Eval ckpt: {ckpt_name} ({ckpt_path}) ===")
        model.load_projector(str(ckpt_path))
        model.eval()

        t0 = time.time()
        # results[pc_j][alpha] = list of z_query_means (per pair)
        # z_query_mean shape (48,)
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

                    # Compute z_query for each candidate
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

            if (pi + 1) % 5 == 0:
                elapsed = time.time() - t0
                log(f"    [{ckpt_name}] pair {pi+1}/{len(pair_data)} elapsed={elapsed:.0f}s")

        log(f"  ckpt eval done in {time.time() - t0:.0f}s")
        results_per_ckpt[ckpt_name] = results

    # Compute metrics
    log("\n\n=== METRICS per ckpt per PC ===")
    from scipy.stats import spearmanr

    metrics_per_ckpt = {}
    for ckpt_name, results in results_per_ckpt.items():
        log(f"\n[{ckpt_name}]")
        per_pc_metrics = {}
        for pc_j in range(N_PCS):
            log(f"\n  PC{pc_j}:")
            # 1. Direction correlation: ρ(α, mean PC_j(z_query))
            per_pair_pc_j = {}
            for alpha in ALPHAS:
                for entry in results[pc_j][alpha]:
                    pi = entry["pair_idx"]
                    per_pair_pc_j.setdefault(pi, {})[alpha] = entry["z_query_mean"][pc_j]

            # For ρ: list of (α, PC_j) per pair, then take pair-mean across alphas
            alphas_for_corr = []
            pcj_means_for_corr = []
            for pi, alpha_dict in per_pair_pc_j.items():
                # Average PC_j across alphas for this pair (should be ~constant if z_query doesn't drift)
                # But we want correlation across α values
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

            # 3. Cross-PC leakage: std of mean PC_{≠j}(z_query) across α per pair, then avg
            leakage_per_pair = []
            for pi, alpha_dict in per_pair_pc_j.items():
                # For each α, compute mean of PC_{≠j}
                # z_query_means at each α:
                z_means_per_alpha = []
                for alpha in ALPHAS:
                    if alpha in alpha_dict:
                        # Get z_query_mean at this α
                        for entry in results[pc_j][alpha]:
                            if entry["pair_idx"] == pi:
                                z_means_per_alpha.append(entry["z_query_mean"])
                                break
                if len(z_means_per_alpha) >= 2:
                    # For each non-perturbed PC, compute std across α
                    z_arr = np.array(z_means_per_alpha)  # (n_alphas, 48)
                    non_j = [k for k in range(48) if k != pc_j]
                    # std of each non-perturbed PC across alphas
                    non_j_stds = np.std(z_arr[:, non_j], axis=0)  # (47,)
                    leakage_per_pair.append(np.mean(non_j_stds))
            leakage_mean = float(np.mean(leakage_per_pair)) if leakage_per_pair else None
            log(f"    3. Cross-PC leakage (mean std of non-perturbed PCs across α): {leakage_mean:.4f}" if leakage_mean is not None else "    3. Leakage: N/A")

            per_pc_metrics[pc_j] = {
                "rho_dir": float(rho_dir),
                "pval_dir": float(pval_dir),
                "sign_accuracy": float(sign_acc),
                "sign_correct": sign_correct,
                "sign_total": sign_total,
                "cross_pc_leakage": leakage_mean,
            }

        # Aggregate: direction success = (rho_dir > 0 AND sign_acc > 0.6) per PC
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
    log(f"{'Ckpt':22s}  {'PC0 ρ':>7s} {'PC0 sig':>7s}  {'PC1 ρ':>7s} {'PC1 sig':>7s}  "
        f"{'PC2 ρ':>7s} {'PC2 sig':>7s}  {'PC3 ρ':>7s} {'PC3 sig':>7s}  "
        f"{'#OK':>3s} {'#REV':>4s}")
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
    log("\n=== VERDICT ===")
    if metrics_per_ckpt:
        cc_metrics = metrics_per_ckpt.get("CC final", {})
        cc_pcs = cc_metrics.get("per_pc", {})
        n_ok = cc_metrics.get("n_direction_ok", 0)
        n_rev = cc_metrics.get("n_reversed", 0)

        if n_ok == N_PCS:
            verdict = "DIRECTION_OK"
            rationale = (f"CC final: all {N_PCS}/{N_PCS} PCs show positive direction "
                         f"(ρ > 0 AND sign_acc > 60%). Phase 6.B.4 worked.")
        elif n_ok >= N_PCS // 2:
            verdict = "DIRECTION_PARTIAL"
            rationale = (f"CC final: {n_ok}/{N_PCS} PCs direction-correct, {n_rev}/{N_PCS} reversed. "
                         f"Partial geometry. Could scale or use Mirror/Ordered Loss.")
        elif n_rev >= N_PCS // 2:
            verdict = "DIRECTION_REVERSED"
            rationale = (f"CC final: {n_rev}/{N_PCS} PCs reversed direction. "
                         f"Confirmed Phase 6.B.4 issue. → Phase 6.B.5 Mirror/Ordered Geometry Loss.")
        else:
            verdict = "DIRECTION_NOISE"
            rationale = (f"CC final: {n_ok} OK / {n_rev} reversed / {N_PCS - n_ok - n_rev} random. "
                         f"Direction signal too weak to conclude. Need more data.")
        log(f"  {verdict}")
        log(f"  {rationale}")

    # Save
    out_path = LOG_DIR / "phase6b4b_directional_audit.json"
    # Convert z_query_mean lists to JSON-safe (they're already floats)
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