#!/usr/bin/env python3
"""Phase 6.B.4-eval — CC training diagnostic.

用户指令 2026-08-30: Phase 6.A ρ=-0.04 (no z_t preference). CC 训练后需要验证:
1. PC sweep ρ median (target=correct z 越远的 z_t 应该 log P 越低)
2. target/anti Δd — 生成在正确 z_t 下 d_self 低, 在 anti z_t 下 d_self 高
3. absolute d_self 整体水平

如果 PC sweep ρ 还是 0 / target vs anti 无差异 → CC loss 没起作用 → 退路.

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

# === Eval config (locked) ===
N_EVAL = 20               # test pairs
N_CANDIDATES = 4          # candidates per (pair, z_test)
N_PC_TEST = 4             # PC sweep over 4 PC strengths × target direction
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 50
SEED = 42
PCA_DIM = 48

# Multiple ckpts to compare (init vs final for speed)
CKPT_PATHS = [
    ("Phase 6.A (init)", SCRATCH / "pca48_ckpt" / "projector_final.pt"),
    ("CC final", CKPT_DIR / "cc_final.pt"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [cc_eval] {msg}", flush=True)


def main():
    log("=== Phase 6.B.4-eval — CC training diagnostic ===")

    # Load Gaussians (for z_t construction)
    gauss = json.load(open(GAUSS_PATH))
    pca_components = np.asarray(gauss["pca_components"], dtype=np.float64)
    pca_mean = np.asarray(gauss["pca_mean"], dtype=np.float64)
    fnames_all = gauss["feature_names_ordered"]
    fnames_f3 = gauss["fnames_f3"]
    col_idx_f3 = [fnames_all.index(n) for n in fnames_f3]
    scaler_mean = np.asarray(gauss["scaler_mean"])
    scaler_scale = np.asarray(gauss["scaler_scale"])

    # Load user Gaussians
    user_gauss = gauss["user_gaussians"] if "user_gaussians" in gauss else gauss.get("users", {})
    if not user_gauss:
        log("ERROR: no user_gaussians in gauss json")
        return

    log(f"  loaded {len(user_gauss)} user Gaussians")

    # Build test cohort
    samples = []
    with open(DATA_DIR / "train.jsonl") as f:
        for line in f:
            samples.append(json.loads(line))
    log(f"  {len(samples)} train samples available")

    # Pick N_EVAL test pairs
    rng = np.random.default_rng(SEED)
    test_pairs = rng.choice(len(samples), size=min(N_EVAL, len(samples)), replace=False)
    test_pairs = [samples[int(i)] for i in test_pairs]

    # For each pair, find user_id and pull mu from gauss
    pair_data = []
    for s in test_pairs:
        uid = s.get("user_id") or s.get("user")
        if uid is None:
            continue
        if uid not in user_gauss:
            continue
        ug = user_gauss[uid]
        mu = np.asarray(ug.get("mu", ug.get("mu_48")), dtype=np.float64)
        if mu.shape != (48,):
            continue
        pair_data.append({
            "user_id": uid,
            "attrs": s["c_i"],
            "z_target": mu,
        })
    pair_data = pair_data[:N_EVAL]
    log(f"  {len(pair_data)} test pairs with valid z_target")

    if not pair_data:
        log("ERROR: no valid pairs for eval")
        return

    # Build z_test matrix:
    #   z_target (correct)
    #   z_anti = 2*global_mean - z_target
    #   z_pc_sweep = z_target + α * PC_i for i in {0, 1, 2, 3}, α ∈ {-2, -1, +1, +2}
    cohort_z = np.stack([np.asarray(user_gauss[u]["mu"], dtype=np.float64) for u in user_gauss])
    global_mean = cohort_z.mean(axis=0)

    # For each pair, construct 6 z conditions: target, anti, pc0 sweep (4 alphas)
    z_conditions_per_pair = []  # list of lists, one per pair
    for pd in pair_data:
        z_t = pd["z_target"]
        z_anti = 2 * global_mean - z_t
        conds = [
            {"label": "target", "z": z_t},
            {"label": "anti", "z": z_anti},
        ]
        for pc_i in range(1):
            for alpha in [-2.0, -1.0, +1.0, +2.0]:
                z_pc = z_t.copy()
                z_pc[pc_i] += alpha
                conds.append({"label": f"pc{pc_i}_{alpha:+.1f}", "z": z_pc})
        z_conditions_per_pair.append(conds)

    n_conds = len(z_conditions_per_pair[0])
    total_gens = n_conds * len(pair_data) * N_CANDIDATES * len([c for c in CKPT_PATHS if Path(c[1]).exists()])
    log(f"  {n_conds} z_conditions × {len(pair_data)} pairs "
        f"× {N_CANDIDATES} candidates × {len(CKPT_PATHS)} ckpts = {total_gens} generations")

    # Load F3 helper for computing d_self of generated query
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

    # Load model ONCE, evaluate all ckpts via save/load
    from syntax_subspace_pca48_projector import Qwen2WithPrefix
    log("loading Qwen2-7B + LoRA + projector ...")
    model = Qwen2WithPrefix()

    # Eval per ckpt
    results_per_ckpt = {}
    for ckpt_name, ckpt_path in CKPT_PATHS:
        if not Path(ckpt_path).exists():
            log(f"  skip {ckpt_name}: not found at {ckpt_path}")
            continue
        log(f"\n=== Eval ckpt: {ckpt_name} ({ckpt_path}) ===")
        model.load_projector(str(ckpt_path))
        model.eval()

        # Generate per (pair, z_condition)
        all_results = []  # (pair_idx, label, z_norm, mean_d_self, generated_query)
        n_total = len(pair_data) * n_conds
        t_start_ckpt = time.time()
        for pi, pd in enumerate(pair_data):
            for cond in z_conditions_per_pair[pi]:
                if cond["z"].shape != (48,):
                    continue
                # 1 z × N_CANDIDATES
                z_tensor = torch.from_numpy(cond["z"].astype(np.float32)).unsqueeze(0).to("cuda")
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

                # Compute d_self for each candidate
                d_self_list = []
                z_query_list = []
                for cand in candidates:
                    zq = query_to_z(cand)
                    if zq is not None:
                        d = float(np.linalg.norm(zq - pd["z_target"]))
                        d_self_list.append(d)
                        z_query_list.append(zq)
                mean_d = float(np.mean(d_self_list)) if d_self_list else None
                all_results.append({
                    "pair_idx": pi,
                    "label": cond["label"],
                    "z_norm": float(np.linalg.norm(cond["z"])),
                    "mean_d_self": mean_d,
                    "n_valid": len(d_self_list),
                })
            # Progress log every 5 pairs
            if (pi + 1) % 5 == 0:
                elapsed = time.time() - t_start_ckpt
                log(f"    [{ckpt_name}] pair {pi+1}/{len(pair_data)} "
                    f"({(pi+1)*n_conds}/{n_total} done) "
                    f"elapsed={elapsed:.0f}s ({(elapsed)/((pi+1)*n_conds):.2f}s/gen)")

        # Aggregate per label
        labels = sorted(set(r["label"] for r in all_results))
        per_label_stats = {}
        for lbl in labels:
            d_vals = [r["mean_d_self"] for r in all_results if r["label"] == lbl and r["mean_d_self"] is not None]
            if d_vals:
                per_label_stats[lbl] = {
                    "n": len(d_vals),
                    "mean": float(np.mean(d_vals)),
                    "median": float(np.median(d_vals)),
                    "std": float(np.std(d_vals)),
                }
        log(f"\n  per-label d_self stats for {ckpt_name}:")
        for lbl in ["target", "anti"] + [f"pc{i}_{a:+.1f}" for i in range(3) for a in [-2.0, -1.0, +1.0, +2.0]]:
            if lbl in per_label_stats:
                st = per_label_stats[lbl]
                log(f"    {lbl:12s}: median={st['median']:.2f}  mean={st['mean']:.2f}  (n={st['n']})")

        # PC sweep ρ: median d_self should INCREASE with |α| (further from target)
        from scipy.stats import spearmanr
        pc_sweep_data = {}
        for pc_i in range(1):
            for lbl in [f"pc{pc_i}_{a:+.1f}" for a in [-2.0, -1.0, +1.0, +2.0]]:
                if lbl in per_label_stats:
                    alpha = float(lbl.split("_")[1])
                    pc_sweep_data.setdefault(pc_i, []).append((alpha, per_label_stats[lbl]["median"]))

        rho_per_pc = {}
        for pc_i, items in pc_sweep_data.items():
            alphas = np.array([x[0] for x in items])
            d_meds = np.array([x[1] for x in items])
            rho, pval = spearmanr(alphas, d_meds)
            rho_per_pc[pc_i] = (float(rho), float(pval))
            log(f"  PC{pc_i}: ρ(|α|, d_self median) = {rho:+.3f}  p={pval:.3f}")

        # target vs anti
        if "target" in per_label_stats and "anti" in per_label_stats:
            t = per_label_stats["target"]["median"]
            a = per_label_stats["anti"]["median"]
            delta = t - a  # negative = target better than anti (closer)
            log(f"  target/anti: target={t:.2f}  anti={a:.2f}  Δd(target-anti)={delta:+.2f}")
        else:
            delta = None

        results_per_ckpt[ckpt_name] = {
            "per_label": per_label_stats,
            "rho_per_pc": {k: list(v) for k, v in rho_per_pc.items()},
            "target_anti_delta": delta,
        }

    # Summary
    log("\n\n=== SUMMARY across ckpts ===")
    log(f"{'Ckpt':25s}  {'d_target':>9s}  {'d_anti':>9s}  {'Δ(t-a)':>8s}  {'ρ_PC0':>7s}")
    for ckpt_name, res in results_per_ckpt.items():
        d_t = res["per_label"].get("target", {}).get("median", float("nan"))
        d_a = res["per_label"].get("anti", {}).get("median", float("nan"))
        dt_a = res["target_anti_delta"] if res["target_anti_delta"] is not None else float("nan")
        rho0 = res["rho_per_pc"].get(0, (float("nan"),))[0]
        log(f"{ckpt_name:25s}  {d_t:9.2f}  {d_a:9.2f}  {dt_a:+8.2f}  {rho0:+7.3f}")

    # Save
    out_path = LOG_DIR / "phase6b4_cc_eval.json"
    # Convert to JSON-safe
    out = {}
    for k, v in results_per_ckpt.items():
        out[k] = {
            "per_label": v["per_label"],
            "rho_per_pc": {pk: list(pv) for pk, pv in v["rho_per_pc"].items()},
            "target_anti_delta": v["target_anti_delta"],
        }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"\nwrote → {out_path}")


if __name__ == "__main__":
    main()
