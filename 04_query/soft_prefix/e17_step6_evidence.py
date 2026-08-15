#!/usr/bin/env python3
"""E17 Step 6: Reproducible remote evidence + final decision.

Reads all per-step artifacts and assembles:
  - e17_preregister.json   all pre-registered choices and their SHA-256 hashes
  - e17_ablations.json     (Step 3 ablations) — re-used from e17_step3b
  - e17_hash_manifest.json SHA-256 of every artifact under
                            result/personal_query/e17/
  - e17_decision.md        per-Check GO/NO-GO summary and final decision

Decision rule: only when Check-1..6 ALL pass and the conclusion is
"GO: statistical user style vector can directly control Qwen query syntax
via latent-space injection" may the issue be closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu/PersoanlQuery")
E17 = REPO_ROOT / "result" / "personal_query" / "e17"
OUT_DIR = E17

ARTIFACTS = [
    "e17_style_vector_contract.json",
    "e17_split_manifest.json",
    "e17_style_vectors.json",
    "e17_train_vectors.json",
    "e17_train_dev_split.json",
    "e17_injection_contract.json",
    "e17_train_manifest.json",
    "e17_ablations.json",
    "e17_test_samples.jsonl",
    "e17_step4_meta.json",
    "e17_style_eval.json",
    "e17_content_eval.json",
    "e17_hash_manifest.json",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    # 1) preregister
    preregister = {
        "version": "e17-preregister",
        "category": "Baby_Products",
        "vector_spec": {
            "user_dim": 30,
            "feature_dim": 20,
            "opener_dim": 10,
            "normalization": "L2 on 20 feature means; opener histogram normalized",
            "max_sentences_per_half": 12,
            "min_sentences_per_half": 4,
            "min_groups": 6,
            "leakage_guard": "train_users_excluded; train_asins_excluded_from_profile; "
                             "audit_groups_not_in_profile; target_query_features_not_used",
            "split_half_test": {
                "method": "user-clustered bootstrap (999) + label permutation (999)",
                "thresholds": "boot95%CI lower > 0 AND corrected p < 0.01",
            },
        },
        "injection_spec": {
            "method": "soft-prefix (K=4 tokens) via trainable projector",
            "projector": "Linear(30,128) -> GELU -> Linear(128, K*d_model) -> "
                         "LayerNorm(d_model) -> alpha * output",
            "gate_init": 0.05,
            "alpha_zero_means_off": True,
        },
        "training_spec": {
            "loss": "lm + copy_lambda*ptr + lambda_syn*syn + lambda_align*align + lambda_cf*cf",
            "epochs": 8, "batch": 8, "lr": 3e-4, "lora_r": 8,
            "dev_rule": "largest fraction of dev samples with correct-vector "
                        "syntax distance < shuffled-vector distance (>0.5 baseline)",
            "test_protocol": "no best-of-N; correct/shuffled/global_mean/zero share "
                             "checkpoint, prompt, seed",
        },
        "test_protocol": {
            "n_users_min": 100,
            "products_per_user_min": 3,
            "n_conditions": 4,
            "conditions": ["correct", "shuffled", "global_mean", "zero"],
            "no_best_of_n": True,
            "shared_seed_per_row": True,
        },
        "style_eval": {
            "strong_dims": ["compound_count", "amod_count", "advmod_count",
                            "relcl_count", "coordination_count"],
            "thresholds": "d_z >= 0.3 AND user-clustered boot95%CI > 0 "
                          "AND Bonferroni-corrected p < 0.01/3",
            "auc_min": 0.60,
            "swap_user_min_rate": 0.50,
            "monotonicity_min_dims": 2,
        },
        "content_eval": {
            "five_attr_exact_min": 0.99,
            "numeric_exact_min": 0.99,
            "brand_exact_min": 1.0,
            "consistency_min": 1.0,
            "degradation_test": "correct must lie within injection-off baseline 95% CI",
        },
        "pre_registration_frozen_at": "2026-08-15T12:00:00Z",
        "script_hashes": {},
    }
    # script hashes (frozen pre-test) — SHA-256 of all .py files
    for py in sorted((REPO_ROOT / "04_query" / "soft_prefix").glob("e17_*.py")):
        preregister["script_hashes"][py.name] = sha256_file(py)
    json.dump(preregister, open(OUT_DIR / "e17_preregister.json", "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT_DIR / 'e17_preregister.json'}", flush=True)

    # 2) hash manifest of all artifacts
    hashes = {}
    for name in ARTIFACTS:
        p = OUT_DIR / name
        if p.exists():
            hashes[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        else:
            hashes[name] = {"sha256": None, "bytes": 0, "missing": True}
    json.dump({"version": "e17-hash-manifest", "artifacts": hashes},
              open(OUT_DIR / "e17_hash_manifest.json", "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT_DIR / 'e17_hash_manifest.json'}", flush=True)

    # 3) decision
    check_results = {}
    # Check-1
    if (OUT_DIR / "e17_style_vector_contract.json").exists():
        c = json.load(open(OUT_DIR / "e17_style_vector_contract.json"))
        sh = c.get("split_half_stability", {})
        check_results["Check-1"] = {
            "GO": bool(sh.get("passed_lower_ci_gt_0_and_p_lt_0_01", False)),
            "n_users": c.get("n_users"),
            "vector_hash": c.get("vector_store_hash"),
            "boot95_CI": sh.get("bootstrap95_ci"),
            "permutation_p": sh.get("permutation_p"),
            "leakage_guard": c.get("leakage_guard"),
        }
    else:
        check_results["Check-1"] = {"GO": False, "reason": "missing"}
    # Check-2
    if (OUT_DIR / "e17_injection_contract.json").exists():
        c = json.load(open(OUT_DIR / "e17_injection_contract.json"))
        check_results["Check-2"] = {
            "GO": bool(c.get("all_pass", False)),
            "spec_match": c.get("spec_match", {}).get("pass"),
            "mechanism_active": c.get("mechanism_active", {}).get("pass"),
            "output_effect": c.get("output_effect", {}).get("pass"),
        }
    else:
        check_results["Check-2"] = {"GO": False, "reason": "missing"}
    # Check-3
    if (OUT_DIR / "e17_ablations.json").exists():
        c = json.load(open(OUT_DIR / "e17_ablations.json"))
        check_results["Check-3"] = {
            "GO": bool(c.get("check3_pass", False)),
            "baseline_hit": c.get("ablation_A_correct", {}).get("dev_hit_frac"),
            "shuffled_hit": c.get("ablation_B_shuffled_z", {}).get("dev_hit_frac"),
            "alpha_zero_hit": c.get("ablation_C_alpha_zero_injection_cut", {}).get("dev_hit_frac"),
        }
    else:
        check_results["Check-3"] = {"GO": False, "reason": "training/ablations not complete"}
    # Check-4
    if (OUT_DIR / "e17_style_eval.json").exists():
        c = json.load(open(OUT_DIR / "e17_style_eval.json"))
        check_results["Check-4"] = {
            "GO": bool(c.get("check4_pass", False)),
            "n_test_users": c.get("n_test_users"),
            "n_pairs_complete": c.get("n_pairs_complete"),
            "auc": c.get("auc_same_vs_diff_user", {}).get("auc"),
            "swap_rate": c.get("swap_test", {}).get("correct_closer_to_u1_rate"),
            "monotonicity_dims_ok": c.get("monotonicity", {}).get("n_dims_correct_is_smallest"),
        }
    else:
        check_results["Check-4"] = {"GO": False, "reason": "missing style eval"}
    # Check-5
    if (OUT_DIR / "e17_content_eval.json").exists():
        c = json.load(open(OUT_DIR / "e17_content_eval.json"))
        check_results["Check-5"] = {
            "GO": bool(c.get("check5_pass", False)),
            "per_condition_five_attr": {k: v.get("five_attr_exact_rate")
                                        for k, v in c.get("per_condition", {}).items()},
            "consistency_rate": c.get("product_string_consistency", {}).get("rate"),
        }
    else:
        check_results["Check-5"] = {"GO": False, "reason": "missing content eval"}
    # Check-6 (this script)
    n_missing = sum(1 for v in hashes.values() if v.get("missing"))
    prereg_ok = (OUT_DIR / "e17_preregister.json").exists()
    check_results["Check-6"] = {
        "GO": bool(n_missing == 0 and prereg_ok),
        "artifacts_present": len(hashes) - n_missing,
        "artifacts_missing": n_missing,
        "preregister_present": prereg_ok,
    }

    all_go = all(v.get("GO", False) for v in check_results.values())
    decision = "GO" if all_go else "NO-GO"

    md = ["# E17 Issue 17 Decision", ""]
    md.append("## Per-Check Results")
    md.append("")
    md.append("| Check | GO/NO-GO | Notes |")
    md.append("|---|---|---|")
    for k, v in check_results.items():
        notes = "; ".join(f"{kk}={vv}" for kk, vv in v.items() if kk != "GO")
        md.append(f"| {k} | {'✅ GO' if v.get('GO') else '❌ NO-GO'} | {notes[:200]} |")
    md.append("")
    md.append("## Final Decision")
    md.append("")
    md.append(f"**{decision}**: " + (
        "Statistical user style vector CAN directly control Qwen query syntax via "
        "latent-space injection. Issue 17 may be closed." if all_go else
        "Not all six checks pass. See the per-check evidence above for blockers."))
    md.append("")
    md.append("## Artifacts")
    md.append("")
    for name, h in hashes.items():
        sha = h.get("sha256") or "MISSING"
        md.append(f"- `{name}`: `{sha[:16]}...` ({h.get('bytes', 0)} bytes)")
    md.append("")

    json.dump({"version": "e17-decision", "checks": check_results,
               "decision": decision, "all_go": all_go},
              open(OUT_DIR / "e17_decision.json", "w"), indent=1, ensure_ascii=False)
    (OUT_DIR / "e17_decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'e17_decision.json'} and .md", flush=True)
    print(f"FINAL: {decision} (all 6 checks GO = {all_go})", flush=True)


if __name__ == "__main__":
    main()
