# Iteration #100 — Audit infrastructure milestone retrospective + next-phase pivot

**日期**: 2026-07-21
**scope**: retrospective on iter #82-#99 (18 iters), pivot to infrastructure execution phase
**prior**: iter #99 completed paper top-down audit navigation; audit infrastructure as deliverable is now COMPLETE

## §A Milestone summary

iter #82-#99 (18 iters, 2026-07-21) 完成了 paper audit infrastructure 作为 deliverable:

### Audit infrastructure layer (6 iters)

| iter | Deliverable | Status |
|------|------------|--------|
| #82 | `paper_claims_audit.py` v1 (12 §3 claims) | ✅ |
| #83 | MAE/RMSE on 5-point scale impl + RQ3_MAE_0.89 verified | ✅ |
| #84 | LLM full-set eval 3 system prompts + RQ3_LLM_Full_Set_94% verified | ✅ |
| #85 | 10-round × 10-candidate regeneration + Pipeline_Regeneration_10x10 verified | ✅ |
| #86 | RQ4 data lineage gap documented + compute_prior_bic_aic.py restored | ✅ |
| #93 | value-validation layer (file+value check) + new status enum | ✅ |

### Paper text update layer (9 iters)

| iter | Paper change | Status |
|------|-------------|--------|
| #86 | §5 #4 wording (Stage 12 outputs missing) | ✅ |
| #90 | §5 Limitations 3 → 5 limitations | ✅ |
| #91 | §2.2 audit (6 new claims) | ✅ |
| #92 | §2.2 K=8 + BIC/AIC/silhouette text tightening | ✅ |
| #94 | §3 Table 2 footnote (4 discrepant metrics) + §5 #5 wording | ✅ |
| #95 | §3 Table 1 footnote (Stage 6 query pool size) | ✅ |
| #96 | §3 Table 3 footnote (Stage 12 outputs missing) | ✅ |
| #98 | §5 Limitations backward cross-refs to Table footnotes (#3 #4) | ✅ |
| #99 | Abstract + §1 Reproducibility audit block | ✅ |

### Cross-reference layer (1 iter)

| iter | Cross-reference work | Status |
|------|---------------------|--------|
| #97 | Table 1/2/3 footnotes ↔ Table 1/2/3 footnotes (forward links) + audit ID verification | ✅ |

## §B Current audit state (iter #93)

```
verified=6, discrepant=4, degenerate=1, partial=1, unverified=5, blocked=0
total claims=17
```

| Status | Count | Resolves via |
|--------|-------|--------------|
| verified (boolean) | 6 | — (no reproduction gap) |
| discrepant (Table 2 metrics) | 4 | Re-run Stage 2 with 120-query annotation |
| degenerate (Table 1 bootstrap CI) | 1 | Stage 6/9 re-run with full query pool (≈1.5-3 h) |
| unverified (Table 3 + §2.2 infra) | 5 | Stage 12 re-run (≈9-21 h) |
| partial (env-var feature) | 1 | No re-run needed (by-design) |
| **blocked** | **0** | — |

## §C Paper state (iter #99)

```
PersonalQuery-Benchmark_evaluating_retrieval.md: 250 lines
```

Paper audit navigation infrastructure complete:
- Abstract (line 7): brief audit mention
- §1 Introduction (line 40): full Reproducibility audit block
- §3 Table 1 footnote (line 129): Stage 6 query pool size caveat
- §3 Table 2 footnote (line 146): 4 discrepant metrics with rel deltas
- §3 Table 3 footnote (line 173): 5 unverified claims list
- §5 Limitations (line 197): 3 backward cross-refs to inline footnotes
- All 3 inline footnotes cross-reference each other (iter #97)

## §D Audit infrastructure deliverables

### D.1 Code (in repo)

- `PersoanlQuery/paper_claims_audit.py` (592 lines)
  - 17 claims registered with code_evidence + expected_outputs + value_check(s)
  - 5 status types: verified_value_match, verified, discrepant, degenerate, partial, unverified, blocked
  - `_extract_value` (dotted-path selector + mean/max/min aggregation, NaN-filtered)
  - `_validate_value_check` (abs_tolerance + rel_tolerance)
  - `_resolve_file` (<cat> placeholder resolver)
- `PersoanlQuery/agreement_metrics.py` (extended iter #83): fleiss_kappa, cohens_kappa, cohens_weighted_kappa, mean_absolute_error, root_mean_squared_error

### D.2 Paper text changes (in repo)

- `PersonalQuery-Benchmark_evaluating_retrieval.md` (250 lines)
  - Abstract audit mention
  - §1 Reproducibility audit block
  - §2.2 K=8 + BIC/AIC/silhouette wording tightening
  - §3.3 Table 2 footnote (4 discrepant metrics)
  - §3 Table 1 footnote (Stage 6 query pool caveat)
  - §3 Table 3 footnote (Stage 12 outputs missing)
  - §5 Limitations 5 numbered items with inline footnote cross-refs

### D.3 Audit data (in result/)

- `result/personal_query/iterations/paper_claims_audit.json` (auto-generated, gitignored)
  - 17 claim entries with status + reason + value_check_results
  - Status summary block (verified/discrepant/degenerate/partial/unverified/blocked counts)

### D.4 Documentation (in iterations/)

- 18 iter docs covering iter #82-#99 with §A-F structure
- loop.md §10 audit trail with one-line summary per iter

## §E Next-phase pivot: infrastructure execution

Audit infrastructure as deliverable is now COMPLETE. Remaining work is infrastructure execution to flip claims:

### E.1 iter #87 — Stage 12 + Stage 10 pilot (5 unverified → verified)

- Re-run `train_vades_lite_sentence_latent_threshold.py` per category on GPU
- Persist `user_profiles.jsonl` + `sentences.jsonl` per category
- Re-run `cluster_strict5550_query_gmm_and_attach_retrieval.py` per category
- Re-run `compute_prior_bic_aic.py` with persisted Stage 12 outputs
- Estimated cost: ≈45-135 min/category × 3 = ≈9-21 h total
- Flip 5 claims: RQ4_GMM_Best_Prior, Sec2_20dim_syntactic_features, Sec2_K8_clusters_BIC_AIC_silhouette, Sec2_K2_GMM_per_user, Sec2_95pct_holdout_threshold

### E.2 iter #88 — Stage 6/9 re-run (1 degenerate → verified)

- Re-run Stage 6 retrieval with full query pool (Baby 95 / Grocery 89 / Pet 97)
- Re-run Stage 7 noisy retrieval with full pool
- Re-run Stage 8 bootstrap_delta_ci with full per-query records
- Estimated cost: ≈30-60 min/category × 3 = ≈1.5-3 h total
- Flip 1 claim: RQ1_Table1_Hit10 (degenerate → verified)

### E.3 iter #101+ — re-run audit + update paper

- After iter #87 / #88, re-run `paper_claims_audit.py` to capture new state
- Update Table 3 footnote (iter #96): mark "5 unverified" → "0 unverified (all verified after iter #87)"
- Update Table 1 footnote (iter #95): mark "degenerate" → "verified (after iter #88 full-pool re-run)"
- Update Table 2 footnote (iter #94): Table 2 metric deltas may shrink with larger sample (iter #102?)

## §F Backlog ordering (next 10 iters)

| Order | iter | Action | Estimated cost |
|-------|------|--------|---------------|
| 1 | #87 | Stage 12 + Stage 10 pilot (flip 5 unverified) | 9-21 h GPU |
| 2 | #88 | Stage 6/9 re-run with full pool (flip 1 degenerate) | 1.5-3 h GPU |
| 3 | #101 | Re-run audit + update Table 1/3 footnotes | < 1 h CPU |
| 4 | #102 | Re-collect Table 2 human-eval with larger sample (flip 4 discrepant) | 4-6 h human |
| 5 | #103 | Add §6 Conclusion with audit reproducibility statement | < 1 h |
| 6 | #104 | Refactor paper_claims_audit.py to CLI-arg-driven (--claim-id, --verbose) | 2-3 h dev |
| 7 | #105 | Add audit regression test (assert no claim flips silently) | 1-2 h dev |
| 8 | #106 | README.md sync to reflect audit infrastructure | < 1 h |
| 9 | #107 | Add CI hook to run paper_claims_audit.py on every commit | 2-3 h dev |
| 10 | #108 | paper_claims_audit dashboard (HTML visualization of audit state) | 3-4 h dev |

## §G File & command

- 模块: `PersoanlQuery/iterations/iter_100_audit_infra_milestone_retrospective.md`
- 命令: 直接 markdown 文档编辑
- 验证: `ls PersoanlQuery/iterations/iter_{82..100}*.md | wc -l` (应该 ≥ 19)

## §H 与 loop.md §8 的关系

iter #100 完成 audit infrastructure milestone retrospective + 下一阶段 pivot:
- 18 iters (#82-#99) audit infra 完整 (paper + code + data + docs)
- 下一阶段 pivot 到 infrastructure execution (#87 Stage 12, #88 Stage 6/9)
- 10-iter backlog 定义清楚 (execution + paper update + dev tooling)

后续 candidate:
- **iter #87** — Stage 12 + Stage 10 pilot (9-21 h GPU, flip 5 unverified)
- **iter #88** — Stage 6/9 re-run with full pool (1.5-3 h GPU, flip 1 degenerate)
- **iter #101** — re-run audit + update paper Table 1/3 footnotes