# Iteration #93 — Paper claims audit: value-validation upgrade

**日期**: 2026-07-21
**scope**: `PersoanlQuery/paper_claims_audit.py` (upgrade from file-existence to value-validation)
**prior**: iter #82-#92 audit (17 claims, 11 verified by file-existence, 5 unverified, 1 partial)

## §A 审稿意见

iter #82-#92 audit 只检查 "expected_output 文件是否存在于磁盘" — 即 file-existence
heuristic。 这导致两类 false-positive verified:

1. **File exists but content degenerate**: 例如 `bootstrap_delta_ci.json` 存在但 1152/1152
   cell 都是 `NaN` (Stage 6 query pool 只有 2 queries, 不足以做 bootstrap)。 Paper §3
   Table 1 数字原本是在完整 query pool 上算的, 当前 release 是降级版 pilot。
2. **File exists with WRONG value**: 例如 `llm_human_eval_cross_domain.json` 里
   `fleiss_kappa_among_3_humans = 0.6235` 但 paper §3 报 0.72;
   `mae_llm_vs_mean_human = 0.62` 但 paper §3 报 0.89;
   `spearman_real_llm_label_vs_gt_label ≈ 0.71` 但 paper §3 报 0.81;
   `llm_full_set_quality_eval.json` 里 `semantic_preservation_after_error_injection = 0.205`
   但 paper §3 报 0.946 — 这是 iter #84 已记录的 preservation gap, 但 audit 没明确
   标记为 discrepant。

Reviewer 看 paper 数字, 对照 release code, 会发现 paper claim 与 code 输出**实质上不一致**
(file 存在 ≠ paper claim 值正确)。 iter #93 升级 audit 加 value-validation layer。

## §B 本轮 (iter #93) 改动

### §B.1 升级 audit 逻辑

`PersoanlQuery/paper_claims_audit.py` 增加:

1. **`_extract_value(file_path, selector)` helper**: 用 dotted path selector 从 JSON
   读值。 支持:
   - `path.to.scalar` — 单值 (返回 float)
   - `path.to.list[*].field` — 通配 list 内每个 element 的 field (返回 list[float])
   - `path.to.list[*].field|mean` — 同上 + aggregation (mean/max/min/sum)
   - `path.to.dict.*.field|mean` — 通配 dict value 内 field + aggregation
   - `3_domain_mean_pass_rate.semantic_plausibility` — 显式 path
   - 文件不存在 / selector 不命中 → return None
   - 任何 NaN/inf → raise 或 skip (走 degenerate path)

2. **`_validate_value(claim, matched_files)`**: 返回
   `(value_status, extracted, expected, abs_delta, rel_delta, note)`:
   - `value_match`: |delta| ≤ abs_tolerance AND rel_delta ≤ rel_tolerance
   - `value_mismatch` (→ status `discrepant`): 超出 tolerance
   - `degenerate`: extracted 全部 NaN/None/0 (file 存在但内容无效)
   - `value_check_skipped`: claim 没有 value_check field

3. **`evaluate_claim` 升级**: 在原 file-existence check 基础上叠加 value check:
   - `verified` + `value_match` → `verified_value_match`
   - `verified` + `value_mismatch` → `discrepant` (NEW status)
   - `verified` + `degenerate` → `degenerate` (NEW status)
   - `verified` + `value_check_skipped` → 保持 `verified`

### §B.2 增加 value_check field to claims

7 个有 paper 数字的 claims 加上 value_check:

| Claim ID | Expected | Selector | Tolerance |
|----------|----------|----------|-----------|
| RQ3_Fleiss_Kappa_0.72 | 0.72 | `[*].fleiss_kappa_among_3_humans\|mean` | abs 0.05, rel 10% |
| RQ3_Spearman_0.81 | 0.81 | `[*].spearman_real_llm_label_vs_gt_label\|mean` | abs 0.05, rel 10% |
| RQ3_MAE_0.89 | 0.89 | `summary.mae_llm_vs_mean_human` | abs 0.05, rel 10% |
| RQ3_LLM_Full_Set_94% (plausibility) | 0.967 | `3_domain_mean_pass_rate.semantic_plausibility` | abs 0.05, rel 10% |
| RQ3_LLM_Full_Set_94% (structure) | 0.973 | `3_domain_mean_pass_rate.target_structure_conformity` | abs 0.05, rel 10% |
| RQ3_LLM_Full_Set_94% (preservation) | 0.946 | `3_domain_mean_pass_rate.semantic_preservation_after_error_injection` | abs 0.10, rel 15% |
| RQ1_Table1_Hit10 | not NaN | 任何 retriever/metric cell | non-degenerate |

为 RQ1 bootstrap_delta_ci.json 加 degenerate check (提取所有 mean 值, 不应全部 NaN)。

### §B.3 更新 `audit_target` 头

```python
"audit_target": "Paper §2.1-§3 RQ1-4 + Table 1-3 (file existence + value validation)"
```

## §C 预期 findings

Pre-impl inspection (grep + head on output JSONs) 已确认:

| Claim | Expected | Actual | Pre-impl status |
|-------|----------|--------|-----------------|
| RQ3_Fleiss_Kappa_0.72 | 0.72 | 0.6235 (all 3 domains) | verified (false positive) |
| RQ3_Spearman_0.81 | 0.81 | 0.7164 (3-domain mean) | verified (false positive) |
| RQ3_MAE_0.89 | 0.89 | 0.62 | verified (false positive) |
| RQ3_LLM_Full_Set_94% plausibility | 0.967 | 0.893 | verified (false positive) |
| RQ3_LLM_Full_Set_94% structure | 0.973 | 0.953 | within tolerance → match |
| RQ3_LLM_Full_Set_94% preservation | 0.946 | 0.205 | verified (false positive, iter #84 已记) |
| RQ1_Table1_Hit10 (bootstrap CI) | non-NaN | all NaN (1152 cells) | verified (degenerate) |

iter #93 完成后 audit summary 应该是:
- **verified_value_match**: 6-7 (原 verified 中真正数字一致的)
- **discrepant**: 4-5 (Fleiss, Spearman, MAE, preservation, plausibility 之一)
- **degenerate**: 1 (RQ1 bootstrap CI)
- **partial**: 1 (Pipeline_Skip_ColBERTv2_SPLADE, no value check)
- **unverified**: 5 (Stage 12 lineage, unchanged)

## §D 与 §5 Limitations 的关系

iter #90 §5 Limitations 列出 5 limitations, 其中包括:
> "Paper §3 Table 2 numerical results are originally-computed values; the
> reproducible release pipeline reports slightly different numbers due to
> small query pool size (N=50 pilot, 21 noisy pairs) and BPE-aware error
> injection differences."

iter #93 audit 将 **量化** 这个 limitation: 每个 discrepancy 的 actual vs paper value
都记录在 audit JSON, reviewer 可以一眼看到 paper claim 与 release output 的具体差距。

## §E 文件 & 命令

- 模块: `PersoanlQuery/paper_claims_audit.py`
- 输出: `result/personal_query/iterations/paper_claims_audit.json`
- 命令: `python3 PersoanlQuery/paper_claims_audit.py`
- 验证: `python3 -m py_compile PersoanlQuery/paper_claims_audit.py` (Rule 10)
- 跑时间: < 1s

## §F 与 loop.md §8 的关系

iter #93 完成 audit value-validation 升级:
- Audit 不再只是 "file exists", 而是 "file exists AND value matches paper claim"
- 5 个原 verified 中 4 个降级为 discrepant, 1 个降级为 degenerate
- Reviewer 现在能看到具体 paper claim value vs release output value 差距

后续 candidate:
- **iter #94** — paper §3 Table 2 text caveat 补 discrepant value 表
- **iter #87** — Stage 12 + Stage 10 pilot (45-135 min GPU) — flip 5 unverified → verified
- **iter #88** — Stage 7 重跑 (multi-hour) — unblock RQ1/RQ2 real Δ CIs (不只是 un-NaN bootstrap)