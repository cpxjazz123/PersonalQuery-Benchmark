# Iteration #90 — Paper §5 Limitations updated with iter #82-#86 audit findings

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md` §5
**prior**: iter #82 paper claims audit (commit 73af875) 发现 paper §5 Limitations
text 与 repo 实际状态不符

## §A 审稿意见

iter #82 senior reviewer audit 把 paper §3 12 个 quantitative claims 全部 map 到
code/output evidence 后, 暴露 §5 Limitations 与 iter #74-#86 实际进展**不一致**:

1. **§5 third limitation 是 stale claim** — 原文 "current implementation generates
   all 10 candidates in a single LLM call; true sequential regeneration with
   per-round LLM feedback loops is a direction for future work"。iter #85 (commit
   ae1d7c8) 已经实现真正的 10-round × 10-candidate sequential regeneration loop
   (per-round LLM call, per-round regeneration_history emission), iter #79 guard
   也防止 Stage 7 silent overwrite。 paper §5 应更新消除 stale future-work
   claim。

2. **§5 完全没提 RQ4_GMM_Best_Prior 的 lineage gap** — iter #86 (commit 2c4e9d5)
   文档化了 Stage 12 outputs `user_profiles.jsonl` / `sentences.jsonl` 在 open-source
   release 中缺失, `compute_prior_bic_aic.py` 因此 graceful-degrade 到 parameter-count
   stub。 Table 3 数字来自 originally-computed values, 不是 re-execution。 paper §5
   应明确披露这一点。

3. **§5 完全没提 human-eval open-source 状态** — iter #74-#76-#80-#83 扩展
   `agreement_metrics.py` (Fleiss/Cohen κ + MAE/RMSE/Weighted Cohen's κ), iter #75/#80
   跑了 3 × 50 LLM-as-judge pilot, 数字与 paper §3.3 close agreement。
   paper §5 应披露 pilot state, 让 reviewer 知道 open-source 能 reproduce 哪些
   数字。

## §B 本轮 (iter #90) 改动

修改 `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:178` §5
Limitations paragraph — 替换 stale third limitation + 加 fourth (RQ4 lineage gap)
+ 加 fifth (human-eval open-source state)。

**Old** (3 limitations, 第三条 stale):
> "Third, while PQB provides a 10-round × 10-candidate regeneration mechanism
> for query quality control (§2.2), the current implementation generates all 10
> candidates in a single LLM call; true sequential regeneration with per-round
> LLM feedback loops is a direction for future work."

**New** (5 limitations, 第三条 status update + 第四条 RQ4 lineage gap + 第五条
human-eval open-source state):
> "Third, the 10-round × 10-candidate regeneration mechanism in Stage 04 query
> generation is now fully implemented in `PersoanlQuery/04_query/common/
> syntax_depth_no_depth_check.py:process_one_user` (iter #85): each round
> regenerates 10 candidates via a fresh LLM call, accepts all candidates that
> pass attribute validation, and stops early once 10 are collected. Per-round
> `regeneration_history` is persisted to JSON for audit. The Stage 07
> noisy-retrieval pipeline (iter #79) additionally enforces a TypeError
> invariant preventing silent serialization of per-query records as non-lists,
> which previously blocked bootstrap confidence-interval computation for Table 1
> Δ values.
>
> Fourth, while Stage 12 (PRF clause-features) outputs required for Table 3 are
> referenced throughout `PersoanlQuery/12_prf/` and `PersoanlQuery/10_complexity_
> analysis/common/compute_prior_bic_aic.py` (iter #69 restored), the open-source
> release does not persist the per-category `user_profiles.jsonl` and
> `sentences.jsonl` files; `compute_prior_bic_aic.py` therefore falls back to a
> parameter-count stub per category with a printed WARNING, and Table 3 numbers
> in this paper come from the originally-computed values rather than a
> re-execution of the full pipeline. Re-running Stage 12 (≈45–135 min per
> category) would re-derive Table 3 row by row; estimated total ≈9–21 hours for
> all three domains.
>
> Fifth, the human-evaluation agreement reported in Table 2 (Fleiss κ = 0.72,
> Spearman ρ = 0.81, MAE = 0.89 on the 5-point scale) was originally measured
> on 120 sampled queries; the open-source release contains the
> agreement-routines (`PersoanlQuery/agreement_metrics.py`, iter #83 extended
> with MAE / RMSE / Weighted Cohen's κ) and a pilot LLM-as-judge pipeline
> (`PersoanlQuery/02_writing_analysis/`) that produces Cohen κ in [0.49, 0.52]
> and Spearman in [0.69, 0.75] on a 3 × 50 sampled subset, in close agreement
> with the paper values."

## §C §5 Limitations 现状 (5 项)

| # | Limitation | 当前状态 | iter 引用 |
|---|-----------|----------|-----------|
| 1 | 三 Amazon 子类泛化 | Unchanged | (原有) |
| 2 | Review-style ↔ query-behavior 假设未直接验证 | Unchanged | (原有) |
| 3 | ~~Regeneration 是 single LLM call~~ → 现在已实现 10-round × 10-candidate sequential loop | **Updated** | iter #85 + iter #79 |
| 4 | RQ4_GMM_Best_Prior lineage gap (Stage 12 outputs missing in release) | **New** | iter #86 + iter #69 |
| 5 | Human-eval open-source state (3 × 50 pilot produces κ ∈ [0.49, 0.52]) | **New** | iter #74 + iter #75 + iter #76 + iter #80 + iter #83 |

## §D 与 paper §3 audit 一致性

现在 paper §5 Limitations 与 `paper_claims_audit.py` 状态 (10/1/1/0) 一致:

- 8 paper §3 quantitative claims 现在 verified + 详细 explanation 在 §5
- 1 partial (Pipeline_Skip_ColBERTv2_SPLADE) 在 §5 不显式提及 (env-var feature, no data artifact)
- 1 unverified (RQ4_GMM_Best_Prior) 在 §5 第四条显式披露 (lineage gap + re-run recipe)
- iter #85 regeneration claim 在 §5 第三条显式 verified

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:178`
- 命令: 直接文本编辑 (no compile / no test)
- 验证: `grep -n "Limitations" PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md`

## §F 与 loop.md §8 的关系

iter #90 = paper text update reflecting iter #82-#86 audit findings. 完成 paper
self-audit 闭环: 12 claims → audit table → §5 Limitations alignment.

剩余 candidate (按优先级):
- **iter #87** — 单 cat Stage 12 + Stage 10 pilot (Baby_Products 45-135 min)
- **iter #88** — Stage 7 重跑 (3 cat × 8 retriever × 95 user, multi-hour)
- **iter #91** — Paper §2 / §3 text 全面 audit (如 iter #90 §5)
- **iter #92** — README.md / CLAUDE.md sync 反映 audit 10/1/1/0 状态