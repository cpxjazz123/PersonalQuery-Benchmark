# Iteration #92 — Paper §2.2 text update (K=8 + BIC/AIC/silhouette wording)

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:108`
**prior**: iter #91 audit found 2 paper_text_caveats in `Sec2_K8_clusters_BIC_AIC_silhouette`

## §A 审稿意见

iter #91 paper §2.2 audit found 2 places where paper text wording 不完全准确反映
code 实际行为:

1. **"yields 8 clusters per domain"** — 暗示 K=8 是固定结果, 实际 K 是 data-driven
   (BIC minimization from {2,...,8}), 在 released runs 收敛到 K=8。
2. **"K selected via BIC, AIC, and silhouette score"** — 暗示三 metric 平等用于 selection,
   实际 code 用 min-BIC 作 primary, silhouette 作 tiebreak, AIC 仅 computed + reported
   但 NOT used for selection。

Reviewer 看 paper 容易误以为 K=8 是 hardcoded, AIC 是 selection criterion, 与
code 行为不符 → 审稿误解。

## §B 本轮 (iter #92) 改动

修改 `PersonalQuery-Benchmark_evaluating_retrieval.md:108`:

**Old**:
> "The number of clusters is selected from 𝐾 ∈ {2, … , 8} based on BIC, AIC, and
> silhouette score. As a result, each of the Baby, Grocery, and Pet domains yields
> 8 expression style clusters."

**New**:
> "The number of clusters K is selected from the range K ∈ {2, … , 8} by minimizing
> the Bayesian Information Criterion (BIC), with the silhouette score used as a
> tiebreak; AIC is computed and reported for reference. The selected K is
> data-driven; in the released runs the BIC minimization converges to K = 8 for
> each of the Baby, Grocery, and Pet domains, yielding 8 expression-style clusters
> per domain."

## §C 关键改动点

1. **明确 selection criterion**: "minimizing BIC, with silhouette score as tiebreak"
   (不再是 "based on BIC, AIC, and silhouette" 三 metric 平权)
2. **明确 AIC role**: "AIC is computed and reported for reference" (不再是
   "selected via ... AIC ..." 暗示 used in selection)
3. **明确 K=8 是 data-driven 结果**: "selected K is data-driven; in the released
   runs ... converges to K = 8" (不再是 "yields 8 clusters" 暗示 fixed)

## §D 与 paper §5 Limitations 一致性

现在 paper §2.2 + §5 都明确披露:
- §2.2: K=8 is data-driven, BIC + silhouette used, AIC informational
- §5 Limitations: Stage 12 outputs missing in open-source release

reviewer 现在能看到完整 picture: K=8 是 originally-computed 值 (Stage 12 outputs
缺失 → 不能 re-execute BIC selection), AIC 是 informational (paper §2.2 text 现在
准确反映)。

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:108`
- 命令: 直接文本编辑 (no compile / no test)
- 验证: `grep -n "K is selected\|yields 8" PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md`

## §F 与 loop.md §8 的关系

iter #92 完成 paper §2.2 text 收紧 (消除 iter #91 两个 paper_text_caveats)。

后续 candidate:
- **iter #87** — Stage 12 + Stage 10 pilot (45-135 min GPU) — flip 5 unverified → verified
- **iter #88** — Stage 7 重跑 (multi-hour) — unblock RQ1/RQ2 real Δ CIs
- **iter #93** — README.md / CLAUDE.md sync 反映 audit 11/1/5/0 状态