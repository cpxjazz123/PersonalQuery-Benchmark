# Iteration #94 — Paper §3 Table 2 footnote: paper-vs-release discrepant values

**日期**: 2026-07-21
**scope**: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:144` (Table 2 footnote) + `:178` (§5 Limitations #5 wording)
**prior**: iter #93 audit quantified 4 discrepant claims (RQ3_Fleiss/Spearman/MAE/LLM_Full_Set preservation)

## §A 审稿意见

iter #93 audit 发现 4 个 paper claim 与 release output 之间有 substantive 差距:
- RQ3_Fleiss_Kappa_0.72: paper 0.72, release 0.6235 (rel 13.4%)
- RQ3_Spearman_0.81: paper 0.81, release 0.7164 (rel 11.6%)
- RQ3_MAE_0.89: paper 0.89, release 0.62 (rel 30.3%)
- RQ3_LLM_Full_Set preservation: paper 0.946, release 0.205 (rel 78.4%)

但当前 paper §3.3 Table 2 (lines 142-144) 直接列 paper numbers 无任何 caveat, §5
Limitations #5 (line 178) 说 "in close agreement with the paper values" — 这与
iter #93 实测数字 (rel delta 13.4-78.4%) 直接矛盾。

Reviewer 看完 Table 2 然后看 release JSON 会发现数字不一致, paper 没说清楚
为什么。 iter #94 修复这个 reviewer-visible gap。

## §B 本轮 (iter #94) 改动

### §B.1 Table 2 footnote (新增, after line 144)

```markdown
Table 2 footnote: The paper's reported numbers (Fleiss κ = 0.72, Spearman ρ = 0.81,
MAE = 0.89 on the 5-point scale, semantic preservation = 94.6%) were computed on the
originally-released 120-query human-annotated sample. The open-source release
contains smaller-scale reproductions due to the limited public query-pool size and
LLM-judge prompt-engineering differences; release values for the same metrics are
Fleiss κ = 0.6235, Spearman ρ = 0.7164, MAE = 0.62, semantic preservation = 20.5%.
Specifically, the semantic preservation gap (paper 94.6% vs release 20.5%) reflects
that Stage 7 BPE-aware noisy-query samples (21 pairs in pilot) carry stronger
semantic perturbations than the originally-sampled noisy pairs used in the paper's
94.6% measurement. The relative deltas for all four metrics are 13.4%, 11.6%,
30.3%, and 78.4% respectively. See `result/personal_query/iterations/paper_claims_audit.json`
for the full per-claim audit (iter #93 value-validation layer).
```

### §B.2 §5 Limitations #5 wording (修订, line 178)

替换:
> "...in close agreement with the paper values."

改为:
> "...somewhat lower than the paper values for Fleiss κ (0.62 vs 0.72), Spearman ρ
> (0.72 vs 0.81), MAE (0.62 vs 0.89), and substantially lower for semantic
> preservation (0.205 vs 0.946); see Table 2 footnote for details."

## §C 关键改动点

1. **Table 2 footnote explicit**: 列出每个 discrepant metric 的 paper vs release 数字 + rel delta,
   reviewer 一眼可见具体差距, 不需要自己 grep audit JSON
2. **preservation gap 解释**: footnote 解释 94.6% → 20.5% 的 78.4% rel delta 是
   Stage 7 BPE-aware noisy-query 的 stronger semantic perturbations 导致 (iter #84 已识别)
3. **§5 Limitations #5 wording fix**: "in close agreement" 改为 quantitative
   "lower than the paper values for Fleiss κ (0.62 vs 0.72)..." — 不再误导 reviewer
4. **audit JSON 引用**: footnote 指向 `paper_claims_audit.json` 让 reviewer 可以
   自主验证

## §D 与 audit summary 一致性

iter #93 audit summary 0/6/4/1/1/5/0:
- 4 discrepant claims ↔ Table 2 footnote 列出全部 4 个 (Fleiss / Spearman / MAE / preservation)
- 1 degenerate (RQ1 bootstrap CI) ↔ 暂不在 paper §3 显示, 由 §5 Limitations #4
  (Stage 12 outputs missing) 间接 cover; iter #87 后 bootstrap CI 重新计算会解决

reviewer 现在能完整追踪: paper claim → release output → 差距原因 → audit JSON 验证。

## §E 文件 & 命令

- 模块: `PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md:144` (Table 2 footnote 新增) + `:178` (§5 #5 wording fix)
- 命令: 直接文本编辑 (no compile / no test, 仅 markdown)
- 验证: `grep -n "footnote:\|lower than the paper values" PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md`

## §F 与 loop.md §8 的关系

iter #94 完成 paper §3 Table 2 reviewer-visible discrepancy disclosure:
- Table 2 footnote 量化 paper vs release 数字差距 (13.4% / 11.6% / 30.3% / 78.4%)
- §5 Limitations #5 wording 从 "in close agreement" 改为 quantitative "lower than"
- audit JSON 引用让 reviewer 可以自主 verify

后续 candidate:
- **iter #95** — paper §3 Table 1 footnote: Stage 6/9 query pool size caveat (现在 Stage 6 只有 2 queries, 不够 bootstrap, RQ1_Table1_Hit10 audit degenerate)
- **iter #87** — Stage 12 + Stage 10 pilot (45-135 min GPU) — flip 5 unverified → verified
- **iter #88** — Stage 7 重跑 (multi-hour infra) — unblock RQ1/RQ2 real Δ CIs + 修复 preservation gap