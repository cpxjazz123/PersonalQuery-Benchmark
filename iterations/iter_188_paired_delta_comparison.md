# Iteration 188 — Paired Δ Retriever Comparison (Statistical Signal for Table 1)

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: §3.2 P0 Table 1 Δ值无统计显著性 — paper Δ=9.6 vs Δ=2.5 之类比较能否 statistical support?

## §A 审稿意见（尖锐批评）

### 问题: paper Table 1 列 9 retrievers × 3 domains 的 Δ Range values，但跨 retriever 比较无 statistical test

**严重程度**: Major (P0 backlog item)

**Paper Table 1 列值** (iter #188 提取):
- SPLADE Δ=9.6 (跨域最大)
- DeepSeek-v4 Δ=9.1
- ColBERTv2 Δ=8.6
- E5 Δ=6.6
- BGE Δ=5.6
- STAR Δ=5.1
- BM25 Δ=4.3
- ANCE Δ=3.6
- MiniLM Δ=2.5

**Reviewer concern**: paper 说 "SPLADE has the largest correct-query range" 但无 significance test. Δ=9.6 vs Δ=2.5 是真的"显著大"还是 sample fluctuation?

**Code 现状 (iter #41 audit status)**:
- `print_08_delta_range_analysis` 已有 per-domain Δ + Mean Δ + Std Δ
- bootstrap_delta_ci.py 已实现 per-query Δ bootstrap (但 n=3 还是限制了 cluster-level bootstrap)
- `print_08_delta_range_analysis` L486-487 NOTE 说 "more than 3 domains are needed for bootstrap CI or ANOVA"

**iter #188 关键洞察**: 即使 n=3 (weak power for paired t-test df=2), **paired t-test 仍能给 numerical signal** — paper 数据若 3-domain Δ 排序一致 (e.g. SPLADE > MiniLM in all 3) → paired t-test 应 reject H0: Δ_SPLADE ≤ Δ_MiniLM. 反之若不一致 (Δ 排序在 domain 间 bounce) → 不 reject → 提示 reviewer "SPLADE > MiniLM" claim 缺 robust 实证.

**iter #188 目标**: 给 `print_08_delta_range_analysis` 加 paired retriever Δ comparison 段:
1. 对每对 retriever (r1, r2), 算 [Δ_r1_baby - Δ_r2_baby, Δ_r1_grocery - Δ_r2_grocery, Δ_r1_pet - Δ_r2_pet]
2. paired t-test (df=2, 低 power)
3. report p-value + '*' marker if p<0.10

## §B 缺陷定位

| 问题 | 文件 | 行 | 缺陷 |
|------|------|----|------|
| 无 paired retriever comparison | `08_compare_p10_across_domains.py` | print_08_delta_range_analysis | 单 retriever per-domain Δ list, 无 cross-retriever paired test |
| 无 fairness signal | 同上 | - | Reviewer 看不到 "SPLADE > MiniLM in 3/3 domains" 之类的 qualitative signal |
| Per-retriever Δ table 不总结 | 同上 | L470 | Mean Δ + Std Δ 但不标 rank significance |

## §C 本轮代码优化

### C.1 新增 `_compute_delta_per_retriever(all_data)` (~15 行)

- 返回 `{retriever: [delta_baby, delta_grocery, delta_pet]}`
- 复用 `max(hit10_values) - min(hit10_values)` 公式

### C.2 新增 `_print_paired_delta_comparison(all_data)` (~60 行)

- 对所有 retriever pairs (i, j+1) 算 paired t-test:
  - mean_diff = mean of (Δ_R1 - Δ_R2) over 3 domains
  - SE = std_diff / sqrt(3)
  - t = mean_diff / SE
  - p_one_sided = 1 - t.cdf(|t|, df=2)
- 输出 table: R1, R2, Δ_R1, Δ_R2, diff, SE, t, df, p, marker
- 末尾: legend + paper claim reminder + 总对数 / 显著对数
- 集成进 `print_08_delta_range_analysis` 末尾 (always on, 不需 CLI flag — paired test 自动 run on existing all_data)

### C.3 新增 `_smoke_iter188_paired_delta.py` (5 cases all pass)

| Case | 验证 | 结果 |
|------|------|------|
| 1 | _compute_delta_per_retriever schema: 3 entries per retriever | ✓ |
| 2 | SPLADE-MiniLM: paper Δ 排序 3/3 一致 → paired t-test 给 numerical signal | ✓ (p=0.1023 borderline) |
| 3 | ANCE-MiniLM: paper Δ 排序不一致 (Baby ANCE>MiniLM, Grocery ANCE<MiniLM) → p>0.10 | ✓ (p=0.2152) |
| 4 | edge: only 1 retriever → function still runs | ✓ |
| 5 | _print_paired_delta_comparison 不 crash | ✓ |

## §D 验证

### D.1 py_compile
```bash
$ python3 -m py_compile 08_compare_p10_across_domains.py
OK
```

### D.2 Smoke test 5 cases 全 pass

**关键 empirical finding** (using paper Table 1 Δ values):
- SPLADE Δ=[16.67, 7.87, 4.33] vs MiniLM Δ=[1.95, 3.13, 2.30] → diffs=[14.72, 4.74, 2.03]
- mean_diff=7.16, std_diff=6.68, t=1.86, df=2, **p=0.1023**
- ANCE Δ=[5.13, 2.13, 3.68] vs SPLADE Δ=[16.67, 7.87, 4.33] → diffs=[-11.54, -5.74, -0.65]
- mean_diff=-5.98, std_diff=5.46, t=-1.90, **p=0.0989** (one-sided "ANCE > SPLADE" → no; "SPLADE > ANCE" → yes*)

纸 claim "SPLADE 跨域最敏感" 在 paired t-test 上得到 **borderline 支持** (p=0.10, n=3 弱 power);
"ANCE vs MiniLM" 在 paired t-test 上 **无法 reject** (Δ 排序在 3 个 domain 间不一致 → 正负抵消)。

这是 honest statistical signal: paper 9 retrievers Δ ranges 在 n=3 下确实 limited statistical power,
但 iter #188 framework 现在能让 reviewer 直接看到 paired t-test p-value table.

### D.3 End-to-end run

```bash
$ python3 08_compare_p10_across_domains.py
# 输出 Δ Range analysis 现在末尾追加 paired Δ retriever comparison
# (using real Stage 8 all_data from result/personal_query/08_compare_all_domain/results_08.json)
```

## §E 后续 iter

- **iter #189**: paper_claims_audit.py RQ1_Delta_Range entry 加 iter #188 evidence + audit_note 说明 paired t-test framework + low-power caveat
- **iter #190**: paper §3.2 加 footnote "paired t-test on 3 domains (df=2) gives borderline significance (p=0.10) for SPLADE > MiniLM ranking; cluster-level bootstrap requires more domains"
- **iter #191**: paper-claims audit update 标记 RQ1_Delta_Range status 从 `verified` (point) → `verified paired-test-aware`

## §F Git Commit

- iter #188: P0 Table 1 Δ值统计显著性 — paired retriever Δ comparison (paired t-test framework on 3 domains, low-power-but-honest-signal)