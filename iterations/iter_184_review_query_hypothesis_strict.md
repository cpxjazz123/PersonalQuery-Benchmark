# Iteration 184 — Review ≠ Query Hypothesis Strict Statistical Test

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: §1+§2.1 P0 review "Review writing style ≠ Query behavior 假设未验证" — 加 Kendall tau + bootstrap CI + permutation test

## §A 审稿意见（尖锐批评）

### 问题: iter #72 pilot 只算了 Pearson + Spearman point estimates, 没有 CI + H0 test

**严重程度**: Major (P0 backlog item, 整个 pipeline 的基础假设)

**iter #72 status**:
- 算了 Pearson + Spearman 在 3 domain × 3 Y (query_words/user_avg_depth/target_depth) = 9 cells
- 输出 point estimates 但无 95% CI, 无 H0 test
- Reviewer 无法判断: r=0.15 是真信号还是 noise? 是否显著不为 0?

**iter #184 目标**: 加 3 个 statistical rigor 改进:
1. **Kendall τ-b** (rank correlation 比 Spearman 更 robust to ties)
2. **Bootstrap 95% CI** (per-cell CI for each correlation)
3. **Permutation test p-value** (H0: ρ=0 vs H1: ρ≠0)

如果 r point estimate 在 CI 之外且 perm_p<0.05 → 显著 ≠ 0, paper §2.1 假设 support
如果 CI 包含 0 且 perm_p>0.05 → 不能拒绝 H0, paper 假设未验证 (review ≠ query)

## §B 缺陷定位

| 问题 | 文件 | 行 | 缺陷 |
|------|------|----|------|
| 缺 Kendall τ | `pilot_review_vs_query_correlation.py` | 全文 | 只有 pearson + spearman, 无 kendall |
| 缺 bootstrap CI | 同上 | _correlate_for_category | 直接输出 point estimate, 无 CI |
| 缺 H0 permutation test | 同上 | - | 无统计显著性检验 |
| 缺 stats inference interpretation | 同上 | main | print 数字不附 significant marker |

## §C 本轮代码优化

### C.1 新增 `_kendall_tau(xs, ys)` 函数 (~30 行)

手写 Kendall's tau-b (no scipy 依赖, 与 _pearson/_spearman 风格一致):
```python
def _kendall_tau(xs, ys):
    """τ_b = (n_concordant - n_discordant) / sqrt((n0-n1)(n0-n2))"""
    n = len(xs)
    if n < 2: return float("nan")
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            if dx == 0 or dy == 0:
                continue  # ties
            if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    n0 = n*(n-1)//2
    n1 = sum_ties(xs); n2 = sum_ties(ys)
    return (concordant - discordant) / sqrt((n0-n1)*(n0-n2))
```

### C.2 新增 `_bootstrap_ci(xs, ys, fn, n_boot=1000)` 函数 (~25 行)

Generic percentile bootstrap CI:
```python
def _bootstrap_ci(xs, ys, fn, n_boot=1000, ci=0.95, rng_seed=42):
    point = fn(xs, ys)
    if math.isnan(point): return (nan, nan, nan)
    rng = random.Random(rng_seed)
    n = len(xs)
    boot_stats = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        b = fn([xs[i] for i in idxs], [ys[i] for i in idxs])
        if not math.isnan(b): boot_stats.append(b)
    boot_stats.sort()
    return (point, boot_stats[lo_idx], boot_stats[hi_idx])
```

### C.3 新增 `_permutation_test(xs, ys, fn, n_perm=1000)` 函数 (~25 行)

Two-sided permutation test p-value:
```python
def _permutation_test(xs, ys, fn, n_perm=1000, rng_seed=42):
    obs = fn(xs, ys)
    if math.isnan(obs): return nan
    rng = random.Random(rng_seed)
    n = len(xs)
    ys_perm = list(ys)
    extreme_count = 0
    for _ in range(n_perm):
        rng.shuffle(ys_perm)
        perm = fn(xs, ys_perm)
        if abs(perm) >= abs(obs): extreme_count += 1
    return extreme_count / n_perm
```

### C.4 扩展 `_correlate_for_category` 输出嵌套 dict

每个 (X, Y) pair 输出:
```python
{
    "point": float,        # correlation point estimate
    "ci95_low": float,     # 2.5 percentile of bootstrap
    "ci95_high": float,    # 97.5 percentile of bootstrap
    "perm_p": float,       # fraction of perms at least as extreme
}
```

3 metrics × 3 (X, Y) pairs = 9 nested dicts per category.

### C.5 main() 输出

- 显著性 marker (`*` if perm_p<0.05)
- Summary table 加 Kendall τ-b 列
- 跨 9 cells 的 side-by-side 对比

## §D 验证

### D.1 py_compile
```bash
$ python3 -m py_compile PersoanlQuery/02_writing_analysis/pilot_review_vs_query_correlation.py
OK
```

### D.2 Smoke test 6 cases (新 `_smoke_iter184_correlation_methods.py`)

| Case | data | Expected | Actual |
|------|------|----------|--------|
| 1 | perfect linear (n=5) | pearson=spearman=kendall=1.0 | 1.0/1.0/1.0 ✓ |
| 2 | anti-correlation | all = -1.0 | -1.0/-1.0/-1.0 ✓ |
| 3 | independent noise (n=50) | perm_p > 0.05 | 0.273 ✓ |
| 4 | bootstrap CI on perfect corr | CI95 = [1.0, 1.0] | [1.0, 1.0] ✓ |
| 5 | bootstrap CI on moderate corr | CI brackets point | [0.997, 1.000] ⊃ 0.999 ✓ |
| 6 | perm_p on perfect corr (n=5) | ≤ 0.05 | 0.016 ✓ |

**All 6 cases pass** — Kendall τ + bootstrap CI + permutation test 数学正确。

### D.3 End-to-end pilot run

不能跑 (Stage 1 + Stage 6 lineage gap, see iter #86/96)。Plumbing ready for future Stage 1+6 outputs。

## §E 后续 iter

- **iter #185**: 真实 Stage 1 + Stage 6 重跑 (~30 min - 3 h, sbatch infra) → pilot 端到端跑通 → 看 CI 是否包含 0 / perm_p 是否 < 0.05 → 回答 paper §2.1 假设 review ≠ query 是否成立
- **iter #186**: paper §2.1 文本基于 iter #185 结果更新 (如果 r>0.3 → 加 caveat; 如果 r<0.1 → 假设 confirmed; 如果混合 → per-domain disclosure)
- **iter #187**: paper_claims_audit Sec2_UserFilter_20_reviews_15_words 加 iter #73 ablation evidence

## §F Git Commit

- iter #184: expand pilot_review_vs_query_correlation.py with Kendall tau-b + bootstrap 95% CI + permutation test p-value (3 new functions + extended output schema + smoke test 6 cases all pass)