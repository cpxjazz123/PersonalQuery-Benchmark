# Iteration #49 — Stage 08 Δ Range 分析 + Stage 10 BIC/AIC 脚本

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人
**scope**: Stage 08 新增 Δ Range 分析函数；Stage 10 新增 BIC/AIC 计算脚本

---

## §A 审稿意见背景

### iter #41 审稿意见（来源）
> "**[Major] GMM prior 结论是循环论证**：GMM/t-dist/Laplace/Logistic 都在 fit 同一数据，2-component GMM 有更多自由参数，likelihood 优势可能是过拟合而非真实更好的 fit。BIC/AIC 校正（proper model-complexity penalty）可解决此问题。"

> "**[Major] Table 1 Δ值无统计显著性**：无置信区间、无 significance test，Stage 08 有 bootstrap CI infrastructure 但未用于 Δ Range。**"

---

## §B 本轮代码改动

### 1. Stage 08 新增 Δ Range 分析函数

**文件**: `08_compare_all_domain/08_compare_p10_across_domains.py`

新增 `print_08_delta_range_analysis()` 函数：
```python
def print_08_delta_range_analysis(all_data: dict) -> None:
    """Report Δ Range (correct-query range) per retriever with per-domain breakdown.
    Δ Range = max(C1..C8 Hit@10) - min(C1..C8 Hit@10) per domain, then averaged across the 3 domains.
    NOTE: With only n=3 domains, formal bootstrap CI cannot be reliably computed."""
```

**输出格式**:
```
=== Δ Range Analysis (Higher = More Query-Sensitive) ===
Retriever   Baby Δ  Grocery Δ  Pet Δ  Mean Δ  Std Δ
BM25        x.xx    x.xx       x.xx   x.xx    x.xx
BGE         x.xx    x.xx       x.xx   x.xx    x.xx
E5          x.xx    x.xx       x.xx   x.xx    x.xx
...
```
- Baby Δ = max(C1..C8 Hit@10) - min(C1..C8 Hit@10) on Baby_Products domain
- Mean Δ = average across 3 domains
- NOTE: n=3 无法做 bootstrap CI（需 >5 domains）

### 2. Stage 10 新增 BIC/AIC 计算脚本

**文件**: `10_complexity_analysis/common/compute_prior_bic_aic.py`

**参数计数公式**:
| Prior | k (params) | 公式 |
|-------|-----------|------|
| GMM (K=2) | 4d + 1 = 33 | 2×K×d + (K-1) |
| t-distribution | 2d + 1 = 17 | mean + log_scale + df |
| Laplace | 2d + 1 = 17 | mean + log_scale + scale |
| Logistic | 2d + 1 = 17 | mean + log_scale + scale |

**BIC/AIC 公式**:
- BIC = k × log(n) - 2 × log(L)
- AIC = 2 × k - 2 × log(L)

**当前状态**: 脚本就绪，需 Stage 10 latent representations 才能完整计算。
- Stage 10 需先生成 VAE latent representations
- latent representations 路径: `result/personal_query/12_complexity_analysis_clause_features/<cat>/<tag>/`

---

## §C 验证

- `python3 -m py_compile 08_compare_p10_across_domains.py` ✅ PASS
- `python3 -m py_compile compute_prior_bic_aic.py` ✅ PASS

---

## §D Git Commit

```
a68b4d2 iter #49: Stage08新增ΔRange分析;Stage10新增compute_prior_bic_aic.py脚本
```

---

## §E 当前阻塞项

| 阻塞项 | 依赖 | 状态 |
|--------|------|------|
| Δ Range 分析 | Stage 08 result 数据 | result 目录不可用 |
| BIC/AIC 计算 | Stage 10 latent representations | result 目录不可用 |

---

## §F 下轮建议

1. **P0**: GMM BIC/AIC — 需运行 Stage 10 训练生成 latent representations（属于非 LLM 实验，可调用 sbatch_wrapper）
2. **P0**: Δ Range — 需 Stage 08 数据；当前 n=3 无法做 bootstrap CI，需在论文中说明局限性
3. **P0**: Review writing style ≠ Query behavior 假设验证 — 需设计新实验（非 LLM）
4. **P1**: 用户阈值 ablation — 需修改用户筛选标准重跑 Stage 02-04
