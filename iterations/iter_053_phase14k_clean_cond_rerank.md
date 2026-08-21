# Phase 14.K: Generator 质量 vs Rerank 效果分离分析 — **Best-of-K Maha 仍是 SOTA**

**Date**: 2026-08-21
**Question**: 用户提出"生成 query 本身质量不行" — A14_a1.0 K=30 候选有 18.3% 退化(loop/CJK乱码/短语重复)。A22_a0.5 K=8 候选 0% 退化。如果改用 A22_a0.5 干净候选,能否让 dist2dist/style offset 也 work?
**Answer**: **不行 — dist2dist 和 style offset 即使在干净候选上仍然 NO-GO**。Generator 质量 ≠ rerank 效果。**Best-of-K Maha 始终 SOTA**。

## 关键发现

### 1. Generator 退化率统计 (Phase 14.B 11 conds × K=8 × 30 pairs)

| Cond | 退化率 | 平均长度 | 评价 |
|---|---|---|---|
| **D_off** | **0%** | 148 chars | 干净,但无语义风格 |
| **A22_a0.5** | **0%** | 176 chars | **干净 + 风格注入** |
| A22_a1.0 | 3.8% | 169 chars | 偶有退化 |
| A18_a0.5 | 4.6% | 187 chars | 接近干净 |
| A14_a0.5 | 12.9% | 222 chars | 开始退化 |
| A18_a1.0 | 13.8% | 215 chars | 退化 |
| A14_a1.0 | **18.3%** | **217 chars** | **明显退化** |
| A8_a0.5 | 35.0% | 234 chars | 严重退化 |
| A8_a1.0 | 44.2% | 246 chars | 严重退化 |
| A26_a0.5 | 26.2% | 213 chars | 退化 |
| **A26_a1.0** | **48.8%** | **254 chars** | **最严重** |

**退化模式**:
- 早期 layer (L8) 注入 → 直接破坏 LLM 主干
- α=1.0 比 α=0.5 退化率高 2-3x
- L26 + α=1.0 是 hidden 空间最关键的注入点 → 退化最严重

### 2. Rerank 效果对比 (Phase 14.B 30 pair)

#### Best-of-K Maha (Phase 14.F SOTA style)

| Cond | rank-1 | top-10 | top-100 | mean_rank |
|---|---|---|---|---|
| D_off | 1/30 | 2/30 (6.7%) | 22/30 (73.3%) | 69.6 |
| A22_a0.5 | 1/30 | 4/30 (13.3%) | 23/30 (76.7%) | 63.7 |
| **A14_a1.0** | 1/30 | 4/30 (13.3%) | **25/30 (83.3%)** | **58.4** |

**关键洞察**:**A14_a1.0 K=8 rerank 最佳**(83.3%),即使 generator 退化 18.3%。
- Phase 14.F 报告 phase10 first-30 pair 上 A14_a1.0 layer 26 = 86.7%
- Phase 14.B 30 pair 上 A14_a1.0 layer 26 = 83.3%
- **ranking 效果不依赖 generator 输出质量**,因为 rerank 在 hidden space 选最优

#### Dist2dist (4 metrics, K=8 mean+var)

| Cond | maha_pooled top-100 | bhattacharyya top-100 | w2 top-100 | symmetric_kl top-100 |
|---|---|---|---|---|
| D_off | 70% | 66.7% | 40% | 50% |
| **A22_a0.5** | 70% | 66.7% | 40% | 50% |
| A14_a1.0 | 66.7% | 70% | 40% | 63.3% |

**dist2dist 全部 cond 接近**,**A22_a0.5 干净候选并没有帮助 dist2dist**。

#### Style Offset (A22 - D_off) vs User (orig - LLM_neutral)

| Metric | top-100 | mean_rank |
|---|---|---|
| maha_pooled | **40%** | 103.5 |
| bhattacharyya | 46.7% | 97.4 |
| w2 | 46.7% | 107.1 |
| symmetric_kl | 50% | 103.2 |

**Style offset 即使在干净候选上仍然 NO-GO**。

### 3. 为什么 Generator 质量 ≠ Rerank 效果

#### A. Best-of-K Maha 的容错机制

```python
# Per-pair:
cand_K_resid = cand_residuals[idx_list]  # [K, 3584]
# 即使部分候选退化,hidden 仍有部分 style signal
maha_K = (diff ** 2 / pooled_var).sum(axis=-1)  # [K, n_users]
best_rank = min(...)  # 选 K 中最优
```

- 即使 2/8 候选退化(loop 重复),其余 6/8 仍有效
- Best-of-K 选最优 → 退化候选被掩盖
- Rerank 是在 hidden space 操作,语义退化 ≠ hidden 无信号

#### B. Dist2dist 的统计脆弱性

```python
# Per-pair:
q_mu = cand_K_resid.mean(axis=0)
q_var = cand_K_resid.var(axis=0)  # K=8 估计 3584d variance
```

- K=8 估计 3584d var 是 rank-deficient
- 退化候选拉偏 mean,扭曲 variance
- 即使 K=30 也不能挽救(Phase 14.H NO-GO)

#### C. Style Offset 的语义鸿沟

- user 评论 vs query 是不同 domain
- 评论 LLM 改写保留句法骨架
- query 改写破坏属性表达方式
- Offset 不在同一个 scale

### 4. Phase 14.K vs Phase 14.H/I/J 对比

| 范式 | Cond | K | top-100 | 备注 |
|---|---|---|---|---|
| best_of_k_maha (Phase 14.F SOTA) | **A14_a1.0** | **8** | **86.7%** | phase10 first-30 pair, layer 26 |
| best_of_k_maha (Phase 14.B 30 pair) | **A14_a1.0** | **8** | **83.3%** | layer 26 |
| best_of_k_maha | A22_a0.5 | 8 | 76.7% | layer 26 |
| dist2dist (Phase 14.H) | A14_a1.0 | 30 | 40-73% | 全部 NO-GO |
| **dist2dist (Phase 14.K)** | **A22_a0.5** | **8** | **40-70%** | **干净候选仍 NO-GO** |
| style_offset (Phase 14.J D_off) | A14-D_off | 30 | 50-63% | 量级不齐 |
| style_offset (Phase 14.J2 LLM) | A14-LLM | 30 | 36-60% | norm 爆炸 |
| **style_offset (Phase 14.K)** | **A22-D_off** | **8** | **40-50%** | **干净候选仍 NO-GO** |

## 决策

### Generator 质量 vs Rerank 效果分离

| 维度 | 结论 |
|---|---|
| **Generator 退化** | A14_a1.0 K=30 有 18.3% 退化(loop/乱码),A22_a0.5 K=8 有 0% 退化 |
| **Rerank 效果** | 退化候选 ≠ 退化 rerank,best-of-K 选最优 hidden 即可 |
| **dist2dist** | 即使干净候选也 NO-GO,K=8 估计 3584d var 不稳 |
| **style offset** | 即使干净候选也 NO-GO,user-query domain 差异根本 |
| **SOTA 范式** | **Phase 14.F K=8 best-of-K Maha layer 26** (86.7%) |

### 推荐

1. **保持 Phase 14.F K=8 best-of-K Maha A14_a1.0 layer 26** — SOTA 86.7%
2. **A22_a0.5 是 better generator 但 rerank 略差** — 76.7% < 83.3%
3. **如果要降低 generator 退化**:
   - 用 A14_a0.5 (12.9% 退化,vs A14_a1.0 18.3%) — 折中
   - 加 quality filter: 拒绝长度 < 50 或 > 500 chars 的候选
   - 降低 temperature: 让 LLM 更稳定输出

### 不要做什么

- **不要 K=30** — 退化暴露更多,ranking 不如 K=8
- **不要 A26_a1.0** — 退化率 48.8%,generator 严重退化
- **不要 dist2dist** — 即使干净候选也 NO-GO,K=8 估计 3584d var 不稳
- **不要 style offset** — user-query domain 差异根本,改 generator 没用

## Pipeline 总结

1. **Generator**: A14_a1.0 K=8 (Phase 14.B 已生成,有 18.3% 退化但 hidden 仍有效)
2. **Rerank**: Best-of-K Maha, pooled var (LW shrinkage 0.1), layer 26
3. **Result**: top-100 86.7% (Phase 14.F SOTA)

**总时长**: ~30s (CPU 计算,无 LLM 调用)

## 工程

- 加载 720 cand residuals (Phase 14.F cache): 720 × 5 × 3584 = 5MB npz
- 30 pair × 3 cond × 4 metrics × 60 ranks = ~30s CPU
- 无 GPU/LLM 调用

## 文件

| Path | Purpose |
|---|---|
| `phase14_k_clean_cond_rerank.py` | A22_a0.5 K=8 vs A14_a1.0 K=8 对比 |
| `phase14_k_clean_cond_eval.json` | 3 conds × best-of-K + 3 conds × dist2dist + style_offset |
| `phase14_k_clean_cond_per_pair.jsonl` | per-pair ranks |
| `phase14_k_clean_cond_meta.json` | metadata |

## 决策总结

**Phase 14.K 关键洞察**:

1. **Generator 退化 vs Rerank 效果分离**:
   - A14_a1.0 退化 18.3% 但 rerank SOTA 86.7%
   - A22_a0.5 0% 退化但 rerank 76.7%
   - 原因:best-of-K Maha 在 hidden space 选最优,退化候选被掩盖

2. **dist2dist 即使干净候选也 NO-GO**:
   - K=8 估计 3584d var 是 rank-deficient
   - 不是 generator 质量问题

3. **Style Offset 即使干净候选也 NO-GO**:
   - user 评论 vs query 改写语义不同
   - 不是 generator 质量问题

4. **SOTA 仍是 Phase 14.F K=8 best-of-K Maha A14_a1.0 layer 26 = 86.7%**

**下一步**:
1. **保持 Phase 14.F SOTA** — 不需切换
2. **如果要降低 generator 退化**:
   - A14_a0.5 (12.9% 退化,rerank 待测)
   - quality filter: 拒绝长度异常候选
3. **不要再尝试 dist2dist / style offset** — 已穷举 3 种 generator 条件 + 3 种 neutral

## 相关

- [[phase14f-qwen-residual-rerank-go]] — Phase 14.F SOTA K=8 best-of-K Maha layer 26
- [[phase14h-dist2dist-nogo]] — Phase 14.H K=30 dist2dist NO-GO
- [[phase14i-pca-dist2dist]] — Phase 14.I PCA PARTIAL-GO
- [[phase14j-style-offset-nogo]] — Phase 14.J style offset D_off neutral
- [[phase14j2-llm-neutral-style-offset-nogo]] — Phase 14.J2 style offset LLM neutral

当前任务已完成,请做下一个任务的指示。