# Iteration 198 — dataset/ vs Paper Table 1 上方面板一致性审计

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 审查视角）
**scope**: 验证 origin/main:dataset/ 是否可用于复现 Paper Table 1 Δ Range

---

## §A 审计发现

### dataset/ 数据规模 vs Paper Table 1 声称

| 域 | dataset/ 用户数 | 实际 cluster 大小（均值） | Paper 声称（~90/cluster） |
|---|---------------|------------------------|------------------------|
| Baby_Products | 6535 | ~817/cluster | ~90/cluster |
| Grocery_and_Gourmet_Food | 6141 | ~768/cluster | ~90/cluster |
| Pet_Supplies | 13933 | ~1742/cluster | ~90/cluster |

**结论**：dataset/ 是全量用户库（26609 users total），Paper Table 1 在**采样子集**（~90/cluster × 8 clusters × 3 domains ≈ 2160 queries）上计算。

### dataset/ 包含字段

```json
{
  "category": "Baby_Products",
  "uuid": "AE23NZUELB4BYWLHKWJXAS73PTSQ",
  "asin": "B08R5BZNSP",
  "queries": [
    {"cluster": 5, "correct_query": "I would like to order Small KK BETO Ties..."}
  ]
}
```

**缺失字段**：无 Hit@10 分数。Hit@10 需由 Stage 06/07 retrieval 生成（retriever 在 dataset/ 上运行）。

### bootstrap_delta_ci.json 状态

```json
{
  "category": "Baby_Products",
  "retriever": "ance",
  "per_metric": {
    "H@10": {
      "n": 0,
      "mean": NaN,
      "ci_low": NaN,
      "ci_high": NaN,
      "deg_warning": "no_per_query_records"
    }
  }
}
```

**结论**：Stage 08 bootstrap CI 全为 NaN，因 retrieval 结果只有 2 query records/cell（iter #79 TypeError guard），不足以做 bootstrap。

---

## §B 核心问题

**Paper Table 1 上方面板（8 GMM clusters Δ Range）无法在本地验证**：

1. dataset/ 有 8-cluster 标签（cluster 0-7），但无 Hit@10 分数
2. Stage 06/07 retrieval 结果不可用（每个 retriever/cell 只有 2 条 records）
3. Paper Δ Range 值（SPLADE=9.6 等）来自采样子集的 retrieval 结果，该采样子集未发布

### 两种可能

- **可能 A**：Paper 的采样子集 + retrieval 结果已发布在 origin/main 某处（未找到）
- **可能 B**：采样子集的 retrieval 结果未发布，仅发布了全量 dataset/ + pipeline 代码

---

## §C 落地

本次 audit 仅验证了 dataset/ 结构，未修改任何代码。

**验证清单**：
- [x] dataset/ 有 3 个域的完整 8-cluster GMM 数据
- [x] 每个 user 有 1 query in their assigned cluster  
- [x] cluster 分布不均匀（cluster 0 最大，cluster 7 最小）
- [x] Hit@10 分数缺失，需 retrieval 阶段生成
- [x] bootstrap CI 全为 NaN（no_per_query_records）

---

## §D 下一步

1. **如选择可能 A**：搜索 origin/main 是否存在采样子集 + retrieval 结果
2. **如选择可能 B**：承认 Paper Table 1 上方面板数值依赖未发布的中间结果，在 Resource Paper Limitations 中说明
3. **修复 bootstrap CI**：需重跑 Stage 06/07 retrieval（在全量 dataset/ 上）产生 per-query Hit@10，然后可计算 bootstrap CI
