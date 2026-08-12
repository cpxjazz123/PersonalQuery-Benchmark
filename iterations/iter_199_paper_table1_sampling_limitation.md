# Iteration 199 — Paper Table 1 上方面板采样数据集未发布 Limitation

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 审查视角）
**scope**: Paper Table 1 上方面板 Δ Range 计算所依赖的采样子集未在 release 中发布

---

## §A 发现

### Paper Table 1 上方面板 (Δ Range across 8 GMM clusters)

**Paper 描述**：
- 8 GMM expression-style clusters (C1-C8)
- 每个 cluster 约 90 queries
- Δ Range = max(C1..C8 Hit@10) - min(C1..C8 Hit@10) per domain

**origin/main:dataset/ 实际情况**：
- Baby_Products: 6535 users × 1 query = 6535 total (≈817/cluster)
- Grocery: 6141 users × 1 query = 6141 total (≈768/cluster)
- Pet: 13933 users × 1 query = 13933 total (≈1742/cluster)

**差距**：Paper 声称 ~90 queries/cluster ≈ 720 queries/domain，但 dataset/ 有 6000-14000/cluster。

### 根因

Paper Table 1 上方面板在**采样子集**（约 90 queries/cluster）上计算，但：
1. 该采样子集**未作为独立文件发布**
2. 仅发布了**全量 dataset/**（26609 users）
3. retrieval 中间结果（Hit@10 per query）**未发布**

### Paper §5 Limitations 现有内容

> "bootstrap CI all-NaN due to Stage 6 query-pool size"

已提到 bootstrap CI 问题，但**未明确说明**：采样子集本身也未发布。

---

## §B 建议修改

在 Paper §5 Limitations 新增第 6 点：

> **Sixth, the sampled query subset for Table 1 upper panel (Δ Range across 8 GMM clusters) is not persisted in the open-source release.** The paper Table 1 upper panel reports Δ Range values computed on a sampled subset of ≈90 queries per cluster (≈720 queries per domain). The open-source release (`dataset/`) contains the full user base (26,609 users, 1 query per user) rather than this sampled subset, and the per-query Hit@10 retrieval scores are not published. As a result, the Δ Range values in Table 1 upper panel (e.g., SPLADE Δ=9.6, E5 Δ=6.58) cannot be independently reproduced from the release artifacts. Re-running Stage 06/07 retrieval on the full dataset and then sampling ~90 queries per cluster would approximate the reported values but may not match exactly due to the different sampling seed and pool size.

---

## §C 落地

本次 audit 仅记录发现，未修改 Paper。

**后续**：如需修复，可考虑：
1. 发布采样子集 + retrieval 结果到 origin/main dataset/
2. 或显式在 Paper §5 Limitations 说明
