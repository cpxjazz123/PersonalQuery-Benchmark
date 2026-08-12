# Iteration 194 — Resource Paper Reproducibility Audit: Stage 06 Query Pipeline断链

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 审查视角）
**scope**: 数据集可复现性 — Stage 06 query 输出断链

---

## §A 审稿意见（Resource Paper 视角）

### 核心问题：Stage 06 query 输出为空，Resource Paper 无法复现

**作为 CIKM 2026 Resource Paper，核心承诺是：发布 PersonalQuery Benchmark 数据集 + 完整代码，community 可复现所有实验。**

但审计发现：

| Stage | 目录 | 状态 |
|-------|------|------|
| Stage 04 query | `result/personal_query/04_query/<cat>/` | ✓ 有输出文件 |
| Stage 06 query (期望输入 GMM) | `result/personal_query/06_query/<cat>/` | **空目录** |
| Stage 12 complexity_analysis (GMM 输出) | `result/personal_query/12_complexity_analysis_clause_features/<cat>/` | **目录不存在** |

### 根因

`cluster_strict5550_query_gmm_and_attach_retrieval.py` 期望读取：
```
result/personal_query/06_query/<cat>/query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json
```

此文件不存在。Stage 06 query pipeline 从未成功运行，或 Stage 04 输出未传递到 Stage 06。

### 对 Resource Paper 的影响

1. **Δ Range 无法计算**：Table 1 依赖 Stage 10 GMM 输出，但 Stage 10 从未运行
2. **8-cluster vs 2-bucket 问题**：Table 1 上方面板标题是 "8 GMM expression-style clusters"，但 release 代码只有 2 bucket
3. **Bootstrap CI 全为 NaN**：每个 retriever/category 只有 2 条 query records，无法做 bootstrap
4. **Community 无法复现**：即使运行完整 pipeline，Stage 06→10 的数据传递存在断链

---

## §B 参考文献（Consensus MCP search，规则 7）

### 论文发现 → PQB 现状 → 改进方案

**[Combining Mixture Components for Clustering](https://consensus.app/papers/details/337fbfe49e6d5f7b9fad7ad078927028/)** [2] — Baudry et al., 2010, 366 citations

**论文发现 [2]**: BIC 选择的是 Gaussian mixture components 数量 K（用于近似密度），而非真实聚类数。非高斯 cluster 会被 BIC 分解为多个 components，导致 BIC 高估 K。解决方案：先用 BIC 选 K，再用层次熵准则合并 components。

**PQB 现状**:
- Stage 10 代码使用 GMM_K_RANGE=[2..8] + min_BIC 选 K，但**没有任何 hierarchical merging 步骤**
- 论文说 K=8 但无依据（未引用 BIC 选择过程）
- Stage 06 query 输出为空导致 Stage 10 从未真正运行

**改进方案**:
- Stage 10 增加 hierarchical merging 步骤（Baudry et al. 2010 方法）
- Paper 补充 K selection 依据（BIC curve 数值 + 选择理由）
- 修复 Stage 06→10 pipeline 使 Stage 10 可运行

**[Unsupervised deep clustering via adaptive GMM modeling](https://consensus.app/papers/details/6b86e5cfafe65f28913de7f938489d52/)** [1] — Wang et al., 2021, 89 citations

**论文发现 [1]**: 联合优化 GMM 参数和数据表征，使 intra-cluster compactness 和 inter-cluster separability 同时提升。

**PQB 现状**: Stage 10 GMM 用固定 PCA embedding，未联合优化特征空间和聚类。

**改进方案**: 考虑用 deep clustering 替代独立 PCA + GMM 流程。

---

## §C 落地验证

### C.1 Stage 06 query pipeline 断链诊断

```bash
# Stage 04 有数据 ✓
ls /fs04/ar57/wenyu/result/personal_query/04_query/Baby_Products/
# → query_by_syntax_depth_no_depth_check_10.json

# Stage 06 query 为空 ✗
ls /fs04/ar57/wenyu/result/personal_query/06_query/Baby_Products/
# → (empty)

# Stage 12 目录不存在 ✗
ls /fs04/ar57/wenyu/result/personal_query/12_complexity_analysis_clause_features/
# → NOT FOUND
```

**结论**：pipeline 在 Stage 04 → Stage 06 之间断裂。

### C.2 下一步修复方向

1. **诊断 Stage 06 为何无输出**：运行 Stage 06 相关脚本，检查数据传递
2. **补充缺失的 query 文件**：确保 `query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json` 存在
3. **Stage 10 hierarchical merging**：参考 Baudry et al. [2] 增加 component merging 步骤
4. **Paper 补充**：Δ Range 计算细节、Bootstrap CI 方法、8-cluster 选择依据

---

## §D 更新 loop.md §11

- P0 #3 (GMM clustering) → 加入 Baudry et al. [2] hierarchical merging 引用
- 新增 P0 条目：**Stage 06 query pipeline 断链**，影响 Resource Paper 可复现性

## §E 后续 iter

- **iter #195**: 诊断 Stage 06 pipeline 断链原因，修复数据传递
- **iter #196**: 运行 Stage 10 GMM（含 hierarchical merging）生成可用的 8-cluster 输出
- **iter #197**: Bootstrap CI 修复（需 Stage 06 full query pool）
