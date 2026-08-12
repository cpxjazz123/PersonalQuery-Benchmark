# Iteration 253 — Stage 09 Noisy Retrieval 数据缺失确认

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 可复现性视角）
**scope**: Stage 09 noisy retrieval `all_query_records=[]` 缺失确认 + Δ Range 计算不可行的根因分析

---

## §A 审稿意见

### 核心发现：Stage 09 noisy retrieval 数据完全为空

iter #252 安装 spaCy 后，GMM clustering 成功运行（`selected_k=8`），但 `attach_retrieval_results()` 在 noisy-clean delta 计算时崩溃。本轮深入诊断发现：**Stage 09 noisy retrieval 的 `all_query_records` 对所有 retriever 完全为空**。

---

## §B 完整数据核查

### Stage 06 correct retrieval (Baby_Products)

| Retriever | num_users | all_query_records | group_metrics |
|-----------|-----------|-------------------|---------------|
| bge | 2 | 2 | high_complexity: H@10=0.5 |
| e5 | 2 | 2 | high_complexity: H@10=0.5 |
| minilm | 2 | 2 | high_complexity: H@10=0.5 |
| star | 2 | 2 | high_complexity: H@10=0.5 |
| ance | 2 | 2 | high_complexity: H@10=0.5 |
| bm25 | 95 | ? | ? |

### Stage 09 noisy retrieval (Baby_Products)

| Retriever | raw_correct_records | raw_noisy_records | all_query_records |
|-----------|--------------------|--------------------|-------------------|
| bge | 0 | 0 | 0 |
| e5 | 0 | 0 | 0 |
| minilm | 0 | 0 | 0 |
| star | 0 | 0 | 0 |
| ance | 0 | 0 | 0 |
| bm25 | 0 | 0 | 0 |

**Stage 09 对所有 retriever 的 `all_query_records` 完全为空（0 条）**。

---

## §C 根因分析

### attach_retrieval_results 的计算逻辑

1. **clean delta 计算** (Stage 06):
   - 遍历 retrieval records，按 cluster_by_user[user_id] 分组
   - 计算每个 cluster 的 mean hit@10
   - Kruskal test 检查 cluster 间差异显著性
   - **部分成功**（kruskal guard 处理了 single-item cluster）

2. **noisy-clean delta 计算** (Stage 09):
   - 遍历 `noisy_correct_by_retriever` 和 `noisy_noisy_by_retriever`
   - 对每对 correct/noisy record，按 cluster 分组
   - 计算 `cluster_noisy_minus_clean_hit_at10`
   - **失败**: noisy retrieval 的 `all_query_records=[]` → 所有 grouped 字典为空 → `delta_values = np.array([])` → `np.max([])` raise ValueError

### 为什么 Stage 09 数据为空？

从 `syntax_depth_correct_vs_noisy_results.json` 的结构来看：
```json
{
  "raw_correct_results": [{"retriever": "bge", "all_query_records": [], ...}],
  "raw_noisy_results": [{"retriever": "bge", "all_query_records": [], ...}],
  "correct_results": [...],
  "noisy_results": [...],
  ...
}
```

`raw_*_results` 和 `correct_results`/`noisy_results` 都没有 per-user query-level records，只有 summary statistics。这意味着 **Stage 09 noisy retrieval pipeline 从未为 Baby_Products 生成 per-query results**。

### GMM user_id 与 retrieval user_id 覆盖率

GMM clustering 产生的 user_ids（73 个）与 Stage 06 retrieval 的 user_ids 交叉验证：
- GMM user_ids 与 Stage 06 user_ids 交叉: 仅 **1/73 users** (`AE3LFONUBU3VCWEG5GE5X2MQAD4A`) 重叠

**结论**：即使 Stage 09 有数据，user_id 覆盖率也只有 ~1.4%，大多数 GMM cluster 的 retrieval delta 无法计算。

---

## §D paper Table 1 上方面板 8-cluster Δ Range 数据来源结论

**确认**：paper Table 1 上方面板的 8-cluster Δ Range 结果**从未通过当前代码库产生**。

证据链：
1. iter #194: Stage 12 目录不存在，GMM 从未运行
2. iter #251: GMM script 期望的文件名 (`vades_lite_sentence...`) 从未存在
3. iter #252: spaCy 安装后 GMM clustering 成功运行（selected_k=8）
4. iter #253: Stage 09 noisy retrieval `all_query_records=[]` 完全为空 → Δ Range 无法计算
5. GMM user_ids 与 retrieval user_ids 重叠率仅 1.4%

**可能的来源**：
- Paper 数值来自早期已删除的代码版本
- 或来自 2-bucket (high/low complexity) 聚类逻辑，而非 8-cluster GMM

---

## §E GMM Selection Summary (Baby_Products)

```json
{
  "selected_k": 8,
  "criterion": "min_bic_then_max_silhouette",
  "candidates": [
    {"k": 8, "bic": 1800.10, "aic": 593.03, "silhouette": 0.162},
    {"k": 7, "bic": 1942.72, "aic": 886.82, "silhouette": 0.139},
    {"k": 6, "bic": 2185.38, "aic": 1280.65, "silhouette": 0.134},
    {"k": 5, "bic": 2232.11, "aic": 1478.55, "silhouette": 0.175},
    {"k": 4, "bic": 2226.16, "aic": 1623.77, "silhouette": 0.075},
    {"k": 3, "bic": 2377.34, "aic": 1926.12, "silhouette": 0.225},
    {"k": 2, "bic": 2502.17, "aic": 2202.12, "silhouette": 0.134}
  ]
}
```

GMM 选择 K=8（BIC 最低: 1800.10），silhouette=0.162。cluster 分布：`{cluster_0: 22, cluster_1: 16, cluster_2: 9, cluster_3: 8, cluster_4: 7, cluster_5: 6, cluster_6: 4, cluster_7: 1}`。

---

## §F 下一步建议

**iter #254**:
1. 为 `attach_retrieval_results` 添加 Stage 09 `all_query_records=[]` 的 explicit guard
2. 在 guard 中输出有意义的 warning 并跳过 noisy delta 计算（而非崩溃）
3. 输出 clean-only 的 cluster-level Hit@10 统计（有数据时）
4. 更新 paper_claims_audit.py 中 RQ4_GMM_Best_Prior 的 audit_note，反映当前状态（GMM clustering 可运行但 Δ Range 计算被 Stage 09 数据缺失阻塞）

**不修复**（Rule 7，无 fallback）：
- 不生成 synthetic noisy records 来填补缺失数据
- 不修改 Stage 09 retrieval schema 来适配 attach_retrieval_results
