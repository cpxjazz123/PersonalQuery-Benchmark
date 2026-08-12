# Iteration 252 — Stage 10 GMM Pipeline spaCy 安装 + 端到端测试

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 可复现性视角）
**scope**: spaCy 安装 + GMM clustering pipeline 端到端测试 + attach_retrieval_results 结构性不兼容确认

---

## §A 审稿意见

### 问题：spaCy 缺失 + retrieval attachment 结构性不兼容

iter #251 修复后，pipeline 仍有两层问题：
1. spaCy 未安装 → GMM feature extraction 失败 ✅ 已修复
2. `attach_retrieval_results()` 设计期望 per-user `user_id` 记录，但实际 retrieval 输出只有 group-level 聚合

---

## §B 修复内容

### Fix 1: spaCy 安装

```bash
pip install spacy  # → spacy 3.8.14
python3 -m spacy download en_core_web_sm  # → en-core-web-sm-3.8.0
```

### Fix 2: `cluster_strict5550_query_gmm_and_attach_retrieval.py` 三处路径修复

```
修复前 (main branch):
1. load_query_rows: syntax_depth_query
2. default query_file: vades_lite_sentence...10_holdout10.json
3. retrieval path: 08_retrieval/...expression_style_*

修复后:
1. load_query_rows: expression_style_query
2. default query_file: query_by_expression_style_no_depth_check_10.json
3. retrieval path: 06_retrieval/...syntax_depth_summary.json
```

### Fix 3: kruskal groups 不足防护

```python
# 修复前: kruskal 遇到 <2 groups 直接 raise ValueError
# 修复后:
try:
    kruskal_stat, kruskal_pvalue = kruskal(...)
except ValueError as e:
    if "Need at least two groups" in str(e):
        log(f"警告: retriever={retriever_name} cluster groups不足，跳过他统计")
        kruskal_stat, kruskal_pvalue = float("nan"), float("nan")
    else:
        raise
```

---

## §C 端到端测试结果

### GMM Clustering 部分 ✅ 成功

```
selected_k: 8
cluster_counts: {cluster_0: 22, cluster_1: 16, cluster_2: 9, cluster_3: 8, cluster_4: 7, cluster_5: 6, cluster_6: 4, cluster_7: 1}
PCA selected_dim: 5 (cumulative variance ≥ 90%)
```

### attach_retrieval_results 部分 ❌ 结构性不兼容

```
错误: ValueError: zero-size array to reduction operation maximum
根因: 
  - Stage 06 retrieval 输出中 all_query_records = [] (空)
  - Stage 06 数据按 group-level 聚合 (high_complexity/low_complexity)
  - attach_retrieval_results 期望 per-user user_id 记录来 attach 到 clusters
  - 实际数据结构: {retriever: {group_metrics: {H@10, MR@10, ...}}}
```

**retrieval 输出结构**:
```python
all_results_combined[0] = {
    'retriever': 'bge',
    'num_users': 2,
    'num_queries': 2,
    'metrics': {'H@10': 0.5, ...},
    'group_metrics': {'high_complexity': {'H@10': 0.5, ...}},
    'all_query_records': [],  # ← 空，无 per-user 数据
}
```

---

## §D 根因：paper 8-cluster Δ Range 数据从未来自当前代码

**确认**：当前代码库中的 GMM clustering (`cluster_strict5550_query_gmm_and_attach_retrieval.py`) 从未被端到端跑通过，原因：
1. iter #194 之前：spaCy 未安装，feature extraction 失败
2. iter #251 之前：文件名/字段名/路径完全不匹配
3. iter #252 现在：retrieval attachment 结构性不兼容

**`selected_k=8` 的 GMM clustering 结果**：已能成功生成（73 queries → 8 clusters，PCA 5维）。但 Δ Range 计算完全依赖 `attach_retrieval_results()`，而该函数与实际 retrieval 数据结构不兼容。

**paper Table 1 上方面板 8-cluster Δ Range 值**：(SPLADE Δ=9.6, DeepSeek-v4 Δ=9.1 等)必然来自其他代码路径——可能：
- 早期已删除的 2-bucket clustering 代码
- 或 paper 数值来自 simulation 而非真实 pipeline 运行

---

## §E 验证

```bash
$ python3 -c "import spacy; print(spacy.__version__)"
3.8.14

$ python3 -m py_compile 10_complexity_analysis/common/cluster_strict5550_query_gmm_and_attach_retrieval.py
(no output = OK)

$ python3 -c "
from cluster_strict5550_query_gmm_and_attach_retrieval import run_query_gmm_pipeline
result = run_query_gmm_pipeline(category='Baby_Products', attach_retrieval=False)
print('GMM clustering OK: selected_k=', result['cluster_selection']['selected_k'])
"
GMM clustering OK: selected_k= 8
```

---

## §F 下一步建议

**iter #253**: 
1. 修复 `attach_retrieval_results()` 使用 group-level metrics（而非 per-user）
2. 或者：确认 paper 8-cluster Δ Range 来自 2-bucket 代码，作为 Limitation 文档化
3. 无论哪种方案，都需要明确 GMM clustering 输出格式与下游 Δ 计算的接口契约

---

## §G 参考文献

- **[Kasa et al., 2020](https://consensus.app/papers/details/d741e6d9e727574392b563956a3204e1/?utm_source=claude_code)** [1] — GMM reproducibility: "intermediate data artifacts must be persisted for downstream verification." Stage 12 输出已有（GMM clustering 结果），但 retrieval attachment 接口断裂使 downstream Δ 计算无法执行。
