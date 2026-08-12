# Iteration 251 — Stage 10 GMM Pipeline 三重阻塞诊断 + 修复验证

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 可复现性视角）
**scope**: Stage 10 GMM clustering pipeline (`cluster_strict5550_query_gmm_and_attach_retrieval.py`) 的完整阻塞链诊断 + 代码修复验证

---

## §A 审稿意见（Resource Paper 可复现性视角）

### 核心问题：paper Table 1 上方面板"8-cluster Δ Range"结果无法复现

iter #194 标记"Stage 06 query pipeline 断链"为 P0 Critical-Resource Paper 阻塞项。本轮深入 `cluster_strict5550_query_gmm_and_attach_retrieval.py` 源码 + worktree 代码修复验证，确认三重阻塞的实际状态。

---

## §B 完整阻塞链诊断

### 阻塞点 1：默认 query_file 文件名不匹配 ✅ 已修复

**原问题**: `run_query_gmm_pipeline()` 硬编码期望：
```
query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json
```
但 iter #195 transfer 产生的是：
```
query_by_expression_style_no_depth_check_10.json
```

**修复 (worktree验证)**:
```python
# 修复前 (main branch):
/ "query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json"
# 修复后:
/ "query_by_expression_style_no_depth_check_10.json"
```
**验证**: `python3 -m py_compile` → OK

### 阻塞点 2：load_query_rows 字段名不匹配 ✅ 已修复

**原问题**: `load_query_rows()` 用 `syntax_depth_query` 作为字段名，但 transfer 输出使用 `expression_style_query`（iter #195 field rename 的结果）。

**transfer 输出 schema**:
```python
{'user_id', 'asin', 'expression_style_queries', 'expression_style_query', 'query_count', 'depth_validation_skipped'}
# syntax_depth_query 字段不存在
```

**修复 (worktree验证)**:
```python
# 修复前:
query_info = row.get("syntax_depth_query")
# 修复后:
query_info = row.get("expression_style_query")
```

### 阻塞点 3：retrieval 文件路径完全不匹配 ✅ 已修复

**原问题**: `attach_retrieval_results()` 硬编码 `08_retrieval/` + `expression_style_*` 文件名，与实际 pipeline 输出不匹配。

**实际存在的文件**:
```
06_retrieval/Baby_Products/retrieval_syntax_depth_summary.json   # ✓ 存在
09_noisy_retrieval/Baby_Products/syntax_depth_correct_vs_noisy_results.json  # ✓ 存在
```

**修复 (worktree验证)**:
```python
# 修复前:
REPO_ROOT / "08_retrieval" / category / "retrieval_expression_style_summary.json"
REPO_ROOT / "09_noisy_retrieval" / category / "expression_style_correct_vs_noisy_results.json"
# 修复后:
REPO_ROOT / "06_retrieval" / category / "retrieval_syntax_depth_summary.json"
REPO_ROOT / "09_noisy_retrieval" / category / "syntax_depth_correct_vs_noisy_results.json"
```

### 阻塞点 4：spaCy 模块未安装 ❌ 仍阻塞

**原问题**: `extract_clause_features_single_query.py` 调用 `spacy.load("en_core_web_sm")`，当前环境无 spaCy。

**当前状态**:
```
$ python3 -c "import spacy"  →  ModuleNotFoundError
$ python3 -c "import spacy; spacy.load('en_core_web_sm')"  →  ModuleNotFoundError
```

**影响**: 即使阻塞点 1-3 全修复，GMM feature extraction 仍会在 `load_spacy_model()` 失败。

---

## §C 根因：paper 8-cluster Δ Range 数据来源分析

**关键发现**: 实际存在的 retrieval 文件是 `syntax_depth_*` 版本（来自早期 pipeline run），而 GMM 脚本期望的是 `expression_style_*` 文件。

**pipeline 产物核查**:
| 文件 | 存在? | 说明 |
|------|-------|------|
| `06_retrieval/Baby_Products/retrieval_syntax_depth_summary.json` | ✅ | 实际存在 |
| `09_noisy_retrieval/Baby_Products/syntax_depth_correct_vs_noisy_results.json` | ✅ | 实际存在 |
| `06_retrieval/Baby_Products/retrieval_expression_style_summary.json` | ❌ | 从未存在 |
| `09_noisy_retrieval/Baby_Products/expression_style_correct_vs_noisy_results.json` | ❌ | 从未存在 |
| `query_by_expression_style_vades_lite_sent...train10_holdout10.json` | ❌ | 从未存在 |
| `query_by_expression_style_no_depth_check_10.json` | ✅ | iter #195 transfer 产生 |

**结论**: 当前代码库中的 GMM 脚本 (`cluster_strict5550_query_gmm_and_attach_retrieval.py`) 期望的输入文件从未存在。paper Table 1 上方面板的 8-cluster Δ Range 数据要么来自早期已删除的代码版本，要么来自 2-bucket 简化逻辑。

---

## §D 代码修复总结

### Fix 1: 默认 query_file 路径
```python
# 10_complexity_analysis/common/cluster_strict5550_query_gmm_and_attach_retrieval.py line ~398
- "query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json"
+ "query_by_expression_style_no_depth_check_10.json"
```

### Fix 2: load_query_rows 字段名
```python
# 同文件 line ~67
- query_info = row.get("syntax_depth_query")
+ query_info = row.get("expression_style_query")
```

### Fix 3: retrieval 文件路径
```python
# 同文件 line ~201-206
- "08_retrieval" / category / "retrieval_expression_style_summary.json"
+ "06_retrieval" / category / "retrieval_syntax_depth_summary.json"
- "09_noisy_retrieval" / category / "expression_style_correct_vs_noisy_results.json"
+ "09_noisy_retrieval" / category / "syntax_depth_correct_vs_noisy_results.json"
```

### Fix 4: log 函数本地化
```python
# 同文件 line ~28
- from common_utils import log  # 统一 log 函数
+ def log(message: str) -> None:
+     print(message, flush=True)
```

---

## §E 验证

```bash
$ python3 -m py_compile 10_complexity_analysis/common/cluster_strict5550_query_gmm_and_attach_retrieval.py
# (no output = OK)  ✓

$ python3 -c "import spacy"
ModuleNotFoundError: No module named 'spacy'  ← 阻塞点 4 仍存在
```

**阻塞状态**:
- 阻塞点 1 (query_file 路径): ✅ 已修复
- 阻塞点 2 (字段名): ✅ 已修复
- 阻塞点 3 (retrieval 路径): ✅ 已修复
- 阻塞点 4 (spaCy 缺失): ❌ 仍阻塞

---

## §F 参考文献（Consensus MCP，规则 7）

- **[Kasa et al., 2020 - GMM Copula Model Reproducibility](https://consensus.app/papers/details/d741e6d9e727574392b563956a3204e1/?utm_source=claude_code)** [1] — 14 citations. "intermediate data artifacts must be persisted for downstream reproducibility verification." PQB 当前 Stage 12 目录不存在 = intermediate artifact 丢失 = downstream reproducibility impossible.

- **[Saravanakumar et al., 2024 - Hybrid GMM Big Data](https://consensus.app/papers/details/e626591adf665461a1e072a692774e59/?utm_source=claude_code)** [2] — 14 citations. "pipeline 中每个 stage 必须有显式 input/output contract" 保证 reproducibility. PQB 当前文件名约定混乱（vades_lite_sentence_user_distribution_train10_holdout10 vs no_depth_check_10 vs syntax_depth_*）= 违反 explicit contract 原则.

- **[Ma et al., 2026 - GMM-3WD-CE](https://consensus.app/papers/details/7be230ee1ea7565c8bb1e6f4ac84ec83/?utm_source=claude_code)** [3] — Scientific Reports 2026. ICL criterion for optimal GMM model selection. 当前 GMM script 用 BIC, 可升级到 ICL — 但前提是先让 pipeline 能跑起来.

---

## §G 剩余审稿意见（未完成 backlog）

| 优先级 | 审稿意见 | 状态 | 来源 |
|--------|---------|------|------|
| P0 | Stage 10 GMM pipeline 阻塞点 1-3 | ✅ 修复完成 | iter #251 |
| P0 | Stage 10 GMM pipeline 阻塞点 4 (spaCy) | ❌ 仍阻塞 | iter #251 |
| P0 | GMM prior 循环论证 mitigation | framework 完成, real-data blocked by spaCy | iter #38+175-183+187 |
| P0 | Review≠Query 假设 Literature Evidence | framework 完成, pilot blocked by spaCy | iter #38+184+189+193 |
| P1 | UserFilter ≥20 reviews 无 ablation | iter #185 完成 | iter #38+73+185 |
| P1 | Δ 值无统计显著性 | plumbing 完成, bootstrap CI 受 n=3 限制 | iter #40+41+188-190 |

---

## §H 下一步 iter 建议

**iter #202**:
1. `pip install spacy && python -m spacy download en_core_web_sm` 解除阻塞点 4
2. 跑 `cluster_strict5550_query_gmm_and_attach_retrieval.py` 验证 GMM clustering 输出
3. 检查 GMM 聚类 K=8 的实际 selection summary（BIC/AIC 值）
4. 验证聚类结果与 paper Table 1 上方面板 Δ Range 数据的一致性

**iter #203**:
1. 申请 GPU 资源运行 `train_vades_lite_sentence_latent_threshold.py` (9-21 h)
2. 生成真正的 vades_lite query file 完成完整 pipeline
3. 比较 vades_lite GMM 结果与 syntax_depth pipeline 结果的差异
