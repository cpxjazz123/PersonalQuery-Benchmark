# Iteration 195 — Stage 06 Query Pipeline断链诊断

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 审查视角）
**scope**: Stage 10 train_vades_lite 从未运行导致核心 query 文件缺失

---

## §A 审稿意见（Resource Paper 视角）

### 核心问题：Stage 10 (train_vades_lite_sentence_latent_threshold.py) 从未运行

**pipeline 链路梳理：**

| Stage | 脚本 | 输出 |
|-------|------|------|
| Stage 04 | `04_query/common/syntax_depth_no_depth_check.py` | `04_query/<cat>/query_by_syntax_depth_no_depth_check_10.json` (73 users, format: syntax_depth_queries) |
| Stage 06 | `06_retrieval/` | **retrieval 构建，非 query 生成** |
| Stage 10 | `10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py` | 应输出 `06_query/<cat>/query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json` + Stage 12 目录 |
| Stage 12 | VAE user profiles + sentences | **目录不存在，从未运行** |

### 根因确认

1. `train_vades_lite_sentence_latent_threshold.py` 的 `QUERY_FILE` 输出路径是：
   ```
   REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / f"query_by_expression_style_{OUTPUT_TAG}.json"
   ```
   即 `06_query/<cat>/query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json`

2. 但 `result/personal_query/12_complexity_analysis_clause_features/<cat>/` 目录不存在 → Stage 10 VAE 训练从未执行

3. `result/personal_query/06_query/Baby_Products/` 为空目录

### 实际数据流

```
Stage 04 output:  query_by_syntax_depth_no_depth_check_10.json
                      ↓ (train_vades_lite reads as RAW_CANDIDATE_QUERY_FILE)
Stage 10: train_vades_lite_sentence_latent_threshold.py
  - INPUT:  04_query/<cat>/query_by_syntax_depth_no_depth_check_10.json
  - OUTPUT: 06_query/<cat>/query_by_expression_style_vades_lite_sentence_*.json  ← 从未生成
            12_complexity_analysis_clause_features/<cat>/                          ← 目录不存在
```

Stage 04 有数据，Stage 10 从未运行 → 中间无传递脚本（数据直接在 Stage 10 内从 04 读，写到 06 + 12）

---

## §B 影响评估

### 对 Resource Paper 可复现性的影响

| 缺失组件 | 影响 |
|---------|------|
| `query_by_expression_style_vades_lite_sentence_*.json` | Stage 10/11/14 后续脚本全部无法运行 |
| `12_complexity_analysis_clause_features/` | GMM clustering 无法运行，Table 1 上方面板 8-cluster Δ Range 无法计算 |
| Stage 10 VAE 训练 (9-21h GPU) | 核心数据生成步骤未执行，Resource Paper 数据集不完整 |

### 当前状态

- Stage 04 query 文件存在（73 users，有 syntax_depth_queries 格式）
- Stage 06 retrieval cache 存在（如 06_retrieval/ 有脚本）
- Stage 10 train_vades_lite **从未运行**
- 所有依赖 Stage 10 输出的脚本都无法运行

---

## §C 落地修复方向

### 修复方案

**方案 A（最小修复）：运行 Stage 10 train_vades_lite**
- 执行 `train_vades_lite_sentence_latent_threshold.py`（需 GPU，约 9-21h）
- 生成 `query_by_expression_style_vades_lite_sentence_*.json` 到 06_query/
- 生成 Stage 12 用户/句子 VAE latent 到 12_complexity_analysis_clause_features/
- 然后运行 `cluster_strict5550_query_gmm_and_attach_retrieval.py` 生成 8-cluster Δ Range

**方案 B（更优设计）：增加 query 文件传递验证**
- 在 `cluster_strict5550_query_gmm_and_attach_retrieval.py` 开头加文件存在性检查
- 若缺失，打印明确错误信息而非 silent failure
- Paper 补充 "Data Generation Pipeline" 章节，说明 Stage 04→10→11 依赖关系

### 当前最高 ROI

先运行 Stage 10 train_vades_lite（需要 GPU 资源分配）。在运行前，可先验证 Stage 04 query 文件格式是否与 Stage 10 输入兼容。

---

## §D 验证检查清单

```bash
# 1. Stage 04 query 文件存在 ✓
ls /fs04/ar57/wenyu/result/personal_query/04_query/Baby_Products/query_by_syntax_depth_no_depth_check_10.json

# 2. Stage 10 输入兼容（检查 RAW_CANDIDATE_QUERY_FILE 是否被 train_vades_lite 读取）
# train_vades_lite_sentence_latent_threshold.py line 35-38:
# RAW_CANDIDATE_QUERY_FILE = ... / "06_query" / CATEGORY / "query_by_syntax_depth_no_depth_check_10.json"
# 注意：train_vades_lite 从 06_query 目录读 Stage 04 输出！
# 但 06_query/Baby_Products/ 为空 → train_vades_lite 也会失败！

# 3. 修复：需先将 Stage 04 输出复制到 06_query/ 再运行 Stage 10
cp 04_query/Baby_Products/query_by_syntax_depth_no_depth_check_10.json \
   06_query/Baby_Products/query_by_syntax_depth_no_depth_check_10.json

# 4. 然后运行 Stage 10 train_vades_lite
```

---

## §E 下一步

- **iter #196**: 执行方案 A：复制 Stage 04 → Stage 06，运行 Stage 10 train_vades_lite
- **iter #197**: 运行 cluster_strict5550_query_gmm_and_attach_retrieval.py 生成 8-cluster 输出
- **iter #198**: 验证 Table 1 上方面板 Δ Range 数值与 paper 一致性
