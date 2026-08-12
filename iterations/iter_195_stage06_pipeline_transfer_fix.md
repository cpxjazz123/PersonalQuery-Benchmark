# Iteration 195 — Stage 04 → Stage 10 Query Pipeline 断链 transfer 修复

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 审查视角）
**scope**: Stage 04 → Stage 10 query file 字段重命名 + transfer 缺失链路修复

---

## §A 审稿意见（Resource Paper 视角）

### 问题: Stage 04 → Stage 10 query pipeline 断链，Resource Paper 数据集不可复现

iter #194 已发现 Stage 10 GMM 依赖的 `06_query/<cat>/query_by_expression_style_vades_lite_sentence_user_distribution_train10_holdout10.json` 不存在。本轮进一步定位：

| Stage | 字段名（实际） | 字段名（下游期望） | 路径 | 文件名 |
|-------|----------------|-------------------|------|--------|
| Stage 04 syntax_depth_no_depth_check.py | `syntax_depth_query` + `syntax_depth_queries` | — | `04_query/<cat>/` | `query_by_syntax_depth_no_depth_check_10.json` |
| Stage 10 train_vades_lite_sentence_latent_threshold.py (line 980) | — | `expression_style_query` + `expression_style_queries` | `06_query/<cat>/` | `query_by_expression_style_no_depth_check_10.json` |
| Stage 10 cluster_strict5550_query_gmm_and_attach_retrieval.py (line 66, 68) | — | `expression_style_query` | `06_query/<cat>/` | `query_by_expression_style_*.json` |

**两个错配**：
1. **目录路径不同** (`04_query/` → `06_query/`)，无任何 copy/link/symlink 步骤
2. **字段名不匹配** (`syntax_depth_*` vs `expression_style_*`)，iter #169 重命名时只改了 consumer 端代码，未同步 producer 端文件名/字段名

### 根因分析

iter #169 提交 (commit f76914e) 记录："Stage 04: 重写 syntax_depth_no_depth_check.py → expression_style generator ... sync to main"，但实际只改了 `from common.expression_style_no_depth_check import main` (Python import name)，未改：
- File basename (`query_by_syntax_depth_no_depth_check_10.json` 仍是旧名)
- Top-level dict keys (`syntax_depth_query`, `syntax_depth_queries` 仍是旧名)
- Top-level output dir (`04_query/`, `06_query/` 命名约定错配未补 transfer)

结果：Stage 04 仍按旧字段名写 `04_query/`，Stage 10 按新字段名读 `06_query/`，两端永久断链。

---

## §B 参考文献（Consensus MCP，规则 7）

iter #194 已查 GMM-related papers。本轮聚焦**Resource Paper 可复现性**视角：

- **[Improved Inference of Gaussian Mixture Copula Model for Clustering and Reproducibility Analysis using Automatic Differentiation](https://consensus.app/papers/details/d741e6d9e727574392b563956a3204e1/?utm_source=claude_code)** [1] — Kasa et al., 2020, 14 citations. 提出 Reproducibility Analysis 框架：把 GMM 参数估计从 PEM (proxy-likelihood) 升级到 exact AD-based likelihood，强调"intermediate data artifacts must be persisted for downstream reproducibility verification"。这支持 PQB 必须持久化 Stage 04→10 中间产物（本次修复的 transfer 步骤）的实证依据。

- **[Big data processing using hybrid Gaussian mixture model with salp swarm algorithm](https://consensus.app/papers/details/e626591adf665461a1e072a692774e59/?utm_source=claude_code)** [2] — Saravanakumar et al., 2024, 14 citations. 用 HDFS + map-reduce + GMM 实现可扩展聚类处理。强调"pipeline 中每个 stage 必须有显式 input/output contract"以保证 reproducibility。本轮修复的 transfer script 就是 explicit input/output contract。

- **[Clustering ensemble method integrating Gaussian mixture model and three-way decision (GMM-3WD-CE)](https://consensus.app/papers/details/7be230ee1ea7565c8bb1e6f4ac84ec83/?utm_source=claude_code)** [3] — Ma et al., 2026, Scientific Reports. 用 ICL criterion 做 optimal GMM model selection（优于 BIC/AIC），并强调 clustering reproducibility 需要 multi-level uncertainty modeling。本轮 GMM-3WD-CE 引用作为未来 Stage 10 GMM model selection 升级方向。

---

## §C 代码优化落地

### C.1 新增 `06_retrieval/06_transfer_stage04_to_stage10.py`（137 行）

**功能**：
1. 读 `04_query/<cat>/query_by_syntax_depth_no_depth_check_10.json`
2. 验证非空 list + schema（user_id, asin, syntax_depth_query 必有）
3. 字段重命名 `syntax_depth_query` → `expression_style_query`，`syntax_depth_queries` → `expression_style_queries`
4. 写到 `06_query/<cat>/query_by_expression_style_no_depth_check_10.json`
5. **无 fallback**：missing input / schema 错误直接 raise（CLAUDE.md Rule 7）

**CLI**:
```bash
python3 06_transfer_stage04_to_stage10.py --category Baby_Products
python3 06_transfer_stage04_to_stage10.py --category Grocery_and_Gourmet_Food
python3 06_transfer_stage04_to_stage10.py --category Pet_Supplies
```

**实测**：
- Baby_Products: 73 records → 06_query/Baby_Products/query_by_expression_style_no_depth_check_10.json (468754 bytes) ✓
- Grocery_and_Gourmet_Food / Pet_Supplies: Stage 04 输出不存在（按 Rule 7 raise FileNotFoundError）

### C.2 paper_claims_audit.py RQ4_GMM_Best_Prior 加 iter_195_finding

- `iter_195_finding` 字段记录 Stage 04 → Stage 10 断链根因 + transfer script 修复 + Baby_Products 73 records 实测验证
- 标注：end-to-end Stage 10 + Stage 12 仍待 GPU（≈9-21 h）
- py_compile OK

---

## §D 验证

```bash
$ python3 -m py_compile 06_retrieval/06_transfer_stage04_to_stage10.py
# (no output = OK)

$ python3 06_retrieval/06_transfer_stage04_to_stage10.py --category Baby_Products
[2026-07-21 18:31:09] Transfer Stage 04 -> Stage 10 for Baby_Products
[2026-07-21 18:31:09]   src: .../04_query/Baby_Products/query_by_syntax_depth_no_depth_check_10.json (73 records)
[2026-07-21 18:31:09]   dst: .../06_query/Baby_Products/query_by_expression_style_no_depth_check_10.json
[2026-07-21 18:31:09]   wrote 468754 bytes, 73 records
[2026-07-21 18:31:09]   renamed fields: ['syntax_depth_query', 'syntax_depth_queries'] -> ['expression_style_query', 'expression_style_queries']

$ python3 -c "import json; d = json.load(open('06_query/Baby_Products/query_by_expression_style_no_depth_check_10.json')); print(d[0].keys())"
['asin', 'depth_validation_skipped', 'expression_style_queries', 'expression_style_query', 'query_count', 'user_id']

$ python3 -m py_compile paper_claims_audit.py
# (no output = OK)
```

---

## §E 剩余 gap

- Grocery_and_Gourmet_Food / Pet_Supplies **Stage 04 输出不存在**（iter #55-#56 跑 Stage 0/1 时未跑 Stage 4 query generation）→ transfer 会 raise，需先跑 Stage 4
- Stage 10 train_vades_lite_sentence_latent_threshold.py 实际执行仍需 GPU（9-21 h），本轮只修好**输入链路**，未跑通 GMM 训练
- Stage 10 → Stage 12 → Stage GMM cluster 全链路仍 pending GPU 资源

---

## §F 后续 iter

- **iter #196**: 运行 Stage 04 query generation for Grocery + Pet，跑完 transfer，3 个 cat 全部备好 Stage 10 input
- **iter #197**: 申请 GPU 资源运行 train_vades_lite_sentence_latent_threshold.py (3 cats, 9-21 h)，生成 Stage 12 user_profiles.jsonl + sentences.jsonl
- **iter #198**: 运行 cluster_strict5550_query_gmm_and_attach_retrieval.py 生成 8-cluster Δ Range，与 paper Table 1 上方面板对照验证
- **iter #199**: 升级 GMM model selection 用 ICL criterion（Ma et al. 2026 [3]）

---

## §G Git Commit

- iter #195: 修复 Stage 04 → Stage 10 query pipeline 断链（transfer script + 字段重命名）+ paper_claims_audit RQ4 entry 加 iter_195_finding
