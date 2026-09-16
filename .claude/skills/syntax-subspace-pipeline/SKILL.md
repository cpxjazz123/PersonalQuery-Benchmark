---
name: syntax-subspace-pipeline
description: 按当前主链路运行 PersonalQuery 语法子空间、query 选择、用户错误画像、typo 注入与 paired retrieval 评估；所有阶段以当前产物 schema 和 signature 为准，不固化易变的数量。
---

# PersonalQuery Syntax-Subspace Pipeline

## 何时使用

当需要从 Amazon Baby Products 原始评论重新生成用户句子、训练监督式语法表示、拟合 per-user Gaussian、生成并选择个性化 query、构建用户错误画像、注入 typo，并评估 typo 对 retrieval 的影响时，使用本技能。主链路是 Stage 01、02、03、04、05、07、08、09、10、11、12；Stage 06 仅提供 Stage 07 所需的既有 SFT 训练产物，不在本技能中重新训练。

本技能只描述流程与数据契约，不写入某次运行的用户数、query 数、pair 数、命中率或耗时。每次运行的实际统计必须从对应结果文件的 `config`、`summary`、`totals` 和 `per_retriever` 字段读取。

## 总体流程

```text
原始 review JSONL
  ↓
Stage 01：商品属性抽取
  ↓
Stage 02：HTML 清洗、句子切分、用户聚合、ASIN→用户映射
  ↓
Stage 03：spaCy/PCFG 规则特征 + supervised syntax encoder + embedding cache
  ↓
Stage 04：在冻结 embedding 上拟合 per-user raw full-covariance Gaussian + ASIN cohort gate
  ↓
Stage 05：对 Gaussian 做 raw covariance validity audit，产出可用用户与 ASIN coverage
  ↓
Stage 07：使用既有 SFT adapter 通过本地 Qwen 批量生成 query pool
  ↓
Stage 08：读取 Stage 05 audit，执行 Mahalanobis target/competitor-exclusive query 选择
  ↓
Stage 09：在 Stage 02 清洗后的句子上运行 GECToR/ERRANT/SErCL，建立用户错误画像
  ↓
Stage 10：读取 Stage 08 selection、Stage 04 Gaussian、Stage 09 profile，执行单次 typo 候选注入与 gates
  ↓
Stage 11：独立运行七类 retriever，保存 canonical retrieval 评估文件
  ↓
Stage 12：读取 Stage 10 成功 pairs，对 original/typo 做 paired hit@k degradation
```

Stage 09 必须在 Stage 02 新产物上重新运行：Stage 02 只负责清洗和聚合句子，不包含 GECToR、ERRANT 或 SErCL 错误画像。Stage 11 必须在 Stage 12 之前独立运行；Stage 12 内部可以复用 Stage 11 的 retriever/cache 实现，但不能把 import Stage 11 模块当作 Stage 11 canonical 评估已经生成。

## 输入、环境与硬性规则

**原始输入**：
- `data/Baby_Products_2023.jsonl`
- `data/meta_Baby_Products_2023.jsonl`

**运行环境**：
- cwd 固定为 `/home/wlia0047/ar57/wenyu/PersoanlQuery`。
- Python 固定为 `/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`。
- GECToR 仅由 Stage 09 指定的独立 `gector_env` 子进程运行。
- 需要 LLM 的业务逻辑只能使用项目 `llm_client.py` 允许的本地 Qwen 后端；不得新增远程 LLM 或业务侧本地推理链路。
- 所有运行参数写在脚本内，不向脚本传命令行参数。
- 长任务使用 `nohup ... &`，日志写入 `/home/wlia0047/hj82_scratch2/wenyu/logs/`；禁止 `sbatch`、`srun` 和 `sleep` 轮询。
- cache、日志和运行时中间文件写入 `/home/wlia0047/hj82_scratch2/wenyu/`；最终结果写入项目 `result/`。
- 新阶段或修改后的阶段先按脚本内 smoke 配置验证数据加载、模型/编码、后处理和输出 schema，再切换 full；批量生成、批量 embedding 和向量化检索必须保持启用。
- 缺失的必需文件、字段、schema、cohort 对齐关系或 signature 必须直接报错，不得静默使用旧输入、默认值或替代路径。

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python
```

## 目录与主脚本

```text
01_attribute_extraction/          extract_product_attrs.py
02_user_review_sentence_extract/  extract_user_sentences.py
03_spacy_encode/                  syntax_pcfg_pipeline.py
04_gaussian/                      fit_per_user_gaussian.py
05_gaussian_audit/                raw_cov_validity.py
06_training_model/                sft_lora/（既有训练产物，不在本技能中训练）
07_gen_query/                     sft_pool_generate.py
08_select_query/                  syntax_select_mahalanobis_gate.py
09_sercl_user_profile/            sercl_user_profile.py
10_typo_injection/                typo_inject.py
11_syntactic_evaluation/          run_stage11_eval.py
11_syntactic_evaluation/          syntax_subspace_retrieval_unified.py
12_typo_evaluation/               typo_retrieval_eval.py
```

**共享 cache**：
- `hj82_scratch2/wenyu/pcfg_cache/`：Stage 03 的 sparse rule cache、vocabulary、用户句子布局、冻结 encoder 和 supervised embeddings。
- `hj82_scratch2/wenyu/gaussian_vades/multiretrieval_embeds/`：Stage 11/12 的 retriever corpus/query embedding cache。
- `result/11_syntactic_evaluation/asin_to_doc.json` 及其 signature：Stage 11/12 的 ASIN 文档 cache。

cache 可以复用，但只能在 corpus、selection 或 query signature 相同且 shape 一致时复用；signature 不一致时必须重建，不得把旧 query embedding 当作当前结果。

## 阶段依赖与运行命令

必须按以下顺序运行。Stage 07 开始前先确认 `result/06_training_model/sft_lora` 已存在；Stage 08 必须等 Stage 05 audit 完成；Stage 09 必须在 Stage 02 完成后重建；Stage 11 必须在 Stage 12 前独立落盘。

```text
Stage 01 → Stage 02 → Stage 03 → Stage 04 → Stage 05
                                      ↓
Stage 06 existing artifact → Stage 07 → Stage 08 → Stage 09 → Stage 10 → Stage 11 → Stage 12
```

每条命令都在项目根目录执行，参数不从命令行传入：

```bash
nohup $PY 01_attribute_extraction/extract_product_attrs.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage01_attrs.log 2>&1 &

nohup $PY 02_user_review_sentence_extract/extract_user_sentences.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage02_sents.log 2>&1 &

nohup $PY 03_spacy_encode/syntax_pcfg_pipeline.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage03_pcfg.log 2>&1 &

nohup $PY 04_gaussian/fit_per_user_gaussian.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage04_gaussian.log 2>&1 &

nohup $PY 05_gaussian_audit/raw_cov_validity.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage05_gaussian_audit.log 2>&1 &

nohup $PY 07_gen_query/sft_pool_generate.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage07_pool.log 2>&1 &

nohup $PY 08_select_query/syntax_select_mahalanobis_gate.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage08_select.log 2>&1 &

nohup $PY 09_sercl_user_profile/sercl_user_profile.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage09_sercl.log 2>&1 &

nohup $PY 10_typo_injection/typo_inject.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage10_typo.log 2>&1 &

nohup $PY 11_syntactic_evaluation/run_stage11_eval.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage11_retrieval.log 2>&1 &

nohup $PY 12_typo_evaluation/typo_retrieval_eval.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage12_typo_eval.log 2>&1 &
```

## 各阶段数据契约与完成标志

### Stage 01：商品属性抽取

读取 review/meta 原始数据，按商品建立后续 query 生成和选择所需的属性结构。完成后检查：

```text
result/01_attribute_extraction/product_attributes.json
```

该文件必须是有效 JSON，并包含 Stage 07/08 所需的商品属性字段；缺失属性应在后续 schema 检查中显式暴露。

### Stage 02：HTML 清洗、用户句子与 ASIN mapping

Stage 02 是下游文本的唯一出口，处理顺序必须是：

```text
raw review text
  → html.unescape() 解码标准 HTML entity
  → 用空格替换 HTML tag，避免词拼接
  → 压缩重复空格并 strip
  → 按句子边界拆分
  → 按用户聚合
  → 删除少于脚本最低句数要求的用户
  → 使用保留用户重建 asin_to_users
  → 写出 JSON 与 PKL
```

这里必须清除 `&#34;`、`&lt;`、`&gt;`、`&amp;` 以及 `<br>` 等 HTML artifact，但不应把普通用户文本中的非标准字符串误当成 HTML。过滤用户后，`asin_to_users.json` 不能继续引用已删除用户，也不能保留无用户的 ASIN。Stage 02 只产生清洗后的句子和映射，不产生 GECToR profile。

完成标志：

```text
result/02_user_review_sentence_extract/uid_to_sentences.json
result/02_user_review_sentence_extract/uid_to_sentences.pkl
result/02_user_review_sentence_extract/asin_to_users.json
```

运行后抽样检查句子中不再出现 HTML entity/tag 残留，并校验 JSON、PKL、ASIN mapping 的用户集合一致。后续 Stage 03 和 Stage 09 都必须读取这批新产物。

### Stage 03：PCFG 特征与 supervised syntax encoder

读取 Stage 02 PKL，使用 spaCy 解析句子并提取不含词项的 dependency/POS 结构规则，建立 sparse sentence-rule cache；随后训练一次 supervised syntax encoder，保存冻结 embedding 和 profile/validation/test 划分。Gaussian 拟合、attribution 和 ablation 不属于 Stage 03。

完成标志至少包括：

```text
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/sent_vectors.npz
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/counts.npz
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/vocab.json
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/user_n_sents.json
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/strict3_embeddings.npz
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/strict3_encoder.pt
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/supervised_embeddings.npy
result/03_spacy_encode/syntax_pcfg_strict_attr.json
```

检查 embedding 的 cohort、句子布局、维度和 finite 值相互一致；发现 cache 属于其他 Stage 02 cohort 时直接停止并重建，不要静默混用。

### Stage 04：per-user Gaussian 与 ASIN cohort gate

Stage 04 只读取 Stage 03 冻结 embedding 和 Stage 02 的 `asin_to_users.json`，在 supervised syntax 空间拟合 per-user raw full covariance Gaussian，并从用户自身 profile/validation 距离生成 cohort gate。Stage 04 是 Gaussian 参数的唯一生产阶段；Stage 08 和 Stage 10 不重新拟合 Gaussian。

生产 artifact 必须明确记录 covariance 类型、gate quantile、embedding 来源和用户/ASIN 对齐信息，并包含每个用户的 `mu`、`sigma_inv`、样本信息及 gate 所需的距离字段。完成标志：

```text
result/04_gaussian/user_gaussian_stats.json
```

先验证 Cholesky/solve、协方差正定性、距离非负以及 user/ASIN 引用完整，再允许 Stage 05 和 Stage 08 使用。rank1、eigenvalue floor、Ledoit–Wolf 等分析变体不能替代 canonical raw full covariance。

### Stage 05：Gaussian validity audit

Stage 05 对 Stage 04 artifact 做 E1–E4 raw covariance audit，定义哪些用户可以进入 Stage 08，以及每个 ASIN 是否具备足够的 valid-user coverage。它不是另一次 Gaussian fitting。

完成标志：

```text
result/05_gaussian_audit/raw_cov_validity.json
result/05_gaussian_audit/asin_coverage_valid_ge2.json
```

必须交叉检查 `raw_cov_validity.json` 中的 valid-user summary 与 `per_user` 标记一致，并确认 coverage 文件只使用 audit 通过的用户。Stage 08 不得绕过 Stage 05 直接使用 Stage 04 的全部 fitted users。

### Stage 07：本地 Qwen SFT query pool

确认既有 SFT adapter 存在后，使用项目允许的本地 Qwen client 批量生成候选 query。生成结果必须保留商品标识、候选文本及属性覆盖所需字段；不能在业务脚本中新增远程 LLM 调用、逐条生成或传入运行参数。

首次运行按脚本内 `SFT_POOL_SMOKE` 配置验证批量生成、属性覆盖和输出 schema，再切换 full。完成标志：

```text
result/07_gen_query/pool_queries.json
```

候选 pool 必须与当前 Stage 04/05 cohort 和商品属性来源一致；旧 pool 若 signature 或来源不一致，必须重新生成。

### Stage 08：Mahalanobis query 选择

Stage 08 必须同时读取：

```text
result/04_gaussian/user_gaussian_stats.json
result/05_gaussian_audit/raw_cov_validity.json
result/05_gaussian_audit/asin_coverage_valid_ge2.json
result/07_gen_query/pool_queries.json
```

对候选 query 使用当前冻结 encoder 编码，在 target 用户 Gaussian 内部通过 gate，并对同一 ASIN 的 competitor 用户执行 exclusive gate；选择逻辑使用固定 Stage 04 full-fit 参数，不在运行时 bootstrap 重拟合。最终 selection 必须保留 ASIN、用户、query 及必要的距离/属性字段，并只输出符合 cohort 约束的 association。

首次运行按脚本内 smoke 配置检查 vectorized/scalar 距离一致性、audit 过滤和输出 schema，再切换 full。完成标志：

```text
result/08_select_query/selected_queries.json
```

运行后检查 selection 中的每个 ASIN/用户都存在于 Stage 04 cohort，且 selection summary、users/selections 结构和候选来源一致。后续 Stage 09、Stage 10、Stage 11 都必须读取同一份 selection。

### Stage 09：GECToR 与 SErCL 用户错误画像

Stage 09 从 Stage 08 selection 提取用户 cohort，再从 Stage 02 清洗后的 `uid_to_sentences.json` 取满足脚本最低句数要求的句子；通过独立 GECToR 子进程批量纠正，使用 ERRANT 对齐原句/纠正句，并用 spaCy UD 信息提取 error type、D3 context、`P_u(e|r)` 与 `L_u(r,e)`。

完成标志：

```text
result/09_sercl_user_profile/user_sercl_profile.json
result/09_sercl_user_profile/cohort_summary.json
```

必须检查 profile 的用户来自当前 Stage 08 selection 且句子来自当前 Stage 02 artifact。原始 JSON 保存的是 `user_profiles` 和 `user_word_edits`；`transformation_history` 是 Stage 10 的 `load_sercl_profile()` 根据词级 edits 动态构建的模型，不能只用 raw JSON 顶层字段搜索来判断其是否为空。GECToR 子进程返回码、输出行数和 index 缺失都必须直接报错。

### Stage 10：selected association 的 typo 注入

Stage 10 读取：

```text
result/09_sercl_user_profile/user_sercl_profile.json
result/04_gaussian/user_gaussian_stats.json
result/08_select_query/selected_queries.json
```

对所有当前 selection 中且有对应用户 profile/Gaussian/cohort gate 的 association 做一次候选注入尝试，不使用旧的每用户 query 截断规则。流程为：

```text
query tokenization + syntax context
  → 按用户上下文 char-level error rate 选择候选 token
  → 优先查找 transformation_history 中的 user-historical typo
  → 无匹配时记录 generic_fallback 来源并按用户机制分布生成候选
  → minimality / edit-distance / length / meaningful-word 检查
  → target 用户 Gaussian Q gate
  → competitor-exclusive cohort gate
  → MiniLM semantic preservation gate
  → 只写入通过全部条件的 char-level typo pair
```

当前输出不包含 case/apostrophe 等 surface-form typo；每个成功记录必须带有 original/typo query、用户与 ASIN、原词/错词、mechanism、`transformation_source`、编辑距离、语义分数和必要上下文。`MAX_EDIT_DISTANCE`、语义阈值、机制集合和 gate quantile 以脚本内配置及输出 `config` 为准，技能不固化某次运行的数量。

完成标志：

```text
result/10_typo_injection/typo_injection_results.json
result/10_typo_injection/cohort_summary.json
```

必须检查 `cohort_summary.json` 中的 attempted、written、gate failure 与 `transformation_source_counts`，重点区分 `user_historical_*` 和 `generic_fallback`；不能把候选尝试数误报为成功注入数。Stage 12 只允许读取该次运行的成功 `results`。

### Stage 11：独立的七类 retriever 评估

Stage 11 必须通过 standalone driver 运行，而不是只 import unified 模块：

```bash
nohup $PY 11_syntactic_evaluation/run_stage11_eval.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage11_retrieval.log 2>&1 &
```

`run_stage11_eval.py` 调用 `syntax_subspace_retrieval_unified.main()`，并在返回后检查 canonical 文件确实写出。评估覆盖当前 registry 中的 BM25、SPLADE、MiniLM、MPNet、BGE、GTE、ColBERTv2；sim09 query-query reference 使用 canonical MiniLM embedding。可复用 corpus/query embedding cache，但 selection signature、corpus signature、query shape 不一致时必须重建。

完成标志必须全部存在于新目录：

```text
result/11_syntactic_evaluation/per_query.json
result/11_syntactic_evaluation/retrieval_summary.json
result/11_syntactic_evaluation/volatility.json
```

`per_query.json` 必须记录当前 selection signature、corpus 信息、retriever registry 和所有 query record；summary/volatility 必须来自同一次运行。历史遗留的旧评估目录不再是 canonical 目录，不得写入、读取或用其文件替代 Stage 11 结果。Stage 11 即使命中 cache，也必须检查 canonical 文件存在、signature 匹配且 schema 完整。

### Stage 12：paired typo retrieval degradation

Stage 12 从当前 Stage 10 的 `typo_injection_results.json` 读取所有成功 pair，把每个 pair 的 original query 与 typo query 组成同一批次，使用与 Stage 11 相同的 retriever registry、corpus 和 cache 逻辑做 paired retrieval。它只报告：

```text
original_hit@k - typo_hit@k
```

正值表示 typo 使目标召回下降；不计算 flip rate，也不把旧 Stage 11 结果或其他输入替换为当前 Stage 10 结果。运行前检查 Stage 10 结果 schema，并将 Stage 12 的 `config.n_pairs` 与 Stage 10 实际写入的成功结果数交叉校验；任何不一致都直接停止。没有成功 pair 时，写出明确的空结果并终止，不改用历史 pair。

完成标志：

```text
result/12_typo_evaluation/per_query.json
result/12_typo_evaluation/retrieval_degradation.json
```

`retrieval_degradation.json` 必须记录 `n_pairs`、corpus/query signature、每个 retriever 的 original/typo hit@k、degradation 及 neutral/help/hurt 计数；`per_query.json` 必须能回溯到 uid、ASIN、original/typo query 和每个 retriever 的 paired rank/hit。

## 最终 canonical 结果

```text
result/01_attribute_extraction/product_attributes.json
result/02_user_review_sentence_extract/uid_to_sentences.json
result/02_user_review_sentence_extract/uid_to_sentences.pkl
result/02_user_review_sentence_extract/asin_to_users.json
result/03_spacy_encode/syntax_pcfg_strict_attr.json
result/04_gaussian/user_gaussian_stats.json
result/05_gaussian_audit/raw_cov_validity.json
result/05_gaussian_audit/asin_coverage_valid_ge2.json
result/07_gen_query/pool_queries.json
result/08_select_query/selected_queries.json
result/09_sercl_user_profile/user_sercl_profile.json
result/09_sercl_user_profile/cohort_summary.json
result/10_typo_injection/typo_injection_results.json
result/10_typo_injection/cohort_summary.json
result/11_syntactic_evaluation/per_query.json
result/11_syntactic_evaluation/retrieval_summary.json
result/11_syntactic_evaluation/volatility.json
result/12_typo_evaluation/per_query.json
result/12_typo_evaluation/retrieval_degradation.json
```

## 快速检查

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
$PY -c 'import json; p=json.load(open("result/02_user_review_sentence_extract/uid_to_sentences.json")); print(len(p), next(iter(p.values()))[:1])'
$PY -c 'import json; p=json.load(open("result/08_select_query/selected_queries.json")); print(p["summary"])'
$PY -c 'import json; p=json.load(open("result/10_typo_injection/cohort_summary.json")); print(p["totals"]); print(p["totals"]["transformation_source_counts"])'
$PY -c 'import json; p=json.load(open("result/11_syntactic_evaluation/per_query.json")); print(p["config"], p["n_queries"])'
$PY -c 'import json; p=json.load(open("result/12_typo_evaluation/retrieval_degradation.json")); print(p["config"])'
```

快速检查只用于读取当前 artifact，不用于推断固定基线。运行中的进程使用 `tail`、`grep`、`ps` 和 `nvidia-smi` 即时检查；禁止用 `sleep` 等待。

## 故障排查

| 症状 | 处理 |
|------|------|
| Stage 02 仍有 HTML artifact | 检查 entity decode、tag strip、space collapse 是否发生在句子切分前；确认输出来自当前 Stage 02 run |
| Stage 02 mapping 引用不存在用户 | 用过滤后的 `uid_to_sentences` 重建 `asin_to_users`，不要沿用旧 mapping |
| Stage 03 cache cohort mismatch | 比较 `meta.json`、`uid_list.json`、`user_n_sents.json`、embedding shape 和 Stage 02 用户集合；不一致就删除对应 cache 后重建 |
| Stage 04 Cholesky/solve 或 D² 异常 | 检查 Stage 03 embedding 来源、raw full covariance、有限值和 gate quantile；不要切换到未批准的正则化变体掩盖问题 |
| Stage 05 valid/coverage 不一致 | 对照 audit 的 summary、per-user 标志和 ASIN coverage；Stage 08 必须等待 audit 完成 |
| Stage 07 生成失败 | 检查既有 SFT adapter、本地 Qwen model cache、GPU 显存和批量生成日志 |
| Stage 08 selection 异常 | 检查 Stage 04/05/pool 是否来自同一 cohort，确认 target/competitor gate 和 candidate encoding cache 的 signature |
| Stage 09 profile 看起来没有 transformation history | 不要在 raw 顶层 JSON 搜索该字段；通过 `load_sercl_profile()` 从 `user_word_edits` 动态构建后再检查 |
| Stage 09 GECToR 子进程失败 | 检查指定 `gector_env` Python、模型 cache、IPC JSONL 和子进程返回码；缺行或错位直接重跑 |
| Stage 10 全部拒绝 | 分别检查 profile 用户交集、char-level history、minimality、Gaussian Q gate、competitor-exclusive gate 和 MiniLM gate |
| Stage 10 source 全是 generic_fallback | 先确认 profile 是由当前 Stage 02 清洗文本重建，再检查 query/review vocabulary 是否有历史 typo 可匹配；不要把候选尝试数当作 historical 命中 |
| Stage 11 canonical 文件缺失 | 运行 `run_stage11_eval.py`，不要只 import `syntax_subspace_retrieval_unified.py`；检查当前 canonical 结果目录而不是历史遗留目录 |
| Stage 11/12 cache signature mismatch | 保留 mismatch 日志，按当前 selection/corpus/query signature 重编码；不得静默复用旧 embedding |
| Stage 12 pair 数不一致 | 比较 Stage 10 成功 `results` 长度、Stage 12 `config.n_pairs` 和 paired records；不一致直接停止 |
| Stage 12 没有 pair | 检查 Stage 10 `n_injected_written`；若确实为零，保留明确空结果，不替换为历史结果 |

## 规则遵守

- 所有阶段按依赖顺序运行；Stage 11 必须通过 `run_stage11_eval.py` 独立产出 canonical 文件后，才能视为 Stage 11 完成。
- 所有运行参数硬编码；固定 `pq_env`，GECToR 固定使用指定 `gector_env`。
- 所有生成、embedding 和解码使用批量路径；不得恢复逐条 LLM 生成或逐元素大矩阵计算。
- 长任务使用 `nohup ... &`；不使用 `sbatch`、`srun` 或 `sleep` 轮询。
- 日志/cache/中间文件写入 `hj82_scratch2`；`result/` 只保存结果 artifact。
- 每个新阶段或修改阶段先 smoke，再 full；修改 Python 后先运行 `python -m py_compile`。
- 所有 LLM 调用只使用项目允许的本地 Qwen backend，不新增远程客户端。
- 必需文件缺失、schema 不一致、用户/ASIN 关联不一致或 signature 不一致时直接 raise。
- 本技能不执行 commit 或 push，除非用户另行明确要求。
