---
name: syntax-subspace-pipeline
description: 按当前主链路运行 PersonalQuery 语法子空间流水线：Stage 01–05 → Stage 07–12；Stage 05 E1–E4 审计定义 Stage 08 的有效 Gaussian 用户与 ASIN cohort。
---

# PersonalQuery Syntax-Subspace Pipeline

## 何时用

当需要从 Amazon Baby Products 评论重新生成用户句子、监督式 32d syntax embedding、per-user Gaussian、个性化 query 选择、用户错误画像、typo 注入以及 7-retriever 评估时，使用本技能。当前主链路包含 Stage 01、02、03、04、05、07、08、09、10、11、12，共 11 个实际运行阶段；Stage 06 是 Stage 07 所需的既有 SFT 训练产物目录，不在本技能中重新训练。

**当前数据契约**：
- Stage 03 生成 adaptive supervised `_SupEncoder(21737→256→32)` 和 profile/val/test 三段 32d embedding；规则特征使用 `count/(1+count)` 归一化，用户至少 80 句，并按 SHA1 hash 做 50/20/30 split。
- Stage 04 直接使用 raw full covariance（无 `λI` ridge），保留 `MIN_EIGEN_RATIO=1e-8`、Cholesky/solve 数值检查，拟合 per-user Gaussian 并生成 Q_0.95 cohort gate；这是 Gaussian 参数的唯一生产阶段。
- Stage 05 对 Stage 04 的 embedding 结果执行 E1–E4 原始 covariance audit，生成 valid Gaussian 用户清单和 ASIN coverage；Stage 08 必须读取并校验这两个产物。
- Stage 08 在 Stage 05 的有效用户/cohort 上，以固定 Stage 04 full-covariance fit 执行 `single_fit_unique`：Mahalanobis target-core + competitor-exclusive 选择；当前主路径不启用 bootstrap 重拟合。
- Stage 10 对 Stage 08 的全部 selected `(uid, asin, query)` association 运行按上下文错误率排序的 single-shot typo injection，不再按用户截断 query 数量，也不执行真正的随机 Bernoulli draw。
- Stage 11 和 Stage 12 都使用相同的 7 个 retriever；Stage 12 比较每个 typo pair 的 original 与 typo hit@k，不计算 flip rate。

## 输入与环境

**原始输入**：
- `data/Baby_Products_2023.jsonl`
- `data/meta_Baby_Products_2023.jsonl`

**运行约束**：
- cwd 必须为 `/home/wlia0047/ar57/wenyu/PersoanlQuery`。
- Python 必须为 `/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`。
- 长任务必须用 `nohup ... &`，日志写入 `/home/wlia0047/hj82_scratch2/wenyu/logs/`；禁止 `sbatch`、`srun` 和 `sleep` 轮询。
- 运行参数全部硬编码在脚本内，不传命令行参数。
- 首次运行每个新阶段必须先使用脚本内的最小 smoke 配置，确认数据加载、模型/编码、后处理和输出 schema 后，再切换为 `SMOKE=False` 或对应 full 配置。
- 中间 cache 写入 `/home/wlia0047/hj82_scratch2/wenyu/`；最终 JSON 结果写入项目 `result/`。
- 所有需要 LLM 的业务逻辑遵守项目 `llm_client.py` 与本地 Qwen 后端约束；不得新增远程 LLM 调用。

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python
```

## 目录与主脚本

```text
01_attribute_extraction/          extract_product_attrs.py
02_user_review_sentence_extract/  extract_user_sentences.py
03_spacy_encode/                   syntax_pcfg_pipeline.py
04_gaussian/                       fit_per_user_gaussian.py
05_gaussian_audit/                 raw_cov_validity.py
06_training_model/                 sft_lora/（Stage 07 的既有训练产物，不在本技能中运行训练）
07_gen_query/                      sft_pool_generate.py
08_select_query/                  syntax_select_mahalanobis_gate.py
09_sercl_user_profile/             sercl_user_profile.py
10_typo_injection/                 typo_inject.py
11_syntactic_evaluation/           syntax_subspace_retrieval_unified.py
12_typo_evaluation/                typo_retrieval_eval.py
```

**共享 cache**：`/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/` 保存 `adaptive_encoder.pt`、`adaptive_embeddings.npz`、`counts.npz`、`vocab.json`、`uid_list.json` 和 `user_n_sents.json`。Stage 08 使用冻结 encoder 编码候选 pool，Stage 11/12 使用带 corpus/query signature 的 retriever cache。

## 关键配置与 gate

| 阶段 | 关键配置 | 当前含义 |
|------|----------|----------|
| 03 | `MODE="adaptive"`, `z_dim=32`, `MIN_SENTS_PER_USER=80` | 21737 维 PCFG counts（`count/(1+count)`）→ 256 → 32d supervised encoder；SHA1 50/20/30 split |
| 04 | `LAMBDA=1e-3`, `MIN_EIGEN_RATIO=1e-8`, `GATE_QUANTILE=0.95` | 稳定 full covariance 与 per-user Q_0.95 gate |
| 05 | `K_DIM=32`, E1–E4 | E1 样本量、E2 留出 log-P、E3 full rank、E4 bootstrap 稳定性；只用于定义 Stage 08 有效 cohort |
| 08 | `FILTER_Q=0.95`, `USE_SINGLE_FIT=True`, `SEMANTIC_SIM_THRESHOLD=0.8` | 固定 full-fit target 核心与 competitor-exclusive；最终 block 再做 query 相似度约束，不走 bootstrap |
| 10 | `D2_THRESHOLD_QUANTILE="q95"`, semantic gate `0.9` | 最高上下文错误率 token → minimality → Mahalanobis → exclusive cohort → MiniLM；不是随机 Bernoulli |
| 11 | 7 retrievers，sim09 reference=`minilm` | BM25、SPLADE、MiniLM、MPNet、BGE、GTE、ColBERTv2 |
| 12 | `KS=(1,5,10)` | paired `original_hit@k - typo_hit@k`，正值表示退化 |

Stage 05 全量当前基线为 `431/5000` 个 valid Gaussian 用户，`asin_coverage_valid_ge2.json` 包含 `7862` 个至少有 2 个 valid 用户的 ASIN。Stage 08 的原始成功 user-ASIN task 为 `4010`，最终输出聚合后为 `467` 个 ASIN block、`959` 条 `(uid, asin, query)` association；后续 Stage 09/10 都以这 959 条 association 为输入。

## 运行顺序

必须按以下顺序运行；Stage 05 完成后才能运行依赖其审计结果的 Stage 08。Stage 07 之前必须确认 `result/06_training_model/sft_lora` 已存在。

```text
Stage 01 → 01_attribute_extraction/extract_product_attrs.py
Stage 02 → 02_user_review_sentence_extract/extract_user_sentences.py
Stage 03 → 03_spacy_encode/syntax_pcfg_pipeline.py
Stage 04 → 04_gaussian/fit_per_user_gaussian.py
Stage 05 → 05_gaussian_audit/raw_cov_validity.py
Stage 07 → 07_gen_query/sft_pool_generate.py
Stage 08 → 08_select_query/syntax_select_mahalanobis_gate.py
Stage 09 → 09_sercl_user_profile/sercl_user_profile.py
Stage 10 → 10_typo_injection/typo_inject.py
Stage 11 → 11_syntactic_evaluation/syntax_subspace_retrieval_unified.py
Stage 12 → 12_typo_evaluation/typo_retrieval_eval.py
```

## 分阶段命令与完成标志

### Stage 01：商品属性抽取

```bash
nohup $PY 01_attribute_extraction/extract_product_attrs.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage01_attrs.log 2>&1 &
```

完成标志：`result/01_attribute_extraction/product_attributes.json` 存在且包含后续选择阶段需要的商品属性。

### Stage 02：用户句子与 ASIN mapping

```bash
nohup $PY 02_user_review_sentence_extract/extract_user_sentences.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage02_sents.log 2>&1 &
```

完成标志：`result/02_user_review_sentence_extract/uid_to_sentences.pkl`、对应 JSON 以及 `asin_to_users.json` 存在。

### Stage 03：PCFG 规则与 32d supervised encoder

```bash
nohup $PY 03_spacy_encode/syntax_pcfg_pipeline.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage03_pcfg.log 2>&1 &
```

完成标志：`pcfg_cache/adaptive_encoder.pt`、`pcfg_cache/adaptive_embeddings.npz` 和 `pcfg_cache/vocab.json` 存在；embedding 至少包含 `z_profile`、`z_val`、`z_test` 及对应 index，并写出 `result/03_spacy_encode/syntax_pcfg_adaptive.json`。

### Stage 04：per-user Gaussian 与 cohort gate

```bash
nohup $PY 04_gaussian/fit_per_user_gaussian.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage04_gaussian.log 2>&1 &
```

首次运行先将脚本内 `SMOKE=True`，确认 schema 后改为 `SMOKE=False` 全量运行。完成标志：`result/04_gaussian/user_gaussian_stats.json` 存在，并包含 `config`、`users`、`cohort_gates`；用户记录含 32d `mu`、32×32 `sigma_inv`、`d2_q50`、`d2_q75`、`d2_q95`。

### Stage 05：E1–E4 raw covariance audit

```bash
nohup $PY 05_gaussian_audit/raw_cov_validity.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage05_gaussian_audit.log 2>&1 &
```

完成标志：同时生成：
- `result/05_gaussian_audit/raw_cov_validity.json`
- `result/05_gaussian_audit/asin_coverage_valid_ge2.json`

必须确认 `raw_cov_validity.json.summary.n_valid` 与 `per_user[*].valid_gaussian=True` 数量一致，并确认 coverage 的 `min_users_per_asin=2`。当前全量基线为 431 valid users、7862 ASINs；Stage 08 不得绕过这些审计产物直接使用 Stage 04 的全部 fitted users。

### Stage 07：Qwen SFT pool 生成

确认 `result/06_training_model/sft_lora` 已存在后运行：

```bash
nohup $PY 07_gen_query/sft_pool_generate.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage07_pool.log 2>&1 &
```

首次运行使用脚本内 `SFT_POOL_SMOKE=True`（5 ASIN）验证批量生成、属性覆盖和输出 schema，再恢复 full 配置。当前生成器使用本地 `Qwen/Qwen2.5-0.5B-Instruct` + `sft_lora`，每轮 `25` 条、共 `2` 轮合计最多 `50` candidates/ASIN，`max_new_tokens=40`、`max_num_seqs=1024`、`gpu_memory_utilization=0.85`。完成标志：`result/07_gen_query/pool_queries.json` 存在，且候选来自 Stage 04 fitted cohort、具有所需属性。

### Stage 08：Stage 05-filtered Mahalanobis 选择

```bash
nohup $PY 08_select_query/syntax_select_mahalanobis_gate.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage08_select.log 2>&1 &
```

Stage 08 必须同时读取：
- `result/04_gaussian/user_gaussian_stats.json`
- `result/05_gaussian_audit/raw_cov_validity.json`
- `result/05_gaussian_audit/asin_coverage_valid_ge2.json`
- `result/07_gen_query/pool_queries.json`

首次运行使用脚本内 `SMOKE=True`（最少 5 个 ASIN）检查 vectorized/scalar D² 一致性和输出 schema，再恢复 full 配置。选择条件为 target `D²≤target.d2_q95` 且对所有 competitor `D²>competitor.d2_q95`；最终只保留至少 2 个 unique users 的 ASIN block。完成标志：`result/08_select_query/selected_queries.json` 存在；当前 full 结果为 4010 个原始 selected user-ASIN tasks，467 个最终 ASIN blocks，959 条 association。

### Stage 09：GECToR 用户错误画像

```bash
nohup $PY 09_sercl_user_profile/sercl_user_profile.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage09_sercl.log 2>&1 &
```

完成标志：`result/09_sercl_user_profile/user_sercl_profile.json` 和 `cohort_summary.json` 存在；画像只覆盖 Stage 08 最终 selection 中出现且至少有 30 句评论的用户，每用户最多处理 50 句，并记录上下文错误率与 mechanism histogram。当前 full 基线为 326 users、16300 sentences、11784 edits、9761 no-edit sentences。

### Stage 10：全量 selected query typo 注入

```bash
nohup $PY 10_typo_injection/typo_inject.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage10_typo.log 2>&1 &
```

首次运行使用脚本内 `SMOKE=True`（按 `N_SMOKE_USERS` 选少量用户）确认四类 gate 和输出 schema，再恢复 `SMOKE=False`。输入为 Stage 09 profile、Stage 04 Gaussian/cohort artifact 和 Stage 08 selection；不再使用 `MAX_QUERIES_PER_USER` 或任何每用户 3 条限制。当前实现先按最高上下文 `P_u(char_level_error|context)` 选择一个 token，再执行 minimality、Mahalanobis Q_0.95、exclusive cohort、MiniLM cosine≥0.9 四个 gate；它不是随机 Bernoulli 抽样，且无 char-level history 时直接不注入。支持的 char-level mechanism 为 `keyboard_adjacent`、`letter_swap`、`letter_repetition`、`letter_insertion`、`letter_deletion`；成功记录保留 uid、asin、原/错 query、token、mechanism、context、距离和语义分数等字段。当前 full 结果为尝试 959 条、写入 76 条 typo pairs、涉及 54 个用户。完成标志：`result/10_typo_injection/typo_injection_results.json` 与 `cohort_summary.json` 存在。

### Stage 11：7-retriever syntactic evaluation

```bash
nohup $PY 11_syntactic_evaluation/syntax_subspace_retrieval_unified.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage11_retrieval.log 2>&1 &
```

首次运行使用 `SMOKE=True`、`N_SMOKE_QUERIES=5`，确认 selection signature、cache signature 和 7 retriever 均能完成，再恢复 full 配置。当前 full 口径为 959 queries、217722 corpus ASIN、467 个有效 ASIN blocks；sim09 的 canonical query-query reference 为 MiniLM 384d。完成标志：
- `result/syntactic_evaluation/retrieval_summary.json`
- `result/syntactic_evaluation/volatility.json`
- `/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_retrieval_per_query.json`

`asin_to_doc.json` 及其 signature 是 Stage 11 的 cache，位于 `result/11_syntactic_evaluation/`，不是 retrieval summary 的 canonical 结果目录。

### Stage 12：paired typo hit@k degradation

```bash
nohup $PY 12_typo_evaluation/typo_retrieval_eval.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage12_typo_eval.log 2>&1 &
```

首次运行使用 `SMOKE=True`、`N_SMOKE_PAIRS=5`（10 combined queries）验证 pair 对齐和 7 retriever，再恢复 full 配置。Stage 12 从 Stage 10 读取全部成功注入 pairs；当前 full 为 76 pairs、152 combined queries、217722 corpus ASIN。只报告 `original_hit@1/5/10 - typo_hit@1/5/10`，不计算 flip rate；没有 typo pair 时写出 `result/12_typo_evaluation/typo_paired_summary.json` 的明确空结果并终止，不替换为其他输入。

完成标志：
- `result/12_typo_evaluation/per_query.json`
- `result/12_typo_evaluation/retrieval_degradation.json`

## 最终结果文件

```text
result/01_attribute_extraction/product_attributes.json
result/02_user_review_sentence_extract/uid_to_sentences.pkl
result/02_user_review_sentence_extract/uid_to_sentences.json
result/02_user_review_sentence_extract/asin_to_users.json
result/04_gaussian/user_gaussian_stats.json
result/05_gaussian_audit/raw_cov_validity.json
result/05_gaussian_audit/asin_coverage_valid_ge2.json
result/07_gen_query/pool_queries.json
result/08_select_query/selected_queries.json
result/09_sercl_user_profile/user_sercl_profile.json
result/09_sercl_user_profile/cohort_summary.json
result/10_typo_injection/typo_injection_results.json
result/10_typo_injection/cohort_summary.json
result/syntactic_evaluation/retrieval_summary.json
result/syntactic_evaluation/volatility.json
result/12_typo_evaluation/per_query.json
result/12_typo_evaluation/retrieval_degradation.json
```

## 快速检查

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
$PY -c 'import json; p=json.load(open("result/05_gaussian_audit/raw_cov_validity.json")); print(p["summary"]["n_valid"])'
$PY -c 'import json; p=json.load(open("result/08_select_query/selected_queries.json")); print(p["summary"])'
$PY -c 'import json; p=json.load(open("result/10_typo_injection/typo_injection_results.json")); print(len(p["results"]))'
$PY -c 'import json; p=json.load(open("result/12_typo_evaluation/retrieval_degradation.json")); print(p["config"]["n_pairs"])'
```

若 cache signature 与当前 selection 或 corpus 不一致，必须让对应脚本重新编码/重建；不得静默复用旧 query 或 corpus cache。运行过程中使用 `tail`、`grep`、`ps`、`nvidia-smi` 即时检查，不使用 `sleep`。

## 故障排查

| 症状 | 检查 |
|------|------|
| Stage 04 出现负 D² 或 Cholesky 失败 | 检查 `LAMBDA=1e-3`、`MIN_EIGEN_RATIO=1e-8` 和 adaptive 32d cache 是否来自同一轮 Stage 03 |
| Stage 05 valid users 不是预期数量 | 检查 E1–E4 每项失败计数、`n_valid` 与 `per_user` 一致性，不要用 Stage 04 fitted user 总数代替 |
| Stage 08 缺少 audit/coverage | 先完整运行 Stage 05，并检查三个 Stage 04/05 JSON 的 schema 和 gate quantile |
| Stage 08 selection 数量异常 | 检查 pool 是否对应当前 Stage 04/05 cohort、`selected_queries.json` 的 summary，以及 candidate query 编码 cache |
| Stage 07 生成失败 | 检查 `result/06_training_model/sft_lora`、本地 Qwen 模型 cache、GPU 显存和批量生成日志 |
| Stage 09 GECToR 子进程失败 | 检查 `/home/wlia0047/hj82_scratch2/wenyu/venvs/gector_env/bin/python` 及其模型 cache |
| Stage 10 全部 gate reject | 检查 Stage 09 是否有当前 326 个 selection users 的 profile、Stage 04 Q_0.95 和 exclusive cohort 是否一致 |
| Stage 11/12 cache signature mismatch | 保留 mismatch 日志并让脚本按当前 signature 重编码，不要复用旧 query cache |
| Stage 12 为 0 pairs | 检查 Stage 10 的 `n_injected_written`；若确实为 0，保留明确的空结果，不改用旧结果 |

## 规则遵守

- 参数硬编码，不传命令行参数。
- Python 解释器固定为 `pq_env`；GECToR 子进程使用指定 `gector_env`。
- 长任务使用 `nohup ... &`，禁止 `sbatch`、`srun` 和 `sleep` 轮询。
- 运行 cwd 固定为项目根；日志、cache 和中间产物写入 `hj82_scratch2`。
- 可运行脚本位于对应功能目录；`result/` 仅存放结果文件。
- 每个新阶段先 smoke，再 full；变更 Python 脚本后先运行 `python -m py_compile`。
- 所有 LLM 调用只使用项目允许的本地 Qwen 后端；不得新增远程客户端或业务侧推理链路。
- 外部文件缺失、schema 不一致、signature 不一致或必需统计不一致时直接报错，不用静默默认值掩盖问题。

完成后在终端输出阶段状态、关键统计和结果路径；本技能本身不执行 commit 或 push，除非用户另行明确要求。
