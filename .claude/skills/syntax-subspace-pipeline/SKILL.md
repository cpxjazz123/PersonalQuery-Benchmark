---
name: syntax-subspace-pipeline
description: 按当前主链路运行 PersonalQuery 语法子空间、查询选择、用户错误画像、typo 注入与 paired retrieval 评估；canonical 路径为 trainable svdmlp Gaussian (SVD→MLP 64d + per-user 可训练 μ/σ² + logp_delta=2.0 per-ASIN gate)。所有阶段以当前产物 schema 和 selection signature 为准，不固化易变的数量。
---

# PersonalQuery Syntax-Subspace Pipeline (svdmlp canonical)

## 何时使用

当需要从 Amazon Baby Products 原始评论重新生成用户句子、训练监督式语法表示（含可选 trainable SVD+MLP 用户风格向量）、拟合 per-user Gaussian（canonical: trainable 对角 σ；legacy: analytical full Σ）、生成并选择个性化 query（canonical: svdmlp logp_delta=2.0；legacy: full Σ + cohort gate_q=0.05）、构建用户错误画像、注入 typo，并评估 typo 对 retrieval 的影响时，使用本技能。

主链路：Stage 01、02、03（含可选 03b）、04（fit + 04b trainable）、05、06（既有 SFT 产物）、07、08（canonical svdmlp；legacy fit path）、09、10、11、12。canonical 产物路径以 `selected_queries_svdmlp.json` (Stage 08 svdmlp default) 为准；legacy baseline 产物（`selected_queries.json`）保留作为对照。Stage 11 必须基于 canonical selection 跑。

技能只描述流程、数据契约和 stage 之间的 schema/signature 对齐；不写入某次运行的用户数、query 数、pair 数、命中率或耗时。每次运行的实际统计必须从对应结果文件的 `config`、`summary`、`totals`、`per_retriever`、`stability_flip` 字段读取。

## 总体流程

```text
原始 review JSONL + meta JSONL
  ↓
Stage 01：商品属性抽取
  ↓
Stage 02：HTML 清洗、句子切分、用户聚合、ASIN→用户映射
  ↓
Stage 03a (canonical)：spaCy/PCFG 规则特征 + supervised syntax encoder + frozen strict3 embeddings
Stage 03b (canonical 前置)：SVD(20000→256) + MLP(256→64) contrastive → svd_mlp_encoder.pt + svd_components.npz
  ↓
Stage 04a (legacy)  ：Numba JIT 拟合 per-user raw full-covariance Gaussian + ASIN cohort gate
                       + main 末尾自动 post-process theoretical χ² gate 字段
Stage 04b (canonical)：PyTorch 训练 per-user trainable 对角 σ Gaussian (NLL + KL(N(0,I)))
                        source = {raw | svd_mlp}，svd_mlp 为 canonical
  ↓
Stage 05：raw covariance validity audit (E1–E4)，产出 valid user + ASIN coverage (legacy gate)
  ↓
Stage 06：既有 SFT adapter（不重新训练）
  ↓
Stage 07：本地 Qwen + vLLM 批量生成 query pool
  ↓
Stage 08 (canonical)：基于 trainable svd_mlp Gaussian 的 logp_delta per-ASIN selection
                     logp_delta = 2.0（至少 1 query 保留）
Stage 08 (legacy)  ：基于 fit Gaussian + Stage 05 cohort gate 的 maha-D² selection (gate_q=0.05)
  ↓
Stage 09：GECToR/ERRANT/SErCL 用户错误画像
  ↓
Stage 10：单次 typo 候选注入 + 多 gate (target Gaussian + competitor-exclusive + MiniLM semantic)
  ↓
Stage 11：7 retriever 评估 (BM25/SPLADE/MiniLM/MPNet/BGE/GTE/ColBERTv2)
         + query-level stability_flip (canonical: svdmlp selection)
  ↓
Stage 12：paired original/typo retrieval degradation
```

canonical selection 是 Stage 04b (trainable svd_mlp) → Stage 08 (svdmlp logp_delta) → Stage 11 (基于 svdmlp selection 跑 7 retriever)。legacy path (Stage 04a → Stage 05 → Stage 08 maha gate_q=0.05) 保留作为对照。

## 输入、环境与硬性规则

**原始输入**：
- `data/Baby_Products_2023.jsonl`
- `data/meta_Baby_Products_2023.jsonl`

**运行环境**：
- cwd 固定为 `/home/wlia0047/ar57/wenyu/PersoanlQuery`（filesystem alias `/fs04/ar57/wenyu/PersoanlQuery` 在部分环境下不可解析，所有 nohup/python 走 `/home/...` 路径）。
- Python 固定为 `/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`（不要用 system python、conda base 或其他 venv；transformers 4.40.2 + torch 2.13.0+cu130 是必需基线）。
- GECToR 仅由 Stage 09 通过 `gector_subprocess.py` 子进程在独立 `gector_env` 中运行；主进程 `pq_env`。
- 需要 LLM 的业务（Stage 07 vLLM）只使用 `pq_env` 内本地 Qwen；不得新增远程 LLM 或业务侧本地推理链路。
- 所有运行参数写在脚本内（Rule 3：硬编码到模块级常量或脚本内默认），不向脚本传命令行参数。
- 长任务用 `nohup ... &` 或 `setsid ... < /dev/null &`；日志写入 `/home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/`（已存在）或 `/home/wlia0047/hj82_scratch2/wenyu/logs/`；禁止 `sbatch` / `srun` / `sleep` 轮询。
- cache、日志、运行时中间文件写入 `/home/wlia0047/hj82_scratch2/wenyu/`；最终结果写入项目 `result/`。
- 新阶段或修改阶段先按脚本内 smoke 配置验证数据加载、模型/编码、后处理和输出 schema，再切换 full；批量生成、批量 embedding 和向量化检索必须保持启用。
- 缺失的必需文件、字段、schema、cohort 对齐关系或 signature 必须直接 raise；不得静默使用旧输入、默认值或替代路径。

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python
```

## 目录与主脚本

每个功能目录只有一个主入口（Rule 3）；同目录内多脚本已被合并或删除。

```text
01_attribute_extraction/          extract_product_attrs.py
02_user_review_sentence_extract/  extract_user_sentences.py
03_spacy_encode/                  syntax_pcfg_pipeline.py     # 含 Stage 03a (strict3) + Stage 03b (svd_mlp) 双路径
04_gaussian/                      fit_per_user_gaussian.py     # Stage 04a: full Σ analytical + theo gate post-process
04_gaussian/                      trainable_per_user_gaussian.py  # Stage 04b: trainable 对角 σ (raw / svd_mlp)
05_gaussian_audit/                raw_cov_validity.py
06_training_model/                sft_pipeline.py              # 既有 SFT adapter，不在本技能中训练
07_gen_query/                     sft_pool_generate.py
08_select_query/                  syntax_select_mahalanobis_gate.py
                                    # 默认 canonical: Stage 04a full Σ + cohort gate (legacy)
                                    # TG_GAUSSIAN_SOURCE=trainable_svdmlp → svd_mlp logp_delta path
09_sercl_user_profile/            sercl_user_profile.py        # 主 driver
09_sercl_user_profile/            gector_subprocess.py         # GECToR 子进程入口，必须独立
10_typo_injection/                typo_inject.py               # 含 error_location/injection_sampler/typo_classifier/valid_words 工具 (前缀 _el_/_is_/_tc_/_vw_)
11_syntactic_evaluation/          run_stage11_eval.py          # canonical driver，必须用此跑
11_syntactic_evaluation/          syntax_subspace_retrieval_unified.py  # 7 retriever 实现，被 driver + Stage 12 importlib 动态加载
12_typo_evaluation/               typo_retrieval_eval.py
analysis/                         diag_exclusive_gate_v2.py
```

**共享 cache**：
- `/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/`：Stage 03a sparse rule cache、vocab、用户句子布局、frozen strict3 encoder 和 supervised embeddings；Stage 03b 的 `svd_components.npz`（Vt）+ `svd_mlp_encoder.pt`（StyleMLP）+ `vocab.json`。
- `/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/multiretrieval_embeds/`：Stage 11/12 的 retriever corpus/query embedding cache，按 retriever 分子目录。
- `result/11_syntactic_evaluation/asin_to_doc.json` 及其 `.sig`：Stage 11/12 的 ASIN 文档 cache（corpus sig 一致才能复用）。

cache 复用条件：corpus sig + selection sig + query text sig + shape 全部一致；不一致时必须重建。

## Stage 间的产物签名（关键对齐点）

每个 stage 写出的 JSON 必须带 selection signature 或 corpus signature，**下游必须 fail-fast** 当上游 signature 与自身期望不符：

- **Selection sig** (`syntax_subspace_retrieval_unified._compute_selection_signature`)：SHA1(n_entries + sorted (asin, uid, query))[:16]，下游 Stage 08/11 都用它判断 selection 是否变化。
- **Corpus sig**：SHA1(META_FILE mtime + 行数)[:16]。
- **Query sig**：SHA1(n + joined queries)[:16]，Stage 11/12 每个 retriever 内部 cache 用。
- **Embedding source**：Stage 04b output 的 `config.source ∈ {"raw", "svd_mlp"}`；Stage 08 必须 match。

如果下游产物 signature 与上游不一致，cache 重建而非静默使用旧值。

## 阶段依赖与运行命令

必须按以下顺序运行。Stage 06 是既有 SFT adapter，不重新训练；Stage 07 之前必须确认 `result/06_training_model/sft_lora` 存在；Stage 08 canonical 路径依赖 Stage 04b (svd_mlp) 完成；Stage 11 必须跑在 Stage 08 canonical selection 之上。

```text
Stage 01 → Stage 02 → Stage 03a (+ Stage 03b 如走 svdmlp) → Stage 04b (svd_mlp) → Stage 08 (canonical)
                                                              ↓
Stage 04a (fit, legacy) → Stage 05 → Stage 08 (legacy baseline)   ↓
                                                              ↓
Stage 06 existing artifact → Stage 07 → Stage 09 → Stage 10 → Stage 11 → Stage 12
```

每条命令在项目根目录执行，参数不传命令行：

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python

# Stage 01–02
nohup $PY 01_attribute_extraction/extract_product_attrs.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage01_attrs.log 2>&1 &

nohup $PY 02_user_review_sentence_extract/extract_user_sentences.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage02_sents.log 2>&1 &

# Stage 03a (canonical strict3 encoder)
nohup $PY 03_spacy_encode/syntax_pcfg_pipeline.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage03a_pcfg.log 2>&1 &

# Stage 03b (trainable SVD+MLP svd_mlp encoder，canonical 前置)
TG_STAGE=03b nohup $PY 03_spacy_encode/syntax_pcfg_pipeline.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage03b_svdmlp.log 2>&1 &

# Stage 04a (legacy analytical full Σ) — 末尾自动加 theoretical χ² gate 字段
nohup $PY 04_gaussian/fit_per_user_gaussian.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage04a_fit.log 2>&1 &

# Stage 04b (canonical trainable svd_mlp Gaussian)
nohup $PY 04_gaussian/trainable_per_user_gaussian.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage04b_trainable.log 2>&1 &

# Stage 05 (legacy baseline path 才需要；svdmlp 不依赖)
nohup $PY 05_gaussian_audit/raw_cov_validity.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage05_audit.log 2>&1 &

# Stage 07 (本地 Qwen + vLLM 批量生成)
nohup $PY 07_gen_query/sft_pool_generate.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage07_pool.log 2>&1 &

# Stage 08 (canonical: svd_mlp logp_delta=2.0)
TG_GAUSSIAN_SOURCE=trainable_svdmlp nohup $PY 08_select_query/syntax_select_mahalanobis_gate.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage08_svdmlp.log 2>&1 &

# Stage 08 (legacy: full Σ + cohort gate_q=0.05) — 用于 baseline 对照
nohup $PY 08_select_query/syntax_select_mahalanobis_gate.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage08_legacy.log 2>&1 &

# Stage 09–10
nohup $PY 09_sercl_user_profile/sercl_user_profile.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage09_sercl.log 2>&1 &

nohup $PY 10_typo_injection/typo_inject.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage10_typo.log 2>&1 &

# Stage 11 (canonical: 必须基于 svdmlp selection 跑)
nohup $PY 11_syntactic_evaluation/run_stage11_eval.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage11_retrieval.log 2>&1 &

# Stage 12 (paired retrieval degradation)
nohup $PY 12_typo_evaluation/typo_retrieval_eval.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage12_typo_eval.log 2>&1 &
```

**Stage 11 SEL_IN**：脚本默认 `SEL_IN = result/08_select_query/selected_queries_svdmlp.json`（canonical）。若需跑 legacy baseline flip 率对比，先把 `syntax_subspace_retrieval_unified.py` 的 `SEL_IN` 临时改为 `selected_queries.json` 并用 `SEL_OUT_SUFFIX=_legacy` 写独立产物，避免覆盖 canonical。

## 各阶段数据契约与完成标志

### Stage 01：商品属性抽取

读取 review/meta 原始数据，按商品建立后续 query 生成和选择所需的属性结构。完成标志：

```text
result/01_attribute_extraction/product_attributes.json
```

该文件必须是有效 JSON，包含 Stage 07/08 所需的商品属性字段；缺失属性在后续 schema 检查中显式暴露。

### Stage 02：HTML 清洗、用户句子与 ASIN mapping

下游文本唯一出口。处理顺序必须是：

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

清除 `&#34;` / `&lt;` / `&gt;` / `&amp;` / `<br>` 等 artifact，但不把普通文本中的非标准字符串误判为 HTML。`asin_to_users.json` 不能引用已删除用户，也不保留无用户 ASIN。Stage 02 只产生清洗后的句子和映射，不产生 GECToR profile。

完成标志：

```text
result/02_user_review_sentence_extract/uid_to_sentences.json
result/02_user_review_sentence_extract/uid_to_sentences.pkl
result/02_user_review_sentence_extract/asin_to_users.json
```

抽样检查句子中无 HTML entity/tag 残留，并校验 JSON、PKL、ASIN mapping 用户集合一致。Stage 03 和 Stage 09 都必须读这批新产物。

### Stage 03a：PCFG 特征与 supervised syntax encoder (canonical strict3)

读取 Stage 02 PKL，spaCy 解析句子并提取不含词项的 dependency/POS 结构规则，建立 sparse sentence-rule cache；随后训练一次 supervised syntax encoder，保存 frozen embedding 和 profile/validation/test 划分。Gaussian 拟合、attribution 和 ablation 不属于 Stage 03a。

完成标志：

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

检查 embedding cohort、句子布局、维度、finite 值相互一致；cache 属于其他 Stage 02 cohort 时直接停止重建。

### Stage 03b：trainable SVD+MLP user-style encoder (canonical 前置)

通过 `TG_STAGE=03b` 跑同一个 `syntax_pcfg_pipeline.py`：TruncatedSVD(20000→256) + InfoNCE 对比学习 MLP(256→64)。输出：

```text
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/svd_components.npz   # Vt (256, vocab_size), row_normalize
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/svd_mlp_encoder.pt  # StyleMLP state_dict + config
/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/vocab.json          # 必须与 Stage 03a 一致
```

Stage 04b `TG_SOURCE=svd_mlp` 与 Stage 08 `TG_GAUSSIAN_SOURCE=trainable_svdmlp` 都依赖此 cache。

### Stage 04a：analytical per-user full Σ Gaussian + theo gate (legacy)

只读取 Stage 03a frozen embedding 和 Stage 02 `asin_to_users.json`，拟合 per-user raw full covariance Gaussian，从用户自身 profile/validation 距离生成 cohort gate。**main 末尾自动调用 `_add_theoretical_gates()`**，给两个产物文件加 `d2_qXX_theoretical` 字段和 cohort `gate_T_high_theoretical` / `gate_T_low_theoretical`。无需单独跑 `rewrite_gaussian_with_theoretical_gate.py`（已合入）。

生产 artifact 记录 covariance 类型、gate quantile、embedding 来源、用户/ASIN 对齐信息，包含每个用户 `mu`、`sigma_inv`、样本信息、gate 距离字段。完成标志：

```text
result/04_gaussian/user_gaussian_stats.json
result/04_gaussian/user_gaussian_stats_rank1.json
```

验证 Cholesky/solve、协方差正定、距离非负、user/ASIN 引用完整后，才允许 Stage 05 和 Stage 08 legacy path 使用。rank1、eigenvalue floor、Ledoit–Wolf 等分析变体不能替代 canonical raw full covariance。

### Stage 04b：trainable per-user 对角 σ Gaussian (canonical)

PyTorch 实现：每个用户 `mu ∈ R^d`、`logsigma^2 ∈ R^d` (Embedding)，NLL + KL(N(0,I)) 训练。`TG_SOURCE` 切换数据源：

- `TG_SOURCE=raw`：在 Stage 03a supervised embedding 上训练 16d Gaussian。
- `TG_SOURCE=svd_mlp` (canonical)：在 Stage 03b svd_mlp 64d embedding 上训练 64d Gaussian，产物被 Stage 08 svdmlp path 消费。

环境变量同时控制 β（KL 权重）、epochs、batch 等 ablation。完成标志：

```text
result/04_gaussian/user_gaussian_stats_trainable.json          # TG_SOURCE=raw
result/04_gaussian/user_gaussian_stats_trainable_svdmlp.json   # TG_SOURCE=svd_mlp (canonical)
```

config 字段必须包含 `source`、`latent_dim`、`n_users_fitted` 等；Stage 08 加载时 fail-fast 校验 `source` 与 `latent_dim`。

### Stage 05：raw covariance validity audit (legacy path only)

对 Stage 04a artifact 做 E1–E4 raw covariance audit，定义哪些用户可进入 Stage 08 legacy path、每个 ASIN 是否有足够 valid-user coverage。不是另一次 Gaussian fitting。

完成标志：

```text
result/05_gaussian_audit/raw_cov_validity.json
result/05_gaussian_audit/asin_coverage_valid_ge2.json
```

交叉检查 `raw_cov_validity.json` summary 与 `per_user` 标记一致；coverage 只用 audit 通过的用户。Stage 08 legacy 必须等 audit 完成；svdmlp path 不依赖此 stage。

### Stage 06：既有 SFT adapter

```text
result/06_training_model/sft_lora/
```

不重新训练；Stage 07 直接消费。

### Stage 07：本地 Qwen SFT query pool

vLLM LLM engine 加载 Qwen + enable_lora=True + `LoRARequest` 指向 `sft_lora`；按 ASIN 批量生成 K_POOL candidates（单次 `generate` 调用）。按脚本内 5-layer strict filter（全覆盖 + 无重复 + extras 阈值）过滤。完成标志：

```text
result/07_gen_query/pool_queries.json
```

pool 必须与当前 Stage 04/05 cohort 和商品属性来源一致；signature 或来源不一致时必须重新生成。

### Stage 08：Mahalanobis / svdmlp query selection

`TG_GAUSSIAN_SOURCE` 切换路径：

- **未设 / `fit`**（legacy）：读 Stage 04a + Stage 05 + Stage 07 pool，对 query 用 Stage 03a frozen strict3 encoder 编码，per-ASIN 用 full Σ maha D² + cohort gate（gate_q=0.05），输出 `selected_queries.json`。
- **`trainable_svdmlp`**（canonical）：读 Stage 04b svdmlp + Stage 07 pool，用 Stage 03b svd_mlp encoder 编码，per-ASIN 用对角 σ 高斯 log-likelihood 选最佳 user（best-fit logp），保留 logp ≥ max_logp − logp_delta（默认 2.0）的所有 query；至少 1 条。输出 `selected_queries_svdmlp.json`。

两个路径共用同一份 `pool_queries.json`，不重新跑 Stage 07。完成标志：

```text
result/08_select_query/selected_queries.json           # legacy baseline
result/08_select_query/selected_queries_svdmlp.json     # canonical
```

输出 schema：`{config, summary, selections:[{asin, users:[{uid, query, d2|logp, ...}]}], ...}`。每个 ASIN/用户都来自当前 cohort；summary、selections 结构、候选来源一致。Stage 09/10/11/12 都必须读同一份 canonical selection。

`TG_LOGP_DELTA` 控制 svdmlp per-ASIN 相对阈值（默认 2.0）；`TG_GATE_MODE=logp_delta|d2`（svdmlp 内 fallback）。

### Stage 09：GECToR 与 SErCL 用户错误画像

从 Stage 08 canonical selection 提取用户 cohort，从 Stage 02 清洗后的 `uid_to_sentences.json` 取满足最低句数要求的句子；通过独立 GECToR 子进程批量纠正（`gector_subprocess.py` 在 gector_env），ERRANT 对齐原句/纠正句，spaCy UD 提取 error type、D3 context、`P_u(e|r)` 与 `L_u(r,e)`。

完成标志：

```text
result/09_sercl_user_profile/user_sercl_profile.json
result/09_sercl_user_profile/cohort_summary.json
```

profile 用户来自当前 Stage 08 selection；句子来自当前 Stage 02 artifact。原始 JSON 保存 `user_profiles` 和 `user_word_edits`；`transformation_history` 由 Stage 10 的 `load_sercl_profile()` 从 `user_word_edits` 动态构建，不能用顶层字段判断。GECToR 子进程返回码、输出行数、index 缺失都直接 raise。

### Stage 10：selected association 的 typo 注入

读取 Stage 09 profile + Stage 04a Gaussian + Stage 08 canonical selection，对所有当前 selection 中有 profile/Gaussian/cohort gate 的 association 做单次候选注入：

```text
query tokenization + syntax context
  → 用户 char-level error rate 选候选 token
  → 优先查 transformation_history 中 user-historical typo
  → 无匹配则 generic_fallback，按用户机制分布生成
  → minimality / edit-distance / length / meaningful-word 检查
  → target Gaussian Q gate (theoretical q 高/低侧，d2_threshold = stats[d2_qXX_theoretical])
  → competitor-exclusive cohort gate
  → MiniLM semantic preservation gate
  → 只写入通过全部条件的 char-level typo pair
```

当前不包含 case/apostrophe 等 surface-form typo；每条记录必须含 original/typo query、uid/asin、原词/错词、mechanism、transformation_source、edit distance、sem sim 和必要上下文。`MAX_EDIT_DISTANCE`、semantic threshold、机制集合、gate quantile 以脚本内配置和产物 `config` 为准。

完成标志：

```text
result/10_typo_injection/typo_injection_results.json
result/10_typo_injection/cohort_summary.json
```

检查 `cohort_summary.json` 的 attempted/written/gate failures/transformation_source_counts，重点区分 `user_historical_*` 和 `generic_fallback`；不能把候选尝试数误报为成功注入数。Stage 12 只读当次成功 results。

### Stage 11：7 retriever 评估 + query-level stability_flip (canonical: svdmlp)

必须通过 standalone driver 跑，而不是只 import unified 模块：

```bash
nohup $PY 11_syntactic_evaluation/run_stage11_eval.py \
  > /home/wlia0047/ar57/wenyu/PersoanlQuery/result/_logs/stage11_retrieval.log 2>&1 &
```

`run_stage11_eval.py` 调 `syntax_subspace_retrieval_unified.main()`，返回后检查 canonical 文件确实写出。

当前 `SEL_IN` 默认指向 `selected_queries_svdmlp.json`（canonical）。评估覆盖 registry 中 BM25、SPLADE、MiniLM、MPNet、BGE、GTE、ColBERTv2；sim09 query-query reference 锚定 canonical MiniLM 384d。可复用 corpus/query embedding cache，但 selection signature、corpus signature、query shape 不一致时必须重建。

**产物指标**：
- `per_query.json`：每条 query × 每个 retriever × rank/hit@k
- `retrieval_summary.json`：headline + 每 retriever 聚合
- `volatility.json`：canonical BM25 + MiniLM flip rate 摘要

**stability_flip**：在每个 ASIN 内部，对该 ASIN ≥2 query 跑同一 retriever，比较 Hit@k 结果在 query 之间的翻转频率（`Hit@k_FlipRate_mean`）。canonical svdmlp selection 典型值（5416 asin_ge2）：BM25 4.7% / SPLADE 3.3% / MiniLM 2.0% / MPNet 1.9% / BGE 3.0% / GTE 4.1% / ColBERTv2 5.3%（Hit@1）。

完成标志：

```text
result/11_syntactic_evaluation/per_query.json
result/11_syntactic_evaluation/retrieval_summary.json
result/11_syntactic_evaluation/volatility.json
```

`per_query.json` 记录当前 selection signature、corpus、retriever registry 和所有 query record；summary/volatility 必须来自同一次运行。历史遗留的旧评估目录不再是 canonical 目录，不得写入或用其文件替代 Stage 11 结果。

**legacy 对照产物**：若需与 baseline（gate_q=0.05 selection）对比 flip 率，把 `syntax_subspace_retrieval_unified.py` 的 `SEL_IN` 临时改为 `selected_queries.json` 并设 `SEL_OUT_SUFFIX=_legacy`：

```text
result/11_syntactic_evaluation/per_query_legacy.json
result/11_syntactic_evaluation/retrieval_summary_legacy.json
result/11_syntactic_evaluation/volatility_legacy.json
```

跑完恢复 SEL_IN 到 `selected_queries_svdmlp.json`。

### Stage 12：paired typo retrieval degradation

从当前 Stage 10 `typo_injection_results.json` 读所有成功 pair，把每个 pair 的 original 与 typo query 组成同一批次，使用与 Stage 11 相同的 retriever registry、corpus、cache 逻辑做 paired retrieval。只报告 `original_hit@k − typo_hit@k`：正值表示 typo 使召回下降。不计算 flip rate，不替换 Stage 11 结果或使用其他输入。

运行前检查 Stage 10 schema；交叉校验 Stage 12 `config.n_pairs` 与 Stage 10 实际写入成功 results 数；不一致直接停止。无成功 pair 时写明确空结果并终止。

完成标志：

```text
result/12_typo_evaluation/per_query.json
result/12_typo_evaluation/retrieval_degradation.json
```

`retrieval_degradation.json` 记录 `n_pairs`、corpus/query signature、每 retriever 的 original/typo hit@k、degradation、neutral/help/hurt 计数；`per_query.json` 能回溯 uid、asin、original/typo query 和每 retriever 的 paired rank/hit。

## 最终 canonical 结果

```text
result/01_attribute_extraction/product_attributes.json
result/02_user_review_sentence_extract/uid_to_sentences.json
result/02_user_review_sentence_extract/uid_to_sentences.pkl
result/02_user_review_sentence_extract/asin_to_users.json
result/03_spacy_encode/syntax_pcfg_strict_attr.json
result/04_gaussian/user_gaussian_stats.json              # Stage 04a full Σ
result/04_gaussian/user_gaussian_stats_rank1.json
result/04_gaussian/user_gaussian_stats_trainable.json    # Stage 04b raw (legacy comparison)
result/04_gaussian/user_gaussian_stats_trainable_svdmlp.json  # Stage 04b svd_mlp (canonical)
result/05_gaussian_audit/raw_cov_validity.json
result/05_gaussian_audit/asin_coverage_valid_ge2.json
result/07_gen_query/pool_queries.json
result/08_select_query/selected_queries.json           # legacy baseline
result/08_select_query/selected_queries_svdmlp.json     # canonical
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
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python

$PY -c 'import json; p=json.load(open("result/02_user_review_sentence_extract/uid_to_sentences.json")); print(len(p), next(iter(p.values()))[:1])'
$PY -c 'import json; p=json.load(open("result/08_select_query/selected_queries_svdmlp.json")); print(p["config"].get("logp_delta"), len(p["kept"]))'
$PY -c 'import json; p=json.load(open("result/10_typo_injection/cohort_summary.json")); print(p["totals"]); print(p["totals"]["transformation_source_counts"])'
$PY -c 'import json; p=json.load(open("result/11_syntactic_evaluation/per_query.json")); print(p["config"], p["n_queries"])'
$PY -c 'import json; p=json.load(open("result/11_syntactic_evaluation/volatility.json")); print(p["stability_flip"])'
$PY -c 'import json; p=json.load(open("result/12_typo_evaluation/retrieval_degradation.json")); print(p["config"])'
```

快速检查只用于读取 artifact 当前状态；不固化任何数量基线。运行中的进程用 `tail` / `grep` / `ps` / `nvidia-smi` 即时检查；禁止 `sleep` 等待。

## 故障排查

| 症状 | 处理 |
|------|------|
| Stage 02 仍有 HTML artifact | 检查 entity decode、tag strip、space collapse 是否发生在句子切分前；确认输出来自当前 Stage 02 run |
| Stage 02 mapping 引用不存在用户 | 用过滤后的 `uid_to_sentences` 重建 `asin_to_users`，不沿用旧 mapping |
| Stage 03a cache cohort mismatch | 比较 `meta.json`、`uid_list.json`、`user_n_sents.json`、embedding shape 和 Stage 02 用户集合；不一致删除 cache 后重建 |
| Stage 03b svd_mlp 与 03a vocab 不一致 | 两个 stage 用同一份 `vocab.json`；03b 不重训 vocab，只 fit SVD + 训 MLP；cache 缺失时直接重建 |
| Stage 04a Cholesky/solve 或 D² 异常 | 检查 Stage 03a embedding 来源、raw full covariance、有限值和 gate quantile；不要切换到未批准的正则化变体掩盖问题 |
| Stage 04a 输出没有 `*_theoretical` 字段 | `_add_theoretical_gates()` 应当 fit main 末尾自动调用；若缺失检查 main() 末尾是否完整执行 |
| Stage 04b `source` mismatch | Stage 04b output `config.source` 必须与 Stage 08 `TG_GAUSSIAN_SOURCE` 路径对齐；svd_mlp 必须 source=svd_mlp |
| Stage 05 valid/coverage 不一致 | 对照 audit summary、per-user 标志、ASIN coverage；Stage 08 legacy 必须等 audit 完成；svdmlp path 不依赖 |
| Stage 07 生成失败 | 检查既有 SFT adapter、本地 Qwen model cache、GPU 显存、批量生成日志 |
| Stage 08 legacy selection 异常 | 检查 Stage 04a/05/pool 是否来自同一 cohort，确认 target/competitor gate 和 candidate encoding cache 的 signature |
| Stage 08 svdmlp selection 异常 | 检查 Stage 04b svdmlp Gaussian 是否存在、Stage 03b svd_components.npz + svd_mlp_encoder.pt cache、`TG_LOGP_DELTA` 阈值 |
| Stage 09 profile 没有 transformation history | 不要在 raw 顶层 JSON 搜索该字段；通过 `load_sercl_profile()` 从 `user_word_edits` 动态构建后再检查 |
| Stage 09 GECToR 子进程失败 | 检查 gector_env Python、模型 cache、IPC JSONL、子进程返回码；缺行或错位直接重跑 |
| Stage 10 全部拒绝 | 分别检查 profile 用户交集、char-level history、minimality、Gaussian Q gate、competitor-exclusive gate、MiniLM gate |
| Stage 10 source 全是 generic_fallback | 确认 profile 来自当前 Stage 02 清洗文本重建；检查 query/review vocabulary 是否有历史 typo 可匹配；不要把候选尝试数当作 historical 命中 |
| Stage 11 canonical 文件缺失 | 必须跑 `run_stage11_eval.py` 而不是只 import `syntax_subspace_retrieval_unified.py`；检查当前 canonical 结果目录而非历史遗留 |
| Stage 11 SEL_IN 仍指向 baseline | 脚本默认 `selected_queries_svdmlp.json`；若之前临时切到 baseline，跑完 flip 率对照后必须改回 svdmlp |
| Stage 11/12 cache signature mismatch | 保留 mismatch 日志，按当前 selection/corpus/query signature 重编码；不得静默复用旧 embedding |
| Stage 12 pair 数不一致 | 比较 Stage 10 成功 results 长度、Stage 12 `config.n_pairs` 和 paired records；不一致直接停止 |
| Stage 12 没有 pair | 检查 Stage 10 `n_injected_written`；若确实为零，保留明确空结果，不替换为历史 |

## 规则遵守

- 所有阶段按依赖顺序运行；Stage 11 必须通过 `run_stage11_eval.py` 独立产出 canonical 文件后，才视为 Stage 11 完成。
- 所有运行参数硬编码；固定 `pq_env`，GECToR 固定使用指定 `gector_env`。
- 所有生成、embedding 和解码使用批量路径；不得恢复逐条 LLM 生成或逐元素大矩阵计算。
- 长任务用 `nohup ... &` 或 `setsid ... < /dev/null &`；不使用 `sbatch`、`srun` 或 `sleep` 轮询。
- 日志/cache/中间文件写入 `hj82_scratch2` 或 `result/_logs/`；`result/` 只保存结果 artifact。
- 每个新阶段或修改阶段先 smoke，再 full；修改 Python 后先运行 `python -m py_compile`。
- 所有 LLM 调用只使用项目允许的本地 Qwen backend，不新增远程客户端。
- 必需文件缺失、schema 不一致、用户/ASIN 关联不一致或 signature 不一致时直接 raise。
- canonical selection = Stage 04b (trainable svd_mlp) → Stage 08 (svdmlp logp_delta=2.0) → Stage 11；legacy baseline 仅作对照保留。
- 本技能不执行 commit 或 push，除非用户另行明确要求。
