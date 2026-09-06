---
name: syntax-subspace-pipeline
description: 跑全 pipeline (10 stages, 2026-09-06)：Step1 attrs → Step2 sents → 03 PCFG 32d supervised → 07 vLLM pool → 08 Mahal select → 09 SErCL GECToR → 10 typo inject → 11 7-retriever eval → 12 typo degradation eval.
---

# PersonalQuery Syntax-Subspace Pipeline — 10 Stages (2026-09-06)

## 何时用

完整端到端跑通 PersonalQuery 流水线：用户评论 → 商品属性 → spaCy PCFG 结构规则 → supervised `_SupEncoder` 32d → per-user Gaussian → vLLM SFT 个性化 pool 生成 → Mahalanobis + non-shared cohort 选择 → SErCL GECToR 错误画像 → 上下文结构 P(error|c) typo 注入（Mahalanobis + MiniLM + exclusive 三 gate）→ 7-retriever 评估 → typo 注入 paired hit@k 退化评估。

**核心设计**：
- 句法表征：spaCy → PCFG 结构规则（21737 binary counts）→ supervised `_SupEncoder(21737→256→32)` → 32d z
- 距离度量：**全 Mahalanobis D²(z, μ_u, Σ_u⁻¹)**（非 cosine），per-user 32×32 Σ_u⁻¹
- 选择 gate：(1) target 内 d² ≤ Q_95 (2) ∀competitor: d² > comp Q_95 (3) bootstrap 稳定
- typo 注入：per-token `P_u(error|c_i)` Bernoulli（7 级 hierarchy: full/d4/d3/clause/depth/sibling/coarse）+ 4-gate (Bernoulli → Mahalanobis Q_95 → non-shared cohort → MiniLM cosine ≥ 0.9)
- LLM：仅本地 Qwen（vLLM 或 `llm_client.py`），无远程后端

**输入**：
- `data/Baby_Products_2023.jsonl`（用户评论）
- `data/meta_Baby_Products_2023.jsonl.gz`（产品元数据）

**预计耗时**：Step1(~5min) → Step2(~3min) → 03 PCfg (~60min, GPU) → 07 vLLM pool (~5min) → 08 Mahal select (~5min) → 09 SErCL GEC (~30min) → 10 typo inject (~5min) → 11 retrieval (~10min) → 12 typo eval (~3min)

## 目录结构（2026-09-06 重整，9 步骤脚本 + 共享 03）

```
01_attribute_extraction/      extract_product_attrs.py       (商品属性抽取，5 attrs/ASIN)
02_user_review_sentence_extract/  extract_user_sentences.py   (uid → sents)
03_spacy_encode/              syntax_pcfg_pipeline.py         (PCFG rules + supervised _SupEncoder + 32d z 缓存)
04_gaussian/                  build_gaussian.py               (legacy 287d, 保留为兼容 artifact，08/10 已不走此)
07_gen_query/                 sft_pool_generate.py            (vLLM Qwen2.5-0.5B + LoRA pool 生成)
08_select_query/              syntax_select_mahalanobis_gate.py  (32d Mahal + Q_0.95 + non-shared cohort)
09_sercl_user_profile/        sercl_user_profile.py           (GECToR-2024 RoBERTa-large subprocess)
                              gector_subprocess.py            (gector_env 子进程入口)
10_typo_injection/            typo_inject.py                  (主脚本: per-ctx Bernoulli + 4-gate)
                              error_location.py               (multi-level hierarchy: full/d4/d3/clause/depth/sibling/coarse)
                              typo_classifier.py              (Levenshtein → mechanism)
                              injection_sampler.py            (per-token Bernoulli + 4-gate 检查)
11_syntactic_evaluation/      syntax_subspace_retrieval_unified.py  (7-retriever: BM25/SPLADE/MiniLM/MPNet/BGE/GTE/ColBERTv2)
12_typo_evaluation/           typo_retrieval_eval.py          (paired orig/typo hit@k 退化)
```

**pcfg_cache**（`/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/`，03 输出，08/10 共享只读）：
- `adaptive_encoder.pt`（冻结的 `_SupEncoder(21737→256→32)`）
- `adaptive_embeddings.npz`（profile/val/test 三段 32d z，3-way hash split）
- `counts.npz`（CSR 句法规则计数，21737 维）
- `sent_vectors.npz`、`vocab.json`、`uid_list.json`、`user_n_sents.json`

## 路径约定（Rule 10/12/13/16）

- **Inputs**（只读）：`data/`, `asin_users/`
- **Intermediate cache**：`/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/`, `…/hf_cache/`
- **Final results**：`result/<NN_module>/<name>.json`
- **Log/output**：`/home/wlia0047/hj82_scratch2/wenyu/logs/<stage>.log`
- **运行 cwd**：`/home/wlia0047/ar57/wenyu/PersoanlQuery`（Rule 11）
- **解释器**：`/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`（Rule 7）

## 配置（硬编码，不可传参）

各 stage 内部常量；典型：
| 常量 | 值 | 说明 |
|------|-----|------|
| `MODE` (03) | `"adaptive"` | 当前主流：supervised 32d encoder |
| `SYNTAX_DIM` | 32 | supervised 32d 句法空间 |
| `N_SENTS_MIN_PROFILE` | 100 | per-user profile 最小句数 |
| `D2_Q` | 0.95 | selection gate 阈值（target & competitor 同号） |
| `QwenLocal` | Qwen2.5-0.5B-Instruct + SFT LoRA | 07 vLLM 加载 |
| `MAX_NEW_TOKENS` | 80 | vLLM pool 生成上限 |
| `K_POOL` | 50 | 每 ASIN pool 大小 |
| `GEC_MODEL` | gector-2024-roberta-large | 09 GEC 模型 |
| `SEMANTIC_THRESHOLD` | 0.9 | 10 typo MiniLM cosine gate |
| `KS` (12) | (1, 5, 10) | hit@k 报告 |

## 运行顺序

```
Step 1 → 01_attribute_extraction/extract_product_attrs.py
Step 2 → 02_user_review_sentence_extract/extract_user_sentences.py
Step 3 → 03_spacy_encode/syntax_pcfg_pipeline.py  (MODE="adaptive", 训练 _SupEncoder)
Step 4 → 07_gen_query/sft_pool_generate.py         (vLLM pool)
Step 5 → 08_select_query/syntax_select_mahalanobis_gate.py  (Mahal + cohort)
Step 6 → 09_sercl_user_profile/sercl_user_profile.py  (GECToR subprocess)
Step 7 → 10_typo_injection/typo_inject.py          (4-gate typo inject)
Step 8 → 11_syntactic_evaluation/syntax_subspace_retrieval_unified.py  (7 retr)
Step 9 → 12_typo_evaluation/typo_retrieval_eval.py  (paired degradation)
```

> **注**：04_gaussian/build_gaussian.py 是 legacy 287d artifact 路径，已被 03+08/10 的 32d 路径取代；除非显式 audit 287d，否则跳过。

## 前置检查

### 1. Python 环境

```bash
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python
$PY --version  # Python 3.10.x
```

### 2. gector_env（Step 6 GEC 子进程需要）

```bash
ls /home/wlia0047/hj82_scratch2/wenyu/venvs/gector_env/bin/python
```

需 Python 3.11 + transformers ≥4.49 + torch ≥2.5（GECToR-2024 RoBERTa-large 要求）。

### 3. vLLM（Step 4 GPU 上 lazy 初始化）

07 脚本内部 `from vllm import LLM` 单例加载，无需外部 HTTP server。GPU util=0.42。

### 4. GPU

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

需 ≥ 16GB free（03 encoder 训练 + 07 vLLM + 11 dense retrievers 共用）。

### 5. HF cache

```bash
ls /home/wlia0047/hj82_scratch2/wenyu/hf_cache/hub/ | head -10
```

需含：sentence-transformers/{all-MiniLM-L6-v2, all-mpnet-base-v2}, BAAI/bge-base-en-v1.5, thenlper/gte-base, naver/splade-cocondenser-ensembledistil, colbert-ir/colbertv2.0, pszemraj/bart-base-grammar-synthesis, roberta-large, gector。

## 跑 9 stages

### Step 1：商品属性抽取

```bash
nohup $PY 01_attribute_extraction/extract_product_attrs.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step1_attrs.log 2>&1 &
```

**完成标志**：`step1_attrs.log` 含 `wrote → ...result/01_attribute_extraction/product_attributes.json`

### Step 2：用户句子提取

```bash
nohup $PY 02_user_review_sentence_extract/extract_user_sentences.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step2_sents.log 2>&1 &
```

**完成标志**：`step2_sents.log` 含 `Done` + `result/02_user_review_sentence_extract/uid_to_sentences.pkl` (also .json)

### Step 3：PCFG 32d supervised encoder 训练 + z 缓存

```bash
nohup $PY 03_spacy_encode/syntax_pcfg_pipeline.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step3_pcfg.log 2>&1 &
```

**MODE = "adaptive"**：PCFG rules → supervised `_SupEncoder(21737→256→32)` → 32d z profile/val/test 3-way hash split。

**完成标志**：`step3_pcfg.log` 含：
- `wrote cache: ...pcfg_cache/adaptive_encoder.pt`
- `wrote cache: ...pcfg_cache/adaptive_embeddings.npz`
- `wrote → ...result/03_spacy_encode/syntax_pcfg_*.json` 系列

**轮询**（Rule 9，禁止长 sleep）：
```bash
for i in $(seq 1 60); do
  if [ -f /home/wlia0047/hj82_scratch2/wenyu/pcfg_cache/adaptive_embeddings.npz ]; then break; fi
  tail -n 1 /home/wlia0047/hj82_scratch2/wenyu/logs/step3_pcfg.log
done
```

### Step 4：vLLM pool 生成

```bash
nohup $PY 07_gen_query/sft_pool_generate.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step4_vllm_pool.log 2>&1 &
```

**配置**：vLLM LLM engine 加载 Qwen2.5-0.5B-Instruct + LoRA（`result/06_training_model/sft_lora`），每 ASIN K_POOL=50 candidates。

**完成标志**：`step4_vllm_pool.log` 含 `wrote → ...result/07_gen_query/pool.json`

### Step 5：Mahal + cohort 个性化选择

```bash
nohup $PY 08_select_query/syntax_select_mahalanobis_gate.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step5_select.log 2>&1 &
```

**完成标志**：`step5_select.log` 含 `wrote → ...result/08_select_query/selected_queries.json`（每 ASIN 健康用户各 1 strict-aligned query）

**Gate**：
1. d_target ≤ target.gate_T（Q_95）
2. ∀competitor: d_competitor > comp.gate_T（non-shared zone）
3. Bootstrap stability P ≥ 0.95

### Step 6：SErCL GECToR 错误画像

```bash
nohup $PY 09_sercl_user_profile/sercl_user_profile.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step6_sercl.log 2>&1 &
```

**完成标志**：`step6_sercl.log` 含 `wrote → ...result/09_sercl_user_profile/user_sercl_profile.json` + `cohort_summary.json`

**流程**：原句 → GECToR-2024 RoBERTa-large (gector_env subprocess) → ERRANT edit alignment → UD parse → per-user `error_rate_by_context[ctx_sig]` + `mechanism_totals`。

### Step 7：typo 注入（4-gate）

```bash
nohup $PY 10_typo_injection/typo_inject.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step7_typo.log 2>&1 &
```

**4-Gate 链**：
1. **Bernoulli per-token**：每个 token 按 7-level hierarchy 查 `error_rate_by_context[sig]`（full → d4 → d3 → clause → depth → sibling → coarse → p_u_err → 0.02）
2. **Mahalanobis D²**：typo query 的 32d z 必须 ≤ target Q_95
3. **Non-shared cohort**：typo query 的 32d z 必须 > 所有 healthy competitor 的 Q_95（asin_cohort_gates.json）
4. **MiniLM cosine ≥ 0.9**：typo query 与原 query 的语义向量 cosine ≥ 0.9

**输入**：
- `result/09_sercl_user_profile/user_sercl_profile.json`
- `result/10_typo_injection/user_mahalanobis_stats.json`（per-user 32d μ + Σ_u⁻¹ + D² Q_50/75/95）
- `result/10_typo_injection/asin_cohort_gates.json`（asin → uid → {μ, Σ⁻¹, gate_T}）
- `result/08_select_query/selected_queries.json`（uid/asin/query 源）

**完成标志**：`step7_typo.log` 含 `wrote → ...result/10_typo_injection/typo_injection_results.json`（成功注入 pairs slim 6 字段）+ `cohort_summary.json`

**机制 taxonomy**：`keyboard_adjacent / letter_swap / letter_repetition` (typo) + `case_error / apostrophe_error` (surface-form)；`semantic_substitution` 不注入。

### Step 8：7-retriever 评估

```bash
nohup $PY 11_syntactic_evaluation/syntax_subspace_retrieval_unified.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step8_eval.log 2>&1 &
```

**7 retrievers**：BM25 · SPLADE · MiniLM-L6-v2 · MPNet-base-v2 · BGE-base-en-v1.5 · GTE-base · ColBERTv2 (768→128 late interaction)

**完成标志**：`step8_eval.log` 含：
- `wrote → ...result/11_syntactic_evaluation/asin_to_doc.json`
- `wrote → ...result/syntactic_evaluation/retrieval_summary.json`
- `wrote → ...result/syntactic_evaluation/volatility.json`

### Step 9：typo paired hit@k 退化评估

```bash
nohup $PY 12_typo_evaluation/typo_retrieval_eval.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step9_typo_eval.log 2>&1 &
```

**配置**：从 `result/10_typo_injection/typo_injection_results.json` 读 45 paired (uid, asin, original, typo)，对每对 orig+typo 都跑 7 retrievers，**只算 hit@1/5/10 退化**（orig - typo），不算 flip。

**完成标志**：`step9_typo_eval.log` 含：
- `wrote → ...result/12_typo_evaluation/per_query.json`
- `wrote → ...result/12_typo_evaluation/retrieval_degradation.json`

## 验证所有 stage 完成

```bash
ls /home/wlia0047/ar57/wenyu/PersoanlQuery/result/{01,02,03_spacy_encode,07,08,09,10,11,12}*/*.{json,pkl} 2>/dev/null
```

期望产物：
- `result/01_attribute_extraction/product_attributes.json`
- `result/02_user_review_sentence_extract/uid_to_sentences.pkl` + `.json`
- `result/03_spacy_encode/syntax_pcfg_user_gaussian.json` 等 + pcfg_cache/adaptive_*
- `result/07_gen_query/pool.json`
- `result/08_select_query/selected_queries.json`
- `result/09_sercl_user_profile/user_sercl_profile.json` + `cohort_summary.json`
- `result/10_typo_injection/{user_mahalanobis_stats.json, asin_cohort_gates.json, typo_injection_results.json, cohort_summary.json}`
- `result/11_syntactic_evaluation/{asin_to_doc.json, retrieval_summary.json, volatility.json}`
- `result/12_typo_evaluation/{per_query.json, retrieval_degradation.json}`

## 规则遵守

- Rule 3：参数硬编码，不传参
- Rule 7：`pq_env` 解释器（Step 6 GEC 子进程走 `gector_env`）
- Rule 8：`nohup ... &` 后台运行
- Rule 9：`tail` + `grep` 轮询，禁止 `sleep X; tail`
- Rule 10：cache/log 写 `/home/wlia0047/hj82_scratch2/wenyu/`，结果写 `result/`
- Rule 11：cwd = 项目根
- Rule 12：脚本在功能目录，不写到 scratch2
- Rule 13：`result/` 只放 JSON/NPZ/CSV/log
- Rule 16：每个功能目录单一 JSON
- Rule 17：脚本/产物名无版本号后缀
- Rule 18：每功能目录单脚本
- Rule 20：每个 stage 第一次跑先 SMOKE=True 验证

## 故障排查

| 症状 | 排查 |
|------|------|
| Step 3 encoder 训练 OOM | 减小 `CHUNK_USERS` 或 `PARSE_BATCH_SIZE` |
| Step 4 vLLM OOM | `--gpu-memory-utilization 0.42`（已在 07 脚本里） |
| Step 6 GEC 子进程失败 | 检查 `/home/wlia0047/hj82_scratch2/wenyu/venvs/gector_env/bin/python` |
| Step 7 typo injection 0 pairs | 检查 SErCL profile 是否加载了 edits；查 `n_edits` per user |
| Step 7 4-gate 全 reject | typo 太"远"出用户 Gaussian — 增大 `D2_THRESHOLD_QUANTILE` 或加宽容 cohort |
| Step 8 retrieval cache stale | 删除 `result/11_syntactic_evaluation/asin_to_doc.json.sig` 重新生成 |
| Step 9 typo eval 0 pairs | 必须先跑 Step 7；检查 `result/10_typo_injection/typo_injection_results.json` |

## 完成后

输出任务摘要 + 9 stage 状态 + 关键统计，并以"当前任务已完成，请做下一个任务的指示。"结尾。