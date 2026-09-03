---
name: syntax-subspace-pipeline
description: 跑全 pipeline：01商品属性 → 02句子提取 → 03 Gaussian → 04 pool生成 → 05个性化选择 → 06检索评估。spaCy 287d + cosine distance，无 TinyStyler。
---

# Syntax Subspace Pipeline — 6 Stages

## 何时用

完整端到端跑通 Syntax Subspace 流水线：用户评论 → spaCy 287d 句法特征 → per-user Gaussian → vLLM 个性化 pool 生成 → 287d 余弦距离选择 → 7-retriever 评估。

**核心变更（Phase 8.V alt）**：
- 移除 TinyStyler/Wegmann，改用 **spaCy 287d 句法特征 + 余弦距离**
- Stage 1 pool 生成：vLLM + 风格锚点 prompt + 287d 余弦距离验证
- Stage 2 Gaussian 拟合：per-user Gaussian in 287d syntax space

**输入**：
- `data/meta_Baby_Products_2023.jsonl.gz`（用户评论）
- `result/product_attributes.json`（Step 1 产出，per-ASIN 属性字典）

**输出**：`/home/wlia0047/ar57/wenyu/PersoanlQuery/result/<dir>/<name>.json`

**预计耗时**：Step 1（~5min）→ Stage 1 pool gen（~2min）→ Stage 2 Gaussian（~30min）→ Stage 3（~5min CPU）→ Stage 4（~10min GPU）

## 目录结构（2026-09-03 更新）

```
01_attribute_extraction/      extract_product_attrs.py       (商品属性抽取)
02_user_review_sentence_extract/  extract_user_sentences.py   (用户句子提取)
03_gaussian/                  compute_syntax_gaussian.py     (287d Gaussian 拟合 + exclusive d_self)
                                  empirical_gaussian.py         (旧版 empirical Gaussian)
                                  vades_pipeline.py            (旧版 VADES，研究用)
                                  compute_exclusive_distance.py
                                  compute_fitting_quality.py
04_gen_query/                 syntax_subspace_pool_regen.py (vLLM pool 生成，spaCy-only)
05_select_query/              syntax_subspace_select_strict_alignment.py  (个性化选择)
06_syntactic_evaluation/      syntax_subspace_retrieval_unified.py  (7-retriever 评估)
common/                       syntax_subspace_utils.py        (共享路径 + 超参)
                                  syntactic_features.py        (287d per_sentence_features_v2)
```

## 配置（硬编码，不可传参）

| 常量 | 值 | 说明 |
|------|-----|------|
| `N_INPUT` | 5 | top-5 非数值属性 |
| `K_POOL` | 50 | 每 ASIN 候选数 |
| `TEMP` | 0.7 | vLLM 温度 |
| `MAX_TOKENS` | 120 | 生成上限 |
| `MAX_QUERY_TOKENS` | 60 | strict 长度阈值 |
| `FEAT_DIM` | 287 | spaCy 句法特征维度 |
| `D_SELF_TARGET` | 0.7 | 余弦距离目标阈值 |

**5-layer strict filter**：
1. `attrs_covered == N`（5/5 属性全覆盖）
2. `!has_invalid_punct`（不以 `here:`/`brand:` 等开头）
3. `has_first_person`（含 `I`/`my`/`looking for` 等）
4. `!has_emoji`（无 emoji）
5. `!has_self_talk`（无自言自语 `thanks!`/`can someone help` 等）
6. `len(query) ≤ 60` tokens

## 运行顺序

```
Step 1  → 01_attribute_extraction/extract_product_attrs.py
Step 2  → 02_user_review_sentence_extract/extract_user_sentences.py
Stage 1 → 03_gaussian/compute_syntax_gaussian.py
Stage 2 → 04_gen_query/syntax_subspace_pool_regen.py
Stage 3 → 05_select_query/syntax_subspace_select_strict_alignment.py
Stage 4 → 06_syntactic_evaluation/syntax_subspace_retrieval_unified.py
```

## 前置检查

### 1. Python 环境

```bash
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python
$PY --version  # Python 3.10.x
```

### 2. 工作目录

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
```

### 3. vLLM HTTP server (:8800)

```bash
curl -s -m 3 http://localhost:8800/v1/models | head -3
```

若空响应，启动 vLLM：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="top_k_top_p_sampling_from_logits,top_k_mask_logits,top_p_renorm_probs,sampling,renorm"
nohup $PY -m vllm.entrypoints.openai.api_server \
  --model /home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct \
  --port 8800 --host 0.0.0.0 --gpu-memory-utilization 0.85 --enforce-eager \
  > /home/wlia0047/hj82_scratch2/wenyu/tmp/vllm_server.log 2>&1 &
```

等 3-5 分钟，轮询直到返回模型列表。

### 4. GPU

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

需 ≥ 20GB free。

## 路径约定

- **Inputs**（只读）：`data/`, `asin_users/`
- **Intermediate cache**：`result/syntax_style_encoder/`（spaCy 特征 + Gaussian）
- **Final results**：`result/<dir>/<name>.json`

## 跑 6 stages

### Step 1：商品属性抽取

```bash
nohup $PY 01_attribute_extraction/extract_product_attrs.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step1_attrs.log 2>&1 &
```

**完成标志**：`step1_attrs.log` 含 `wrote → ...result/product_attributes.json`

### Step 2：用户句子提取

```bash
nohup $PY 02_user_review_sentence_extract/extract_user_sentences.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/step2_sents.log 2>&1 &
```

**完成标志**：`step2_sents.log` 含 `wrote → ...result/03_gaussian/uid_to_sentences.pkl`

### Stage 1：287d Gaussian 拟合 + exclusive d_self

```bash
nohup $PY 03_gaussian/compute_syntax_gaussian.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage1_gaussian.log 2>&1 &
```

**完成标志**：`stage1_gaussian.log` 含：
- `wrote → ...result/syntax_style_encoder/user_syntax_gaussian.json`（per-user Gaussian）
- `wrote → ...result/syntax_style_encoder/syntax_exclusive_distance.json`（per-(ASIN, user) exclusive d_self）

**输出**：
- `user_syntax_gaussian.json`：5144 users，mu 287d + sigma 287d
- `syntax_exclusive_distance.json`：per-(ASIN, user) exclusive d_self 分布，>0 占 33%

### Stage 2：vLLM pool 生成（spaCy-only，无 TinyStyler）

```bash
nohup $PY 04_gen_query/syntax_subspace_pool_regen.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage2_pool.log 2>&1 &
```

**配置**：
- vLLM prompt 含 target_user 历史句子作为风格锚点
- spaCy 287d 编码 + 余弦距离
- d_self 阈值：`max(t_cos, 0.7)`，其中 `t_cos = t_eucl / 12.0`

**完成标志**：`stage2_pool.log` 含：
- `strict: XXX (XX%)`
- `d_self_pass: XXX (XX%)`
- `both pass: XXX (XX%)`
- `wrote → ...result/gen_query/pool.json`

**期望**：
- strict rate：~60-77%
- d_self_pass rate：~69%
- both pass rate：~52%（目标 d_self < 0.7）

**轮询**（rule 9，禁止长 sleep）：
```bash
for i in 1 2 3 4 5; do
  sleep 30; tail -n 5 /home/wlia0047/hj82_scratch2/wenyu/logs/stage2_pool.log
  if grep -q "wrote →" /home/wlia0047/hj82_scratch2/wenyu/logs/stage2_pool.log; then break; fi
done
```

### Stage 3：个性化选择（Mahalanobis + PCA48）

```bash
nohup $PY 05_select_query/syntax_subspace_select_strict_alignment.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage3_select.log 2>&1 &
```

**完成标志**：`stage3_select.log` 含：
- `STAGE 2 — FEATURES`
- `saved cache: ...05_select_query/stage7b_query_features.jsonl.gz`
- `wrote → ...result/syntax_style_encoder/stage8_5_selection_stats.json`

### Stage 4：7-retriever 评估

```bash
nohup $PY 06_syntactic_evaluation/syntax_subspace_retrieval_unified.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage4_eval.log 2>&1 &
```

**7 个 retriever**：BM25 · SPLADE · MiniLM · MPNet · BGE-base · GTE-base · ColBERTv2

**完成标志**：`stage4_eval.log` 含：
- `wrote → ...result/syntactic_evaluation/retrieval_summary.json`
- `wrote → ...result/syntactic_evaluation/volatility.json`

**轮询**：
```bash
for i in $(seq 1 20); do
  sleep 60; tail -n 3 /home/wlia0047/hj82_scratch2/wenyu/logs/stage4_eval.log
  if grep -q "wrote →" /home/wlia0047/hj82_scratch2/wenyu/logs/stage4_eval.log; then break; fi
done
```

## 验证所有 stage 完成

```bash
ls -la /home/wlia0047/ar57/wenyu/PersoanlQuery/result/*/*.json
```

期望产物：
- `result/product_attributes.json`（Step 1）
- `result/03_gaussian/uid_to_sentences.pkl`（Step 2）
- `result/syntax_style_encoder/user_syntax_gaussian.json`（Stage 1）
- `result/syntax_style_encoder/syntax_exclusive_distance.json`（Stage 1）
- `result/gen_query/pool.json`（Stage 2）
- `result/syntax_style_encoder/stage8_5_selection_stats.json`（Stage 3）
- `result/syntactic_evaluation/retrieval_summary.json`（Stage 4）
- `result/syntactic_evaluation/volatility.json`（Stage 4）

## 规则遵守

- Rule 3：参数硬编码，不传参
- Rule 7：`pq_env` 解释器
- Rule 8：`nohup ... &` 后台运行，禁止 sbatch/srun
- Rule 9：`tail` + `grep` 轮询，禁止 `sleep X; tail`
- Rule 10：cache/log 写 `/home/wlia0047/hj82_scratch2/wenyu/`，结果写 `result/`
- Rule 11：cwd = 项目根
- Rule 12：脚本在功能目录，不写到 scratch2
- Rule 13：`result/` 只放 JSON/NPZ/CSV/log
- Rule 16：每个功能目录单一 JSON

## 故障排查

| 症状 | 排查 |
|------|------|
| Stage 1/2 卡住 | `tail vllm_server.log`，确认 vLLM 已加载 |
| Stage 2 strict=0 | 检查 `count_attrs_covered` 是否正确处理 "No"/"Yes" 否定属性 |
| Stage 2 d_self_pass=0 | 检查 `t_eucl` 是否为负（shared zone 用户），负数阈值应跳过 filter |
| Stage 3 OOM | spaCy n_process 减小 |
| Stage 4 GPU OOM | SPLADE 用 chunked-stream（默认） |

## 完成后

输出任务摘要 + 6 stage 状态 + 关键统计，并以"当前任务已完成，请做下一个任务的指示。"结尾。
