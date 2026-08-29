---
name: syntax-subspace-pipeline
description: 跑全 5-stage Syntax Subspace 流水线。Stage 1 默认 N=5 attrs + 5-layer strict filter (attrs/invalid/1st-person/emoji/self-talk/length≤60)。LLM pool → spaCy 特征 → per-user Gaussian → v6m Mahalanobis strict alignment → 7-retriever 评估 (BM25/SPLADE/4 dense/ColBERTv2, NO rerank) + minilm-canonical sim09 volatility
---

# Syntax Subspace Pipeline — 5 Stages

## 何时用

完整端到端跑通 Syntax Subspace 流水线:从用户评论 → 句法特征 → per-user Gaussian → Mahalanobis 选 query → 7-retriever 评估 + 波动率标定。

**输入**:
- `scratch2/wenyu/gaussian_vades/stage8_5_asins.json`(target users + ASIN,1619 个)
- `result/product_attributes.json`(从 `data/meta_Baby_Products_2023.jsonl.gz` 抽取的 per-ASIN 属性字典,Stage 1 取 top-N=5 attrs)

**输出**:`/home/wlia0047/ar57/wenyu/PersoanlQuery/result/<dir>/<name>.json`(最终聚合结果,按 Rule 13/16 写到 result/)

**预计耗时**:Stage 1(LLM 5-10min) → Stage 2-3(spaCy CPU 30min) → Stage 4(CPU 5min) → Stage 5(GPU MiniLM+SPLADE+ColBERTv2 ~10min)。总计 ~1.5h。

**用户指令 2026-08-28 锁定配置**:
- `N_INPUT = 5`(从 `product_attributes.json` 取 top-5 **非数值** attrs)
- **属性数值过滤**:value 含数字字符(`5.02 x 4.04`、`3.87 ounces`、`DXR-8`)的属性被排除。若某 ASIN 去掉数值属性后 <5 个,直接过滤该 ASIN。
- **5-layer strict filter**(Stage 1 自动打标 `strict: true/false`,下游 Stage 4/5 只取 strict):
  1. `attrs_covered == N`(5/5 属性全部提及)
  2. `!has_invalid_punct`(开头不是 `here:`/`brand:`/`attribute:` 等元数据格式)
  3. `has_first_person`(`I`/`my`/`hoping to find` 等第一人称标识)
  4. `!has_emoji`(Unicode emoji block 0x1F000-0x1FFFF, 0x2600-0x27BF 等)
  5. `!has_self_talk`(`can someone help`、`thanks!`、`let's try again`、`just those exact attributes` 等自言自语)
  6. `len(query) ≤ MAX_QUERY_TOKENS=60`(避免 80+ token 长尾失控污染 strict 池)

## 规范目录结构(2026-08-29 整理后)

**每个功能目录按职责划分,无 ablation/legacy 残留**。所有冗余脚本已归档删除,git history 保留可查:

- `attribute_extraction/` `gaussian/` `gen_query/` `select_query/` `syntactic_evaluation/`:主脚本 + (按需)数据准备脚本
- `common/` / `syntactic_analysis/`:共享底座或预留目录

```
attribute_extraction/   extract_product_attrs.py                  (Step 1: product attrs + select_top_attrs 工具)
common/                 syntax_subspace_utils.py                  (318d features + paths)
gaussian/               syntax_subspace_user_gaussians.py         (Stage 3: per-user Gaussian)
                        build_user_cohort.py                      (Steps 2-4: review scan + query_records + stage8_5 cohort)
gen_query/              syntax_subspace_pool_regen.py             (Stage 1+2: pool + features, --stage)
select_query/           syntax_subspace_select_v6m_strict_alignment.py  (Stage 4: v6m main)
syntactic_evaluation/   syntax_subspace_retrieval_unified.py      (Stage 5: 7-retriever, NO rerank)
syntactic_analysis/     (空)
```

**目录职责(2026-08-29 拆分后)**:
- `attribute_extraction/`: 仅做商品属性抽取(Step 1)。`select_top_attrs()` 是工具函数,供下游 `gaussian/build_user_cohort.py` 的 Step 3/4 共用。
- `gaussian/`: 用户相关操作。`build_user_cohort.py`(Steps 2-4: 评论扫描 + query_records + cohort)负责数据准备,`syntax_subspace_user_gaussians.py`(Stage 3)负责 per-user Gaussian 拟合,二者解耦但都在用户域。
- **运行顺序**: 先 `python attribute_extraction/extract_product_attrs.py`(Step 1) → 再 `python gaussian/build_user_cohort.py`(Steps 2-4)。`build_user_cohort.py` 会自动检测 `result/product_attributes.json`,缺失则报错。

## 前置检查

### 1. Python 环境(rule 7)

```bash
PY=/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python
$PY --version  # 应输出 Python 3.10.x
```

### 2. 工作目录(rule 11)

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
```

### 3. vLLM HTTP server(:8800)

```bash
curl -s -m 3 http://localhost:8800/v1/models | head -3
```

若空响应,**启动 vLLM**(后台,nohup):
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="top_k_top_p_sampling_from_logits,top_k_mask_logits,top_p_renorm_probs,sampling,renorm"
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct \
  --port 8800 --host 0.0.0.0 --gpu-memory-utilization 0.85 --enforce-eager \
  > /home/wlia0047/hj82_scratch2/wenyu/tmp/vllm_server.log 2>&1 &
```

等 3-5 分钟,轮询 `curl http://localhost:8800/v1/models` 直到返回模型列表。

### 4. GPU

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```

需 ≥ 20GB free(MiniLM 6GB + SPLADE 13GB chunked-stream + ColBERTv2 8GB + vLLM 14GB)。

### 5. 输入数据

```bash
ls -la /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json
```

不存在则需先跑两段上游数据构造:
```bash
# Step 1 — 商品属性抽取
/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
  attribute_extraction/extract_product_attrs.py

# Steps 2-4 — 用户 cohort 构造(扫描 reviews + query_records + stage8_5_asins)
/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
  gaussian/build_user_cohort.py
```

## 路径约定(Rule 13/16)

- **Inputs**(`/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json` + `data/`):只读
- **Intermediate cache**(`/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7b_query_features.jsonl.gz` + `stage8_5_selection.json` + `stage8_5_retrieval_per_query.json` + `multiretrieval_embeds/<retr_name>/`):下游 stage 读取
- **Final results**(`/home/wlia0047/ar57/wenyu/PersoanlQuery/result/<dir>/<name>.json`):
  - `result/gen_query/pool.json`
  - `result/gaussian/user_gaussians.json`
  - `result/syntactic_evaluation/retrieval_summary.json`(7 retriever volatility only)
  - `result/syntactic_evaluation/volatility.json`(canonical BM25+MiniLM slice)

## 跑 5 stages

### Stage 1: vLLM pool generation(Stage 1 of 5)

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
  gen_query/syntax_subspace_pool_regen.py --stage pool_regen \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage1_pool_regen.log 2>&1 &
```

**配置**(`common/syntax_subspace_utils.py`):
- `N_INPUT = 5`(从 `product_attributes.json` 取 top-5 attrs)
- `K_POOL = 50`(每 ASIN 50 个候选 query)
- `TEMP = 0.7`,`MAX_TOKENS = 120`(vLLM 生成上限)
- `MAX_QUERY_TOKENS = 60`(strict filter 长度阈值)
- 5-layer strict filter 自动应用,见 "何时用" 段落

**轮询**(rule 9):
```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 60; tail -n 3 /home/wlia0047/hj82_scratch2/wenyu/logs/stage1_pool_regen.log
  if grep -q "wrote →" /home/wlia0047/hj82_scratch2/wenyu/logs/stage1_pool_regen.log; then break; fi
done
```

**完成标志**:`stage1_pool_regen.log` 含 `wrote → ...result/gen_query/pool.json` + 6 行 filter breakdown(显示每个 filter 的通过率)

**期望输出**:严格率约 50-65%(根据经验:N=5 first-person 下,严格率 ~60%;N=10 降到 21%;N=7 居中 ~37%)。如果 strict 远低于 50% 说明 5-layer filter 太严,可临时调小 `MAX_QUERY_TOKENS` 或放宽 self-talk pattern。

### Stage 2: spaCy features(Stage 2 of 5)

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
  gen_query/syntax_subspace_pool_regen.py --stage features \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage2_features.log 2>&1 &
```

**轮询**:
```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 60; tail -n 3 /home/wlia0047/hj82_scratch2/wenyu/logs/stage2_features.log
  if grep -q "saved cache:" /home/wlia0047/hj82_scratch2/wenyu/logs/stage2_features.log; then break; fi
done
```

**完成标志**:`stage2_features.log` 含 `saved cache: ...stage7b_query_features.jsonl.gz`(intermediate cache,在 scratch2)

### Stage 3: per-user Gaussians(Stage 3 of 5)

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
  gaussian/syntax_subspace_user_gaussians.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage3_user_gaussians.log 2>&1 &
```

**轮询**:
```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 60; tail -n 3 /home/wlia0047/hj82_scratch2/wenyu/logs/stage3_user_gaussians.log
  if grep -q "wrote →" /home/wlia0047/hj82_scratch2/wenyu/logs/stage3_user_gaussians.log; then break; fi
done
```

**完成标志**:`stage3_user_gaussians.log` 含 `wrote → ...result/gaussian/user_gaussians.json`

### Stage 4: v6m Mahalanobis strict alignment(Stage 4 of 5)

v6m 是当前 main pipeline:PCA48 whitening + R_99 from real historical + L2 margin。
无参数(规则 3),直接跑:

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
  select_query/syntax_subspace_select_v6m_strict_alignment.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage4_select.log 2>&1 &
```

**轮询**:
```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 30; tail -n 3 /home/wlia0047/hj82_scratch2/wenyu/logs/stage4_select.log
  if grep -qE "wrote →|Stage 4" /home/wlia0047/hj82_scratch2/wenyu/logs/stage4_select.log; then break; fi
done
```

**完成标志**:`stage4_select.log` 含 `wrote → ...scratch2/stage8_5_selection.json`(intermediate cache)+ `stage8_5_selection_stats.json`(~6KB 统计,6357 strict personalized + 15318 no_strict_candidate)

### Stage 5: 7-retriever 评估 + minilm-canonical sim09 volatility(Stage 5 of 5)

**集成 7 个 retriever(NO rerank)**:
1. **BM25**(lexical_sparse)— `bm25s` lib, k1=1.5 b=0.75
2. **SPLADE**(learned_sparse)— `naver/splade-cocondenser-ensembledistil`,chunked-stream encoding 避免 13GB OOM
3. **MiniLM**(dense_biencoder 384d, 22M)— `sentence-transformers/all-MiniLM-L6-v2`
4. **MPNet**(dense_biencoder 768d, 110M)— `sentence-transformers/all-mpnet-base-v2`
5. **BGE-base-v1.5**(dense_biencoder 768d, 110M)— `BAAI/bge-base-en-v1.5`
6. **GTE-base**(dense_biencoder 768d, 110M)— `thenlper/gte-base`
7. **ColBERTv2**(late_interaction, 768→128 投影)— `colbert-ir/colbertv2.0`

**sim09 锚定**:所有 retriever 的 query-query 相似度阈值统一参考 minilm 384d dense embedding(用户指令 2026-08-29)。BM25/SPLADE 无 dense query embed,自动借用 minilm;这样 7 个 retriever 的 volatility 数值可比(共享同一聚类)。

无参数(规则 3),直接跑:

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
  syntactic_evaluation/syntax_subspace_retrieval_unified.py \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage5_retrieval.log 2>&1 &
```

**轮询**(Stage 5 较慢 ~8min for 7 retrievers;vLLM 不需要):
```bash
for i in $(seq 1 30); do
  sleep 60; tail -n 3 /home/wlia0047/hj82_scratch2/wenyu/logs/stage5_retrieval.log
  if grep -qE "wrote →" /home/wlia0047/hj82_scratch2/wenyu/logs/stage5_retrieval.log; then break; fi
done
```

**完成标志**:`stage5_retrieval.log` 含两条 `wrote →`:
- `scratch2/.../stage8_5_retrieval_per_query.json`(7.9MB,signature=`1442eea563ce0a2d`,per-query rank/RR/hit@1/5/10)
- `result/syntactic_evaluation/retrieval_summary.json`(per-retriever sim09 volatility,7 retrievers,`config.sim09_reference = "minilm"`)
- `result/syntactic_evaluation/volatility.json`(canonical BM25+MiniLM slice)

**Cache 命中**:如果 `stage8_5_selection.json` 的 signature 未变且 `stage8_5_retrieval_per_query.json` 已存在,Stage 5 直接读 cache + 重算 volatility,只需 ~2 秒。

## 验证所有 stage 完成

```bash
ls -la /home/wlia0047/ar57/wenyu/PersoanlQuery/result/*/*.json
```

期望产物:
- `result/gen_query/pool.json`(Stage 1)
- `scratch2/stage7b_query_features.jsonl.gz`(Stage 2 intermediate)
- `result/gaussian/user_gaussians.json`(Stage 3)
- `scratch2/stage8_5_selection.json` + `scratch2/stage8_5_selection_stats.json`(Stage 4 intermediate)
- `result/syntactic_evaluation/retrieval_summary.json`(Stage 5 Part B,7 retriever volatility)
- `result/syntactic_evaluation/volatility.json`(Stage 5 canonical BM25+MiniLM slice)

## Stage 5 期望输出参考(v7 on v6m strict alignment, 1095 ASINs sim09)

| retriever | Hit@1_flip | Hit@5_flip | Hit@10_flip | Hit@20_flip | RR_Std |
|---|---:|---:|---:|---:|---:|
| bm25 | 4.70% | 7.68% | 9.06% | 8.82% | 0.0499 |
| splade | 3.95% | 4.94% | 5.80% | 4.60% | 0.0419 |
| minilm | 2.71% | 2.98% | 4.76% | 4.90% | 0.0272 |
| mpnet | 2.75% | 4.61% | 5.23% | 6.66% | 0.0325 |
| bge_base_v15 | 3.01% | 4.56% | 5.29% | 6.03% | 0.0339 |
| gte_base | 4.05% | 5.58% | 6.85% | 7.59% | 0.0420 |
| colbertv2 | 2.40% | 4.99% | 6.21% | 8.14% | 0.0347 |

**关键观察**:
- BM25 在 minilm-sim09 聚类下 Hit@10_flip 最高(9.06%)— 同义改写最脆弱
- minilm 综合最稳(RR_Std=0.0272)— 自我引用最自洽
- colbertv2 Hit@1_flip=2.40%(rank-1 最稳)但 Hit@20_flip=8.14%(top-20 最不稳)— max-sim 对 rank-1 边界敏感,top-20 散开
- gte_base Hit@K_flip 随 K 单调递增(4.05% → 5.58% → 6.85% → 7.59%)— 检索强但不稳

## 规则遵守(CLAUDE.md)

- Rule 3:所有参数硬编码到脚本,运行统一 `python <script>`(可选 `--stage X` 仅 Stage 1/2)。Stage 3/4/5 无参数传入
- Rule 7:只用 `pq_env` (`/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`)
- Rule 8:所有 stage 用 `nohup ... &` 后台运行,**禁止 sbatch/srun**
- Rule 9:用 `tail` + `grep` 轮询完成标志,**禁止 `sleep X; tail`** 长休眠
- Rule 10:中间 cache + log 写 `/home/wlia0047/hj82_scratch2/wenyu/`,**最终聚合结果写 `/home/wlia0047/ar57/wenyu/PersoanlQuery/result/<dir>/`**
- Rule 11:cwd = `/home/wlia0047/ar57/wenyu/PersoanlQuery`
- Rule 12:脚本留在 6 个项目目录(attribute_extraction/common/gaussian/gen_query/select_query/syntactic_evaluation/),不写到 scratch2
- Rule 13:`result/` 只放 JSON/NPZ/CSV/log 产物,无 .py/.sh
- Rule 14:每个功能目录 1 个主脚本(2026-08-29 整理后)
- Rule 16:每个功能目录在 result/ 下对应单一 JSON(允许 stage 唯一 JSON + summary 配对)

## 故障排查

| 症状 | 排查 |
|------|------|
| Stage 1 卡住 | `tail vllm_server.log`,确认 vLLM 已加载 Qwen2-7B |
| Stage 2 OOM | spaCy n_process=8 占用过大,可改 `n_process=4` |
| Stage 3 找不到 review | 确认 `data/Baby_Products_2023.jsonl.gz` 存在 |
| Stage 5 GPU OOM | 减小 batch_size;SPLADE 用 chunked-stream(默认开启) |
| Stage 5 dense retriever stale cache | 删除 `scratch2/multiretrieval_embeds/<retr_name>/query_embeds.npy` 让脚本重编码 |
| vLLM 起不来 (flashinfer JIT nvcc 错) | `export VLLM_USE_FLASHINFER_SAMPLER=0` + `export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="..."` 后再启 vLLM |
| pq_env 找不到 | `ls /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`,确认环境存在 |

## 完成后

按 AGENTS.md 规则 4 输出任务摘要 + 5 stage 状态 + 关键统计(选 query 数 / per-retriever Hit@1_flip / Hit@10_flip / RR_Std 等),并以"当前任务已完成,请做下一个任务的指示。"结尾。
