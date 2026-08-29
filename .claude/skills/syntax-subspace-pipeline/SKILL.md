---
name: syntax-subspace-pipeline
description: 跑全 5-stage Syntax Subspace 流水线。Stage 1 默认 N=5 attrs + 5-layer strict filter (attrs/invalid/1st-person/emoji/self-talk/length≤60)。LLM pool → spaCy 特征 → per-user Gaussian → Mahalanobis 选 query → MiniLM/BM25 评估 + 波动率标定
---

# Syntax Subspace Pipeline — 5 Stages

## 何时用

完整端到端跑通 Syntax Subspace 流水线:从用户评论 → 句法特征 → per-user Gaussian → Mahalanobis 选 query → MiniLM/BM25 评估 + 波动率标定。

**输入**:
- `scratch2/wenyu/gaussian_vades/stage8_5_asins.json`(target users + ASIN,1619 个)
- `result/product_attributes.json`(从 `data/meta_Baby_Products_2023.jsonl.gz` 抽取的 per-ASIN 属性字典,Stage 1 取 top-N=5 attrs)

**输出**:`/home/wlia0047/ar57/wenyu/PersoanlQuery/result/<dir>/<name>.json`(最终聚合结果,按 Rule 13/16 写到 result/)

**预计耗时**:Stage 1(LLM 5-10min) → Stage 2-3(spaCy CPU 30min) → Stage 4(CPU 5min) → Stage 5(GPU MiniLM 30min)。总计 ~1.5h。

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

需 ≥ 20GB free(MiniLM 6GB + spaCy 小 + vLLM 14GB)。

### 5. 输入数据

```bash
ls -la /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json
```

不存在则需先跑 `attribute_extraction/build_query_records.py` 生成上游数据。

## 路径约定(Rule 13/16)

- **Inputs**(`/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage8_5_asins.json` + `data/`):只读
- **Intermediate cache**(`/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/stage7b_query_features.jsonl.gz` + `stage8_5_selection.json` + `stage8_5_retrieval_per_query.json`):下游 stage 读取
- **Final results**(`/home/wlia0047/ar57/wenyu/PersoanlQuery/result/<dir>/<name>.json`):
  - `result/gen_query/pool.json`
  - `result/gaussian/user_gaussians.json`
  - `result/syntactic_evaluation/retrieval_summary.json`
  - `result/syntactic_evaluation/volatility.json`

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

### Stage 4: Mahalanobis select(Stage 4 of 5)

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
  select_query/syntax_subspace_select.py --stage select \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage4_select.log 2>&1 &
```

**轮询**:
```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 30; tail -n 3 /home/wlia0047/hj82_scratch2/wenyu/logs/stage4_select.log
  if grep -qE "wrote →|Stage 4" /home/wlia0047/hj82_scratch2/wenyu/logs/stage4_select.log; then break; fi
done
```

**完成标志**:`stage4_select.log` 含 `wrote → ...scratch2/stage8_5_selection.json`(intermediate cache)

### Stage 5: retrieval + volatility(Stage 5 of 5,GPU MiniLM)

Part A: Stage 4 选出的 {selected, random, farthest} 三组查询全量跑 BM25 + MiniLM。
Part B: 同一个脚本末尾自动跑 V_low / V_user / V_high 波动率标定。

```bash
cd /home/wlia0047/ar57/wenyu/PersoanlQuery
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
  syntactic_evaluation/syntax_subspace_retrieval.py --stage retrieval \
  > /home/wlia0047/hj82_scratch2/wenyu/logs/stage5_retrieval.log 2>&1 &
```

**轮询**(Stage 5 较慢 ~30min):
```bash
for i in $(seq 1 20); do
  sleep 90; tail -n 3 /home/wlia0047/hj82_scratch2/wenyu/logs/stage5_retrieval.log
  if grep -qE "wrote →" /home/wlia0047/hj82_scratch2/wenyu/logs/stage5_retrieval.log; then break; fi
done
```

**完成标志**:`stage5_retrieval.log` 含两条 `wrote →`:
- `result/syntactic_evaluation/retrieval_summary.json`(Part A)
- `result/syntactic_evaluation/volatility.json`(Part B)

## 验证所有 stage 完成

```bash
ls -la /home/wlia0047/ar57/wenyu/PersoanlQuery/result/*/*.json
```

期望产物:
- `result/gen_query/pool.json`(Stage 1)
- `scratch2/stage7b_query_features.jsonl.gz`(Stage 2 intermediate)
- `result/gaussian/user_gaussians.json`(Stage 3)
- `scratch2/stage8_5_selection.json` + `scratch2/stage8_5_selection_stats.json`(Stage 4 intermediate)
- `result/syntactic_evaluation/retrieval_summary.json`(Stage 5 Part A)
- `result/syntactic_evaluation/volatility.json`(Stage 5 Part B)

## 规则遵守(CLAUDE.md)

- Rule 3:所有参数硬编码,运行统一 `python <script> [--stage X]`(无参数传入)
- Rule 7:只用 `pq_env` (`/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`)
- Rule 8:所有 stage 用 `nohup ... &` 后台运行,**禁止 sbatch/srun**
- Rule 9:用 `tail` + `grep` 轮询完成标志,**禁止 `sleep X; tail`** 长休眠
- Rule 10:中间 cache + log 写 `/home/wlia0047/hj82_scratch2/wenyu/`,**最终聚合结果写 `/home/wlia0047/ar57/wenyu/PersoanlQuery/result/<dir>/`**
- Rule 11:cwd = `/home/wlia0047/ar57/wenyu/PersoanlQuery`
- Rule 12:脚本留在项目目录(gen_query/gaussian/select_query/syntactic_evaluation/),不写到 scratch2
- Rule 13:`result/` 只放 JSON/NPZ/CSV/log 产物,无 .py/.sh
- Rule 16:每个功能目录在 result/ 下对应单一 JSON(允许 stage 唯一 JSON + summary 配对)

## 故障排查

| 症状 | 排查 |
|------|------|
| Stage 1 卡住 | `tail vllm_server.log`,确认 vLLM 已加载 Qwen2-7B |
| Stage 2 OOM | spaCy n_process=8 占用过大,可改 `n_process=4` |
| Stage 3 找不到 review | 确认 `data/Baby_Products_2023.jsonl.gz` 存在 |
| Stage 5 GPU OOM | 减小 batch_size,或单独跑 BM25 不用 MiniLM |
| vLLM 起不来 (flashinfer JIT nvcc 错) | `export VLLM_USE_FLASHINFER_SAMPLER=0` + `export VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS="..."` 后再启 vLLM |
| pq_env 找不到 | `ls /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`,确认环境存在 |

## 完成后

按 AGENTS.md 规则 4 输出任务摘要 + 5 stage 状态 + 关键统计(选 query 数 / rank-1 hit rate / V_user 中位数等),并以"当前任务已完成,请做下一个任务的指示。"结尾。