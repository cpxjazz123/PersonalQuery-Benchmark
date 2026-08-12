# iter #55 (完成): Stage 4 vLLM 端到端跑通 (Pet + Grocery)

**目标**: Stage 4 query generation (10 queries/user × 100 users) 走本地 vLLM (Qwen2.5-7B-Instruct @ GPU 3, max_model_len=8192), 不依赖 MiniMax API.

## 关键根因 + 修复

### 根因: `max_tokens=32768` 超过 vLLM `max_model_len=8192`

`PersoanlQuery/04_query/common/llm_runner.py` line 83 硬编码 `max_tokens=32768`,
原为 MiniMax API 设计. 改 vLLM 后所有 100 个用户调用都返回 `HTTP 400 Bad Request`.

### Fix (commit `9526155`)

```python
# llm_runner.py:80-91
response, cache_info = _minimax_client.call_with_cache(
    system_base=system_base,
    user_content=prompt,
    # Local vLLM (Qwen2.5-7B with --max-model-len 8192): cap output at 4096
    # tokens to leave headroom for the ~760-token system+user prompt. The
    # original 32768 was meant for the MiniMax API which supports much
    # larger contexts; sending it to vLLM causes a 400 Bad Request.
    max_tokens=4096 if use_local else 32768,
    temperature=0.8,
    retry_on_empty_response=False,
    stream=not use_local,
)
```

- `use_local` = `os.environ.get("VLLM_USE_LOCAL", "")` ∈ {"1","true","yes","on"}
- vLLM 模式: max_tokens=4096, stream=False
- MiniMax API: max_tokens=32768, stream=True (与原行为一致)

## Stage 4 跑通结果

| Category | Users | Success | Failed | Time | Output |
|----------|------:|--------:|-------:|-----:|--------|
| Pet_Supplies | 100 | 97 | 3 | 226s | `06_query/Pet_Supplies/query_by_syntax_depth_no_depth_check_10.json` (617 KB) |
| Grocery_and_Gourmet_Food | 100 | 89 | 11 | 280s | `06_query/Grocery_and_Gourmet_Food/query_by_syntax_depth_no_depth_check_10.json` (573 KB) |

## 失败原因分析 (reviewer perspective)

Pet 3% / Grocery 11% 失败: 全部因 5-attribute validation 不通过 — LLM 生成的 query 没有用上指定的 5 个属性, 或用了多次, 或混入额外属性值. 失败用户的 query_by_syntax_depth_no_depth_check_10.json 不写记录 (代码逻辑: 只有 ≥1 有效候选才计入 records).

## 配套改动 (同 commit)

1. **`PersoanlQuery/llm_client.py`** +200 行: `VLLMLocalClient` 类 (OpenAI 兼容 HTTP, `/v1/chat/completions`, 自动从 `VLLM_MODEL` 环境变量读模型名, 默认 `Qwen2.5-7B-Instruct`).
2. **`06_retrieval/06_build_retriever_indices_{Cat}.py`** ×3: 加 `SKIP_COLBERTV2=1` 环境变量绕过 colbertv2 torch 扩展编译失败 (thrust/complex.h 缺失). 跳过 colbertv2 后, 6 retrievers 仍可用: bge-small / bge-base / contriever / e5 / bm25 / splade.
3. **`PersoanlQuery/run_stage4_vllm.sh`** (新): wrapper 脚本 — 自动从 Stage 3 复制到 03_syntactic_analysis, swap query_config 100-user, 用 vLLM 跑, 跑完还原 config. 日志输出到 `/home/wlia0047/ar57/wenyu/result/personal_query/logs/` (符合 Rule 11).

## 复现命令

```bash
# 1. 启动 vLLM (GPU 3, 8K context):
vllm serve /home/wlia0047/ar57_scratch/wenyu/hf_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
  --port 8000 --gpu 3 --max-model-len 8192 --served-model-name Qwen2.5-7B-Instruct

# 2. 跑 Stage 4 Pet 或 Grocery:
bash /home/wlia0047/ar57/wenyu/PersoanlQuery/run_stage4_vllm.sh Pet_Supplies
bash /home/wlia0047/ar57/wenyu/PersoanlQuery/run_stage4_vllm.sh Grocery_and_Gourmet_Food
```

## 后续 iter #56 计划

1. Stage 06: `SKIP_COLBERTV2=1 python3 06_build_retriever_indices_{Cat}.py` for 3 domains (Pet / Grocery / Baby) — 6 retrievers each.
2. Stage 07: 同样跳过 colbertv2-based noisy query cache.
3. Stage 08: Δ Range 评估 (BIC/AIC).
4. Commit wrapper to LIVE repo (currently only in worktree).

## Reviewer 观察

- Grocery 失败率比 Pet 高 8%: 可能因为 Grocery 产品属性更长 (e.g. "0.01 Ounces" 包含小数+单位, 5-attr validator 难以匹配). 后续可考虑放宽 A15 (weight) 验证规则.
- vLLM 8 worker 并发 100 用户 226s (Pet) / 280s (Grocery) — 平均 ~2.5s/user, 满足生产性能要求.