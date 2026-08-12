# Iteration #54 — vLLM 本地部署 + Stage 0/1/3/4 端到端跑通

**日期**: 2026-07-20 → 2026-07-21
**角色**: NLP/IR 专业审稿人 + 平台工程师
**scope**: 用本地 vLLM 替换 MiniMax API，跑通 Stage 0/1/3/4 端到端，生成 100-user query 数据集

---

## §A 背景与动机

### 审稿视角：外部 LLM API 的可复现性危机

论文 §3.2-3.4 的所有 query 生成、writing error injection、LLM rerank 都强依赖 LLM 调用。如果继续用 MiniMax API：
1. **API 不稳定**：模型版本、Safety filter、rate limit 都可能变 → 实验无法长期复现
2. **成本不可控**：每轮 stage 04 query generation 跑 30000 user × 10 candidate ≈ 30 万次 LLM 调用
3. **论文评审质疑点**："How was the query set generated?" — 用外部闭源模型是关键 reproducibility concern

**结论**：必须替换成本地开源模型。Qwen2.5-7B-Instruct 是当前 7B 量级中文/英文任务 SOTA。

---

## §B 本轮实施

### 1. vLLM 部署 (GPU 3)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --enforce-eager
```

**已知问题修复**:
- `libnvrtc.so.13` 缺失 → `pip install nvidia-cuda-nvrtc-cu12`
- `Backend TORCH_SDPA must be registered` → 移除 `--attention-backend TORCH_SDPA` flag
- `flashinfer ModuleNotFoundError` → 重装 `flashinfer-python==0.6.13`
- `/usr/local/cuda/bin/nvcc: No such file or directory` → `PATH=/usr/local/cuda-12.5/bin:$PATH` (系统有 cuda-12.5/13.0，symlink 指错)

**结果**: ✅ Qwen2.5-7B-Instruct @ http://localhost:8000/v1

### 2. 新增 VLLMLocalClient

`/home/wlia0047/ar57/wenyu/PersoanlQuery/llm_client.py`:
- 新增 `class VLLMLocalClient` (200 行)
- 接口兼容 `MiniMaxAnthropicClient.call_with_cache()` → `(text, usage_dict)`
- 使用 `urllib.request` POST 到 `/v1/chat/completions`，零额外依赖

`/home/wlia0047/ar57/wenyu/PersoanlQuery/04_query/common/llm_runner.py`:
- 检测 `VLLM_USE_LOCAL=1` 环境变量
- 选择 `VLLMLocalClient` 替代 `MiniMaxAnthropicClient`
- `stream=not use_local` (vLLM 不需要 stream)

### 3. Stage 0/1/3/4 端到端

| Stage | 输入 | 输出 | 状态 |
|-------|------|------|------|
| 0 | HF `Baby_Products.jsonl` (2.8GB) | 88966 user review profiles | ✅ |
| 1 | Stage 0 + meta | 7681 matched users, attributes_Baby_Products.json | ✅ |
| 2 | Stage 1 | writing_error.json | ⏭️ 跳过 (用 MiniMax API，且 Stage 4 不需要) |
| 3 | Stage 1 + spaCy | user_average_syntax_depth.json (avg=7.73), attr_density_user_profiles.json | ✅ |
| 4 | Stage 1+3 + vLLM | 73 users × 10 candidates = 730 valid queries | ✅ |

### 4. 路径修复

Stage 3 主脚本写到 `05_syntactic_analysis/{Cat}/`，但 Stage 4 从 `03_syntactic_analysis/{Cat}/` 读：
```bash
cp 05_syntactic_analysis/Baby_Products/user_average_syntax_depth.json 03_syntactic_analysis/Baby_Products/
cp 05_syntactic_analysis/Baby_Products/attr_density_user_profiles.json 03_syntactic_analysis/Baby_Products/
```

### 5. 依赖补装

| 包 | 用途 |
|----|------|
| `ujson` | Stage 1 multiprocessing chunk parsing |
| `PySocks` | MiniMax API SOCKS tunnel |
| `spacy` | Stage 3 syntactic analysis |
| `en_core_web_sm` | spaCy English model |

### 6. common_utils.py 补丁

```python
# 兼容旧代码: from common_utils import log
log = log_with_timestamp
```

---

## §C 关键审稿观察

### 观察 1: Stage 4 73/100 成功率偏低

**问题**: 100 个用户中只有 73 个通过 5-attribute 验证（27% 失败率）。
**根因**: LLM 在 5-attribute 严格约束（每个 attr 值必须出现恰好 1 次）下经常出错：
- 漏写 attribute（如 `A3='15.99' 出现 0 次; A4='Large' 出现 0 次`）
- 重复 attribute（如 `A6='PVC' 出现 2 次`）

**审稿意义**: 论文 §3.2 报告 query 数量时未明确报告"经过多少轮 rejection 才得到 N 个 valid queries"。**审稿建议**：论文应报告 mean attempts per valid query，作为 LLM controllability 的可复现性指标。

### 观察 2: Stage 2 跳过 = 论文 §2.2 writing error analysis 无 ground truth

跳过 Stage 2 (writing_analysis) 是工程取舍，但严格来说论文 §2.2 "writing style 包含错别字、语法错误" 的描述依赖此 Stage 输出。**审稿意义**：未来需要补跑 Stage 2 (改用 vLLM 替换 MiniMax API) 才能完整论文 ground truth。

### 观察 3: vLLM 8192 max-model-len 可能不够

Stage 4 system prompt + 10-candidate user content ≈ 500-700 tokens，但 system_base 包含 100+ 行模板 instructions (~700 tokens)。8192 应该够，但如果是 Stage 09/12 更复杂的 prompt，需要调大 `--max-model-len`。

---

## §D 验证

| 检查项 | 结果 |
|--------|------|
| vLLM `/v1/models` 返回 `Qwen2.5-7B-Instruct` | ✅ |
| VLLMLocalClient smoke test (`2+2=?`) | ✅ `4. equals 4.` |
| Stage 0 → 88966 users | ✅ |
| Stage 1 → 7681 matched users, attrs JSON 11MB | ✅ |
| Stage 3 → user_average_syntax_depth.json 2.0MB | ✅ |
| Stage 3 → attr_density_user_profiles.json 8.2MB | ✅ |
| Stage 4 → 73 users × 10 valid queries | ✅ |

---

## §E 改动文件清单

```
PersoanlQuery/llm_client.py          (新增 VLLMLocalClient 200 行)
PersoanlQuery/04_query/common/llm_runner.py  (添加 VLLM_USE_LOCAL 检测)
PersoanlQuery/common_utils.py        (添加 log = log_with_timestamp 别名)
```

---

## §F 剩余任务 (Backlog 更新)

| 优先级 | 任务 | 依赖 |
|--------|------|------|
| P0 | Stage 0/1/3 跑 Grocery_and_Gourmet_Food, Pet_Supplies | review 数据已下载完 |
| P0 | Stage 4 跑 Grocery, Pet (用 vLLM) | Stage 0/1/3 完成 |
| P0 | Stage 2 改用 vLLM (替代 MiniMax API) | 改 extract_errors_common.py |
| P1 | Stage 06 3-category parallel builds (GPU 0/1/2) | 需要 retriever data |
| P1 | Stage 08 Δ Range analysis | 需要 Stage 06/07 完成 |
| P1 | Stage 10 BIC/AIC compute | 需要 Stage 10 latent representations |

---

## §G Git Commit

待合并到主分支后统一 commit。