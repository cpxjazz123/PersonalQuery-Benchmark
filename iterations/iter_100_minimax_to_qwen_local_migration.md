# iter #100 — MiniMax → 本地 Qwen-7B/8B 推理迁移

## 背景

`PersoanlQuery/llm_client.py` 原先仅注册 `MiniMaxAnthropicClient` 与
`MiniMaxIOAnthropicClient` 两个远程客户端，通过 Anthropic SDK 调用
`api.minimaxi.com/anthropic` / `api.minimax.io/anthropic`。本次按用户
需求下线所有远程 LLM 后端，把 LLM 改造为本地 Qwen-7B/8B 量级模型推理。

## 模型权重

默认路径（已确认存在）：

```
/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct
```

可通过 `QWEN_MODEL_PATH` 环境变量覆盖，指向其他本地权重目录或 HF repo id。

## 文件改动一览

| 文件 | 改动 |
| --- | --- |
| `/fs04/ar57/wenyu/CLAUDE.md` | Rule 9 重写：LLM 后端白名单改为「仅本地 Qwen」；删除 MiniMax / OpenAI / Anthropic 等远程后端条目 |
| `/fs04/ar57/wenyu/PersoanlQuery/llm_client.py` | 删除 `MiniMaxAnthropicClient` / `MiniMaxIOAnthropicClient` 与全部 SSH SOCKS 网络补丁；新增 `QwenLocalClient`（transformers / vllm 双后端、线程安全单例、chat-template 拼接、空响应 60s 重试、OOM/RuntimeError 30s 重试、保持与原 MiniMax 客户端同构的 `call` / `call_with_thinking` / `call_with_cache` 接口） |
| `02_writing_analysis/common/extract_errors_common.py` | `from llm_client import QwenLocalClient`；`BatchErrorExtractor.__init__` 类型注解同步；`use_minimaxio` 分支删除，统一实例化 `QwenLocalClient()` |
| `04_query/common/llm_runner.py` | 全局变量 `_minimax_client` → `_qwen_client`；`load_minimax_client()` 内部改为实例化 `QwenLocalClient()`（函数名保留以兼容 04_query 内部 caller） |
| `14_llm_rerank/common/rerank_config.json` | `client_class`: `MiniMaxAnthropicClient` → `QwenLocalClient`；`model`: `"M2.5"` → `"/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"` |
| `14_llm_rerank/14_llm_rerank/common/rerank_config.json` | 同上 |

## 接口契约（保持兼容）

`QwenLocalClient` 暴露与原 MiniMax 客户端同构的方法签名，便于 14_llm_rerank
通过 `cfg["llm"]["client_class"]` 切换后端，业务代码不动：

```python
QwenLocalClient(model: str = DEFAULT_QWEN_MODEL_PATH)

call(prompt: str, max_tokens: int = 4096, temperature: Optional[float] = None,
     max_retries: int = 350) -> str

call_with_thinking(prompt: str, max_tokens: int = 8192,
                   temperature: Optional[float] = None,
                   max_retries: int = 350) -> tuple[str, str]
# 返回 (thinking_text, text_content)
# 本地 Qwen2 不原生支持 thinking，统一 thinking_text=""

call_with_cache(system_base: str, user_content: str,
                max_tokens: int = 4096,
                temperature: Optional[float] = None,
                max_retries: int = 350,
                retry_on_empty_response: bool = True,
                stream: bool = False) -> tuple[str, dict]
# 返回 (text_content, usage_dict)
# usage_dict: {"cache_creation_input_tokens": 0,
#               "cache_read_input_tokens": 0,
#               "input_tokens": int,
#               "output_tokens": int}
```

## 运行环境要求

- Python 3.10+
- PyTorch ≥ 2.1（CUDA 11.8+）
- transformers ≥ 4.40（`trust_remote_code=True` 需要模型仓支持）
- 可选：vllm ≥ 0.5（设 `QWEN_BACKEND=vllm` 切换）
- 单 GPU ≥ 16GB（Qwen2-7B bf16 ≈ 15GB；CPU 推理可行但极慢）

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `QWEN_MODEL_PATH` | `/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct` | 模型权重路径或 HF repo id，覆盖构造参数 |
| `QWEN_BACKEND` | `transformers` | `transformers` 或 `vllm` |
| `QWEN_DTYPE` | auto（CUDA → bf16） | `bfloat16` / `float16` / `float32` |
| `QWEN_DEVICE_MAP` | `auto` | `auto` / `cuda:0` / `cpu` |
| `QWEN_MAX_MODEL_LEN` | 模型 config | vllm 模式下生效 |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.9` | vllm 模式下生效 |
| `QWEN_ENABLE_THINKING` | (未使用) | Qwen3 chat template 可选参数占位 |

## 验证

- `python3 -m py_compile llm_client.py` → OK
- 全量 py_compile（修改的 8 个 .py 文件）→ ALL_PY_COMPILE_OK
- import 时类已注册：`from llm_client import QwenLocalClient` 正常
- rerank_config.json 通过 `json.load` 校验
- 模型权重目录经 `ls -la` 确认存在 `config.json` /
  `model-00001-of-00004.safetensors` 等 Qwen2 标准产物

## 后续 TODO

- 实际跑通一次 stage 14_llm_rerank 的 rerank 脚本（需要 GPU）
- 若 `transformers` 报 `Qwen2ForCausalLM` 未注册，确认 `transformers ≥ 4.40` 或
  加载时正确传入 `trust_remote_code=True`（已实现）
- 若显存不足，考虑切换 `QWEN_DTYPE=float16` 或量化（目前未实现，需扩展 `QwenLocalClient`）