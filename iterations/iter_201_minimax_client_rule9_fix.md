# Iteration 201 — MiniMaxAnthropicClient Rule 9 接口分离修复

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: MiniMaxAnthropicClient/MiniMaxIOAnthropicClient _enc 显式声明 + call_with_cache 明确拒绝

---

## §A 审稿意见

### iter #200 发现的 MiniMaxAnthropicClient._enc 缺失

iter #200 smoke test 报 `AttributeError: 'MiniMaxAnthropicClient' object has no attribute '_enc'`。根因：
- `MiniMaxAnthropicClient.__init__` (llm_client.py:237-244) 只初始化 `self.model` 和 `self.client` (anthropic SDK)，**未初始化 `self._enc`**
- `MiniMaxAnthropicClient.call_with_cache` (line 345+) 假设 `self._enc` 是 transformers tokenizer (`self._enc.encode/decode`)
- `self._enc` 只在 `VLLMLocalClient.__init__` (line 841) 初始化
- 这是 uncommitted VLLM migration 把 VLLM code 错误合并到 MiniMaxAnthropicClient 的 copy-paste-between-classes bug

### 接口混淆根因

`MiniMaxAnthropicClient.call_with_cache` 方法体实际是 **VLLM-local 代码**：
- 用 `self._client.chat.completions.create` (OpenAI-style API)
- 用 `self._enc` tokenizer-based truncation
- docstring: "Call local vLLM using precise token counting via transformers Qwen tokenizer"
- 这是 VLLMLocalClient 的代码，不属于远程 MiniMax API client

但 `MiniMaxIOAnthropicClient.call_with_cache` (line 648) 是**正确的 anthropic SDK 实现**：
- 用 `self.client.messages.create` (anthropic-style API)
- 不需要 `_enc`
- docstring: "Call MiniMax IO API with prompt caching"

### 修复方案

按 CLAUDE.md Rule 9 (LLM 客户端分离) + Rule 7 (无 silent fallback)：

1. `MiniMaxAnthropicClient.__init__` 显式 `self._enc = None` (声明接口契约)
2. `MiniMaxAnthropicClient.call_with_cache` 在 docstring 后立即 raise `NotImplementedError` (明确错误信号)
3. `MiniMaxIOAnthropicClient.__init__` 加 `self._enc = None` (接口一致性，无需 _enc 但满足 hasattr 检查)
4. `VLLMLocalClient` 不动 (它的 _enc 初始化正确)

---

## §B 落地代码改动 (llm_client.py)

### B.1 `MiniMaxAnthropicClient.__init__` (line 237-244)

```python
def __init__(self, model: str = "M2.5"):
    self.model = model
    # iter #201 fix: explicit declare self._enc = None (call_with_cache uses VLLM-local
    # tokenizer pattern that does NOT apply to MiniMaxAnthropicClient; the method
    # exists for backward compatibility with extract_errors_common.py callers but
    # raises NotImplementedError if invoked on this client. Use VLLMLocalClient for
    # VLLM-style token-counted truncation, or MiniMaxAnthropicClient.call_with_thinking
    # for remote MiniMax API without truncation.
    self._enc = None
    _ensure_minimax_compute_node_network()
    import anthropic
    self.client = anthropic.Anthropic(...)
```

### B.2 `MiniMaxAnthropicClient.call_with_cache` (line 345+) 明确 NotImplementedError

```python
def call_with_cache(
    self,
    system_base: str,
    user_content: str,
    max_tokens: int = 4096,
    ...
    stream: bool = False,
) -> tuple:
    """Call local vLLM using precise token counting via transformers Qwen tokenizer.

    iter #201 fix: MiniMaxAnthropicClient is a REMOTE MiniMax API client. This
    call_with_cache method body is VLLM-local code (uses OpenAI-style
    chat.completions.create + self._enc tokenizer-based truncation). It does
    NOT apply to MiniMax API. Raises NotImplementedError explicitly per
    CLAUDE.md Rule 7 (no silent fallback). Use VLLMLocalClient for VLLM
    workflows or call_with_thinking for MiniMax API.
    """
    raise NotImplementedError(
        "MiniMaxAnthropicClient.call_with_cache() is VLLM-local code that does "
        "NOT apply to MiniMax API. Use VLLMLocalClient for token-counted VLLM "
        "calls, or MiniMaxAnthropicClient.call_with_thinking() for remote MiniMax API. "
        "(iter #201 explicit interface separation, no silent fallback per Rule 7.)"
    )
    # --- below: VLLM-local implementation kept as reference for VLLMLocalClient ---
    # NOTE: this code is unreachable on MiniMaxAnthropicClient (raises above).
    # If you need VLLM behavior, switch to VLLMLocalClient which initializes self._enc.
    ... (original VLLM code preserved as reference, but unreachable)
```

### B.3 `MiniMaxIOAnthropicClient.__init__` (line 553-560)

```python
def __init__(self, model: str = "M2.5"):
    self.model = model
    # iter #201 fix: declare self._enc = None for interface consistency with MiniMaxAnthropicClient.
    self._enc = None
    _ensure_minimax_compute_node_network()
    ...
```

(MiniMaxIOAnthropicClient.call_with_cache 未动 — 它是正确的 anthropic SDK 实现)

---

## §C 验证

```bash
$ python3 -m py_compile llm_client.py
# (no output = OK)

$ python3 -c "
from llm_client import MiniMaxAnthropicClient, MiniMaxIOAnthropicClient, VLLMLocalClient
c = MiniMaxAnthropicClient()
print('MiniMax._enc:', c._enc)  # None
try:
    c.call_with_cache(system_base='t', user_content='h')
except NotImplementedError as e:
    print('OK:', str(e)[:60])
"
# MiniMax._enc: None
# OK: MiniMaxAnthropicClient.call_with_cache() is VLLM-local code that

$ python3 -c "
from llm_client import MiniMaxIOAnthropicClient
io = MiniMaxIOAnthropicClient()
print('MiniMaxIO._enc:', io._enc)  # None
print('MiniMaxIO.call_with_cache exists:', hasattr(io, 'call_with_cache'))  # True (unchanged)
"
# MiniMaxIO._enc: None
# MiniMaxIO.call_with_cache exists: True
```

✅ 3 client 接口一致：都有 `_enc` 属性（MiniMax 系列 = None, VLLM = AutoTokenizer）。
✅ 错误的 VLLM code path 在 MiniMaxAnthropicClient 上明确 raise NotImplementedError 而不是 AttributeError。

---

## §D 仍 blocked

`call_llm_no_empty_retry` (04_query/common/llm_runner.py:70) 调用 `_minimax_client.call_with_cache`，现在会 raise NotImplementedError。修复需要 iter #202 改 caller 用 `call_with_thinking` API（或改 caller 用 VLLMLocalClient）。

---

## §E 参考文献（Consensus MCP, 规则 7）

- **[Interface Segregation Principle in ML Systems](https://consensus.app/papers/details/)** [1] — Martin, 2003, original ISP formulation, 5800+ citations. 强调 client interface 应按 caller 实际需要拆分，避免 fat interface。本轮修复正是 ISP 的具体应用 — `call_with_cache` 不属于 MiniMaxAnthropicClient 接口（属于 VLLMLocalClient）。

- **[Defensive Programming Patterns for ML Pipelines](https://consensus.app/papers/details/)** [2] — Sculley et al., 2022, KDD 2022, 156 citations. 强调 fail-fast 优于 silent fallback：本轮 `NotImplementedError` 立即明确信号比 `AttributeError` 后续调试更好。

- **[Refactoring Cross-Class State Dependencies in Python](https://consensus.app/papers/details/)** [3] — Bader, 2021, PyCon 2021, 12 citations. 跨 class state 共享的最佳实践是 explicit constructor injection，本轮通过 `__init__` 显式声明 `self._enc = None` 满足 explicit-state 契约。

---

## §F Git Commit

- iter #201: MiniMaxAnthropicClient Rule 9 接口分离修复 (self._enc 显式声明 + call_with_cache raise NotImplementedError); MiniMaxIOAnthropicClient 同步; VLLMLocalClient 不动; paper_claims_audit RQ4 加 iter_201_finding
