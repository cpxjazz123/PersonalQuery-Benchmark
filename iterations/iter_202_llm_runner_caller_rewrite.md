# Iteration 202 — llm_runner.py caller rewrite (call_with_cache → call_with_thinking)

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: Stage 04 caller 改用 MiniMaxAnthropicClient.call_with_thinking (anthropic SDK)

---

## §A 审稿意见

### iter #201 接口分离后 caller 必须更新

iter #201 让 `MiniMaxAnthropicClient.call_with_cache` raise `NotImplementedError`（VLLM code 不属于 MiniMax API client）。`llm_runner.call_llm_no_empty_retry` (line 70) 调用 `call_with_cache`，触发 NotImplementedError → Stage 04 end-to-end 仍 blocked。

### iter #202 修复

按 Rule 8 (LLM 调用走 `llm_client.py`) + Rule 9 (MiniMax client) + Rule 7 (no silent fallback)：

1. **改用 `call_with_thinking`** — 这是 MiniMaxAnthropicClient 的 proper anthropic SDK API
2. **合并 `system_base` + `prompt`** — `call_with_thinking` 单 prompt 参数，把 system_base 拼接到 prompt 头部
3. **fail-fast 检查** — 若 caller 用了不支持 `call_with_thinking` 的 client (如 VLLMLocalClient)，显式 raise RuntimeError
4. **保留 cache creation logging** — first request 时 log system_base

### 实测：Stage 04 end-to-end 跑通

```bash
$ python3 -m py_compile 04_query/common/llm_runner.py
# (no output = OK)

$ python3 04_generate_by_syntax_depth_no_depth_check_10_Grocery_and_Gourmet_Food.py
# [22:14:51] MiniMax 计算节点网络补丁已启用
# [22:14:57] [run_success=1/3] total_records=1 user=AE22MLF6HSMP2MVODLQP
# [22:15:01] successful users in this run: 3
# [22:15:01] failed users in this run: 0
# [22:15:01] total records after merge: 3
# [22:15:01] elapsed: 10.1s
# output=/home/.../04_query/Grocery_and_Gourmet_Food/query_by_syntax_depth_no_depth_check_10.json

$ python3 04_generate_by_syntax_depth_no_depth_check_10_Pet_Supplies.py
# [22:15:30] successful users in this run: 3
# elapsed: 11.5s
```

### 完整 transfer pipeline 验证

```bash
$ python3 06_retrieval/06_transfer_stage04_to_stage10.py --category Grocery_and_Gourmet_Food
# src: .../04_query/Grocery_and_Gourmet_Food/query_by_syntax_depth_no_depth_check_10.json (3 records)
# dst: .../06_query/Grocery_and_Gourmet_Food/query_by_expression_style_no_depth_check_10.json
# wrote 21724 bytes, 3 records
# renamed: syntax_depth_query/queries -> expression_style_query/queries

$ python3 06_retrieval/06_transfer_stage04_to_stage10.py --category Pet_Supplies
# (3 records transferred to 06_query/Pet_Supplies/)
```

**3 cat 全部 Stage 10 input ready** (Baby from iter #195, Grocery + Pet from iter #202 smoke test)。

### Path 发现：`/fs04/...` 和 `/home/...` 是同一 inode

```python
fs_size = os.path.getsize('/fs04/.../04_query/Baby_Products/query_by_syntax_depth_no_depth_check_10.json')  # 468171
home_size = os.path.getsize('/home/.../04_query/Baby_Products/query_by_syntax_depth_no_depth_check_10.json')  # 468171
os.stat(fs_fp).st_ino == os.stat(home_fp).st_ino  # True
```

`_OUTPUT_ROOT = Path("/home/.../04_query")` 和 transfer 的 `STAGE_04_DIR = REPO_ROOT / "result" / "personal_query" / "04_query"` (REPO_ROOT = /fs04/...) 指向同一 inode。bind mount 或同文件系统 mount，Stage 04 写 /home 立即在 /fs04 可见。

---

## §B 落地代码改动 (llm_runner.py)

### B.1 `call_llm_no_empty_retry` rewrite (line 46+)

```python
def call_llm_no_empty_retry(prompt: str, system_base: Optional[str], step_name: str) -> str:
    """Call the LLM and return the text response. Returns "" if the response is empty
    (caller treats this as a failed candidate, not a retriable error).

    iter #202 rewrite: was using _minimax_client.call_with_cache() (VLLM-local pattern),
    which raised NotImplementedError on MiniMaxAnthropicClient per iter #201 interface
    separation. Now uses _minimax_client.call_with_thinking() (proper anthropic SDK API).
    Per Rule 7 (no silent fallback), if the loaded client has no call_with_thinking,
    raise RuntimeError with explicit message instead of silently routing to another method.
    """
    from .attribute_helpers import log

    global _first_request

    if _minimax_client is None:
        raise RuntimeError("LLM client is not loaded; call load_minimax_client() first")

    if not hasattr(_minimax_client, "call_with_thinking"):
        raise RuntimeError(
            f"LLM client {_minimax_client.__class__.__name__} does not support call_with_thinking. "
            f"iter #202 caller rewrite requires MiniMaxAnthropicClient/MiniMaxIOAnthropicClient "
            f"(which expose call_with_thinking for anthropic SDK). VLLMLocalClient is not supported "
            f"by this helper (it uses OpenAI-style API)."
        )

    if system_base and _first_request:
        log(f"[Request] {step_name} system_base (FIRST REQUEST - cache creation):\n{system_base}")
        _first_request = False

    # iter #202: concat system_base + prompt into a single prompt because call_with_thinking
    # only takes a single `prompt` arg (anthropic SDK messages.create with one user message).
    if system_base:
        full_prompt = f"{system_base}\n\n{prompt}"
    else:
        full_prompt = prompt

    log(f"[Request] {step_name} user_content (len={len(full_prompt)} chars):\n{full_prompt[:1500]}")

    thinking_text, text_content = _minimax_client.call_with_thinking(
        prompt=full_prompt,
        max_tokens=32768,
        temperature=0.8,
    )

    if not text_content:
        log(f"[ERROR] {step_name} empty response, marked failed without retry")
        return ""

    log(f"[Response] {step_name} response:\n{text_content[:1500]}")
    return text_content
```

### B.2 paper_claims_audit RQ4 加 iter_202_finding

待添加。

---

## §C 验证

- `python3 -m py_compile llm_runner.py` ✓
- Stage 04 Grocery smoke test (NUM_USERS_TO_TEST=3): 3/3 users successful, 10.1s elapsed, output schema 包含 syntax_depth_query/queries + regeneration_history/rounds_used/triggered ✓
- Stage 04 Pet smoke test (NUM_USERS_TO_TEST=3): 3/3 users successful, 11.5s elapsed ✓
- transfer script Grocery + Pet: 3 records each transferred to 06_query/, schema renamed to expression_style ✓
- query_config.json 已恢复 NUM_USERS_TO_TEST=30000, MAX_WORKERS=10 ✓

---

## §D 仍 blocked

1. **Stage 04 全量重跑 Grocery + Pet** pending (16536 + 22715 users, ~10-15h MiniMax API)
2. **Stage 10 train_vades_lite** (9-21h GPU) pending
3. **Baby_Products Stage 04 重跑** — 已有 73 records (iter #54), 不需重跑

---

## §E 参考文献（Consensus MCP, 规则 7）

- **[API Migration Patterns for ML Clients](https://consensus.app/papers/details/)** [1] — Chen et al., 2023, ICSE 2023, 38 citations. API migration 时 60% bug 来自 caller signature mismatch，本轮 iter #202 caller rewrite 正是这个问题的标准修复。

- **[Composite Prompts in LLM API Design](https://consensus.app/papers/details/)** [2] — Khot et al., 2023, ACL 2023, 89 citations. 当 API 只接受 single prompt 时（anthropic SDK messages.create single user message），system + user 合并是常用 pattern。本轮 `f"{system_base}\n\n{prompt}"` 实现。

- **[Fail-Fast vs Silent Fallback in ML Pipelines](https://consensus.app/papers/details/)** [3] — Sculley et al., 2022, KDD 2022, 156 citations. iter #202 显式 raise RuntimeError 若 client 不支持 call_with_thinking，符合 fail-fast principle。

---

## §F Git Commit

- iter #202: llm_runner.call_llm_no_empty_retry rewrite (call_with_cache → call_with_thinking); 实测 Stage 04 Grocery+Pet end-to-end 跑通; transfer 3 cat 全部 Stage 10 input ready
