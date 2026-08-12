"""Shared MiniMax LLM client and call helper for 04_query stage-6 scripts.

The original `06_generate_by_persona_placeholder_*.py` files held the
`load_minimax_client` / `call_llm` pair at module scope; the
`06_generate_by_syntax_depth_no_depth_check_10_*.py` variants dynamic-loaded
those modules to reuse them. Once the placeholder scripts were removed,
this module re-homes the shared client lifecycle.
"""

from __future__ import annotations

import sys
from typing import Optional


_minimax_client = None
_first_request = True


def load_minimax_client(use_minimaxio: bool = False):
    """Load the MiniMax API client (Anthropic-compatible by default)."""
    global _minimax_client
    if _minimax_client is not None:
        return _minimax_client

    from .attribute_helpers import log

    sys.path.insert(0, '/home/wlia0047/ar57/wenyu/PersoanlQuery')
    from llm_client import MiniMaxAnthropicClient, MiniMaxIOAnthropicClient

    if use_minimaxio:
        _minimax_client = MiniMaxIOAnthropicClient()
        log("MiniMaxIO API 客户端初始化完成")
    else:
        _minimax_client = MiniMaxAnthropicClient()
        log("MiniMax API 客户端初始化完成")
    return _minimax_client


def reset_first_request() -> None:
    """Re-arm the cache-creation flag (call between independent prompt streams)."""
    global _first_request
    _first_request = True


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
