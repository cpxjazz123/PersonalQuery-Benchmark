#!/usr/bin/env python3
"""LLM Clients (MiniMax + ZAI + VectorEngine + Qwen8b)."""

import atexit
import socket
import subprocess
import threading
import time
import re
import json
import traceback
from datetime import datetime
from typing import Optional

VLLM_CONFIG_FILE = '/workspace/PersonalQuery/PersoanlQuery/06_query/vllm_config.json'

_MINIMAX_SSH_TARGET = "m3-login2"
_MINIMAX_HOST_IP_MAP = {
    "api.minimaxi.com": "47.79.2.234",
    "api.minimax.io": "47.252.72.253",
}
_MINIMAX_NETWORK_READY = False
_MINIMAX_TUNNEL_PROCESS = None
_MINIMAX_TUNNEL_PORT = None
_MINIMAX_NETWORK_LOCK = threading.Lock()
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_ORIGINAL_SOCKET_CLASS = socket.socket


def _log(msg: str):
    """带时间戳的日志打印"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_raw_socket_class():
    if getattr(socket.socket, "__module__", "") == "socks":
        import socks
        return socks._orgsocket
    return _ORIGINAL_SOCKET_CLASS


def _choose_local_socks_port() -> int:
    raw_socket_class = _get_raw_socket_class()
    with raw_socket_class(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_for_socks_tunnel(process, port: int, timeout_seconds: int = 10):
    raw_socket_class = _get_raw_socket_class()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            stderr = process.stderr.read().strip() if process.stderr else ""
            raise RuntimeError(
                f"MiniMax SSH SOCKS 隧道启动失败: return_code={return_code}, stderr={stderr}"
            )

        with raw_socket_class(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return

        time.sleep(0.1)

    process.terminate()
    raise TimeoutError(f"MiniMax SSH SOCKS 隧道启动超时: 127.0.0.1:{port}")


def _cleanup_minimax_socks_tunnel():
    global _MINIMAX_TUNNEL_PROCESS
    if _MINIMAX_TUNNEL_PROCESS is None:
        return
    if _MINIMAX_TUNNEL_PROCESS.poll() is None:
        _MINIMAX_TUNNEL_PROCESS.terminate()
        try:
            _MINIMAX_TUNNEL_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _MINIMAX_TUNNEL_PROCESS.kill()
            _MINIMAX_TUNNEL_PROCESS.wait(timeout=5)


def _patch_minimax_dns():
    def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        lookup_host = host.decode("ascii") if isinstance(host, bytes) else host
        mapped_host = _MINIMAX_HOST_IP_MAP.get(lookup_host)
        if mapped_host is not None:
            return _ORIGINAL_GETADDRINFO(mapped_host, port, family, type, proto, flags)
        return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)

    socket.getaddrinfo = patched_getaddrinfo


def _ensure_minimax_compute_node_network():
    """计算节点外网 DNS/直连不可用时，通过登录节点 SSH SOCKS 隧道访问 MiniMax。"""
    global _MINIMAX_NETWORK_READY, _MINIMAX_TUNNEL_PROCESS, _MINIMAX_TUNNEL_PORT
    if _MINIMAX_NETWORK_READY:
        return

    with _MINIMAX_NETWORK_LOCK:
        if _MINIMAX_NETWORK_READY:
            return

        import socks

        port = _choose_local_socks_port()
        cmd = [
            "ssh",
            "-q",
            "-N",
            "-D",
            f"127.0.0.1:{port}",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            _MINIMAX_SSH_TARGET,
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_for_socks_tunnel(process, port)

        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", port, rdns=True)
        socket.socket = socks.socksocket
        _patch_minimax_dns()

        _MINIMAX_TUNNEL_PROCESS = process
        _MINIMAX_TUNNEL_PORT = port
        _MINIMAX_NETWORK_READY = True
        atexit.register(_cleanup_minimax_socks_tunnel)
        _log(
            f"MiniMax 计算节点网络补丁已启用: socks=127.0.0.1:{port}, "
            f"ssh_target={_MINIMAX_SSH_TARGET}, dns_map={_MINIMAX_HOST_IP_MAP}"
        )


def _log_exception_details(prefix: str, exc: Exception):
    """打印底层 API 异常细节，避免只看到 SDK 的泛化错误。"""
    exc_type = type(exc)
    _log(f"{prefix} exception_type: {exc_type.__module__}.{exc_type.__name__}")
    _log(f"{prefix} exception_repr: {repr(exc)}")
    _log(f"{prefix} exception_message: {str(exc)}")

    for attr in ("status_code", "request_id", "code", "type"):
        if hasattr(exc, attr):
            _log(f"{prefix} {attr}: {getattr(exc, attr)}")

    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "reason_phrase"):
            if hasattr(response, attr):
                _log(f"{prefix} response.{attr}: {getattr(response, attr)}")
        response_text = getattr(response, "text", None)
        if response_text:
            _log(f"{prefix} response.text(first_2000): {response_text[:2000]}")

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        cause_type = type(cause)
        _log(f"{prefix} cause_type: {cause_type.__module__}.{cause_type.__name__}")
        _log(f"{prefix} cause_repr: {repr(cause)}")
        _log(f"{prefix} cause_message: {str(cause)}")

    context = getattr(exc, "__context__", None)
    if context is not None and context is not cause:
        context_type = type(context)
        _log(f"{prefix} context_type: {context_type.__module__}.{context_type.__name__}")
        _log(f"{prefix} context_repr: {repr(context)}")
        _log(f"{prefix} context_message: {str(context)}")

    tb = "".join(traceback.format_exception(exc_type, exc, exc.__traceback__))
    _log(f"{prefix} traceback:\n{tb}")


class MiniMaxAnthropicClient:
    """MiniMax LLM Client using Anthropic SDK.

    Uses the Anthropic-compatible API endpoint at https://api.minimaxi.com/anthropic
    Supports MiniMax-M2.7 and other models with thinking/text content blocks.
    """
    def __init__(self, model: str = "MiniMax-M2.7-highspeed"):
        self.model = model
        _ensure_minimax_compute_node_network()
        import anthropic
        self.client = anthropic.Anthropic(
            base_url="https://api.minimaxi.com/anthropic",
            api_key="sk-cp-jqg2XWIob99HfZTveS5CqjO1h8BAQguTCcHG0p_vZlQ_rNqJgQLqNMwJ7AHMMwRhogi2I8A7o9FZ-f1dR2jsVNfwUsdLzicgrXm9tM8bqodav3ZhtQ0Ig-Y"
        )

    def call_with_thinking(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> tuple:
        """Call MiniMax API and return both thinking and text.

        Returns:
            tuple: (thinking_text, text_content)
        """
        import anthropic

        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return "", ""

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        retry_count = 0
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    messages=[
                        {"role": "user", "content": safe_prompt}
                    ]
                )

                thinking_text = ""
                text_content = ""

                for block in message.content:
                    if block.type == "thinking":
                        thinking_text = block.thinking
                    elif block.type == "text":
                        text_content = block.text

                text_content = text_content.strip()
                # 空响应也需要重试
                if not text_content:
                    if attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        print(f"[MiniMax-Anthropic] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    else:
                        return "", ""
                if retry_count > 0:
                    print(f"[MiniMax-Anthropic] Empty response retry succeeded after {retry_count} retries.")
                return thinking_text, text_content

            except Exception as e:
                error_str = str(e)
                # 429/529/500/502/503/504/520/522/530 等错误都需要重试
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"[MiniMax-Anthropic] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                print(f"[MiniMax-Anthropic] Error calling API: {e}")
                return "", ""

        return "", ""

    def call(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> str:
        """Call MiniMax API and return text content only.

        Returns:
            str: The text content from the response.
        """
        _, text_content = self.call_with_thinking(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        return text_content

    def call_with_cache(
        self,
        system_base: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> tuple:
        """Call MiniMax API with prompt caching.

        Uses MiniMax's built-in ephemeral cache mechanism.
        The system content with cache_control is cached by MiniMax automatically.

        Args:
            system_base: Static system prompt (with cache_control on first call)
            user_content: Dynamic user content (varies per call)
            max_tokens: Max output tokens
            temperature: Sampling temperature
            max_retries: Max retry attempts

        Returns:
            tuple: (text_content, cache_read_input_tokens)
        """
        import anthropic

        safe_system = system_base.strip() if isinstance(system_base, str) else str(system_base)
        safe_user = user_content.strip() if isinstance(user_content, str) else str(user_content)
        if not safe_system or not safe_user:
            return "", 0

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        retry_count = 0
        for attempt in range(max_retries):
            try:
                # 构建 messages
                messages = [{"role": "user", "content": safe_user}]

                # 始终使用 cache_control，让 MiniMax 自动管理缓存
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    system=[
                        {"type": "text", "text": safe_system, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=messages
                )

                text_content = ""
                cache_read_input_tokens = 0
                cache_creation_input_tokens = 0
                input_tokens = 0
                output_tokens = 0

                for block in message.content:
                    if block.type == "text":
                        text_content = block.text
                    elif block.type == "thinking":
                        pass

                # 提取所有 token 相关字段
                if hasattr(message, 'usage') and message.usage:
                    cache_read_input_tokens = getattr(message.usage, 'cache_read_input_tokens', 0)
                    cache_creation_input_tokens = getattr(message.usage, 'cache_creation_input_tokens', 0)
                    input_tokens = getattr(message.usage, 'input_tokens', 0)
                    output_tokens = getattr(message.usage, 'output_tokens', 0)

                text_content = text_content.strip()
                if not text_content:
                    if attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        _log(f"[MiniMax-Cache] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    else:
                        return "", {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

                if retry_count > 0:
                    _log(f"[MiniMax-Cache] Empty response retry succeeded after {retry_count} retries.")
                return text_content, {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

            except Exception as e:
                error_str = str(e)
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    _log(f"[MiniMax-Cache] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                _log(f"[MiniMax-Cache] Error calling API: {e}")
                return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}


class MiniMaxIOAnthropicClient:
    """MiniMax IO LLM Client using Anthropic SDK.

    Uses the Anthropic-compatible API endpoint at https://api.minimax.io/anthropic
    Supports MiniMax-M2.7 and other models with thinking/text content blocks.
    """
    def __init__(self, model: str = "MiniMax-M2.7"):
        self.model = model
        _ensure_minimax_compute_node_network()
        import anthropic
        self.client = anthropic.Anthropic(
            base_url="https://api.minimax.io/anthropic",
            api_key="sk-cp-GpGDO4bCprD-LScnuPL2JUzK0B3VNC6e6zFFkjRYP7OPTgY-sxuezrlwM-qLcri2zl0VmgGXmfv_FaNlOp-lVGDeiJnR6qVppp8fW4280iC4k4gSggDXWn8"
        )

    def call_with_thinking(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> tuple:
        """Call MiniMax IO API and return both thinking and text.

        Returns:
            tuple: (thinking_text, text_content)
        """
        import anthropic

        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return "", ""

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        retry_count = 0
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    messages=[
                        {"role": "user", "content": safe_prompt}
                    ]
                )

                thinking_text = ""
                text_content = ""

                for block in message.content:
                    if block.type == "thinking":
                        thinking_text = block.thinking
                    elif block.type == "text":
                        text_content = block.text

                text_content = text_content.strip()
                if not text_content:
                    if attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        print(f"[MiniMaxIO-Anthropic] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    else:
                        return "", ""
                if retry_count > 0:
                    print(f"[MiniMaxIO-Anthropic] Empty response retry succeeded after {retry_count} retries.")
                return thinking_text, text_content

            except Exception as e:
                error_str = str(e)
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"[MiniMaxIO-Anthropic] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                _log(f"[MiniMaxIO-Anthropic] Error calling API: {e}")
                _log_exception_details("[MiniMaxIO-Anthropic]", e)
                return "", ""

        return "", ""

    def call(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> str:
        """Call MiniMax IO API and return text content only."""
        _, text_content = self.call_with_thinking(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        return text_content

    def call_with_cache(
        self,
        system_base: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> tuple:
        """Call MiniMax IO API with prompt caching."""
        import anthropic

        safe_system = system_base.strip() if isinstance(system_base, str) else str(system_base)
        safe_user = user_content.strip() if isinstance(user_content, str) else str(user_content)
        if not safe_system or not safe_user:
            return "", 0

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        retry_count = 0
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": safe_user}]

                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    system=[
                        {"type": "text", "text": safe_system, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=messages
                )

                text_content = ""
                cache_read_input_tokens = 0
                cache_creation_input_tokens = 0
                input_tokens = 0
                output_tokens = 0

                for block in message.content:
                    if block.type == "text":
                        text_content = block.text
                    elif block.type == "thinking":
                        pass

                if hasattr(message, 'usage') and message.usage:
                    cache_read_input_tokens = getattr(message.usage, 'cache_read_input_tokens', 0)
                    cache_creation_input_tokens = getattr(message.usage, 'cache_creation_input_tokens', 0)
                    input_tokens = getattr(message.usage, 'input_tokens', 0)
                    output_tokens = getattr(message.usage, 'output_tokens', 0)

                text_content = text_content.strip()
                if not text_content:
                    if attempt < max_retries - 1:
                        wait_time = min(60, (2 ** attempt) * 3)
                        _log(f"[MiniMaxIO-Cache] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    else:
                        return "", {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

                if retry_count > 0:
                    _log(f"[MiniMaxIO-Cache] Empty response retry succeeded after {retry_count} retries.")
                return text_content, {"cache_creation_input_tokens": cache_creation_input_tokens, "cache_read_input_tokens": cache_read_input_tokens, "input_tokens": input_tokens, "output_tokens": output_tokens}

            except Exception as e:
                error_str = str(e)
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    _log(f"[MiniMaxIO-Cache] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                _log(f"[MiniMaxIO-Cache] Error calling API: {e}")
                _log_exception_details("[MiniMaxIO-Cache]", e)
                return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}

        return "", {"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "input_tokens": 0, "output_tokens": 0}


class ZAIAnthropicClient:
    """ZAI LLM Client using requests.

    Uses the Anthropic-compatible API endpoint at https://api.z.ai/api/anthropic
    Default model: glm-4.5-air.
    """
    def __init__(self, model: str = "glm-4.5-air"):
        self.model = model
        self.api_key = "7b18c427b59d4da088955e565bde3b93.Ik8tOlLvHAlyywyq"
        self.base_url = "https://api.z.ai/api/anthropic"

    def call_with_thinking(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> tuple:
        """Call ZAI API and return both thinking and text.

        Returns:
            tuple: (thinking_text, text_content)
        """
        import requests

        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return "", ""

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": self.model,
            "max_tokens": safe_max_tokens,
            "temperature": safe_temp,
            "messages": [{"role": "user", "content": safe_prompt}],
        }

        retry_count = 0
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=120,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    thinking_text = ""
                    text_content = ""

                    for block in result.get("content", []):
                        if block.get("type") == "thinking":
                            thinking_text = block.get("thinking", "")
                        elif block.get("type") == "text":
                            text_content = block.get("text", "")

                    text_content = text_content.strip()
                    # 如果返回内容为空，重试
                    if not text_content:
                        if attempt < max_retries - 1:
                            wait_time = min(60, (2 ** attempt) * 3)
                            print(f"[ZAI-Anthropic] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                            time.sleep(wait_time)
                            retry_count += 1
                            continue
                        else:
                            return "", ""
                    if retry_count > 0:
                        print(f"[ZAI-Anthropic] Empty response retry succeeded after {retry_count} retries.")
                    return thinking_text, text_content
                else:
                    print(f"[VectorEngine] Non-200 response: status={resp.status_code}, body={resp.text[:500] if resp.text else 'empty'}")
                    error_data = resp.json() if resp.text else {}
                    error_msg = error_data.get("error", {}).get("message", resp.text)
                    raise Exception(f"Error code: {resp.status_code} - {error_msg}")

            except Exception as e:
                error_str = str(e)
                # 429/529/500/502/503/504/520/522/530 等错误都需要重试
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"[ZAI-Anthropic] Rate limited ({e}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                print(f"[ZAI-Anthropic] Error calling API: {e}")
                return "", ""

        return "", ""

    def call(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
    ) -> str:
        """Call ZAI API and return text content only.

        Returns:
            str: The text content from the response.
        """
        _, text_content = self.call_with_thinking(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        return text_content


class VectorEngineClient:
    """VectorEngine LLM Client using OpenAI-compatible Responses API.

    Uses the API endpoint at https://api.vectorengine.ai/v1/responses
    Default model: gpt-5-nano-2025-08-07.
    """
    def __init__(self, model: str = "gpt-5-nano-2025-08-07"):
        self.model = model
        self.api_key = "sk-60CeMDwtGBBn4yQmrl4XOESAGHcWS8eMY0Z1p3UG82EGzuyx"
        self.base_url = "https://api.vectorengine.ai"

    def call_with_thinking(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: Optional[float] = None,
        max_retries: int = 100,
        response_format: Optional[dict] = None,
    ) -> tuple:
        """Call VectorEngine API and return thinking and text.

        Returns:
            tuple: (thinking_text, text_content)
        """
        import requests

        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return "", ""

        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        data = {
            "model": self.model,
            "input": safe_prompt,
            "max_tokens": safe_max_tokens,
            "temperature": safe_temp,
        }

        if response_format:
            data["response_format"] = response_format

        retry_count = 0
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/responses",
                    headers=headers,
                    json=data,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    text_content = ""

                    # 使用 output 字段获取响应内容
                    output = result.get("output", [])
                    if isinstance(output, list) and output:
                        for item in output:
                            if item.get("type") == "message":
                                text_content = item.get("content", [{}])[0].get("text", "")

                    text_content = text_content.strip()
                    # 如果返回内容为空，重试
                    if not text_content:
                        # 打印完整响应用于调试
                        print(f"[VectorEngine] Empty response. Full result: {result}")
                        if attempt < max_retries - 1:
                            wait_time = min(60, (2 ** attempt) * 3)
                            print(f"[VectorEngine] Empty response. Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                            time.sleep(wait_time)
                            retry_count += 1
                            continue
                        else:
                            return "", ""
                    if retry_count > 0:
                        print(f"[VectorEngine] Empty response retry succeeded after {retry_count} retries.")
                    return "", text_content
                else:
                    print(f"[VectorEngine] Non-200 response: status={resp.status_code}, body={resp.text[:500] if resp.text else 'empty'}")
                    error_data = resp.json() if resp.text else {}
                    error_msg = error_data.get("error", {}).get("message", resp.text)
                    raise Exception(f"Error code: {resp.status_code} - {error_msg}")

            except Exception as e:
                error_str = str(e)
                # 429/529/500/502/503/504/520/522/530 等错误都需要重试
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"[VectorEngine] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                print(f"[VectorEngine] Error calling API: {e}")
                return "", ""

        return "", ""

    def call(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 100,
        response_format: Optional[dict] = None,
    ) -> str:
        """Call VectorEngine API and return text content only.

        Returns:
            str: The text content from the response.
        """
        _, text_content = self.call_with_thinking(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            response_format=response_format,
        )
        return text_content


class Qwen8bClient:
    """Qwen3-8B 本地 vLLM 服务器客户端。

    使用 vllm_config.json 中配置的 base_url。
    支持 32768 tokens 上下文长度。
    """

    def __init__(self):
        self._load_config()

    def _load_config(self):
        """从 vllm_config.json 加载配置"""
        try:
            with open(VLLM_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            self.base_url = config.get('base_url', 'http://localhost:8000')
            self.model_name = config.get('model_name', 'Qwen/Qwen3-8B')
            self.max_model_len = config.get('max_model_len', 32768)
            self.status = config.get('status', 'unknown')
            print(f"[Qwen8b] Config loaded: base_url={self.base_url}, model={self.model_name}, max_len={self.max_model_len}, status={self.status}")
        except Exception as e:
            print(f"[Qwen8b] Failed to load config: {e}")
            self.base_url = 'http://localhost:8000'
            self.model_name = 'Qwen/Qwen3-8B'
            self.max_model_len = 32768
            self.status = 'unknown'

    def call_with_thinking(
        self,
        prompt: str,
        max_tokens: int = 16384,
        temperature: Optional[float] = None,
        max_retries: int = 100,
        stop: Optional[list] = None,
    ) -> tuple:
        """调用 vLLM API，返回 (thinking, text)。

        vLLM 不支持 thinking，直接返回空字符串和文本内容。
        Qwen3-8B 会输出 <think>...</think> 标签，需要过滤。
        """
        import requests

        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return "", ""

        safe_max_tokens = max(128, min(int(max_tokens), self.max_model_len))
        safe_temp = max(0.0, min(2.0, float(temperature if temperature is not None else 0.3)))
        safe_stop = stop if stop is not None else []

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": safe_prompt}],
                        "max_tokens": safe_max_tokens,
                        "temperature": safe_temp,
                        "stop": safe_stop,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=180,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    choices = result.get("choices", [])
                    if choices:
                        text_content = choices[0].get("message", {}).get("content", "")
                        # 过滤掉思考标签 <think>...</think>
                        text_content = re.sub(r'<think>[\s\S]*?</think>', '', text_content)
                        text_content = text_content.strip()
                        if text_content:
                            return "", text_content
                    print(f"[Qwen8b] Empty response. Retry {attempt + 1}/{max_retries}")
                else:
                    print(f"[Qwen8b] Non-200: {resp.status_code} - {resp.text[:200]}")
                    error_data = resp.json() if resp.text else {}
                    error_msg = error_data.get("error", {}).get("message", resp.text)
                    raise Exception(f"Error code: {resp.status_code} - {error_msg}")

            except Exception as e:
                error_str = str(e)
                retryable = any(code in error_str for code in ["429", "529", "500", "502", "503", "504", "520", "522", "530", "timeout", "Connection"])
                if retryable:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"[Qwen8b] Rate limited ({e}). Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                print(f"[Qwen8b] Error calling API: {e}")
                return "", ""

        return "", ""

    def call(
        self,
        prompt: str,
        max_tokens: int = 16384,
        temperature: Optional[float] = None,
        max_retries: int = 100,
        stop: Optional[list] = None,
    ) -> str:
        """调用 vLLM API，返回文本内容。"""
        _, text_content = self.call_with_thinking(prompt, max_tokens, temperature, max_retries, stop=stop)
        return text_content
