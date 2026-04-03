#!/usr/bin/env python3
"""LLM Clients (MiniMax and ChatGLM) using OpenAI SDK."""

# 权限验证
import re
import time
from openai import OpenAI
from typing import Optional


class LLMClient:
    """MiniMax LLM Client."""
    def __init__(self, model: str = "MiniMax-M2.7"):
        self.model = model
        self.client = OpenAI(
            api_key="sk-cp-jqg2XWIob99HfZTveS5CqjO1h8BAQguTCcHG0p_vZlQ_rNqJgQLqNMwJ7AHMMwRhogi2I8A7o9FZ-f1dR2jsVNfwUsdLzicgrXm9tM8bqodav3ZhtQ0Ig-Y",
            base_url="https://api.minimaxi.com/v1"
        )

    def call(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        max_retries: int = 5,
    ) -> str:
        """Call MiniMax API using OpenAI SDK with reasoning_split=True."""
        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return ""

        safe_max_tokens = max(128, min(int(max_tokens), 4096))
        safe_temp = temperature if temperature is not None else 0.7

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": safe_prompt}],
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    extra_body={"reasoning_split": True},
                )

                if not response.choices:
                    print(f"[MiniMax] No choices in response")
                    return ""

                message = response.choices[0].message

                if hasattr(message, 'content') and message.content:
                    return message.content.strip()

                # Fallback to reasoning_details if content is empty
                if hasattr(message, 'reasoning_details') and message.reasoning_details:
                    reasoning = message.reasoning_details
                    if isinstance(reasoning, list) and len(reasoning) > 0:
                        first_item = reasoning[0]
                        if isinstance(first_item, dict) and 'text' in first_item:
                            return first_item['text'].strip()
                    print(f"[MiniMax reasoning_details]: {reasoning}")

                return ""

            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                print(f"Error calling MiniMax API: {e}")
                return ""

        return ""


class ChatGLMClient:
    """ChatGLM (智谱AI) LLM Client using OpenAI SDK.

    API Docs: https://open.bigmodel.cn/api/paas/v4
    Supports GLM-4.5 series models.
    """
    def __init__(self, model: str = "glm-5", api_key: str = None):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or "YOUR_CHATGLM_API_KEY",
            base_url="https://api.z.ai/api/coding/paas/v4"
        )

    def call(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        max_retries: int = 5,
        thinking_threshold: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call ChatGLM API using OpenAI SDK.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (128-4096)
            temperature: Sampling temperature (0-1)
            max_retries: Number of retry attempts on rate limit
            thinking_threshold: Optional threshold for thinking content (GLM-4.5 feature)
            system_prompt: Optional system prompt to guide response format

        Returns:
            Generated text content (thinking content excluded if not requested)
        """
        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return ""

        safe_max_tokens = max(128, min(int(max_tokens), 4096))
        safe_temp = temperature if temperature is not None else 0.7

        extra_body = {"thinking": {"type": "disabled"}}
        if thinking_threshold is not None:
            extra_body["thinking_threshold"] = thinking_threshold

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": safe_prompt})

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=safe_max_tokens,
                    temperature=safe_temp,
                    extra_body=extra_body if extra_body else None,
                )

                if not response.choices:
                    print(f"[ChatGLM] No choices in response")
                    return ""

                message = response.choices[0].message

                if hasattr(message, 'content') and message.content:
                    # Filter out thinking content tags if present
                    content = message.content
                    if '<|thinking|>' in content:
                        # Extract only non-thinking content
                        parts = content.split('<|thinking|>')
                        result = []
                        for i, part in enumerate(parts):
                            if i % 2 == 0:  # Non-thinking content
                                result.append(part)
                        content = ''.join(result)
                    if content.strip():
                        return content.strip()

                # Fallback to reasoning_content if content is empty
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    reasoning = message.reasoning_content.strip()
                    # Try to find JSON object in the reasoning content
                    import json as json_module
                    json_match = re.search(r'\{[\s\S]*\}', reasoning)
                    if json_match:
                        try:
                            parsed = json_module.loads(json_match.group())
                            return json_module.dumps(parsed)
                        except Exception:
                            pass
                    return reasoning

                return ""

            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = min(60, (2 ** attempt) * 3)
                    print(f"[ChatGLM] Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                print(f"[ChatGLM] Error calling API: {e}")
                return ""

        return ""

    def call_json(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        max_retries: int = 5,
    ) -> str:
        """Call ChatGLM API and force JSON output.

        Shorthand for call() with JSON-only system prompt.
        Automatically strips markdown code block wrappers.
        Attempts to parse JSON from reasoning_content if content is empty.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate (128-4096)
            temperature: Sampling temperature (0-1)
            max_retries: Number of retry attempts on rate limit

        Returns:
            Generated text in JSON format (markdown code blocks removed)
        """
        import json as json_module

        result = self.call(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            thinking_threshold=0.0,
            system_prompt="You must respond in valid JSON format only, no other text.",
        )

        # Strip markdown code block wrappers if present
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        elif result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()

        # Try to parse as JSON directly
        try:
            json_module.loads(result)
            return result
        except Exception:
            pass

        # Try to extract JSON from reasoning content (when content is mixed with thinking)
        json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
        if json_match:
            candidate = json_match.group()
            try:
                json_module.loads(candidate)
                return candidate
            except Exception:
                pass

        # Return as-is if no JSON found
        return result


class MiniMaxClient(LLMClient):
    """MiniMax LLM Client (alias for backward compatibility)."""
    pass
