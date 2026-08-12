#!/usr/bin/env python3
"""LLM Clients (local Qwen inference).

项目唯一 LLM 入口（见 /fs04/ar57/wenyu/CLAUDE.md Rule 8/9）。
仅提供本地 Qwen-7B/8B 级别推理，远程后端（MiniMax / OpenAI / Anthropic 等）已下线。

默认模型路径：
    /home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct

可通过环境变量覆盖：
    QWEN_MODEL_PATH   模型权重路径或 HF repo id
    QWEN_DTYPE        bfloat16 | float16 | float32（默认 bfloat16）
    QWEN_MAX_MODEL_LEN  int（默认取模型 config 的 max_position_embeddings）
    QWEN_GPU_MEMORY_UTILIZATION  float（默认 0.9）

后端硬编码为 vllm（不再支持 transformers 后端，不再读取 QWEN_BACKEND）。
"""

import os
import threading
import time
import traceback
from datetime import datetime
from typing import Optional


DEFAULT_QWEN_MODEL_PATH = (
    "/home/wlia0047/hj82_scratch2/wenyu/RAG/cfrag_project/LLMs/Qwen2-7B-Instruct"
)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _usage_value(usage, key: str) -> int:
    """兼容 dict / 对象两种 usage 表达，返回 token 数。"""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(key, 0)
    else:
        value = getattr(usage, key, 0)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class _LocalBackend:
    """线程安全的本地 vllm 推理单例。

    第一次访问时按 model_path 加载 vllm 模型，之后所有 QwenLocalClient
    实例复用同一份模型对象。后端硬编码为 vllm，不再支持 transformers 路径。
    """

    _lock = threading.Lock()
    _initialized: bool = False
    backend_name: Optional[str] = None
    model = None
    tokenizer = None
    device: Optional[str] = None
    dtype = None
    model_path: Optional[str] = None

    @classmethod
    def get(cls, model_path: str):
        if cls._initialized and cls.model_path == model_path:
            return cls
        with cls._lock:
            if cls._initialized and cls.model_path == model_path:
                return cls
            cls._load(model_path)
            return cls

    @classmethod
    def _load(cls, model_path: str) -> None:
        # 后端硬编码为 vllm，不再支持 transformers 路径
        backend = "vllm"

        cls.model_path = model_path
        cls.backend_name = backend

        if not os.path.exists(model_path) and not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Qwen 模型路径不存在: {model_path}。"
                f"可通过环境变量 QWEN_MODEL_PATH 指向其他权重目录或 HF repo id。"
            )

        cls._load_vllm(model_path)

        cls._initialized = True
        _log(
            f"[QwenLocal] 已加载模型: path={model_path}, backend={backend}, "
            f"dtype={cls.dtype}, device={cls.device}"
        )

    @classmethod
    def _load_vllm(cls, model_path: str) -> None:
        # 硬编码禁用 flashinfer sampler：当前环境 CUDA toolkit 12.0 与 flashinfer
        # 0.6.13 自带 CCCL 不兼容（sampling.cuh 编译报
        # BlockAdjacentDifference::FlagHeads 不存在）。必须在 import vllm 之前
        # 注入，因为 vllm.envs 在首次访问时才读 os.environ。
        # 注入后，vllm sampler 走 PyTorch 原生 (Triton for bs>=8) 路径，无 JIT。
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        from vllm import LLM
        from vllm.config import KernelConfig

        cls.dtype = None
        max_model_len_env = os.environ.get("QWEN_MAX_MODEL_LEN")
        gpu_mem_env = os.environ.get("QWEN_GPU_MEMORY_UTILIZATION", "0.9")
        try:
            gpu_memory_utilization = float(gpu_mem_env)
        except ValueError:
            raise ValueError(
                f"QWEN_GPU_MEMORY_UTILIZATION={gpu_mem_env!r} 非法，必须是 0~1 的 float"
            )

        kwargs = {
            "model": model_path,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": os.environ.get("QWEN_DTYPE", "bfloat16"),
            # 硬编码 enforce_eager=True：跳过 cuda graphs + torch.compile
            "enforce_eager": True,
            # 硬编码 kernel_config.enable_flashinfer_autotune=False：跳过
            # flashinfer JIT autotune（attention 路径已默认走 FLASH_ATTN backend，
            # 这里关闭 sampling autotune 是冗余保险）
            "kernel_config": KernelConfig(enable_flashinfer_autotune=False),
        }
        if max_model_len_env:
            try:
                kwargs["max_model_len"] = int(max_model_len_env)
            except ValueError:
                raise ValueError(f"QWEN_MAX_MODEL_LEN={max_model_len_env!r} 非法")

        llm = LLM(**kwargs)
        cls.model = llm
        cls.tokenizer = llm.get_tokenizer()
        cls.device = "cuda"

    @classmethod
    def reset(cls) -> None:
        """显式释放模型与显存（用于换模型/换 backend）。"""
        with cls._lock:
            cls.model = None
            cls.tokenizer = None
            cls._initialized = False
            cls.backend_name = None
            cls.model_path = None
            cls.dtype = None
            cls.device = None


def _apply_chat_template(tokenizer, system_text: str, user_text: str) -> str:
    """使用 tokenizer.apply_chat_template 拼出 Qwen chat prompt。

    Qwen2/Qwen3 Instruct 模型必须经过 chat template 才能正确生效 system 角色；
    直接拼 "<system>...</system>\n<user>...</user>" 会绕过 special token 与模板逻辑。
    """
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        # 不允许静默回退到 string concat（见 CLAUDE.md Rule 7）—— 直接抛错
        raise RuntimeError(
            f"tokenizer.apply_chat_template 失败: {type(exc).__name__}: {exc}"
        ) from exc


def _build_prompt_from_user_only(tokenizer, user_text: str) -> str:
    """无 system 场景下，仍走 chat template 以保证与训练分布一致。"""
    return _apply_chat_template(tokenizer, system_text="", user_text=user_text)


def _run_vllm(
    backend: _LocalBackend,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float],
) -> tuple[str, int, int]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=float(temperature) if temperature is not None else 0.0,
        top_p=0.95 if temperature is not None and temperature > 0 else 1.0,
    )
    outputs = backend.model.generate([prompt], sampling)
    if not outputs:
        raise RuntimeError("vllm 返回空输出列表")
    request_output = outputs[0]
    if not request_output.outputs:
        raise RuntimeError("vllm RequestOutput.outputs 为空")
    text = request_output.outputs[0].text.strip()
    input_tokens = (
        len(request_output.prompt_token_ids)
        if request_output.prompt_token_ids is not None
        else 0
    )
    output_tokens = (
        len(request_output.outputs[0].token_ids)
        if request_output.outputs[0].token_ids is not None
        else 0
    )
    return text, input_tokens, output_tokens


def _run_generate(
    backend: _LocalBackend,
    prompt: str,
    max_tokens: int,
    temperature: Optional[float],
) -> tuple[str, int, int]:
    """后端硬编码为 vllm，直接调用 _run_vllm。"""
    return _run_vllm(backend, prompt, max_tokens, temperature)


class QwenLocalClient:
    """本地 Qwen-7B/8B 量级 Instruct 模型客户端。

    接口契约与原 MiniMaxAnthropicClient 保持一致，便于业务侧通过
    cfg["llm"]["client_class"] 切换后端而无需改动其他代码。
    """

    def __init__(self, model: str = DEFAULT_QWEN_MODEL_PATH):
        # 业务侧可能传 HF repo id；env 优先，再 fallback 到入参
        env_path = os.environ.get("QWEN_MODEL_PATH", "").strip()
        self.model_name = env_path or model or DEFAULT_QWEN_MODEL_PATH
        self._backend = _LocalBackend.get(self.model_name)
        # 暴露给业务侧 debug 使用
        self.model = self.model_name
        # Qwen3 才有 thinking；Qwen2 一律关闭
        self._supports_thinking = "Qwen3" in os.path.basename(self.model_name)

    # ---------- public API ----------

    def call(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 350,
    ) -> str:
        """单轮 user prompt 调用，返回纯文本。"""
        _, text = self.call_with_thinking(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )
        return text

    def call_with_thinking(
        self,
        prompt: str,
        max_tokens: int = 8192,
        temperature: Optional[float] = None,
        max_retries: int = 350,
    ) -> tuple:
        """单轮 user prompt 调用，返回 (thinking_text, text_content)。

        本地 Qwen2 模型不原生支持 thinking；Qwen3 通过 QWEN_ENABLE_THINKING
        控制 tokenizer 的 enable_thinking 参数；为保持向后兼容，thinking_text
        一律返回空字符串，text_content 返回主生成内容。
        """
        safe_prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
        if not safe_prompt:
            return "", ""
        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        full_prompt = _build_prompt_from_user_only(self._backend.tokenizer, safe_prompt)

        retry_count = 0
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                text, _in_tok, _out_tok = _run_generate(
                    self._backend,
                    full_prompt,
                    safe_max_tokens,
                    safe_temp,
                )
                if not text:
                    if attempt < max_retries - 1:
                        wait_time = 60
                        _log(
                            f"[QwenLocal] Empty response. "
                            f"Retry {attempt + 1}/{max_retries}, waiting {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    return "", ""
                if retry_count > 0:
                    _log(f"[QwenLocal] Empty response retry succeeded after {retry_count} retries.")
                return "", text
            except Exception as e:
                last_error = e
                error_str = str(e)
                retryable = any(
                    code in error_str
                    for code in ["CUDA out of memory", "OOM", "RuntimeError"]
                )
                if retryable:
                    wait_time = 30
                    _log(
                        f"[QwenLocal] Inference failed ({type(e).__name__}: {e}). "
                        f"Retry {attempt + 1}/{max_retries}, waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                # 不可重试错误：直接 raise（Rule 7）
                _log(f"[QwenLocal] Non-retryable error: {type(e).__name__}: {e}")
                _log(
                    f"[QwenLocal] traceback:\n"
                    f"{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
                )
                raise

        if last_error is not None:
            _log(
                f"[QwenLocal] Exhausted {max_retries} retries; last error: "
                f"{type(last_error).__name__}: {last_error}"
            )
        return "", ""

    def call_with_cache(
        self,
        system_base: str,
        user_content: str,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        max_retries: int = 350,
        retry_on_empty_response: bool = True,
        stream: bool = False,
    ) -> tuple:
        """与 MiniMaxAnthropicClient 同构的 system + user 调用。

        本地推理无远程 cache 概念，但保持 usage dict 形态以兼容业务侧统计：
            {"cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0,
             "input_tokens": int,
             "output_tokens": int}

        stream 参数保留以兼容旧调用签名；本地 vllm 单请求
        同步生成，stream=True 走与 False 完全相同的路径。
        """
        safe_system = system_base.strip() if isinstance(system_base, str) else str(system_base)
        safe_user = user_content.strip() if isinstance(user_content, str) else str(user_content)
        if not safe_system or not safe_user:
            return "", {
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }
        safe_max_tokens = max(128, int(max_tokens))
        safe_temp = temperature if temperature is not None else 0.7

        full_prompt = _apply_chat_template(
            self._backend.tokenizer, safe_system, safe_user
        )

        retry_count = 0
        last_error: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                text, in_tok, out_tok = _run_generate(
                    self._backend,
                    full_prompt,
                    safe_max_tokens,
                    safe_temp,
                )
                if not text:
                    if retry_on_empty_response and attempt < max_retries - 1:
                        wait_time = 60
                        _log(
                            f"[QwenLocal-Cache] Empty response. "
                            f"Retry {attempt + 1}/{max_retries}, waiting {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    return "", {
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                    }
                if retry_count > 0:
                    _log(
                        f"[QwenLocal-Cache] Empty response retry succeeded after {retry_count} retries."
                    )
                return text, {
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                }
            except Exception as e:
                last_error = e
                error_str = str(e)
                retryable = any(
                    code in error_str
                    for code in ["CUDA out of memory", "OOM", "RuntimeError"]
                )
                if retryable:
                    wait_time = 30
                    _log(
                        f"[QwenLocal-Cache] Inference failed ({type(e).__name__}: {e}). "
                        f"Retry {attempt + 1}/{max_retries}, waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                _log(
                    f"[QwenLocal-Cache] Non-retryable error: {type(e).__name__}: {e}"
                )
                _log(
                    f"[QwenLocal-Cache] traceback:\n"
                    f"{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
                )
                raise

        if last_error is not None:
            _log(
                f"[QwenLocal-Cache] Exhausted {max_retries} retries; last error: "
                f"{type(last_error).__name__}: {last_error}"
            )
        return "", {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def _call_with_cache_streaming(
        self,
        safe_system: str,
        safe_user: str,
        safe_max_tokens: int,
        safe_temp: float,
        max_retries: int,
        retry_on_empty_response: bool,
        log_prefix: str,
    ) -> tuple:
        """兼容旧接口：本地推理无真正的 streaming，统一走同步路径。"""
        return self.call_with_cache(
            system_base=safe_system,
            user_content=safe_user,
            max_tokens=safe_max_tokens,
            temperature=safe_temp,
            max_retries=max_retries,
            retry_on_empty_response=retry_on_empty_response,
            stream=False,
        )


# 工厂函数（与 14_llm_rerank/common/llm_rerank_common.py 中的 get_llm_client 兼容）
def create_qwen_local_client(model: Optional[str] = None) -> QwenLocalClient:
    """便捷工厂：model 缺省时使用 QWEN_MODEL_PATH / DEFAULT_QWEN_MODEL_PATH。"""
    if model is None:
        model = os.environ.get("QWEN_MODEL_PATH", "").strip() or DEFAULT_QWEN_MODEL_PATH
    return QwenLocalClient(model=model)


__all__ = ["QwenLocalClient", "create_qwen_local_client", "DEFAULT_QWEN_MODEL_PATH"]