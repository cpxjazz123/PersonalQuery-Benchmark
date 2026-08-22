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

# 业务侧可调用 QwenLocalClient.get_hidden_states(texts, layers) 提取
# 指定层的 pooled hidden activations（mean over tokens weighted by attention
# mask）。vllm 不原生暴露 per-layer hidden states，因此单独建一个 transformers
# 后端通道；与 _LocalBackend（vllm 生成）解耦，按需懒加载、互不抢占显存。
_HIDDEN_DEFAULT_DTYPE = os.environ.get("QWEN_DTYPE", "bfloat16")
_HIDDEN_DEFAULT_BATCH = int(os.environ.get("QWEN_HIDDEN_BATCH", "32"))
_HIDDEN_MAX_LENGTH = int(os.environ.get("QWEN_HIDDEN_MAX_LENGTH", "160"))


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


class _HiddenBackend:
    """transformers-only 后端，专门给 hidden state 提取用。

    与 _LocalBackend（vllm 生成）完全解耦：独立 singleton、独立显存占用、
    按需懒加载。vllm 不原生暴露 per-layer hidden states，所以单独建一个
    transformers 通道；AGENTS.md Rule 8 禁止业务代码直接 import
    transformers，所以把这路径封到 llm_client.py 内部，业务侧只用
    QwenLocalClient.get_hidden_states(texts, layers)。
    """

    _lock = threading.Lock()
    _initialized: bool = False
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
        # transformers 直接 import：只允许出现在 llm_client.py 内部
        # （AGENTS.md Rule 8 业务代码不允许 import transformers）
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"_HiddenBackend 模型路径不存在: {model_path}"
            )
        dt = _HIDDEN_DEFAULT_DTYPE
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dt)
        if torch_dtype is None:
            raise ValueError(
                f"QWEN_DTYPE={dt!r} 非法，必须是 bfloat16/float16/float32"
            )

        cls.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        if cls.tokenizer.pad_token_id is None:
            cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id
        cls.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch_dtype, device_map="cuda",
            trust_remote_code=True)
        cls.model.eval()
        for p in cls.model.parameters():
            p.requires_grad = False
        cls.device = "cuda"
        cls.dtype = dt
        cls.model_path = model_path
        cls._initialized = True
        _log(
            f"[QwenLocal-Hidden] 已加载 transformers 模型: path={model_path}, "
            f"dtype={dt}, n_layer={cls.model.config.num_hidden_layers}"
        )

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls.model = None
            cls.tokenizer = None
            cls._initialized = False
            cls.model_path = None
            cls.dtype = None
            cls.device = None


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

    def __init__(
        self,
        model: str = DEFAULT_QWEN_MODEL_PATH,
        with_vllm: bool = True,
    ):
        """初始化 QwenLocalClient。

        参数:
            model: HF repo id 或本地模型路径（env QWEN_MODEL_PATH 优先）
            with_vllm: 是否同时初始化 vllm 生成后端。False 时只初始化
                transformers hidden-states 后端，可用于「仅需 hidden state
                提取、不需要 generation」的场景（例如 E24 StyleVector 网格
                扫描）。当前 pq_env 上 vllm 0.27.1 + torch 2.13/cu130 引擎
                初始化失败（TypeError: 'type' object is not subscriptable），
                hidden-states-only 工作流因此必须传 with_vllm=False。
        """
        # 业务侧可能传 HF repo id；env 优先，再 fallback 到入参
        env_path = os.environ.get("QWEN_MODEL_PATH", "").strip()
        self.model_name = env_path or model or DEFAULT_QWEN_MODEL_PATH
        if with_vllm:
            self._backend = _LocalBackend.get(self.model_name)
        else:
            self._backend = None
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

        if self._backend is None:
            raise RuntimeError(
                "QwenLocalClient 实例化时 with_vllm=False：vllm 生成后端未"
                "初始化。generation 方法（call / call_with_cache）需要"
                "with_vllm=True。"
            )
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

        if self._backend is None:
            raise RuntimeError(
                "QwenLocalClient 实例化时 with_vllm=False：vllm 生成后端未"
                "初始化。call_with_cache 需要 with_vllm=True。"
            )
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

    # ---------- hidden state extraction ----------
    # 业务侧用：抽取指定层的 pooled hidden activations（mean over tokens
    # weighted by attention_mask），与 style_vector / PACS / steering 一类
    # 任务配套。transformers 直跑，不走 vllm（vllm 不暴露 per-layer hidden
    # states），但封装在 llm_client.py 内部以满足 AGENTS.md Rule 8。
    def get_hidden_states(
        self,
        texts: list[str],
        layers: list[int],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> dict[int, "np.ndarray"]:
        """对一组文本提取指定层的 pooled hidden states。

        参数:
            texts: 字符串列表
            layers: 层索引列表（0-indexed，Qwen2-7B 共 28 层，取值范围 0..27）
            batch_size: 默认 _HIDDEN_DEFAULT_BATCH (32)
            max_length: 默认 _HIDDEN_MAX_LENGTH (160)

        返回: dict[layer_idx, np.ndarray]，shape=(len(texts), hidden_dim)
              pooled 方式：mean over tokens weighted by attention_mask
        """
        import numpy as np  # 仅在本方法内 import（业务代码可用 numpy）

        if not texts:
            raise ValueError("texts must be non-empty")
        if not layers:
            raise ValueError("layers must be non-empty")
        n_layers_total = self._hidden_backend.model.config.num_hidden_layers
        bad = [l for l in layers if not (0 <= l < n_layers_total)]
        if bad:
            raise ValueError(
                f"layer indices out of range [0, {n_layers_total}): {bad}"
            )
        bs = batch_size or _HIDDEN_DEFAULT_BATCH
        ml = max_length or _HIDDEN_MAX_LENGTH
        tok = self._hidden_backend.tokenizer
        m = self._hidden_backend.model

        # 按层收集 (text_idx, pooled_vec)
        per_layer: dict[int, list] = {l: [] for l in layers}
        import torch
        with torch.no_grad():
            for st in range(0, len(texts), bs):
                chunk = texts[st:st + bs]
                enc = tok(
                    chunk, return_tensors="pt", padding=True,
                    truncation=True, max_length=ml).to(self._hidden_backend.device)
                o = m(**enc, use_cache=False, output_hidden_states=True)
                hs = torch.stack([o.hidden_states[l] for l in layers])
                # hs shape: (n_layers, B, T, H)
                am = enc["attention_mask"].unsqueeze(0).unsqueeze(-1).to(hs.dtype)
                pooled = (hs * am).sum(2) / am.sum(2).clamp_min(1)
                # pooled shape: (n_layers, B, H)
                for li, l in enumerate(layers):
                    per_layer[l].append(pooled[li].float().cpu().numpy())
                _log(
                    f"  hidden {min(st + bs, len(texts))}/{len(texts)} "
                    f"layers={layers}"
                )
        return {l: np.concatenate(per_layer[l], axis=0) for l in layers}

    @property
    def _hidden_backend(self):
        return _HiddenBackend.get(self.model_name)

    # ---------- hidden state injection during generation ----------
    # 业务侧用：把 per-row 风格向量（user_mu 投影到 hidden_dim）作为 bias 注入
    # 指定 transformer layer 的 hidden state，引导生成靠近用户风格。
    # 走 transformers _HiddenBackend 的 model.generate + forward hook，batched
    # (Rule 4)，不暴露 transformers 给业务代码（Rule 8）。
    #
    # 使用限制：
    #   - 当前 vllm 0.27.1 在 pq_env 引擎初始化失败（type object is not
    #     subscriptable），本接口走 transformers path，无 vllm PagedAttention
    #     优化，单 batch 速度低于 vllm；高频生成仍建议走 vllm 路径。
    #   - frequency_penalty / presence_penalty 是 vllm 扩展参数，transformers
    #     4.40 不支持；如需去模板套话，prompt 层用 system 约束更稳定。
    def generate_with_hidden_injection(
        self,
        system_text: str,
        user_texts: list[str],
        injection_per_row: list,
        injection_layers: list[int],
        injection_alpha: float = 1.0,
        max_new_tokens: int = 96,
        temperature: float = 0.5,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        batch_size: int = 16,
        max_input_length: int = 384,
        mask_cjk: bool = False,
    ) -> list[str]:
        """Batched generation with per-row hidden-state bias injection.

        参数:
            system_text: system prompt 文本（同一 batch 共享）
            user_texts: 用户 prompt 列表
            injection_per_row: 长度 = len(user_texts) 的 list，每项是
                shape=(hidden_dim,) 的 1-D tensor 或 None；None 表示该行
                不注入（用作 D_off baseline）
            injection_layers: 0-indexed layer indices，注入 bias 到该层
                decoder layer 的 output hidden state 上（broadcast over T）
            injection_alpha: scalar multiplier on bias，控制注入强度
            max_new_tokens: 每条生成的最大新 token 数
            temperature / top_p: 采样参数
            repetition_penalty: 重复惩罚（transformers 4.40 支持）
            batch_size: 模型生成时的 batch size
            max_input_length: tokenize 截断长度
            mask_cjk: True 时把 vocab 里所有 decode 后含 CJK 字符的 token 的
                logit 设为 -inf, 防止注入风格偏移引入非英文噪声（用户指令
                2026-08-22）。vocab 152K 一次性扫描 ~5s, 缓存到 _cjk_mask。

        返回: list[str]，长度 = len(user_texts)，已 trim 到第一换行
        """
        import threading
        import torch
        import unicodedata

        if not user_texts:
            return []
        if len(injection_per_row) != len(user_texts):
            raise ValueError(
                f"injection_per_row length ({len(injection_per_row)}) != "
                f"user_texts length ({len(user_texts)})"
            )
        hidden_backend = self._hidden_backend
        m = hidden_backend.model
        tok = hidden_backend.tokenizer
        device = hidden_backend.device
        n_layers = m.config.num_hidden_layers
        for L in injection_layers:
            if not (0 <= L < n_layers):
                raise ValueError(
                    f"layer {L} out of range [0, {n_layers}); model has "
                    f"{n_layers} layers"
                )

        # 应用 chat template (transformers 4.40 AutoTokenizer 支持 apply_chat_template)
        full_prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": system_text},
                 {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True,
            )
            for u in user_texts
        ]

        # 预乘 alpha 并搬上 GPU/dtype
        # hidden_backend.dtype 是字符串 ("bfloat16" 等), 需先 map 成 torch.dtype
        _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        target_dtype = _DTYPE_MAP.get(hidden_backend.dtype)
        if target_dtype is None:
            raise ValueError(
                f"unsupported hidden backend dtype: {hidden_backend.dtype!r}"
            )
        biases: list = []
        for b in injection_per_row:
            if b is None:
                biases.append(None)
            else:
                t = b.to(device=device, dtype=target_dtype)
                if injection_alpha != 1.0:
                    t = t * float(injection_alpha)
                biases.append(t)

        # 线程本地上下文:每个 chunk 重新设置 ctx.biases
        ctx = threading.local()

        def make_hook(L_idx: int):
            def hook(module, input, output):
                # Qwen2DecoderLayer.forward 返回 (hidden_states, present_kv)
                # 或仅 hidden_states（use_cache=False）
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # h shape: (B, T, H)
                bs_local = ctx.biases  # 当前 chunk 的 per-row biases
                for b_idx, bias in enumerate(bs_local):
                    if bias is not None:
                        h[b_idx] = h[b_idx] + bias  # broadcast over T
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            return hook

        handles = []
        # decoder-only 模型必须用 left-padding，否则首步预测会落在 padding token
        # 位置上（model.generate 仍会输出文本但内容断裂 — 参见 Rule 4 注意事项）。
        # 这里临时把 padding_side 切成 "left"，生成完恢复为原值（默认 right，
        # 业务侧用 get_hidden_states 提取时也用 right-padding，不能持久改）。
        orig_padding_side = tok.padding_side
        tok.padding_side = "left"

        # CJK 字符屏蔽 mask (用户指令 2026-08-22): vocab 里 decode 后含 CJK 字符
        # 的 token 都把 logit 设为 -inf, 防止注入风格偏移引入中文噪声。
        # vocab 152K 一次性扫描 ~5s, 缓存到 _cjk_token_ids 属性。
        logits_processor = None
        if mask_cjk:
            if not hasattr(self, "_cjk_token_ids") or self._cjk_token_ids is None:
                cjk_ids: list[int] = []
                for tid in range(len(tok)):
                    s = tok.decode([tid])
                    if not s:
                        continue
                    if any(
                        '一' <= ch <= '鿿'            # CJK Unified Ideographs
                        or '㐀' <= ch <= '䶿'        # CJK Ext A
                        or '\U00020000' <= ch <= '\U0002a6df'  # CJK Ext B
                        or '豈' <= ch <= '﫿'          # CJK Compatibility
                        or '　' <= ch <= '〿'          # CJK Symbols/Punctuation (含日文)
                        or '぀' <= ch <= 'ゟ'          # Hiragana
                        or '゠' <= ch <= 'ヿ'          # Katakana
                        or '가' <= ch <= '힯'          # Hangul Syllables
                        for ch in s
                    ):
                        cjk_ids.append(tid)
                self._cjk_token_ids = cjk_ids
                _log(f"[QwenLocal-Hidden] CJK mask: {len(cjk_ids)} tokens masked out of {len(tok)}")
            else:
                _log(f"[QwenLocal-Hidden] CJK mask cache hit: {len(self._cjk_token_ids)} tokens")

            class _CJKMaskProcessor:
                """LogitsProcessor: 把 CJK token 的 logit 设为 -inf。"""
                def __init__(self, token_ids: list[int]):
                    self.token_ids = token_ids
                def __call__(self, input_ids, scores):
                    scores[:, self.token_ids] = -float("inf")
                    return scores

            logits_processor = [_CJKMaskProcessor(self._cjk_token_ids)]

        try:
            for L in injection_layers:
                handles.append(
                    m.model.layers[L].register_forward_hook(make_hook(L))
                )

            all_texts: list[str] = []
            for st in range(0, len(full_prompts), batch_size):
                chunk = full_prompts[st:st + batch_size]
                chunk_biases = biases[st:st + batch_size]
                ctx.biases = chunk_biases

                enc = tok(
                    chunk, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_input_length,
                ).to(device)

                do_sample = temperature > 0
                with torch.no_grad():
                    out_ids = m.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        top_p=top_p if do_sample else 1.0,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                        logits_processor=logits_processor,
                    )

                # 解码：只取 prompt 长度之后的新 token
                prompt_len = enc["input_ids"].shape[1]
                new_ids = out_ids[:, prompt_len:]
                for row in new_ids:
                    text = tok.decode(row, skip_special_tokens=True).strip()
                    text = text.split("\n")[0].strip()  # 防御性截断
                    all_texts.append(text)

            return all_texts
        finally:
            for h in handles:
                h.remove()
            tok.padding_side = orig_padding_side  # 恢复成原 padding_side


# 工厂函数（与 14_llm_rerank/common/llm_rerank_common.py 中的 get_llm_client 兼容）
def create_qwen_local_client(
    model: Optional[str] = None,
    with_vllm: bool = True,
) -> QwenLocalClient:
    """便捷工厂：model 缺省时使用 QWEN_MODEL_PATH / DEFAULT_QWEN_MODEL_PATH。

    with_vllm=False 时跳过 vllm 初始化（pq_env 当前 vllm engine 不可用）；
    此模式仅支持 hidden state 提取（get_hidden_states），不能调用
    call / call_with_cache 等生成接口。
    """
    if model is None:
        model = os.environ.get("QWEN_MODEL_PATH", "").strip() or DEFAULT_QWEN_MODEL_PATH
    return QwenLocalClient(model=model, with_vllm=with_vllm)


__all__ = ["QwenLocalClient", "create_qwen_local_client", "DEFAULT_QWEN_MODEL_PATH"]