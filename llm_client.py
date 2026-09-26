"""LLM client wrapper — single source of LLM access (Rule 8/9).

2026-09-19: Created for Stage 11/12 rerank evaluation. 所有 LLM 调用必须通过此 wrapper,
不允许业务代码直接 import vllm / transformers / openai 等.

Backend whitelist: Qwen3-Reranker (vLLM) and Llama-3.1-8B-Instruct (vLLM).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional


# 2026-09-23: pq_env 多版本 dist-info 残留（torch ×5, transformers ×4, accelerate ×2, torchvision ×4），
# 实际加载 transformers 4.57.6 + torchvision 0.19.0 vs torch 2.14.0 不兼容，
# import torchvision 时 _meta_registrations 注册 fake `torchvision::nms` 抛 RuntimeError，
# 级联导致 transformers.image_utils / Qwen3ForCausalLM / TrainerCallback 等全部炸。
# 解决：导入 transformers 之前 stub torchvision.transforms（提供 InterpolationMode enum）
# 和 torchvision.transforms.v2.functional（空模块），同时把 4.57.x 已移除的
# AutoProcessor 从 processing_auto 子模块重新 export 到顶层。
import enum as _enum
import importlib.machinery as _im
import types as _types

_tv = _types.ModuleType("torchvision")
_tv.__path__ = ["/home/wlia0047/ar57_scratch/wenyu/pq_env/lib/python3.10/site-packages/torchvision"]
_tv.__spec__ = _im.ModuleSpec("torchvision", None, is_package=True)
sys.modules.setdefault("torchvision", _tv)


class _InterpolationMode(_enum.Enum):
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


_tv_t = _types.ModuleType("torchvision.transforms")
_tv_t.__spec__ = _im.ModuleSpec("torchvision.transforms", None)
_tv_t.InterpolationMode = _InterpolationMode
sys.modules.setdefault("torchvision.transforms", _tv_t)
_tv.transforms = _tv_t
_tv_t_v2 = _types.ModuleType("torchvision.transforms.v2")
_tv_t_v2.__spec__ = _im.ModuleSpec("torchvision.transforms.v2", None)
_tv_t_v2.functional = _types.SimpleNamespace()
sys.modules.setdefault("torchvision.transforms.v2", _tv_t_v2)
_tv_t.v2 = _tv_t_v2

import transformers as _transformers_mod
from transformers.models.auto.processing_auto import AutoProcessor as _AutoProcessor
_transformers_mod.AutoProcessor = _AutoProcessor


# 禁止 vLLM/Outlines 向 /home 写 telemetry 或 SQLite cache；统一写入 scratch。
os.environ["VLLM_USAGE_STATS"] = "0"
os.environ["OUTLINES_CACHE_DIR"] = "/fs04/scratch2/hj82/wenyu/outlines_cache"


# ---------- 默认 backend 配置 (Rule 9: 仅本地 Qwen / Llama 3.1) ----------
DEFAULT_QWEN_MODEL = "/home/wlia0047/hj82_scratch2/wenyu/RAG/Qwen3-Reranker-8B"
DEFAULT_SFT_ADAPTER = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/06_training_model/sft_lora")
DEFAULT_BACKEND = "vllm"  # Qwen3 and Llama 3.1 rerankers all use vLLM now.
DEFAULT_MAX_MODEL_LEN = 2048  # Qwen3-Reranker 0.6B strict
DEFAULT_MAX_NUM_SEQS = 32  # placeholder, unused under transformers backend
DEFAULT_MAX_BATCHED_TOKENS = 4096  # placeholder
DEFAULT_GPU_MEM_UTIL = 0.85  # unused under transformers backend
DEFAULT_DTYPE = "bfloat16"
DEFAULT_ENABLE_PREFIX_CACHING = False  # 2026-09-20: 关掉 prefix caching 避免 KV 重建开销
DEFAULT_VLLM_LOGPROBS = 20


class QwenLocalClient:
    """本地 Qwen / Llama 推理 client. 支持 vLLM (preferred, 批量并发) 或 transformers.

    使用方法:
        client = QwenLocalClient()
        outputs = client.generate(prompts, n=1, temperature=0.0, max_tokens=64)
    """

    def __init__(self,
                 model: str = DEFAULT_QWEN_MODEL,
                 backend: str = DEFAULT_BACKEND,
                 lora_adapter: Optional[Path] = None,
                 max_model_len: int = DEFAULT_MAX_MODEL_LEN,
                 max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
                 max_num_batched_tokens: int = DEFAULT_MAX_BATCHED_TOKENS,
                 gpu_memory_utilization: float = DEFAULT_GPU_MEM_UTIL,
                 dtype: str = DEFAULT_DTYPE,
                 enforce_eager: bool = False,
                 enable_prefix_caching: bool = DEFAULT_ENABLE_PREFIX_CACHING):
        self.model = model
        self.backend = backend
        self.lora_adapter = Path(lora_adapter) if lora_adapter else None
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching
        self._backend = None

    def _init_backend(self):
        if self._backend is not None:
            return
        backend = self.backend
        if backend == "vllm":
            from vllm import LLM

            print(f"[llm_client] loading vLLM {self.model} ...", flush=True)
            self._backend = LLM(
                model=self.model,
                tokenizer=self.model,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                enforce_eager=self.enforce_eager,
                trust_remote_code=True,
                enable_prefix_caching=self.enable_prefix_caching,
            )
            self.backend_kind = "vllm"
        elif backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[llm_client] loading transformers (auto) {self.model} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype=getattr(torch, self.dtype),
                device_map="auto", trust_remote_code=True)
            model.eval()
            self._backend = (model, tok)
            self.backend_kind = "transformers"
        else:
            raise ValueError(f"unknown backend: {backend}")

    def generate(self,
                 prompts: List[str],
                 n: int = 1,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 max_tokens: int = 64,
                 stop: Optional[List[str]] = None) -> List[List[str]]:
        """批量生成. Returns list of lists: outputs[i] = list of n generations for prompt i."""
        self._init_backend()
        if not prompts:
            return []
        if self.backend_kind == "vllm":
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
            )
            outputs = self._backend.generate(
                prompts, sampling_params, use_tqdm=False
            )
            return [[item.text for item in output.outputs] for output in outputs]

        import torch
        model, tok = self._backend
        results = []
        for prompt in prompts:
            inputs = tok(prompt, return_tensors="pt", truncation=True,
                         max_length=self.max_model_len - max_tokens).to(model.device)
            gen = model.generate(
                **inputs, do_sample=(temperature > 0), temperature=max(temperature, 1e-5),
                top_p=top_p, max_new_tokens=max_tokens, num_return_sequences=n,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            texts = []
            for g in gen:
                txt = tok.decode(g[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                if stop:
                    for s in stop:
                        if s in txt:
                            txt = txt.split(s)[0]
                texts.append(txt)
            results.append(texts)
        return results

    def score_logit_diff(self,
                         prompts: List[str],
                         yes_tokens: List[int] | None = None,
                         no_tokens: List[int] | None = None,
                         batch_size: int = 1024) -> List[float]:
        """Score Qwen3 / Llama 3.1 rerank prompts from the first generated token.

        vLLM returns the selected token plus the requested top logprobs. The
        ``allowed_token_ids`` constraint keeps all Yes/No variants in that
        returned set without changing their logit difference. The returned
        value is ``sigmoid(logP(Yes) - logP(No))`` so it has the same [0, 1]
        scale as the Transformers ``score_pairs`` implementation.
        """
        import math
        from vllm import SamplingParams

        self._init_backend()
        if self.backend_kind != "vllm":
            raise RuntimeError(
                "score_logit_diff requires the vLLM backend; "
                f"backend_kind={self.backend_kind}"
            )
        if not prompts:
            return []

        tokenizer = self._backend.get_tokenizer()
        if yes_tokens is None:
            yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
        if no_tokens is None:
            no_tokens = tokenizer.encode(" No", add_special_tokens=False)
        yes_ids = set(yes_tokens)
        no_ids = set(no_tokens)
        allowed_ids = sorted(yes_ids | no_ids)
        if not yes_ids or not no_ids:
            raise ValueError(
                f"Empty Yes/No token set: yes={sorted(yes_ids)} "
                f"no={sorted(no_ids)}"
            )

        batch_size = max(1, batch_size)
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        scores: List[float] = []
        started_at = last_log_at = time.time()
        print(
            f"[llm_client {time.strftime('%H:%M:%S')}] "
            f"score_logit_diff start: {len(prompts)} prompts in {n_batches} "
            f"batches (batch_size={batch_size})",
            flush=True,
        )

        for batch_num, start in enumerate(
                range(0, len(prompts), batch_size), start=1):
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=1,
                logprobs=max(DEFAULT_VLLM_LOGPROBS, len(allowed_ids)),
                allowed_token_ids=allowed_ids,
            )
            outputs = self._backend.generate(
                prompts[start:start + batch_size],
                sampling_params,
                use_tqdm=False,
            )
            for output in outputs:
                if not output.outputs or output.outputs[0].logprobs is None:
                    raise RuntimeError(
                        "vLLM returned no first-token logprobs for a Qwen3 "
                        "reranker prompt"
                    )
                first_position = output.outputs[0].logprobs[0]
                yes_values = [
                    info.logprob
                    for token_id, info in first_position.items()
                    if token_id in yes_ids
                ]
                no_values = [
                    info.logprob
                    for token_id, info in first_position.items()
                    if token_id in no_ids
                ]
                if not yes_values and not no_values:
                    raise RuntimeError(
                        "vLLM returned no Yes/No token logprobs: "
                        f"yes_ids={sorted(yes_ids)} no_ids={sorted(no_ids)} "
                        f"returned={sorted(first_position)}"
                    )
                # Top-k logprobs may omit the unlikely label; mirror the
                # transformers path by clamping missing sides to -60.
                logprob_floor = -60.0
                yes_log = max(yes_values) if yes_values else logprob_floor
                no_log = max(no_values) if no_values else logprob_floor
                diff = yes_log - no_log
                diff = max(-60.0, min(60.0, diff))
                scores.append(1.0 / (1.0 + math.exp(-diff)))

            now = time.time()
            if batch_num == n_batches or now - last_log_at >= 30:
                processed = min(start + len(outputs), len(prompts))
                print(
                    f"[llm_client {time.strftime('%H:%M:%S')}] "
                    f"score_logit_diff progress: {processed}/{len(prompts)} "
                    f"prompts ({batch_num}/{n_batches}), "
                    f"elapsed={now - started_at:.1f}s",
                    flush=True,
                )
                last_log_at = now

        print(
            f"[llm_client {time.strftime('%H:%M:%S')}] "
            f"score_logit_diff complete: {len(scores)}/{len(prompts)} "
            f"prompts in {time.time() - started_at:.1f}s",
            flush=True,
        )
        return scores

    def score_pairs(self,
                    pairs: List[tuple[str, str]],
                    prompt_style: str = "qwen3",
                    batch_size: int = 8,
                    yes_tokens: List[int] | None = None,
                    no_tokens: List[int] | None = None) -> List[float]:
        """Score query-passage pairs using the Qwen3-Reranker transformers backend.

        Supported prompt_style: ``"qwen3"`` (Qwen3-Reranker chat template ending
        in ``"assistant\\n<think>\\n</think>\\n\\n"``). The yes/no logit is
        read at the LAST non-padding position (where generation would start),
        which matches the official Qwen3-Reranker scoring convention when
        served via transformers.

        Returns sigmoid-normalised relevance scores in [0, 1], one per pair.

        Forces the ``transformers`` backend. For batched Qwen3 reranking use
        ``score_logit_diff`` with ``backend="vllm"`` instead.
        """
        import math
        import torch
        backend_started = time.time()
        print(f"[llm_client {time.strftime('%H:%M:%S')}] "
              f"score_pairs loading backend for {self.model}", flush=True)
        self._init_backend()
        print(f"[llm_client {time.strftime('%H:%M:%S')}] score_pairs backend ready "
              f"in {time.time() - backend_started:.1f}s", flush=True)
        if self.backend_kind != "transformers":
            raise RuntimeError(
                f"score_pairs requires the transformers backend; "
                f"backend_kind={self.backend_kind}"
            )
        model, tok = self._backend
        if prompt_style != "qwen3":
            raise ValueError(f"Unknown prompt_style: {prompt_style}")
        # logits_to_keep=1 returns the final padded column. Left padding
        # makes that column the final real token for every batch item.
        tok.padding_side = "left"
        if yes_tokens is None:
            yes_tokens = tok.encode("Yes", add_special_tokens=False)
        if no_tokens is None:
            no_tokens = tok.encode("No", add_special_tokens=False)

        model.eval()
        scores: List[float] = []
        eos_id = tok.eos_token_id
        pad_id = tok.pad_token_id or eos_id
        n_batches = (len(pairs) + batch_size - 1) // batch_size
        started_at = last_log_at = time.time()
        print(f"[llm_client {time.strftime('%H:%M:%S')}] score_pairs start: "
              f"{len(pairs)} pairs in {n_batches} batches "
              f"(batch_size={batch_size}, style={prompt_style})", flush=True)

        with torch.inference_mode():
            for batch_num, start in enumerate(
                    range(0, len(pairs), batch_size), start=1):
                batch = pairs[start:start + batch_size]
                prompts = []
                for q, d in batch:
                    q_clean = q.strip()
                    d_clean = d.strip()
                    # Official Qwen3-Reranker chat template.
                    prompts.append(
                        "<|im_start|>system\n"
                        "You are Qwen, created by Alibaba Cloud. "
                        "You are a helpful assistant.<|im_end|>\n"
                        "<|im_start|>user\n"
                        "<Instruct>: Given a web search query, "
                        "retrieve relevant passages that answer the query\n"
                        f"<Query>: {q_clean}\n"
                        f"<Document>: {d_clean}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                        "<think>\n</think>\n\n"
                    )
                enc = tok(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_model_len - 1,
                ).to(model.device)
                # Qwen3ForCausalLM supports logits_to_keep and only needs
                # the final position for the Yes/No relevance score. This
                # avoids materializing sequence_length × vocab_size logits.
                logits = model(**enc, logits_to_keep=1).logits
                last_logits = logits[:, 0, :]
                z_yes = last_logits[:, yes_tokens].max(dim=-1).values
                z_no = last_logits[:, no_tokens].max(dim=-1).values
                m = torch.maximum(z_yes, z_no)
                p_yes = torch.exp(z_yes - m) / (
                    torch.exp(z_yes - m) + torch.exp(z_no - m)
                )
                scores.extend(p_yes.float().cpu().tolist())
                now = time.time()
                if batch_num == n_batches or now - last_log_at >= 30:
                    processed = min(start + len(batch), len(pairs))
                    print(f"[llm_client {time.strftime('%H:%M:%S')}] "
                          f"score_pairs progress: {processed}/{len(pairs)} pairs "
                          f"({batch_num}/{n_batches} batches), "
                          f"elapsed={now - started_at:.1f}s", flush=True)
                    last_log_at = now
        print(f"[llm_client {time.strftime('%H:%M:%S')}] score_pairs complete: "
              f"{len(scores)}/{len(pairs)} pairs in "
              f"{time.time() - started_at:.1f}s", flush=True)
        return scores

    def shutdown(self):
        """释放 vLLM 或 Transformers GPU 资源."""
        if self._backend is None:
            return
        try:
            if getattr(self, "backend_kind", None) == "vllm":
                backend = self._backend
                del backend
            else:
                model, tok = self._backend
                del model
                del tok
        finally:
            self._backend = None


# ---------- 单例 cache (lazy) ----------
_client_singleton: Optional[QwenLocalClient] = None


def _is_qwen3_checkpoint(model_path: str) -> bool:
    import json as _json
    import os as _os
    cfg_path = _os.path.join(model_path, "config.json")
    if not _os.path.isfile(cfg_path):
        return False
    with open(cfg_path) as _f:
        cfg = _json.load(_f)
    return cfg.get("model_type") == "qwen3"


def get_client(**kwargs) -> QwenLocalClient:
    """Get or create singleton client."""
    global _client_singleton
    if _client_singleton is None:
        # Ensure the current module-level DEFAULT_QWEN_MODEL is honored when caller
        # does not explicitly pass `model=...` (Python default args bind at def-time,
        # not at call-time, so we re-inject from the module attr here).
        kwargs.setdefault("model", DEFAULT_QWEN_MODEL)
        _client_singleton = QwenLocalClient(**kwargs)
    return _client_singleton


def reset_client():
    """Reset singleton (used after rerank)."""
    global _client_singleton
    if _client_singleton is not None:
        _client_singleton.shutdown()
        _client_singleton = None
