"""LLM client wrapper — single source of LLM access (Rule 8/9).

2026-09-19: Created for Stage 11/12 rerank evaluation. 所有 LLM 调用必须通过此 wrapper,
不允许业务代码直接 import vllm / transformers / openai 等.

Backend whitelist: 本地 Qwen 系列 (Qwen2.5 / Qwen3) 通过 vLLM 或 transformers 加载。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Union


# ---------- 默认 backend 配置 (Rule 9: 仅本地 Qwen) ----------
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SFT_ADAPTER = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/06_training_model/sft_lora")
DEFAULT_BACKEND = "vllm"     # 'vllm' 或 'transformers'
DEFAULT_MAX_MODEL_LEN = 2048  # rerank 时需要更长 prompt
DEFAULT_MAX_NUM_SEQS = 1024   # 2026-09-19: vLLM 并发从 256 提升到 1024 (Stage 11/12 LLM rerank)
DEFAULT_GPU_MEM_UTIL = 0.90   # 2026-09-19: 0.85→0.90 提速 (更多 GPU mem 给 vLLM KV cache)
DEFAULT_DTYPE = "bfloat16"
DEFAULT_ENABLE_PREFIX_CACHING = True   # 2026-09-19: 共享 prompt prefix cache


class QwenLocalClient:
    """本地 Qwen 推理 client. 支持 vLLM (preferred, 批量并发) 或 transformers.

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
                 gpu_memory_utilization: float = DEFAULT_GPU_MEM_UTIL,
                 dtype: str = DEFAULT_DTYPE,
                 enforce_eager: bool = False,
                 enable_prefix_caching: bool = DEFAULT_ENABLE_PREFIX_CACHING):
        self.model = model
        self.backend = backend
        self.lora_adapter = Path(lora_adapter) if lora_adapter else None
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching
        self._backend = None

    def _init_backend(self):
        if self._backend is not None:
            return
        if self.backend == "vllm":
            from vllm import LLM
            print(f"[llm_client] loading vLLM {self.model} (lora={self.lora_adapter}) ...", flush=True)
            kwargs = dict(
                model=self.model,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                enforce_eager=self.enforce_eager,
                trust_remote_code=True,
                enable_prefix_caching=self.enable_prefix_caching,
            )
            if self.lora_adapter is not None and self.lora_adapter.exists():
                kwargs["enable_lora"] = True
                kwargs["max_lora_rank"] = 8
            self._backend = LLM(**kwargs)
        elif self.backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[llm_client] loading transformers {self.model} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model, torch_dtype=getattr(torch, self.dtype), device_map="auto",
                trust_remote_code=True)
            self._backend = (model, tok)
        else:
            raise ValueError(f"unknown backend: {self.backend}")

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
        if self.backend == "vllm":
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
            sp = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                                max_tokens=max_tokens, stop=stop or [])
            kwargs = {}
            if self.lora_adapter is not None and self.lora_adapter.exists():
                kwargs["lora_request"] = LoRARequest("sft_adapter", 1, str(self.lora_adapter))
            outputs = self._backend.generate(prompts, sp, use_tqdm=False, **kwargs)
            return [[o.text for o in out.outputs] for out in outputs]
        else:  # transformers
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

    def shutdown(self):
        """释放 vLLM / transformers GPU 资源."""
        if self._backend is None:
            return
        try:
            if self.backend == "vllm":
                del self._backend
            else:
                model, tok = self._backend
                del model
                del tok
        finally:
            self._backend = None


# ---------- 单例 cache (lazy) ----------
_client_singleton: Optional[QwenLocalClient] = None


def get_client(**kwargs) -> QwenLocalClient:
    """Get or create singleton client."""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = QwenLocalClient(**kwargs)
    return _client_singleton


def reset_client():
    """Reset singleton (used after rerank)."""
    global _client_singleton
    if _client_singleton is not None:
        _client_singleton.shutdown()
        _client_singleton = None


# ---------- high-level rerank helper ----------
def llm_rerank_pointwise(query: str,
                         candidates: List[dict],
                         asin_to_doc: dict,
                         top_k: int = 10,
                         client: Optional[QwenLocalClient] = None,
                         max_chars: int = 200) -> List[int]:
    """对 query + 每个 candidate doc 拼 prompt → Qwen 输出相关度分数 → top_k 排序.

    candidates: list of dict, must contain 'asin' field. Document text 从 asin_to_doc[asin] 取,
    截断 max_chars 字符避免超长 prompt.

    Returns: top_k candidate indices (sorted by descending relevance).
    """
    client = client or get_client()
    prompts = []
    for cand in candidates:
        asin = cand.get("asin")
        doc_text = asin_to_doc.get(asin, "").replace("\n", " ").strip()[:max_chars]
        prompts.append(_build_rerank_prompt(query, doc_text))
    raw = client.generate(prompts, n=1, temperature=0.0, top_p=1.0, max_tokens=4)
    scores = [_parse_rerank_score(r[0]) if r else 0.0 for r in raw]
    ranked = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return ranked[:top_k], scores


def _build_rerank_prompt(query: str, doc_text: str) -> str:
    """Qwen 通用 rerank prompt. Output: 单 token 数字 0-3."""
    return (
        "Rate the relevance of the document to the query on a scale of 0 (irrelevant) "
        "to 3 (highly relevant). Output only a single digit.\n\n"
        f"Query: {query}\n"
        f"Document: {doc_text}\n\n"
        "Relevance:"
    )


def _parse_rerank_score(text: str) -> float:
    """Parse Qwen 0-3 single digit output."""
    t = text.strip()
    if not t:
        return 0.0
    # Take first digit
    for c in t:
        if c in "0123":
            return float(c)
    # Fallback: any digit
    import re
    m = re.search(r"\d", t)
    return float(m.group(0)) if m else 0.0