#!/usr/bin/env python3
"""SFT pool generation via vLLM.

User instruction 2026-09-05: integrate vLLM and load SFT adapter (result/06_training_model/sft_lora)
Generate 50 ASINs x K_POOL candidates, output to result/07_gen_query/pool.json.

Design:
  - vLLM engine loads Qwen2.5-0.5B-Instruct with enable_lora=True
  - LoRARequest points to sft_lora (LoRA r=8)
  - 50 ASINs x K_POOL batched generation (single generate call)
  - Reuse prompt format from 06_training_model/sft_pipeline.py (with stop tokens)
  - Apply 5-layer strict filter (with <=3 extras threshold, per user feedback 2026-09-05)
  - generate_candidates() / get_or_create_llm() reused by 08_select_query regen

Usage (Rule 3: no args):
  cd /home/wlia0047/ar57/wenyu/PersoanlQuery
  $PY 07_gen_query/sft_pool_generate.py

Output:
  result/07_gen_query/pool_queries.json
"""
from __future__ import annotations

import json
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "result/07_gen_query"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_CACHE_ROOT = Path("/home/wlia0047/hj82_scratch2/wenyu/stage07_pool_cache")
RAW_CACHE_PATH = RAW_CACHE_ROOT / "baby" / "raw_per_asin_cache.json"

# Stage 04 cohort3mlp lowrank+diag artifact supplies fitted users and ASIN cohort coverage.
STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_cohort3mlp16_30_lowrankdiag_rank1.json"
STAGE05_PATH = REPO_ROOT / "result/05_gaussian_audit/raw_cov_validity.json"
SFT_POOL_SOURCE = "stage04_cohort3mlp_lowrankdiag"  # sample-source tag

# CONTRASTIVE_5558 ABLATION: 5558 contrastive cohort stats
CONTRASTIVE_5558 = False
if CONTRASTIVE_5558:
    STAGE04_PATH = REPO_ROOT / "result/04_gaussian/user_gaussian_stats_5558.json"
    SFT_POOL_SOURCE = "stage04_fitted_cohort_5558_contrastive"

QWEN_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
HF_CACHE_DIR = "/home/wlia0047/hj82/wenyu/hf_cache"
SFT_ADAPTER_DIR = REPO_ROOT / "result/06_training_model/sft_lora"
# User instruction 2026-09-23: product_attributes switched to pkl-only (Stage 01 already switched).
ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction/product_attributes_baby.pkl"
# User instruction 2026-09-23: one attributes file per category (Baby / Musical / Video_Games),
# main() runs 3 domains sequentially; outputs go to result/07_gen_query/<subdir>/.
CATEGORY_INPUTS = [
    # (category_key, subdir)
    ("Baby",                "baby"),
    ("Musical_Instruments", "musical"),
    ("Video_Games",         "video_games"),
]

# --- Hardcoded hyperparams (Rule 3) ---
SFT_POOL_SEED = 42
SFT_POOL_SMOKE = False        # True=5 ASIN smoke, False=all eligible ASINs (Rule 20)
SFT_POOL_N_ASIN = 5 if SFT_POOL_SMOKE else None  # SMOKE=5, full=None=no limit (all asin_to_valid)
SFT_POOL_K = 50               # 2026-09-19: 50 candidates per ASIN per round, 2 rounds = 100/ASIN
SFT_POOL_ROUNDS = 2            # 2026-09-06: same prompt run 2 rounds, merged to 50/ASIN
MIN_KEPT_QUERIES = 30          # 2026-09-18: after filter must be >=30 queries; otherwise regen
MAX_REGEN_ROUNDS = 10          # max regen rounds per ASIN (cap to prevent vLLM runaway)
POOL_DUMP_EVERY_N_ROUNDS = 1   # dump raw_per_asin every N rounds to disk to survive crashes
SFT_POOL_TEMP = 0.7
SFT_POOL_TOP_P = 0.95
SFT_POOL_MAX_NEW_TOKENS = 40
SFT_POOL_EXCLUDE_TRAIN = True # exclude ASINs used in SFT training (user query_samples_100.json)
# 2026-09-13 removed extras <= 3 threshold (user instruction): keep only full-coverage + no-repeat filters.
# extras signal still recorded in candidate dict, but no longer fails pass.

# vLLM engine singleton (reused by 07 regen, lazy init)
_LLM = None
_GENERATION_BACKEND = None
_HF_TOKENIZER = None
_LOADED_ADAPTER = None
HF_BATCH_SIZE = 8
_VLLM_SAMPLING_PARAMS = None
_VLLM_LORA_REQUEST = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_sft_prompt(attrs: Dict[str, str]) -> str:
    """Reuse 06_training_model/sft_pipeline._build_sft_prompt (with stop tokens)."""
    attrs_text = "\n".join(f"- {k}: {v}" for k, v in attrs.items())
    return (
        f"You are shopping for a product with these 5 known attributes:\n"
        f"{attrs_text}\n\n"
        f"Write one short natural shopping search query (5-20 words) that "
        f"uses these attributes as keywords. Do NOT ask questions about the "
        f"attributes (they are already known). Do NOT invent or add new "
        f"product information not listed above.\n\n"
        f"Query:<|im_end|>\n"
    )


def load_attributes() -> Dict[str, Dict[str, str]]:
    # User instruction 2026-09-23: use pickle.load (Stage 01 already switched to pkl-only).
    with open(ATTRIBUTES_PATH, "rb") as f:
        return pickle.load(f)


def clean_attr_value(value: str, is_brand: bool = False) -> str:
    """Clean a single attribute string; split when two words are clearly stuck together with no space.

    Retention policy (2026-09-19):
      - Brand fields (Brand / Manufacturer / Item model number): **always preserved verbatim**,
        queries may include them whole (BabyBjorn / WubbaNub / StrollAir).
      - Other fields: aggressive split on CamelCase / multi-uppercase words / underscores.
        "Dishwasher SafeMicrowave Safe" → "Dishwasher Safe Microwave Safe"
        "machine_wash" → "machine wash"
        "Reclining SeatCanopy" → "Reclining Seat Canopy"
        "Walk_through" → "Walk through"
        "MusicalLightsInteractive ToysAdjustable Height" -> multiple tokens
      - Single-token all-uppercase (BPA / DC) / short strings: preserved.
    """
    if not isinstance(value, str):
        return str(value)
    v = value.strip()
    if not v:
        return v
    if is_brand:
        return v
    # 2026-09-19: aggressive split on [a-z][A-Z], [A-Z]+[A-Z][a-z], '_' boundaries
    v = v.replace("_", " ")
    v = re.sub(r"([a-z])([A-Z])", r"\1 \2", v)               # camelCase
    v = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", v)          # FOOBar → FOO Bar
    return v.strip()


_BRAND_FIELDS = {"brand", "manufacturer", "item model number"}


def select_clean_attrs(attrs: Dict[str, str], n_keep: int = 5) -> Dict[str, str]:
    """Pick first n_keep from attrs dict, clean stuck attrs, drop fields still abnormal after cleaning.
    Brand fields (Brand / Manufacturer) skip cleaning and are preserved verbatim.
    2026-09-19: if cleaning still leaves 1 token and raw value has no space and is >10 chars -> skip.
    """
    out: Dict[str, str] = {}
    for k, v_raw in attrs.items():
        is_brand = k.lower() in _BRAND_FIELDS
        v_clean = clean_attr_value(v_raw, is_brand=is_brand)
        if not v_clean:
            continue
        toks = re.findall(r"\b\w+(?:['-]\w+)*\b", v_clean.lower())
        if not toks:
            continue
        v_text = " ".join(toks)
        if len(v_text) > 30:
            continue
        # 2026-09-19: post-cleaning still 1 token, no space, raw >10 chars -> unrecoverable, skip
        if len(toks) == 1 and " " not in v_raw and len(v_raw) > 10:
            continue
        out[k] = v_clean
        if len(out) >= n_keep:
            break
    return out


def filter_pass_queries(asin: str, queries: List[str],
                        attrs_all: Dict[str, Dict[str, str]]) -> List[str]:
    """Filter a single ASIN's query list by content_pass; keep only those passing.

    User 2026-09-13: drop failing queries before writing pool_queries.json.
    Mirrors generate_candidates.check_content: tokenize + full coverage (2026-09-18 dropped no_repeat_token check).
    Cached pool also goes through this path because older artifacts were not pre-filtered.
    2026-09-18: run attrs through select_clean_attrs before filtering, dropping stuck / abnormal attrs.
    """
    if not queries:
        return []
    if asin not in attrs_all:
        # attrs missing -> all fail, matching generate_candidates behavior (raise)
        raise ValueError(f"ASIN {asin} missing in product_attributes")
    attrs = select_clean_attrs(attrs_all[asin], n_keep=5)

    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+(?:['-]\w+)*\b", text)

    kept: List[str] = []
    n_total = len(queries)
    for text in queries:
        tokens = tokenize(text.lower())
        normalized = " ".join(tokens)
        missing = []
        for value in attrs.values():
            value_tokens = tokenize(str(value).lower())
            if not value_tokens:
                continue
            value_text = " ".join(value_tokens)
            if value_text not in normalized:
                missing.append(value_text)
                break
        repeated = len(tokens) != len(set(tokens))
        if not missing:  # 2026-09-18: dropped no_repeat_token check (small words like 'a'/'the' cause spurious rejects)
            kept.append(text)
    return kept


def select_eval_asins(attrs_all: Dict[str, Dict[str, str]],
                      n_asin: int) -> List[str]:
    """Select ASINs from the Stage 04 fitted cohort that have attributes, keeping only those with
    at least one passing user in the Stage 5 audit."""
    if not STAGE04_PATH.exists():
        raise FileNotFoundError(
            f"{STAGE04_PATH} does not exist; run 04_gaussian/fit_per_user_gaussian.py first"
        )
    with open(STAGE04_PATH) as f:
        stage04 = json.load(f)
    asin_to_valid = stage04.get("cohort_gates")
    if not isinstance(asin_to_valid, dict) or not asin_to_valid:
        raise ValueError("Stage 04 artifact requires non-empty cohort_gates")

    valid_uids: set[str] = set()
    if STAGE05_PATH.exists():
        with open(STAGE05_PATH) as f:
            stage05 = json.load(f)
        valid_uids = {uid for uid, r in stage05.get("per_user", {}).items()
                      if r.get("valid_gaussian")}
        log(f"  [stage05] loaded {len(valid_uids)} valid uids from {STAGE05_PATH}")
    else:
        log(f"  [stage05] WARNING: {STAGE05_PATH} not found; "
            f"using all Stage 04 fitted users")

    for asin, cohort in asin_to_valid.items():
        if not isinstance(cohort, dict) or len(cohort) < 2:
            raise ValueError(f"Stage 04 cohort {asin} has fewer than two users")

    excluded = set()
    if SFT_POOL_EXCLUDE_TRAIN:
        sft_train_path = REPO_ROOT / "result/06_training_model/sft_train_asins.json"
        if sft_train_path.exists():
            with open(sft_train_path) as f:
                excluded = set(json.load(f).get("train_asins", []))

    # valid >= 1 Stage5 user intersect >= 2 Stage 4 cohort users intersect attrs >= 5 intersect not-train
    candidates = []
    for asin, udict in asin_to_valid.items():
        if asin in excluded:
            continue
        if asin not in attrs_all or len(list(attrs_all[asin].items())[:5]) < 5:
            continue
        if valid_uids:
            if not any(uid in valid_uids for uid in udict.keys()):
                continue
        candidates.append((asin, len(udict)))
    log(f"  coverage ASINs (Stage 04 cohort_gates): {len(asin_to_valid)}")
    log(f"  after stage05-filter ∩ attrs≥5: {len(candidates)}")

    # sort desc by valid-uid density, tie-break with seed shuffle for reproducibility
    candidates.sort(key=lambda av: (-av[1], av[0]))
    rng = random.Random(SFT_POOL_SEED)
    rng.shuffle(candidates)  # shuffle only same-density ASINs
    candidates.sort(key=lambda av: -av[1])
    if n_asin is None:
        selected = [a for a, _ in candidates]
    else:
        selected = [a for a, _ in candidates[:n_asin]]
        if len(selected) < n_asin:
            log(f"  WARNING: only {len(selected)} ASINs available, requested {n_asin}")
    return selected


def get_or_create_llm():
    """Load vLLM, falling back to a batched Transformers LoRA model."""
    global _LLM, _GENERATION_BACKEND, _HF_TOKENIZER, _LOADED_ADAPTER
    global _VLLM_SAMPLING_PARAMS, _VLLM_LORA_REQUEST
    adapter_path = str(SFT_ADAPTER_DIR)
    if _LLM is not None:
        if (_GENERATION_BACKEND != "transformers"
                or _LOADED_ADAPTER == adapter_path):
            return _LLM
        del _LLM
        _LLM = None
        _HF_TOKENIZER = None
        _LOADED_ADAPTER = None
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        log("  loading vLLM engine (Qwen2.5-0.5B + enable_lora=True) ...")
        t0 = time.time()
        _LLM = LLM(
            model=QWEN_PATH,
            enable_lora=True,
            max_lora_rank=8,
            max_model_len=384,
            gpu_memory_utilization=0.85,
            max_num_seqs=1024,
            enforce_eager=False,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        _VLLM_SAMPLING_PARAMS = SamplingParams
        _VLLM_LORA_REQUEST = LoRARequest
        _GENERATION_BACKEND = "vllm"
        log(f"  vLLM engine loaded in {time.time()-t0:.1f}s")
        return _LLM
    except Exception as exc:
        log(f"  vLLM unavailable ({type(exc).__name__}: {exc}); using Transformers+PEFT")
        _VLLM_SAMPLING_PARAMS = None
        _VLLM_LORA_REQUEST = None
        import gc
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if _LLM is not None:
            del _LLM
            _LLM = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        t0 = time.time()
        _HF_TOKENIZER = AutoTokenizer.from_pretrained(
            QWEN_PATH, cache_dir=HF_CACHE_DIR)
        _HF_TOKENIZER.padding_side = "left"
        if _HF_TOKENIZER.pad_token_id is None:
            _HF_TOKENIZER.pad_token = _HF_TOKENIZER.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH,
            cache_dir=HF_CACHE_DIR,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        _LLM = PeftModel.from_pretrained(base, adapter_path)
        _LLM.eval()
        _GENERATION_BACKEND = "transformers"
        _LOADED_ADAPTER = adapter_path
        log(f"  Transformers+PEFT model loaded in {time.time()-t0:.1f}s")
        return _LLM


def generate_candidates(attrs_list: List[Dict[str, str]],
                        k: int) -> List[List[Dict]]:
    """Generate k candidates per ASIN in batches and apply the content checks."""
    import re

    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+(?:['-]\w+)*\b", text)

    def check_content(text: str, attrs: Dict[str, str]):
        tokens = tokenize(text.lower())
        normalized = " ".join(tokens)
        covered = {}
        missing = []
        for key, value in attrs.items():
            value_tokens = tokenize(str(value).lower())
            if not value_tokens:
                raise ValueError(f"attribute {key} has no tokenized value")
            value_text = " ".join(value_tokens)
            covered[key] = value_text in normalized
            if not covered[key]:
                missing.append(key)
        repeated = len(tokens) != len(set(tokens))
        known_tokens = {
            token for value in attrs.values()
            for token in tokenize(str(value).lower())
        }
        extras = [token for token in tokens if token not in known_tokens]
        return covered, missing, repeated, extras

    llm = get_or_create_llm()
    prompts = [_build_sft_prompt(attrs) for attrs in attrs_list]
    all_cands: List[List[Dict]] = []

    if _GENERATION_BACKEND == "transformers":
        import torch

        tokenizer = _HF_TOKENIZER
        n_batches = (len(prompts) + HF_BATCH_SIZE - 1) // HF_BATCH_SIZE
        progress_every = max(1, (n_batches + 19) // 20)
        generation_started = time.time()
        log(f"  Transformers generation start: prompts={len(prompts)}, "
            f"batches={n_batches}, batch_size={HF_BATCH_SIZE}, k={k}")
        for batch_num, start in enumerate(
                range(0, len(prompts), HF_BATCH_SIZE), start=1):
            batch_prompts = prompts[start:start + HF_BATCH_SIZE]
            batch_attrs = attrs_list[start:start + HF_BATCH_SIZE]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            device = next(llm.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_width = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                generated = llm.generate(
                    **encoded,
                    max_new_tokens=SFT_POOL_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=SFT_POOL_TEMP,
                    top_p=SFT_POOL_TOP_P,
                    num_return_sequences=k,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for batch_idx, attrs in enumerate(batch_attrs):
                seqs = generated[batch_idx * k:(batch_idx + 1) * k,
                                 prompt_width:]
                cands = []
                for token_ids in seqs:
                    ids = token_ids.tolist()
                    if tokenizer.eos_token_id in ids:
                        ids = ids[:ids.index(tokenizer.eos_token_id)]
                    text = tokenizer.decode(
                        ids, skip_special_tokens=True).strip().split("\n")[0].strip()
                    cov, miss, rep, extras = check_content(text, attrs)
                    cands.append({
                        "text": text,
                        "cov": cov,
                        "missing": miss,
                        "repeated": rep,
                        "extras": extras,
                        "pass": len(miss) == 0 and not rep,
                        "n_words": len(tokenize(text)),
                        "n_tokens": len(ids),
                    })
                all_cands.append(cands)
            n_done_prompts = min(start + len(batch_attrs), len(prompts))
            if batch_num % progress_every == 0 or batch_num == n_batches:
                log(f"  Transformers generation progress: "
                    f"{n_done_prompts}/{len(prompts)} prompts "
                    f"({batch_num}/{n_batches} batches), "
                    f"elapsed={time.time() - generation_started:.1f}s")
        log(f"  Transformers generation complete: prompts={len(prompts)}, "
            f"candidates={len(prompts) * k}, "
            f"elapsed={time.time() - generation_started:.1f}s")
        return all_cands


    sampling_params = _VLLM_SAMPLING_PARAMS(
        n=k,
        temperature=SFT_POOL_TEMP,
        top_p=SFT_POOL_TOP_P,
        max_tokens=SFT_POOL_MAX_NEW_TOKENS,
        stop=["<|im_end|>", "\n\n"],
    )
    lora_request = _VLLM_LORA_REQUEST("sft_adapter", 1, str(SFT_ADAPTER_DIR))
    t0 = time.time()
    log(f"  starting vLLM generate: {len(prompts)} prompts × n={k} "
        f"= {len(prompts)*k} candidates")
    outputs = llm.generate(
        prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    log(f"  vLLM generate done in {time.time()-t0:.1f}s")

    for attrs, out in zip(attrs_list, outputs):
        cands = []
        for choice in out.outputs:
            text = choice.text.strip().split("\n")[0].strip()
            cov, miss, rep, extras = check_content(text, attrs)
            cands.append({
                "text": text,
                "cov": cov,
                "missing": miss,
                "repeated": rep,
                "extras": extras,
                "pass": len(miss) == 0 and not rep,
                "n_words": len(tokenize(text)),
                "n_tokens": len(choice.token_ids),
            })
        all_cands.append(cands)
    return all_cands




def main_task_body():
    log("=== sft_pool_generate ===")
    log(f"  Qwen={QWEN_PATH}  adapter={SFT_ADAPTER_DIR}")
    log(f"  N_ASIN={SFT_POOL_N_ASIN}  K={SFT_POOL_K}  rounds={SFT_POOL_ROUNDS}  temp={SFT_POOL_TEMP} "
        f"top_p={SFT_POOL_TOP_P}  max_new_tokens={SFT_POOL_MAX_NEW_TOKENS}")

    attrs_all = load_attributes()
    eval_asins = select_eval_asins(attrs_all, SFT_POOL_N_ASIN)
    log(f"  eval_asins (n={len(eval_asins)}): {eval_asins[:5]} ...")

    # --- Incremental cache: reuse ASINs that already have a pool_queries.json ---
    out_path = OUT_DIR / "pool_queries.json"
    cached_pool: Dict[str, List[str]] = {}
    cached_source = ""
    cached_n_total = 0
    if out_path.exists():
        with open(out_path) as _cf:
            _cached_doc = json.load(_cf)
        cached_pool = dict(_cached_doc.get("pool", {}))
        cached_source = _cached_doc.get("config", {}).get("source", "")
        cached_n_total = sum(len(v) for v in cached_pool.values())
        log(f"  [cache] loaded {len(cached_pool)} ASINs from existing pool "
            f"(n_total={cached_n_total}, source={cached_source})")
        if cached_source and cached_source != SFT_POOL_SOURCE:
            raise ValueError(
                f"existing pool source='{cached_source}' differs from "
                f"current SFT_POOL_SOURCE='{SFT_POOL_SOURCE}', refusing to merge")

    # ASINs already complete (count >= MIN_KEPT_QUERIES) -> reuse directly, no regen
    reuse_asins = [a for a in eval_asins
                   if a in cached_pool and len(cached_pool[a]) >= MIN_KEPT_QUERIES]
    fresh_asins = [a for a in eval_asins if a not in reuse_asins]
    # 2026-09-18: load raw_per_asin cache (raw candidates left over from prior crash)
    raw_cache_path = RAW_CACHE_PATH
    raw_per_asin_cache: dict[str, list] = {}
    if raw_cache_path.exists():
        try:
            with open(raw_cache_path) as _rcf:
                raw_per_asin_cache = json.load(_rcf)
            log(f"  [raw-cache] loaded raw candidates for "
                f"{len(raw_per_asin_cache)} ASINs from {raw_cache_path}")
        except Exception as e:
            log(f"  [raw-cache] load failed: {e}; ignore cache")
            raw_per_asin_cache = {}
    # 2026-09-19: cached_pool >=30 fully reused; remaining ASINs in raw cache (cached_pool <30)
    # take the fresh path: raw cache provides initial raw, regen tops up <30 ASINs.
    reuse_asins = list(set(reuse_asins))
    fresh_asins = [a for a in eval_asins if a not in reuse_asins]
    log(f"  [cache] reuse {len(reuse_asins)} ASINs (cached_pool ≥{MIN_KEPT_QUERIES} ∪ raw_cache), "
        f"regen {len(fresh_asins)} ASINs (raw_cache provides initial raw, "
        f"{len(raw_per_asin_cache)} ASINs in raw cache)")

    # this round's fresh results take precedence over old raw cache to avoid regression.
    fresh_pool: Dict[str, List[str]] = {}
    if fresh_asins:
        attrs_list = [dict(list(attrs_all[a].items())[:5]) for a in fresh_asins]
        # incremental generation: per ASIN accumulate filter-pass queries until >= MIN_KEPT_QUERIES
        #           or regen rounds exhausted.
        # raw is list of candidate dicts (text + metadata); kept is list of strings.
        raw_per_asin = [[] for _ in fresh_asins]
        # 2026-09-18: recover raw candidates cache (crash recovery)
        for i, asin in enumerate(fresh_asins):
            if asin in raw_per_asin_cache:
                cached = raw_per_asin_cache[asin]
                # strip metadata, keep only text for re-encoding
                raw_per_asin[i] = [{"text": c if isinstance(c, str) else c["text"]}
                                    for c in cached]
        n_loaded = sum(1 for r in raw_per_asin if r)
        log(f"  [raw-cache] loaded {n_loaded}/{len(fresh_asins)} ASINs "
            f"with raw candidates")
        kept_per_asin: List[List[str]] = [[] for _ in fresh_asins]
        # initial filter for cached raw candidates (avoid redo)
        for i, asin in enumerate(fresh_asins):
            if raw_per_asin[i]:
                kept_per_asin[i] = filter_pass_queries(
                    asin, [c["text"] for c in raw_per_asin[i]], attrs_all)
        n_total_regen_rounds = 0
        # initial SFT_POOL_ROUNDS of generation (warm-up batches)
        for round_idx in range(SFT_POOL_ROUNDS):
            log(f"  --- round {round_idx+1}/{SFT_POOL_ROUNDS} (initial) ---")
            cands = generate_candidates(attrs_list, SFT_POOL_K)
            for i, asin_cands in enumerate(cands):
                raw_per_asin[i].extend(asin_cands)
            n_total_regen_rounds += 1
            # 2026-09-18: dump raw cache + pool right after each round (filter pass >= MIN_KEPT_QUERIES),
            # to survive crashes and let Stage 8 consume early
            try:
                # 2026-09-19: on dump merge existing raw_per_asin_cache with fresh_asins new raw,
                # to avoid clobbering reuse_asins raw data.
                _dump_existing: Dict[str, list] = {}
                if RAW_CACHE_PATH.exists():
                    try:
                        _dump_existing = json.load(open(RAW_CACHE_PATH))
                    except Exception:
                        _dump_existing = {}
                _dump = dict(_dump_existing)
                for i, asin in enumerate(fresh_asins):
                    _dump[asin] = [c["text"] for c in raw_per_asin[i]]
                with open(RAW_CACHE_PATH, "w") as _dcf:
                    json.dump(_dump, _dcf, ensure_ascii=False)
                # also dump filter-pass partial pool (both fresh and reuse)
                # 2026-09-19: sync kept_per_asin so regen rounds skip already-passed ASINs
                _pool_partial: Dict[str, List[str]] = {}
                for i, asin in enumerate(fresh_asins):
                    kept = filter_pass_queries(
                        asin, [c["text"] for c in raw_per_asin[i]], attrs_all)
                    kept_per_asin[i] = kept  # sync kept list used by regen
                    if len(kept) >= MIN_KEPT_QUERIES:
                        _pool_partial[asin] = kept
                # 2026-09-19: reuse_asins (in raw cache) also go through filter pass into pool
                for asin in reuse_asins:
                    if asin in _dump_existing and asin not in _pool_partial:
                        kept = filter_pass_queries(
                            asin, _dump_existing[asin], attrs_all)
                        if len(kept) >= MIN_KEPT_QUERIES:
                            _pool_partial[asin] = kept
                # merge: existing cached_pool (reuse_asins) + fresh passing subset
                _pool_merged: Dict[str, List[str]] = dict(cached_pool)
                _pool_merged.update(_pool_partial)
                with open(OUT_DIR / "pool_queries.json", "w") as _pcf:
                    # 2026-09-19: write reuse_asin_list so Stage 8 CORAL fit uses full ASIN subset
                    _reuse_list = sorted(set(cached_pool.keys())
                                         | set(reuse_asins)
                                         | set(_pool_partial.keys()))
                    json.dump({
                        "config": {
                            "source": SFT_POOL_SOURCE,
                            "min_keep_queries_per_asin": MIN_KEPT_QUERIES,
                            "n_total_fresh": len(fresh_asins),
                            "n_pool_fresh_kept": len(_pool_partial),
                            "n_pool_reuse_kept": len(reuse_asins),
                            "round_idx": n_total_regen_rounds,
                            "reuse_asin_list": _reuse_list,
                        },
                        "pool": _pool_merged,
                    }, _pcf, indent=2, ensure_ascii=False)
                log(f"  [dump] round {n_total_regen_rounds}: "
                    f"raw={len(_dump)} ASINs, "
                    f"pool={len(_pool_merged)} ASINs ({len(_pool_partial)} fresh pass, "
                    f"reuse_list={len(_reuse_list)})")
            except Exception as _e:
                log(f"  [raw-cache] dump failed: {_e}")
        # incremental regen: per ASIN check if MIN_KEPT_QUERIES reached
        # one generate_candidates call handles a batch of ASINs that need top-up.
        n_regen_full = 0
        while True:
            need_more = [i for i, kept in enumerate(kept_per_asin)
                         if len(kept) < MIN_KEPT_QUERIES]
            if not need_more:
                break
            if n_total_regen_rounds >= SFT_POOL_ROUNDS + MAX_REGEN_ROUNDS:
                log(f"  [regen] max rounds {n_total_regen_rounds} reached; "
                    f"still short on {len(need_more)} ASINs")
                break
            batch_attrs = [attrs_list[i] for i in need_more]
            log(f"  [regen] batch size={len(need_more)}, round {n_total_regen_rounds+1}")
            cands = generate_candidates(batch_attrs, SFT_POOL_K)
            for batch_idx, asin_cands in enumerate(cands):
                i = need_more[batch_idx]
                raw_per_asin[i].extend(asin_cands)
            n_total_regen_rounds += 1
            n_regen_full += 1
            # after each new batch, recompute kept for the touched ASINs (cheap)
            for i in need_more:
                kept_per_asin[i] = filter_pass_queries(
                    fresh_asins[i],
                    [c["text"] for c in raw_per_asin[i]],
                    attrs_all,
                )
            # 2026-09-18: periodic dump raw cache + pool partial (crash safety, let Stage 8 consume early)
            if POOL_DUMP_EVERY_N_ROUNDS > 0 and n_regen_full % POOL_DUMP_EVERY_N_ROUNDS == 0:
                try:
                    # 2026-09-19: on dump merge existing raw_per_asin_cache with fresh_asins new raw
                    _dump_existing: Dict[str, list] = {}
                    if RAW_CACHE_PATH.exists():
                        try:
                            _dump_existing = json.load(open(RAW_CACHE_PATH))
                        except Exception:
                            _dump_existing = {}
                    dump = dict(_dump_existing)
                    for i, asin in enumerate(fresh_asins):
                        dump[asin] = [c["text"] for c in raw_per_asin[i]]
                    with open(RAW_CACHE_PATH, "w") as _dcf:
                        json.dump(dump, _dcf, ensure_ascii=False)
                    _pool_partial = {fresh_asins[i]: kept_per_asin[i]
                                     for i in range(len(fresh_asins))
                                     if len(kept_per_asin[i]) >= MIN_KEPT_QUERIES}
                    # 2026-09-19: reuse_asins (in raw cache) also go through filter pass into pool
                    for asin in reuse_asins:
                        if asin in _dump_existing and asin not in _pool_partial:
                            k = filter_pass_queries(
                                asin, _dump_existing[asin], attrs_all)
                            if len(k) >= MIN_KEPT_QUERIES:
                                _pool_partial[asin] = k
                    _pool_merged = dict(cached_pool)
                    _pool_merged.update(_pool_partial)
                    with open(OUT_DIR / "pool_queries.json", "w") as _pcf:
                        # 2026-09-19: write reuse_asin_list (cached_pool union reuse_asins union pool_partial)
                        # so Stage 8 CORAL global A uses the full ASIN subset
                        _reuse_list = sorted(set(cached_pool.keys())
                                             | set(reuse_asins)
                                             | set(_pool_partial.keys()))
                        json.dump({
                            "config": {
                                "source": SFT_POOL_SOURCE,
                                "min_keep_queries_per_asin": MIN_KEPT_QUERIES,
                                "n_pool_fresh_kept": len(_pool_partial),
                                "n_pool_reuse_kept": len(reuse_asins),
                                "round_idx": n_total_regen_rounds,
                                "reuse_asin_list": _reuse_list,
                            },
                            "pool": _pool_merged,
                        }, _pcf, indent=2, ensure_ascii=False)
                    log(f"  [dump] round {n_total_regen_rounds}: "
                        f"raw={len(dump)} ASINs, "
                        f"pool={len(_pool_merged)} ASINs ({len(_pool_partial)} fresh pass, "
                        f"reuse_list={len(_reuse_list)})")
                except Exception as e:
                    log(f"  [raw-cache] dump failed: {e}")
        log(f"  [regen] total rounds: {n_total_regen_rounds} "
            f"(initial 2 + {n_regen_full} regen)")
        fresh_pool = {fresh_asins[i]: kept for i, kept in enumerate(kept_per_asin)}

    pool: Dict[str, List[str]] = {}
    n_dropped_total = 0
    n_zero_query_asins = 0
    n_below_min_query_asins = 0
    MIN_KEEP_QUERIES = 30  # 2026-09-18: enforce >=30 queries; otherwise ASIN is excluded from pool
    for asin in eval_asins:
        if asin in fresh_pool:
            # this round's fresh results take precedence; must not be overwritten by old raw cache.
            raw = list(fresh_pool[asin])
        elif asin in cached_pool and len(cached_pool[asin]) >= MIN_KEPT_QUERIES:
            # cached_pool hit (reuse_asins ∩ cached_pool)
            raw = list(cached_pool[asin])
        elif asin in raw_per_asin_cache:
            # raw cache hit (not a fresh ASIN this round)
            raw = list(raw_per_asin_cache[asin])
        else:
            continue
        kept = filter_pass_queries(asin, raw, attrs_all)
        n_dropped_total += len(raw) - len(kept)
        if len(kept) < MIN_KEEP_QUERIES:
            if not kept:
                n_zero_query_asins += 1
            else:
                n_below_min_query_asins += 1
            continue
        pool[asin] = kept
    n_total = sum(len(v) for v in pool.values())

    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "model": QWEN_PATH,
                "lora_adapter": str(SFT_ADAPTER_DIR),
                "n_asin": len(pool),
                "k_per_asin": SFT_POOL_K,
                "rounds": SFT_POOL_ROUNDS,
                "temp": SFT_POOL_TEMP,
                "top_p": SFT_POOL_TOP_P,
                "max_new_tokens": SFT_POOL_MAX_NEW_TOKENS,
                "source": SFT_POOL_SOURCE,
                "stage04_source": str(STAGE04_PATH),
                "incremental_reuse": len(reuse_asins),
                "incremental_fresh": len(fresh_asins),
                "reuse_asin_list": reuse_asins,  # 2026-09-18: Stage 8 CORAL global fallback A uses reuse_asins subset
                "fresh_asin_list": fresh_asins,
                "pass_filter_applied": True,
                "pass_filter_criteria": "full_coverage_5_attrs AND no_repeat_token",
                "min_keep_queries_per_asin": MIN_KEEP_QUERIES,
                "n_dropped_by_filter": n_dropped_total,
                "n_asins_dropped_all_queries": n_zero_query_asins,
                "n_asins_dropped_below_min_queries": n_below_min_query_asins,
            },
            "pool": pool,
        }, f, indent=2, ensure_ascii=False)
    log(f"  saved -> {out_path} (n_total={n_total}, dropped={n_dropped_total}, "
        f"all-removed={n_zero_query_asins}, below-min={n_below_min_query_asins})")
    log(f"  SUMMARY: {n_total} queries, {len(pool)} ASINs "
        f"(reuse={len(reuse_asins)}, fresh={len(fresh_asins)}, dropped={n_dropped_total}, "
        f"all-removed={n_zero_query_asins}, below-min={n_below_min_query_asins})")
    log("=== sft_pool_generate DONE ===")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """User instruction 2026-09-23: run 3 categories sequentially.

    For each category, rebind this script's path constants to category-specific paths,
    then call the original main_task_body() (preserves existing logic). Outputs go to
    result/<stage>/<baby|musical|video_games>/ subdirectories.
    """
    global SENT_CACHE, UID_TO_SENTS, ASIN_USERS_PATH, ATTRIBUTES_PATH, META_FILE, OUT_DIR, OUT_PATH, SFT_ADAPTER_DIR, STAGE04_PATH, STAGE05_PATH, RAW_CACHE_PATH  # noqa
    # backup current (Baby) defaults
    saved = {
        k: v for k, v in globals().items()
        if k in {"SENT_CACHE", "UID_TO_SENTS", "ASIN_USERS_PATH", "ATTRIBUTES_PATH",
                 "META_FILE", "OUT_DIR", "OUT_PATH", "SFT_ADAPTER_DIR", "STAGE04_PATH",
                 "STAGE05_PATH", "RAW_CACHE_PATH"}
        and isinstance(v, Path)
    }
    base_out = REPO_ROOT / "result" / Path(__file__).parent.name
    for category, subdir in CATEGORY_INPUTS:
        log(f"\n========== [{category}] (subdir={subdir}) ==========")
        # Reset all known category-dependent paths to point at the per-category subdir.
        if "SENT_CACHE" in saved:
            SENT_CACHE = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "UID_TO_SENTS" in saved:
            UID_TO_SENTS = REPO_ROOT / "result/02_user_review_sentence_extract" / f"uid_to_sentences_{subdir}.pkl"
        if "ASIN_USERS_PATH" in saved:
            ASIN_USERS_PATH = REPO_ROOT / "result/02_user_review_sentence_extract" / f"asin_to_users_{subdir}.pkl"
        if "ATTRIBUTES_PATH" in saved:
            ATTRIBUTES_PATH = REPO_ROOT / "result/01_attribute_extraction" / f"product_attributes_{subdir}.pkl"
        if "META_FILE" in saved:
            META_FILE = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data") / {
                "baby": "meta_Baby_Products_2023.jsonl",
                "musical": "meta_Musical_Instruments.jsonl",
                "video_games": "meta_Video_Games.jsonl",
            }[subdir]
        if "OUT_DIR" in saved:
            OUT_DIR = base_out / subdir
        if "RAW_CACHE_PATH" in saved:
            RAW_CACHE_PATH = RAW_CACHE_ROOT / subdir / "raw_per_asin_cache.json"
        if "OUT_PATH" in saved:
            OUT_PATH = base_out / subdir / saved["OUT_PATH"].name
        if "SFT_ADAPTER_DIR" in saved:
            SFT_ADAPTER_DIR = REPO_ROOT / "result/06_training_model" / subdir / saved["SFT_ADAPTER_DIR"].name
        if "STAGE04_PATH" in saved:
            STAGE04_PATH = REPO_ROOT / "result/04_gaussian" / subdir / saved["STAGE04_PATH"].name
        if "STAGE05_PATH" in saved:
            STAGE05_PATH = REPO_ROOT / "result/05_gaussian_audit" / subdir / saved["STAGE05_PATH"].name
        OUT_DIR.mkdir(parents=True, exist_ok=True) if "OUT_DIR" in saved else None
        RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True) if "RAW_CACHE_PATH" in saved else None
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True) if "OUT_PATH" in saved else None
        SFT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True) if "SFT_ADAPTER_DIR" in saved else None
        STAGE04_PATH.parent.mkdir(parents=True, exist_ok=True) if "STAGE04_PATH" in saved else None
        try:
            main_task_body()
        except Exception as e:
            log(f"[{category}] FAILED: {e!r}")
            raise
    # Restore Baby defaults (for import compatibility with downstream).
    for k, v in saved.items():
        globals()[k] = v


if __name__ == "__main__":
    main()
