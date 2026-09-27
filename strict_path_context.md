# STRICT absolute-path elimination in parent repo root 01-14 / 15 / llm_client

## Hard rule (user instruction 2026-09-27)
NO absolute paths may remain anywhere in `.py` files, including inside `os.environ.get("VAR", "DEFAULT")` default strings. A user moving this project to a new environment must be able to run it with only `export PQ_*=...` overrides — never need to edit source.

## Target directory
`/home/wlia0047/ar57/wenyu/PersoanlQuery/` (NOT `PersonalQuery-pipeline/` — leave that alone).

## Files in scope
```
01_attribute_extraction/extract_product_attrs.py
02_user_review_sentence_extract/extract_user_sentences.py
03_spacy_encode/syntax_encoder.py
04_gaussian/fit_per_user_gaussian.py
04_gaussian/fit_per_user_gaussian_cohort3mlp.py
04_gaussian/fit_per_user_gaussian_cohort3mlp_lowrankdiag.py
04_gaussian/trainable_per_user_gaussian.py
05_gaussian_audit/raw_cov_validity.py
06_training_model/sft_pipeline.py
07_gen_query/sft_pool_generate.py
08_select_query/syntax_select.py
09_sercl_user_profile/gector_subprocess.py
09_sercl_user_profile/sercl_user_profile.py
10_typo_injection/typo_inject.py
11_syntactic_evaluation/build_asin_to_doc.py
11_syntactic_evaluation/syntax_subspace_retrieval_unified.py
12_typo_evaluation/typo_retrieval_eval.py
13_syntactic_rerank/syntactic_rerank_eval.py
14_typo_rerank/typo_rerank_eval.py
15_build_dataset/build_dataset.py
llm_client.py
```

(21 files total)

## Replacement policy

### Rule A — REPO_ROOT
Already done:
```python
REPO_ROOT = Path(__file__).resolve().parent.parent
```
Keep as-is. Don't touch.

### Rule B — In-repo absolute paths (under REPO_ROOT)
Original anywhere:
```python
Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/...")
"/home/wlia0047/ar57/wenyu/PersoanlQuery/query_samples_100.json"
"/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode"
```
Replace with:
```python
REPO_ROOT / "result" / "03_spacy_encode" / "..."
REPO_ROOT / "query_samples_100.json"
REPO_ROOT / "03_spacy_encode"
```

### Rule C — DATA / SCRATCH / HF cache defaults
**HARD requirement: no absolute paths in defaults.** Replace with REPO_ROOT-relative defaults.

Original:
```python
DATA_DIR = Path(os.environ.get("PQ_DATA_DIR", "/home/wlia0047/hj82/wenyu/PersoanlQuery/data"))
SCRATCH = Path(os.environ.get("PQ_SCRATCH", "/home/wlia0047/hj82_scratch2/wenyu"))
HF_CACHE_DIR = os.environ.get("PQ_HF_CACHE", "/home/wlia0047/hj82/wenyu/hf_cache")
```

New (relies on REPO_ROOT being defined):
```python
DATA_DIR = Path(os.environ.get("PQ_DATA_DIR", str(REPO_ROOT / "data")))
SCRATCH = Path(os.environ.get("PQ_SCRATCH", str(REPO_ROOT / "scratch")))
HF_CACHE_DIR = os.environ.get("PQ_HF_CACHE", str(REPO_ROOT / "scratch" / "hf_cache"))
```

For sub-paths that previously inherited hj82_scratch2 (e.g. `pcfg_cache`, `stage07_pool_cache`, `stage11_corpus_cache`, `stage11_retrieval_cache`, `stage12_typo_cache`, `stage13_smoke`, `stage14_smoke`, `typo_eval`, `gaussian_vades/multiretrieval_embeds`, `RAG/Qwen3-Reranker-8B`, `tmp/`, `pool_query_cache`, `trainable_gaussian.log`, `coral_asin_cohort2_mlp16_30`, `external/gector/src`):

Build them relative to SCRATCH (or HF_CACHE_DIR) using `/` operator, NOT as separate env vars with hardcoded defaults. Example:
```python
PCFG_CACHE = SCRATCH / "pcfg_cache"
PCFG_CACHE_BABY = SCRATCH / "pcfg_cache_baby"
CACHE_DIR = SCRATCH / "pcfg_cache"
RAW_CACHE_ROOT = SCRATCH / "stage07_pool_cache"
ASIN_TO_DOC_CACHE = SCRATCH / "stage11_corpus_cache" / "baby" / "asin_to_doc.json"
TOPK_SAVE_DIR = SCRATCH / "stage11_retrieval_cache" / "baby" / "top100_cache"
EMBED_CACHE_DIR = SCRATCH / "gaussian_vades" / "multiretrieval_embeds"
TOPK_DIR_ORIG = SCRATCH / "stage12_typo_cache" / "baby" / "top100_cache_orig"
TOPK_DIR_TYPO = SCRATCH / "stage12_typo_cache" / "baby" / "top100_cache_typo"
TYPO_TOPK_DIR = SCRATCH / "stage12_typo_cache" / "baby" / "top100_cache_typo"
SMOKE_OUT_DIR = SCRATCH / "stage13_smoke" / subdir
STAGE13_RESULTS_DIR = SCRATCH / "stage13_smoke" / subdir
PER_QUERY_OUT = SCRATCH / "typo_eval" / "per_query_stage11.json"
SUMMARY_OUT = SCRATCH / "typo_eval" / "retrieval_summary_stage11.json"
VOLATILITY_OUT = SCRATCH / "typo_eval" / "volatility_stage11.json"
QWEN_RAG = SCRATCH / "RAG" / "Qwen3-Reranker-8B"
HF_HUB_CACHE = os.environ.get("PQ_HF_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
TEMP_DIR = SCRATCH / "tmp"
GEC_IN_JSONL = SCRATCH / "tmp" / "gec_in.jsonl"
GEC_OUT_JSONL = SCRATCH / "tmp" / "gec_out.jsonl"
POOL_QUERY_CACHE = SCRATCH / "pool_query_cache"
CORAL_ASIN_PATH = SCRATCH / "coral_asin_cohort2_mlp16_30" / "coral_asin_cohort2_mlp16_30.npz"
CORAL_ART_DIR = SCRATCH / "coral_asin_cohort2_mlp16_30"
GECTOR_EXTERNAL_SRC = SCRATCH / "external" / "gector" / "src"
GECTOR_WEIGHTS = HF_CACHE_DIR / "gector" / "gector-2024-roberta-large.th"
GECTOR_VOCAB = HF_CACHE_DIR / "gector" / "vocab"
TRAINABLE_GAUSSIAN_LOG = SCRATCH / "trainable_gaussian.log"
```

For log path strings: use `str(SCRATCH / "logs")` instead of `"/home/.../logs/"`.

### Rule D — pq_env Python interpreter

Original:
```python
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
SERCL_GEC_SUBPROCESS_PY = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"
${PQ_PYTHON:-/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python}
```

Replace with `sys.executable`:
```python
# Python constant:
PQ_PYTHON = os.environ.get("PQ_PYTHON", sys.executable)

# In docstring shell examples:
${PQ_PYTHON:-$(which python3)}
# or just:
python3
```

### Rule E — _tv.__path__ override (Stage 11)
Original:
```python
_tv.__path__ = ["/home/wlia0047/ar57_scratch/wenyu/pq_env/lib/python3.10/site-packages/torchvision"]
```
Replace with:
```python
import torchvision
_tv.__path__ = list(torchvision.__path__)  # no override needed; keep site-packages layout
```

If override is truly required for some venv, derive it from `sys.exec_prefix` or `sysconfig`:
```python
import sysconfig
_tv.__path__ = [Path(sysconfig.get_paths()["purelib"]) / "torchvision"]
```

### Rule F — sys.path.insert with in-repo path
Original:
```python
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery")
sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode")
```
Replace with:
```python
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "03_spacy_encode"))
```

For Stage 09 `sys.path.insert(0, "/home/wlia0047/hj82_scratch2/wenyu/external/gector/src")`:
```python
GECTOR_EXTERNAL_SRC = SCRATCH / "external" / "gector" / "src"
sys.path.insert(0, str(GECTOR_EXTERNAL_SRC))
```

### Rule G — docstring usage examples (e.g. `cd /home/...`)
Replace `cd /home/wlia0047/ar57/wenyu/PersoanlQuery` with `cd <REPO_ROOT>`.
Replace other absolute paths in docstrings with `<REPO_ROOT>/...`, `${VAR}` defaults, or "see env.example".

### Rule H — llm_client.py

Original (4 absolute paths):
```python
_tv.__path__ = ["/home/wlia0047/ar57_scratch/wenyu/pq_env/lib/python3.10/site-packages/torchvision"]
os.environ["OUTLINES_CACHE_DIR"] = "/fs04/scratch2/hj82/wenyu/outlines_cache"
DEFAULT_QWEN_MODEL = "/home/wlia0047/hj82_scratch2/wenyu/RAG/Qwen3-Reranker-8B"
DEFAULT_SFT_ADAPTER = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/06_training_model/sft_lora")
```

Replace with:
```python
import torchvision
_tv.__path__ = list(torchvision.__path__)

# OUTLINES_CACHE_DIR: use env var with no hardcoded default
os.environ.setdefault("OUTLINES_CACHE_DIR", str(Path("./outlines_cache").resolve()))

DEFAULT_QWEN_MODEL = os.environ.get("PQ_QWEN_RAG", str(Path("./RAG/Qwen3-Reranker-8B").resolve()))

DEFAULT_SFT_ADAPTER = REPO_ROOT / "result" / "06_training_model" / "sft_lora"
```

For `llm_client.py`, REPO_ROOT is `Path(__file__).resolve().parent`.

## Verification (must pass after edits)

For each file:
```bash
grep -nE '/home/wlia0047|/fs04|/opt/|/usr/local|/scratch[0-9]?/|/mnt/|/data/|/root/' <file>
# MUST return ZERO matches (no absolute path anywhere, including env-var defaults)
python3 -c "import ast; ast.parse(open('<file>').read())"
# MUST exit 0 silently
```

Any remaining hit = incomplete edit. Re-read and fix.

## Rules

- Read the file in FULL first.
- Use the `edit` tool for surgical replacements.
- Don't change logic, variable names, formatting outside of replaced lines.
- Don't add new dependencies beyond `os`, `sys`, `sysconfig`, `pathlib.Path`, `torchvision`.
- Don't touch files outside your assignment.
- Don't touch `PersonalQuery-pipeline/`.

## Deliverable

Report:
- Number of absolute path occurrences eliminated.
- Confirmation grep + ast.parse pass.