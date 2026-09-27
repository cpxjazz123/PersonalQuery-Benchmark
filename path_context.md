# Replace absolute paths in parent repo root 01-14 .py files

## Target directory
`/home/wlia0047/ar57/wenyu/PersoanlQuery/` (NOT `PersonalQuery-pipeline/` — leave that alone for now).

## Goal
Replace absolute paths in `0[1-9]_*/*.py` and `1[0-4]_*/*.py` files at the repo root with portable equivalents. Do NOT touch `/home/wlia0047/ar57/wenyu/PersoanlQuery/PersonalQuery-pipeline/`.

## Path replacement rules

### 1. REPO_ROOT (in-repo paths under PersoanlQuery itself)

Original:
```python
REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery")
```

Replace with (top-level stage files like `01_attribute_extraction/extract_product_attrs.py`):
```python
REPO_ROOT = Path(__file__).resolve().parent.parent
```

The `.parent.parent` accounts for `<REPO_ROOT>/<NN_stage>/<script>.py` — parent is the stage dir, parent.parent is REPO_ROOT.

### 2. In-repo absolute paths (under REPO_ROOT)

Original:
```python
Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/result/03_spacy_encode/cohort2_mlp16_30_30ep.pt")
Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/03_spacy_encode")
```

Replace with:
```python
REPO_ROOT / "result" / "03_spacy_encode" / "cohort2_mlp16_30_30ep.pt"
REPO_ROOT / "03_spacy_encode"
```

### 3. sys.path.insert with absolute in-repo path

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

### 4. Data / cache directories (hj82* paths, NOT in repo)

These point to data and scratch directories outside the repo. Use environment variables with sensible defaults.

Original:
```python
DATA_DIR = Path("/home/wlia0047/hj82/wenyu/PersoanlQuery/data")
SCRATCH = Path('/home/wlia0047/hj82_scratch2/wenyu')
CACHE_DIR = Path("/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache")
HF_CACHE_DIR = "/home/wlia0047/hj82/wenyu/hf_cache"
RAW_CACHE_ROOT = Path("/home/wlia0047/hj82_scratch2/wenyu/stage07_pool_cache")
# ... many similar
```

Replace with:
```python
import os
DATA_DIR = Path(os.environ.get("PQ_DATA_DIR", "/home/wlia0047/hj82/wenyu/PersoanlQuery/data"))
SCRATCH = Path(os.environ.get("PQ_SCRATCH", "/home/wlia0047/hj82_scratch2/wenyu"))
CACHE_DIR = SCRATCH / "pcfg_cache"  # if appropriate, OR keep env var
HF_CACHE_DIR = os.environ.get("PQ_HF_CACHE", "/home/wlia0047/hj82/wenyu/hf_cache")
RAW_CACHE_ROOT = SCRATCH / "stage07_pool_cache"
```

Use these env var names consistently:
- `PQ_DATA_DIR` for data dir (default `/home/wlia0047/hj82/wenyu/PersoanlQuery/data`)
- `PQ_SCRATCH` for scratch root (default `/home/wlia0047/hj82_scratch2/wenyu`)
- `PQ_HF_CACHE` for HF cache (default `/home/wlia0047/hj82/wenyu/hf_cache`)

For subdirs under scratch (e.g. `pcfg_cache`, `stage07_pool_cache`, `tmp/gec_in.jsonl`, `pool_query_cache`), build them relative to `SCRATCH` or to a named base. For deep paths like `/home/wlia0047/hj82_scratch2/wenyu/pcfg_cache_baby`, write:
```python
PCFG_CACHE_BABY = SCRATCH / "pcfg_cache_baby"
```

### 5. pq_env Python interpreter

Original (3 occurrences in 04_gaussian/trainable_per_user_gaussian.py and 09_sercl_user_profile/):
```python
nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python \
SERCL_GEC_SUBPROCESS_PY = "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python"
```

Replace with environment variable:
```python
PQ_PYTHON = os.environ.get("PQ_PYTHON", "/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python")
```

Then use `PQ_PYTHON` in shell commands and the constant. If used in a docstring/usage example, also replace.

### 6. /tmp/ paths

For `/home/wlia0047/hj82_scratch2/wenyu/tmp/gec_in.jsonl`:
```python
GEC_IN_JSONL = SCRATCH / "tmp" / "gec_in.jsonl"
```

### 7. Inside docstrings / usage examples

Replace there too. Read the line carefully — if it's inside `"""..."""` docstring, the replacement still applies.

## Verification (must pass after edits)

For each file you touched:
```bash
grep -nE "/home/wlia0047/ar57/wenyu/PersoanlQuery" <file>  # expect ONLY __file__ references or no matches
grep -nE "/home/wlia0047/hj82" <file>  # expect ONLY env var defaults (os.environ.get(...)) or no matches
python3 -c "import ast; ast.parse(open('<file>').read())"  # silent
```

A line containing `"/home/wlia0047/ar57/wenyu/PersoanlQuery"` is OK if it's inside `os.environ.get("X", "/home/.../PersoanlQuery")` — the env var default is allowed. Bare `Path("/home/.../PersoanlQuery/...")` assignments are NOT OK — replace them.

## Rules

- One file per agent.
- Read the file in FULL first to understand context before editing.
- Use the `edit` tool for surgical replacements.
- Don't change logic, variable names, formatting outside of replaced lines.
- Don't add new dependencies beyond `os` (already imported or trivially added).
- Don't touch files outside your assignment. Don't touch `PersonalQuery-pipeline/`.
- Don't add new documentation.

## Deliverable

Report:
- Number of in-repo absolute paths replaced.
- Number of hj82/scratch paths replaced with env vars.
- Confirmation that `grep` and `ast.parse` pass.