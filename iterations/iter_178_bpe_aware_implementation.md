# Iteration 178 — E5 BPE-aware Error Injection 实现

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: Stage 05 真正实现 subword-aware error injection (补救 iter #44 phantom + iter #177 audit 发现)

## §A 审稿意见（尖锐批评）

### 问题: iter #44 描述的 BPE-aware 注入从未 commit — iter #178 真正落地

**严重程度**: Major（iter #44 是 phantom spec, iter #177 audit 已记录; iter #178 补救）

**iter #177 关键发现**:
- `token_level_lambdamart_user_based.py` 与 `apply_lambdamart_userbased_noisy.py` 仅 4 个 distinct commits
- `compute_bpe_token_diff` / `_get_bpe_tokenizer` / `BPE_SIMILARITY_THRESHOLD=0.8` 全部未 commit
- 实际 char-Jaccard 自 initial commit 就在但**不是** BPE token overlap
- loop.md §11 backlog P1「E5 subword 敏感场景未模拟」**从未**真正实现

**iter #178 真实交付**:

1. **新增 `_get_bpe_tokenizer()`**: lazy-load `intfloat/e5-base-v2` tokenizer via `transformers.AutoTokenizer`
2. **新增 `compute_bpe_token_diff()`**: 计算两词的 BPE-tokenization Jaccard similarity (返回 ∈ [0, 1])
3. **`process_batch()` 新增 `bpe_aware: bool = False` 参数**: 当 True 时仅注入 `bpe_similarity < 0.8` 的错误
4. **CLI 新增 `--bpe-aware` flag**: argparse 接 main()
5. **applied_error 输出 schema**: 加 `bpe_similarity` + `bpe_impact` 字段 (only when bpe_aware=True)

## §B 对应代码缺陷

| 问题 | 文件 | 修复前 | 修复后 |
|------|------|--------|--------|
| 无 BPE tokenizer 加载 | `apply_lambdamart_userbased_noisy.py` | ❌ | ✅ `_get_bpe_tokenizer()` lazy-load `intfloat/e5-base-v2` |
| 无 BPE Jaccard 函数 | 同上 | ❌ | ✅ `compute_bpe_token_diff()` Jaccard over BPE tokens |
| 无 subword-aware gating | 同上 `process_batch` | word-level only | ✅ `bpe_aware=True` 时仅注入 subword-divergent 错误 (similarity < 0.8) |
| 无 CLI 入口 | 同上 `__main__` | sys.argv only | ✅ argparse `--bpe-aware` flag |

## §C 本轮代码优化

### C.1 修改文件清单

| 文件 | 行号 | 改动 |
|------|------|------|
| `PersoanlQuery/05_inject_noisy/common/apply_lambdamart_userbased_noisy.py` | L8-19 + L20-71 | import + 3 个 BPE 常量/函数 |
| 同上 `process_batch` | L376-403 | 加 `bpe_aware: bool = False` 参数 + docstring |
| 同上 `process_batch` body | L425-432 | 加 bpe_similarity + bpe_impact 计算 + gating |
| 同上 `applied_error` 输出 | L450-456 | 输出 schema 加 2 字段 (gated by bpe_aware) |
| 同上 `main()` | L489-498 | 接受 bpe_aware 参数, header 加 BPE-aware 标识 |
| 同上 `main()` 调用处 | L530 | `process_batch(..., bpe_aware=bpe_aware)` |
| 同上 `__main__` | L557-571 | argparse + `--bpe-aware` flag |
| `PersoanlQuery/05_inject_noisy/common/_smoke_bpe_aware_injection.py` (NEW) | 全文 | 7-case smoke test (transformers availability-aware) |

### C.2 BPE 函数设计 (核心代码)

```python
_BPE_TOKENIZER_NAME = 'intfloat/e5-base-v2'
_BPE_SIMILARITY_THRESHOLD = 0.8
_BPE_TOKENIZER = None  # lazy-loaded


def _get_bpe_tokenizer():
    """Lazy-load E5 BPE tokenizer for subword impact scoring (iter #178).
    Per loop.md §1 Rule 7: raises on failure, no fallback.
    """
    global _BPE_TOKENIZER
    if _BPE_TOKENIZER is None:
        from transformers import AutoTokenizer
        _BPE_TOKENIZER = AutoTokenizer.from_pretrained(_BPE_TOKENIZER_NAME)
    return _BPE_TOKENIZER


def compute_bpe_token_diff(original_word: str, corrected_word: str) -> float:
    """Compute BPE-tokenization Jaccard similarity (iter #178).
    Returns float in [0, 1]:
        1.0 → identical BPE token sequences (no subword impact)
        0.0 → disjoint BPE token sequences (max subword divergence)
    """
    tok = _get_bpe_tokenizer()
    orig_tokens = tok.tokenize(original_word)
    corr_tokens = tok.tokenize(corrected_word)
    if not orig_tokens or not corr_tokens:
        return 0.0  # per loop.md §1 Rule 7: explicit, no fallback
    orig_set = set(orig_tokens)
    corr_set = set(corr_tokens)
    intersection = len(orig_set & corr_set)
    union = len(orig_set | corr_set)
    return intersection / union if union > 0 else 0.0
```

### C.3 process_batch bpe_aware gating (核心代码)

```python
for token in clean_tokens:
    err = find_similar_error(token, user_errors)
    if err:
        # iter #178: BPE-aware gating
        if bpe_aware:
            orig = err.get('original') or err.get('corrected') or token
            corr = err.get('corrected') or err.get('original') or token
            bpe_similarity = compute_bpe_token_diff(orig, corr)
            bpe_impact = bpe_similarity < _BPE_SIMILARITY_THRESHOLD
            if not bpe_impact:
                continue  # Skip: not subword-impactful
        best_token = token
        error_case = err
        break
```

### C.4 applied_error 输出 schema (when bpe_aware=True)

```python
applied_error = {
    'original': error_case.get('original'),
    'corrected': error_case.get('corrected'),
    'error_type': error_case.get('error_type', 'writing_error'),
    'bpe_similarity': bpe_similarity,  # NEW: iter #178
    'bpe_impact': bpe_impact,          # NEW: iter #178 (bool)
}
```

## §D 验证

### D.1 py_compile
```bash
$ python3 -m py_compile PersoanlQuery/05_inject_noisy/common/apply_lambdamart_userbased_noisy.py
OK

$ python3 -m py_compile PersoanlQuery/05_inject_noisy/common/_smoke_bpe_aware_injection.py
COMPILE_OK
```

### D.2 smoke test (7 cases)
```bash
$ PYTHONPATH=... python3 _smoke_bpe_aware_injection.py
[Case 1-3] SKIPPED — 'transformers' not installed (in this dev env)
[Case 4] process_batch signature: (model, tasks, category, bpe_aware: bool = False)
[Case 5] argparse accepts --bpe-aware: True
[Case 6] process_batch emits 'bpe_similarity' + 'bpe_impact' (verified by AST scan)
[Case 7] bpe_aware=False legacy path correctly omits BPE fields
ALL CASES PASS (BPE cases skipped if 'transformers' unavailable)
```

### D.3 后续验证 (需 Stage 06 用的 conda env)

Stage 06 build/eval 用的 conda env 装了 `transformers` + `sentence-transformers` (用于 E5 embedder)。
需要在该 env 下重跑 smoke test 让 Case 1-3 (BPE 数值验证) 也跑通：

```bash
# 找 Stage 06 用的 python:
# (一般是在 sbatch SLURM env 或 user-conda env)
# 假设 /home/wlia0047/.conda/envs/<name>/bin/python3
$ /home/wlia0047/.conda/envs/<name>/bin/python3 _smoke_bpe_aware_injection.py
[Case 1] identical 'bottle' vs 'bottle' → bpe_sim=1.0000
[Case 2] 'bottle' vs 'bottel' (phonetic) → bpe_sim=<0.5  (divergent)
[Case 3] 'pacifier' vs 'PACIFIER' (case-only) → bpe_sim≈1.0
ALL CASES PASS
```

后续 iter (iter #179) 可在 conda env 下 re-validate + 用 Stage 05 driver 跑 end-to-end BPE-aware 注入 pipeline。

## §E 影响与限制

### E.1 实现已完成 (plumbing level)

- ✅ BPE tokenizer lazy-loader
- ✅ BPE-token Jaccard computation
- ✅ process_batch bpe_aware gating
- ✅ CLI --bpe-aware flag
- ✅ applied_error schema extension

### E.2 限制

- **transformers 包依赖**: 当前 dev env 没装, 需 Stage 06 conda env 跑数值验证 (Case 1-3)。
- **未在 Stage 05 driver 实际跑过**: smoke test 只验证 plumbing; 实际 noisy_query.json 重生成需用 Stage 06 的 sbatch 环境。
- **未量化 bpe_aware=True vs False 对 E5/MiniLM/STAR 的 Δ 分布影响**: 这需要 (a) Stage 05 重生成 noisy queries + (b) Stage 7 retrieval + (c) Stage 8 Δ Range 分析, 是 multi-hour infra, 留给 iter #180。
- **`process_batch` 仍用规则匹配 `find_similar_error`**: LambdaMART model 参数仍 unused (iter #47 audit 已记录); iter #178 不动此 dead path。

### E.3 后续 iter 候选

- **iter #179**: 在 Stage 06 conda env 重跑 smoke test + 端到端单 query BPE-aware 注入
- **iter #180**: Stage 05 + 6 + 7 全链路重跑, 比较 bpe_aware=True vs False 在 E5/MiniLM/STAR 的 Δ Range (paper §3.2 E5 Δ=11.16 的 subword sensitivity 实证)
- **iter #181**: paper_claims_audit.py `BPE_aware_Error_Injection` claim 加新 evidence (`_smoke_bpe_aware_injection.py`, `compute_bpe_token_diff`, `--bpe-aware` CLI flag), 状态 unverified → verified

## §F Git Commit

- iter #178: implement E5 BPE-aware error injection (apply_lambdamart_userbased_noisy.py: _get_bpe_tokenizer + compute_bpe_token_diff + process_batch bpe_aware param + --bpe-aware CLI flag + applied_error schema extension); smoke test plumbing verified