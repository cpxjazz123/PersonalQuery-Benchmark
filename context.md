# Cleanup Chinese in PersonalQuery-pipeline

## Goal
Remove ALL Chinese characters from Python files under `/home/wlia0047/ar57/wenyu/PersoanlQuery/PersonalQuery-pipeline/` (Stages 01-14). After your work, `grep -P '[\x{4e00}-\x{9fff}]' <file>` must return 0 matches.

## Strategy
1. **First preference: DELETE** redundant Chinese.
   - Dev-history comments like `# 2026-09-23: ...` belong in git log, not code. Delete them.
   - Docstring content that merely restates the function/class name in Chinese (e.g. `"""Stage 04 — 拟合 32d per-user Gaussian..."""` where the function name already says "fit_per_user_gaussian") can be shortened to a brief English summary or removed.
2. **If the Chinese carries essential information** (API contract, parameter meaning, behavior detail that is NOT obvious from the symbol name), **translate to concise English**.
   - Keep technical names (asin, uid, sigma, logp_delta, q95, chi2, GECToR, LoRA, etc.) unchanged.
   - Keep version dates like `2026-09-23` unchanged.
   - Keep English terms embedded in Chinese (e.g. `规则`, `权重`, `维度`) — translate to English technical equivalents (`rule`, `weight`, `dimension`).
3. **DO NOT** change:
   - Variable names, function names, class names (they may already be English).
   - Print/log statements (user explicitly said log is not the concern — but avoid introducing new Chinese).
   - Code logic, indentation, or whitespace outside of lines that contain Chinese.
   - The `04_gaussian_cohort3mlp*.py` files have minimal Chinese — easy cases.

## File-by-file ownership
Each agent owns ONE file. Read the file in full first to understand context before editing. Use the `edit` tool for surgical changes, `write` only if you need to rewrite a large contiguous block.

## Verification
After editing, run:
```bash
grep -cP '[\x{4e00}-\x{9fff}]' <your-file>
```
Expected output: `0`. If non-zero, find and fix remaining Chinese.

Also run a syntax check:
```bash
python3 -c "import ast; ast.parse(open('<your-file>').read())"
```
Expected: no error (silent output).

## Deliverable
- One file modified.
- File passes both checks above.
- Report back the count of Chinese lines removed vs translated.

## Don't
- Don't touch any file outside your assignment.
- Don't add new dependencies.
- Don't rewrite non-Chinese parts of the file.