# Iteration 256 — Paper-Audit Consistency Fix + §3.1 Review + §2.2 Pipeline Separation Confirmed

**日期**: 2026-07-22
**角色**: NLP/IR 专业审稿人
**scope**: 修正 paper footnote line 133/139 与 audit.json 的 `verified`/`degenerate` 状态矛盾；更新 paper line 40 审稿摘要以反映 iter #254/#255 后的真实状态（3 degenerate → 5 partial）；§3.1 9-retriever 实现覆盖确认；§2.2 Stage 04/10 分离问题记录

---

## §A 发现的 Paper-Audit 不一致

### 问题 1: Paper line 133 错误标注 RQ1_Delta_Range 和 RQ2_Table1_Drop 为 verified

**原文本**:
> `RQ1_Delta_Range` (status `verified`) ... `RQ2_Table1_Drop` (status `verified`)

**实际 audit.json 状态**:
- `RQ1_Delta_Range`: `degenerate` (selector returned no non-NaN values)
- `RQ2_Table1_Drop`: `degenerate` (selector returned no non-NaN values)
- `RQ1_Table1_Hit10`: `degenerate`

**根因**: iter #204 paper footnote 更新时将 status 标记为 `verified`，但 audit.json 从未实际更新为 verified（始终为 degenerate，因为 Stage 06/09 数据为空）。

**修复**: 将 line 133 两处 `verified` 替换为 `degenerate`，并补充 Stage 09 empty 导致的 blocking 说明。

### 问题 2: Paper line 139 footnote 声称 RQ2 已 resolved，但 audit 仍为 degenerate

**原文本**:
> `RQ2_Table1_Drop` audit_note updated from `discrepant` to `resolved-by-iter-#204-code-impl-plus-paper-text-clarification`

**实际情况**: iter #204 仅提供了 formula clarification，RQ2 degenerate 的根本原因（Stage 09 all_query_records=[]）在 iter #253 才确认。

**修复**: 移除 "resolved" 措辞，说明 Stage 09 degenerate blocking 现状。

### 问题 3: Paper line 40 审稿摘要数字过时

**原摘要**: "1 degenerate, 5 unverified, 1 partial"

**实际（iter #254/#255 后）**: 3 degenerate + 5 partial + 3 unverified

**修复**: 更新为 "3 degenerate, 5 partial, 3 unverified"。

---

## §B §3.1 Experimental Setup 审查

### 9-retriever 实现覆盖确认

| Retriever | Paper family | 代码位置 | 状态 |
|-----------|-------------|---------|------|
| BM25 | Lexical | `06_retrieval/utils/retrievers.py:77` | ✅ |
| SPLADE | Sparse expansion | `retrievers.py:2845` | ✅ |
| BGE | Dense bi-encoder | `retrievers.py:622` | ✅ |
| E5 | Dense bi-encoder | `retrievers.py:384` | ✅ |
| MiniLM | Dense bi-encoder | `retrievers.py:264` | ✅ |
| ANCE | Dense bi-encoder | `retrievers.py:2140` | ✅ |
| STAR | Learned sparse/hybrid | `retrievers.py:2281` | ✅ |
| ColBERTv2 | Multi-vector late interaction | `retrievers.py:10` | ✅ |
| DeepSeek-v4 Reranker | LLM-based reranker | `14_llm_rerank/common/llm_rerank_common.py:70` | ✅ |

9 个 retriever 全部有代码实现。

### RQ2 (error-effect range) 与 Stage 09 empty 的关系

Paper §3.1 RQ2 问："After authentic user writing errors are injected... how much variation is there in the error-effect range?" Table 1 lower panel 报告该值（BM25 Δ=8.22, SPLADE Δ=6.51 等）。

**现状**: Stage 09 noisy retrieval all_query_records=[] 对所有 retriever 完全为空，无法计算 per-query noisy-correct delta → RQ2 lower panel `degenerate`。

Paper footnote line 133（已修复）现在正确标注为 `degenerate`。

---

## §C §2.2 Stage 04/10 分离确认（来自 iter #255）

Paper §2.2 line 110 描述："if a candidate query fails this check, the system regenerates it, generating 10 candidate sentences per round for a maximum of 10 iterations; samples that still fail after 10 rounds are excluded."

**实际情况**: Stage 04（generation）和 Stage 10（95th percentile filtering）是完全分离的两个 Stage：
- Stage 04: 一次性生成 10 candidates，只做 attribute validation
- Stage 10: 用 VAE 计算 surrogate log-likelihood，per-user 95th percentile threshold filtering
- Stage 10 输出 `regeneration_needed=True` 但无 consumer，Stage 04 不读取

Paper footnote 无专门描述此分离问题，但 iter #255 确认该问题存在，不影响 Table 1/2/3 数值（仅 pipeline 架构问题）。

---

## §D 本轮 Paper 改动汇总

| 位置 | 改动 | 理由 |
|------|------|------|
| line 40 | 审稿摘要更新：1 degenerate→3 degenerate, 5 unverified→3 unverified, +5 partial | 反映 iter #254/#255 真实状态 |
| line 133 footnote | 移除 "status verified" 标注，3 个 claim 全部改为 degenerate，补充 Stage 09 blocking 说明 | paper-audit 一致性 |
| line 139 footnote | 移除 "RQ2 resolved" 措辞，说明 Stage 09 degenerate blocking | paper-audit 一致性 |

**无新增审稿意见**（loop.md §11 无需新增 backlog 项，footnote 已充分覆盖）。

---

## §E 验证

- `python3 -m py_compile PersonalQuery-Benchmark_evaluating_retrieval.md` → N/A（纯文本）
- paper 修改通过人工审阅：lines 40, 133, 139 全部核实

---

## §F 下一步

**iter #257**: 继续 §3.1 审查，检查 paper RQ4 (§3.4) 的 Δ Range 数值与 Stage 10 实际输出的差距；验证 paper §3.4 Table 3 内容是否与 Stage 12 输出对齐。
