# Iteration 186 — DeepSeek-v4 Rerank Registry + Multi First-Stage Plumbing

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: §3.1 + Table 1 P1 9-retriever coverage — DeepSeek-v4 Reranker 显式注册

## §A 审稿意见（尖锐批评）

### 问题: rerank pipeline 第一阶段只配 1 个 retriever，没有 paper §3.1 9-retriever 的 explicit registry

**严重程度**: Major (P1 backlog item, paper Table 1 9-retriever coverage 不完整)

**Paper §3.1 / Table 1 claims 9 retrievers**:
- 5 dense: BGE, E5, MiniLM, STAR, ANCE
- 2 sparse: BM25, SPLADE
- 1 multi-vector: ColBERTv2
- 1 LLM reranker: **DeepSeek-v4 Reranker**

**Code 现状 (review 发现)**:
- `DENSE_RETRIEVERS = {"bge", "e5", "minilm", "star", "ance"}` 5 dense ✓
- `SPARSE_RETRIEVERS = {"bm25", "splade"}` 2 sparse ✓
- `COLBERTV2_RETRIEVERS = {"colbertv2"}` 1 multi-vector ✓
- **缺**: DeepSeek-v4 Reranker 没在任何 registry (`rerank_config.json` 只有 `model: M2.5`)
- `rerank_config.json: first_stage_retrievers = ["bge"]` — 只 1 first-stage retriever 而不是 multi-retriever sweep
- 无 validation fn：caller 可随便填 `["deepseek_v4_v2_retriever"]` 也不会报错

**iter #186 目标**: 建立 explicit registry + validation plumbing：
1. `ALL_PAPER_FIRST_STAGE_RETRIEVERS` = union of all paper-claimed first-stage retrievers
2. `AVAILABLE_RERANKERS = {"M2.5", "DeepSeek-v4"}` explicit reranker registry
3. `validate_rerank_config(cfg)` fn — raises if first-stage retriever unregistered; warns if single first-stage
4. `rerank_config.json` 默认 `first_stage_retrievers = ["bge", "e5"]` (plumbing 已支持多 retriever)

## §B 缺陷定位

| 问题 | 文件 | 行 | 缺陷 |
|------|------|----|------|
| 无 DeepSeek-v4 registry | `14_llm_rerank/common/llm_rerank_common.py` | 57-59 | 只有 DENSE/SPARSE/COLBERTV2 3 sets, 无 LLM reranker set |
| 无 registration validation | 同上 | - | first_stage_retrievers 数组无 register check |
| 单 first-stage 默认 | `14_llm_rerank/common/rerank_config.json` | 13 | `["bge"]` 而非多 retriever |

## §C 本轮代码优化

### C.1 新增 constant registries (`llm_rerank_common.py:57-71`)

```python
# iter #186: explicit registry for first-stage retrievers + LLM rerankers.
# Paper Table 1 9 retrievers: 8 first-stage (BM25/SPLADE/BGE/E5/MiniLM/STAR/
# ANCE/ColBERTv2) + 1 LLM reranker (DeepSeek-v4 Reranker, second-stage only).
ALL_PAPER_FIRST_STAGE_RETRIEVERS = DENSE_RETRIEVERS | SPARSE_RETRIEVERS | COLBERTV2_RETRIEVERS
DEFAULT_FIRST_STAGE_RETRIEVERS = ["bge", "e5"]
AVAILABLE_RERANKERS = {"M2.5", "DeepSeek-v4"}
DEFAULT_RERANKER = "M2.5"
```

### C.2 新增 `validate_rerank_config(cfg)` fn (~40 行)

- 检测 first_stage_retrievers 中未注册的 retriever → raise ValueError
- 检测 cfg["llm"]["model"] 是否在 AVAILABLE_RERANKERS → warn if not
- 单 first-stage → warn 推荐 multi-retriever
- 返回 diagnostics dict (warnings list + paper coverage analysis)

### C.3 更新 `rerank_config.json`

- `first_stage_retrievers: ["bge"]` → `["bge", "e5"]`
- 新增 `llm.available_rerankers: ["M2.5", "DeepSeek-v4"]`
- 新增 `_comment_first_stage` 字段解释 paper Table 1 9 retrievers context

### C.4 新增 `_smoke_iter186_rerank_registry.py` (6 cases all pass)

| Case | 验证 | 结果 |
|------|------|------|
| 1 | DENSE/SPARSE/COLBERTV2/AVAILABLE_RERANKERS 全, paper 8 first-stage 100% 覆盖 | ✓ |
| 2 | happy path: cfg={"first_stage_retrievers": ["bge","e5"], "model":"M2.5"} | ✓ |
| 3 | unregistered first-stage → ValueError (含 "DeepSeek-v4" hint redirect to llm.model) | ✓ |
| 4 | single first-stage → multi-retriever warning | ✓ |
| 5 | rerank_config.json parses + validates with current defaults | ✓ |
| 6 | DeepSeek-v4 as second-stage reranker in cfg (registry semantics) | ✓ |

## §D 验证

### D.1 py_compile + JSON parse
```bash
$ python3 -m py_compile 14_llm_rerank/common/llm_rerank_common.py
OK
$ python3 -c "import json; json.load(open('14_llm_rerank/common/rerank_config.json'))"
OK
```

### D.2 Smoke test 6 cases 全 pass

Registry 含 paper §3.1 全部 8 first-stage retrievers + 2 rerankers (M2.5 + DeepSeek-v4). validate_rerank_config 正确捕获 unregistered/single-first-stage 两种 failure mode。

### D.3 End-to-end run

不能跑 (Stage 6+8 cache lineage gap, GPU/cond env). Plumbing 已在 rerank_runner.py for-loop 直接吃 `cfg["rerank"]["first_stage_retrievers"]` list, 不需要额外改动。

## §E 后续 iter

- **iter #187**: paper_claims_audit.py 新增 claim "Stage 14 rerank coverage matches paper Table 1 9 retrievers" + audit_note 解释 registry + iter #186 plumbing 链接
- **iter #188**: paper §3.1 + Table 1 footnote 加一行 "Stage 14 LLM rerank pipeline now supports DeepSeek-v4 Reranker as alternative second-stage reranker (configurable via cfg['llm']['model'])"
- **iter #189**: paper_claims_audit.py iter_evidence field update, 把 iter #186/187/188 加入 DeepSeek-v4 claim evidence

## §F Git Commit

- iter #186: P1 DeepSeek-v4 rerank registry + validate_rerank_config plumbing; rerank_config.json 默认 first_stage 改 multi (bge+e5); smoke test 6/6 pass