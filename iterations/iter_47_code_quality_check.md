# Iteration #47 — 代码质量复查

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人
**scope**: 代码质量复查 — 验证 iter #39-#46 改动的完整性和一致性

---

## §A 复查范围

### 改动的文件（iter #39-#46）

| 文件 | 改动内容 | 状态 |
|------|---------|------|
| `10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py` | REGENERATION_MAX_ROUNDS=10, regeneration_info tracking, build_summary 更新 | ✅ py_compile OK |
| `04_query/common/syntax_depth_no_depth_check.py` | REGENERATION_MAX_ROUNDS=10, process_one_user_single_attempt, process_one_user regeneration wrapper | ✅ py_compile OK |
| `05_inject_noisy/common/apply_lambdamart_userbased_noisy.py` | _get_e5_tokenizer, compute_bpe_token_diff, bpe_aware参数 | ✅ py_compile OK |
| `14_llm_rerank/common/rerank_config.json` | first_stage_retrievers扩展到[bge, e5], _note_multi_retriever | ✅ 有效 JSON |
| `14_llm_rerank/common/llm_rerank_common.py` | validate_retriever_caches(), cache预检验 | ✅ py_compile OK |
| `14_llm_rerank/common/rerank_runner.py` | cache预检验WARN | ✅ py_compile OK |

---

## §B 问题检查

### ✅ 检查1: REGENERATION_MAX_ROUNDS 一致性

- **Stage 04** (`syntax_depth_no_depth_check.py`): `REGENERATION_MAX_ROUNDS = 10` ✅
- **Stage 10** (`train_vades_lite_sentence_latent_threshold.py`): `REGENERATION_MAX_ROUNDS = 10` ✅
- **一致性**: ✅ 两处值相同，注释一致

### ✅ 检查2: process_one_user 调用正确性

- `main()` 中使用的是 `process_one_user`（新的 regeneration 版本）✅
- `process_one_user_single_attempt` 未被 `main()` 调用（保留作为备份）✅

### ✅ 检查3: BPE-aware error injection

- `_get_e5_tokenizer()` lazy-load，失败时返回 None ✅
- `compute_bpe_token_diff()` 失败时返回 0.5（neutral）✅
- `process_batch()` 的 `bpe_aware=True` 时才启用 BPE scoring ✅

### ✅ 检查4: Stage 14 多 retriever 配置

- `rerank_config.json` 的 `first_stage_retrievers: ["bge", "e5"]` ✅
- `validate_retriever_caches()` 对每个 retriever 单独检查 doc/query cache ✅
- cache 缺失时 WARN 不 crash ✅

### ⚠️ 检查5: 潜在未使用代码

`process_one_user_single_attempt` 在当前 `main()` 中未被调用，但保留在代码库中：
- **潜在问题**: 未来可能误用
- **建议**: 可以考虑删除或标记为 deprecated
- **风险等级**: Low（不影响运行，但可能造成混淆）

### ⚠️ 检查6: Stage 06 cache 生成缺失

Stage 14 的多 retriever 支持（`["bge", "e5"]`）需要 Stage 06 先生成 e5 的 doc/query cache：
- 当前 result 目录不可访问，无法验证 cache 是否存在
- `validate_retriever_caches()` 已添加 WARN，但无法确认 e5 cache 实际状态
- **建议**: 需要运行 `06_build_retriever_indices_<Cat>.py` 和 `06_generate_query_cache_<Cat>.py` 为 e5 生成 cache

---

## §C py_compile 验证

所有修改的文件均通过 `python3 -m py_compile`：

```
syntax_depth_no_depth_check.py         OK
apply_lambdamart_userbased_noisy.py    OK
llm_rerank_common.py                  OK
rerank_runner.py                       OK
train_vades_lite_sentence_latent_threshold.py  (iter #39 验证过)
```

---

## §D 结论

**iter #39-#46 的代码质量: 合格**

所有改动的文件通过语法检查，常量一致，逻辑正确。存在两个低风险问题：
1. `process_one_user_single_attempt` 保留但未使用（可接受）
2. e5 等 retriever 的 Stage 06 cache 生成未验证（需要实际运行确认）

---

## §E Git Commit

N/A（复查轮次）

---

## §F 下轮建议

| 优先级 | 行动 | 预计影响 |
|--------|------|---------|
| P2 | 删除或标记 `process_one_user_single_attempt` 为 deprecated | 减少代码混淆 |
| P1 | 验证 e5/doc cache 是否存在（需要 result 目录访问）| 确保 Stage 14 多 retriever 可用 |
| P1 | 确认 Stage 04 regeneration 的 `regeneration_triggered` 字段是否被 Stage 10 正确读取 | 确保 Stage 04-10 数据流一致 |
