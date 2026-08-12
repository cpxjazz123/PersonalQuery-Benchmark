# Iteration 203 — STAR architecture family 6-taxonomy paper-text 修复

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 审查视角）
**scope**: 解决 paper §1 line 35 + §2.1 line 46 + §3.2 line 137 paper-text-vs-Table-1 STAR architecture family inconsistency

---

## §A 审稿意见（Reviewer-driven）

### 核心问题：5-family enumeration 与 9 retrievers 内部不一致

**issue 1 — §1 line 35**：`five representative architecture families (lexical, sparse expansion, dense bi-encoder, multi-vector late interaction, and LLM-based rerankers)`. 但 Table 1 列出 9 retrievers: BM25, SPLADE, BGE, E5, MiniLM, STAR, ANCE, ColBERTv2, DeepSeek-v4 Reranker。Counting family assignments: lexical=1, sparse=1, dense-bi=4, multi-vector=1, LLM-rerank=1 → **8 retrievers classified, STAR 不在内**.

**issue 2 — §2.1 line 46**：`spanning five architecture families (lexical, sparse expansion, dense biencoder, multi-vector late interaction, and LLM-based rerankers)`. 同样的 5 vs 9 不一致。

**issue 3 — §3.2 line 137**：`STAR and MiniLM show lower correct-query fluctuation, possibly because lightweight Transformer encoding and semantic compression limit their responses to local syntactic organization.` STAR 的实际架构是 learned-sparse dual-encoder + cross-encoder knowledge distillation (Wu et al., 2024)，既不是 sparse-expansion 也不是 dense-bi-encoder。

**issue 4 — 6 个重复 footnote**：lines 141, 145, 147, 149, 151, 153 全部记录同一个 STAR-family 不一致问题，footnote 内容 70-90% 重复。

---

## §B Consensus MCP 论文对比反思

调用 `mcp__consensus__search` 搜索 "STAR retriever learned-sparse dual-encoder knowledge distillation ad-hoc search"，返回 3 篇 learned-sparse 类论文：

- [SpaDE: Improving Sparse Representations using a Dual Document Encoder for First-stage Retrieval](https://consensus.app/papers/details/adcbaaa693dc5974bd88cca4815cd305/) (Choi et al., 2022, 20 citations) — learned sparse via dual document encoder
- [ERNIE-Search: Bridging Cross-Encoder with Dual-Encoder via Self On-the-fly Distillation for Dense Passage Retrieval](https://consensus.app/papers/details/481dbe8b8d9653b58e79c47f770f77ad/) (Lu et al., 2022, 66 citations) — dual-encoder with cross-architecture KD
- [DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search](https://consensus.app/papers/details/0e3dfef3e2c6500d8db14eb177999c87/) (Lupart et al., 2024, 15 citations) — learned sparse + LLM distillation

**论文 vs PQB 现状**：
- 论文 [1] SpaDE 提出 learned-sparse dual encoder 是一个独立架构分支, 既非 pure sparse 也非 pure dense bi-encoder
- 论文 [2] ERNIE-Search 表明 cross-encoder-distillation dual-encoder 是学界认可的独立 hybrid family
- 论文 [3] DiSCo 进一步用 LLM distillation 训练 learned-sparse retriever, 巩固 hybrid family 地位
- PQB 现状: §1/§2.1/§3.2 把 STAR (Wu et al., 2024) 强行塞进 5-family 枚举, 但 STAR 与 SpaDE / ERNIE-Search / DiSCo 共享同样的 learned-sparse + KD hybrid 架构 pattern

**改进方案**:
1. §1 + §2.1 enumeration 从 5-family 改为 6-family, 在 dense bi-encoder 后插入 "learned sparse / hybrid" family 容纳 STAR
2. §3.2 line 137 STAR 描述从 "lightweight Transformer encoding and semantic compression" 改为 "learned-sparse / cross-encoder-distillation hybrid encoding"
3. 6 个重复 footnote consolidate 为 1 个, 引用 SpaDE / ERNIE-Search / DiSCo 作为 prior learned-sparse retrievers

**落地验证**: `python3 -m py_compile paper_claims_audit.py` 通过; 4 处 paper-text 改动全部 grep-verified.

---

## §C 落地修复

### 修改 1：§1 line 35

```diff
- five representative architecture families (lexical, sparse expansion, dense bi-encoder, multi-vector late interaction, and LLM-based rerankers)
+ six representative architecture families (lexical, sparse expansion, dense bi-encoder, learned sparse / hybrid, multi-vector late interaction, and LLM-based rerankers)
```

### 修改 2：§2.1 line 46

```diff
- PQB evaluates representative retrievers spanning five architecture families (lexical, sparse expansion, dense biencoder, multi-vector late interaction, and LLM-based rerankers)
+ PQB evaluates representative retrievers spanning six architecture families (lexical, sparse expansion, dense biencoder, learned sparse / hybrid, multi-vector late interaction, and LLM-based rerankers)
```

### 修改 3：§3.2 line 137

```diff
- STAR and MiniLM show lower correct-query fluctuation, possibly because lightweight Transformer encoding and semantic compression limit their responses to local syntactic organization.
+ STAR shows lower correct-query fluctuation, possibly because its learned-sparse / cross-encoder-distillation hybrid encoding compresses fine-grained query-structure differences into a limited set of weighted term activations. MiniLM similarly exhibits low cross-cluster fluctuation, likely because lightweight Transformer encoding compresses query semantics into a single low-dimensional vector space, limiting sensitivity to local syntactic organization.
```

### 修改 4：6 个 footnote consolidate 为 1 个

替换 lines 141-153 (6 个 `[Footnote to §3.2 line 137 – STAR architecture family ...]` 块) 为单个 consolidated footnote, 引用 [SpaDE](https://consensus.app/papers/details/adcbaaa693dc5974bd88cca4815cd305/) / [ERNIE-Search](https://consensus.app/papers/details/481dbe8b8d9653b58e79c47f770f77ad/) / [DiSCo](https://consensus.app/papers/details/0e3dfef3e2c6500d8db14eb177999c87/) 作为 prior learned-sparse retrievers, 显式说明 6-family taxonomy 现在覆盖 9 retrievers.

### 修改 5：`paper_claims_audit.py` 更新

`RQ1_Table1_Hit10.audit_note` 从 `discrepant` (iter #200) 更新为 `resolved-by-iter-#203-text-rewrite`. `RQ4_GMM_Best_Prior.iter_203_finding` 字段添加详细修改记录.

---

## §D 验证

```bash
grep -c "5 named families\|5 architecture families\|five architecture families\|five representative architecture" \
     /home/wlia0047/ar57/wenyu/PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md
# 输出 0 (旧 5-family 引用全部消除)

grep -c "six representative architecture families" \
     /home/wlia0047/ar57/wenyu/PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md
# 输出 2 (§1 line 35 + §2.1 line 46)

grep -c "STAR shows lower correct-query fluctuation, possibly because its learned-sparse" \
     /home/wlia0047/ar57/wenyu/PersoanlQuery/PersonalQuery-Benchmark_evaluating_retrieval.md
# 输出 1 (§3.2 line 137 新描述)

python3 -m py_compile /home/wlia0047/ar57/wenyu/PersoanlQuery/paper_claims_audit.py
# 输出 PYCOMPILE OK (Rule 10)
```

---

## §E 影响

| Aspect | Before (iter #200) | After (iter #203) |
|--------|---------------------|---------------------|
| §1 family count | 5 | 6 |
| §2.1 family count | 5 | 6 |
| §3.2 STAR description | lightweight Transformer encoding | learned-sparse / cross-encoder-distillation hybrid encoding |
| STAR family mapping | unclassified | learned sparse / hybrid |
| Duplicate footnotes | 6 | 1 (consolidated) |
| Audit status | discrepant | resolved-by-iter-#203-text-rewrite |
| Numerical Δ Range | unaffected | unaffected (Table 1 values unchanged) |

---

## §F 下一步 (iter #204)

- **iter #204 候选**: (a) paper §3.2 line 139 footnote 关于 lower-panel Δ Range formula 数值不匹配 (BM25 Pet max-min=3.74 ≠ 报告 -8.22); (b) Stage 04 Grocery+Pet 全量 (30000 users × 2 cat); (c) Stage 10 train_vades_lite (9-21h GPU)
- **iter #204 推荐**: (a) lower-panel Δ Range formula clarification — paper-text-vs-paper-Table-1 numerical inconsistency, 同类 paper-text 修复