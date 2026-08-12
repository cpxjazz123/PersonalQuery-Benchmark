# Iteration #40 — 审稿人视角：§3.1 Retriever 架构批判分析

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人
**scope**: §3.1 Retriever 架构 + Table 1 数据可信度审查

---

## §A 审稿意见（尖锐批评）

---

### 问题 1: Major — "DeepSeek-v4 Reranker" 不是独立 Retriever，是 BGE + LLM Reranking，论文夸大 Retriever 覆盖率

**论文位置**: §3.1, Table 1 caption + line 131
**原文引用**:
> "DeepSeek-v4 Reranker also shows a large correct-query range (Δ=9.1) and moderate error-effect range (Δ=6.62)"

**具体批评**:
论文 Table 1 将 "DeepSeek-v4 Reranker" 列为第 9 个独立 Retriever，与 BM25/SPLADE/BGE/E5 等并列。但**代码中的 DeepSeek-v4 Reranker 不是独立的第一阶段检索器**。

从 `14_llm_rerank/rerank_config.json`:
```json
"first_stage_retrievers": ["bge"]
```
Stage 14 的 pipeline 是：**BGE (first-stage) → LLM reranking (DeepSeek-v4)**。DeepSeek-v4 只是一个重排序组件，不是独立检索器。

`rerank_runner.py` 的处理流程：
```
1. BGE encode query → vector
2. FAISS top-100 retrieval via BGE
3. LLM (M2.5) 对 top-100 做 rerank → top-10
```

**这意味着论文的 Table 1 里 DeepSeek-v4 Reranker 的 Δ=9.1 实际上测的是 "(BGE+DeepSeek-v4)组合" 的性能，不是 DeepSeek-v4 单独的检索能力。**

更严重的问题：**DeepSeek-v4 Reranker 独占一个 Stage（14），而 BM25/SPLADE 等却没有专属 Stage**。这暗示论文可能在用不同的实验设置评估不同的 retriever，导致横向比较不公平。

**审稿人会问**:
> "Table 1 lists DeepSeek-v4 Reranker as a first-stage retriever alongside BM25 and BGE. However, Stage 14 code shows it is a reranking layer applied on top of BGE. How does this architectural difference affect the fairness of comparison? Moreover, why does only DeepSeek-v4 have a dedicated Stage while other retrievers share Stage 06/07?"

---

### 问题 2: Major — Table 1 的 Δ 值完全没有统计显著性支撑，置信区间缺失

**论文位置**: §3.1, Table 1 + §3.2
**原文引用**:
> "SPLADE has the largest correct-query range (Δ =9.6) and a moderate error-effect range (Δ =6.51)"

**具体批评**:
Table 1 报告的 Δ 值（correct-query range 和 error-effect range）是**点估计值**，没有任何置信区间或统计显著性测试。

以 SPLADE Δ=9.6 为例：
- 这个 9.6 是 3 个 domain 的均值
- **没有任何关于这个均值方差的报告**
- 也没有任何 pair-wise significance test（SPLADE 是否显著优于 BGE Δ=5.6？）

如果这不是在 3 个 domain 上的 repeated measure，样本量只有 3，那么：
- **无法做 statistical significance test**（n=3 太少了）
- 置信区间无法估计
- 论文声称的 "observable architecture-related differences" 完全没有统计依据

**审稿人会问**:
> "How many independent runs were used to compute the Δ values in Table 1? What are the confidence intervals? Was any statistical significance test performed? With only 3 domains, how can the paper claim 'observable architecture-related differences' without any statistical evidence?"

---

### 问题 3: Major — E5 subword 敏感性与代码的 full-word error injection 不匹配

**论文位置**: §3.1, line 131
**原文引用**:
> "E5 has a medium correct-query range (Δ =6.6) but the largest error-effect range (Δ =11.16), suggesting sensitivity to spelling-error-induced subword tokenization changes"

**具体批评**:
论文对 E5 敏感性的解释是 **"spelling-error-induced subword tokenization changes"**。这个解释非常具体：E5 使用 BPE/WordPiece subword tokenization，拼写错误会导致 token 序列改变，进而影响检索性能。

**但代码的 error injection 发生在 full-word 级别**：

`05_inject_noisy/common/apply_lambdamart_userbased_noisy.py` 使用的 error 类型：
- Edit distance errors (insertion, deletion, substitution)
- Transposition (相邻字母交换)
- Visual similarity errors
- Keyboard adjacency errors
- Vowel substitution
- Double letters

这些都是 **character-level 或 whole-word-level** 的操作。当 E5 对 "bottle" 和 "bottel" 做 tokenization 时：
- BPE tokenizer 可能将 "bottle" → ["bott", "le"]，"bottel" → ["bott", "el"]
- 这种 token 序列差异是 **subword-level 的**，不是 character-level 的

**代码完全模拟不了 BPE tokenization 对错误拼写的切分效果**。这就是为什么 E5 的 Δ=11.16 实际上可能更高或更低——测试条件与真实场景不符。

**审稿人会问**:
> "E5 uses subword (BPE/WordPiece) tokenization. The paper attributes E5's high error-effect range (Δ=11.16) to 'spelling-error-induced subword tokenization changes.' However, Stage 05 error injection operates at the full-word level. How does full-word error injection simulate subword tokenization effects in E5? What is the evidence that the observed Δ=11.16 reflects subword sensitivity rather than other factors?"

---

### 问题 4: Minor — §3.1 Q2 缺少 Retriever × Expression Style Cluster 的交互效应分析

**论文位置**: §3.1, Q2
**原文引用**:
> "(2) After authentic user writing errors are injected ... how much variation is there in the error-effect range across personalized expression styles between different retrievers?"

**具体批评**:
Q2 问的是 "error-effect range across personalized expression styles between different retrievers"。Table 1 的 Δ Range 列确实报告了跨 cluster 的 error-effect range。

但论文没有回答一个更深的问题：**不同 retriever 的敏感性是否在不同 expression style cluster 上有不同的模式？**

例如：
- BM25 在 C1-C8 的 error-effect 变化是否与 SPLADE 不同？
- 是否存在 retriever × cluster 的交互效应？

Table 1 的 per-cluster Hit@10 变化已经收集了，但没有做任何交互效应分析。如果存在显著的交互效应，论文的结论（"SPLADE 最敏感"）可能是对复杂模式的过度简化。

---

## §B 对应代码缺陷

| 审稿问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| 问题1: DeepSeek-v4 非独立 retriever | `14_llm_rerank/rerank_config.json:13` | `first_stage_retrievers` 只有 `["bge"]`，DeepSeek-v4 是 reranking 层而非独立 retriever |
| 问题1: DeepSeek-v4 非独立 retriever | `14_llm_rerank/rerank_runner.py` | rerank pipeline = BGE first-stage + LLM second-stage，DeepSeek-v4 不独立做检索 |
| 问题2: Δ 值无统计显著性 | `08_compare_all_domain/08_compare_p10_across_domains.py` | 无置信区间报告，无 significance test |
| 问题3: E5 subword 敏感性 | `05_inject_noisy/common/apply_lambdamart_userbased_noisy.py` | error injection 在 full-word 级别，不模拟 BPE tokenization 效果 |
| 问题3: E5 subword 敏感性 | `07_noisy_retrieval/noisy_syntax_depth_eval_common.py` | 评估逻辑依赖 Stage 05 的 error injection，无 subword-aware 评估 |

---

## §C 本轮代码优化

**无代码改动**（本轮为审稿分析轮，发现了深层次的架构性问题）

---

## §D 验证

N/A（审稿分析轮次）

---

## §E Git Commit

N/A（审稿分析轮次）

---

## §F 最需要 rebuttal 的审稿意见

**Top 3 致命审稿意见**（本轮新增）：

1. **[Major] DeepSeek-v4 Reranker 不是独立 Retriever，与其他 8 个 Retriever 的比较不公平**
   - 必须重新定义 DeepSeek-v4 的实验设置：是将它作为独立的 retriever 评估，还是作为 BGE 的 reranking upgrade？
   - 如果是 reranking upgrade，应该比较 "BGE vs BGE+DeepSeek-v4" 的 Δ 值差异，而不是将两者并列

2. **[Major] Table 1 Δ 值无统计显著性支撑**
   - 至少需要报告 3 个 domain 上的均值 ± 标准差
   - 需要明确的统计假设检验（Kruskal-Wallis + post-hoc pairwise comparisons）

3. **[Major] E5 Δ=11.16 的 subword 敏感性结论与代码实现不符**
   - 论文声称 E5 敏感的原因是 "spelling-error-induced subword tokenization changes"
   - 但代码只在 full-word 级别注入错误，无法验证这个假设
   - 需要在代码中添加 subword-level error injection 或承认这个实验无法验证 subword 假设

---

## §G 下轮建议（针对以上问题）

| 优先级 | 行动 |
|--------|------|
| P0 | 确认 DeepSeek-v4 Reranker 在 Table 1 中的定位：独立 retriever 还是 BGE+reranking？若为后者，需重新设计实验对比 |
| P0 | 在 Stage 08 添加 Δ 值的 bootstrap confidence interval 报告 |
| P1 | 研究在 Stage 05 添加 BPE-aware error injection 的可行性（模拟 E5 subword 敏感性） |
| P2 | Stage 14 rerank_runner 支持多 first-stage retriever，不只是 BGE |

