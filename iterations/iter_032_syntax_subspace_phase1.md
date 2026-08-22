# iter 032: Syntax-only Subspace — Phase 1 探索发现

**日期**: 2026-08-22
**动机**: 用户指出 review-space hidden residual ≠ query-space style (transductive
assumption 错误)。review 残差包含 sentiment / evaluation / product category /
discourse patterns,直接注入 query LLM 会污染 benchmark information need。

**Phase 1 目标**: 在 hidden residual 空间里**辨识**哪些 PCA dim 主要跟 syntax 相关、
哪些被 noise (sentiment/category) 主导,作为 Phase 2 refit Gaussian 的依据。

---

## 方法 (syntax_subspace/phase1_explore.py)

输入:
- `residual_hidden.npz` (299 sentences × 3584-dim × 4 layers 16/20/24/26)
- 每句 syntax features (spaCy batch parse, 14 维):
  sentence_length, clause_count, function_word_ratio, coordination_count,
  subordinator_count, relative_clause_count, dependency_depth, avg_word_length,
  noun_ratio, verb_ratio, adj_ratio, adv_ratio, pronoun_ratio, punctuation_count
- 每句 noise features (sentiment lexicon + category one-hot + rating, 39 维):
  polarity, subjectivity, rating, word_count + 35 类目 one-hot

流程:
1. Standardize features (zero mean unit var)
2. PCA(20) on residuals per layer
3. 对每个 PC j, 与所有 syntax/noise features 算 |Pearson r|
4. Verdict: KEEP_SYNTAX if max_syntax_r > max_noise_r AND > 0.10
           DROP_NOISE  if max_noise_r > 0.10 (else WEAK)

---

## 结果 (per layer KEEP_SYNTAX dims)

| Layer | KEEP_SYNTAX dims   | exp_var of dim 0 | 说明 |
|-------|--------------------|-------------------|------|
| L16   | [0, 2, 9, 11]      | 88.6%             | 4-dim, dim 0 占绝对优势 |
| L20   | [0, 12, 15, 16, 17]| 94.1%             | **5-dim 最多, 最优选** |
| L24   | [0, 10, 15]        | 91.8%             | 3-dim |
| L26   | [0, 5, 11, 12]     | 84.9%             | 4-dim |
| **跨 layer 稳定** | **[0]** | 85-94% | dim 0 跨 4 layer 全 KEEP, 信号最强 |

### dim 0 的具体信号 (跨 layer 一致)

| Layer | top_syntax | top_noise | verdict |
|-------|------------|-----------|---------|
| L16   | sentence_length(-0.22), clause_count(-0.16) | sentiment_subjectivity(0.18), word_count(-0.14) | KEEP_SYNTAX |
| L20   | sentence_length(-0.22), clause_count(-0.16) | sentiment_subjectivity(0.18), word_count(-0.14) | KEEP_SYNTAX |
| L24   | sentence_length(-0.23), clause_count(-0.17) | sentiment_subjectivity(0.18), sentiment_polarity(0.13) | KEEP_SYNTAX |
| L26   | sentence_length(-0.23), clause_count(-0.17) | sentiment_subjectivity(0.18), sentiment_polarity(0.13) | KEEP_SYNTAX |

**dim 0 = "长难句 vs 短句"轴**: 长句 + 多 clause + function_word_ratio 偏低;
短句 + sentiment 主语化强(短评价句更主观)。是 dim 0 中最纯净的 syntax 信号。

### 大多数 PC 是 WEAK / DROP_NOISE

例如 L26 dim 8 (WEAK): sentence_length(-0.31), clause_count(-0.24) | word_count(-0.30), sentiment_subjectivity(0.26)
  → word_count 与 sentence_length 在 noise 端,导致 noise correlation > syntax correlation。

L26 dim 14 (WEAK): 所有 |r| < 0.10, 完全无法识别。

L26 dim 4 (DROP_NOISE): sentiment_subjectivity(-0.19) > clause_count(0.23),被 sentiment 主导。

**整体观察**: hidden residual 在 3584-dim 里几乎被 sentiment + category + word_count
主导,纯净 syntax 信号被困在 dim 0 一处,其它 dim 要么弱要么被 noise 反超。

---

## Phase 2 设计 (per-user Gaussian refit on syntax-only dims)

**关键选择**: 每个 layer 单独 refit。**Layer 20 是首选** (5-dim: [0, 12, 15, 16, 17], exp_var=94.1%)。

### Phase 2 脚本: phase2_refit_gaussian.py

输入:
- `residual_hidden.npz` (299 × 3584 × 4 layers)
- `phase1_explore.py` 产出的 `pca_layer_*.npz` (components, mean, Z)
- `sentences_for_rewrite.jsonl` (sentence → user_id)

流程 per layer:
1. 加载 KEEP_SYNTAX dims (来自 correlation_layer_*.json 的 verdict)
2. 把 PCA components 限制到这些 dim (rows = KEEP_SYNTAX dims, cols = 3584)
3. U_keep = components[keep_dims]  # (n_keep, 3584)
4. 对每 user:
   - 取该 user 所有句子的 Z[:, keep_dims]  # (n_user_sent, n_keep)
   - fit Gaussian: z_u ~ N(mu_u, Sigma_u) (full covariance)
   - 保存 (mu_u, Sigma_u)
5. 输出: per-user (mu_u, Sigma_u) 在 syntax-only subspace

注入阶段 (Phase 3):
- sample z_u ~ N(mu_u, Sigma_u)  # (n_keep,)
- Δh_syntax = z_u @ U_keep.T + mean_residual_in_keep_subspace  # (3584,)
- 注入 query LLM layer output

---

## 下一步 (按优先级)

1. ✅ Phase 1 完成 (本 iter)
2. ⏭ Phase 2: per-user Gaussian refit (Layer 20 优先, 其它 layer 备选)
3. ⏭ Phase 3: query generation with syntax-only Δh_syntax injection (K-sample + best-of-K)
4. ⏭ Phase 4: 评估 — 与 D_off (no inject) 和 direct_residual (L20 α=0.5) 对比
   - 关键 metric: per-user mean_rank, attribute coverage, hallucination rate
   - **如果 syntax-only 不优于 direct_residual, 说明 hidden space 里真的没有
     clean syntax signal, 需要换路线 (e.g. StyleVector 在 768d AnnaWegmann
     或 Qwen residual rerank, 已 Phase 14.F SOTA)**

## 输出文件

- `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/syntax_features.npz`
- `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/noise_features.npz`
- `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/pca_layer_{16,20,24,26}.npz`
- `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/correlation_layer_{16,20,24,26}.json`
- `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/syntax_dims_summary.json`

## Phase 1 关键 takeaway

> 1 个 dim 跨 layer 稳定 KEEP_SYNTAX (dim 0, sentence_length 主信号)。
> Layer 20 是 syntax-only dim 最多的 layer (5 个: [0,12,15,16,17])。
> 绝大多数 dim 被 sentiment/category/word_count 主导 → 这正是用户假设的
> "review-space residual ≠ query-space style" 在数据上的验证。
> Phase 2 用 syntax-only dim refit Gaussian, 看是否比 direct_residual 更 discriminative。
