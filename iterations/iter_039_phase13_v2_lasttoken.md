# Phase 13/14/15 v2: Last-Token Pool vs Mean-Pool — NO-GO

**Date**: 2026-08-21
**Status**: NO-GO — last-token extraction 比 mean-pool **更弱**,但**语义退化减半**
**Engineering**: ~10 min 总(Qwen load 4 min × 2 + 生成 150s + 评估 < 1 min)

## 实验动机

Phase 13.D (mean-pool) 生成时 StyleVector 注入让 query 偏离 generic attribute-listing,
出现 reviewer-style commentary(如 "Model: Search for: ...") — 这是 mean-pool 抓到的
"整句 aggregate style" 信号。

ACL StyleVector 论文用 **last-token** 抽取 hidden state(因为 generation-direction 决定下一步),
理论上更精准。但我们没验证 last-token 是否在本任务(短 query attribute-listing)上比 mean-pool 更好。

**Phase 13/14/15 v2**:把整个 chain 切到 last-token 抽取,对比 v1 (mean-pool)。

## 实施

### 修改
- 新增 `last_token_hidden_states()` 替代 `get_hidden_states()` 的 mean-pool
  - `idx = attention_mask.sum(dim=1) - 1`
  - `last_h = o.hidden_states[l].gather(1, idx.view(-1,1,1).expand(-1,1,H)).squeeze(1)`
- 复制 13.A/B/C/D/F + 14 v2 scripts + 15.A/B/C with `_lasttoken` suffix
- 输入/输出路径全部 rename → `_v2_*_lasttoken.*`

### Chain 结果
| Stage | Result | Verdict |
|-------|--------|---------|
| 13.A v2 | 298 users × 10 sents × 28 layers, L2 norm 23.41 (mean-pool was ~50) | OK |
| 13.B v2 | per-user Gaussian, own vs other cos margin 0.63 at layer 12-14 | GO |
| 13.C v2 | smoke OK (4 cond × 3 users),α=1.0 ρ=0.5 work | GO |
| 13.D v2 | 30 pairs × 4 cond × K=8 = 960 queries, 150.6s | OK |
| 13.F v2 | 768d style rerank | **NO-GO** |
| 14 v2 | rank-1 in 768d | GO (weak) |
| 15.A v2 | Author ID top-10 best-K 73.3% vs D_off 73.3% | tie |
| 15.B v2 | Semantic sim A=0.732, D=0.768 | PARTIAL-GO |
| 15.C v2 | 4-dim panel | **NO-GO** |

## 关键对比 — last-token vs mean-pool

### D1: 768d Style Margin (best-of-K)
| Cond | v1 mean-pool | v2 last-token | Δ |
|------|--------------|---------------|---|
| A_sampled | **0.323** | 0.257 | **-0.066** |
| B_mean | 0.288 | **0.270** | -0.018 |
| C_shuffled | 0.272 | 0.225 | -0.047 |
| D_off | 0.225 | **0.240** | **+0.015** |
| **A vs D diff** | **+0.098 CI [0.019, 0.191]** excludes 0 ✓ | +0.017 CI [-0.060, 0.091] includes 0 ✗ | lift disappeared |

### D2: Author Identification top-10 (best-K, in 30 users)
| Cond | v1 mean-pool | v2 last-token |
|------|--------------|---------------|
| A_sampled | **83.3%** | 73.3% |
| D_off | 63.3% | **73.3%** |
| A vs D Δ | +0.200 [+0.000, +0.400] ✓ | 0.0 (tie) |

### D4: Semantic Preservation
| Metric | v1 mean-pool | v2 last-token | Δ |
|--------|--------------|---------------|---|
| A sem_sim | 0.6715 | **0.7321** | **+0.060** |
| A coverage | 0.807 | **0.8508** | +0.044 |
| A perfect | 12.5% | **31.7%** | **+19.2pp** |
| A vs D sem_diff | -0.084 [-0.120, -0.051] | **-0.036 [-0.057, -0.017]** | regress halved |
| A vs D cov_diff | -0.072 [-0.098, -0.046] | **-0.034 [-0.055, -0.014]** | regress halved |

### 14: 768d Rank-1 Coverage (top-100)
| Metric | v1 mean-pool | v2 last-token |
|--------|--------------|---------------|
| A_sampled cos top-100 | 56.7% | 50.0% |
| D_off cos top-100 | ~26% | **40.0%** |
| A vs D rank_diff | significant lift | +20.2 CI [-46.9, +81.6] includes 0 |

### 15.C 4-Dim Panel Verdict
| | v1 mean-pool | v2 last-token |
|--|--------------|---------------|
| Composite | A=0.459 > D=0.405 | A=0.433 < D=0.436 (D_off 略胜) |
| Verdict | GO (per user clarification) | **NO-GO** |

## 解读 — Why last-token is weaker for short queries

**last-token 抽取原理**:抓 generation-direction 的 style 信号 — 在 LLM 预测下一个 token 时,
last position hidden state 编码"应该生成什么类型的下一个词"的偏好。

**问题**:我们的任务是生成 **短 attribute-listing query**(15-25 tokens, "Brand X, Material Y,
Dimensions Z"),所有 token 的分布都很类似:
- 句首 Brand
- 句中 attribute name
- 句末 attribute value

**last-token 在 attribute-listing 几乎所有 query 都是 "attribute value"(数字/名词)**,
**信息密度太低,user-to-user 差异不显著**。
而 **mean-pool 包含整句 aggregate**,能抓住 user 在 attribute-ordering / punctuation /
adjective-choice 上的差异。

**生成对比**(同一个 user, A_sampled):
- v1 mean-pool: 出现 "Model: Search for: ..." 等 reviewer-style commentary, 偏离 generic
- v2 last-token: 干净的 attribute-listing,几乎和 D_off 一致

## 结论 — mean-pool v1 is better for this task

| Aspect | v1 mean-pool | v2 last-token | Choice |
|--------|--------------|---------------|--------|
| Style margin | +0.098 CI excludes 0 ✓ | +0.017 CI includes 0 ✗ | v1 |
| Author ID | +0.200 CI excludes 0 ✓ | 0.0 (tie) ✗ | v1 |
| Semantic regress | -0.084 | -0.036 | v2 |
| Composite | 0.459 > D=0.405 ✓ | 0.433 < D=0.436 ✗ | v1 |
| 4-dim verdict | GO | NO-GO | **v1** |

虽然 last-token 让 semantic regress 减半,但它同时让 style lift 几乎消失 → 用户 task
("我们只关心风格") 在 v1 mean-pool 下被满足,v2 不满足。

## 路线调整

- **保留 Phase 13.D/14/15 v1 (mean-pool) 作为主路线** (StyleVector injection + 768d rerank, 4-dim GO)
- Phase 13 v2 (last-token) **NO-GO** — 不替换 mean-pool 版本
- Phase 15.D (α sweep) 应继续基于 mean-pool v1
- 如果未来任务换成 long-form generation(review continuation),可再试 last-token

## 文件位置

### Scripts (in result/phaseXX/scripts/)
- `result/phase13/scripts/phase13_a_v2_extract_residuals_lasttoken.py`
- `result/phase13/scripts/phase13_b_v2_fit_gaussian_lasttoken.py`
- `result/phase13/scripts/phase13_c_v2_inject_lasttoken.py`
- `result/phase13/scripts/phase13_d_v2_generate_lasttoken.py`
- `result/phase13/scripts/phase13_f_v2_style_eval_lasttoken.py`
- `result/phase14/scripts/phase14_v2_rank1_style768d_lasttoken.py`
- `result/phase15/scripts/phase15_a_author_id_v2_lasttoken.py`
- `result/phase15/scripts/phase15_b_semantic_v2_lasttoken.py`
- `result/phase15/scripts/phase15_c_4d_panel_v2_lasttoken.py`

### Results
- `result/phase13/phase13_a_v2_meta_lasttoken.json` (+ per_sentence_residuals_qwen_lasttoken.npz, user_hiddens_lasttoken.npz, neutral_hiddens_lasttoken.npz)
- `result/phase13/phase13_b_v2_meta_lasttoken.json`, `phase13_b_v2_validation_lasttoken.json` (+ user_gaussians_qwen_lasttoken.npz)
- `result/phase13/phase13_c_v2_smoke_test_lasttoken.json`, `phase13_c_v2_smoke_meta_lasttoken.json`
- `result/phase13/phase13_f_v2_style_eval_lasttoken.json`, `phase13_f_v2_per_pair_lasttoken.jsonl`, `phase13_f_v2_meta_lasttoken.json`
- `result/phase14/phase14_v2_rank1_eval_lasttoken.json`, `phase14_v2_per_pair_lasttoken.jsonl`, `phase14_v2_meta_lasttoken.json`
- `result/phase15/phase15_a_v2_author_id_lasttoken.json`, `phase15_a_v2_per_cand_lasttoken.jsonl`
- `result/phase15/phase15_b_v2_semantic_lasttoken.json`, `phase15_b_v2_per_cand_lasttoken.jsonl`
- `result/phase15/phase15_c_v2_4d_panel_lasttoken.json`, `phase15_c_v2_4d_per_pair_lasttoken.csv`

### Heavy data files (not committed)
- `phase13_a_v2_per_sentence_residuals_qwen_lasttoken.npz` (560MB)
- `phase13_a_v2_user_hiddens_lasttoken.npz` (465MB)
- `phase13_a_v2_neutral_hiddens_lasttoken.npz` (464MB)
- `phase13_a_v2_rewrites_cache_lasttoken.jsonl`
- `phase13_b_v2_user_gaussians_qwen_lasttoken.npz`
- `phase13_d_v2_e2e_queries_lasttoken.jsonl`
- `phase13_f_v2_cand_embs_768d_lasttoken.npy`

These live in `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/` (per Rule 10).