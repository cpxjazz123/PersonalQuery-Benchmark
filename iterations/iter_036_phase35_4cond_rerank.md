# iter 036: Phase 35 — 4-condition Rerank Coverage Result + Intra-product 数据缺口

**日期**: 2026-08-22
**承接**: iter_035 (Phase 34 light eval) + 用户反馈"style score 必须与 coverage/length 解耦,加 S_residual, 加 intra-product Rank-1"
**目的**: 验证 S_residual (Qwen 3584d Maha) 是否能成为新的 style signal,
        并与 S_syntax (length-controlled 14d Maha) 形成 hybrid rerank。

---

## Pipeline

```
STRICT v8 + 3 user exemplars (Qwen 7B, K=4)
  → candidates.json (20 records × 4 cands = 80 cands)
  → per-candidate:
       S_residual = -Maha(cand_qwen_resid, user_qwen_resid_mu)  ← Phase 14.F SOTA style signal
       S_syntax   = -||cand_syntax_ctrl - user_mean_syntax_ctrl||
                    (14d syntax features, residualized by [length, clause_count, attr_count])
  → 4 conditions (argmax within full_cov):
       - baseline:  cand[0]
       - residual:  argmax S_residual
       - syntax:    argmax S_syntax
       - hybrid_05: argmax 0.5*z(S_residual) + 0.5*z(S_syntax)
       - hybrid_07: argmax 0.7*z(S_residual) + 0.3*z(S_syntax)
  → intra-product Rank-1 (size≥N subsets)
```

**Records**: 20 (subset of query_records.json, limited by user exemplar pool of 20 users)
**K**: 4 (因 phase34_generate 的 K_SAMPLES hard-coded 4)

---

## 主结果: 4-condition Coverage (20 records)

| condition | full_cov (5/5) | delta vs baseline |
|-----------|----------------|-------------------|
| baseline | **7/20 (35%)** | — |
| residual | **10/20 (50%)** | +15% ✓ |
| syntax | **10/20 (50%)** | +15% ✓ |
| hybrid_05 | **10/20 (50%)** | +15% ✓ |
| hybrid_07 | **10/20 (50%)** | +15% ✓ |

**观察**:
- 所有 4 个 rerank conditions 都到 10/20 (records 中有 10 条至少 1 个 cand 满足 full coverage)
- baseline greedy (cand[0]) 仅覆盖 7 条 → rerank **统一 +15%** (3 条新增)
- 4 conditions 间无差异 → 在 K=4 / pool size 小的设置下, coverage filter 是主驱动力,
  score 区分力被 coverage=1 候选数限制
- Phase 35 真正要看的不是 coverage,而是 **intra-product Rank-1** (style 区分力)

---

## Length-controlled Syntax Distance

每 cand 的 14d syntax features 被 orthogonalized by [length, clause_count, attr_count] (3-dim):
- y_ctrl[d] = y[d] - X*beta  (X = z-scored control, beta = OLS coef)
- 然后 S_syntax = - ||cand_syntax_ctrl - user_mean_syntax_ctrl||

→ 这强制 syntax 距离不再因 coverage 提升而增加 (已 regress 掉 length / clause / attr 影响)

但 intra-product Rank-1 还没法跑 (见下)。

---

## Intra-product Rank-1 数据缺口

**核心问题**: 当前 query_records.json 20 条记录的 asin 全部唯一 (每条 asin 仅 1 user)。
intra-product Rank-1 需要 size≥N 的 asin pool (Phase 15.7 标准是 size≥10),
但当前 records 完全不满足。

```
intra-product pool sizes: top 10 = [('B0BBM75S58', 1), ('B09WRKYRQZ', 1), ...]
                            每个 asin pool size = 1
```

→ Phase 35 当前**无法评估 intra-product Rank-1**。

### Phase 35.B 计划 (size≥10 subset)

需要重新选 records:
1. 从 500 条 query_records 中,筛 asin 出现在 ≥10 个 user 中的 records
2. Phase 15.7 已做过这件事 → 找相关 code / output 重用
3. 对每条记录复用当前 phase34 generator + phase35 scorer + rerank
4. 评估 size≥10 subset 的 intra-product Rank-1

estimated work:
- 找 size≥10 subset: 5 min (greplookup phase15.7 output)
- generation: ~30 min (198 users × K=4 = 792 candidates)
- scoring: ~10 min (Qwen forward 792 candidates)
- intra-product: ~5 min (Maha in 3584d)
- **total**: ~50 min

但用户原本 success criteria 是 19.2% → > 22% intra-product Rank-1,
仍需要 K=8 才能让 BoK rerank 有足够候选 (Phase 14.Q10 的 BoK-184 SOTA 用 K=8)。
所以 Phase 35.B 实际上 = Phase 14.Q10 size≥10 子集 + Qwen residual scorer 替代 pure syntax rerank。

---

## Qwen Residual Signal 初步观察 (informal)

从 20 records 内的 coverage 提升看 (35% → 50%, +15%):
- S_residual 选出的 candidate 跟 S_syntax / hybrid 都指向相同 10 条 records
- 这暗示 S_residual 也学到了"coverage 倾向",但**不一定是 style 信号**

需要 intra-product 才能区分:
- 如果 S_residual intra-product Rank-1 > S_syntax intra-product Rank-1 → Qwen hidden 比 handcraft syntax 更 style-aware
- 如果 ≤ → handcraft syntax 已经接近 Qwen hidden 的 style signal 上限

---

## 决策点

| Phase 35.B result | Decision |
|-------------------|----------|
| S_residual intra-product Rank-1 > 22% (Phase 14.F SOTA) | Qwen residual is the new style signal; abandon 14d syntax |
| S_residual intra-product Rank-1 in [19.2%, 22%] | parity with Phase 14.F; combine with hybrid for marginal gain |
| S_residual intra-product Rank-1 < 19.2% | handcraft syntax and Qwen residual both fail intra-product; abandon rerank line |
| (Phase 35.B uses Phase 14.Q10 generator + K=8, size≥10 subset) |

---

## 输出文件

- `syntax_subspace/phase35_full.py` (整合 pipeline)
- `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/user_qwen_residuals.pt` (用户 Qwen residual cache, 20 users × ~14 sents × 3584d = 80KB)
- `result/phase35/scores.json` (per-record per-candidate scores for 4 conditions)
- `result/phase35/intra_product_rank1.json` (intra-product result, all pool size=1)
- `result/phase35/eval_summary.json` (aggregate)

## 最终 takeaway

> Phase 35 在 20 records 上验证 **4-condition rerank 一致 +15% coverage**,
> 真正区分 style 信号的是 **intra-product Rank-1** 但当前数据 size=1 pool 不可行。
> 必须 Phase 35.B: 用 size≥10 subset (Phase 15.7) + K=8 candidates (Phase 14.Q10) 才能
> 区分 S_residual vs S_syntax 哪个 style signal 强。
