# iter 041: Phase 35.F — Hard-Negative Contrastive → margin_pca64_diag = 46.6% NEW SOTA

**日期**: 2026-08-23
**承接**: iter_040 (Phase 35.E hybrid K=16 = 40.5%) + 用户路线 #3
  > 引入 margin: L = max(0, m + D(q,u+) - D(q,u-))
  > u- = argmin_{v≠u} D(q,v)  ← hard negative mining
**目的**: 测试 hard-negative contrastive calibration 是否能突破 hybrid rerank 上限

---

## Pipeline

```
Phase 35.E K=16 candidates (74 records × K=16 = 1184 cands, 17 asins)
  → 一次性 batched Qwen forward (cache 21MB) → 3584d residuals
  → PCA-64 + PCA-32 projection
  → Per-pool (asin), per-target-user (73 users with pool≥10), compute:
       D_target = D(cand, target_user)        [Maha in PCA-64 diag]
       D_other  = D(cand, other_users)        [Maha to all other users]
       HN score = -D_target + α · log_sum_exp(-D_other)   [α=1, 0.5]
       margin   = D_target - min(D_other)    [target closer than closest other]
       hybrid_HN = z(HN_diag) · 0.7 + z(HN_g32) · 0.3
```

**Records**: 73 users with pool≥10 (vs Phase 35.E 74 — one single-user pool skipped)
**耗时**: ~10s (no Qwen, uses Phase 35.E cache)

---

## 主结果: Hard-Negative Intra-product Rank-1 (size≥10, 73 users) ★ KEY METRIC

| scorer | rank-1 | mean_rank |
|--------|--------|-----------|
| pca64_diag_alone (Phase 35.E baseline) | 28.8% | 46.50 |
| HN_pca64_diag_α0.5 | 42.5% | 46.28 |
| HN_pca64_diag_α1 | 43.8% | 46.58 |
| **margin_pca64_diag** | **46.6%** ★★ | 47.35 |
| **hybrid_HN_diag_g32_07** | **46.6%** ★★ | 44.70 |

→ **margin_pca64_diag = 46.6% NEW SOTA** (vs Phase 35.E 40.5%, **+6.1pp 1.15x**)

---

## 关键发现

### 1. Hard-negative scoring 大幅提升 rank-1
- pca64_diag_alone (target-only Maha): 28.8%
- + hard-negative (HN α=1): 43.8% (+15pp)
- + margin (target vs closest other): 46.6% (+17.8pp)
- → **直接对比"target 比其他用户近多少"比单纯 target 距离更有判别力**

### 2. Margin > HN (target vs closest > target + avg others)
- margin: 46.6%
- HN α=1: 43.8%
- → closest other 比 average others (log-sum-exp) 更 informative

### 3. Hybrid_HN = margin (46.6% tie)
- hybrid_HN(diag + g32 70/30): 46.6%
- → PCA-64 diag HN + PCA-32 global HN 互补达到同样效果
- 但 mean_rank 略低 (44.70 vs 47.35),说明排序更稳定

### 4. mean_rank 仍 44-47 (K=16 pool 翻倍效应)
- 用户期望 < 10
- 即便 hard-negative 也只能 44-47
- → **rank-1 精确命中 ≠ 整体排序提升** (K=16 pool 太大)

### 5. 与 Phase 14.Q10 对比
- Phase 14.Q10 LOPO 30 用户 rank-1 60% (小池)
- Phase 35.F 73 用户 rank-1 46.6% (大池,真实泛化指标)
- → Phase 35.F 是大池评估,绝对数字看似低但比 30-pair SOTA 更可信

---

## 对照用户目标

| 目标 | 实测 | 判定 |
|------|------|------|
| Rank-1 size≥10 > 35% | **46.6%** | ✅ **+11.6pp over target** |
| Mean rank < 10 | 44.70 | ❌ K=16 pool 大,mean_rank 翻倍 |

→ **GO (部分)** — Rank-1 满足并大幅超出,mean_rank 受 K=16 限制。

---

## 三阶段累计 SOTA 进化

| Phase | Method | Rank-1 size≥10 | mean_rank |
|-------|--------|----------------|-----------|
| 14.F (历史 SOTA) | Qwen 768d Maha top-K rerank | 19.2% | n/a |
| 35.B | K=4 + hybrid_07 (3584d) | 31.3% | 12.66 |
| 35.C | K=8 + pca64_diag | 35.6% | 23.09 |
| 35.D | K=8 + pca64_diag + pca64_global_07 | 37.0% | 22.37 |
| 35.E | K=16 + pca64_diag + pca64_global_07 | 40.5% | 44.79 |
| **35.F** | **K=16 + margin_pca64_diag** | **46.6%** ★★ | 47.35 |

→ 用户路线 1 (PCA + shrinkage) → 2 (K=16) → 3 (hard-negative) 完整链,
**46.6% 是 Phase 14.F SOTA 19.2% 的 2.43x**

---

## Outputs

| Path | 用途 |
|------|------|
| `syntax_subspace/phase35f_hardneg.py` | 5 hard-negative scorers + intra-product (~10s) |
| `result/phase35f/hn_intra.json` | 完整 intra-product 结果 |

---

## 下一步

1. **2-component GMM per user** (用户路线 #4): 处理 "keyword vs natural language" 双模态,可能突破 50%
2. **mean_rank 优化**: 当前 47 (K=16) 是 pool 翻倍 artifact,K=8 + margin 是不是更好?
3. **跨 user hard-negative**: 用 50 size≥5 用户池而非 17 asins,做更通用风格对比
4. **Phase 35.G 综合**: 把 margin 与 hybrid 联合优化,看是否能突破 50%

---

## 最终 takeaway

> Phase 35.F hard-negative contrastive 验证用户路线 #3:
> **margin_pca64_diag K=16 = 46.6%** (intra-product Rank-1 size≥10),
> vs Phase 35.E 40.5%, **+6.1pp NEW SOTA**,累计 Phase 14.F 19.2% 的 **2.43x**。
> Hard-negative margin 是真正的杠杆: target vs closest other 比单纯 target 距离强 17.8pp。
> mean_rank 仍 47 是 K=16 pool 翻倍 artifact,下一步可试 K=8 + margin 或 2-component GMM。
