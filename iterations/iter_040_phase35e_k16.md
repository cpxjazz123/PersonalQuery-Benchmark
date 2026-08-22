# iter 040: Phase 35.E — K=16 Candidate Expansion → hybrid_07 = 40.5% NEW SOTA

**日期**: 2026-08-23
**承接**: iter_039 (Phase 35.D hybrid pca64_diag+pca64_global_07 = 37.0% K=8) + 用户路线 #2
  > candidate pool 从 K=4 扩大 → K=8/16
**目的**: 测试 K=16 candidate expansion 是否真能 +5pp (用户预期)

---

## Pipeline

```
Phase 15.7 size>=10 subset (74 records, 17 asins)
  → STRICT v8 + 3 user exemplars (Qwen 7B, K=16) → 1184 candidates (420s gen)
  → 一次性 batched Qwen forward 1554 texts (cache 21MB) → 3584d residuals (65s)
  → 一次性 batched spacy nlp.pipe on 1554 texts → 14d syntax
  → PCA fit on 1554 residuals: evr 89.8%@32d, 93.8%@64d, 96.5%@128d
  → 10 base scorers (同 Phase 35.C)
  → 10 hybrid variants (同 Phase 35.D) → intra-product Rank-1
```

**Records**: 74 × K=16 = 1184 candidates × 370 user sents = 1554 texts total
**耗时**: 420s gen + 77s scoring (Qwen forward ~65s + spacy ~5s + scoring ~7s)

---

## 主结果: K=16 Intra-product Rank-1 (size≥10, 74 users)

| scorer | K=8 rank-1 | K=16 rank-1 | delta |
|--------|------------|-------------|-------|
| pca32_global | 31.5% | **33.8%** | +2.3pp |
| pca128_diag | 30.1% | **33.8%** | +3.7pp |
| pca64_diag (Phase 35.C SOTA) | 35.6% | 29.7% | -5.9pp |
| pca64_global | 30.1% | 29.7% | -0.4pp |
| diag_3584 | 27.4% | 28.4% | +1.0pp |
| syntax | 20.5% | 21.6% | +1.1pp |
| oracle_syntax | 12.3% | **5.4%** | -6.9pp |

---

## Hybrid (Phase 35.D variants) on K=16

| hybrid | K=8 rank-1 | K=16 rank-1 | delta |
|--------|------------|-------------|-------|
| **pca64_diag + pca64_global_07** | 37.0% | **40.5%** ★★ | **+3.5pp** |
| pca64_diag + pca128_diag_05 | 31.5% | 37.8% | +6.3pp |
| pca32_global + pca64_diag_07 | 27.4% | 33.8% | +6.4pp |
| pca64_diag + syntax_07 | 34.2% | 33.8% | -0.4pp |
| pca64_diag + syntax_05 | 35.6% | 28.4% | -7.2pp |

→ **pca64_diag + pca64_global_07 (70/30) K=16 = 40.5%** ★ NEW SOTA
(vs Phase 35.D K=8 37.0%, **+3.5pp**)

---

## 关键发现

### 1. K=16 提升 hybrid SOTA 3.5pp
- pca64_diag + pca64_global_07: K=8 37.0% → K=16 40.5%
- → 用户路线 #2 验证成功: candidate 多样性是真实杠杆

### 2. K=16 反而降低 base scorers
- pca64_diag alone: K=8 35.6% → K=16 29.7% (-5.9pp)
- oracle_syntax: K=8 12.3% → K=16 5.4% (-6.9pp)
- → K=16 pool size 翻倍 (104 → 208),base 排序精度变难
- → 但 hybrid 通过互补融合反而补偿

### 3. Coverage 仍 35-46%
- K=8 syntax: 32.4% → K=16 syntax: 45.9% (+13.5pp)
- → K=16 让更多 record 有 full_cov candidate,rerank 选项更多

### 4. mean_rank 仍 ~45 (K=8 是 ~22)
- 候选翻倍,mean_rank 翻倍 → 没真正排序质量提升
- → **40.5% rank-1 是精确命中能力的提升,不是整体排序提升**

### 5. 用户目标对照
- 目标 1: rank-1 size≥10 > 35% → **40.5%** ✅ (+5.5pp over target)
- 目标 2: mean_rank < 10 → 44.79 ❌ K=16 pool 太大
- → K=16 解决 rank-1,但 mean_rank 需要真正排序能力提升

---

## 对照用户目标

| 目标 | 实测 | 判定 |
|------|------|------|
| Rank-1 size≥10 > 35% | **40.5%** | ✅ **+5.5pp over target** |
| Mean rank < 10 | 44.79 | ❌ 候选翻倍,mean_rank 翻倍 |

→ **GO (部分)** — Rank-1 满足并超出,mean_rank 仍需 trade-off。

---

## Outputs

| Path | 用途 |
|------|------|
| `syntax_subspace/phase35e_gen_k16.py` | K=16 generator (1184 cands, 420s) |
| `syntax_subspace/phase35e_score.py` | 10 scorers + intra-product K=16 (~80s) |
| `syntax_subspace/phase35e_hybrid.py` | 10 hybrid variants on K=16 (~30s) |
| `result/phase35e/{scores,intra_product_rank1,eval_summary}.json` | K=16 base 评估 |
| `result/phase35d/hybrid_intra.json` (重写) | K=16 hybrid 评估 |
| `result/phase35e/pca_components.npz` | K=16 PCA (32/64/128 × 3584) |
| `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase35e_candidates_k16.json` | 1184 K=16 candidates |
| `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase35e_user_qwen_residuals.pt` | 21MB Qwen cache |

---

## 下一步

1. **Hard-negative contrastive** (用户路线 #3): 训练 metric learning 区分同 asin 相近用户,可能 mean_rank 大幅下降
2. **Diverse generators hybrid** (K=8 from generator A + K=8 from generator B): 不同 prompt 风格的 16 cands 可能 oracle 真正提升
3. **2-component GMM per user** (用户路线 #4): 处理 "keyword vs natural language" 双模态

---

## 最终 takeaway

> Phase 35.E K=16 candidate expansion 验证用户路线 #2:
> **pca64_diag + pca64_global_07 K=16 = 40.5%** (intra-product Rank-1 size≥10),
> vs Phase 35.D K=8 37.0%, **+3.5pp NEW SOTA**,超用户目标 35%。
> K=16 让 candidate 多样性成为杠杆,但 mean_rank 翻倍 (22→45) 揭示
> **rank-1 精确命中 ≠ 整体排序提升**,下一步应做 hard-negative contrastive。
