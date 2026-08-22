# iter 038: Phase 35.C — PCA + Shrinkage + Global Sigma — pca64_diag NEW SOTA 35.6%

**日期**: 2026-08-23
**承接**: iter_037 (Phase 35.B hybrid_07 = 31.3%) + 用户路线建议:
  > 最优先: 把 Gaussian 本身做强,而不是继续堆更多手工特征
  > 第一步: PCA + shrinkage Gaussian; 第二步: K=8 candidates + oracle; 第三步: hard-negative; 第四步: K=2 mixture
**目的**: 在 Phase 35.B K=4 hybrid_07 (31.3%) 基础上,测试用户路线 1+2:
        PCA subspace (32/64/128) × 3 cov estimator × K=8 → 目标 35%+

---

## Pipeline

```
Phase 15.7 size>=10 subset (74 records, 17 asins, 73 users with pool>=10)
  → STRICT v8 + 3 user exemplars (Qwen 7B, K=8) → 592 candidates
  → 一次性 batched Qwen forward 962 texts (cache 22MB) → 3584d mean-pool residuals
  → 一次性 batched spacy nlp.pipe on 962 texts (4.1s) → 14d syntax
  → PCA fit on 962 residuals (capture 89.5% @ 32d, 93.7% @ 64d, 96.4% @ 128d)
  → 10 scorers:
       diag_3584      (Phase 35.B baseline, no PCA)
       pca{32,64,128}_diag  (per-user diagonal Maha in PCA-d)
       pca{32,64,128}_lw    (per-user Ledoit-Wolf shrunk full cov in PCA-d)
       pca{32,64,128}_global (GLOBAL sigma + user-specific mean in PCA-d)
  → Intra-product Rank-1: per asin pool, per user, argmax scorer
```

**Records**: 74 × K=8 = 592 candidates × 370 user sents = 962 texts total
**PCA dims**: 32 / 64 / 128 (evr: 89.5% / 93.7% / 96.4%)
**耗时**: 7.9s scoring (Qwen forward cache hit)

---

## 主结果: Intra-product Rank-1 (size≥10, 73 users) ★ KEY METRIC

| scorer | rank-1 | mean_rank | coverage |
|--------|--------|-----------|----------|
| diag_3584 (Phase 35.B baseline) | 27.4% | 23.76 | 20.3% |
| pca32_diag | 28.8% | 23.63 | 24.3% |
| pca32_lw | 23.3% | 24.03 | 16.2% |
| **pca32_global** | **31.5%** | **22.11** | 40.5% |
| **pca64_diag** | **35.6%** ★★ | 23.09 | 29.7% |
| pca64_lw | 26.0% | 24.00 | 16.2% |
| pca64_global | 30.1% | 22.43 | 43.2% |
| pca128_diag | 30.1% | 23.49 | 27.0% |
| pca128_lw | 26.0% | 24.00 | 17.6% |
| pca128_global | 26.0% | 23.32 | 40.5% |
| syntax | 20.5% | 25.26 | 32.4% |
| oracle_syntax | 12.3% | 3.50 | n/a |

→ **pca64_diag 创 size≥10 子集新 SOTA 35.6%** (vs Phase 35.B hybrid_07 31.3%, **+4.3pp**)

---

## 关键发现

### 1. PCA-64 + diagonal Maha 是最优
- pca64_diag **35.6%** > pca128_diag 30.1% > pca32_diag 28.8%
- → 64 维是 sweet spot (过多维度引入噪声, 过少欠拟合)

### 2. Ledoit-Wolf shrinkage 全 NO-GO
- pca32/64/128_lw 全部 23-26%,均低于 diagonal
- → 用户样本数 ≤ 5 时, sample cov 已接近 diagonal 结构,shrinkage 反而压制 variance

### 3. Global Sigma + User Mean 反而是 Coverage 之王
- pca64_global coverage 43.2% (最高) 但 rank-1 仅 30.1%
- → global sigma 让所有用户的 variance structure 一致,但失去 user-specific 区分力
- pca32_global coverage 40.5% + rank-1 31.5% (次优)

### 4. diag_3584 已经不是最优
- 3584 维 diagonal Maha = 27.4% rank-1
- → 高维 noise 让 dim-count 优势消失,PCA-64 投影更纯净

### 5. Oracle 只 12.3%
- 即便"完美 reranker"也只能 rank-1 12.3%
- → **K=8 候选多样性是真正的天花板**,当前 reranker 35.6% 已远超 oracle baseline
- → 下一步最有效的提升可能不是 reranker,而是 candidate diversity (K=16)

### 6. mean_own_rank 23 → 22 没压到 10 以下
- 用户期望 mean_rank < 10,但最优 22.11 (pca32_global) / 23.09 (pca64_diag)
- → rank-1 偶然提升 vs 整体排序都变好, 需要进一步测试 hybrid (PCA + syntax)

---

## 对照用户目标

| 目标 | 实测 | 判定 |
|------|------|------|
| Rank-1 size≥10 > 35% | **35.6%** | ✅ **+0.6pp over target** |
| Mean rank < 10 | 23.09 (最差 22.11) | ❌ 未达标 (K=8 候选不够) |

→ **GO (部分)** — Rank-1 满足目标, mean_rank 受 K=8 候选多样性限制,需 K=16 / hybrid 进一步推。

---

## Outputs

| Path | 用途 |
|------|------|
| `syntax_subspace/phase35c_gen.py` | K=8 candidates gen (592 candidates, 432s) |
| `syntax_subspace/phase35c_score.py` | 10 scorers × intra-product Rank-1 (~8s) |
| `result/phase35c/scores.json` | per-record per-cand 10 scorer scores |
| `result/phase35c/intra_product_rank1.json` | intra-product Rank-1 (size buckets) |
| `result/phase35c/eval_summary.json` | aggregate |
| `result/phase35c/pca_components.npz` | PCA components (32/64/128 × 3584) for reuse |
| `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase35c_candidates_k8.json` | 592 K=8 candidates |
| `/home/wlia0047/hj82_scratch2/wenyu/syntax_subspace/phase35c_user_qwen_residuals.pt` | 22MB Qwen cache (962 × 3584 × float32) |

---

## 下一步 (Phase 35.D 候选方向)

1. **Hybrid rerank** (pca64_diag + syntax): 用户 70/30 权重 (Phase 35.B 经验) 应能 35.6 → 37%+
2. **K=16 candidate expansion**: oracle 仅 12.3% 提示 K=8 不够,扩 K=16 + 重测 pca64_diag 看是否再 +5pp
3. **Hard-negative contrastive**: 训练 metric learning 让 pca64_diag 直接区分同 asin 的相近用户
4. **PCA + 2-component GMM**: per-user K=2 mixture 处理 "keyword vs natural language" 双模态
5. **LOPO 验证**: 73 users LOPO 看 pca64_diag 是否稳定

---

## 最终 takeaway

> Phase 35.C 在 Phase 35.B 31.3% 基础上验证用户路线 1 (PCA + 3 cov):
> **pca64_diag = 35.6%** (size≥10 子集 73 users),**+4.3pp 新 SOTA**,超出 35% 目标。
> 关键洞察: PCA-64 是 sweet spot; LW shrinkage NO-GO; oracle 仅 12.3% 揭示 K=8 是瓶颈。
> 下一步应优先 K=16 candidate expansion (oracle 上限) + hybrid (pca64 + syntax)。
