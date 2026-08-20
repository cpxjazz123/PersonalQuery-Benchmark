# Phase 11.F: Rank-1 Eval on 11.D Candidates — 总结

**Date**: 2026-08-20
**Status**: NO-GO (rank-1 = 0%, top-10 = 3.3%, but margin +93% shows style signal works)

## 实验动机

Phase 11.D 跑了 600 candidates (Architecture B: LLM neutral + diffusion + style hints + hard-copy)。
需要 rank-1 eval 看是否突破 1.26% baseline。

## Pipeline

1. Load 11.D 600 candidates (30 pairs × 20 cands)
2. spaCy batch 318d 句法特征提取 (USE_NEUTRALIZE=True, 与 Phase 10.16.C 一致)
3. 复用 Phase 10.15 的 user_mu_318d cache + LedoitWolf cov
4. Per-pair full distance matrix (20 cands × 3000 users)
5. Compute target rank (0-indexed); rank-1 = target is nearest user

## 关键结果

```
Total pairs: 30
Rank-1 coverage (any cand):  0/30 = 0.0%
Rank-3 coverage (any cand):  0/30 = 0.0%
Rank-5 coverage (any cand):  0/30 = 0.0%
Rank-10 coverage (any cand): 1/30 = 3.3%
Best margin > 0:             28/30 = 93.3%
Best rank=1 + margin > 0:    0/30 = 0.0%

Mean best rank: 476.5 [301.8, 684.2]   (Phase 10.15: 307.8)
Mean best margin: +268.0 [153.7, 404.0]
```

## 对比历史最佳

| 方法 | Pair 数 | Cand/Pair | Rank-1 | Top-10 | Top-100 |
|------|---------|-----------|--------|--------|---------|
| Phase 10.15 baseline | 876 | 96 | **1.26%** | 9.25% | - |
| Phase 10.18 exemplar search | 30 | - | 0% | 0% | - |
| Phase 10.19 contrastive RAG | 30 | - | - | - | **17%** |
| **Phase 11.D Architecture B** | 30 | 20 | **0%** | **3.3%** | - |

## 解读

1. **rank-1 = 0%**: Architecture B 没有突破 1.26% baseline (虽然样本量小 30 vs 876,统计意义有限)
2. **margin +93% positive**: 93% 的 pair 都有 best cand 比 wrong user mean 近,说明 style 信号 **真的** 有用
3. **mean rank 476.5 vs baseline 307.8**: 平均 rank 更差,可能是因为 11.D 的 20 cands 太少,或候选过于 attr-listing 格式
4. **Top-10 3.3% > 0% rank-1**: 比 Phase 10.18 exemplar search (0/30 top-10) 好,但仍低于 Phase 10.19 contrastive RAG (5/30 top-100)

## 为什么 Architecture B 失败

观察 11.D 输出的 query:
```
"Brand: TL Care; Item Weight: 1.21 pounds; Product Dimensions: 1 x 1 x 1 inches; Color: Ecru; Material: Cotton"
"Search for TL Care products; specifically, an item weighing 1.21 pounds; ensure it measures..."
```

11.D cands 几乎都是 **semi-structured attr-listing** (semicolon 分隔, "Brand: X; Item Weight: Y; ...") 或 **search-instruction format** ("Search for ...; specifically, ...")。这两种格式在 318d 句法特征空间里非常 generic,与所有 user 的 generic query 都很像,无法区分 target user。

而 Phase 10.19 contrastive RAG 通过让 LLM 看到 user 真实 query exemplars,产生更自然、更个性化的 phrasing("need a sturdy item, used for 3 months, mostly..." style),所以 top-100 命中更高。

## Phase 11 整体决策: NO-GO (Phase 11 范式不优于 Phase 10.19)

| Phase | Status | 关键结果 |
|-------|--------|----------|
| 11.A Gaussian cache | GO (内部) | held-out rank-1 73.5% (AnnaWegmann 768d 用户空间) |
| 11.B Diversity test | GO (内部) | sampled-z 1.57x diversity gain |
| 11.C Diffusion training | PARTIAL-GO | s_u signal 0.42, recon err 0.10, style transfer cos ~0 |
| 11.D E2E pipeline | COMPLETE | 600 cands in 10 min, attr 93%, style margin +0.95 |
| 11.E Validation (proxies) | PARTIAL-GO | Diversity + style margin OK, attr 93% |
| **11.F Rank-1 eval** | **NO-GO** | **rank-1 = 0/30 = 0%, top-10 = 3.3%** |

**Conclusion**: 
- ✅ **Gaussian N(μ, Σ) 范式有用** (diversity 1.57x, held-out 73.5% rank-1 在 AnnaWegmann 用户空间)
- ✅ **Conditional diffusion 学到了 s_u 信号** (recon err 0.10, s_u signal 0.42)
- ❌ **Architecture B e2e 没有突破 rank-1 ceiling** (0% on 30 pairs)
- ❌ **Phase 11.D 不优于 Phase 10.19 contrastive RAG** (3.3% top-10 vs 17% top-100)

**Root cause**: Architecture B 让 LLM 生成 semi-structured attr-listing (style hints 推动 generic 格式),这与所有 user 的 generic query 都很像,无法做 user-id signal。Contrastive RAG 直接让 LLM 看 user 真实 query exemplars,反而更能模仿 user 风格。

## 下一步

- 暂不推进 Phase 11 主系统路线
- Phase 10.19 contrastive RAG (rank-1 top-100 17%) 仍是最佳用户生成 baseline
- 后续改进方向 (不在 Phase 11 scope):
  - 让 Architecture B 的 style hints 更具体 (用真实 user exemplars 取代 z_personalized → NL translation)
  - 训练更深的 denoiser,直接 generation 而非 LLM 二次解码

## 文件

- Eval script: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase11_f_rank1_eval.py`
- Features: `phase11_d_candidates_318d.npy` (600 × 318)
- Report: `phase11_f_rank1_eval.json`

## 工程教训

1. **rank-1 eval 必须跑**: proxy metrics (diversity, attr coverage, style margin) 不能替代 VADES rerank
2. **318d 句法特征惩罚 attr-listing**: 当候选都是 "Brand: X; Item Weight: Y; ..." 格式,318d 看到的是 generic 句法,user-id signal 缺失
3. **Sample size matters**: 30 pairs vs 876 pairs,统计差异可能掩盖真实信号;后续应扩到 100-300 pairs
4. **Architecture B 限制**: diffusion 输出是 128d 风格向量,NL translation 损失大;直接用 user exemplars (Phase 10.19) 更有效
5. **Phase 11 5 阶段全部完成**: GO + PARTIAL-GO + NO-GO 各阶段结果都已记录,可作为后续研究基础