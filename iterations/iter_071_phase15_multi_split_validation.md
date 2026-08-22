# Phase 15: 大样本 LOPO 多方案验证 (5 splits × 500 test users)

## Context

Phase 14.Q10-L 在 2,041-fold LOPO 上暴露关键问题: 30 pair 60% rank-1 是 overfit 假象,
真实 2041-scale rank-1 = 14/2041 = 0.69%。瓶颈不是候选数量,而是:
1. 用户之间真实风格重叠 (style overlap)
2. 当前表示空间未充分学习"内容无关的用户差异"

本 phase 设计 5 种方案,在 5 splits × 500 test users (= 2,500 test pairs) 严格大样本上对比。
所有方案复用 Phase 14.F SOTA PCA-300+500 Maha rerank + Qwen layer 26 mean-pool residual。

## 方案与结果

| 方案 | rank-1 | top-100 | MRR | mean_rank | vs baseline |
|------|--------|---------|-----|-----------|-------------|
| **15.1 D_off baseline** | 0.56% (14/2500) | 20.48% | 0.0182 | 666 | — |
| 15.2 supervised contrastive | 0.24% (6/2500) | 4.44% | 0.0058 | 1017 | **NO-GO** (-57%) |
| 15.3 hierarchical K=30 | 0.56% (14/2500) | 20.48% | 0.0182 | 666 | NO-GO (=) |
| **15.4 per-user exemplars** | **1.40% (35/2500)** | 20.40% | **0.0325** | 646 | **GO (+150%)** ★ |
| 15.5 distinctiveness-aware | 0.16% (4/2500) | 20.44% | 0.0151 | 646 | NO-GO (-71%) |

CI95: 15.4 MRR [0.0276, 0.0380] vs 15.1 MRR [0.0152, 0.0216] — 不重叠,统计显著。

## 关键发现

### 1. Per-user Exemplar Generation (15.4) 是 GO — NEW SOTA

为每个用户选 3 个最 distinct 句子 (按 z-score 长度/punct/connectors 选),
strip 掉 brand/model/size/price 等产品词,作为 few-shot exemplars 生成 query。
- **rank-1: 0.56% → 1.40% (2.5x)**
- **MRR: 0.0182 → 0.0325 (1.79x)**
- 每个 split 都 6-9 个 rank-1 (vs baseline 2-5 个)

为什么有效: exemplars 让 LLM 直接模仿用户实际写作风格,生成 query 携带用户语言特征
(句长、连接词、标点、口头禅),Qwen hidden 更接近 target 用户真实分布。

### 2. Hierarchical K=30 (15.3) 是 NO-GO

K-means K=30 聚类,然后对每个 cand 找 top-3 cluster,在 cluster 内重排。
- Result: rank-1 0.56% (与 baseline 完全一致)
- **target_reachable 0.1%** (3/2500): D_off 生成的 query 风格过于平均,
  几乎不在 target user cluster 的 top-3 邻居中
- 启示: 用户风格空间是**连续的**,不是 cluster 离散的;硬聚类反而丢失用户区分信号

### 3. Distinctiveness-Aware Scoring (15.5) 是 NO-GO — 反直觉

直觉: 给"独特"用户加权 → 他们的 query 应该更易识别
- 实际: rank-1 **降低 3.5x** (0.56% → 0.16%)

**Stratified by distinctiveness quartile**:
| Quartile | rank-1 | top-100 | mean_rank |
|----------|--------|---------|-----------|
| Q1 (least distinctive) | 0.16% | **36.00%** | **323** |
| Q2 | 0.32% | 19.68% | 529 |
| Q3 | 0.00% | 15.36% | 704 |
| Q4 (most distinctive) | 0.16% | **10.72%** | **1027** |

**反直觉**: 最不独特的用户 top-100 = 36%,最独特的 top-100 = 11%。
**独特用户的 query 反而最难被检索到** (style 太独特,生成器无法稳定复现)。

→ 推论: 用 user distinctiveness 做加权是错的;应该**反转**(给平均用户加权)
或用 distinctiveness 做**难度分层报告**(不参与评分)。

### 4. Supervised Contrastive (15.2) 是 NO-GO

NT-Xent 30 epochs,manual gradient computation
- loss 卡在 6.77,完全未收敛
- rank-1 = 0.24% (vs baseline 0.56%) → 比 baseline 还差
- 说明: 简单投影 + 3584d→128d + NT-Xent 在 1541 train user 上不足以学到有意义的 style 空间
- 改进方向: SimCSE / 真实 triplet (positive=user其他句,negative=其他user句) / InfoNCE
  重新设计,但 1541 用户可能 too small

## Stratified by Distinctiveness 的方法论意义

D_off baseline 在 Q1 用户上 top-100 = 36%,Q4 用户 top-100 = 11%。
意味着**当前系统对所有用户的 retrieval 难度高度不均**:
- "简单"用户 (~25%, 顶级普通风格) → 几乎都能进 top-100
- "难"用户 (~25%, 高度独特风格) → 仅 11% 进 top-100

未来要破 rank-1 ceiling,核心是**对难用户(Q3-Q4)做特殊处理**:
- 给难用户更多 exemplar shots (5 → 10)
- 用 user 自适应 prompt 强调独特 style (Phase 14.Q9 已尝试但弱)
- 改 hard negatives mining,让 ranking 学到"uniqueness vs uniqueness"的细微差异

## 文件清单

| Path | 用途 |
|------|------|
| `phase15_baseline_lopo.py` | 15.1 multi-split baseline |
| `phase15_2_contrastive.py` | 15.2 NT-Xent (NO-GO) |
| `phase15_3_hierarchical.py` | 15.3 K-means K=30 cluster rerank |
| `phase15_4_exemplar_generate.py` | 15.4 per-user 3-shot exemplar generation |
| `phase15_4_lopo.py` | 15.4 LOPO encoding + rerank |
| `phase15_5_distinctiveness.py` | 15.5 distinctiveness-weighted scoring |
| `phase15_6_combined_analysis.py` | 15.6 全部对比 + stratified report |
| `phase15_*_lopo_eval.json` | 每方案 cross-split aggregate |
| `phase15_4_generated.jsonl` | 16,328 per-user-exemplar queries |
| `phase15_6_combined_summary.json` | 最终对比表 + verdict |

## 下一步建议

**短期** (继续在 2041 用户上探索):
- **15.4 + 15.5 反转**: 给"非独特"用户加权 (Q1 36% top-100 拉到更高 rank-1)
- **15.4 + 难用户优化**: 对 Q3-Q4 distinctiveness 用户用 10 exemplars 而非 3
- **15.4 BoK-24**: 15.4 K=8 × 3 subsets = 24 cands,看 rank-1 能否翻倍

**中期** (破 rank-1 ceiling):
- 跨 5 splits 联合训练一个 reranker (pairwise loss)
- 用 Qwen 句级别 hidden 替代句子级 mean-pool,捕捉更细粒度风格

**长期**:
- 10K user scale requires 扩大数据基础 (Baby Products 2023 仅 2,043 用户 ≥30 评论,
  需跨多个 category 拼接或换数据集)

## Verdict

**15.4 per-user exemplars 是 NEW SOTA**,MRR 1.79x baseline,rank-1 2.5x baseline。
15.5 distinctiveness 反转 (down-weight distinct users) 是有希望的下一步。
15.2/15.3 NO-GO,无后续价值。
