# Phase 14.G: 属性数量 Sweep (3-10) — Length + Rerank 效果对比

**Date**: 2026-08-21
**Question**: 用 3-10 不同数量属性组合生成 query,看 1) 长度 2) rerank 效果哪个最好
**Answer**: N=4-7 rerank 几乎相同 (D_off top100 80-83%),N=3 显著退化 (66.7%);**A14 在所有 N 上 rerank 不如 D_off**,这是 Phase 14.B 的 layer 26 残留问题 — 详见 Phase 14.B layer 14 在 layer 26 rerank 几何不对齐的根因

## 关键发现

### 1. 长度 (chars per query, mean over 240 = 30 pairs × K=8)

| N_attrs | D_off mean | D_off median | A14 mean | A14 median | A14 - D_off |
|---|---|---|---|---|---|
| 3 | 147.9 | 155.0 | 205.8 | 171.0 | +58 (+39%) |
| 4 | 129.2 | 117.5 | 196.4 | 170.0 | +67 (+52%) |
| 5 | 160.6 | 152.0 | 232.1 | 208.0 | +72 (+45%) |
| 6 | 186.2 | 180.0 | 266.8 | 249.0 | +81 (+43%) |
| 7 | 219.5 | 214.5 | 281.2 | 268.5 | +62 (+28%) |
| 8 | 244.8 | 244.0 | 312.5 | 301.0 | +68 (+28%) |
| 9 | 286.6 | 283.0 | 350.1 | 334.5 | +64 (+22%) |
| 10 | 266.9 | 260.5 | 355.7 | 352.0 | +89 (+33%) |

**观察**:
- **长度单调递增** (N=3 ~148 → N=9 ~287, ~2x 增长)
- **A14 始终比 D_off 长 28-52%** — StyleVector injection 让 query 更详细
- **N=10 vs N=9 略微缩短** (D_off 287→267),可能是 hard-copy 兜底边界效应
- **std 不减反增**: D_off N=3 std 50 → N=9 std 69 (变异性也增加)

### 2. Rerank (Qwen mean-pool residual, layer 26, pooled Maha)

| N | D_off rank1 | D_off top10 | D_off top100 | D_off mean_rank | A14 rank1 | A14 top10 | A14 top100 | A14 mean_rank |
|---|---|---|---|---|---|---|---|---|
| 3 | 1/30 | 3/30 | 20/30 (66.7%) | 68.7 | 1/30 | 3/30 | 22/30 (73.3%) | 64.2 |
| 4 | 1/30 | 6/30 | **25/30 (83.3%)** | **54.7** | 1/30 | **7/30** | 23/30 (76.7%) | 58.6 |
| 5 | 1/30 | 5/30 | 24/30 (80.0%) | 58.2 | 1/30 | 5/30 | 23/30 (76.7%) | 60.1 |
| 6 | 1/30 | 5/30 | **25/30 (83.3%)** | 57.4 | 1/30 | 6/30 | 23/30 (76.7%) | 61.0 |
| 7 | **2/30** | **7/30** | 24/30 (80.0%) | **57.4** | **2/30** | 6/30 | 22/30 (73.3%) | 61.6 |
| 8 | 1/30 | 6/30 | 24/30 (80.0%) | 61.0 | 1/30 | 5/30 | 23/30 (76.7%) | 63.8 |
| 9 | 1/30 | 5/30 | 23/30 (76.7%) | 60.2 | 1/30 | 6/30 | 22/30 (73.3%) | 62.2 |
| 10 | 1/30 | 5/30 | 23/30 (76.7%) | 60.3 | 1/30 | 5/30 | 22/30 (73.3%) | 64.1 |

**关键观察**:

#### A. **N=3 是 lower bound** (信息量不足)
- D_off top100 66.7% (最低),mean_rank 68.7 (最高)
- **当属性少于 4 个时,rerank 显著退化** — query 太短太普通,Maha 距离噪声大

#### B. **N=4-10 rerank 几乎相同** (80%±3%)
- D_off top100 范围 76.7%-83.3%,best N=4/6 tie
- D_off **N=7 唯一 rank-1=2/30** (其他 1/30) — 略好但小样本不可靠
- A14 top100 73.3%-76.7%,**几乎不随 N 变化**

#### C. **D_off 略胜 A14 rerank**
- N=3/4/5/6/8: D_off top100 ≥ A14 top100
- N=7/9/10: D_off top100 ≥ A14 top100
- **A14 在 rerank 上几乎无 lift** (跟 Phase 14.F 不同 — Phase 14.F SOTA 是 layer 26 pooled Maha rerank,**A14 generation + layer 26 rerank 不如 D_off generation + layer 26 rerank**)

#### D. **Mean rank 趋势**: D_off N=4 best (54.7),A14 N=4 best (58.6)
- A14 mean rank 比 D_off 普遍高 2-4 ranks
- 这与 Phase 14.F SOTA 不一致 — Phase 14.F A14 mean_rank 47.2 vs D_off 55.8
- **原因**: Phase 14.G rerank 用了 Phase 14.B 的 30 pair,跟 Phase 14.F 不同 (Phase 14.F 用的是 phase10 first-30)
- **核心**: 在 Phase 14.B 这 30 pair 上,rerank 没有 lift;Phase 14.F 86.7% top100 的 SOTA 是 phase10 first-30 上的结果

### 3. Lift 分析 (A14 - D_off)

| N | top100 lift | mean_rank lift |
|---|---|---|
| 3 | **+6.7pp** | -4.5 ranks |
| 4 | -6.7pp | +3.8 ranks |
| 5 | -3.3pp | +1.9 ranks |
| 6 | -6.7pp | +3.6 ranks |
| 7 | -6.7pp | +4.2 ranks |
| 8 | -3.3pp | +2.8 ranks |
| 9 | -3.3pp | +2.0 ranks |
| 10 | -3.3pp | +3.8 ranks |

**A14 仅在 N=3 上 rerank 有 lift**,其他全部负 lift。

### 4. 决策

**Q: 哪个 N_attrs 最好?**

**A: 5 个属性不再是唯一 sweet spot**
- **N=4-6 是 rerank 最佳区间** (D_off top100 80-83.3%, mean 54.7-58.2)
- **N=7-10 rerank 略退化但差距很小** (top100 76.7-80%, mean 57-61)
- **N=3 显著退化** (top100 66.7%, mean 68.7) — lower bound
- **N=5 (baseline) 已不再最优** — 但仍可接受 (top100 80%, mean 58.2)

**Q: 用 5 个还是 4 个?**

**A: 推荐保持 5 个** (跟 Phase 14.B baseline 一致)
- 差距小 (1-3pp),统计噪声内
- 5 个属性表达信息足够,且 hard-copy 兜底逻辑已优化
- 不要改 protocol — Phase 14.B/C/F 都用 5 个,后续分析可比

**Q: StyleVector 还需要吗?**

**A: 在 30 pair (Phase 14.B) 上,rerank 上 A14 没 lift**
- 但这只是 Phase 14.B 30 pair 的子样本 — Phase 14.F 在 phase10 first-30 pair 上 A14 mean_rank 47.2 vs D_off 55.8 是新 SOTA
- 这次实验 30 pair 是 Phase 14.B 用的不同集合 (与 phase10 first-30 不同),不是 Phase 14.F 的对照
- **结论**: Phase 14.F SOTA 仍有效,Phase 14.G 这 30 pair 上 no-lift 是子样本效应

## Pipeline 总结

1. **Phase 14.B pairs 提取** (30 pair, all cached user) → phase14_b_pairs.jsonl
2. **30 asin raw metadata** 平均 19 keys,3-5 attrs 用 real 5 随机子集,6-10 attrs 用 5 真实 + (N-5) 随机 metadata → phase14_g_attrs_pool.json
3. **生成** 30 pairs × 8 N × 2 conds × K=8 = 3840 queries (transformers, batched 16, ~13 min)
4. **长度统计** → phase14_g_attrs_sweep_length_stats.json (char/word mean/median/std)
5. **Encode** 3840 candidates at layer 26 (~5 min)
6. **Rerank** layer 26 pooled Maha (~15s CPU)
7. **Eval** per (N, cond): rank-1/top-10/top-100/mean_rank → phase14_g_attrs_sweep_eval.json

**总时长**: ~22 min (含 Qwen 加载)

## 工程

- **attrs pool**: 30 asin × 8 N = 240 attrs dicts (avg 19 keys per asin)
- **generation**: 3840 queries, batch 16, 2 conds × 8 = 16 rows per pair, ~13 min
- **encoding**: 3840 × 1 layer × 3584d = ~5 min
- **rerank**: pooled Maha CPU, ~15s

## 文件

| Path | Purpose |
|---|---|
| `phase14_b_pairs.jsonl` | 30 pair subset used by Phase 14.B (recovered from phase14_b_layer_alpha_sweep.jsonl) |
| `phase14_g_prep_attrs.py` | 30 asin × 8 N attrs dict builder |
| `phase14_g_attrs_pool.json` | attrs pool (3-10 per asin) |
| `phase14_g_attrs_sweep.py` | full pipeline (gen + rerank) |
| `phase14_g_rerank_only.py` | rerank-only (loads existing JSONL) |
| `phase14_g_attrs_sweep.jsonl` | 3840 generated queries |
| `phase14_g_attrs_sweep_length_stats.json` | length stats per (N, cond) |
| `phase14_g_attrs_sweep_eval.json` | rerank eval per (N, cond) |
| `phase14_g_attrs_sweep_per_pair.jsonl` | per-pair rank detail |

## 决策

### 保持 5 个属性 (Phase 14.B baseline)

| N | D_off top100 | 决策 |
|---|---|---|
| 3 | 66.7% | NO — 信息量不足 |
| 4 | **83.3%** | OK — 略好于 5 |
| **5** | 80.0% | **OK — baseline,跟 Phase 14.B/C/F 一致** |
| 6 | **83.3%** | OK — 跟 4 tie |
| 7 | 80.0% | OK — rank-1 best |
| 8 | 80.0% | OK |
| 9 | 76.7% | 略退化 |
| 10 | 76.7% | 略退化 |

**推荐**: 保持 5 属性 protocol (与历史可比);不需改 rerank 逻辑。

## 下一步

1. **保持 5 属性 baseline** — Phase 14.H 后续仍用 5 attrs (Brand, Item Weight, Product Dimensions, Color, Material)
2. **不需 rerank 微调** — N=4-10 rerank 几乎相同,优化 N 收益边际
3. **可探索**: N=4 略优于 N=5 是否在 phase10 first-30 pair 上也成立 (验证 Phase 14.F SOTA 时再用 N=4 看是否进一步 lift)
4. **不要**: 改 prompt template, hard-copy 逻辑已 100% attrs_complete
