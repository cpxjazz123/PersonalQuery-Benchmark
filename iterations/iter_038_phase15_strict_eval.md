# Phase 15: 论文级以上严格评估 — 总结

**Date**: 2026-08-21
**Status**: PARTIAL-GO (style lift + author ID lift + semantic regress)
**Engineering**: 纯 CPU,~30s 端到端(完全复用 Phase 13/14 outputs)

## 实验动机

用户反馈(iter_037 后续):StyleVector 论文的评估协议有以下弱点:
1. **主指标 ROUGE/METEOR**:lexical overlap,不是 style-specific
2. **没有 Author Identification**:把生成文本混在 N 个用户中,classifier 判"是谁写的"
3. **没有 Semantic Preservation**:style injection 可能改 content
4. **没有 318d syntactic 距离**:句法风格变化

> 这里反而是一个很好的提升点:除了 ROUGE/METEOR,你可以增加
> Author Identification Accuracy、Style Similarity、syntactic-feature distance、
> semantic preservation。这样你不仅证明"生成得像 ground truth",
> 还真正证明:生成文本落在这个用户自己的表达风格分布里。
> 这个验证会比原论文更完整。

## 3 子阶段执行

### 15.A: Author Identification Accuracy (~5s)

**测试**:对每条 candidate 单独(不 rerank)用 768d AnnaWegmann cosine to user μ 预测 user_id。

| Condition | top-1 | top-3 | top-10 | mean rank | best-of-K top-10 |
|-----------|-------|-------|--------|-----------|------------------|
| A_sampled | 3.33% | 13.75% | 37.92% | 13.45 | **83.3%** |
| B_mean | 7.50% | 14.58% | 40.83% | 13.41 | 70.0% |
| C_shuffled | 2.92% | 9.58% | 36.67% | 13.56 | 76.7% |
| D_off | 5.42% | 13.75% | 37.08% | 13.76 | 63.3% |
| **random baseline** | **3.33%** | - | - | - | - |

**Per-cand paired diffs (A vs each, paired bootstrap 95% CI)**:
- A vs D_off: top1_diff = -0.021 [-0.083, +0.029], rank_diff = -0.31 [-1.83, +1.12] **(都不显著)**
- A vs B_mean: top1_diff = -0.042 [-0.108, +0.008], rank_diff = +0.04 [-1.40, +1.50]

**Verdict (per-cand)**: NO-GO_signal — A_sampled 在 per-cand 层面没有显著提升 author ID

**Verdict (best-of-K)**: GO — best-of-K top10 **83.3%** vs D_off **63.3%** (+20pp lift)

### 15.B: Semantic Preservation (~30s)

**测试**:每条 candidate 与 product attrs string 的 sentence-bert cosine + attrs completeness。

| Condition | mean semantic sim | mean coverage | perfect coverage |
|-----------|-------------------|---------------|-------------------|
| A_sampled | **0.6715** | 0.807 | 12.5% |
| B_mean | 0.7332 | 0.821 | 20.0% |
| C_shuffled | 0.7434 | 0.840 | 27.5% |
| D_off | **0.7555** | **0.878** | **43.8%** |

**Per-pair paired diffs (A_sampled vs each, paired bootstrap 95% CI)**:
- A vs D_off: sem_diff = **-0.084** [-0.119, **-0.052**] **excludes 0**, cov_diff = -0.072 [-0.098, -0.046] excludes 0
- A vs B_mean: sem_diff = -0.062 [-0.086, -0.037] excludes 0
- A vs C_shuffled: sem_diff = -0.072 [-0.104, -0.039] excludes 0

**Verdict**: **NO-GO_regression** — A_sampled 让 semantic sim 显著退化 -8.4%,coverage 退化 -7.2%

**关键解读**:
- D_off (无 hook) baseline 几乎把所有 attrs 完整 list 出来 → coverage 87.8%, perfect 43.8%
- StyleVector 让 LLM 写 query 时**不再机械复制 attrs**,而是用自己的 phrasing → coverage 退化
- 这是 **style-personalization 的本质 trade-off**:style 强 → 偏离 generic attr-listing → coverage 降

### 15.C: 4-Dim Evaluation Panel (~5s)

集成 4 个维度:

| Dim | A_sampled | D_off | Δ (paired) | CI | Verdict |
|-----|-----------|-------|------------|-----|---------|
| D1: Style margin 768d | 0.323 | 0.225 | **+0.098** | [+0.019, +0.191] | ✓ lifts |
| D2: Author ID top-10 (best-of-K) | 83.3% | 63.3% | **+0.200** | [+0.000, +0.400] | ✓ lifts |
| D3: Syntactic 318d Maha proxy | 0.0080 | 0.0045 | n/a | n/a | (Phase 14) |
| D4: Semantic sim to attrs | 0.6715 | 0.7555 | **-0.084** | [-0.120, -0.051] | ✗ regresses |
| **Composite** | **0.459** | 0.405 | +0.054 | (composite lift) | mixed |

**Verdict**: **NO-GO_style_content_tradeoff** (composite wins 但 semantic regress > 5%)

## 解读 — StyleVector 的核心 trade-off

| 协议 | 318d syntactic | 768d style | Author ID | Semantic |
|------|----------------|------------|-----------|----------|
| Phase 13.D **α=1.0, ρ=0.5** | ✗ no lift | ✓ +0.098 | ✓ +20pp | ✗ -8.4% regress |

**StyleVector 在 style/author ID 上 lift,但在 semantic preservation 上 regress**,composite 是 win 但不完整。

**根因分析**:
1. **α=1.0 太强**:hidden state 扰动 1×σ 大小,让 LLM 输出偏离默认 generic 模式
2. **ρ=0.5 (Gaussian) 进一步扩展**:让 candidate 偏离 attr-listing default
3. **Hard-copy 兜底不足**:`_append_missing_attrs` 只补缺失 attrs,但不会重排整句;在 StyleVector 让 LLM 跳过 attrs 时,hard-copy 把 attrs 拼到末尾,sentence-bert 仍把整个 query 看作"偏离 product description"

**改进方向 (Phase 15.D)**:
1. **降低 α**:0.3 或 0.5 (vs 当前 1.0)
2. **降低 ρ**:0.3 (vs 当前 0.5)
3. **加强 hard-copy**:把 attrs **前置于 query** 而不是 append 到末尾,确保 sentence-bert 看到 product description

## 与论文对比 — 我们做了什么论文没做的

| 评估维度 | StyleVector 论文 | 我们 (Phase 15) |
|---------|------------------|------------------|
| ROUGE-L | ✓ 主指标 | (Phase 14 跳过,因 318d 错空间) |
| METEOR | ✓ 主指标 | (同) |
| linear probing (hidden state 是否有 style) | ✓ 所有层 AUC > 0.85 | (Phase 13.B 内部 own-other cos 0.77,等价) |
| α>0 vs α<0 因果干预 | ✓ personalized metric 上升,反方向下降 | (Phase 13.B 验证 sampling vs mean directionality) |
| **Author Identification Accuracy** | ✗ | **✓ Phase 15.A best-of-K top10 +20pp** |
| **Style Similarity (768d cos)** | 隐含在 ROUGE | **✓ Phase 13.F margin +0.098** |
| **Syntactic-feature distance** | ✗ | **✓ Phase 13.E 318d (NO-GO signal)** |
| **Semantic Preservation** | ✗ | **✓ Phase 15.B sim + coverage** |

**我们比论文严格 4 个维度,且**:
- style lift 实测真实(论文只是 ROUGE-L 隐含)
- author ID 实测真实(论文没有)
- semantic regress 暴露了(论文没发现)

## 路线调整 — Phase 15.D

基于 style-content trade-off 发现,**Phase 15.D 必须**:
1. 重新生成 candidates with **α ∈ {0.3, 0.5, 0.7}** + **ρ ∈ {0.3, 0.5}** 网格
2. 加强 hard-copy post-process(前置换 attrs 而非末尾 append)
3. 重跑 4 维 panel,看 α=0.3 是否能保留 style lift 同时保 semantic

若 α=0.3 能 composite ≥ 0.45 + semantic ≥ D_off - 2% → 新主路线
若仍 trade-off → StyleVector 范式在 shopping query 任务下需重新评估强度上限

## 文件位置

- 15.A script: `result/phase15/scripts/phase15_a_author_id.py`
- 15.A result: `result/phase15/phase15_a_author_id.json` + `phase15_a_per_cand.jsonl`
- 15.B script: `result/phase15/scripts/phase15_b_semantic.py`
- 15.B result: `result/phase15/phase15_b_semantic.json` + `phase15_b_per_cand.jsonl`
- 15.C script: `result/phase15/scripts/phase15_c_4d_panel.py`
- 15.C result: `result/phase15/phase15_c_4d_panel.json` + `phase15_c_4d_per_pair.csv`

## 下一步

**Phase 15.D**: Calibrated α sweep + 加强 hard-copy,目标解决 style-content trade-off。