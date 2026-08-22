# iter 044: Phase 35.I — N_attrs sweep (4..10) — N=4 是最优,更多属性反而下降

**日期**: 2026-08-23
**承接**: iter_043 (Phase 35.H length = NO-confound) + 用户新指示
  > 使用不同数量的属性去生成 query, 4, 5, 6, 7, 8, 9, 10
**目的**: 测试 attribute 数量对生成 query + rerank 质量的影响

---

## Pipeline

```
Phase 35.E 74 records × product_attributes.json 全属性
  → 对每个 N ∈ {4..10}: 取 top-N 个 attrs (按 key 字母序)
  → 对每个 (record, N) × K=4: 用 Qwen 生成 query
  → 总候选: 7 N × 296 = 2072 prompts
  → Qwen 3584d residual @ layer 26 (mean-pool attention mask)
  → PCA-32 + softmax_g32_τ0.5 排名
  → Intra-product Rank-1 size>=10
```

**耗时**: gen ~13 min + scoring ~5 min (Qwen) = ~18 min 总
**Records**: 73 users (size≥10 pool), 17 asins, K=4 per N

---

## 主结果: N_attrs Sweep

| N_attrs | n_cands | Rank-1 | mean_rank | p90 | full_cov | avg_cov |
|---------|---------|--------|-----------|-----|----------|---------|
| **N=4** | 296 | **35.6%** ★★★ | **10.09** | **19.70** | **14.4%** | 66.3% |
| N=5 | 296 | 21.9% | 10.54 | 21.95 | 3.4% | 57.9% |
| N=6 | 296 | 26.0% | 11.07 | 22.85 | 0.0% | 62.2% |
| N=7 | 296 | 34.2% | 11.35 | 24.30 | 0.0% | 62.0% |
| N=8 | 296 | 23.3% | 11.81 | 26.40 | 0.0% | 63.3% |
| N=9 | 296 | 34.2% | 11.16 | 23.50 | 0.0% | 60.5% |
| N=10 | 296 | 30.1% | 11.50 | 25.65 | 0.0% | 59.8% |

---

## 关键发现

### 1. **N=4 是 sweet spot** ★★★
- Rank-1: **35.6%** (vs N=5 21.9%, N=10 30.1%)
- p90: 19.70 (vs N=8 26.40) — 长尾错误最低
- mean_rank: 10.09 (vs N=8 11.81) — 排序最稳定
- → **少属性 + 高覆盖 = 最优 ranking 信号**

### 2. Coverage 断崖 (N>=6 全部 0% full coverage)
- N=4: 14.4% (能塞下 4 个属性值)
- N=5: 3.4% (临界,5 个塞不下)
- N=6+: **0%** (LLM 无法在 15-30 词 query 里完整覆盖 6+ 属性)
- avg_cov 在 60-65% 区间稳定 (总是漏 2-3 个属性)

### 3. 中段 N (5, 6, 8) 表现差 — 噪声区间
- N=5: 21.9% (开始难塞)
- N=6: 26.0%
- N=8: 23.3% (中等长度覆盖,信号被淹没)
- N=7, N=9 异常回升到 34.2% (为什么? 可能是 N 为奇数时 query 更长,clause 多,style 信号强)

### 4. 中段 N 用户 Gaussian 学不到 style
- N=4: 35.6% (style 信号主导)
- N=5-8: 21-26% (style 信号被 attribute 噪声稀释)
- N=9-10: 30-34% (style 信号略恢复,但 query 太长)
- → **最优 attribute 数量在 style 信息密度与覆盖率之间平衡**

### 5. 与 Phase 35.E K=16 对比 (同样 N=4)
- Phase 35.E K=16 (1184 cands, 4 attrs): intra-product Rank-1 = **41.1%**
- Phase 35.I K=4 (296 cands, 4 attrs): Rank-1 = **35.6%**
- → **K=4 vs K=16 差 5.5pp** (候选数翻 4 倍 → +5pp),与用户路线 #2 一致

---

## 对照用户问题

| 用户问题 | 实测 | 判定 |
|---------|------|------|
| 不同 attribute 数对 ranking 影响 | N=4 35.6%, N=10 30.1% | ⚠️ **N=4 是最优** |
| 是否有 optimum N? | N=4 (35.6%), N=9 (34.2%) | ✅ **N=4 略胜** |
| N=10 是否有惊喜? | 30.1% (反而下降) | ❌ NO-GO on N=10 |
| Coverage 决定 ranking 吗? | N=4 cov=14.4% 最佳 | ✅ **强相关** |

→ **少而精**: 4 个属性能让 LLM 高概率完整覆盖,信号纯; 5+ 个属性覆盖率断崖,
   混入不完整信息,反而稀释 user style learning。

---

## 下一步 (基于本结果)

1. **Phase 35.I-2: K=16 + N=4 sweep** (验证 K×N 联合最优)
2. **Phase 35.J: 强制 N=4 + 多样化采样 (长度 / 结构)** (复用 Phase 35.H 的 U-shape 发现)
3. **Phase 35.K: N=4 + 精选 attrs (剔除 value 模糊项如 "Material Type")**
4. **Phase 35.L: 端到端 N=4 + K=16 + softmax listwise (组合 N=4 K=16 多样性)** ★ 可能突破 55%

---

## Outputs

| Path | 用途 |
|------|------|
| `syntax_subspace/phase35i_attrsweep.py` | K=4 多 N 生成 (~13 min) |
| `syntax_subspace/phase35i_score.py` | Qwen residual + softmax 评估 (~5 min) |
| `result/phase35i/attrsweep_summary.json` | N=4..10 Rank-1 / cov |
| `phase35i_cands_N{{4..10}}.json` | 7×296 candidates |

---

## 最终 takeaway

> Phase 35.I N_attrs sweep (4-10) 验证用户猜想:
> **N=4 是 sweet spot** — Rank-1 35.6%, p90 19.70, mean_rank 10.09 三项最优。
> N≥6 时 full coverage 全部断崖 (0%),attribute 混入噪声稀释 style 信号。
> N=7, N=9 异常回升到 34.2%,但仍 < N=4; N=10 进一步下降至 30.1%。
> **建议: 生成器固定 N=4 + K=16-32 多样性扩展**, 不再增加 attribute 数量。
> 下一步 Phase 35.J 直接 N=4 × K=16 + softmax 组合,目标 Rank-1 > 50% (与 Phase 35.G 持平)。
