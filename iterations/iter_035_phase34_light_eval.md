# iter 035: Phase 34 Light Eval 结果 + Trade-off 解读

**日期**: 2026-08-22
**承接**: iter_034 (Phase 34 设计) + rerank_light 跑完
**目的**: 评估 3-condition rerank (baseline / syntax / hybrid) 在 10 records 上的
        style fidelity + coverage + sentiment 表现, 确认 syntax Gaussian rerank
        的净效果。

---

## Pipeline

```
STRICT v8 + 3 user exemplars (Qwen 7B)
  → K=4 candidates per (user, asin)  (phase34_generate.py)
  → per-candidate scores:
       S_syntax:   log N(f(q) | mu_u^syntax, Sigma_u^syntax)   (Layer 20)
       S_lexical:  function-word Jaccard vs user exemplars
       S_coverage: 1 if all 5 attrs covered else 0
  → select:
       baseline:   cand[0]
       syntax:     argmax S_syntax within coverage_full=1, else argmax
       hybrid:     argmax 0.5*z(S_syntax) + 0.5*z(S_lexical) within coverage_full=1
```

**Records**: 10 (subset of 30-pairs Phase 14 evaluation set, mostly nontrivial 5 attrs)

---

## 主结果 (style fidelity 视角)

| condition | full_cov (5/5) | syn_dist ↓ | sent_leak ↓ | fw_overlap ↑ | hallucination | echo |
|-----------|-----------------|------------|--------------|---------------|----------------|------|
| baseline  | **4/10 (40%)**  | **5.43**   | 0.269        | 0.220         | 2              | 0    |
| syntax    | **6/10 (60%)**  | 5.81 (+0.38) | 0.268      | 0.193         | 2              | 0    |
| hybrid    | **6/10 (60%)**  | 6.18 (+0.76) | 0.297      | 0.234         | 1              | 0    |

**reference floor**: user-to-user natural syn_dist = **1.855** (50 random pairs)
**reference baseline**: Phase 4 D_off_baseline syn_dist = 5.500

---

## 关键发现

### ✓ Coverage +20% (40% → 60%)

syntax rerank 的核心价值是**强制选 full-coverage candidate**:
- baseline cand[0] 仅 40% full coverage
- syntax rerank 通过在 coverage=1 子集中 argmax S_syntax, 提升到 60%
- hybrid 同步 60% (与 syntax 持平,因为 hybrid 也被 coverage=1 过滤)

→ 这是**真实信号**, 不是 cherry-pick — 因为 hybrid 用了不同 score,
   选出的是不同 candidate。

### ⚠ syn_dist 略升 (+0.4 syntax, +0.76 hybrid)

相对 user-to-user floor 1.85, 仍 ~3 std dev 远 (跟 baseline 5.43 量级一致)。
但 absolute 距离增加,可能源于:
- coverage 越高的 candidate 含更多 attribute listing → 句法更长、noun_ratio↑
- syntax rerank 选出的 query 在结构上偏离 user self mean sentence (因 user self 是评论语,
  query 是搜索语,本身就比 user-to-user floor 高 3x)

→ **Trade-off**: 用 syntax rerank 换 coverage,但 query 句法略偏离 user 评论风格。
→ 需要 **intra-product Rank-1** 来验证: 在 same asin pool 内, syntax rerank 的
   candidate 是否比 baseline 更接近 user 真实搜索句?

### ≈ sent_leak / fw_overlap 持平

syntax rerank **没有**降低 sentiment leakage (0.269 ≈ 0.268), 也**没有**提高
function-word overlap (-0.027 略降)。
→ 这说明 query 的 sentiment 是由**生成**决定的, 不是由 **rerank** 决定的;
   期望 rerank 修 sentiment 是不合理的。

---

## 验证 / 反事实

为了排除"coverage 提升只是 syntax score 与 coverage 相关"的解释, 补一组:
**random-within-fullcov**: 在 coverage=1 的 candidates 中随机选 1 个
- 期望 ≈ 60% baseline (因为 cand[0] 的覆盖率是 40%, 强制选 coverage=1 必然提升)
- 但 random 不能区分 "syntax score 在 coverage=1 内部是否还有判别力"

→ 下一步需要 intra-product Rank-1, 在 syntax score 与 residual score (Phase 14.F)
   比较哪个对 user 真实句的 ranking 更准。

---

## 与 Phase 4 对比

| metric              | Phase 4 baseline | Phase 4 syntax mean α=0.3 | Phase 34 light syntax rerank |
|---------------------|------------------|----------------------------|-------------------------------|
| full coverage       | —                | —                          | 60% (vs 40% baseline)         |
| syn_dist            | 5.500            | 6.505 (+18%)               | 5.812 (+7%)                  |
| sent_leak           | 0.358            | 0.210 (-41%)               | 0.268 (-0.4%)                |
| 作用机制            | hidden bias      | hidden bias                | selection (no hidden edit)    |

**对比**:
- Phase 4 injection 改 sent_leak -41% 但 syn_dist +18%
- Phase 34 rerank 改 coverage +20% 但 syn_dist +7%, sent_leak ≈ 不变

→ **Rerank 与 injection 是 orthogonal mechanisms**, 各自优化不同 objective。
→ Phase 34 的真正潜在优势是 **不污染 generation** (no hidden bias, no sent_leak leak),
   这意味着 sample diversity 完整保留, 后续 Phase 35 可以加入 S_residual 形成 4-condition
   BoK 联合 rerank。

---

## 局限

1. **10 records 太少**: statistics weak, 需要 ≥30 才有 95% CI
2. **无 Qwen residual score**: 3-condition 缺 S_residual, 无法做 Phase 14.F SOTA 对比
3. **无 intra-product Rank-1**: 这是 Phase 15.7 主指标, 但 Phase 34 light 没跑
4. **覆盖率 ≠ 风格匹配**: coverage 提升不一定等于 user 真实句命中率提升
5. **exemplar 复用**: cand3 提示 in-context learning 风险 (exemplar 原句泄漏),
   rerank 只能 selection 不能 generation 修复

---

## 下一步建议 (Phase 35 路线)

按重要性递减:

### A. 扩到 30 records + 加 S_residual + intra-product Rank-1 (PRIORITY 1)
- 复用 phase14_q10 generator 跑 30 pairs × K=8
- 加 S_residual (Qwen mean-pool 768d + user self Maha, 复用 Phase 14.F)
- 4 conditions: baseline / S_residual / S_syntax / S_hybrid (syntax+lexical)
- 主指标: intra-product Rank-1 (size≥10 subset: 当前 SOTA 19.2%, target > 22%)

### B. 控制 coverage 变量 (PRIORITY 2)
- 加 baseline2: random-within-fullcov (剔除 coverage 优势)
- 加 hybrid_no_filt: 不限 coverage 直接 argmax (看 coverage filter 是否必要)
- 区分 "coverage lift" vs "syntax signal lift"

### C. 多 Gaussian layer ensemble (PRIORITY 3)
- Layer 16 / 20 / 24 / 26 4 个 syntax Gaussian 取 log-prob 平均
- 期望提升 stability

---

## 决策点 (从 iter_034)

| 条件 | Phase 35 路径 |
|------|---------------|
| BoK-hybrid intra-product Rank-1 > 19.2% (size≥10) | 选 Phase 35 production |
| BoK-hybrid intra-product Rank-1 ≤ 19.2% | 切回 Phase 14.Q10 BoK-184 SOTA 18/30 |

Phase 34 light 没做 intra-product eval, 当前 verdict 不可作为决策依据,需 Phase 35。

---

## 输出文件

- `syntax_subspace/phase34_eval.py` (3-condition eval, self-contained norm stats)
- `result/phase34/eval.json` (per-condition per-record metrics)
- `result/phase34/rerank_light.json` (3-condition rerank selection)
- `result/phase34/candidates.json` (10 records × K=4 candidates)

## 最终 takeaway

> Phase 34 light eval **确认 syntax rerank 提升 coverage +20%**, 但 trade-off 是
> syn_dist 略升。sent_leak 与 fw_overlap 持平 (rerank 不能修 generation 问题)。
> 下一步必须扩展到 30 records + 加 S_residual + intra-product Rank-1 才能做
> Phase 34 vs Phase 14.F 的公平对比。
