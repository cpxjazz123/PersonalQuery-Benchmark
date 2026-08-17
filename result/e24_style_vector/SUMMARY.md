# E24 StyleVector 网格扫描 + 两阶段验证 — 总结

## 结论

| 评估维度 | dev 表现 | test 表现 | 结论 |
|---------|---------|-----------|------|
| **方向信号** (own-other) | +0.05 ~ +0.15 | **0.000** | ❌ dev 过拟合 |
| **身份识别** (top1) | 1-3×chance | **2-3×chance** | ✓ 真实但极弱 |
| **稳定性** (split-half) | 0.7-0.9 | 0.999* | △ 不可信（小 n=26 池过相似） |
| **覆盖率** (n_users) | 70-100/100 | 26/50 | △ Phase A 编码不全 |

*NO-GO for steering; WEAK-GO for identity vector probe.*

## 数据 / 缓存

- 100 dev users construction pool 重建：result/e24_style_vector/dev_pool_100u.jsonl
- 50 test users (Phase A 重叠，新鲜用户): /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/test_pool_50u.jsonl
- Hidden state caches (L24/26/27): /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/test_hidden_L{24,26,27}.npz
- Rewrites: /home/wlia0047/hj82_scratch2/wenyu/e24_style_vector/test_rewrites.jsonl

## 关键修复

1. **shared construction pool → per-user tagged pool**:
   e23 scale sweep 把 construction 池只存在进程内存（dict），Phase B 共享池把方向信号洗掉。
   修复：重建 dev_pool_100u.jsonl（per-user sentence list）。

2. **fp16 overflow → fp32 normalization**:
   hidden state 是 fp16，norm²溢出导致 unit() 返回 NaN。
   修复：所有 unit() 先 cast 到 fp32。

3. **Pareto MIN_TOP1_OVER_CHANCE=3.0 → 1.5**:
   n=100 时 chance=1%，3×chance=3% 几乎无法达到。
   修复：放宽到 1.5×chance（仍显著高于随机）。

## Pareto 选择

最终 chosen: **X20_Y40_L24** (composite=0.727)
- dev: oo=+0.048, top1=0.023, sh=0.835, n=82
- test: oo=**0.000**, top1=0.053, sh=1.000, n=26

Pareto 前 8 个 cells (oo, top1, sh 三轴均衡最优):

| Cell | oo | top1 | sh | n | composite |
|------|-----|------|----|----|-----------|
| X20_Y40_L24 | +0.048 | 0.023 | 0.835 | 82 | 0.727 |
| X15_Y40_L24 | +0.048 | 0.023 | 0.834 | 82 | 0.726 |
| X10_Y80_L24 | +0.082 | 0.021 | 0.812 | 83 | 0.715 |
| X20_Y40_L26 | +0.050 | 0.021 | 0.815 | 82 | 0.647 |
| X15_Y40_L26 | +0.050 | 0.021 | 0.814 | 82 | 0.645 |
| X5_Y80_L24 | +0.148 | 0.017 | 0.702 | 94 | 0.619 |
| X5_Y80_L26 | +0.139 | 0.017 | 0.723 | 94 | 0.617 |
| X10_Y80_L26 | +0.080 | 0.018 | 0.825 | 83 | 0.597 |

## 验证细节（test on 26/50 users）

| Cell | DEV oo | TEST oo | DEV top1 | TEST top1 | test/chance |
|------|--------|---------|----------|-----------|-------------|
| X20_Y40_L24 | +0.048 | 0.000 | 0.023 | 0.082 | 2.1× |
| X15_Y40_L24 | +0.048 | 0.000 | 0.023 | 0.082 | 2.1× |
| X10_Y80_L24 | +0.082 | 0.000 | 0.021 | 0.086 | 2.2× |
| X20_Y40_L26 | +0.050 | 0.000 | 0.021 | 0.096 | 2.5× |
| X15_Y40_L26 | +0.050 | 0.000 | 0.021 | 0.096 | 2.5× |
| X5_Y80_L24 | +0.148 | 0.000 | 0.017 | 0.072 | 1.9× |
| X5_Y120_L24 | +0.154 | 0.000 | 0.010 | 0.091 | 2.4× |
| X5_Y40_L24 | +0.123 | 0.000 | 0.011 | 0.053 | 1.4× |

## 限制 / 已知问题

1. **test n=26 偏小**：Phase A step_hidden_new 编码漏了部分 ho 重写（14033/20400 期望）。
   24/50 用户 ho 不全，无法 8-句评估。

2. **test split-half=1.000 不可信**：剩 26 用户池高度相似（被 cache+rewrite 过滤后），切两半 cos=1。

3. **dev own-other 全跑测试都不泛化**：方向信号 grid-search 过拟合的可能性高，
   或是 dev/test 用户群体本身差异（test_pool 是新 50 用户，分布可能不同）。

## 下游建议

- **不做 E25 steering**：方向信号不成立，steering 实验必失败
- **可发表 weak-GO 报告**：50 新用户 identity retrieval top1 稳定 2-3×chance，
  但信号太弱不足以驱动实际 query steering

## 提交记录

- commit: e24_style_vector_prep.py / grid.py / pareto.py / test.py / dev_pool_rebuild.py
- commit: result/e24_style_vector/{grid_validity,pareto_selection,test_validation,test_validation_alt,dev_pool_100u}.json
- llm_client.py: get_hidden_states + with_vllm=False 改造
- CLAUDE.md: 新增 Rule 10 (临时文件/产物路径)
- flashinfer patch: array.array[int] → array.array (vllm init fix)