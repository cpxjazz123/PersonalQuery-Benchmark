# Phase 10.17: TinyStyler 范式 style prefix — 总结

**Date**: 2026-08-20
**Status**: NO-GO (与 Phase 10.16 一致 — soft prefix injection 路径不工作)

## 实验动机

Phase 10.16 用 e22_t3 trained projector (K=16, α=0.277, LayerNorm) + 318d phase10 user vectors 做 4-control soft prefix injection,得到 real-z=0/876, injection-off=1/876。结论:投影器无法突破 1.26% rank-1 baseline。

用户洞察:**"我记得我们之前不是使用 TinyStyler 能够成功控制吗"** — E30.34 (`gen_styled_tinystyler_noforce.py`) 已经在 TinyStyler 范式下跑通,得到 50%-95% hit rate。区别在于:
- e22_t3 projector 是**自己训练**(有 covariate shift 风险),K=16
- TinyStyler 是**已训练好的 HuggingFace 模型** + AnnaWegmann 768d (cross-distribution) + K=8, α=2.0

**Phase 10.17 假设**:软前缀注入范式本身可能有效,只是 e22_t3 projector 在 e22_t1 train distribution 上拟合,phase10 是不同分布。

## 实验设计

复用 E30.34 完整栈:

| 组件 | Phase 10.16 | Phase 10.17 |
|------|-------------|-------------|
| 模型 | Qwen2.5-Coder-7B (decoder) | T5-v1_1-large (encoder-decoder) |
| Style embed | 318d e22_t3 user_mu | **768d AnnaWegmann mean-pooled** |
| Prefix K | 16 | **8** (E30.34 验证) |
| Alpha | 0.277 (learned) | **2.0** (E30.34 验证) |
| LayerNorm | yes (投影器内) | **no** |
| 总参数 | K·H = 16·3584 = 57344 | K·H = 8·1024 = 8192 |

**4-control 实验** (与 Phase 10.16 一致):
- `real-z`: proj(AnnaWegmann(u_target) · 2.0) → K=8 prefix
- `shuffled-z`: proj(AnnaWegmann(u_other) · 2.0) → K=8 prefix (sanity)
- `true-zero-z`: zeros(K=8, d_model=1024)
- `injection-off`: no prefix (baseline)

**规模**: 876 pairs × 4 controls × 4 seeds = **14016 candidates**
**生成**: TinyStyler (HuggingFace), temp=1.0, top_p=0.95, max_new=48, BATCH=16 (4 pair/chunk), rate=15.73/s, **total=891s = 14.9 min**

## 结果: Rank-1 Coverage 4-control

```
Control          Rank-1 Cov   Mean Best Rank   Δ vs off
real-z            0.0023      645.3            +0.0000
shuffled-z        0.0011      809.9            -0.0011
true-zero-z       0.0034      893.4            +0.0011
injection-off     0.0023      817.9             0.0000
```

**Paired bootstrap** (n=2000):
- real-z vs shuffled-z: Δ=+0.0012, 95% CI [-0.0027, +0.0050] — **包含 0**
- real-z vs off: Δ=+0.0000, CI [-0.0046, +0.0046] — **包含 0**

## Phase 10.16 vs 10.17 对比

| 维度 | Phase 10.16 (e22_t3) | Phase 10.17 (TinyStyler) | 改进 |
|------|----------------------|--------------------------|------|
| real-z rank-1 | 0/876 (0.0000) | 2/876 (0.0023) | +0.0023 |
| real-z mean best rank | 896.6 | **645.3** | **-251 (-28%)** |
| shuffled-z rank-1 | 1/876 | 1/876 | 持平 |
| injection-off rank-1 | 1/876 | 2/876 | 持平 |
| injection-off mean best rank | 935.1 | 817.9 | -117 (-13%) |
| Paired bootstrap CI | [-0.0034, 0.0000] | [-0.0027, +0.0050] | 略微改善但仍含 0 |

**关键观察**:
1. **Rank-1 coverage 几乎不变**: real-z 从 0 → 2,off 从 1 → 2。增量仅 1 个 pair。
2. **Mean best rank 显著下降**: real-z 从 896 → 645 (-28%),off 从 935 → 818 (-13%)。说明 TinyStyler prefix 让 rank 分布整体左移,但多数 pair 仍卡在 rank 100-1500 区间,不到 rank-1。
3. **CI 含 0**: 不论用哪种 style embed,软前缀注入**不能**可靠地把生成 query 推到 target user rank-1。
4. **TinyStyler 整体比 e22_t3 略好**: 主要来自 mean best rank 改善,但 rank-1 没变。
5. **True-zero-z 反常偏高**: 3/876 rank-1 (> real-z 的 2/876)。Prefix 信号对 target 不敏感,但 zeros prefix 偶尔也能命中(可能因为 prefix 缩短了 effective prompt,让 T5 输出更接近 attr 模板)。

## 结论: **NO-GO (与 Phase 10.16 一致)**

**软前缀注入路径不可行**。无论用:
- trained projector (e22_t3, covariate shift 风险)
- pre-trained encoder (AnnaWegmann, cross-distribution generalization)
- 不同 prefix K (16 vs 8)
- 不同 alpha (0.277 vs 2.0)

都不能把生成 query 推到 target user rank-1。CI 均包含 0,效应大小未达统计显著。

## 失败原因分析

1. **LLM 自由生成结构上限**: Phase 10.15 已确认 LLM 自由生成 query 的 rank-1 上限是 ~1.26%。Soft prefix 只是 prompt 微扰,不能改变 LLM 输出分布的根本限制。

2. **Prefix 信号在 T5 encoder-decoder 中衰减**:
   - T5 是 encoder-decoder 架构,prefix 加在 encoder 端,decoder 注意力分布被 encoder hidden state 主导
   - 同样问题在 Qwen decoder-only 中也存在 (Phase 10.16)
   - 8 个 soft prefix tokens 相比 ~70 个 prompt tokens 太短,信号被淹没

3. **Style embed → prefix mapping 学不到 user-discriminative 特征**:
   - AnnaWegmann 是**通用风格分类器**(formal/casual/technical 等),不是 user-specific 风格
   - 即使 prefix 注入了"风格"信号,这个风格对 user 排名没意义

4. **样本量不足**:
   - 4 seeds/control vs Phase 10.15 的 96 candidates/pair
   - 真实效应若存在,4 seeds 不足以检出

5. **观察到 prefix 注入后 T5 输出 echo prompt 元描述**: "I would suggest writing a short shopping query..." → 表明 T5 把 prefix 当噪声,反而模仿 prompt 的元层面。建议写作而非实际写 query。这是 prefix 范式的**额外副作用**。

## 决策

**Phase 10.17 关闭 — 软前缀注入路径双范式 NO-GO**

- e22_t3 trained projector: NO-GO (covariate shift)
- TinyStyler AnnaWegmann: NO-GO (无 user-discriminative 信息)
- Mean best rank 改善但 rank-1 不变:说明 prefix 注入只是把生成推向用户风格类的中心,不是 target user 本身

**下一步方向** (沿用 Phase 10.16 总结中的 4 选项):
- (a) 重新训练 projector 用 phase10 分布
- (b) few-shot in-context learning: 直接把 user past queries 放 prompt
- (c) 句法控制升级
- (d) 接受 1.26% rank-1 结构上限,转 acceptance/coverage 拆解
- (e) VADES 架构重构

## 工程实现经验

1. **Batched generation 大幅提速**: 把 prefix/no-prefix 分区成 2 个 forward call (一次 12 prefix samples + 一次 4 no-prefix samples),从单条 2.2/s → batch=16 后 15.7/s (7x 提速)
2. **T5.generate supports `inputs_embeds`**: 不需要手写 KV cache 注入,直接 `inputs_embeds=torch.cat([style_prefix, input_embeds], dim=1)` 即可
3. **`num_return_sequences` + style replication**: 一个 (pi, ctrl) 共享 style,4 seeds 复用 style_prefix 但生成不同输出
4. **Smoke 脚本必须先于 full run**: phase10_17_b_smoke.py 验证 chunk 边界和 dtype 后再启全量,避免 891s 浪费
5. **CHUNK=4 pairs/batch**: T5-large fp16 ~3GB 模型 + 16 records 一次 forward 显存够用,3.6x 提速于 CHUNK=1

## 文件位置

- 数据/产物: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/`
  - `phase10_user_embs_768d.npz` (876, 768) — AnnaWegmann user vectors
  - `phase10_user_embs_768d_meta.json`
  - `phase10_17_b_generations_4control.jsonl` (14016 records)
  - `phase10_17_b_meta.json`
  - `phase10_17_c_candidates_318d.npy` (14016, 250)
  - `phase10_17_c_rank1_eval.json`
- 脚本: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/`
  - `phase10_17_a_anna_embs.py` — AnnaWegmann 768d 提取
  - `phase10_17_b_tinystyler_gen.py` — 4-control TinyStyler 生成
  - `phase10_17_b_smoke.py` — 5-pair smoke
  - `phase10_17_c_rank1_eval.py` — Rank-1 评估
  - `phase10_17_d_summary.md` — 本文档
- 复用组件:
  - `/home/wlia0047/ar57/wenyu/PersoanlQuery/TinyStyler/tinystyler/tinystyler.py` — TinyStyler 类
  - `/home/wlia0047/ar57/wenyu/PersoanlQuery/query_gen/e30/gen_styled_tinystyler_noforce.py` — E30.34 范式参考
  - AnnaWegmann Style-Embedding (HF cache)
  - TinyStyler weights (HF cache)
