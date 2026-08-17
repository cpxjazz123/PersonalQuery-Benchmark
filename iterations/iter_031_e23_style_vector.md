# Iter #031 (E23 P0): StyleVector — 100 随机用户评论重写 → hidden activation 差异 → 个人 style vector

**Date**: 2026-08-17
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/23
**Status**: 完成 — 100 用户 × 10 句 × 3 策略 × 7 层 style vector 全部产出；sanity 全部通过（用户 vs 中性激活 cosine≈0.96-0.999 内容受控、风格向量跨用户可区分）

## §A 任务（用户指令 2026-08-17）

应用 StyleVector 论文方法到本项目：随机 100 用户，把他们的评论句重写为 style-agnostic（中性）版本，取两者 Qwen hidden activation 之差作为用户写作风格方向。

## §B 方法（`query_gen/e23_style_vector.py`，硬编码配置，seed=7777）

```
用户历史评论 (Baby_Products_2023.jsonl.gz)
  → 两遍 gz 扫描：计数候选(≥30 评论) → 随机 240 候选 → spaCy 批量切句
  → 100 用户 × 10 句 (5-60 词)
  → Qwen2-7B-Instruct 批量改写（system prompt: 中性、事实保持、去除口语/情绪）
  → 用户句 + 中性句 分别过 Qwen，取层 {8,12,16,20,24,26,27} mean-pooled activation
  → Δ_i = a_user − a_neutral
  → 每用户三种聚合: Mean Difference（主）/ Logistic Regression / PCA 第一主成分
  → 单位化保存 per (user, layer, method)
```

- 重写缓存 `rewrites.jsonl`（按句文本，可断点续跑）
- hidden 缓存 `hidden_cache.npz`（按句文本，可断点续跑）
- 批量解码（rewrite batch 16 / hidden batch 32）、spaCy `nlp.pipe`、单次文件读取
- 与 pacs_layer 长任务共用 GPU（剩余 ~24GB）无冲突

## §C 结果

### C1. 数据规模
- 候选池：≥30 评论用户 2330 → 随机 240 → 合格（≥10 句）237 → 取 100
- 句子：100 用户 × 10 句 = 1000 条（987 唯一文本）；重写成功 987/987（空/短 0）

### C2. 重写示例（用户 → 中性）
| user 句 | 中性改写 |
|---|---|
| This set arrived in a nice reusable storage container with the elastics in plastic packaging. | The set was delivered in a reusable storage container and came with elastic bands packaged in plastic. |
| Some colors had much fewer, like the grey, whereas others had a lot. | The distribution of colors varied, with some, such as grey, having fewer options compared to others. |
| To tie it securely in my daughter's slippery hair, I would have to loop it so many times that it would be a thick band. | The hair tie would need to be looped multiple times to secure it in the slippery hair, resulting in a thick band. |

→ 语义/事实保持，语气中性化（"nice"→无、口语/修辞删除），符合 style-agnostic 定义。

### C3. 每层 sanity 指标（100 用户 × 10 句对）

| layer | user-neutral cosine (内容受控) | Δ norm | style vec norm | 用户间 pairwise cosine | cos<0.9 占比 | cos<0.5 占比 |
|---|---|---|---|---|---|---|
| 8  | 0.9993 | 167.9 | 93.3 | 0.053 | 0.62 | 0.52 |
| 12 | 0.9990 | 175.5 | 97.1 | 0.054 | 0.64 | 0.53 |
| 16 | 0.9988 | 180.4 | 99.6 | 0.058 | 0.64 | 0.52 |
| 20 | 0.9977 | 183.9 | 100.9 | 0.078 | 0.69 | 0.53 |
| 24 | 0.9929 | 196.8 | 105.8 | 0.128 | 0.81 | 0.55 |
| 26 | 0.9883 | 206.4 | 109.1 | 0.167 | 0.89 | 0.56 |
| 27 | 0.9632 | 133.7 | 66.4 | 0.342 | 1.00 | 0.65 |

**解读**：
1. **内容受控**：user-neutral cosine 0.96-0.999 —— 重写只改变表达不改内容，差异主要为风格（呼应论文"同内容 vs 中性版本"前提）
2. **风格信号随层深增**：浅层 cosine 更接近 1（Δ 小、主要是词表面差异），深层（27）差异最大（cosine 0.963）——style 信息集中在高层语义表征
3. **用户可区分**：每用户 style vector 与其他用户 pairwise cosine 均值 0.05-0.34，deep 层 100% 用户对 cos<0.9、65% cos<0.5 —— 100 个用户向量互不塌缩，为后续 steering 提供可辨识方向

## §D 产物（result/e23_style_vector/）

- `style_vectors_100u.npz`：users/sentences/layers/methods + mean_diff/logreg/pca 及其单位化版本 + user/neutral activations (fp16)
- `style_vectors_100u.json`：manifest + 每层 sanity 指标 + 重写示例
- `rewrites.jsonl`、`hidden_cache.npz`：可断点续跑缓存
- 向量尺寸：100 × 7 层 × 3584 维 × 3 策略（单位方向可直接用于 h ← h + α·s 注入）

## §E 文件与提交

- 脚本：`query_gen/e23_style_vector.py`
- 结果：`result/e23_style_vector/*`
- 日志：`result/e23_style_vector.log`

## §F 后续建议（待用户决策）

1. 用 layer26（=STEER_LAYER=-2，与 e22_t3 注入层一致）的 mean_diff 向量做生成注入实验
2. 对比三种策略（mean_diff/logreg/pca）在个性化生成上的效果
3. 单用户存储量：7 层 × 3 策略 × 3584 维 × 4B ≈ 300KB/用户（单位向量）
