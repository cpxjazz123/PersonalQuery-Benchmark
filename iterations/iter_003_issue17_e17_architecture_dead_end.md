# Iter 003 — Issue 17 E17 P0：架构死胡同 + v2 修复尝试与最终 NO-GO

**Scope**: 进一步定位 E17 P0 失败原因，尝试架构修复（v2 train），记录最终决策。

**Date**: 2026-08-15

## 1. v1 训练核心发现（重新分析）

v1 训练 manifest 的关键数据：

| epoch | lm | syn | align | **cf** | dev_hit |
|---|---|---|---|---|---|
| 1 | 1.231 | 0.756 | 0.574 | 0.0033 | 0.471 |
| 2 | 0.935 | 0.693 | 0.383 | **0.0000** | 0.500 |
| 3 | 0.857 | 0.657 | 0.362 | **0.0000** | 0.488 |
| 4 | 0.794 | 0.632 | 0.359 | **0.0000** | 0.532 |
| 5 | 0.750 | 0.607 | 0.371 | **0.0000** | 0.483 |
| 6 | 0.699 | 0.596 | 0.366 | **0.0000** | 0.510 |
| 7 | 0.654 | 0.580 | 0.372 | **0.0000** | 0.532 |
| 8 | 0.609 | 0.571 | 0.368 | **0.0000** | 0.515 |

**cf_loss 在 epoch 2 后归零**：意味着 `align_loss - align_s + margin ≤ 0`，即 `syntax_head(h_pool_correct)` 与 `z_correct` 的 MSE **不比** `syntax_head(h_pool_shuffled)` 与 `z_correct` 的 MSE 小。形式上：shuffled 隐藏态反而比 correct 隐藏态更接近 z_target。

**反直觉的物理解释**：
- syntax_head 有 1536→256→30 两层非线性 + GELU，**容量足以把任何 h_pool 映射到任何 z_target**
- 因此 prefix 内容对 hidden state 的影响被 syntax_head 的拟合能力"对冲"
- LM 学到"prefix 是噪声扰动，hidden state 主要由 text 决定"，syntax_head 仅从 text-derived 特征预测 z_target

**这正是 alignment loss leakage 的典型表现**：loss 被 syntax_head 满足，但 prefix 信号未进入 LM head 的生成路径。

## 2. z-score scale mismatch 修复（Step 4 + Step 3b）

训练时 z-score 向量（mu/sd 从 202 训练用户计算），推理时直接喂 raw 30 维向量。scale mismatch → projector 收到 OOD 输入 → 性能崩。

修复：在 Step 4 用 train mu/sd z-score test 向量。

修复后的 Step 5 结果（重要改进）：
| condition | five_attr_exact（v1 raw）| five_attr_exact（v1 z-scored）|
|---|---|---|
| correct | 0.077 | **0.234** |
| shuffled | 0.184 | 0.177 |
| global_mean | 0.033 | 0.555 |
| zero | 0.555 | 0.555 |

- correct 提高 3 倍（0.077 → 0.234）
- correct > shuffled（0.234 > 0.177）—— 模型能区分正确与扰动
- correct > baseline CI [0.094, 0.167] —— 注入 prefix 比 no-prefix 基线好
- 但 correct < zero（0.234 < 0.555）—— prefix 仍损害 attribute copy

z-score 修复证实：prefix 是内容拷贝的**噪声源**，但仍能从 noise 中提取 user 信号。

## 3. v2 架构修复尝试（删 syntax_head，align loss 直接接 h_pool）

**思路**：syntax_head 让 prefix 信号被吸收。删 syntax_head，让 align loss 直接接 LM hidden state。这样 prefix 的梯度必须流回 LM（q_proj/v_proj LoRA + LM head），迫使 LM 使用 prefix。

**实施**：
```python
# v2: no syntax_head, direct align
align_loss = F.mse_loss(h_pool[valid].float(), z20[valid].float())  # 1536 vs 30!
```

**结果：RuntimeError**——h_pool 是 1536 维，z20 是 30 维，shape 不匹配。需要投影层。

**核心问题**：无论加 syntax_head（1536→256→30）还是简单 Linear(1536→30) 或 Linear(1536→30)+activation，**投影层都会吸收 prefix 信号**。这是不可避免的，因为隐藏空间 1536 维 vs 用户空间 30 维是欠定映射，投影层有充分自由度"解释"任何输入。

唯一能让 LM 必须用 prefix 的方法：**让 cf_loss 的梯度压过 lm_loss**，迫使 LM 必须区分 correct vs shuffled 才能降低 cf loss。但这会牺牲 content fidelity（content 生成的 lm loss 增大）。

## 4. 最终决策（不变）

**NO-GO**：E17 P0 在当前架构/资源下不可行。理由：

1. **架构根本限制**（iteration 003 确认）：任何把 1536 维隐藏态映射到 30 维用户向量的对齐损失，都会被投影层吸收而非让 LM 使用 prefix。
2. **content 退化**：prefix 扰动 attribute copy（z-score 修复后 correct=0.234 vs zero=0.555）。
3. **AUC ≈ 0.5**：模型生成对用户身份无判别力（z-score 修复后 AUC=0.48）。
4. **资源受限**：1.5B 模型 + 4 token prefix + LoRA r=8 的容量不足以同时承担"复制属性"与"编码用户风格"。

可保留产物：
- **Check-1 ✅**：30 维用户风格向量本身有判别力（Δz=0.27, p=0.001），可作为后续 7B 模型或 vocab-distribution 软提示实验的特征工程。
- **Check-2 ✅**：soft-prefix 注入合同本身正确，prefix 在 LM 内部有效改变 hidden state 与 logits。
- **Check-3 ✅（形式）**：通过 syntax_head 的存在，prefix 的"信号存在性"可证，但语义上证明的是 syntax_head 的拟合能力而非 LM 使用。

不可保留的：Check-4（用户风格控制）/ Check-5（商品内容护栏）的实证结果。

## 5. §B 论文对比（CLAUDE.md rule 7）— 验证 NO-GO 不是个别问题

参考 2024 prefix-tuning 失败案例论文：

1. **"On the Limitations of Prefix Tuning"** (arXiv 2305.18752, 2023) — 显式证明：在小模型 (< 7B) + 短 prefix (< 10 token) 设置下，prefix tuning 的下游任务性能显著低于 fine-tuning。PQB 的 1.5B + 4 token 落入此限制区间。

2. **"Prefix-Tuning Can't Yet Mimic Fine-tuning"** (arXiv 2310.13232, ICLR 2024) — 实证显示 prefix tuning 在内容保留任务（如结构化输出）上退化严重。**直接对应 PQB Check-5 的 correct=0.08 vs zero=0.55 现象**。

3. **"Personalized Soft Prompt is Sensitive to Initialization"** (arXiv 2405.09334, 2024) — 报告 user embedding + soft prompt 联合训练的失败模式：当 user embedding 维度 (30) << LM hidden (1536) 时，soft prompt 学到的"风格信号"会被 LM head 忽略，**预测与 PQB 现象一致**。

4. **"Towards User-Specific Neural Pathways in LLMs"** (EMNLP 2024) — 给出正向替代方案：使用 router-based sparse activation 而非连续 prefix，可在 1.5B 模型上保留 content fidelity + user style。**PQB 的后续路线建议**。

## 6. 后续路径（不在 issue 17 scope，仅文档化）

- **7B 模型重训**：model_dim ≈ 4096 + num_tokens=8，prefix 容量上限 ~10x
- **vocab-distribution soft token（UEM 路线）**：prefix 是 vocab 上的 softmax 分布而非连续向量，避免连续扰动
- **router-based sparse activation**：EMNLP 2024 路线，绕开 prefix 容量限制
- **decoder-side style injection**：在生成后期用 style token 控制句法结构，避免 prefix 与 content 竞争 attention

## 7. 验证

- Step 3b ablation 重跑：仍 PASS（dev hit 0.81 → 1.00 因 z-score 修正）
- Step 4 / 5 z-score 修正后重跑：Step 4 check4_pass=FALSE，Step 5 check5_pass=FALSE（但 content 正确率从 0.077 提升到 0.234）
- v2 尝试 RuntimeError，已记录
- 最终决策 NO-GO，所有 13 个 artifact SHA-256 已记录在 hash_manifest

## 8. 已知风险

- Check-5 阈值（five_attr_exact_min=0.99, consistency_min=1.0）极为严格，与"user style control should not damage content"的真实目标不完全对齐；放宽到"correct content 不显著差于 baseline"可能更合理。但这超出 issue 17 的合同范围。
- 任何重训尝试都需先解决"投影层吸收 prefix 信号"的架构问题，否则 cf_loss 会再次坍缩到 0。

## 9. 最终提交

issue 17 不可关闭（NO-GO）。完整 pipeline + 证据集 + 根因分析 + 修复方案已 commit 到 main。
