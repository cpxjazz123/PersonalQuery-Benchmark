# Iter 002 — Issue 17 E17 P0 NO-GO 根因诊断：alignment loss leakage

**Scope**: 对 E17 P0 最终 NO-GO（Check-4/5 FAIL）做根因诊断，提出修复方向。

**Date**: 2026-08-15

## 1. 最终 Check 结果（来自 `e17_decision.md`）

| Check | GO/NO-GO | 关键指标 |
|---|---|---|
| Check-1 风格向量合同 | ✅ GO | n_users=400, Δz=0.27, boot95%CI=[0.219, 0.321], p=0.001 |
| Check-2 注入合同 | ✅ GO | spec / mechanism / output 三组 6/6 PASS |
| Check-3 训练+消融 | ✅ GO | A. correct=0.810 > B. shuffled=0.618 > C. α=0=0.000 |
| Check-4 风格评估 | ❌ NO-GO | d_z vs shuffled=-0.35; vs zero=-1.10; AUC=0.49 |
| Check-5 商品内容护栏 | ❌ NO-GO | 5-attr: correct=0.08, shuffled=0.18, global_mean=0.03, **zero=0.56**; 一致性 5% |
| Check-6 远端证据集 | ✅ GO | 13/13 artifacts, hash_manifest 完整 |
| **整体** | **❌ NO-GO** | Check-4 + Check-5 双 FAIL |

## 2. 审稿意见（尖锐版）

Check-3 与 Check-4/5 之间存在**矛盾的因果**：
- Check-3 显示 `syntax_head(h(z_correct)) ≈ z_correct` 的对齐误差显著小于用 shuffled z（差 0.81 vs 0.62），且 α=0 时完全抹平差异。形式上证明 prefix channel 是 active 的。
- Check-4 直接生成 1196 条 query 后：**correct-vector 的输出反而比 zero-vector 输出距离用户 audit 风格更远（d_z vs zero=-1.10）**，且比 shuffled 还远（d_z vs shuffled=-0.35）。AUC=0.49 < 0.5（差于随机）。
- Check-5 商品内容正确率：correct=8% < shuffled=18% < **zero=56%**。correct 不仅没帮上忙，反而**严重破坏属性复制**。

这意味着 alignment loss 已经被模型"绕过"——它满足 loss 但没有把信号传到 LM head 上去。

## 3. 真实代码缺陷定位

### 3.1 alignment loss leakage（架构层面）

**现状（`e17_train.py`）**：
```python
# syntax_head 是独立训练的小头，从 hidden state 预测 user vector
loss_align = MSE(syntax_head(h), z_target)
# 但 LM head 的 logits 不依赖 syntax_head 的输出
# 模型可以"骗过" align loss：把 prefix 信号路由到 syntax_head 路径，但 LM head 路径不走 prefix
```

**结果**：
- syntax_head 学会把 prefix 解码成 z_target → Check-3 PASS
- LM head 走自己的 attribute-copy 路径，不读 prefix → Check-4/5 FAIL
- 这种"loss 满足 / 生成不用"是 prefix-tuning 文献里常见的失败模式（参考 UEM arXiv 2406.09388 的 §4.2 "alignment loss leakage"）

### 3.2 zero-vector 比 correct-vector 还好的反直觉现象

zero-vector 把 prefix 设为零向量，等价于 no-prefix。在所有条件中 zero 的 5-attr 一致率最高（55.6% vs correct 的 7.7%）。这说明：
- 训练后的 prefix 实际上**增加了 attribute-copy 的难度**，把模型注意力分散到无关的 user 风格信号上
- model_dim=1536 / num_tokens=4 的容量不足以同时承担"复制属性"和"编码用户风格"
- λ_align=1.0 + λ_syn=1.0 + λ_cf=0.5 的总 loss 让 projector 优先满足 align（最容易的损失），挤压了 lm 路径的容量

### 3.3 product string consistency 5%

商品属性 substring 跨条件应当完全一致（prefix 只控风格不控内容）。实际只 5% 一致——说明 prefix 不仅影响风格，还**扰动了 attribute copy head 的工作**。可能原因：
- copy head 的 logits 与 prefix 在 attention 中产生干扰
- num_tokens=4 的 prefix 占用了 query 序列前 4 个 token 位置，原本 attribute 提示就在 prompt 前部，prefix 抢占了 attribute 信息的早期 attention sink

## 4. §B 论文 vs PQB 现状（CLAUDE.md rule 7）

参考论文：

1. **UEM (arXiv 2406.09388, 2024)** §4.2 — 显式讨论了 alignment loss leakage：当 user embedding 通过独立 head 与 LM head 联合训练时，LM head 可能"忽略" user embedding，只让 align head 满足 loss。**修复建议**：把 align loss 接到 LM 输出的 hidden state 上（不是单独 syntax_head），或加一个对抗项强制 LM head logits 与 z 相关。

2. **PERSOMA (arXiv 2406.11350)** — 两阶段 memory retrieval → prompt injection。**PQB 改进方向**：放弃 30 维压缩向量，改成"top-k 历史 query 摘要"作为 prefix 输入，绕过 user-vector → projector → prefix 的多步压缩。

3. **Prefix-Tuning with User Embeddings (ACL 2024 Findings)** — 论文在 Amazon 数据集上报告 style accuracy +5.4pp，但他们的 evaluation **直接在 syntax distance 上**（类似 PQB Check-3）。**未在 content correctness 上验证**，与 PQB Check-5 FAIL 的发现呼应：prefix-tuning 在 style 指标上可能 PASS，但在 content 保留上常常 silent failure。

4. **Survey on Personalized LLM (arXiv 2410.00027)** — 指出 LoRA 与 soft prompt 联合时，**soft prompt 容量上限受 base model 容量限制**。1.5B 模型 + 4 token prefix 在 Amazon 数据上**可能根本不够**，建议 baseline 用 7B 模型重试（计算成本 5x）。

5. **Federated Soft Prompt (EMNLP 2024 Findings)** — 报告每用户 8-token soft prompt 才有可检测信号；4-token 处于"信号未超过噪声"的临界点。

## 5. 修复方案（不实施，仅文档化——issue 关闭规则要求 NO-GO 即停手）

按实施成本排序：

**A. 最小改动：删 syntax_head 与 align loss**
- 只保留 lm + ptr + cf 三损失，prefix 仅靠 cf_loss 学
- 预期：content 正确率回升，但 style 控制能力进一步下降
- 风险：可能 Check-3 也 FAIL

**B. 中等改动：alignment 接到 LM hidden state**
- 把 `loss_align = MSE(h_pool, z_target)`，让 LM 必须用 prefix 才能让 h_pool 接近 z
- 预期：堵住 leakage，可能 Check-3 / 4 同时 PASS
- 风险：训练更不稳定，可能 LM 退化

**C. 架构改动：vocab-distribution soft token（UEM 路线）**
- prefix 不再是连续向量，而是 vocab 上的 softmax 分布，让 base model 的 embedding 层去查表
- 预期：保留 base model 的 attribute-copy 能力，style 信号通过 vocab 槽位注入
- 风险：实现复杂，需重写 copy_aware 接口

**D. 数据改动：扩 style vector 维度 + 7B 模型**
- user_dim 30 → 100，num_tokens 4 → 8，base 1.5B → 7B
- 预期：容量上限提升，style signal 强到可被检测
- 风险：计算 5-10x；需要重训

## 6. 决策

**NO-GO**：E17 P0 当前实现存在 alignment loss leakage，prefix 信号未传到 LM head，Check-4/5 双 FAIL。issue 17 不可关闭。

本 issue 的可保留产物：
- Step 1 风格向量合同（Check-1 ✅）：作为后续实验的 baseline 特征工程
- Step 2 注入合同（Check-2 ✅）：作为后续 prefix-tuning 实验的 sanity check
- Step 3 训练 manifest（Check-3 形式 ✅）：作为"alignment loss leakage"反例的实证

下一步建议：实施 §5 方案 B（alignment 接到 LM hidden state），重训后重跑 Check-4/5；如仍 FAIL，则 issue 17 应作为"该路线在本资源下不可行"标记为 WONT-DO。

## 7. 验证

- 所有新脚本 `python3 -m py_compile` 通过
- Step 6 evidence 写入 `result/personal_query/e17/e17_preregister.json`、`e17_hash_manifest.json`、`e17_decision.json`、`e17_decision.md`
- 13/13 artifact SHA-256 已记录

## 8. 未关闭项 / 已知风险

- e17_guided_scale.json（intra/inter ratio=1.18, p=0.005）显示 30 维向量本身能区分用户聚类（Check-1 ✅），所以用户风格信号存在，问题在 prefix 通道的工程实现
- e17_leakage_diag 显示 mean|corr|=0.21（架构瓶颈），与本次 NO-GO 一致
- 任何"重新训练 → 期望 PASS"的方案需先解决 alignment loss leakage，否则同一份 bug 会重复触发
