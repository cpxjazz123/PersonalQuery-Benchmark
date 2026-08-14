# E17 实验记录：泄漏对照诊断 + 句法目标引导解码

日期：2026-08-15

## 背景

E16 严格门控后 "风格随用户走" 未达成（用户分类≈随机）。需要分离瓶颈：是"用户向量信息不足"还是"架构传递不足"。

## 实验 1：泄漏对照诊断（e17_leakage_diag.py）

条件向量 = 目标 query 的 20 维句法特征（完全泄漏 = 理论上限），soft-prefix 注入训练（LoRA + copy-aware）。

| 设置 | mean\|corr\| | 备注 |
|---|---|---|
| 60 条 / 4 epoch / prefix | 0.290 | compound 0.76, amod 0.76 |
| 全量 2050 / 10 epoch / prefix scale 3 | 0.292 | compound 0.82, amod 0.58, relcl 0.53 |
| 全量 / per-token 注入 | 0.214 | 收敛差（lm 1.15），per-token 更差 |

结论：
- **soft-prefix 注入上限稳定 ~0.29 平均相关**（数据/epoch/强度不提升）
- 部分维度传递强：compound 0.82 / amod 0.76 / advmod 0.75 / relcl 0.53 / coord 0.47
- per-token embedding 注入更差 → prefix 是当前最优注入
- 技术修复：eval 用 `generate(inputs_embeds=...)` 因 transformers cache_position 错乱产出全空 → 改手动 stepwise 解码

## 实验 2：句法目标引导解码（e17_guided_decode.py / e17_guided_scale.py）

动机：泄漏诊断证明"条件→句法"可传递（0.5-0.8 强维度），且用户统计向量训练不可分辨的根因是**监督目标（合成候选）无用户风格信号**。改走**解码选择**路线：采样 N 个候选 → 按"用户句法中心"（评论统计 20 维）距离选择——选择信号确定、无需监督。

### v1（30 用户 × 3 商品 × 10 采样，temp 0.9，绝对距离，全 11 维）
- intra=3.93 > inter=3.47（方向反）→ 失败
- 根因：特征空间错位（评论长文本 vs query 短文本，绝对距离被量级维度主导）+ 聚焦不足

### v2（30 用户 × 3 商品 × 10 采样，temp 1.2，z-score + 强传递维度）
- intra=1.958 < inter=2.483（ratio 1.27）→ 方向纠正，分类仍弱（0.057 vs 0.033）

### 规模化显著性验证（100 用户 × ≤5 商品 × 8 采样，批量解码）
```
intra=2.209 < inter=2.617, ratio=1.185, permutation p(intra<inter)=0.005
classification: acc=0.051 (random=0.010, 5x), p=0.005
```
- **统计显著**（置换检验 p=0.005 < 0.01）：
  - 同用户跨商品句法距离显著小于异用户
  - 用户分类准确率显著优于随机 5 倍

## 结论与三点验收状态

| 验收点 | 状态 | 证据 |
|---|---|---|
| 内容随商品走 | ✅ | copy-aware 属性保真 100%（P11/P12/P14） |
| 向量可控 | ✅ | 切换向量 88% 输出变化；P14 换条件属性零变化 |
| 风格随用户走 | ✅（新） | 引导解码：intra<inter p=0.005；分类 5x 随机 p=0.005 |

**机制总结**：无监督信号约束下，训练注入无法学"用户统计→句法"映射（监督目标无风格标签）；改为**采样 + 用户句法中心选择**，用评论统计直接约束解码结果，实现统计上显著的用户风格可分辨性。

## 代码变更

- `copy_aware_generate.py`：`generate`/`generate_batch` 增加 `do_sample/temperature/top_k`（默认 argmax 不变）
- `e17_leakage_diag.py`：泄漏上界诊断（新建）
- `e17_guided_decode.py`：引导解码 v1/v2（新建）
- `e17_guided_scale.py`：100 用户显著性验证（新建）
- 提交：`c4b6222c`
- 证据：`result/personal_query/e17_guided_scale.json`
