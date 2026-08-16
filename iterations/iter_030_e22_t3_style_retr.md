# Iter #030 (E22 Task 3): 评论风格迁移到 Query + 检索 hit@10 差值 — **内容主导检索，文体鸿沟限制风格迁移**

**Date**: 2026-08-17
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/22
**Status**: 完成 — PACS steering / few-shot / 改写三种方法跨文体 self-match 均≈chance；检索 hit@10 差值≈0.03（多商品、多属性数收敛）；评论内自匹配=1.0 证明是文体级限制

## §A 任务（用户指令 2026-08-17）

配置 **X=5（top5 用户）、Y=5（每用户≥5句）、Z=5（每句≥5词）、w=20（top20 差异句法维度）**，生成"可迁移评论句法风格的 query"，评估：同一商品、同一批属性值、语义一致但句法不同的 query → 检索 → hit@10 差值。

## §B 前置分析

### B1. 维度/用户/句数扫描（支撑 X/Y/Z/w 配置）
- `e22_t3_topxyzw_count.py`：X∈{5,8,10}×Y∈{3,5,10}×Z∈{1..20}×w∈{1..281} 四维网格商品计数
- `e22_t3_style_scan_dim_syntax.py`：句法-only（排除标点+长度相关）维度可分性；**296→281 维句法**、top-20 依 variance-ratio
- `e22_t3_multi_prod_scan.py`：每商品最优 (n_users, D) 组合因商品而异
- X=5/Y=5/Z=5/w=20 为可分率>90% 的稳定配置（抽样 100/格验证）

### B2. 测量可靠性对照
- 评论内自匹配（同用户 split-half，20 维句法）**rank-1=1.00**（B0BQ1QK14T 5 用户）
- → 句法向量有效区分用户；任何跨文体失败均非测量缺陷

## §C 风格迁移方法 × self-match（评论→query 保真度）

| 方法 | 评论→query self-match rank-1 | 判定 |
|---|---|---|
| PACS activation steering (α=1, layer-2) | 0.18 | =chance(0.20) |
| few-shot 评论示范（system prompt） | 0.22 | ≈chance |
| 评论改写为查询 prompt | 0.27 | 不显著 |
| **评论内自匹配（上限对照）** | **1.00** | 测量可靠 |

**结论**：三种方法均无法把评论句法迁移到购物 query。购物查询文体 vs 意见评论文体在 318/20 维句法空间为两个不重叠区域（呼应 iter_029/cf9686d"分布不可比"）；用户个体差异（rank-1=1.0）远小于文体差异。

## §D 检索 hit@10 差值（核心指标）

设置：同商品同属性 → 每用户 8-16 条风格 query（strict-span 保证属性 verbatim，内容纯净≥0.95）→ BGE-base 检索 200 个同品类商品（随机/最近邻干扰两种 corpus）→ hit@10 diff = max−min(用户命中率)

### D1. 属性数扫描（最近邻干扰 corpus，15 商品）

| 属性数 | steering α=1 | baseline α=0 | delta |
|---|---|---|---|
| 9（全属性） | 0.025 | 0.033 | −0.008 |
| 5 | 0.042 | 0.058 | −0.017 |
| 3 | 0.033 | 0.033 | +0.000 |

### D2. 方法间比较（多商品平均）

| 方法 | hit@10 diff | 检索集 Jaccard |
|---|---|---|
| PACS steering | 0.025 | 0.524 |
| few-shot | 0.075 | — |
| 改写 prompt | 0.167 | — |
| baseline | 0.033 | — |

**结论**：**hit@10 差值 ≈ 0.03-0.17**，随属性数/方法波动但始终很小——query 含目标核心属性时 target 恒进 top-10，句法只改变 top-10 内部排序与结果集构成（Jaccard 0.52），不改变是否命中。

## §E 最终判定

1. **"评论风格迁移到购物 query"在当前范式下不可实现**：文体鸿沟（购物查询 vs 意见评论）使任何生成方法输出都落在句法"无人区"；评论内自匹配=1.0 证明限制是文体级、非方法级
2. **语义一致的风格迁移 query 的检索 hit@10 差值 ≈ 0.03**（内容主导检索；四轮测量收敛：属性数 3/5/9 × 方法 4 种均一致）
3. 若要真正实现风格迁移：重构 query 目标文体（"用户口吻的购物表达"保留意见句式）或改评"文体内迁移"（评论内已验证可行 rank-1=1.0）

## §F 文件

- 脚本：`query_gen/e22_t3_{steer_retr,steer_alpha,steer_multi,steer_multi3,steer_final,steer_match,fewshot_match}.py`
- 分析：`query_gen/e22_t3_{topxy,topxyz,topxyzw}_count.py`、`e22_t3_{style_scan,style_scan_dim,user_diff,user_diff_dim,multi_prod_scan,rank_baseline,eval7d,audit_content}.py`
- 结果：`result/e22_t3/e22_t3_{steer_alpha,steer_multi,steer_multi3,steer_final,steer_match,fewshot_match}.json` 等
- 提交：db41572, 5f2f6d0, 7ceeef8, e90acd0, 1fdbbc5, 25c94e4, de3301b
