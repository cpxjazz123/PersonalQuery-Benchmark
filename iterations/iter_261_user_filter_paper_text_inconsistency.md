# Iteration 261 — Paper §2.1 User Filter Claim：Paper 内部文字不一致 + Code 实现与 Paper 描述的差异

**日期**: 2026-07-22
**角色**: NLP/IR 专业审稿人
**scope**: Paper §2.1 line 48 描述的 user filtering criteria 与 paper_claims_audit.json 中 claim 描述不一致；code 实际用 MIN_LONG_SENTENCES=10 而非"≥20 reviews"

---

## §A 审稿意见

### 问题：Paper §2.1 line 48 描述 vs claim 描述 vs code 实现 三者不一致

**严重程度**: Major
**论文位置**: §2.1 line 48 + paper_claims_audit.json `UserFilter_20_reviews_15_words`

#### 1. Paper §2.1 line 48 原文：

> "To ensure the reliability of user profiles, PQB retains only users with **at least 10 long sentences per user (where each long sentence contains 15–35 words)**, based on the MIN_LONG_SENTENCES=10 threshold in the released code."

#### 2. Paper §2.1 claim text（来自 audit.json）：

> "PQB retains users with **≥20 reviews** and ≥15 words per review."

#### 3. Code 实际（`00_batch_prepare_data_Baby_Products.py:200-202`）：

```
MIN_WORDS = 15            # 每句话最少词数
MAX_WORDS = 35            # 每句话最多词数
MIN_LONG_SENTENCES = 10   # 每个用户最少长句子数
```

### 不一致分析

| 维度 | Paper §2.1 line 48 | Audit claim text | Code 实现 |
|------|---------------------|------------------|-----------|
| 用户筛选标准 | ≥10 long sentences (15-35 words each) | ≥20 reviews + ≥15 words/review | ≥10 long sentences (15-35 words) |
| 阈值来源 | MIN_LONG_SENTENCES=10 | ≥20 reviews | MIN_LONG_SENTENCES=10 |
| 句子词数窗口 | 15-35 words | ≥15 words/review | 15-35 words |

**三处不一致**：
1. Paper §2.1 line 48：10 long sentences (correct)
2. Audit claim text：≥20 reviews (wrong — conflates review count with long-sentence count)
3. Code：≥10 long sentences, 15-35 words (correct — matches line 48)

### 核心问题

Paper §2.1 line 48 的描述是**正确的**（10 long sentences, 15-35 words each），且与 code 实现一致。但 audit claim 描述为"≥20 reviews"——**这是 paper 内部错误**。

iter #73 的 ablation 已经证明：
- MIN_LONG_SENTENCES=10 是 Pareto-optimal threshold
- 10 vs 5 threshold 仅损失 2-6% overlap users
- 15/20 threshold 损失 24-58% users

### 对应代码缺陷

无代码缺陷——code 实现与 paper line 48 描述一致。问题是 **paper claim 描述与 paper line 48 描述不一致**。

---

## §B 下一步

**代码修改**: 无（code 实现正确）

**Paper 修改**: audit.json 中 `UserFilter_20_reviews_15_words` 的 claim text 应更正为与 paper §2.1 line 48 一致：
- 当前（错误）："PQB retains users with ≥20 reviews and ≥15 words per review."
- 应改为："PQB retains users with at least 10 long sentences per user (where each long sentence contains 15–35 words), based on MIN_LONG_SENTENCES=10."

**验证**: iter #73 ablation 结果已支持此 threshold 的合理性。
