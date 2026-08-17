# E25 Rewrite 质量综合分析 — 总结

## 数据规模

3 个 rewrite 文件共计 **25166 条** 改写：
- e23 scale: 13984
- e23 heldout: 800
- e24 test: 10382

## 关键质量指标

| 维度 | 数值 | 解读 |
|------|------|------|
| **空 rewrite** | 827 (3.3%) | LLM 失败 / 提前截断 |
| **< 3 词太短** | 2192 (9%) | LLM 输出过短（含 prompt 片段） |
| **与原文完全相同** | 47 (0.2%) | LLM 没改写直接返回 |
| **> 100 词超长** | 2 (0.01%) | LLM 输出过长（含原句+改写） |
| **edit distance (norm)** | mean=0.628, median=0.655 | 改写幅度中等 |
| **jaccard** | mean=0.263 | 词级重叠低 |
| **word overlap (recall)** | mean=0.356 | 36% 原句词保留 |
| **len_ratio (rw/src)** | mean=0.87 | 改写略短 |
| **数字保留率** | **36.4%** | ⚠️ **严重问题** |

## 风格清理效果（基本到位）

| 类型 | 源含此词 | 改写全部清理 | 清理率 |
|------|---------|--------------|--------|
| 人称代词 (I/my/we...) | 14666 | 10033 | 68.4% |
| 情感词 (love/amazing...) | 8647 | 7117 | 82.3% |
| 感叹号 (!) | 2325 | 2043 | 87.9% |
| 全大写 (GREAT/LOVE...) | 976 | 703 | 72.0% |

## ⚠️ 数字保留率仅 36%（关键污染源）

2450 条原句含数字，其中只有 891 (36.4%) 改写后数字完全保留，
130 (5.3%) 部分保留，**1299 (58%) 完全丢失数字**。

**对 StyleVector 的影响**：
- delta = hidden_user - hidden_neutral 本应只反映「风格差异」
- 但改写丢失数字时，delta 同时包含「事实差异」
- 这污染了方向信号，可能是 E24 dev→test own-other 崩溃的部分原因

## LLM 失败模式（5 种典型）

1. **空字符串**：LLM 输出截断或 model.generate 未取到 tokens
2. **system prompt 片段泄漏**：
   - "review sentence in a plain, neutral, matter-of-fact style"
   - "rewritten sentence, nothing else"
   - "remove personal tone, slang, exclamations, and emotional words"
3. **原句尾片段截断**：
   - src="...like my little one is very safe" → rw="like my little one is very safe"
4. **无关内容幻觉**：
   - src="We try to only give him the pacifier..." → rw="Illegal Tender" 电影评论
5. **格式泄漏**（e23 scale 长样本特有）：
   - rw="Original: ...\n\nRewritten: \n..." （包含原句+改写元信息）

## 对 StyleVector 的影响

1. **空/太短 rewrite（12.4%）**：cache lookup miss，无 delta 可算
2. **数字丢失（58%）**：delta 含事实差异，方向信号污染
3. **system prompt 泄漏（部分）**：delta 指向 instruction 文本，不是中性句
4. **无关幻觉**：delta 完全没有语义对应

## 改进建议

1. **改写 prompt 强化**：
   - 加 "Output ONLY the rewritten sentence. Do not include any explanation, label, or quotation marks."
   - 加 "If the sentence contains numbers, dates, or measurements, preserve them exactly."

2. **后处理过滤**：
   - 过滤掉 rw 与 system prompt 重叠的（leakage detection）
   - 过滤掉数字从 src 中减少的（除非真的该改）
   - 过滤掉长度 < 50% src 的（truncation）

3. **多候选 + 验证**：
   - 生成 N=3 个候选，选选 jaccard 0.3-0.5 的（保留事实且改了风格）

## 提交

- 脚本: `query_gen/e25_rewrite_quality.py`
- 结果: `result/e25_rewrite_quality.json` + `result/e25_rewrite_quality/SUMMARY.md`