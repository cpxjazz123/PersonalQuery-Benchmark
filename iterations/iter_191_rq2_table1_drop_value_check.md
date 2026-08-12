# Iteration 191 — RQ2_Table1_Drop per-cluster value_checks + Δ Range formula audit finding

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: §11 follow-up from iter #190 — 给 RQ2_Table1_Drop (paper Table 1 lower panel) 加 per-cluster value_checks, 同时 audit 发现 paper §3.2 Δ Range 公式与 Table 1 lower panel column 不一致.

## §A 审稿意见（尖锐批评）

### 问题: paper Table 1 lower panel "Δ Range" column 不符合 §3.2 定义的 max-min 公式

**严重程度**: Major (paper-code / paper-text inconsistency — 公式与表格 column 数字对不上)

**Paper §3.2 line 135** (Δ Range 公式定义):
> "Δ Range = max(C1..C8 Hit@10) - min(C1..C8 Hit@10) within each domain"

(公式针对 upper panel correct-query Hit@10; lower panel 错误注入 drop 应该按同样公式: max(|drop|) 或 max - min of drops)

**Paper Table 1 lower panel BM25 Pet** (paper line 128):
- 8 cluster drops: -2.37, -2.58, -5.23, -4.12, -4.17, -1.49, -1.96, -3.33
- Δ Range column 报: -8.22
- 公式 max(drops) - min(drops) = (-1.49) - (-5.23) = **3.74 ≠ 8.22**

**Paper Table 1 lower panel BM25 Baby**:
- 8 cluster drops: -5.71, -3.41, -4.82, -12.12, +0.00, +0.00, +0.00, +0.00
- Δ Range column 报: -8.71
- 公式 max(drops) - min(drops) = 0 - (-12.12) = **12.12 ≠ 8.71**

**Paper Table 1 lower panel SPLADE Grocery**:
- 8 cluster drops: -5.00, -12.38, -11.39, 0.00, -2.17, -8.70, 0.00, +5.88
- Δ Range column 报: -10.21
- 公式 = 5.88 - (-12.38) = **18.26 ≠ 10.21**

**Reviewer concern**: 表格 column 数字与 §3.2 文字公式不一致 — 这是 paper-text-vs-Table-numerical-consistency 问题. Reviewer 会问 "Δ Range 到底怎么算的?"

**Possible alternative formulas** (audit-tested):
- |max - min| (upper panel 公式)? Same as max-min for negatives.
- max(|drop|) - median(|drop|)? 不匹配
- max(|drop|) - mean(|drop|)? 不匹配
- sum of |drops| / count? 不匹配
- mean of drops? 不匹配

**iter #191 目标**:
1. 给 RQ2_Table1_Drop 加 per-(retriever, domain, cluster) value_checks (不依赖 Δ Range column)
2. audit_note 显式 flag Δ Range formula discrepancy 留给 reviewer

## §B 缺陷定位

| 问题 | 文件 | 影响 |
|------|------|----|
| Paper §3.2 Δ Range 公式与 Table 1 lower panel column 数字不一致 | paper §3.2 line 135 vs paper Table 1 line 128 | Reviewer 会问: Δ Range 怎么算的? 公式应该 clarifiy |
| RQ2_Table1_Drop 无 value_check | paper_claims_audit.py L87-99 (before iter #191) | 仅 file-existence verification, 同 iter #190 RQ1_Delta_Range 问题 |
| 公式 missing 在 paper §3.2 文字描述 | paper §3.2 line 135 | 应该 explicitly 说 "negative direction Δ Range" 还是 "absolute effect" 还是 其他 variant |

## §C 本轮代码优化

### C.1 RQ2_Table1_Drop: 加 7 per-cluster value_checks (Baby domain, selected cells)

```python
"value_checks": [
    {"subclaim": "baby_splade_c1_drop",   "selector": "drop.baby.splade.c1",  "expected": -7.62,  ...},
    {"subclaim": "baby_splade_c2_drop",   "selector": "drop.baby.splade.c2",  "expected": -4.55,  ...},
    {"subclaim": "baby_e5_c1_drop",      "selector": "drop.baby.e5.c1",     "expected": -6.67,  ...},
    {"subclaim": "baby_e5_c3_drop",      "selector": "drop.baby.e5.c3",     "expected": -15.66, ...},
    {"subclaim": "baby_ance_c8_drop",    "selector": "drop.baby.ance.c8",   "expected":  10.00, ...},  # 注意 positive drop (Hit@10 增加)
    {"subclaim": "baby_deepseek_c1_drop","selector": "drop.baby.deepseek.c1","expected": -5.56,  ...},
    {"subclaim": "baby_bm25_c1_drop",    "selector": "drop.baby.bm25.c1",   "expected": -5.71,  ...},
],
```

**选择 7 cells rationale**:
- 跨 6 retrievers (SPLADE/E5/ANCE/DeepSeek/BM25 + 重复 SPLADE)
- Baby domain (single-domain verification, 避免 3-D × 8-cluster = 24 value_checks 太长)
- 包含 negative + positive drop (ANCE C8 = +10.00 测试 positive drop handling)
- abs_tolerance 2.0-3.0, rel_tolerance 0.20-0.40 容纳 Stage 7 BPE-aware error injection variance

### C.2 audit_note 显式 flag Δ Range formula discrepancy

```python
"audit_note": "iter #191 added per-(retriever, domain, cluster) drop value_checks (paper Table 1 lower panel Baby domain 7 selected cells). 
Reviewer 真实 verify 7 specific drop values once Stage 7 + Stage 9 re-run 完成. 
**Δ Range formula audit finding**: paper §3.2 line 135 defines 'Δ Range = max(C1..C8) - min(C1..C8)' formula, 
but paper Table 1 lower panel rightmost 'Δ Range' column does NOT match this formula for BM25 
(Baby drops=[-5.71,-3.41,-4.82,-12.12,0,0,0,0]; max-min=12.12 ≠ Table=-8.71; 
Pet drops=[-2.37,-2.58,-5.23,-4.12,-4.17,-1.49,-1.96,-3.33]; max-min=3.74 ≠ Table=-8.22). 
This is a genuine reviewer concern: either Table 1 Δ Range column uses an undocumented formula 
(e.g. mean-of-cluster-drops, max-|drop|-min-|drop|, or sample-weighted variant), 
or paper text formula in §3.2 line 135 is inconsistent with Table 1 column. 
iter #191 audit_note documents this so reviewer can request paper clarification. 
iter #188 paired t-test + iter #190 RQ1_Delta_Range value_checks (upper panel) 不受影响 
(upper panel uses different formula semantics).",
```

## §D 验证

### D.1 Δ Range formula 不匹配实证

Python 验证 (BM25 Baby):
```python
drops = [-5.71,-3.41,-4.82,-12.12,0,0,0,0]
max(drops) - min(drops) = 0 - (-12.12) = 12.12
Table 1 reports Δ = -8.71
|12.12| ≠ |8.71|  → 公式不匹配
```

```python
drops = [-2.37,-2.58,-5.23,-4.12,-4.17,-1.49,-1.96,-3.33]
max(drops) - min(drops) = -1.49 - (-5.23) = 3.74
Table 1 reports Δ = -8.22
|3.74| ≠ |8.22|  → 公式不匹配
```

### D.2 py_compile
```bash
$ python3 -m py_compile paper_claims_audit.py
PY_COMPILE_OK
```

### D.3 audit_note 显式 flag

- paper §3.2 line 135 Δ Range 公式与 Table 1 lower panel column 数字不一致
- 提供 3 retriever × 2 domain examples (BM25 Baby, BM25 Pet, SPLADE Grocery) as concrete evidence
- 留给 reviewer clarification request: 公式到底怎么算?

## §E 后续 iter

- **iter #192**: 给 paper §3.2 line 135 加 footnote 解释 Δ Range 真实公式 (例如 "Δ Range = mean(absolute cluster drop)" 或 document Table 1 lower panel 用 sample-weighted 公式)
- **iter #193**: RQ3_Fleiss/Spearman/MAE re-evaluate 现在 release values vs paper — flip 到 `discrepant`
- **iter #194**: 修复 paper Table 1 lower panel Δ Range column 与 §3.2 line 135 公式一致 (recompute from cluster drops)

## §F Git Commit

- iter #191: RQ2_Table1_Drop value_check enforcement + Δ Range 公式 audit finding — paper_claims_audit.py 加 7 per-cluster drop value_checks (Baby domain) + audit_note 显式 flag paper §3.2 line 135 公式与 Table 1 lower panel column 不一致 (BM25 Baby 12.12≠8.71, BM25 Pet 3.74≠8.22, SPLADE Grocery 18.26≠10.21)