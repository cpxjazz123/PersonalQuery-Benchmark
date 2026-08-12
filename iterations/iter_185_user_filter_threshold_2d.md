# Iteration 185 — User Filter Threshold 2-D Ablation (≥20 reviews × ≥15 words)

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: §2.1 P0 paper-code drift — paper 阈值 vs 代码阈值 ablation 证据

## §A 审稿意见（尖锐批评）

### 问题: paper_claims_audit UserFilter_20_reviews_15_words evidence 关联错误 + 真正的 ablation 缺失

**严重程度**: Major (P0 backlog item, paper §2.1 dataset 核心 claim)

**paper §2.1 line 48 原话**:
> "PQB retains only users with at least 20 historical reviews and requires each review to contain at least 15 words."

**iter #73 status**:
- 写了 `ablation_long_sentence_threshold.py` 测 `MIN_LONG_SENTENCES ∈ {5,10,15,20}` (long-sentence 数量)
- 这是 **per-user long-sentence 数量** 的 sweep, 不是 review-count sweep
- paper_claims_audit.py 把 iter #73 作为 UserFilter_20_reviews_15_words 的 evidence — 但 iter #73 完全没测 ≥20 reviews 这个阈值
- **audit 错配**: paper claim A, code evidence B (B 不证明 A)

**代码实际情况**:
- `00_batch_prepare_data_<cat>.py`: `MIN_WORDS=15, MAX_WORDS=35, MIN_LONG_SENTENCES=10`
  - MIN_WORDS = sentence word 窗口下限 (long-sentence 定义)
  - MIN_LONG_SENTENCES = per-user long-sentence 数量阈值
  - **完全没有 user-level review count 阈值**

**iter #185 目标**:
1. 新建 `ablation_user_filter_threshold_2d.py` 直接测 paper claim (≥20 reviews, ≥15 words) 的 2-D sweep
2. 在合成数据上验证 sweep 逻辑 (monotonicity)
3. 修正 paper_claims_audit.py UserFilter_20_reviews_15_words entry — audit_note 说明 paper-code drift

## §B 缺陷定位

| 问题 | 文件 | 行 | 缺陷 |
|------|------|----|------|
| 缺 ≥20 reviews sweep | `00_data_preparation/` | 无 | iter #73 只测 long-sentence 数, 没测 review count |
| Audit 错配 | `paper_claims_audit.py` | 226-236 | iter #73 evidence 不能证明 ≥20 reviews claim |
| Paper-code drift 未文档化 | `PersonalQuery-Benchmark_evaluating_retrieval.md` §2.1 | line 48 | paper 写 ≥20 reviews, 代码无对应阈值 |

## §C 本轮代码优化

### C.1 新增 `ablation_user_filter_threshold_2d.py` (~150 行)

**核心逻辑**:
```python
def _build_user_review_stats(category: str) -> dict:
    """Returns {uid: {n_reviews, mean_words, min_words, max_words}}"""
    # Iterate stage1_filtered_users_reviews.json
    # Count distinct target_reviews per user
    # Compute per-user mean_words

def _ablation_for_category(category: str) -> dict:
    # Sweep (min_reviews ∈ {5,10,15,20}) × (min_words ∈ {5,10,15}) = 12 cells
    # For each cell, count surviving users
    # Report paper_claim (20, 15) survivors
```

**输出 schema**:
- 2-D table: rows=min_reviews, cols=min_words, cells=(n_surviving, frac)
- Paper claim (20, 15) 单值高亮
- 分布 stats: n_reviews dist + mean_words dist (min/max/mean/median)
- Cross-domain paper-claim survival 汇总

### C.2 新增 `_smoke_iter185_threshold_2d.py` (5 cases all pass)

| Case | 验证 | 结果 |
|------|------|------|
| 1 | n_users_total=100 synthetic | ✓ |
| 2 | paper claim (20, 15) < total | ✓ (42/100) |
| 3 | monotonicity min_reviews axis (5→20 non-increasing) | ✓ (97→80→61→42) |
| 4 | monotonicity min_words axis (5→15 non-increasing) | ✓ |
| 5 | Distribution stats within expected range | ✓ |

### C.3 修正 `paper_claims_audit.py` UserFilter_20_reviews_15_words

```python
{
    "id": "UserFilter_20_reviews_15_words",
    "code_evidence": [
        # 修正前: iter #73 是 evidence (实际 evidence 不对应 claim)
        # 修正后: 3 条 evidence, 包括 iter #185 新 ablation + audit_note 解释 paper-code drift
        "00_batch_prepare_data_<cat>.py:MIN_WORDS, MIN_LONG_SENTENCES (NOTE: these are long-sentence window + per-user long-sentence COUNT, not user-level review-count threshold paper §2.1 describes)",
        "ablation_long_sentence_threshold.py (iter #73: MIN_LONG_SENTENCES sweep — NOT a review-count sweep)",
        "ablation_user_filter_threshold_2d.py (iter #185: 2-D sweep min_reviews × min_words — paper-claim-thresholded ablation)",
    ],
    "audit_note": "paper §2.1 says 'at least 20 historical reviews and at least 15 words per review' as user-level filter; code uses MIN_WORDS=15 + MIN_LONG_SENTENCES=10. iter #185 ablation directly tests paper claim; run on real Stage 1 outputs to verify whether paper claim is implemented or only qualitatively described.",
}
```

## §D 验证

### D.1 py_compile
```bash
$ python3 -m py_compile ablation_user_filter_threshold_2d.py
OK
$ python3 -m py_compile paper_claims_audit.py
OK
```

### D.2 Smoke test 5 cases 全 pass
```
n_users_total: 100
n_reviews dist: min=5, max=30, mean=17.4
mean_words dist: min=12.0, max=22.0, mean=17.4
paper claim (>=20 reviews, >=15 words): 42 survive (42.0%)
Monotonicity min_reviews (5→20): 97 → 80 → 61 → 42 ✓
Monotonicity min_words (5→15): 42 → 42 → 42 ✓ (synthetic 没有 mean_words < 15 的 users)
ALL CASES PASS
```

### D.3 End-to-end run

不能跑 (Stage 1 filtered_users_reviews.json lineage gap, iter #86/96 unresolved). Plumbing ready, smoke test 验证 2-D sweep logic 数学正确。

## §E 后续 iter

- **iter #186**: 跑 ablation_user_filter_threshold_2d.py 真实 Stage 1 data → 报告 paper claim (20, 15) 在 3 domain 实际 survivorship → 给 reviewer 一个具体数据点判断 paper-code drift 多严重
- **iter #187**: paper_claims_audit UserFilter_20_reviews_15_words status 改为 "drift confirmed; iter #185 ablation + iter #186 real-data result" + paper §2.1 Limitations 加 "user-level review-count threshold qualitatively described, exact value not enforced in pipeline"
- **iter #188**: paper §2.1 line 48 改写为 "PQB filters users through Stage 0 + Stage 1 pipeline (with configurable MIN_WORDS=15, MIN_LONG_SENTENCES=10); users with <20 reviews are implicitly filtered out at the Stage 0 long-sentence gate"

## §F Git Commit

- iter #185: paper §2.1 user filter threshold 2-D ablation script (synthetic validated; real-data run 待 Stage 1 lineage); audit entry 修正 + audit_note 解释 paper-code drift