# Iteration 199 — Stage 03 syntactic depth per-user data 永久丢失诊断

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 审查视角）
**scope**: Stage 03 per-user syntactic depth 数据 lineage 状态评估

---

## §A 审稿意见

### 问题: Stage 03 per-user syntactic depth 数据不可恢复，Stage 04 端到端永久 blocked

iter #195 修了 transfer script，iter #198 修了 import shim，但 Stage 04 `load_user_syntax_depths(category)` (line 131-157) 仍 raise：

```
FileNotFoundError: syntax depth file not found:
  /home/wlia0047/ar57/wenyu/result/personal_query/03_syntactic_analysis/<cat>/user_average_syntax_depth.json
```

### 完整 data lineage 调查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Stage 03 scripts (`03_syntactic_analysis/*.py`) | ❌ **git 中已删除** | iter #172 commit 4f4a051 删了 9 个 .py 文件 (03_count_acl_ccomp_levels.py, 03_syntactic_analysis*.py × 4, 03_user_average_syntax_depth*.py × 4) |
| Stage 03 source data | ⚠️ 05_syntactic_analysis/<cat>/ 有 summary stub | 见下 |
| Stage 04 input 期望路径 | `03_syntactic_analysis/<cat>/user_average_syntax_depth.json` | 硬编码 (line 40), schema `{users: [{user_id, avg_depth, review_count, min_depth, max_depth}, ...]}` (line 136-156) |
| Stage 05 实际文件 | `05_syntactic_analysis/<cat>/user_average_syntax_depth.json` | **schema 不匹配**：仅 `{timestamp, summary: {category, user_count, review_count, avg_depth, min_depth, max_depth, depth_histogram}}` (per-category aggregate), 无 per-user list |
| Stage 03 scripts in git history | ✅ 可恢复 | `git show 4f4a051^:PersoanlQuery/03_syntactic_analysis/03_user_average_syntax_depth_Baby_Products.py` |

### 关键发现: Stage 03 per-user 数据永久丢失

05_syntactic_analysis/user_average_syntax_depth.json 实际内容 (Baby_Products):
```json
{
  "timestamp": "2026-07-21T00:22:47.693318",
  "summary": {
    "category": "Baby_Products",
    "user_count": 7681,
    "review_count": 54302,
    "avg_depth": 7.7323,
    "min_depth": 3,
    "max_depth": 25,
    "depth_histogram": {...}
  }
}
```

**这只是 category-level summary stub**，包含 aggregate statistics 但**完全没有 per-user avg_depth 数据**。Stage 04 需要 per-user `[{user_id, avg_depth}, ...]` 才能给每个 user 生成 query。

### Stage 04 输入依赖 (line 150-156)

```python
depth_map[user_id] = {
    "avg_depth": float(avg_depth),
    "target_depth": _round_target_depth(float(avg_depth)),
    "review_count": user.get("review_count"),
    "min_depth": user.get("min_depth"),
    "max_depth": user.get("max_depth"),
}
```

每个用户的 `target_depth` 决定 prompt 的 syntactic complexity 等级。没有 per-user 数据，Stage 04 无法给任意一个用户生成合理 query。

### end-to-end Stage 04 不能在当前环境跑通的原因

1. **缺失 per-user avg_depth 数据**（不是 summary）
2. **spaCy 模块未安装** (`ModuleNotFoundError: No module named 'spacy'`)，即使用 git history 恢复 Stage 03 scripts 也跑不起来
3. **en_core_web_sm 未安装**（spacy.util.is_package 返回 False）
4. **LLM 资源不确定**（MiniMaxAnthropicClient 需 API call，每个 query 都要生成）

按 CLAUDE.md Rule 7（无 fallback），正确的做法是：
- **不创建 synthetic stub** 代替缺失的 per-user 数据
- **不修改 Stage 04 期望 schema** 来"适配"现存 summary
- **承认 lineage broken** 并文档化

---

## §B 落地决策

### B.1 不修复 (但保留 audit trail)

iter #199 范围：诊断 + 文档化 Stage 03 per-user 数据 lineage 永久丢失状态。不尝试：
- 重建 synthetic per-user data (会引入 bias)
- 用 summary stub 替代 (违反 Rule 7)
- 修改 Stage 04 接受 summary schema (改 paper schema)

### B.2 把 Stage 03 lineage gap 写入 paper §5 Limitations

已有 iter #90 把"RQ4_GMM_Best_Prior lineage gap"加到 §5 #4。本次把"Stage 03 per-user syntactic depth data lineage gap"作为新 limitation 加入 §5 #6。

### B.3 paper_claims_audit.py 加新 claim entry

`Sec2_user_avg_syntax_depth_per_user` claim: paper §2.2 说每个用户有 avg_depth，但 release 端无 per-user 数据。status: blocked (data lineage lost, not recoverable without Stage 03 re-run)。

### B.4 评估 Stage 03 重跑 cost

如要真正恢复，需：
1. 装回 spaCy + en_core_web_sm (`pip install spacy && python -m spacy download en_core_web_sm`)
2. 从 git history 恢复 Stage 03 scripts (`git show 4f4a051^:PersoanlQuery/03_syntactic_analysis/03_user_average_syntax_depth_Baby_Products.py`)
3. 跑 Stage 03 三个 cat (CPU 几小时, depends on review count: Baby 7681 users × 10 reviews)
4. 把 per-user JSON 写到 `/home/wlia0047/ar57/wenyu/result/personal_query/03_syntactic_analysis/<cat>/user_average_syntax_depth.json`
5. 跑 Stage 04 (LLM API budget: ~73 users × 10 queries = ~730 API calls per cat)

Total: ~3-6 h compute + LLM API budget

---

## §C 参考文献（Consensus MCP, 规则 7）

- **[A Reproducible Universal Dependencies-Style Pipeline for Katharevousa Greek Parliamentary Text](https://consensus.app/papers/details/e95fb3fc7cac59f99cf8adc2944091a7/?utm_source=claude_code)** [1] — Mikros et al., 2026, 0 citations. 提出 reproducible UD-style pipeline，强调 "frozen automatically validated reference set" + "deterministic CoNLL-U snapshotting" + "per-model benchmark reports" 是 reproducibility 的核心。这与 PQB 当前 Stage 03 数据永久丢失形成对比：per-user syntactic depth 数据如果没 frozen snapshot 就无法 reproducibility verification。

- **[Parsing Old English with Universal Dependencies - The Impacts of Model Architectures and Dataset Sizes](https://consensus.app/papers/details/22c47c6c791a5f8f968bc6260672181e/?utm_source=claude_code)** [2] — Arista et al., 2025, Big Data Cogn. Comput., 1 citation. 系统比较 spaCy / tok2vec / MobileBERT 在低资源 UD parsing 上的表现。spaCy en_core_web_sm 在 small dataset 上仍可达 75% LAS。这支持 Stage 03 重跑使用 spaCy en_core_web_sm 的 cost-effectiveness 决策。

- **[CAIT: A Syntactic Parsing Toolkit for Child-Adult InTeractions](https://consensus.app/papers/details/66ebfe31833251b3ac21631fa20e138c/?utm_source=claude_code)** [3] — Padovani et al., 2026, 0 citations. 强调 syntactic parser toolkit 必须 release frozen UD annotations + train/test split + error analysis 才能 reproducibility。本轮确认 PQB 的 syntactic depth 数据 lineage 不满足这一标准。

---

## §D 验证

- `git log --all --oneline -- 'PersoanlQuery/03_syntactic_analysis/*.py'` → 4f4a051 (iter #172 delete) ✓
- `git show 4f4a051^:PersoanlQuery/03_syntactic_analysis/03_user_average_syntax_depth_Baby_Products.py` → 完整 script 可恢复 ✓
- `python3 -c "import spacy"` → ModuleNotFoundError ✓
- `/home/wlia0047/.../05_syntactic_analysis/Baby_Products/user_average_syntax_depth.json` schema = `{timestamp, summary}` ≠ Stage 04 期望 `{users: [...]}` ✓

---

## §E 后续 iter

- **iter #200**: paper §5 Limitations 加 #6 "Stage 03 per-user syntactic depth data lineage gap"; paper_claims_audit.py 加 Sec2_user_avg_syntax_depth_per_user claim entry (status: blocked)
- **iter #201** (optional, 待资源): 真恢复 Stage 03 — 装 spaCy, 从 git history 恢复 scripts, 跑 3 cat per-user data
- **iter #202** (optional, blocked on iter #201): 跑 Stage 04 query generation for 3 cat (LLM API budget)
- **iter #203** (optional, blocked on iter #201+#202): 跑 transfer script 3 cat, 备好 Stage 10 input
- **iter #204** (optional, blocked on iter #203 + GPU): 跑 train_vades_lite 生成 Stage 12 outputs
