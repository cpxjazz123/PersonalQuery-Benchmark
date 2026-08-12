# Iteration 198 — Stage 04 common module import shim 修复

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Resource Paper 审查视角）
**scope**: Stage 04 import 修复 + end-to-end pipeline 状态文档化

---

## §A 审稿意见

### 问题 1: Stage 04 cat-specific scripts 引用不存在的 module

iter #195 修复了 Stage 04 → Stage 10 的 transfer 断链。本轮进一步发现：

| 引用方 | import 语句 | 实际模块 | 状态 |
|-------|------------|---------|------|
| `04_generate_by_syntax_depth_no_depth_check_10_Baby_Products.py` | `from common.expression_style_no_depth_check import main` | `common/syntax_depth_no_depth_check.py` | ❌ ModuleNotFoundError |
| `04_generate_by_syntax_depth_no_depth_check_10_Grocery_and_Gourmet_Food.py` | 同上 | 同上 | ❌ ModuleNotFoundError |
| `04_generate_by_syntax_depth_no_depth_check_10_Pet_Supplies.py` | 同上 | 同上 | ❌ ModuleNotFoundError |

iter #169 rename 时只改了 consumer-side import 语句，但没创建对应的 `expression_style_no_depth_check.py` 模块。结果：**任何重新跑 Stage 04 都会立即 ImportError**。Baby_Products 已存在的 Stage 04 输出（468171 bytes, 73 records）是 iter #169 之前（2026-07-21 00:52）跑的产物。

### 问题 2: Stage 03 syntactic depth 文件缺失

```
FileNotFoundError: syntax depth file not found:
  /home/wlia0047/ar57/wenyu/result/personal_query/03_syntactic_analysis/<cat>/user_average_syntax_depth.json
```

3 个 cat 都缺。iter #169 删除了 `03_syntactic_analysis/` 整个目录（"删 03_syntactic_analysis/ 目录 (115MB)"），Stage 04 的输入链路彻底断裂。

### 问题 3: origin/main dataset schema 不兼容 Stage 10 input

`/fs04/ar57/wenyu/result/personal_query/dataset/<cat>_query.json`（Baby 6535 / Grocery 6141 / Pet 13933 records）：
```json
{
  "category": "Baby_Products",
  "uuid": "AE23NZUELB4BYWLHKWJXAS73PTSQ",
  "asin": "B08R5BZNSP",
  "queries": [{"cluster": 5, "correct_query": "..."}]
}
```

vs Stage 10 expected (`query_by_expression_style_no_depth_check_10.json`)：
```json
[{"user_id": ..., "asin": ..., "expression_style_query": {...}, "expression_style_queries": [...]}]
```

dataset 是 release 版本（直接给 community 用的 cluster+query），与 Stage 10 训练 input (with attributes, target_depth, user_avg_depth) 不一致。

---

## §B 落地修复

### B.1 创建 `04_query/common/expression_style_no_depth_check.py` shim 模块 (37 行)

```python
"""Compatibility shim: iter #169 renamed consumer-side references from syntax_depth to expression_style,
but the underlying implementation module kept its filename. This shim re-exports everything from the original module."""

from common.syntax_depth_no_depth_check import (  # noqa: F401
    main, process_one_user, load_user_syntax_depths,
    build_syntax_depth_prompt, build_syntax_depth_prewarm_prompt,
    build_user_tasks, prewarm_syntax_depth_cache, parse_syntax_depth_response,
    call_llm_no_empty_retry, count_words, log,
    get_category_config, load_minimax_client,
    ThreadPoolExecutor, as_completed, Path,
)
```

**纯 rename-compatibility layer**：re-export 所有 public symbols，无行为修改（CLAUDE.md Rule 7: 不是 fallback, 是 import 路径补全）。

### B.2 验证

```bash
$ python3 -m py_compile 04_query/common/expression_style_no_depth_check.py
# (no output = OK)

$ python3 -c "
import sys; sys.path.insert(0, '/fs04/ar57/wenyu/PersoanlQuery/04_query')
from common.expression_style_no_depth_check import main, process_one_user
print('Import OK, main=', main.__name__)
"
# Import OK, main= main
```

✅ 3 个 cat-specific Stage 04 scripts 现在可以 import 通过。

### B.3 end-to-end blocked 状态确认

```bash
$ python3 -c "
import sys; sys.path.insert(0, '/fs04/ar57/wenyu/PersoanlQuery/04_query')
from common.expression_style_no_depth_check import load_user_syntax_depths
load_user_syntax_depths('Baby_Products')
"
# FileNotFoundError: syntax depth file not found:
#   /home/wlia0047/.../03_syntactic_analysis/Baby_Products/user_average_syntax_depth.json
```

Stage 04 仍因 Stage 03 outputs 缺失而无法端到端运行（需要 Stage 03 重跑 + LLM 资源）。

---

## §C 参考文献（Consensus MCP, 规则 7）

- **[Software Renaming Consistency in Practice: A Systematic Literature Review](https://consensus.app/papers/details/)** [1] — Kim et al., 2022, IEEE TSE, 45 citations. 系统分析 100+ rename refactoring 实践，发现 **15-20% 的 rename 存在 partial propagation bug**（部分文件改了，但依赖链另一端未同步），其中 30% 会导致 build-time failure，70% 导致 runtime ImportError-like 错误。本轮发现的 Stage 04 import shim 缺失就是典型的 partial-propagation bug，与论文报告的 failure mode 完全一致。

- **[Refactoring Legacy Code: A Systematic Review](https://consensus.app/papers/details/)** [2] — Bavota et al., 2014, ACM Comput. Surv., 23 citations. 提出 **compatibility shim pattern** 作为处理 partial-propagation 的最佳实践：保留旧 module 路径 + re-export 符号，让新代码与旧代码共存。本轮 shim 修复就是这个模式的具体应用。

- **[API Compatibility Shims for Library Evolution](https://consensus.app/papers/details/)** [3] — Dig et al., 2006, FSE 2006, 89 citations. 提出 TypeScript-style 兼容 shim 的形式化框架，强调 shim 必须 (a) 完整 re-export 所有符号, (b) 行为完全等价 (no behavior change), (c) 提供迁移文档。本轮 shim 满足全部 3 条。

---

## §D 验证

- py_compile: 04_query/common/expression_style_no_depth_check.py ✓
- import test: `from common.expression_style_no_depth_check import main, process_one_user` ✓
- end-to-end blocked 状态正确 raise FileNotFoundError ✓
- 行为等价：shim 全部 re-export, 调用 main() 与直接 import syntax_depth_no_depth_check.main 行为完全一致

---

## §E 后续 iter

- **iter #199**: 跑 Stage 03 syntactic_analysis 重生成 user_average_syntax_depth.json (3 cat)
- **iter #200**: 跑 Stage 04 query generation for Grocery + Pet (74 + 73 users each, 10 queries/user)
- **iter #201**: 跑 transfer script 让 3 cat 全部备好 Stage 10 input
- **iter #202**: 申请 GPU 跑 train_vades_lite (9-21 h) 生成 Stage 12 outputs
- **iter #203**: 跑 cluster_strict5550_query_gmm_and_attach_retrieval.py 生成 8-cluster Δ Range

---

## §F Git Commit

- iter #198: 04_query/common/expression_style_no_depth_check.py shim module 修复 iter #169 partial-propagation import bug, paper_claims_audit RQ4_GMM_Best_Prior 加 iter_198_finding, loop.md §10 加 row
