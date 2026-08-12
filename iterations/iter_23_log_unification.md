# Iteration #23 — 跨 stage `log` 函数统一

**日期**: 2026-07-20
**scope**: 跨 stage `log` 函数统一到 `common_utils.py`
**关联 stage**: 跨 stage（02/04/05/07/10/12/13/14）

## §A 冗余代码发现
- Stage 05 `common.py`：文件损坏（regex 替换失误），含孤立 log 函数残块（IndentationError）
- Stage 04 `attribute_helpers.py`：本地 `def log` + `from datetime import datetime` 仅服务于 log，可同步删除 import
- Stage 02/07/10/12/13/14 共 11 个 common/ 文件含本地 `def log`，与 `common_utils.log` 重复

## §B 不合理逻辑发现
- Stage 12 `prf_common.py`：`def log` 封装 `_09_log(f"[14] {msg}")`，为 Stage 12 专用前缀设计，**保留**
- Stage 06 `06_fast_fullscale_eval_{Baby,Grocery,Pet}.py`：3 个域脚本含本地 `def log`，iter #22 确认逻辑差异 100+ 处，**保留**

## §C 可合并文件发现
无新发现。

## §D 本轮已实施改动
- 修复: `05_inject_noisy/common/common.py`：删除孤立 log 残块，添加 `from common_utils import log`
- 修复: `04_query/common/attribute_helpers.py`：删除 `def log` + `from datetime import datetime`，添加 `from common_utils import log`
- 修复: `02_writing_analysis/common/extract_errors_common.py`：删除 `def log`，添加 `from common_utils import log`
- 修复: `02_writing_analysis/common/classify_writing_errors_common.py`：删除 `def log`，添加 `from common_utils import log`
- 修复: `07_noisy_retrieval/noisy_syntax_depth_eval_common.py`：删除 `def log`，添加 `from common_utils import log`
- 修复: `10_complexity_analysis/common/cluster_strict5550_query_gmm_and_attach_retrieval.py`：删除 `def log`，添加 `from common_utils import log`
- 修复: `10_complexity_analysis/common/evaluate_review_query_alignment.py`：删除 `def log`，添加 `from common_utils import log`
- 修复: `10_complexity_analysis/common/evaluate_vades_style_vector_probe_svr10fold.py`：删除 `def log`，添加 `from common_utils import log`
- 修复: `10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py`：删除 `def log`（含多行残块），添加 `from common_utils import log`
- 修复: `12_prf/common/prf_syntax_depth_eval_common.py`：删除 `def log`，添加 `from common_utils import log`
- 修复: `13_query_template/common/template_noisy_query.py`：删除 `def log`，添加 `from common_utils import log`
- 修复: `13_query_template/common/eval_template_driver.py`：删除 `def log`，添加 `from common_utils import log`
- 修复: `14_llm_rerank/common/llm_rerank_common.py`：删除 `def log`，添加 `from common_utils import log`

## §E 验证
- `python3 -m py_compile` 全部 13 个修改文件通过
- 已同步到 fs04

## §F Git Commit
- `git commit -m "iter #23: 跨stage log函数统一到common_utils.py,删除11个common文件本地def log"`

## §G 下轮建议
- Stage 06 `fast_fullscale_eval` 三个域脚本逻辑合并评估（iter #22 暂缓，可重审）
- Stage 02/04 入口脚本 smoke test（确认删除了 datetime import 后不影响功能）
