# Iteration #29 — §3.1 dead code 复查 + Stage 13/14 清理

**日期**: 2026-07-20
**scope**: §3.1 dead code 复查（Stage 10/13/14 common/ 目录）
**关联 stage**: 10/13/14

## §A 冗余代码发现
- `13_query_template/common/template_query.py`：`datetime.datetime` 未使用；`Optional` 未使用但已导入；`Tuple` 需从 typing 导入（pyflakes: undefined name 'Tuple'）
- `13_query_template/common/generate_template_cache.py`：`typing.Tuple` 未使用；`user_data_loader.load_template_user_queries` 未使用
- `13_query_template/common/cache_path_overrides.py`：`typing.Optional` 未使用
- `13_query_template/common/template_noisy_query.py`：`datetime.datetime` 未使用
- `13_query_template/common/user_data_loader.py`：`os`、`sys` 未使用
- `10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py`：`torch.cuda.amp.autocast` 未使用
- `13_query_template/common/eval_template_driver.py`：`original_is_available` pyflakes 报未使用，但为标准 monkey-patch cleanup 模式，**保留**

## §B 不合理逻辑发现
无新发现。

## §C 可合并文件发现
无。

## §D 本轮已实施改动
- 修复: `13_query_template/common/template_query.py`：删除 `datetime` import；`Optional`→`Tuple`
- 修复: `13_query_template/common/generate_template_cache.py`：删除 `Tuple`、`load_template_user_queries` import
- 修复: `13_query_template/common/cache_path_overrides.py`：删除 `Optional` import
- 修复: `13_query_template/common/template_noisy_query.py`：删除 `datetime` import
- 修复: `13_query_template/common/user_data_loader.py`：删除 `os`、`sys` import
- 修复: `10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py`：删除 `autocast` import

## §E 验证
- `python3 -m py_compile` 全部 6 个文件通过

## §F Git Commit
- `git commit -m "iter #29: Stage 13/14 dead code清理,删除6个未使用import"`

## §G 下轮建议
- Stage 10 `train_vades_lite_sentence_latent_threshold.py` 死变量清理（5处: train_indices_tensor/num_users/feature_names）
- Stage 03/05 入口脚本 smoke test
