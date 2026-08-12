# Iteration #30 — Stage 10 train_vades_lite 死变量/f-string 修复

**日期**: 2026-07-20
**scope**: Stage 10 `train_vades_lite_sentence_latent_threshold.py` 死变量清理
**关联 stage**: 10_complexity_analysis

## §A 冗余代码发现
- 行 787：`f"开始过滤重复 target_reviews 用户"` 无占位符的 f-string
- 行 2044：`f"缓存格式错误，将重新提取句子"` 无占位符的 f-string
- 行 ~1096：`train_indices_tensor = torch.tensor(...)` 死赋值（赋值后未使用）
- 行 ~1202：`num_users = user_mu_all.size(0)` 死赋值（赋值后未使用）
- 注：`GradScaler` 未定义和 `feature_names` 未使用两处 pyflakes 警告为原文件已有问题，非本次引入

## §B 不合理逻辑发现
无新发现。

## §C 可合并文件发现
无。

## §D 本轮已实施改动
- 修复: `10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py`：
  - `f"开始过滤..."` → `"开始过滤..."`（去 f 前缀）
  - `f"缓存格式错误..."` → `"缓存格式错误..."`（去 f 前缀）
  - 删除 `train_indices_tensor` 死赋值
  - 删除 `num_users` 死赋值

## §E 验证
- `python3 -m py_compile` 通过
- `GradScaler`/`feature_names` 警告为原文件既有，保留

## §F Git Commit
- `git commit -m "iter #30: Stage 10 train_vades_lite清理2个f-string前缀+2个死变量"`

## §G 下轮建议
- Stage 10 其他脚本 pyflakes 警告复查
- Stage 05/07 入口 smoke test
