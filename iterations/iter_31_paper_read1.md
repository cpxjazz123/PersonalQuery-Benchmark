# Iteration #31 — 论文第1次通读：§1 Introduction + §2.1 任务定义

**日期**: 2026-07-20
**scope**: 论文第1次通读：§1 Introduction + §2.1 任务定义
**关联 stage**: 跨 stage

## §A 论文发现

### §1 Introduction 关键结论
- **Personalized expression differences 两个维度**：
  1. **Syntactic expression structure**：用户组织 product attributes/modifiers 的方式（noun phrase vs clause extension）
  2. **Writing errors**：用户在历史 review 中的真实拼写/输入错误（如 "lactose-free" → "lactsoe-free"）
- **核心问题**：现有 benchmark 无法评估"同一需求不同表达方式"下 retriever 的稳定性
- **数据集规模**：26,609 queries，覆盖 Baby/Grocery/Pet 三个 domain
- **用户筛选标准**：≥20 条历史 reviews，每条 ≥15 词

### §2.1 数据源与 pipeline 概览
- **数据来源**：Amazon reviews + product metadata
- **Pipeline 阶段**：
  1. 用户历史 review → 提取 expression pattern + writing error pattern
  2. LLM 生成含 5 个 attribute value 的 candidate queries
  3. Inject writing errors → error-injected query
  4. GMM filtering → style-aligned queries
  5. GMM clustering → expression style clusters
  6. Retriever evaluation

## §B 代码问题

- **Stage 02（writing error 提取）**：代码是否完整实现了从用户历史 review 提取真实 writing error pattern 的逻辑？需要对照检查。
- **Stage 03（syntactic analysis）**：论文用 spaCy dependency parser 提取 20 维 syntactic style features，代码是否实现了这 20 维？
- **Stage 04（syntax depth query generation）**：是否生成了包含不同 syntactic structure 的 queries（如 noun phrase vs clause extension）？
- **用户筛选标准**：代码是否实现了 ≥20 reviews、每条 ≥15 词的过滤？

## §C 本轮已实施改动

- 重写 `loop.md`：从"代码去重重构"切换为"论文驱动代码优化"
- 更新 §2 执行流程、§3 检查清单、§5 论文分 6 次读完计划
- loop.md §9/§10/§11 全部更新

## §D 验证

- `python3 -m py_compile` N/A（仅修改 loop.md）
- 论文已转换为 markdown（215行，17个标题，11个表格）

## §E Git Commit

- `git commit -m "iter #31: 重写loop.md为论文驱动方向，启动论文第1次通读"`

## §F 下轮建议

- 论文第2次：§2.2 GMM/VAE 训练目标（Eq.1-8）——对照 Stage 10 `train_vades_lite_sentence_latent_threshold.py` 检查 loss 实现完整性
- 检查 Stage 02 writing error extraction 是否覆盖了 3 个 domain 的真实 error pattern
