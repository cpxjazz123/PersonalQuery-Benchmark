# CLAUDE.md — PersonalQuery 项目约束

## 硬性规则

1. **禁止将可视化（dashboard/HTML/heatmap/sortable-column）作为优化目标**
   - 可视化增强是审计基础设施的 sugar，不是论文贡献
   - 每次迭代必须从论文审稿意见出发，找真实代码缺陷
   - dashboard / HTML / CLI output 样式改进一律跳过

2. **核心目标**：扮演专业 NLP/IR 审稿人（ACL/NeurIPS/ICLR/SIGIR 级别），每次迭代从论文中发现问题，驱动代码优化

3. **每次迭代产出**：
   - 一条审稿意见（尖锐、有据可查）
   - 对应代码缺陷定位
   - 代码修复 + py_compile 验证
   - 写入 `iterations/iter_<NN>_<scope>.md`
   - 更新 `loop.md` 迭代日志

4. **不允许的方向**（即使技术上可行）：
   - dashboard 颜色 / 布局 / 交互改进
   - CLI 输出的格式化 sugar（颜色、unicode 装饰）
   - 新增 `--heatmap-*`、`--sort-*`、`--list-*` 等纯展示性 flag（已存在的可维护）
   - 任何不解决论文实际问题的"基础设施优化"

5. **Retriever index 缓存必须统一**：
   - 所有使用 retriever index 的 stage（Stage 06 / 08 / 12 等）必须共享同一份缓存
   - 缓存路径统一在 `{result}/06_retrieval/retriever_<cat>_cache/`（通过 `config.py` 的 `get_category_config()["retriever_cache_dir"]` 获取）
   - **禁止**在 `08_retrieval/` 或其他 stage 目录下重复构建 retriever index 缓存
   - 新增 stage 必须复用已有 cache路径，不允许新建独立 cache 目录
   - `08_retrieval_config.json` 的 `retriever_cache_dir` 已配置为 `{scratch_result}/06_retrieval/...`，Stage 08 必须沿用此路径

6. **禁止使用 git worktree**：
   - 所有代码修改必须直接在原文件上完成，不允许通过 worktree 隔离后 cat/diff 同步到 main 的方式
   - 修改 → py_compile → 直接 commit 到 main

7. **每次优化必须基于 Consensus MCP 论文搜索进行"论文-现状"对比反思并落地优化**：
   - 每次迭代（定位到具体代码缺陷后），先调用 `mcp__consensus__search` 搜索与当前问题相关的 peer-reviewed 论文（Query 围绕问题本质或具体技术挑战）
   - 搜索结果必须进行**"论文 vs PQB 现状"的对比反思**，回答：论文的发现/方法/结论与 PQB 当前实现有何差距？PQB 的哪些具体做法需要改进？应当如何改进？
   - **不允许仅搜索论文而不落地优化**：搜索完成后必须有实质性的代码或论文修改（新增代码路径/修改已有逻辑/更新 paper 描述/新增 test case 等）
   - 将对比反思结论和优化落地写入 `iterations/iter_<NN>_*.md` 的 §B（参考文献后），格式：**"论文 [N] 发现 → PQB 现状 → 改进方案 → 落地验证"**
   - CLAUDE.md 本条规则在每次迭代时显式执行，不依赖记忆

8. **不需要分析人工评估部分的数据**：
   - §3.3 LLM-human eval (Fleiss κ / Spearman / MAE) 不需要分析
   - 不需要把 paper Fleiss κ=0.72 / Spearman=0.81 / MAE=0.89 与 release 值做对比
   - 不需要针对 agreement_metrics.py / 02_writing_analysis/ 下的 LLM-human eval scripts 做优化
   - 这部分数据是 reviewer pilot 抽样，开放源码里仅复现 Cohen κ/Spearman 在 3×50 subset 上的小规模 pilot，不需要对齐到 paper 的 120-query 全量

9. **禁止使用传入参数运行脚本**：
   - 所有脚本运行参数（数据路径、模型路径、epochs、batch_size、seed、category、out_dir 等）一律**硬编码到脚本内**（模块级常量或脚本内默认值），不允许通过命令行参数（argparse/sys.argv）传参运行
   - 运行方式统一为：`python script.py`（无参数），或 `python script.py --help` 查看硬编码配置
   - 已存在的 argparse 参数可保留（作为配置覆盖后备），但**实际执行必须依赖硬编码默认值**，不允许在运行命令里传参数
   - 分片/并行等场景同样不允许传参：在脚本内用硬编码的 offset/limit 或按数据分片逻辑实现
