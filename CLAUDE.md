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

10. **解码必须使用批量解码（加快速度）**：
   - 所有 LLM 生成/解码（训练验证、推理、干预实验等）一律使用批量解码（同一 batch 内多条样本共享 forward pass），不允许逐条顺序生成
   - 参考实现：`copy_aware_generate.py` 的 `generate_batch`（batch 16，~18x 提速）、`e15_validate_h1.py` 的批量 `model.generate(batch_ids)`
   - 批量场景注意事项：
     - right-padding 时首步预测必须取每行**最后有效位置**的 logits（`attention_mask.sum(dim=1)-1`），不能取 `logits[:,-1]`（会预测到 padding 位置）
     - 后续步（KV cache 模式）logits 只有新 token 位置，用 `logits[:,-1]`
     - batch 内各样本独立 prompt/属性 span/hook 方向时，逐样本构建 mask/position/span 后整批 forward
   - 已实现单条生成的脚本（如无批量版本）必须改造或注明 TODO，不得新增逐条生成逻辑

11. **已实现的提速方法（实现相关业务时必须使用）**：

   **a. 批量解码（~18x）**：LLM 生成/解码一律批量（规则 10）；参考 `copy_aware_generate.py::generate_batch`（batch 16）与 `e15_validate_h1.py`（批量 `model.generate(batch_ids)`）。注意 right-padding 首步取 `attention_mask.sum(dim=1)-1` 位置 logits。

   **b. 向量化张量操作（1.7x 训练）**：避免 Python 逐元素循环（如 `for b in range(B): for t in range(T): index_add_`）；用批量 scatter_add_（参考 `copy_aware.py` 的 `[B*T,S]→[B*T,V]` 单次 scatter）。新增模块不得写逐样本/逐 token 的 Python 循环实现张量逻辑。

   **c. spaCy/特征提取批处理（108x）**：
      - 一律用 `nlp.pipe(texts, batch_size=...)` 批量解析，禁止逐条 `nlp(text)`
      - 评论/文本先切句再提取句子级特征（段落级特征慢且语义错）
      - 大文本源**只读一次**构建内存索引，禁止每用户/每条重新 `json.load` 大文件
      - 参考 `user_stat_vector.py`（单次 pipe + 单次文件读取 + 切句）

   **d. 中间结果缓存**：确定性计算结果（统计向量、特征、tokenization、activations）必须写文件缓存并在二次调用时直接加载，禁止重复计算；缓存需支持**合并更新**（部分运行不得覆盖整体缓存）。参考 `user_stat_vector.py::CACHE_PATH`（900 用户 150s→0.01s）。缓存元数据（dim/version）必须与调用方校验一致，不一致时 fail-fast 而非静默重算。

   **e. 模型规模选择**：优先用能满足任务的最小编号模型（7B→1.5B 样本吞吐 3.7x）；大规模生成时优先批量化而非多进程并行（GPU 饱和时多进程无增益）。

   **f. 训练数据预编码**：Dataset 的 `__getitem__` 不得每次重复 tokenize/编码——一次性预编码（ids/masks/spans）缓存后复用；多进程数据加载（num_workers>0）在数据预处理重时启用。

   **g. vLLM 推理引擎（高频生成 3-5x）**：大规模/高频 LLM 生成（批量、采样、评估、干预实验的纯 LM 生成路径）优先使用 vLLM 服务（PagedAttention + continuous batching），吞吐提升 3-5x；本环境已有 `pypkks_vllm` 目录与使用经验。注意：
      - vLLM 只适用于**标准 generate 路径**（无自定义 hook/copy head 干预的纯 LM 生成）；带自定义层 hook、copy head 或中间激活注入的生成必须走原生批量解码（规则 10/11a），不得硬套 vLLM
      - 使用 vLLM 时必须保持解码配置一致（temperature/top-p/penalty/max_tokens 与预注册或对照配置逐字段一致），禁止不同引擎引入解码差异
      - vLLM 服务启动/关闭纳入脚本生命周期；小批量（<100 条）或单次生成直接用原生批量解码即可，引擎开销不值得

   这些方法均为已验证的实测提速；新实现/改造相关业务时必须选用适用的方法，禁止回退到逐条生成、逐元素循环、重复加载大文件等低效模式。

12. **任务完成必须 commit + comment + push + close 四步闭环**：
   - 每个实验/任务完成时必须严格执行以下四步，缺一不可：
     1. **commit**：`git add` 所有相关文件（脚本 + iter 文档 + result JSON/log，result/ 在 .gitignore 用 `git add -f`），用 `git diff --cached --stat` 验证文件齐全后再 commit
     2. **comment**：用 `glab issue note <id>` post 进度/勘误/最终 note 到对应 GitLab issue；至少含设计、量化结果、文件位置、提交号四要素
     3. **push**：`git push` 到 main（已 commit 但未 push 视为未完成）
     4. **close**：实验有明确 GO/NO-GO 结论后用 `glab issue close <id>` 关闭 issue；如果结论仍待用户决策，先 post note 等回复，不要擅自关闭
   - 任何一步遗漏都会破坏可复现性或决策链（如 v3 漏 push result JSON 导致误判、漏 close 导致 issue 状态错乱）
   - **不允许**"代码写完 + 文档写完"就报告完成，必须四步全做完
   - **不允许**在用户未明确同意前关闭 issue（即使是 NO-GO 也先 post note 等用户确认）
