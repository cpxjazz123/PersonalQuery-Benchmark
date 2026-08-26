# CLAUDE.md — PersonalQuery 项目约束

## 硬性规则

1. **Retriever index 缓存必须统一**：
   - 所有使用 retriever index 的 stage（Stage 06 / 08 / 12 等）必须共享同一份缓存
   - 缓存路径统一在 `{result}/06_retrieval/retriever_<cat>_cache/`（通过 `config.py` 的 `get_category_config()["retriever_cache_dir"]` 获取）
   - **禁止**在 `08_retrieval/` 或其他 stage 目录下重复构建 retriever index 缓存
   - 新增 stage 必须复用已有 cache路径，不允许新建独立 cache 目录
   - `08_retrieval_config.json` 的 `retriever_cache_dir` 已配置为 `{scratch_result}/06_retrieval/...`，Stage 08 必须沿用此路径

2. **禁止使用 git worktree**：
   - 所有代码修改必须直接在原文件上完成，不允许通过 worktree 隔离后 cat/diff 同步到 main 的方式
   - 修改 → py_compile → 直接 commit 到 main

3. **禁止使用传入参数运行脚本**：
   - 所有脚本运行参数（数据路径、模型路径、epochs、batch_size、seed、category、out_dir 等）一律**硬编码到脚本内**（模块级常量或脚本内默认值），不允许通过命令行参数（argparse/sys.argv）传参运行
   - 运行方式统一为：`python script.py`（无参数），或 `python script.py --help` 查看硬编码配置
   - 已存在的 argparse 参数可保留（作为配置覆盖后备），但**实际执行必须依赖硬编码默认值**，不允许在运行命令里传参数
   - 分片/并行等场景同样不允许传参：在脚本内用硬编码的 offset/limit 或按数据分片逻辑实现

4. **解码必须使用批量解码（加快速度）**：
   - 所有 LLM 生成/解码（训练验证、推理、干预实验等）一律使用批量解码（同一 batch 内多条样本共享 forward pass），不允许逐条顺序生成
   - 参考实现：`copy_aware_generate.py` 的 `generate_batch`（batch 16，~18x 提速）、`e15_validate_h1.py` 的批量 `model.generate(batch_ids)`
   - 批量场景注意事项：
     - right-padding 时首步预测必须取每行**最后有效位置**的 logits（`attention_mask.sum(dim=1)-1`），不能取 `logits[:,-1]`（会预测到 padding 位置）
     - 后续步（KV cache 模式）logits 只有新 token 位置，用 `logits[:,-1]`
     - batch 内各样本独立 prompt/属性 span/hook 方向时，逐样本构建 mask/position/span 后整批 forward
   - 已实现单条生成的脚本（如无批量版本）必须改造或注明 TODO，不得新增逐条生成逻辑
   - **vLLM 后端必须直接 batched 调用**：`QwenLocalClient.call()` 内部走 `backend.model.generate([prompt])` 一次只传一个 prompt (batch=1)，仍属于逐条生成。**禁止**用 `client.call()` 串行循环生成多条 prompt。正确做法：
     - 直接调 `client._backend.model.generate([prompts], sampling_params)` 一次传入所有 prompt
     - 或封装 `batch_generate(client, prompts, temp, max_tokens)` 辅助函数（参考 `phase11_g_diverse_z_v2.py::batch_generate`，K=10 prompts → 1 次 vLLM 调用 vs 10 次串行，~3x 提速）
     - vLLM 内部 `apply_chat_template` + `tokenize=False` + `add_generation_prompt=True` 与 `call()` 一致
     - vLLM 对**完全相同**的 prompt 会 dedup 跳过采样；如需同 prompt 多 sample 获得不同输出，必须在 prompt 加唯一后缀（如 `(variant {seed_idx})`）或使用 `SamplingParams(seed=...)`

5. **已实现的提速方法（实现相关业务时必须使用）**：

   **a. 批量解码（~18x）**：LLM 生成/解码一律批量（规则 4）；参考 `copy_aware_generate.py::generate_batch`（batch 16）与 `e15_validate_h1.py`（批量 `model.generate(batch_ids)`）。注意 right-padding 首步取 `attention_mask.sum(dim=1)-1` 位置 logits。

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
      - vLLM 只适用于**标准 generate 路径**（无自定义 hook/copy head 干预的纯 LM 生成）；带自定义层 hook、copy head 或中间激活注入的生成必须走原生批量解码（规则 4/5a），不得硬套 vLLM
      - 使用 vLLM 时必须保持解码配置一致（temperature/top-p/penalty/max_tokens 与预注册或对照配置逐字段一致），禁止不同引擎引入解码差异
      - vLLM 服务启动/关闭纳入脚本生命周期；小批量（<100 条）或单次生成直接用原生批量解码即可，引擎开销不值得
      - **vLLM 必须 batched 调用，禁止串行 `client.call()` 循环**：见规则 4 末尾注意事项，`call()` 内部 batch=1 失去 batching 优势；正确做法是 `client._backend.model.generate([prompts], sampling)` 一次传所有 prompt

   **h. vLLM batched 调用参考实现（必读）**：
      - 路径：`phase11_g_diverse_z_v2.py::batch_generate(client, prompts, temp, max_tokens)`
      - 实现：
        ```python
        from vllm import SamplingParams
        sampling = SamplingParams(
            max_tokens=max_tokens, temperature=temp,
            top_p=0.95 if temp > 0 else 1.0,
        )
        full_prompts = [
            client._backend.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        outputs = client._backend.model.generate(full_prompts, sampling)
        texts = [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]
        return texts
        ```
      - 实测效果（Phase 11.G 30 pairs × K=10 personal prompts）：v1 `call()` 串行 593s → v2 batched 194s（**3x 提速**）
      - 适用场景：所有需要 K 个不同 prompt（style samples / beam candidates / diverse sampling）一次生成的业务
      - **陷阱**：vLLM 对完全相同 prompt dedup；同 prompt 多 sample 必须加 `(variant {idx})` 后缀或 `SamplingParams(seed=...)`

   这些方法均为已验证的实测提速；新实现/改造相关业务时必须选用适用的方法，禁止回退到逐条生成、逐元素循环、重复加载大文件等低效模式。

6. **任务完成必须 commit + push 两步闭环**：
   - 每个实验/任务完成时必须严格执行以下两步，缺一不可：
     1. **commit**：`git add` 所有相关文件（脚本 + iter 文档 + result JSON/log，result/ 在 .gitignore 用 `git add -f`），用 `git diff --cached --stat` 验证文件齐全后再 commit
     2. **push**：`git push` 到 main（已 commit 但未 push 视为未完成）
   - 任何一步遗漏都会破坏可复现性
   - **不允许**"代码写完 + 文档写完"就报告完成，必须两步全做完

7. **虚拟环境必须使用 `pq_env`（本项目专用）**：
   - 唯一允许的 Python 解释器：`/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`（python 3.10）
   - **禁止**使用系统 python、base conda、`genrec_env`、`rqvae_repro_env` 等其他环境运行本项目脚本
   - 运行脚本统一为：`/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python script.py`
   - `pq_env` 已安装：torch 2.13.0+cu130、transformers 4.40.2、spacy 3.8.4 + en_core_web_sm、scikit-learn 1.7.2、numpy 1.26.4、accelerate
   - **安装新依赖的注意点**：`~/.pip/pip.conf` 全局设置了 `target=/home/wlia0047/ar57/wenyu/.local`，直接 `pip install` 会装到 user-site 而非环境内；必须用 `--target $(pq_env python -c "import site; print(site.getsitepackages()[0])")` 显式安装到环境 site-packages
   - 新增/验证脚本运行前先确认解释器是 pq_env 的 python（避免误用旧环境导致依赖版本漂移，如 transformers 4.40.2 → 4.33.2 造成 Qwen 加载失败）

8. **后台任务必须使用 nohup，禁止 sbatch/srun**：
   - 所有需要长时间运行的脚本（训练、生成、评估、扫描等）一律用 `nohup ... &` 后台运行
   - **禁止**使用 `sbatch` / `srun` 提交作业（本项目不通过 slurm 队列运行）
   - 标准启动方式：
     ```bash
     cd /fs04/ar57/wenyu/PersoanlQuery
     nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python -c "
     import sys; sys.path.insert(0, 'query_gen'); sys.path.insert(0, 'syntactic_analysis')
     import query_gen_main as m
     m.MAIN_STAGE = 'generate'
     m.main()
     " > result/xxx.log 2>&1 &
     ```
   - 需先 `scancel` 取消已提交的 sbatch 作业；用 `ps aux | grep xxx` + `kill` 管理本地 nohup 进程

9. **禁止用 sleep 等待后台进程，改用定期手动轮询**：
   - 后台任务（nohup/setsid 启动的训练、生成、评估）启动后，**禁止** `sleep 300; tail xxx.log` 这类休眠等待
   - 应通过多次 `tail -n 20 <log>` / `ps aux | grep <proc>` 快速轮询检查进度，每次检查间隔由用户或上下文决定
   - 轮询时若进程已完成（log 出现 DONE/完成标志），立即进行下一步，不要多余等待

10. **临时文件 / 产物路径必须使用 `hj82_scratch2`，禁止写入 `/fs04`**：
    - 所有临时文件（log、debug output、intermediate cache、no_stdout redirect 等）、所有业务产物（实验 result JSON/npz/jsonl、scale_validity 等）一律写入 `/home/wlia0047/hj82_scratch2/wenyu/<sub_path>`
    - **禁止**写入 `/fs04/ar57/wenyu/PersoanlQuery/result/` 或 `/tmp/claude-*/.../tasks`（`/fs04` 总容量 500G 已 89% 占用，`/tmp` 多用户共享易冲突）
    - 项目代码 commit 仅保留在 `/fs04/ar57/wenyu/PersoanlQuery/`（git repo）；运行时的 log / cache / npz / jsonl 走 `hj82_scratch2`
    - 标准做法：
      - 业务代码内硬编码 OUT 路径为 `Path("/home/wlia0047/hj82_scratch2/wenyu/<feature>/result/...")`
      - 启动后台任务前显式 `export TMPDIR=/home/wlia0047/hj82_scratch2/wenyu/tmp`（避免 torch / vllm 把临时文件写到 `/tmp`）
      - 已有 `/fs04` 下的 result 文件需迁移到 `hj82_scratch2` 并在脚本里更新引用路径
    - 命名建议：`/home/wlia0047/hj82_scratch2/wenyu/<feature>/<artifact>.{json,jsonl,npz,log}`

11. **运行脚本必须放在 `/home/wlia0047/ar57/wenyu/PersoanlQuery` 目录下**：
    - 所有 `python script.py` / `nohup python ...` 启动前必须先 `cd /home/wlia0047/ar57/wenyu/PersoanlQuery`
    - 该目录与 `/fs04/ar57/wenyu/PersoanlQuery` 指向同一份 git repo（filesystem alias），但**所有运行/进程的 cwd 一律用 `/home/wlia0047/ar57/wenyu/PersoanlQuery`，禁止用 `/fs04/ar57/wenyu/PersoanlQuery`**
    - 原因：`/fs04` 总容量 500G 已 89% 占用且 IO 带宽受限；`/home/wlia0047/ar57/wenyu/PersoanlQuery` 走主盘, 启动 vLLM / 读 dataset / 写临时 cache 更稳
    - 标准启动方式：
      ```bash
      cd /home/wlia0047/ar57/wenyu/PersoanlQuery
      nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python script.py > /home/wlia0047/hj82_scratch2/wenyu/<feature>/xxx.log 2>&1 &
      ```
    - 验证 cwd：`pwd` 应输出 `/home/wlia0047/ar57/wenyu/PersoanlQuery`，否则视为违规
    - 例外：仅 `git` / `cat` / `Edit` 操作 git tracked 文件可走 `/fs04/...` 路径（IDE 兼容），但 Python 进程一律走 `/home/wlia0047/...`

12. **可运行脚本必须放在项目主目录 `/home/wlia0047/ar57/wenyu/PersoanlQuery` 下（禁止放在 `hj82_scratch2` 等外部目录）**：
    - 所有 `.py` 可运行脚本必须位于 `/home/wlia0047/ar57/wenyu/PersoanlQuery/` 内（包括任何子目录，如 `gen_query/`、`gaussian/`、`attribute_extraction/`、`syntactic_analysis/`、`select_query/` 等），"主目录"指项目根及其子目录树
    - **禁止**将可运行脚本放在 `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/` 或 `/home/wlia0047/hj82_scratch2/wenyu/postfilter/`、`e29_paper/`、`gaussian_cluster/`、`disentangle/` 等任何外部目录
    - `hj82_scratch2/wenyu/` 只允许存放 **运行时产物**：log / cache / jsonl / json / npz / pt / txt / png，**不允许存放 .py 可运行脚本**
    - 历史遗留在 scratch2 的脚本（实测 247 个 .py，集中在 `vades_prototype/` 207 个 + `postfilter/` 28 + `e29_paper/` 8 + `gaussian_cluster/` 3 + `disentangle/` 1）必须迁移回项目内对应位置并 `git add -f` + `git commit` + `git push` 闭环
    - 新增脚本必须直接写在项目内的对应子目录（按 phase 编号归档到 `result/phaseXX/` 下或对应模块目录），**禁止**先放 scratch2 再"将来迁移"
    - 迁移后须同步更新所有引用旧路径的脚本（import 路径、PATH 拼接、文档说明）指向新位置
    - 与规则 10 的分工：规则 10 管**运行时产物**路径（log/result），规则 12 管**脚本本体**路径；前者 scratch2，后者项目内

13. **`/result/` 目录只允许存放生成的产物，脚本必须放在对应的功能目录下**：
    - `/home/wlia0047/ar57/wenyu/PersoanlQuery/result/`（以及其任何子目录）**只允许存放生成的产物**：
      - 实验 result 文件：`*.json`、`*.jsonl`、`*.npz`、`*.pt`、`*.csv`、`*.log`
      - 各 phase 的子目录（如 `result/phase35b/`）只放该 phase 的产物（scores.json、eval_summary.json、intra_product_rank1.json、pca_components.npz 等）
    - **禁止**在 `result/` 任何位置（包括 `result/phase15/scripts/`、`result/phase16/scripts/` 等模式）放 `.py` 或 `.sh` 脚本
    - 脚本必须直接放在对应功能模块目录下，按业务归属划分（已存在目录）：
      | 功能模块 | 路径 | 用途 |
      |----------|------|------|
      | query 生成 | `gen_query/` | generate_strict_nohallu.py、build_user_style_vectors_*.py 等 |
      | query 选择 | `select_query/` | pick_best_cand_*.py、eval_query_quality_*.py、eval_rank1_*.py 等 |
      | VADES Gaussian | `gaussian/` | gaussian_vades.py、build_raw_candidate_queries.py、build_real_user_style_vectors.py 等 |
      | 用户筛选 / records | `attribute_extraction/` | build_query_records.py、filter_10k_active_users.py 等 |
      | 句法特征 | `syntactic_analysis/` | spacy/feature 相关脚本 |
      | 评测 / 训练编排 | 项目根或子模块根 | run_pipeline_10k.sh 等 orchestrator |
    - 历史遗留：当前 `result/phase15/scripts/` 下有 7 个脚本（filter_10k_active_users、build_raw_candidate_queries、build_user_style_vectors_stub_10k、build_real_user_style_vectors、eval_rank1_10k、eval_query_quality_10k、run_pipeline_10k.sh）**必须迁移**：
      - `filter_10k_active_users.py` → `attribute_extraction/filter_10k_active_users.py`
      - `build_raw_candidate_queries.py` → `gaussian/build_raw_candidate_queries.py`
      - `build_user_style_vectors_stub_10k.py` → `gen_query/build_user_style_vectors_stub.py`
      - `build_real_user_style_vectors.py` → `gaussian/build_real_user_style_vectors.py`
      - `eval_rank1_10k.py` → `select_query/eval_rank1_intra_pool.py`
      - `eval_query_quality_10k.py` → `select_query/eval_query_quality.py`
      - `run_pipeline_10k.sh` → 项目根 `run_pipeline_10k.sh`（顶层 orchestrator）
    - 新增脚本时**先按功能归属决定目录**，再写代码；不允许新建 `result/phaseXX/scripts/` 这种"phase 子目录里再嵌套 scripts"的反模式
    - 与规则 10 的分工：规则 10 管**运行时产物**路径（log/result → scratch2 或 result/），规则 13 管**项目内脚本**的目录组织（按功能模块而非按 phase）；前者管路径，后者管组织
    - 与规则 12 的关系：规则 12 规定"脚本必须在项目内"（不在 scratch2），规则 13 在此基础上进一步规定"脚本必须按功能模块组织"（不在 result/ 下）

14. **Syntax Subspace 流水线的目录分工（gaussian/ 只放用户先验分布）**：
    Syntax Subspace 流水线（Stage 1-7）按"业务职责"拆到 4 个目录，**`gaussian/` 只保留"计算用户先验分布"逻辑**，其它 3 个目录分别承担生成 / 选择 / 评估：

    | 目录 | 职责 | 当前主脚本 | 包含 stage |
    |------|------|------------|------------|
    | `gaussian/` | **只**放"计算用户先验分布"逻辑（per-user Mahalanobis Gaussian 拟合） | `syntax_subspace_user_gaussians.py` + `syntax_subspace_utils.py`（共享 path / hyperparam / feat_key / `_syntax_subspace_prepare`） | Stage 3 only |
    | `gen_query/` | 通过 vLLM LLM 为 100 个 ASIN 各生成 K=50 共享候选池，并抽取 spaCy 182d 句法特征 | `syntax_subspace_pool_regen.py` | Stage 1（pool_regen）+ Stage 2（features） |
    | `select_query/` | 基于 per-user Gaussian 从共享池选 Mahalanobis 最小候选，并跑 mean_t50 / A1_reject_repeat / C3_kmeans_k{2-5} 多策略 | `syntax_subspace_select.py` | Stage 4（select）+ Stage 7（a1_select） |
    | `syntactic_evaluation/` | 用 bm25s + GPU MiniLM 评估选出的查询能否把对应 ASIN 拉回 rank-1，并标定 V_low / V_user / V_high 波动率 | `syntax_subspace_retrieval.py` | Stage 5（retrieval）+ Stage 6（volatility） |

    **硬性约束**：
    - **`gaussian/` 只放 per-user Gaussian 拟合逻辑**：禁止在 `gaussian/` 下新增 pool_regen / feature 抽取 / selection / retrieval / volatility 业务脚本
    - **`syntax_subspace_utils.py` 必须留在 `gaussian/`**：它是 4 个兄弟脚本的共享底座（路径 / 超参 / `_syntax_subspace_prepare` / `feat_key` / `log`），删掉它整个流水线无法 import
    - **任何新加的 stage 脚本必须按业务职责放到对应目录**：
      - 新加"LLM 生成相关" → `gen_query/`
      - 新加"基于 user / query 分布做选择" → `select_query/`
      - 新加"retrieval / 评估指标" → `syntactic_evaluation/`
      - 新加"per-user 分布拟合"（如多高斯混合、t 分布等）→ `gaussian/`
    - **禁止**回到旧的 `gaussian/syntax_subspace_main.py` 一锅炖模式：历史上的 7-in-1 main.py 已废弃（commit a684bc8 删除），任何"重新合并"的提议都视为违规
    - 4 个脚本各自的 CLI 通过 `--stage {name|all}` 子命令路由（保留 argparse 选择器），但**实际运行参数全部硬编码在脚本顶部**（遵循规则 3）

    **共享关系**：
    - `syntax_subspace_utils.py` 被 `gen_query/` / `select_query/` / `syntactic_evaluation/` 三个脚本通过 `sys.path.insert(0, "<repo>/gaussian")` 导入
    - 4 个脚本互相**不直接 import**（避免循环依赖），只通过 `scratch/` 下的 JSON 文件通信（POOL → GAUSSIANS → SELECTION → RETRIEVAL → A1_SELECTION）
    - `syntactic_analysis/main.py::per_sentence_features_v2` 是 spaCy 182d 特征提取函数，被 `gen_query/syntax_subspace_pool_regen.py` 和 `gaussian/syntax_subspace_user_gaussians.py` 通过 sys.path 导入

    **用法**：
    ```bash
    # Stage 1 + 2 (gen_query/)
    cd /home/wlia0047/ar57/wenyu/PersoanlQuery
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python gen_query/syntax_subspace_pool_regen.py --stage pool_regen > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/pool_regen.log 2>&1 &
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python gen_query/syntax_subspace_pool_regen.py --stage features > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/features.log 2>&1 &

    # Stage 3 (gaussian/)
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python gaussian/syntax_subspace_user_gaussians.py > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/user_gaussians.log 2>&1 &

    # Stage 4 (select_query/)
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python select_query/syntax_subspace_select.py --stage select > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/select.log 2>&1 &

    # Stage 5 + 6 (syntactic_evaluation/)
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python syntactic_evaluation/syntax_subspace_retrieval.py --stage retrieval > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/retrieval.log 2>&1 &
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python syntactic_evaluation/syntax_subspace_retrieval.py --stage volatility > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/volatility.log 2>&1 &

    # Stage 7 (select_query/)
    nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python select_query/syntax_subspace_select.py --stage a1_select > /home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/a1_select.log 2>&1 &
    ```

    **与规则 13 的关系**：规则 13 给出"query 生成 / 选择 / 评测"的高层划分；规则 14 在此基础上**精确约束** Syntax Subspace 这条流水线的 4 个目录边界，特别是"gaussian/ 只放用户先验分布"。任何跨边界搬运（如把 select 逻辑搬进 gaussian/）都视为违反规则 14。
