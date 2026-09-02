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

3. **每个功能目录下只维护一个主脚本**：
   - 每个功能目录（如 `gaussian/`、`gen_query/`、`select_query/`、`gaussian/` 等）只允许存放一个主脚本，禁止在同一目录下堆积多个版本脚本（如 `learned_syntax_gaussian_8a.py` + `learned_syntax_gaussian_8a2.py` + `learned_syntax_gaussian_8a3.py` ……）
   - 每次对功能进行迭代时，直接在主脚本上修改，不在同目录下新建脚本；不同实验/ablation/smoke 的差异通过配置文件或环境变量管理，或通过 `result/` 下的产物文件区分
   - 历史实验结果在 commit message 中追溯

4. **禁止使用传入参数运行脚本**：
   - 所有脚本运行参数（数据路径、模型路径、epochs、batch_size、seed、category、out_dir 等）一律**硬编码到脚本内**（模块级常量或脚本内默认值），不允许通过命令行参数（argparse/sys.argv）传参运行
   - 运行方式统一为：`python script.py`（无参数），或 `python script.py --help` 查看硬编码配置
   - 已存在的 argparse 参数可保留（作为配置覆盖后备），但**实际执行必须依赖硬编码默认值**，不允许在运行命令里传参数
   - 分片/并行等场景同样不允许传参：在脚本内用硬编码的 offset/limit 或按数据分片逻辑实现

5. **解码必须使用批量解码（加快速度）**：
   - 所有 LLM 生成/解码（训练验证、推理、干预实验等）一律使用批量解码（同一 batch 内多条样本共享 forward pass），不允许逐条顺序生成
   - 参考实现：`copy_aware_generate.py` 的 `generate_batch`（batch 16，~18x 提速）、`e15_validate_h1.py` 的批量 `model.generate(batch_ids)`
   - 批量场景注意事项：
     - right-padding 时首步预测必须取每行**最后有效位置**的 logits（`attention_mask.sum(dim=1)-1`），不能取 `logits[:,-1]`（会预测到 padding 位置）
     - 后续步（KV cache 模式）logits 只有新 token 位置，用 `logits[:,-1]`
     - batch 内各样本独立 prompt/属性 span/hook 方向时，逐样本构建 mask/position/span 后整批 forward
   - 已实现单条生成的脚本（如无批量版本）必须改造或注明 TODO，不得新增逐条生成逻辑

6. **已实现的提速方法（实现相关业务时必须使用）**：

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

   这些方法均为已验证的实测提速；新实现/改造相关业务时必须选用适用的方法，禁止回退到逐条生成、逐元素循环、重复加载大文件等低效模式。

7. **Commit + push 仅在用户显式要求时执行**：禁止在每个任务完成后自动 commit + push；仅当用户明确要求"commit"或"commit+push"时，才执行 `git add` + `git diff --cached --stat` 验证 + `git commit` + `git push origin main` 两步闭环。

8. **虚拟环境必须使用 `pq_env`（本项目专用）**：
   - 唯一允许的 Python 解释器：`/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`（python 3.10）
   - **禁止**使用系统 python、base conda、`genrec_env`、`rqvae_repro_env` 等其他环境运行本项目脚本
   - 运行脚本统一为：`/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python script.py`
   - `pq_env` 已安装：torch 2.13.0+cu130、transformers 4.40.2、spacy 3.8.4 + en_core_web_sm、scikit-learn 1.7.2、numpy 1.26.4、accelerate
   - **安装新依赖的注意点**：`~/.pip/pip.conf` 全局设置了 `target=/home/wlia0047/ar57/wenyu/.local`，直接 `pip install` 会装到 user-site 而非环境内；必须用 `--target $(pq_env python -c "import site; print(site.getsitepackages()[0])")` 显式安装到环境 site-packages
   - 新增/验证脚本运行前先确认解释器是 pq_env 的 python（避免误用旧环境导致依赖版本漂移，如 transformers 4.40.2 → 4.33.2 造成 Qwen 加载失败）

9. **后台任务必须使用 nohup，禁止 sbatch/srun**：
   - 所有需要长时间运行的脚本（训练、生成、评估、扫描等）一律用 `nohup ... &` 后台运行
   - **禁止**使用 `sbatch` / `srun` 提交作业（本项目不通过 slurm 队列运行）
   - 标准启动方式：
     ```bash
     cd /home/wlia0047/ar57/wenyu/PersoanlQuery
     nohup /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python -c "
     import sys; sys.path.insert(0, 'gaussian')
     import query_gen_main as m
     m.MAIN_STAGE = 'generate'
     m.main()
     " > result/xxx.log 2>&1 &
     ```
   - 需先 `scancel` 取消已提交的 sbatch 作业；用 `ps aux | grep xxx` + `kill` 管理本地 nohup 进程

10. **禁止用 sleep 等待后台进程，改用定期手动轮询**：
   - 后台任务（nohup/setsid 启动的训练、生成、评估）启动后，**禁止** `sleep 300; tail xxx.log` 这类休眠等待
   - 应通过多次 `tail -n 20 <log>` / `ps aux | grep <proc>` 快速轮询检查进度，每次检查间隔由用户或上下文决定
   - 轮询时若进程已完成（log 出现 DONE/完成标志），立即进行下一步，不要多余等待

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

12. **`/result/` 目录只允许存放生成的产物，脚本必须放在对应的功能目录下**：
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
    - 与规则 10 的关系：规则 10 管"运行 cwd"（必须在 `/home/wlia0047/ar57/wenyu/PersoanlQuery`），规则 11 管"项目内脚本的目录组织"（按功能模块而非按 phase）

19. **被引用的模块必须放在 `common/`**：跨目录被其他模块 import 的工具/函数必须放在 `/home/wlia0047/ar57/wenyu/PersoanlQuery/common/`，禁止散落在功能目录内；功能目录内只允许放主脚本本身（可运行入口）。当前已识别：`common/syntactic_features.py`（句法特征）、`common/syntax_subspace_utils.py`（路径/超参）、`common/build_user.py`（Stage 1-2 cohort 入口，被 attribute_extraction/gen_query/select_query 引用）等。
