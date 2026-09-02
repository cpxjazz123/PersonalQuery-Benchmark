# CLAUDE.md — PersonalQuery 项目约束

## 硬性规则

1. **Retriever index 缓存必须统一**: 所有使用 retriever index 的 stage 必须共享 `{scratch_result}/06_retrieval/retriever_<cat>_cache/` 同一份缓存,禁止在各 stage 目录重复构建。

2. **禁止使用 git worktree**: 所有代码修改直接 commit 到 main,不允许通过 worktree 隔离后再 cat/diff 同步。

3. **禁止使用传入参数运行脚本**: 所有运行参数(数据路径 / 模型 / epochs / batch_size / seed / category / out_dir 等)必须硬编码到脚本内,运行统一为 `python script.py`(无参数)。

4. **解码必须使用批量解码**: 所有 LLM 生成 / 解码一律批量(`model.generate([batch_ids])` 或 `client._backend.model.generate([prompts], sampling)`),禁止逐条串行生成(特别注意 vLLM `client.call()` 内部 batch=1,必须直接 batched 调用)。

5. **已实现的提速方法必须使用**: 实现相关业务时必须选用适用的方法(批量解码 / 向量化张量 / `nlp.pipe` / 中间缓存 / 最小模型 / 预编码 / vLLM / batched vLLM),禁止回退到逐条生成 / 逐元素循环 / 重复加载大文件。

5b. **spaCy 特征提取必须用 `spacy_features_parallel.py`**: 所有涉及 spaCy 句法特征提取的代码(包括 `per_sentence_features_v2` 调用),必须统一走 `common/spacy_features_parallel.py` 中的 `extract_styles_batch()` 或 `extract_styles_parallel()`,禁止直接在循环中调用 `nlp(text)` 逐句处理。

6. **Commit + push 仅在用户显式要求时执行**: 禁止在每个任务完成后自动 commit + push；仅当用户明确要求"commit"或"commit+push"时,才执行 `git add` + `git diff --cached --stat` 验证 + `git commit` + `git push origin main` 两步闭环。

7. **虚拟环境必须使用 `pq_env`**: 唯一允许的 Python 解释器是 `/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`(python 3.10 + torch 2.13.0+cu130 + transformers 4.40.2 + spacy 3.8.4 + sklearn 1.7.2 + numpy 1.26.4 + accelerate),禁止使用系统 python / base conda / `genrec_env` / `rqvae_repro_env` 等其他环境。

8. **后台任务必须使用 nohup,禁止 sbatch/srun**: 所有长时间运行的脚本(训练 / 生成 / 评估 / 扫描)一律 `nohup ... &` 后台运行,禁止通过 slurm 队列提交。

9. **禁止用 sleep 等待后台进程**: 后台任务启动后用 `tail -n 20 <log>` / `ps aux | grep <proc>` 多次快速轮询,禁止 `sleep 300; tail xxx.log` 这类休眠等待。**严禁在 Bash 工具调用中使用 sleep 命令(无论长短)监控后台日志**,必须只用 `tail` / `grep` / `ps` / `nvidia-smi` 等即时查询,让后台进程完成时由 harness 通知,或主动 grep 日志关键字判断进度。

10. **临时文件 / 产物路径必须使用 `hj82_scratch2`**: 所有 log / debug output / cache / npz / jsonl / json 一律写 `/home/wlia0047/hj82_scratch2/wenyu/<sub_path>`,禁止写入 `/fs04`(89% 占用,IO 受限)或 `/tmp`(多用户共享冲突)。

11. **运行脚本必须放在 `/home/wlia0047/ar57/wenyu/PersoanlQuery` 目录下**: 所有 Python 进程的 cwd 一律 `/home/wlia0047/ar57/wenyu/PersoanlQuery`(主盘,IO 带宽稳),禁止用 `/fs04/ar57/wenyu/PersoanlQuery`(仅 git / cat / Edit 允许)。

12. **可运行脚本必须放在项目主目录 `/home/wlia0047/ar57/wenyu/PersoanlQuery` 下**: 所有 `.py` 可运行脚本必须位于项目根及其子目录树(`gen_query/` / `gaussian/` / `attribute_extraction/` / `syntactic_analysis/` / `select_query/` 等)内,禁止放在 `/home/wlia0047/hj82_scratch2/wenyu/` 等任何外部目录(该目录只允许 log / cache / jsonl / json / npz / pt / txt / png 运行时产物)。

13. **`/result/` 目录只允许存放生成的产物,脚本必须放在对应的功能目录下**: `/home/wlia0047/ar57/wenyu/PersoanlQuery/result/` 只放 `*.json / *.jsonl / *.npz / *.pt / *.csv / *.log` 产物,禁止放 `.py` / `.sh`;`.py` 脚本必须放在功能模块目录(`gen_query/` 生成 / `select_query/` 选择 / `gaussian/` / `attribute_extraction/` / `syntactic_analysis/` / 项目根 orchestrator),禁止新建 `result/phaseXX/scripts/` 反模式。

14. **Syntax Subspace 流水线按业务职责拆到 4 个目录,`gaussian/` 只放用户先验分布**: `gaussian/` 只放 per-user Mahalanobis Gaussian 拟合(Stage 3),`gen_query/` 放 vLLM pool 生成 + spaCy 特征抽取(Stage 1 + 2),`select_query/` 放 Mahalanobis + A1 reject-repeat(Stage 4 + 7),`syntactic_evaluation/` 放 bm25s + GPU MiniLM 评估 + V_low/V_user/V_high 标定(Stage 5 + 6);`common/` 放 2 个共享底座(`syntax_subspace_utils.py` 路径/超参/`_syntax_subspace_prepare`,`syntactic_features.py` 318d `per_sentence_features_v2`),禁止回退到 7-in-1 `main.py` 大锅炖模式。

15. **禁止写 iter / iterator 文档文件**: 任务完成直接在 chat 输出总结(按规则 4 打印任务摘要并以"当前任务已完成,请做下一个任务的指示。"结尾),不要写 `iter_*.md` / `iter_*_*_*.md` / `iteration_*.md` / `*_iter.md` 之类的过程文档到任何目录(包括 `iterations/`、`result/`、项目根)。

16. **每个功能目录只对应一个结果文件**: 每个功能目录(`gen_query/` / `select_query/` / `gaussian/` / `syntactic_evaluation/` / `attribute_extraction/` / `syntactic_analysis/` 等)运行后**只允许**在 `result/` 产出对应其职责的**单一结果文件**,禁止在同一目录下堆叠 ablation / sweep / variant 的多个变体文件。不同超参的多次实验结果必须收缩到一个文件,或在文件名前加 hash 后缀避免堆积。

17. **名字不允许有版本号(脚本和产物都算)**: 所有脚本文件名与产物文件名(`.py` / `.json` / `.jsonl` / `.npz` / `.pt` / `.csv` / `.log` / 任何扩展名)禁止出现版本号后缀(`v6m` / `v6k` / `v2` / `_v10` 等),包括但不限于:
    - 算法迭代版本号(`v6m` / `v6k` / `v6g` / `v6h` / `v6i` / `v6j` / `v6l` / `v6f` 等 Stage 4 ablation 编号)
    - 通用版本后缀(`v1` / `v2` / `v3` / `version_X` / `*_v2_backup` 等)
    - 命名含义应反映"做什么"而非"第几版"(如 `syntax_subspace_select_strict_alignment.py`,不是 `syntax_subspace_select_v6m_strict_alignment.py`)。历史迭代信息应在 commit message / PR description / CLAUDE.md / docs/ 中追溯,而非藏在文件名里。

18. **每个功能目录下只维护一个主脚本**: 每个功能目录(`gaussian/` / `gen_query/` / `select_query/` / `gaussian/` / `syntactic_evaluation/` 等)只允许存放一个主脚本,禁止在同一功能目录下堆积多个版本脚本。已被其他模块 import 的工具函数(如 `build_user.py`)必须移动到 `common/` 目录。每次对功能进行迭代时,直接在主脚本上修改,不在同目录下新建脚本;不同实验/ablation/smoke 的差异通过配置文件或环境变量管理,或通过 `result/` 下的产物文件区分。历史实验结果在 commit message 中追溯。

19. **被引用的模块必须放在 `common/`**: 跨目录被其他模块 import 的工具/函数必须放在 `/home/wlia0047/ar57/wenyu/PersoanlQuery/common/`,禁止散落在功能目录内;功能目录内只允许放主脚本本身(可运行入口)。当前已识别: `common/syntactic_features.py`(句法特征) / `common/syntax_subspace_utils.py`(路径/超参) / `common/build_user.py`(Stage 1-2 cohort 入口,被 attribute_extraction/gen_query/select_query 引用)等。

20. **所有方案必须先做 smoke 版本,确认无问题再全量运行**:
    - ✅ **ALWAYS**: 任何新脚本 / 新训练 / 新 eval / 新 sweep / 新 ablation 第一次运行必须先用**最小 smoke 配置**验证 pipeline 端到端跑通(无报错、无 OOM、无 silent bug、输出结构正确)
    - ❌ **NEVER**: 不做 smoke 直接跑全量(避免浪费几十分钟到几小时 GPU/IO 发现脚本有 bug)
    - **Smoke 配置示例**(以 Phase 6.B.4b audit 为例,2026-08-30):
      - 第一次跑: `N_PAIRS=3, N_PCS=1, ALPHAS=[-2.0, +2.0]` → 24 generations, <1 min
      - smoke 通过后再: `N_PAIRS=20, N_PCS=2, ALPHAS=[-2.0, +2.0]` → 320 generations, ~5 min
      - 全量(只在用户明确批准后): `N_PAIRS=30, N_PCS=4, ALPHAS=[-2.0, 0.0, +2.0]` → 2880 generations, ~80 min
    - **Smoke 必须覆盖**: 数据加载 → 模型加载 → forward pass → 后处理 → 输出 JSON schema → 与 reference 数据形状对比
    - **Smoke 失败的常见原因**(必须先排查再扩大): Shape 错位(PCA 维度转置)、silent try/except 吞错、checkpoint 路径错、seed 不固定、GPU OOM (需 batch 砍半)、spaCy/torch 版本不兼容
