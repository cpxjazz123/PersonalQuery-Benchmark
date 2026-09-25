# CLAUDE.md — PersonalQuery 项目约束

## 硬性规则

2. **禁止使用 git worktree**: 所有代码修改直接 commit 到 main,不允许通过 worktree 隔离后再 cat/diff 同步。

3. **禁止使用传入参数运行脚本**: 所有运行参数(数据路径 / 模型 / epochs / batch_size / seed / category / out_dir 等)必须硬编码到脚本内,运行统一为 `python script.py`(无参数)。

4. **解码必须使用批量解码**: 所有 LLM 生成 / 解码一律批量(`model.generate([batch_ids])` 或 `client._backend.model.generate([prompts], sampling)`),禁止逐条串行生成(特别注意 vLLM `client.call()` 内部 batch=1,必须直接 batched 调用)。

5. **已实现的提速方法必须使用**: 实现相关业务时必须选用适用的方法(批量解码 / 向量化张量 / `nlp.pipe` / 中间缓存 / 最小模型 / 预编码 / vLLM / batched vLLM),禁止回退到逐条生成 / 逐元素循环 / 重复加载大文件。

6. **Commit + push 远程仓库固定为 GitHub**: 远程仓库已切换为 `https://github.com/cpxjazz123/PersonalQuery-Benchmark.git`(`git push origin main`)。推送须在用户显式要求时执行,禁止自动 commit + push；仅当用户明确要求"commit"或"commit+push"时,才执行 `git add` + `git diff --cached --stat` 验证 + `git commit` + `git push origin main` 两步闭环。若上游未配置凭据(`403 / permission denied`),先提示用户配置 `gh`/token/SSH,不要降级到只本地 commit。

7. **虚拟环境必须使用 `pq_env`**: 唯一允许的 Python 解释器是 `/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`(python 3.10 + torch 2.13.0+cu130 + transformers 4.40.2 + spacy 3.8.4 + sklearn 1.7.2 + numpy 1.26.4 + accelerate),禁止使用系统 python / base conda / `genrec_env` / `rqvae_repro_env` 等其他环境。

7a. **pip 安装必须装到 pq_env 自身的 site-packages**:
    - ✅ **ALWAYS**: 走 `/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python -m pip install --target=/home/wlia0047/ar57_scratch/wenyu/pq_env/lib/python3.10/site-packages <pkg>`,或先 `source /home/wlia0047/ar57_scratch/wenyu/pq_env/bin/activate` 再 `pip install <pkg>`。依赖会装到 `/home/wlia0047/ar57_scratch/wenyu/pq_env/lib/python3.10/site-packages`,venv 自动加载,不走 USER_SITE 路径。
    - **全局 pip 陷阱**:`/etc/pip.conf` 强制设置 `target = /home/wlia0047/ar57/wenyu/.local`,所有 `pip install` 默认被重定向到该目录,pq_env 永远装不到,**所以必须显式 `--target=` 覆盖**。装完用 `/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python -c "import <pkg>; print(<pkg>.__version__)"` 自检,确认包装进了 pq_env。
    - 按规则 7(项目统一用 pq_env,禁止 base/system/conda),禁止 `pip install --user`、不要设 `PYTHONUSERBASE`,否则包会落到 `/home/wlia0047/.local/lib/python3.10/site-packages` 而 venv 加载不到,跑脚本时会 `ModuleNotFoundError`。

8. **后台任务必须使用 nohup,禁止 sbatch/srun**: 所有长时间运行的脚本(训练 / 生成 / 评估 / 扫描)一律 `nohup ... &` 后台运行,禁止通过 slurm 队列提交。

9. **禁止用 sleep 等待后台进程**: 后台任务启动后用 `tail -n 20 <log>` / `ps aux | grep <proc>` 多次快速轮询,禁止 `sleep 300; tail xxx.log` 这类休眠等待。**严禁在 Bash 工具调用中使用 sleep 命令(无论长短)监控后台日志**,必须只用 `tail` / `grep` / `ps` / `nvidia-smi` 等即时查询,让后台进程完成时由 harness 通知,或主动 grep 日志关键字判断进度。

10. **临时文件 / 产物路径必须使用 `hj82_scratch2`**: 所有 log / debug output / cache / npz / jsonl / json 一律写 `/home/wlia0047/hj82_scratch2/wenyu/<sub_path>`,禁止写入 `/fs04`(89% 占用,IO 受限)或 `/tmp`(多用户共享冲突)。

11. **运行脚本必须放在 `/home/wlia0047/ar57/wenyu/PersoanlQuery` 目录下**: 所有 Python 进程的 cwd 一律 `/home/wlia0047/ar57/wenyu/PersoanlQuery`(主盘,IO 带宽稳),禁止用 `/fs04/ar57/wenyu/PersoanlQuery`(仅 git / cat / Edit 允许)。

12. **可运行脚本必须放在项目主目录 `/home/wlia0047/ar57/wenyu/PersoanlQuery` 下**: 所有 `.py` 可运行脚本必须位于项目根及其子目录树(`01_attribute_extraction/` / `02_user_review_sentence_extract/` / `03_spacy_encode/` / `04_gaussian/` / `07_gen_query/` / `08_select_query/` / `09_sercl_user_profile/` / `10_typo_injection/` / `11_syntactic_evaluation/` / `12_typo_evaluation/` 等)内,禁止放在 `/home/wlia0047/hj82_scratch2/wenyu/` 等任何外部目录(该目录只允许 log / cache / jsonl / json / npz / pt / txt / png 运行时产物)。

13. **`/result/` 目录只允许存放生成的产物,脚本必须放在对应的功能目录下**: `/home/wlia0047/ar57/wenyu/PersoanlQuery/result/` 只放 `*.json / *.jsonl / *.npz / *.pt / *.csv / *.log` 产物,禁止放 `.py` / `.sh`;`.py` 脚本必须放在功能模块目录(`03_spacy_encode/` / `04_gaussian/` / `07_gen_query/` / `08_select_query/` / `09_sercl_user_profile/` / `10_typo_injection/` / `11_syntactic_evaluation/` 等),禁止新建 `result/phaseXX/scripts/` 反模式。

14. **当前流水线目录结构 (2026-09-11 合并重整)**: `03_spacy_encode/` 只放 PCFG rules + 统一 supervised `_SupEncoder` (31737→256→32) 训练 + 3-way profile(50%)/val(20%)/test(30%) SHA1 split + 32d z 缓存 (`strict3_embeddings.npz` + `strict3_encoder.pt` + `supervised_embeddings.npy`),不包含任何 Gaussian fitting 或 attribution 逻辑,**只训练一次**;`04_gaussian/` 放 per-user Gaussian 拟合全链路 (`fit_per_user_gaussian.py` 读 `strict3_embeddings.npz`,在 32d supervised 空间拟合 raw full Σ + ASIN cohort gates → `user_gaussian_stats.json`,canonical Gaussian artifact),不重训 encoder;`analysis/` 放 attribution 复盘 (`attribution_ablation.py` 7-method M1-M8 + 高/低方差分层 lift, `variance_factor_rules_vocab.py` / `variance_factor_with_semantic.py` σ²_u 多因子回归, `covariance_reproducibility.py` Σ_u profile/test 跨半稳定性, `syntax_pcfg_adaptive_eval.py` τ sweep + held-out eval 读 strict3);`07_gen_query/` 放 vLLM pool 生成;`08_select_query/` 放 Mahalanobis + Q_0.95 gate 选择;`09_sercl_user_profile/` 放 GECToR user error profile 抽取;`10_typo_injection/` 放 typo 注入 pipeline (Bernoulli + position weighted + mechanism sampling + 32d Gaussian constraint + MiniLM ≥0.9 + non-shared zone);`11_syntactic_evaluation/` 放 7-retriever 评估 (BM25 / SPLADE / MiniLM / MPNet / BGE / GTE / ColBERTv2);`12_typo_evaluation/` 放 typo 注入后的 paired hit@k 退化评估 (orig vs typo, 不算 flip)。

15. **禁止写 iter / iterator 文档文件**: 任务完成直接在 chat 输出总结(按规则 4 打印任务摘要并以"当前任务已完成,请做下一个任务的指示。"结尾),不要写 `iter_*.md` / `iter_*_*_*.md` / `iteration_*.md` / `*_iter.md` 之类的过程文档到任何目录(包括 `iterations/`、`result/`、项目根)。

16. **每个功能目录只对应一个结果文件**: 每个功能目录(`03_spacy_encode/` / `04_gaussian/` / `07_gen_query/` / `08_select_query/` / `09_sercl_user_profile/` / `10_typo_injection/` / `11_syntactic_evaluation/` / `12_typo_evaluation/` 等)运行后**只允许**在 `result/` 产出对应其职责的**单一结果文件**,禁止在同一目录下堆叠 ablation / sweep / variant 的多个变体文件。不同超参的多次实验结果必须收缩到一个文件,或在文件名前加 hash 后缀避免堆积。

17. **名字不允许有版本号(脚本和产物都算)**: 所有脚本文件名与产物文件名(`.py` / `.json` / `.jsonl` / `.npz` / `.pt` / `.csv` / `.log` / 任何扩展名)禁止出现版本号后缀(`v1` / `v2` / `_v10` / `version_X` 等)。命名含义应反映"做什么"而非"第几版"。历史迭代信息应在 commit message / PR description / CLAUDE.md / docs/ 中追溯,而非藏在文件名里。

18. **每个功能目录下只维护一个主脚本**: 每个功能目录(`04_gaussian/` / `07_gen_query/` / `08_select_query/` / `09_sercl_user_profile/` / `10_typo_injection/` / `11_syntactic_evaluation/` / `12_typo_evaluation/` 等)只允许存放一个主脚本,禁止在同一功能目录下堆积多个版本脚本。每次对功能进行迭代时,直接在主脚本上修改,不在同目录下新建脚本;不同实验/ablation/smoke 的差异通过脚本内的 smoke flag / 配置文件 / 环境变量管理,或通过 `result/` 下的产物文件区分。历史实验结果在 commit message 中追溯。

20. **所有方案必须先做 smoke 版本,确认无问题再全量运行**:
    - ✅ **ALWAYS**: 任何新脚本 / 新训练 / 新 eval / 新 sweep / 新 ablation 第一次运行必须先用**最小 smoke 配置**验证 pipeline 端到端跑通(无报错、无 OOM、无 silent bug、输出结构正确)
    - ❌ **NEVER**: 不做 smoke 直接跑全量(避免浪费几十分钟到几小时 GPU/IO 发现脚本有 bug)
    - **Smoke 配置示例**(以 typo injection Stage 10 为例, 2026-09-06):
      - 第一次跑: `SMOKE=True` → 5 (uid, asin, query) pairs = 5 pairs, <30s
      - smoke 通过后再: `SMOKE=False` → 217 users × ~3 queries = 376 pairs, ~60s
    - **Smoke 必须覆盖**: 数据加载 → 模型加载 → forward pass → 后处理 → 输出 JSON schema → 与 reference 数据形状对比
    - **Smoke 失败的常见原因**(必须先排查再扩大): Shape 错位、silent try/except 吞错、checkpoint 路径错、seed 不固定、GPU OOM (需 batch 砍半)、spaCy/torch 版本不兼容

21. **回复必须简单直接,尽量只使用一段话**: 每次回复只用一段连贯的话讲清结论和下一步,禁止列表/表格/分点,禁止解释过程,直接给答案 + 行动项。需要时给出文件路径、命令、状态值等关键事实,但不展开。详细日志写到文件,chat 里只给一句话总结。

22. **每次业务修改必须立即 commit + push 到 GitHub main**: 每完成一次业务代码 / 配置 / 文档 / 产物的修改,必须立刻执行 `git add -A` + `git status --short` 自检 + `git commit -m "<scope>: <what>"` + `git push origin main` 四步闭环(remote 已固定为 `https://github.com/cpxjazz123/PersonalQuery-Benchmark.git`),**不等待用户指示、不积攒多次改动一起提交**。本规则覆盖 Rule 6 中的"仅在用户显式要求时执行"——所有任务完成后默认触发自动 commit + push。仅维护 `main` 一个分支(同步 Rule 2),禁止 checkout / 创建其他分支,所有修改直接 commit 到 main。例外:`/home/wlia0047/ar57/wenyu/PersoanlQuery` 项目根的 `.claude/` 目录(本地 Claude 配置)、`.git/`、临时 scratch 产物不在 commit 范围内(由 `.gitignore` 控制)。如推送失败(`403 / permission denied` 等),先提示用户配置 `gh`/token/SSH,不要降级到只本地 commit。
