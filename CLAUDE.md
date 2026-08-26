# CLAUDE.md — PersonalQuery 项目约束

## 硬性规则

1. **Retriever index 缓存必须统一**: 所有使用 retriever index 的 stage 必须共享 `{scratch_result}/06_retrieval/retriever_<cat>_cache/` 同一份缓存,禁止在各 stage 目录重复构建。

2. **禁止使用 git worktree**: 所有代码修改直接 commit 到 main,不允许通过 worktree 隔离后再 cat/diff 同步。

3. **禁止使用传入参数运行脚本**: 所有运行参数(数据路径 / 模型 / epochs / batch_size / seed / category / out_dir 等)必须硬编码到脚本内,运行统一为 `python script.py`(无参数)。

4. **解码必须使用批量解码**: 所有 LLM 生成 / 解码一律批量(`model.generate([batch_ids])` 或 `client._backend.model.generate([prompts], sampling)`),禁止逐条串行生成(特别注意 vLLM `client.call()` 内部 batch=1,必须直接 batched 调用)。

5. **已实现的提速方法必须使用**: 实现相关业务时必须选用适用的方法(批量解码 / 向量化张量 / `nlp.pipe` / 中间缓存 / 最小模型 / 预编码 / vLLM / batched vLLM),禁止回退到逐条生成 / 逐元素循环 / 重复加载大文件。

6. **任务完成必须 commit + push 两步闭环**: 每个任务完成必须先 `git add` 所有相关文件 + `git diff --cached --stat` 验证 + `git commit`,再 `git push origin main`,缺一不可。

7. **虚拟环境必须使用 `pq_env`**: 唯一允许的 Python 解释器是 `/home/wlia0047/ar57_scratch/wenyu/pq_env/bin/python`(python 3.10 + torch 2.13.0+cu130 + transformers 4.40.2 + spacy 3.8.4 + sklearn 1.7.2 + numpy 1.26.4 + accelerate),禁止使用系统 python / base conda / `genrec_env` / `rqvae_repro_env` 等其他环境。

8. **后台任务必须使用 nohup,禁止 sbatch/srun**: 所有长时间运行的脚本(训练 / 生成 / 评估 / 扫描)一律 `nohup ... &` 后台运行,禁止通过 slurm 队列提交。

9. **禁止用 sleep 等待后台进程**: 后台任务启动后用 `tail -n 20 <log>` / `ps aux | grep <proc>` 多次快速轮询,禁止 `sleep 300; tail xxx.log` 这类休眠等待。

10. **临时文件 / 产物路径必须使用 `hj82_scratch2`**: 所有 log / debug output / cache / npz / jsonl / json 一律写 `/home/wlia0047/hj82_scratch2/wenyu/<sub_path>`,禁止写入 `/fs04`(89% 占用,IO 受限)或 `/tmp`(多用户共享冲突)。

11. **运行脚本必须放在 `/home/wlia0047/ar57/wenyu/PersoanlQuery` 目录下**: 所有 Python 进程的 cwd 一律 `/home/wlia0047/ar57/wenyu/PersoanlQuery`(主盘,IO 带宽稳),禁止用 `/fs04/ar57/wenyu/PersoanlQuery`(仅 git / cat / Edit 允许)。

12. **可运行脚本必须放在项目主目录 `/home/wlia0047/ar57/wenyu/PersoanlQuery` 下**: 所有 `.py` 可运行脚本必须位于项目根及其子目录树(`gen_query/` / `gaussian/` / `attribute_extraction/` / `syntactic_analysis/` / `select_query/` 等)内,禁止放在 `/home/wlia0047/hj82_scratch2/wenyu/` 等任何外部目录(该目录只允许 log / cache / jsonl / json / npz / pt / txt / png 运行时产物)。

13. **`/result/` 目录只允许存放生成的产物,脚本必须放在对应的功能目录下**: `/home/wlia0047/ar57/wenyu/PersoanlQuery/result/` 只放 `*.json / *.jsonl / *.npz / *.pt / *.csv / *.log` 产物,禁止放 `.py` / `.sh`;`.py` 脚本必须放在功能模块目录(`gen_query/` 生成 / `select_query/` 选择 / `gaussian/` / `attribute_extraction/` / `syntactic_analysis/` / 项目根 orchestrator),禁止新建 `result/phaseXX/scripts/` 反模式。

14. **Syntax Subspace 流水线按业务职责拆到 4 个目录,`gaussian/` 只放用户先验分布**: `gaussian/` 只放 per-user Mahalanobis Gaussian 拟合(Stage 3),`gen_query/` 放 vLLM pool 生成 + spaCy 特征抽取(Stage 1 + 2),`select_query/` 放 Mahalanobis + A1 reject-repeat(Stage 4 + 7),`syntactic_evaluation/` 放 bm25s + GPU MiniLM 评估 + V_low/V_user/V_high 标定(Stage 5 + 6);`syntax_subspace_utils.py` 必须留在 `gaussian/` 作 4 个脚本的共享底座,禁止回退到 7-in-1 `main.py` 大锅炖模式。

15. **禁止写 iter / iterator 文档文件**: 任务完成直接在 chat 输出总结(按 AGENTS.md 规则 4 打印任务摘要并以"当前任务已完成,请做下一个任务的指示。"结尾),不要写 `iter_*.md` / `iter_*_*_*.md` / `iteration_*.md` / `*_iter.md` 之类的过程文档到任何目录(包括 `iterations/`、`result/`、项目根)。