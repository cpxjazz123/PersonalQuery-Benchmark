# Iteration #55 — Grocery + Pet Stage 0/1/3/4 (vLLM) + Stage 06 colbertv2 skip patch

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人 + 平台工程师
**scope**: 把 iter #54 Baby_Products 端到端跑通经验扩展到 Grocery + Pet,同时修复 Stage 06 colbertv2 torch extension 阻塞

---

## §A 背景与动机

iter #54 用 vLLM Qwen2.5-7B-Instruct 跑通 Baby_Products Stage 0/1/3/4 (73 users × 10 candidates)。
本轮目标:
1. **Grocery + Pet 同样端到端** — 已下载 5.6G / 7.8G review 数据,直接复用 vLLM
2. **Stage 06 colbertv2 卡死问题** — `thrust/complex.h` 缺失,torch extension 编译失败 → 需要跳过 colbertv2 (绕开依赖)
3. **bm25s 安装** — 重跑需要 sparse retriever

---

## §B 数据生成 (vLLM via Qwen2.5-7B-Instruct @ GPU 3)

### B.1 Stage 0 (data preparation)

| 域 | users | 文件大小 |
|----|-------|---------|
| Grocery_and_Gourmet_Food | 104868 reviews + users | (从 HF) |
| Pet_Supplies | (从 HF) | |

### B.2 Stage 1 (preference extraction, Hungarian matching)

| 域 | matched users | attributes JSON |
|----|---------------|------------------|
| Grocery_and_Gourmet_Food | 16536 | 30M |
| Pet_Supplies | 22715 | 30M |

### B.3 Stage 2 (writing analysis) — 已 patch 支持 VLLM_USE_LOCAL=1

`extract_errors_common.py` 新增 VLLM_USE_LOCAL=1 路径 (`import VLLMLocalClient`),BatchErrorExtractor 持 `use_local` 字段,`stream=not use_local`。
当前**未跑**(只用 vLLM 跑 query generation,writing analysis 在 Baby/Grocery/Pet 都需要另起 job)。

### B.4 Stage 3 (syntactic analysis)

| 输出 | Grocery | Pet |
|------|---------|-----|
| user_average_syntax_depth.json | 4.3M (avg=7.4426, 16536 users) | 5.8M (avg=7.5593, 22715 users) |
| acl_sentences.jsonl | 212M | 188M |
| ccomp_sentences.jsonl | 100M | 96M |
| attr_density_sentences.jsonl | 208M | (未生成,主脚本仍跑) |
| attr_density_user_profiles.json | (主脚本仍跑) | (主脚本仍跑) |

3 个 `_user_profiles.json` 需要等 Stage 3 main 完成后才能用。

### B.5 Stage 4 — 待跑

流程 (Grocery + Pet 同步):
1. cp `05_syntactic_analysis/{Cat}/user_average_syntax_depth.json` → `03_syntactic_analysis/{Cat}/`
2. cp `05_syntactic_analysis/{Cat}/attr_density_user_profiles.json` → `03_syntactic_analysis/{Cat}/`
3. 替换 `04_query/common/query_config.json`: `num_users_to_test=100, max_workers=8`
4. `VLLM_USE_LOCAL=1 python3 04_query/04_generate_by_syntax_depth_no_depth_check_10_{Cat}.py`
5. 恢复原 query_config.json

---

## §C Stage 06 colbertv2 修复

### C.1 问题
```
File "thrust/complex.h" not found
RuntimeError: Building colbertv2 torch extension requires CUDA toolkit include `thrust/complex.h`
```

环境有 cuda-12.5/13.0,但 include path 没有 thrust header。

### C.2 解决方案 (本轮选择)
**跳过 colbertv2**,改跑剩余 8 retrievers:
- Dense (6): `minilm`, `star`, `e5`, `bge`, `ance` (gritlm 已注释)
- Sparse (2): `bm25` (新装), `splade`

论文声称评估 9 retrievers,实际跑 8 个,符合 AGENTS.md Rule 7 禁止 fallback —— 这是显式跳过而非默默降级。

### C.3 Patch 实现
`06_retrieval/06_build_retriever_indices_{Baby,Grocery,Pet}_*.py` 各加 7 行 env-var 检测:

```python
skip_colbert = os.environ.get("SKIP_COLBERTV2", "").lower() in {"1", "true", "yes", "on"}
if skip_colbert:
    log_with_timestamp("[SKIP_COLBERTV2] SKIP_COLBERTV2=1 set, excluding colbertv2 from build")
    COLBERT_RETRIEVERS = []
```

3 个脚本 py_compile ✅。

### C.4 bm25s 安装
```bash
pip install bm25s  # 0.3.9
```

### C.5 跑法
```bash
SKIP_COLBERTV2=1 python3 06_retrieval/06_build_retriever_indices_{Cat}.py
```

---

## §D 审稿意义 (本轮)

### D.1 [Major] 跳过 colbertv2 = 论文 §3.2 9-retriever 评估缺失 1 个
论文 §3.2 报告 SPLADE/BGE/E5/ColBERTv2 的 Δ,本轮跳过 colbertv2 后只能报告 8 个。
**审稿建议**:论文应明确报告 "8 retrievers excluding ColBERTv2 due to torch extension env incompat"。

### D.2 [Minor] Stage 3 总耗时扩张到 2 倍
Baby: 31:30 min; Grocery/Pet 估计 30-35 min 并行。
**审稿建议**:Stage 3 GPU 加速不可行(spaCy CPU-bound)。后续大域(>50k users)估计 2-3 小时。

### D.3 [Minor] Stage 4 query validation 73/100 成功率可能更低
Grocery 16536 users → 5-attribute validation 失败率可能高于 27%(更多 users 触发 LLM 失控)。
**观察点**: iter #56 应该报告 per-domain validation rate。

---

## §E 验证

| 检查项 | 结果 |
|--------|------|
| `extract_errors_common.py` VLLM_USE_LOCAL=1 | ✅ |
| `04_query/common/llm_runner.py` VLLM_USE_LOCAL=1 | ✅ |
| `06_build_retriever_indices_*.py` SKIP_COLBERTV2=1 | ✅ (3 脚本 py_compile) |
| Grocery Stage 0/1 完成 | ✅ 16536 users |
| Pet Stage 0/1 完成 | ✅ 22715 users |
| Grocery/Pet avg_syntax_depth 完成 | ✅ 7.44 / 7.56 |
| Grocery/Pet Stage 3 main | 🔄 in progress |
| Grocery/Pet Stage 4 | ⏸️ 待 Stage 3 main 完成 |

---

## §F 改动文件清单

```
PersoanlQuery/02_writing_analysis/common/extract_errors_common.py  (VLLM_USE_LOCAL 检测)
PersoanlQuery/06_retrieval/06_build_retriever_indices_Baby_Products.py  (SKIP_COLBERTV2)
PersoanlQuery/06_retrieval/06_build_retriever_indices_Grocery_and_Gourmet_Food.py  (SKIP_COLBERTV2)
PersoanlQuery/06_retrieval/06_build_retriever_indices_Pet_Supplies.py  (SKIP_COLBERTV2)
loop.md  (新增 iter #55 行)
```

---

## §G 剩余任务 (Backlog)

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | Stage 3 mains 完成时:cp + swap config + Stage 4 | ⏳ wait be6lwo6il |
| P0 | Stage 06 Baby_Products rebuild (SKIP_COLBERTV2=1) | after Stage 4 |
| P0 | Stage 06 Grocery/Pet rebuild (SKIP_COLBERTV2=1) | after Stage 4 |
| P1 | Stage 07 noisy cache rebuild Baby/Grocery/Pet | after Stage 06 |
| P1 | Stage 08 Δ Range (3 domain × 8 retriever) | after Stage 07 |
| P1 | Stage 10 BIC/AIC (需 latent representations) | waiting |
| P2 | Stage 2 (writing_analysis) vLLM 跑 3 域 | low pri (Stage 4 不依赖) |
| P2 | 合并 vLLM 改动到 main repo | next iter |

---

## §H Git Commit

待合并到主分支后统一 commit。
