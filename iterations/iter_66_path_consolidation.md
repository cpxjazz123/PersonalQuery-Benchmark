# iter #66: Stage 5/6/7 路径整合 — 08_retrieval → 06_retrieval (post iter #48)

## 背景
iter #48 重命名 stage 目录 `08_retrieval` → `06_retrieval`,但 config JSON 和若干 Stage 7 脚本仍引用 `08_retrieval` 路径。Stage 7 Baby 跑通是因为旧 `08_retrieval` 目录碰巧还存在。

## 修复

### 1. `06_retrieval/08_retrieval_config.json` (工作树与 LIVE 同步)
- 所有 `08_retrieval` 路径模板 → `06_retrieval` (retriever_cache_dir / query_cache_dir / output_dir / metadata_cache_file / document_cache)
- 注释 `_comment` 更新

### 2. Stage 7 脚本 (4 个文件)
- `07_noisy_retrieval/07_generate_noisy_query_cache_Baby_Products.py` (line 48): `retrieval_root = current_dir.parent / "08_retrieval"` → `"06_retrieval"`
- 同上 for Grocery 和 Pet
- `07_noisy_retrieval/noisy_syntax_depth_eval_common.py` (line 37): `RETRIEVAL_ROOT = CURRENT_DIR.parent / "08_retrieval"` → `"06_retrieval"`

### 3. `common_utils.py` 路径别名
- 添加 `log = log_with_timestamp` (iter #23 漏做的 alias)
- 修复 Stage 02 `from common_utils import log` ImportError

### 4. `05_inject_noisy/common/apply_lambdamart_userbased_noisy.py`
- 添加 `_PROJECT_ROOT = _SCRIPT_DIR.parent.parent` 到 sys.path
- 修复 Stage 5 `from common_utils import log` 找不到模块

### 5. Stage 14 死代码清理
- 删除 `14_build_bge_query_cache_{Baby,Grocery,Pet}_Products.py` (Stage 7 已包含)
- 删除 `14_eval_rerank_*` (3 个, 不在评估流程)
- 删除 `14_run_llm_rerank_*` 和 `14_run_llm_rerank_noisy_*` (6 个)
- 重命名 `14_build_bge_query_cache_Pet_Supplies.py` → `14_build_bge_query_cache.py` (通用版)

## 验证

```bash
python3 -m py_compile <files>  # OK
git commit -m "iter #61: Stage 5/7 路径修复 + common_utils log alias"
```

## 运行时问题

Stage 7 Pet + Grocery 第一次用 `source genrec_env/bin/activate` 失败 — 因为该 env 没有 `activate` 脚本(只用 `conda activate`)。改用 `export PATH=...genrec_env/bin:$PATH` 解决。

第二次仍失败 — `CUDA_VISIBLE_DEVICES=1` 在某些节点导致 `_require_cuda_device` 找不到 GPU。改用默认 CUDA 设备后两个任务正常编码。

## 现状

- Stage 7 Baby: ✅ 已完成 (06_retrieval/query_cache_Baby_Products/, 已从旧 08_retrieval 目录迁移)
- Stage 7 Pet + Grocery: 🔄 运行中 (m3g112, m3g115)
