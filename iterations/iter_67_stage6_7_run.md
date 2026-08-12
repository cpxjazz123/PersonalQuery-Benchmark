# iter #67: Stage 6/7 BM25 + 完整 query cache 跑通 (3 domains)

## 背景
iter #66 修了 Stage 7 路径后,又发现两个新问题:
1. Stage 7 query cache 默认包含 ColBERTv2 → 实际跑会因 `from colbert.infra import ColBERTConfig` ModuleNotFoundError 失败
2. Stage 6 build 也硬编码 ColBERTv2 → 即便设置了 SKIP_COLBERTV2,也会在 splade 编码时 OOM 后被强制 build ColBERTv2 失败

## 修复 (iter #62 + iter #63)

### Stage 7 query cache: SKIP_COLBERTV2 filter
所有 3 个 `07_generate_noisy_query_cache_*.py` 添加:
```python
if (os.environ.get("SKIP_COLBERTV2", "").lower() in {"1","true","yes","on"}
        and 'ColBERTv2' in args.retrievers):
    log_with_timestamp("[SKIP_COLBERTV2] SKIP_COLBERTV2=1 set, excluding ColBERTv2 ...")
    args.retrievers = [r for r in args.retrievers if r != 'ColBERTv2']
```

### Stage 6 build: SKIP_COLBERTV2 filter
所有 3 个 `06_build_retriever_indices_*.py` 添加:
```python
skip_colbert = os.environ.get("SKIP_COLBERTV2", "").lower() in {"1","true","yes","on"}
if skip_colbert and COLBERT_RETRIEVERS:
    COLBERT_RETRIEVERS = []
```

### sbatch wrapper 改动
- 必须用 `--export=ALL,SKIP_COLBERTV2=1` (不能仅在脚本里 `export`,因为 SLURM 计算节点会 strip 变量)
- Stage 6 必须 `--mem=256G` (BM25 在 603K docs 上 compute_scores 峰值 ~50GB,默认 4G 直接 OOM)

### Path consolidation
- `06_retrieval/08_retrieval_config.json`: 所有 `08_retrieval` → `06_retrieval` (与 LIVE 同步)
- Stage 7 (4 个文件) 同步 `retrieval_root = current_dir.parent / "06_retrieval"`
- 旧 `08_retrieval/query_cache_Baby_Products/*` 移到 `06_retrieval/query_cache_Baby_Products/`
- 旧 `08_retrieval/retriever_Pet_Supplies_cache/bm25_*.pkl` 移到 `06_retrieval/`

## 结果

| Domain | Stage 6 索引 | Stage 7 query cache |
|--------|--------------|----------------------|
| Baby_Products | dense 5 + splade + bm25 (705MB) ✅ | ✅ all 6 retrievers (BGE/E5/MiniLM/STAR/ANCE/SPLADE/BM25), 14 files, 0.3 MB |
| Pet_Supplies | dense 5 + bm25 ✅ (splade 没建,但 query cache 不需要)| ✅ all 7 retrievers, 16.8s |
| Grocery_and_Gourmet_Food | dense 5 + bm25 (705MB) ✅ (splade 超时被杀) | ✅ 6 retrievers (skip splade), 92.9s |

## 后续

- iter #68: 跑 Stage 08 Δ Range BIC/AIC 评估 (P0)
- splade 单独补建 Grocery (24min)
