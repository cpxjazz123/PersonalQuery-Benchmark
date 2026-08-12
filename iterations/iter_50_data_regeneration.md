# Iteration #50 — 数据重新生成

**日期**: 2026-07-20
**角色**: NLP/IR 专业审稿人
**scope**: 重新生成缺失的 result 数据

---

## §A 背景

### 问题：所有 result 数据缺失

检查发现 `/home/wlia0047/ar57/wenyu/result/personal_query/` 目录不存在（Stage 00-14 的所有输出数据都没有）。

### 原因分析
1. 大量 category-specific 脚本被删除（git status 显示 "deleted"）
2. Amazon review 原始数据文件缺失
3. git commit 94cd02b 重命名了文件（从 08_ 前缀改为 06_），导致本地工作目录的脚本丢失

---

## §B 本轮行动

### 1. 恢复所有已删除的脚本

```bash
cd /fs04/ar57/wenyu/PersoanlQuery
git restore $(git ls-files --deleted)
```

**结果**: 所有 3x category-specific 脚本（Baby_Products/Grocery_and_Gourmet_Food/Pet_Supplies）从 Stage 00 到 Stage 10 全部恢复。

### 2. 下载 Amazon Reviews 2023 Metadata

从 HuggingFace Hub (`McAuley-Lab/Amazon-Reviews-2023`) 下载 3 个 category 的 metadata：

```bash
python3 /home/wlia0047/ar57/wenyu/download_amazon_reviews.py
```

**结果**: ✅ 下载成功
| 文件 | 大小 |
|------|------|
| meta_Baby_Products.jsonl.gz | 165.9 MB |
| meta_Grocery_and_Gourmet_Food.jsonl.gz | 307.7 MB |
| meta_Pet_Supplies.jsonl.gz | 379.2 MB |

### 3. 修正路径问题

脚本期望路径: `/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/meta_Baby_Products.jsonl.gz`
下载保存路径: `/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/raw/meta_categories/meta_Baby_Products.jsonl.gz`

**修复**: 创建软链接
```bash
ln -sf raw/meta_categories/meta_Baby_Products.jsonl.gz \
       /home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2023/meta_Baby_Products.jsonl.gz
```

### 4. 安装缺失依赖

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/pip install beautifulsoup4 lxml -q
```

### 5. 运行 Stage 06 (Baby_Products)

使用 Python 环境: `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3`

```bash
cd /fs04/ar57/wenyu/PersoanlQuery
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3 \
    06_retrieval/06_build_retriever_indices_Baby_Products.py \
    2>&1 | tee /home/wlia0047/ar57/wenyu/result/stage06_baby.log
```

**当前状态**: 运行中
- 文档数: 217,724 products
- Retriever 列表: minilm ✅, star 🔄, e5 ⏳, bge ⏳, ance ⏳, colbertv2 ⏳, bm25 ⏳, splade ⏳
- ColBERTv2 已知问题: CUDA thrust/complex.h 缺失（无法构建）
- BM25 已知问题: bm25s 未安装

**已构建 cache**:
- minilm: 319MB embeddings (完整)
- star: 进行中
- bge: 128 bytes (空 → 需重建)

---

## §C 下一步

| 优先级 | 任务 | 依赖 |
|--------|------|------|
| P0 | 等待 Stage 06 Baby_Products 完成 | 当前运行中 |
| P0 | 运行 Stage 06 Grocery + Pet | 需依次执行 |
| P0 | 运行 Stage 04 query 生成 | 需要 LLM client |
| P1 | 运行 Stage 08 Δ Range 分析 | 需要 Stage 06-07 数据 |
| P1 | 运行 Stage 10 BIC/AIC | 需要 Stage 10 latent |

---

## §D 当前阻塞

1. **Stage 04 (query 生成)**: 需要 LLM 调用，无法在无人工干预下运行
2. **ColBERTv2**: CUDA 扩展缺失，无法构建
3. **BM25**: bm25s 未安装

---

## §E Git Commit

```
(待完成 — 需在 Stage 06 完成后统一 commit)
```
