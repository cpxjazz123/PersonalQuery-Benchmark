# iter #56a: 修复 torchcodec 阻塞 sentence_transformers → Stage 06 dense 索引重建

## 根因

`sentence-transformers==5.6.0` 在模块导入时执行:
```python
# sentence_transformers/base/modality_types.py:16
from torchcodec.decoders import AudioDecoder, VideoDecoder
```

`torchcodec==0.15.0` 的 C 扩展 `libtorchcodec_core4.so` 需要 FFmpeg 8, 但环境只有 FFmpeg 5.1.4 → 加载失败 → sentence_transformers 整个 import 失败 → 所有 5 个 dense retriever (minilm/star/e5/bge/ance) 全部 BUILD_FAILED。

### 错误链
```
sentence_transformers/__init__.py
  → base/__init__.py
    → base/model.py:30 (import modality_types)
      → modality_types.py:16: from torchcodec.decoders import AudioDecoder, VideoDecoder
        → torchcodec/_core/__init__.py:8 (load .so)
          → OSError: Could not load /.../torchcodec/libtorchcodec_core4.so
            ("FFmpeg version 8: ... libtorchcodec_core4.so: cannot open shared object file")
```

## 修复

```bash
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/pip uninstall -y torchcodec
```

sentence_transformers 5.6 检测 torchcodec 不存在后自动跳过 modality_types, 模块加载成功 (验证: `from sentence_transformers import SentenceTransformer` OK;  MiniLM encode OK)。

## Stage 06 Pet 重跑进度 (2026-07-21 03:09 起)

| Retriever | Status | Time |
|-----------|--------|-----:|
| minilm    | ✅ BUILD_SUCCESS | 534.5s (~9 min) |
| star      | 🔄 running | ... |
| e5        | pending | |
| bge       | pending | |
| ance      | pending | |
| bm25      | ✅ already cached | 188s |
| splade    | pending (was at batch 64/3850 when killed) | est. 17 min |

单 dense 模型平均 ~10 min, 总 Stage 06 Pet 预计 ~75 min。

## 后续计划

1. Pet 完成后立即跑 Grocery 和 Baby (类似时间)。
2. Stage 07 noisy query cache 同样需要 dense — torchcodec 修复已生效, 可直接跑。
3. 写入 loop.md §10 iter log。

## Reviewer 观察

- dense 模型在 GPU 0/1/2 空闲的情况下还只用单卡 (sentence_transformers 默认行为) — 后续可以改造支持多卡并行加速。
- BM25 用 bm25s 0.3.9 (init 时可能 OOM, 实际跑没问题)。
- SPLADE 是 sparse 检索但走 dense encoder (SPLADE++ ED), 比 BM25 慢 5x — 总 17 min 单 domain。