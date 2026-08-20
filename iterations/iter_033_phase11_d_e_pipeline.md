# Phase 11.D + 11.E: End-to-End Architecture B + Validation — 总结

**Date**: 2026-08-20
**Status**: PARTIAL-GO (e2e pipeline works, validation positive but rank-1 not run)

## 实验动机

Phase 11.C 训练了 denoiser,验证 s_u signal 0.42(模型真的用了 style 输入)。下一步:
1. **11.D**: 把 denoiser 嵌入完整 pipeline (q_neutral → diffusion → z_personalized → style hint → LLM → q_personalized)
2. **11.E**: 4-condition 头对头对比 (11.B sampled-z vs 11.D vs Phase 10.15 baseline)

## 11.D Pipeline

每个 pair 产生 5 × 4 = 20 candidates:
1. **q_neutral**: LLM with attrs only, temperature=0.3 → clean content
2. **z_neutral**: encode q_neutral (Qwen layer 14 hidden) → PCA → normalize → 128d
3. **s_u samples**: 从 N(μ_u, Σ_u) 采 K=5 个 128d style vectors
4. **diffusion reverse**: z_neutral + s_u (100 steps) → z_personalized (K=5 × 128d)
5. **z_to_style_hint**: 把 z_personalized (vs mu_u) 转成 NL style hints (top dim shifts)
6. **q_personalized**: LLM with style hints + attrs, temperature=0.7 → K=5 prompts × M=4 seeds = 20 cands
7. **hard-copy post-process**: append missing Brand/Color/Material

30 pairs × 20 cands = 600 records,total 593s (~10 min GPU)

## 11.D Bug 修复 (5 项)

1. **PCA npz 路径**: `phase11_c_content_pca.npy` 实际生成 `.npy.npz`,改 `.npz` 后缀
2. **Qwen call signature**: 不接受 `top_p`/`top_k`/messages 列表,改用 prompt (str) only
3. **Float32 alpha tensors**: numpy float64 在 GPU 与 bf16 dtype mismatch
4. **CUDA OOM**: with_vllm=True (71GB) + transformers hidden_states (14GB) > 79GB. 修复:`QWEN_GPU_MEMORY_UTILIZATION=0.5` 让 vLLM 用 40GB,transformers 14GB = 54GB ✓
5. **QwenLocalClient 限制**: with_vllm=False 不支持 call() (只支持 hidden_states),必须双开

## 11.E Validation

### Diversity (1 - mean_pair_cos)

```
11.B sampled-z      : 0.6999  (best — 直接 Gaussian 采样 TinyStyler prefix)
11.B injection-off  : 0.5732
11.D q_personalized : 0.5069  ← 中间
11.B mean-z         : 0.4458
11.B shuffled-z     : 0.4394
```

11.D 比 single-mean baseline 高 (0.51 > 0.45),但比 Gaussian sampling 低 (0.51 < 0.70)。

### Attr Coverage

```
11.D q_personalized: 93.2% (559/600) all 3 attrs (Brand+Color+Material)
- Brand:    97.0%
- Color:    97.7%
- Material: 95.5%
```

LLM with style hints 自然列出 3 个 attrs,hard-copy 是 no-op。Material 最易漏。

### Style Preservation (cos margin)

```
11.D q_personalized : target=2.11, other=1.16, margin=+0.95
11.B sampled-z      : target=3.78, other=2.53, margin=+1.25
```

两者 margin 都是正的,cands 偏向 target user style。11.B sampled-z 略强(+1.25 > +0.95),但 11.D 也有明显风格信号。

### Phase 10.15 Baseline (参考)

- rank-1 coverage: 1.26% (876 pairs × 96 cands)
- top-10 coverage: 9.25%
- mean best rank: 307.8

## 决策: PARTIAL-GO

- ✅ Diversity: 11.D > 11.B mean-z
- ✅ Attr coverage: 93.2% (without hard-copy active)
- ✅ Style margin: +0.95 (positive, target user preferred)
- ⏳ **Open**: rank-1 eval on 30 pairs not run
- ⏳ Direct head-to-head with Phase 10.15 / 10.19 contrastive RAG (rank-1 17% top-100) not done

## Phase 11 整体决策: 需 rank-1 eval

| Phase | Status | Key Result |
|-------|--------|-----------|
| 11.A Gaussian cache | GO | held-out rank-1 73.5% |
| 11.B Diversity test | GO | sampled-z 1.57x diversity gain |
| 11.C Diffusion training | PARTIAL-GO | s_u signal 0.42, recon err 0.10, style transfer cos ~0 |
| 11.D E2E pipeline | COMPLETE | 600 cands in 10 min |
| 11.E Validation | PARTIAL-GO | Diversity + Attr + Style margin all OK |

**下一步**(如果要做):
1. 跑 11.D 的 600 cands 通过 VADES rerank (类似 phase10_15_rank1_coverage.py),看 rank-1 是否 > 1.26%
2. 如果 rank-1 > 5% → GO (commit as Phase 11 主线)
3. 如果 rank-1 ~ 1% → NO-GO (commit 为记录,保留 11.D 作为 baseline)

## 文件位置

- Pipeline: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase11_d_e2e_pipeline.py`
- Validation: `phase11_e_validation.py`
- Outputs:
  - `phase11_d_e2e_queries.jsonl` (600 records)
  - `phase11_d_e2e_meta.json`
  - `phase11_e_validation_report.json`
  - `phase11_e_summary.md`

## 工程教训

1. **np.savez 自动加 .npz 后缀**: 不要把变量名定义为 `something.npy` 然后 `np.savez(path)` — 会变成 `something.npy.npz`
2. **QwenLocalClient 双开内存管理**: `QWEN_GPU_MEMORY_UTILIZATION=0.5` 让 vLLM + transformers 共存
3. **call() 不支持 top_p/k/messages**: 只接 prompt (str)。需要 top_p/k 时改 vLLM 直接调用
4. **Float32 alpha tensors**: numpy float64 在 GPU 与 bf16 不兼容,diffusion reverse 必须显式 .astype(np.float32)
5. **cos > 1 不一定是 bug**: 如果 mu 没归一化,dot product 可以 > 1。相对比较时使用相同 mu 即可