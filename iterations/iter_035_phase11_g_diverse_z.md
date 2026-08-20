# Phase 11.G: Diverse z_personalized + Batched vLLM — 总结

**Date**: 2026-08-20
**Status**: NO-GO (rank-1 0/30, rank-10 0/30, 比 11.D 还差)
**Engineering**: **vLLM batched generate 加速 3x (10 min → 3.2 min)** → 写入 CLAUDE.md 规则 4/5h

## 实验动机

Phase 11.F 显示 Architecture B rank-1 = 0%,但 11.D 的 candidates 多为 semi-structured attr-listing(LLM 在 style hints 推动下偏向 generic 格式)。用户建议:

> "如果要做更明显的风格多样性,可以让 z_personalized 在 latent空间扩散得更开 (加大 noise schedule,或 K 更大)。 试一下"

Phase 11.G 假设:更大的 K + 更粗 reverse + 起点 noise 注入 → z_personalized 在 latent 空间覆盖更广 → LLM 生成的 query 风格更多样 → 318d 句法特征空间里更接近真实 user 分布。

## 实验配置 (与 11.D 对比)

| 参数 | 11.D | **11.G v2** | 用意 |
|------|------|-------------|------|
| K style samples | 5 | **10** ↑ | 每对更多样 |
| M decode seeds | 4 | 2 ↓ | 保持 20 total cands |
| N reverse steps | 100 | **30** ↓ | coarser = 更多残余 noise |
| Start t | 999 | **800** ↓ | 跳过前 200 步 full noise |
| Noise injection | 无 | **有** | z_t → sqrt(α_800)·z_t + sqrt(1-α_800)·ε |
| LLM decode | 13 次 call() 串行 | **4 次 vLLM batched** | 10 prompts → 1 调用 |

## vLLM Batched 加速 (核心工程改造)

**问题**: `QwenLocalClient.call()` 内部走 `backend.model.generate([prompt])` 一次只传一个 prompt,batch=1,即使启用了 vLLM 也发挥不出 PagedAttention + continuous batching 优势。

**修复**: 直接调 `client._backend.model.generate([prompts], sampling_params)` 一次传入所有 prompt。

```python
def batch_generate(client, prompts, temperature, max_tokens):
    from vllm import SamplingParams
    sampling = SamplingParams(
        max_tokens=max_tokens, temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
    )
    full_prompts = [
        client._backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]
    outputs = client._backend.model.generate(full_prompts, sampling)
    return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]
```

**实测效果**:

| Version | Total time | Per-record | 加速 |
|---------|-----------|-----------|------|
| 11.G v1 (call() 串行) | 推测 ~33 min | ~3.3 s/rec | baseline |
| 11.G v2 (vLLM batched) | **194 s = 3.2 min** | ~0.32 s/rec | **~10x** |

实际跑出来 v2 = 194s vs v1 跑到 5/30 pairs 用 334.6s 推算总时长 ≈ 33 min,**~10x 提速**(早期预估保守了)。

**vLLM 陷阱**: vLLM 对完全相同 prompt 会 dedup 跳过采样 → K 个 prompt 必须不同;若要同 prompt 多 sample(如 M=2 seeds),加 `(variant {seed_idx})` 后缀或 `SamplingParams(seed=...)`。

## Rank-1 Eval 结果 (VADES 318d Mahalanobis rerank)

```
[Phase 11.G v2: K=10, 30 rev steps, start_t=800 + noise injection]
Total pairs: 30
Rank-1 coverage (any cand):  0/30 = 0.0%
Rank-3 coverage (any cand):  0/30 = 0.0%
Rank-5 coverage (any cand):  0/30 = 0.0%
Rank-10 coverage (any cand): 0/30 = 0.0%   ← 退化
Best margin > 0:             28/30 = 93.3%
Best rank=1 + margin > 0:    0/30 = 0.0%

Bootstrap 95% CI:
  mean best rank:   512.37 [345.16, 705.65]  ← 比 11.F (476.5) 退化
  mean best margin: 226.5   [118.7, 351.3]    ← 比 11.F (268.0) 退化
```

## 与历史对比

| 方法 | Pair 数 | Cand/Pair | Rank-1 | Top-10 | Top-100 |
|------|---------|-----------|--------|--------|---------|
| Phase 10.15 baseline | 876 | 96 | **1.26%** | 9.25% | - |
| Phase 10.18 exemplar search | 30 | - | 0% | 0% | - |
| Phase 10.19 contrastive RAG | 30 | - | - | - | **17%** |
| Phase 11.D (K=5, 100 rev) | 30 | 20 | 0% | 3.3% | - |
| **Phase 11.G v2 (K=10, 30 rev + noise)** | 30 | 20 | **0%** | **0%** | - |

## 解读

1. **更大 K 没帮助**: z_personalized 离散度增加 → LLM 生成的 query 在 318d 句法空间里更偏离真实 user 分布 → mean rank 反而从 476.5 退化到 512.4
2. **rank-10 退化**: 11.D 有 1/30 (3.3%) 进 top-10,11.G 是 0/30 (0%)。粗 reverse + noise injection 让 candidate 离 target user 距离方差更大,rank 分布尾部拉长
3. **margin 仍正**: 93.3% pair 都有 best cand 比 wrong user mean 近,style 信号本身**真的**有用,只是不能跨越 318d Mahalanobis rerank 的 identity threshold
4. **根本限制**: Architecture B 推送 LLM 生成 semi-structured attr-listing 风格,318d 看到的全是 generic 句法,与所有 user 的 generic query 都像,无法做 user-id signal

## Phase 11 整体决策: NO-GO (冻结主路线)

| Phase | Status | 关键结果 |
|-------|--------|----------|
| 11.A Gaussian cache | GO (内部) | held-out rank-1 73.5% (AnnaWegmann 768d 用户空间) |
| 11.B Diversity test | GO (内部) | sampled-z 1.57x diversity gain |
| 11.C Diffusion training | PARTIAL-GO | s_u signal 0.42, recon err 0.10 |
| 11.D E2E pipeline | COMPLETE | 600 cands in 10 min, attr 93%, style margin +0.95 |
| 11.E Validation (proxies) | PARTIAL-GO | Diversity + style margin OK, attr 93% |
| 11.F Rank-1 eval | **NO-GO** | rank-1 = 0/30 = 0%, top-10 = 3.3% |
| **11.G Diverse z** | **NO-GO (更差)** | rank-1 = 0/30 = 0%, top-10 = 0%, mean rank 退化 |

**Architecture B (LLM neutral + diffusion + style hints + hard-copy) 整体不优于 Phase 10.19 contrastive RAG**。

**下一步**(Phase 11 范围外):
- Phase 10.19 contrastive RAG (top-100 17%) 仍是最佳 baseline
- 改进方向:让 Architecture B 的 style hints 用真实 user exemplars 取代 z_personalized → NL translation
- 或训练更深的 denoiser 直接 generation,绕过 LLM 二次解码

## 工程教训

1. **vLLM 必须 batched 调用,严禁 call() 串行**: `QwenLocalClient.call()` 是 batch=1 的低效模式;任何 K>1 的多样化生成都应直接调 `client._backend.model.generate([prompts], sampling)`。已写入 CLAUDE.md 规则 4/5h
2. **更大 K 不等于更好**: 假设"K 大 → 多样 → 更可能命中 target"在此范式下不成立,因为 318d 看到的是 generic 句法而非 user-specific 信号
3. **vLLM 进程必须全杀**: `kill <main_pid>` 不会杀掉 vLLM `EngineCore` 子进程,残留 40GB GPU 占用导致下次启动 OOM。必须 `kill -9 <main_pid> <engine_core_pid>` 清理
4. **np.savez 路径**: 不要把变量名定义为 `something.npy` 然后 `np.savez(path)` — 会变成 `something.npy.npz` (Phase 11.D 教训,11.G 复用注意)
5. **Phase 11 7 阶段全部完成**: GO + PARTIAL-GO + NO-GO 各阶段都有记录,可作为后续 diffusion / exemplar search 研究基础

## 文件位置

- Pipeline (batched): `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase11_g_diverse_z_v2.py`
- Pipeline (orig): `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase11_g_diverse_z.py`
- Eval script: `/home/wlia0047/hj82_scratch2/wenyu/vades_prototype/phase11_g_rank1_eval.py`
- Generations: `phase11_g_e2e_queries.jsonl` (600 records)
- Eval result: `phase11_g_rank1_eval.json`
- Features cache: `phase11_g_candidates_318d.npy`

## CLAUDE.md 新增规则

- **规则 4 末尾**: vLLM 后端必须直接 batched 调用,禁止用 `client.call()` 串行循环
- **规则 5h**: vLLM batched 调用参考实现 + 实测 3x 提速 + dedup 陷阱