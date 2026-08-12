# Iteration #71 — Stage 10 BIC/AIC blocked on missing Stage 12 outputs

**日期**: 2026-07-21
**scope**: 10_complexity_analysis / BIC/AIC 表 (paper §3.4)

## §A 状态

`compute_prior_bic_aic.py` 在 iter #69 已恢复, 但实际跑它需要 latent
representations, 来自 `10_complexity_analysis/common/train_vades_lite_sentence_latent_threshold.py`。

检查发现:

```
$ find /home/wlia0047/ar57/wenyu/result/personal_query -name '*latent*'
(empty)

$ find /home/wlia0047/ar57/wenyu/result/personal_query -path '*12_complexity*' -type d
(empty)
```

Stage 12 (`12_complexity_analysis_clause_features/<cat>/...jsonl`) 还没跑过。
Stage 10 VAE 训练依赖 Stage 12 输出:
```python
CANDIDATE_QUERY_FILE = INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
```
所以 Stage 10 → Stage 12 整条链都没跑。

## §B 阻塞拓扑

| 阶段 | 状态 | 缺什么 |
|------|------|--------|
| 11 | 未查 | review_clauses features |
| 12 | 输入 json 不存在 | Stage 11 outputs |
| 10 / BIC-AIC | 输入 json 不存在 | Stage 12 outputs |

要我重头跑 Stage 11 + 12 + 10, 这超出了 review-only loop 的范围 (单次跑 Stage 12
可能要 60min+ on CPU/GPU, 需要 LLM 之外的非 sbatch 实验)。

## §C 结论

iter #71 暂时无法完成 Stage 10 BIC/AIC。文档化此 gap, 留给未来的 iter 处理。P0 backlog 里的 "[Major] GMM prior 结论是循环论证" 仍然未通过实证解决 — 这是一个写论文时
必须谨慎的局限, 文档里必须在 §3.4 或 §5 Limitations 里明说。

## §D 候选下一个 (按 review significance)

1. **P0 [Major] Review writing style ≠ Query behavior 假设未验证 (iter #38)** — 这条需要设计新实验 (例如: review style A 的用户 vs style B 的用户, 看生成的 query 是否有统计差异)。先做 pilot: 取 2 个 review style 极端不同的用户, 比较他们的 query length / readability。
2. **P0 [Major] 用户筛选阈值 (≥20 reviews, ≥15 words) ablation (iter #38)** — 比较 n_review_in [10, 15, 20, 30] 时的 query quality distribution
3. **P0 [Major] §3.3 LLM-human eval 代码缺失 (iter #45)** — Fleiss Kappa / Spearman 实现
4. P1 SPLADE 重建 (~90min on Pet/Grocery, 让 Δ Range 表多 1 列)
5. P1 08 Δ Range 表用真的合成数据校验 (e.g. 喂一个 fake-but-clean JSON 进去, 确认 GroupMetrics 计算是对的)

下一步决定走 #1 还是 #5 还是再静下心考虑。已遵循 loop.md §8 持续运行。
