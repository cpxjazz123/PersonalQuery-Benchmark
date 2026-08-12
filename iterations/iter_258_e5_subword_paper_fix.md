# Iteration 258 — E5 Δ=11.16 subword sensitivity claim + paper fix

**日期**: 2026-07-23
**角色**: NLP/IR 专业审稿人
**scope**: iter #258 E5 BPE-aware pipeline reproducibility + paper line 137 fix

---

## §A 审稿意见

### 问题 1: E5 Δ=11.16 subword sensitivity claim 与 pipeline 行为矛盾

**严重程度**: Major
**具体批评**:
> Paper §3.2 line 137 将 E5 的 error-effect Δ 最大（Δ=11.16）归因于 "suggesting sensitivity to spelling-error-induced subword tokenization changes"。这暗示 BPE-aware 是 injection 设计的一个特性。但实际上 `bpe_aware=True` 仅在 iter #178 smoke test 中验证通过，从未在 production pipeline 中被调用。

**证据支撑**:
- Entry scripts (`05_generate_noisy_queries_by_lambdamart_userbased_Baby_Products.py` 等) 调用 `main("Baby_Products")`，无 `--bpe-aware` 参数
- `apply_lambdamart_userbased_noisy.py` 中 `bpe_aware=False` 默认值（line 376, 487, 566）
- 所有历史 noisy queries 均通过 char-set Jaccard only 生成（无 BPE tokenization gating）
- Paper claim 暗示 BPE-aware 是 E5 Δ=11.16 的解释机制，但该机制从未在 pipeline 中启用

---

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| E5 subword claim 缺乏 pipeline 证据 | `05_inject_noisy/05_generate_noisy_queries_by_lambdamart_userbased_*.py` | Entry scripts 调用 `main(cat)` 无 `bpe_aware=True` |
| E5 subword claim 缺乏 pipeline 证据 | `05_inject_noisy/common/apply_lambdamart_userbased_noisy.py:566` | `main()` 默认 `bpe_aware=False` |

---

## §C 本轮代码优化

**Paper 文本修正**:
- Paper line 137 E5 描述从：
  `"suggesting sensitivity to spelling-error-induced subword tokenization changes"`
  改为：
  `"suggesting sensitivity to character-level writing errors and their interaction with subword tokenization boundaries (see iter #258 for pipeline reproducibility caveat regarding the BPE-aware injection flag)"`
- 新文本保留了 E5 对 subword tokenization 敏感性的 observation，但明确了 pipeline reproducibility caveat

**paper_claims_audit.json 更新**:
- Table 1 Δ Range degenerate claim (index 1) audit_note 追加 iter #258 说明

---

## §D 验证

- Paper line 137 E5 text: ✅ 已修正为 qualified 表述 + iter #258 caveat reference
- Audit JSON updated: ✅ generated_at 更新
- `python3 -m py_compile` on paper: N/A (markdown)

---

## §E Git Commit

- `git commit -m "iter #258: qualify E5 subword sensitivity claim with BPE-aware pipeline caveat"`

---

## §F 剩余审稿意见（待后续迭代）

1. **BPE-aware pipeline 端到端验证**：需 rerun Stage 05+06+07 with `--bpe-aware` 验证 E5 Δ Range 是否真的有显著差异
2. **Claim 2 状态升级**：Baby+Pet Stage 07 已验证 noisy Hit@10 退化，可升级为 verified（需更多 pairs 统计支撑）
3. **Δ Range degenerate 解除**：需 query file 提供 word_count_bucket 元数据，使 Stage 08 可计算跨复杂度分组波动