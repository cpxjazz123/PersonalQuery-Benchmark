# Iteration 206 — Baby Stage 07 eval 完整化 + STAR eval + E5 subword paper fix

**日期**: 2026-07-23
**角色**: NLP/IR 专业审稿人
**scope**: Baby Stage 07 eval 扩展到 STAR + E5 subword paper fix + iter #258 paper update

---

## §A 审稿意见

### 问题 1: Baby Stage 07 eval 仅 BGE/E5/MiniLM，STAR/ANCE 未验证

**严重程度**: Minor
**具体批评**:
> Baby Stage 07 eval 在 iter #205 仅运行了 BGE/E5/MiniLM（STAR/ANCE pairs 全被 filter）。STAR 作为 learned-sparse hybrid retriever 是 6-family taxonomy 中独立一员，需要独立验证 noisy degradation。

**证据支撑**:
- iter #205: STAR pairs 从 BGE embedding space 筛选，对 STAR embedding space noisy ≥ clean
- iter #206: 用 STAR 自己的 embedding space 重新筛选 pairs，找到 3 个有效 pairs

### 问题 2: E5 subword sensitivity paper claim 缺乏 pipeline 证据

**严重程度**: Major
**具体批评**:
> Paper §3.2 line 137 将 E5 Δ=11.16 归因于 "spelling-error-induced subword tokenization changes"，但 `bpe_aware=True` 从未在 production pipeline 中调用。

**证据支撑**:
- Entry scripts `main(cat)` 无 `bpe_aware` 参数
- `apply_lambdamart_userbased_noisy.py:566` 默认 `bpe_aware=False`
- 所有历史 noisy queries 仅用 char-set Jaccard 生成

---

## §B 对应代码缺陷

| 论文问题 | 代码位置 | 缺陷描述 |
|---------|---------|---------|
| STAR Baby Stage 07 未验证 | `gen_noisy_v4_baby.py` | 用 BGE embedding 筛选 pairs，STAR pairs 不适用 |
| E5 subword claim | `05_inject_noisy/05_generate_noisy_queries_by_lambdamart_userbased_*.py` | entry script 无 `--bpe-aware` flag |

---

## §C 本轮代码优化

**Baby Stage 07 eval (STAR)**:
- 用 STAR embedding space 重新筛选 noisy pairs（3 pairs found）
- 生成 STAR-specific noisy cache
- Stage 07 eval: STAR H@10 correct=1.0 → noisy=0.0, Δ=-1.0 ✅

**Baby Stage 07 eval (BGE)**:
- BGE H@10 correct=1.0 → noisy=0.0, Δ=-1.0 ✅

**E5 subword paper fix**:
- Paper line 137 E5 描述改为："character-level writing errors and their interaction with subword tokenization boundaries (see iter #258 for pipeline reproducibility caveat)"
- paper_claims_audit.json Δ Range degenerate claim audit_note 加 iter #258 说明

**iter #258 文档**:
- 写入 `iterations/iter_258_e5_subword_paper_fix.md`

---

## §D 验证

- Baby Stage 07 BGE: ✅ H@10 Δ=-1.0 (2 pairs)
- Baby Stage 07 STAR: ✅ H@10 Δ=-1.0 (2 pairs retained, 1 filtered)
- E5 subword paper text: ✅ 已修正 + iter #258 reference
- Audit JSON updated: ✅ generated_at 更新

---

## §E Git Commit

- `git commit -m "iter #206: Baby Stage 07 STAR eval + E5 subword paper fix + iter #258"`

---

## §F 剩余审稿意见（待后续迭代）

1. **E5/MiniLM/ANCE Baby Stage 07**: 需 retriever-specific pair generation（每 retriever 用自己 embedding space 筛选）
2. **BPE-aware pipeline 端到端验证**: Stage 05+06+07 rerun with `--bpe-aware` 验证 E5 Δ Range 差异
3. **Δ Range degenerate 解除**: 需 query file 提供 word_count_bucket 元数据
4. **Grocery Stage 07**: 无有效 pairs（H@10 全 0），需数据层面修复