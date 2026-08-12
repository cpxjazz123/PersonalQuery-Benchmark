# Iteration 177 — Stage 05 iter #44 Phantom BPE Implementation

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: Stage 05 iter #44 BPE-aware error injection spec drift — git history forensic

## §A 审稿意见（尖锐批评）

### 问题: iter #44 是 phantom iteration record — 详细描述的 BPE 代码从未 commit

**严重程度**: Major（影响 E5 Δ=11.16 subword 敏感性这一 P0 review concern 的实证状态）

**具体批评**:

iter #44 iteration record (`iterations/iter_44_bpe_aware_error_injection.md`) 详细描述了一个"实现"：
- 新增 `_get_bpe_tokenizer()` 函数（lazy-load E5 tokenizer）
- 新增 `compute_bpe_token_diff()` 函数（BPE Jaccard similarity）
- `process_batch` 新增 `bpe_aware` 参数
- `BPE_SIMILARITY_THRESHOLD=0.8` gating

但 **git history 中不存在任何 commit 添加了这些代码**：

```bash
$ git log --all --oneline --follow PersoanlQuery/05_inject_noisy/common/token_level_lambdamart_user_based.py
066f091 iter #33: Stage 05 common dead code清理...
a5ba958 refactor(stages): renumber stage directories for contiguous 00-14
4fdd6c3 feat: add new noise injection methods and user-based lambdamart
4d745f8 feat(writing_analysis): add multi-threading and checkpoint to error extraction

$ git log --all --oneline --follow PersoanlQuery/05_inject_noisy/common/apply_lambdamart_userbased_noisy.py
066f091 iter #33: Stage 05 common dead code清理...
a5ba958 refactor(stages): renumber stage directories for contiguous 00-14
4fdd6c3 feat: add new noise injection methods and user-based lambdamart
```

两文件总共 4 个 distinct commits。iter #44 commit 时间是 2026-07-20。但 iter #44 之后没有任何 commit 修改这两个文件 — 4fdd6c3 (initial commit) 之后只有 a5ba958 (rename) 和 066f091 (iter #33 dead code 清理, 2026-07-20 17:24)。

**初始 commit (4fdd6c3, 2026-06-01)** 已经在 `token_level_lambdamart_user_based.py:103` 定义了 `char_similarity`：

```python
def char_similarity(s1: str, s2: str) -> float:
    """计算字符集相似度（Jaccard）"""
    set1 = set(s1.lower())
    set2 = set(s2.lower())
    ...
```

这是字符集 Jaccard，**不是** BPE token overlap。从 initial commit 至今 (4 commits 跨度 ~46 天) 这个函数完全没变过。

**iter #44 phantom spec 与实际实现的差异**:

| spec 描述（iter #44） | 实际实现 | 差异 |
|----------------------|---------|------|
| `_get_bpe_tokenizer()` lazy-load E5 tokenizer | ❌ 不存在 | 整个 BPE 加载机制都没实现 |
| `compute_bpe_token_diff(original, corrected)` | ❌ 不存在 | 没有任何 BPE 函数 |
| `BPE_SIMILARITY_THRESHOLD = 0.8` | ❌ 不存在 | 文件中 0.8 只在 LightGBM `bagging_fraction` (L547) |
| `process_batch(bpe_aware=True)` 参数 | ❌ 不存在 | 无 bpe_aware 参数 |
| E5 subword 敏感性模拟 | ❌ 未实现 | 没有任何 subword-level 注入 |
| char-Jaccard 作为 LambdaMART feature | ✅ `char_similarity` L103 + `top_k_similarity` L173 + `compute_token_features` L286 `max_sim` → `topk_sim` L338 | 实现，但作为 continuous feature 不是 gating |

**对论文 E5 Δ=11.16 subword sensitivity 论断的影响**:

iter #40 提出「E5 Δ=11.16 反映 subword 敏感性，但代码在完整单词级别注入」，iter #44 spec 设计 BPE-aware injection 来补救，loop.md backlog line 338 标注 "✅ iter #44: 完全实现"。

**实证**: iter #44 没有真正实现 BPE-aware injection。E5 Δ=11.16 的 subword-sensitivity claim 没有 Stage 05 subword-level 注入的代码支撑。Stage 05 仍然在 word-level 注入（与 char-Jaccard similarity 无关，char-Jaccard 只决定哪个候选错误被 LambdaMART 选中，不影响注入的形态）。

**结论**: 这是 **method-level claim vs implementation gap** —— paper §3.2 描述了「writing error 注入应反映 subword-level 敏感性」，但 release pipeline 完全没实现这个 mechanism。Audit claim `BPE_aware_Error_Injection` (经 iter #176 修复) 现在如实描述 char-Jaccard, 揭示了 E5 Δ=11.16 的实证 infra 不完整。

## §B 对应代码缺陷

| 问题 | 文件 | 行/范围 | 缺陷描述 |
|------|------|---------|---------|
| phantom iteration record | `iterations/iter_44_bpe_aware_error_injection.md` | 全文 | 描述的 BPE-aware 注入从未 commit；Stage 05 实际是 char-Jaccard-only LambdaMART feature |
| stale backlog claim | `loop.md` | L338 | 「✅ iter #44: E5 subword 敏感性 完全实现」是 stale claim；iter #177 后应改为 ❌ 未实现 |
| E5 subword-sensitivity infra missing | `05_inject_noisy/common/token_level_lambdamart_user_based.py` | 全文 | 没有任何 subword-level error injection 机制（既不是 BPE-aware 也不是 character-edit-aware） |

## §C 本轮代码优化

### C.1 修正 loop.md backlog

更新 `loop.md` L338 把 ✅ 改为 ❌ 未实现，引用 iter #177。

### C.2 写 iteration record 记录 phantom spec 现象

写 `iterations/iter_177_stage05_phantom_bpe_iteration.md` 记录 git forensic findings。

### C.3 不重新实现 BPE-aware 注入（避免 scope creep）

iter #177 是 audit/记录性质，**不**重新实施 BPE-aware injection。如果要真正实现，需要：
- 加 `transformers.AutoTokenizer.from_pretrained('intfloat/e5-base-v2')` 依赖
- `_get_bpe_tokenizer()` lazy loader
- `compute_bpe_token_diff()` BPE Jaccard
- `process_batch(bpe_aware=True)` parameter
- 在 LambdaMART score 上加 BPE-aware 加权

但这是新 scope（不是 audit），需单独 iter（如 iter #178）。当前 backlog P1「E5 subword 敏感场景未模拟」需要重新激活。

## §D 验证

- `git log --all --oneline --follow token_level_lambdamart_user_based.py` → 4 commits, 无 BPE
- `git log --all --oneline --follow apply_lambdamart_userbased_noisy.py` → 3 commits, 无 BPE
- `git show 4fdd6c3:.../token_level_lambdamart_user_based.py | grep "def char_similarity"` → ✅ initial commit 已有 char_similarity
- `grep -rn "compute_bpe_token_diff" /home/wlia0047/ar57/wenyu/PersoanlQuery/` → 仅 paper_claims_audit.json + iterations/iter_44 + iter_47 record (全部是 audit/record 引用, 无代码)
- `grep -rn "BPE_SIMILARITY_THRESHOLD\|bpe_aware\|_get_bpe_tokenizer" /home/wlia0047/ar57/wenyu/PersoanlQuery/05_inject_noisy/` → 无 match
- `python3 -m py_compile PersoanlQuery/05_inject_noisy/common/token_level_lambdamart_user_based.py` → ✅ (no changes needed)

## §E 对 §11 backlog 的影响

### E.1 P1「E5 subword 敏感场景未模拟」重新激活

iter #40 提出的 review concern「E5 Δ=11.16 反映 subword 敏感性，但代码在完整单词级别注入」经过 iter #44 (phantom) 和 iter #177 (audit) 仍未真正解决。需 iter #178 实现真正的 subword-aware error injection（按 iter #44 spec 或改进版本）。

修改 backlog (loop.md §11):
```
| ~~P1~~ | **~~E5 subword 敏感场景未模拟~~** ❌ iter #44 phantom, iter #177 审计发现 spec 从未 commit | iter #36 + #40 | ❌ 未实现 (iter #177 audit findings) |
```

替换原来的 "✅ iter #44: 完全实现"。

### E.2 iter #44 iteration record 加 caveat

`iterations/iter_44_bpe_aware_error_injection.md` 末尾加一段 "⚠️ Caveat (iter #177)" 说明本 iter 描述的实现从未 commit，audit findings 见 iter #177。

## §F Git Commit

- iter #177: audit Stage 05 spec drift — iter #44 BPE-aware injection never committed; char-Jaccard is from initial commit (4fdd6c3); update loop.md §11 backlog P1 status