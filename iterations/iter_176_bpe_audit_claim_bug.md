# Iteration 176 — BPE-aware Error Injection Audit Claim Bug

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: §2.2 `BPE_aware_Error_Injection` paper claim × actual Stage 05 implementation

## §A 审稿意见（尖锐批评）

### 问题: Audit claim 引用不存在的函数，与实际实现不符

**严重程度**: Minor（claim 与代码都正确地描述了"相似度感知"的注入机制，但 audit reference 的函数名是错的）

**具体批评**:

`PersoanlQuery/paper_claims_audit.py:211-220` 的 `BPE_aware_Error_Injection` claim 引用了 `PersoanlQuery/05_inject_noisy/common/apply_lambdamart_userbased_noisy.py:compute_bpe_token_diff`，但该函数在代码库中**完全不存在**：

```bash
$ grep -rn "compute_bpe_token_diff" /home/wlia0047/ar57/wenyu/PersoanlQuery/
# (no matches in any .py file)
```

iter #44 iteration record 确实记录了"新增 compute_bpe_token_diff" + "BPE_SIMILARITY_THRESHOLD=0.8"，但实际代码并没有这两个构造物。这是 **iteration record 与实际实现的 divergent state** —— iter #44 设计 spec 与最终实现不符，audit script 引用了 spec 而不是实际。

**实际实现**（`PersoanlQuery/05_inject_noisy/common/token_level_lambdamart_user_based.py`）：

- **L103** `char_similarity(s1, s2)`: 计算字符集的 Jaccard similarity
  ```python
  def char_similarity(s1: str, s2: str) -> float:
      """计算字符集相似度（Jaccard）"""
      set1 = set(s1.lower())
      set2 = set(s2.lower())
      ...
  ```
  这是**字符级 Jaccard**，不是 BPE token 重叠（后者需要 `_get_bpe_tokenizer()` + BPE encode + token overlap ratio）。

- **L173** `top_k_similarity(token, error_words, k=5)`: 返回 top-5 char-Jaccard 相似度
- **L286** `compute_token_features`: `max_sim` 作为 LambdaMART 输入特征之一（`topk_sim`，写入 L338），**不是 0.8 gating threshold**
- **L547** `bagging_fraction: 0.8`: 这个 0.8 是 LightGBM bagging fraction，与相似度阈值无关

**iter #47 audit**:
> "compute_bpe_token_diff() 失败时返回 0.5（neutral）✅"
> 这是基于 spec 的 false positive —— 该函数从未被 commit。

**对 audit 可信度的影响**:

audit script 的 code_evidence 字段如果引用不存在的函数，reviewer 用 grep 验证时会发现 mismatch，导致：
1. audit claim 整体被质疑（即使实现是正确的）
2. `paper_claims_audit.json` 与代码状态 drift

**修复**: 用实际实现的 file:line reference + 准确描述 claim 文本。

## §B 对应代码缺陷

| 审计问题 | 文件 | 行 | 缺陷描述 |
|---------|------|-----|---------|
| `BPE_aware_Error_Injection` claim | `paper_claims_audit.py` | L211-220 | code_evidence 引用不存在的 `compute_bpe_token_diff` 函数；claim 描述与实际 char-Jaccard 实现不符 |

## §C 本轮代码优化

### C.1 修改 `paper_claims_audit.py:210-220`

**before**:
```python
{
    "id": "BPE_aware_Error_Injection",
    "paper_section": "§2.2",
    "claim": "Writing error injection uses BPE-similarity-aware matching (BPE_SIMILARITY_THRESHOLD=0.8).",
    "code_evidence": [
        "PersoanlQuery/05_inject_noisy/common/apply_lambdamart_userbased_noisy.py:compute_bpe_token_diff",
    ],
    "expected_outputs": [
        "result/personal_query/07_inject_noisy/<cat>/noisy_query.json",
    ],
},
```

**after**:
```python
{
    "id": "BPE_aware_Error_Injection",
    "paper_section": "§2.2",
    "claim": "Writing error injection uses character-set Jaccard similarity (not BPE token overlap) between query token and error candidates as one LambdaMART feature (topk_sim); there is no 0.8 gating threshold — sim_score is a continuous feature, not a gate.",
    "code_evidence": [
        "PersoanlQuery/05_inject_noisy/common/token_level_lambdamart_user_based.py:char_similarity (line 103, Jaccard on character sets, NOT BPE)",
        "PersoanlQuery/05_inject_noisy/common/token_level_lambdamart_user_based.py:top_k_similarity (line 173, top-5 char-Jaccard sim)",
        "PersoanlQuery/05_inject_noisy/common/token_level_lambdamart_user_based.py:compute_token_features (line 286, uses max_sim as 'topk_sim' feature — NOT a 0.8 gate)",
    ],
    "expected_outputs": [
        "result/personal_query/07_inject_noisy/<cat>/noisy_query.json",
    ],
},
```

### C.2 修改要点

1. **id 保留** `BPE_aware_Error_Injection`：保留 audit claim ID 不变（避免 breaking downstream paper_claims_audit.json consumers + dashboard URLs）；但 claim text 改为如实描述 char-Jaccard。
2. **claim text 重写**: 明确说"character-set Jaccard similarity (not BPE token overlap)" + "one LambdaMART feature (topk_sim), no 0.8 gating"。
3. **code_evidence 三条**: L103 `char_similarity` (定义) + L173 `top_k_similarity` (top-5 ranking) + L286 `compute_token_features` (max_sim 用法)。
4. **expected_outputs**: 保留（Stage 7 noisy query 路径，iter #58 待跑；iter #176 不阻塞）。

## §D 验证

- `python3 -m py_compile PersoanlQuery/paper_claims_audit.py` ✅
- `python3 PersoanlQuery/paper_claims_audit.py --claim-id BPE_aware_Error_Injection --verbose` ✅
  - 输出新 claim text + 新 code_evidence（3 行 file:line reference）
  - 状态 `unverified`（仅因 expected_outputs 路径下文件未生成，与 code_evidence 无关）
- 全部 17 claim 总状态: `0 verified_value_match / 0 verified / 0 discrepant / 0 degenerate / 0 partial / 1 unverified / 0 blocked` (--claim-id filter 只看这条 claim)
- `paper_claims_audit.json` 已重写：下次 reviewer 跑 audit 时看到正确的 char-Jaccard 描述

## §E 旁支发现

### E.1 Stage 05 spec 漂移

iter #44 iteration record 描述的 `compute_bpe_token_diff` + `BPE_SIMILARITY_THRESHOLD=0.8` 与实际代码不一致。可能原因：
- (a) iter #44 实施过程中改了 spec 但 iteration record 没更新
- (b) iter #44 的 spec 写在某一 commit，但后续 commit (e.g. iter #46-#50 重构) 把 BPE 实现替换成 char-Jaccard 而没改 iter record

需 iter #177 audit Stage 05 完整 history：`git log --follow PersoanlQuery/05_inject_noisy/common/token_level_lambdamart_user_based.py` 找 commit chain。

### E.2 `apply_lambdamart_userbased_noisy.py` vs `token_level_lambdamart_user_based.py`

audit claim 引用 `apply_lambdamart_userbased_noisy.py`，但实际 char-Jaccard 在 `token_level_lambdamart_user_based.py`。两个文件的分工：
- `token_level_lambdamart_user_based.py` — 实际 features (char_similarity, LambdaMART features)
- `apply_lambdamart_userbased_noisy.py` — 顶层 apply/orchestration (consume token_level features)

iter #177 需 audit 这两个 file 的 import graph 确认 Stage 05 entry point 究竟是哪个。

## §F Git Commit

- iter #176: fix paper_claims_audit BPE_aware_Error_Injection claim text + code_evidence references to actual char-Jaccard implementation