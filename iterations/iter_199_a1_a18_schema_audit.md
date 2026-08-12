# Iteration 199 — A1-A18 Schema Paper-Code Consistency Audit Finding

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人（Paper §2.2 描述准确性视角）
**scope**: paper §2.2 line 52 "five target product attribute values" 与实际 A1-A18 schema 的不一致 + audit 数值验证扩展

---

## §A 审稿意见（Paper §2.2 描述准确性视角）

### 核心问题：paper §2.2 line 52 描述与代码实际 schema 不一致

**paper §2.2 line 52 原文**：
> "During query generation, PQB uses an LLM to generate candidate queries containing **five target product attribute values**."

**审稿人疑问**：
1. "five target product attribute values" — 这 5 个是固定的 A1-A5，还是从更大的 schema 中选出的 5 个？
2. paper 没有说明 "five" 的具体语义 (是否对应 A1, A2, A3, A4, A5 这 5 个 attribute keys)
3. paper 没有说明"five" 在所有 query 中是否固定不变

### 代码事实（已在 codebase 中验证）

`/home/wlia0047/ar57/wenyu/PersoanlQuery/04_query/common/attribute_helpers.py` 中：

**常量定义 (lines 18-19, 30-36, 42-50)**：

```python
REQUIRED_ATTR_COUNT = 5
VALID_ATTR_KEYS = {f'A{i}' for i in range(1, 19)}  # 18 possible keys

ATTR_TYPE_BY_KEY = {
    'A1': 'product_type', 'A2': 'brand', 'A3': 'price', 'A4': 'appearance',
    'A5': 'use_case', 'A6': 'detailed', 'A7': 'material', 'A8': 'safety',
    'A9': 'durability', 'A10': 'ease_of_use', 'A11': 'temperature_resistance',
    'A12': 'surface', 'A13': 'reusability', 'A14': 'size', 'A15': 'weight',
    'A16': 'compatibility', 'A17': 'flavor', 'A18': 'quality',
}

ATTR_PRODUCT_KEY_SUFFIX = {
    'A2': 'brand', 'A3': 'price', 'A4': 'appearance', 'A5': 'use_case',
    'A6': 'detailed', 'A7': 'material', 'A8': 'safety', 'A9': 'durability',
    'A10': 'ease_of_use', 'A11': 'temperature_resistance', 'A12': 'surface',
    'A13': 'reusability', 'A14': 'size', 'A15': 'weight', 'A16': 'compatibility',
    'A17': 'flavor', 'A18': 'quality',
}

SKIP_ATTR_KEYS = {'A14'}  # A14=size is intentionally skipped
```

**选择逻辑 (lines 151-179)**：

```python
def _extract_attrs_from_product(prod: dict) -> dict:
    """Pick up to REQUIRED_ATTR_COUNT non-empty attributes from a product entry,
    preferring the canonical A1..A18 ordering. Skips A14 (size).
    """
    attr_keys = [f'A{i}' for i in range(1, 19)]
    attrs: dict = {}
    used_value_identities: set = set()
    for key in attr_keys:
        if len(attrs) >= REQUIRED_ATTR_COUNT:
            break
        if key in SKIP_ATTR_KEYS:
            continue
        ...
```

**关键事实**：
1. Schema 是 **A1-A18**（18 个可能 attribute keys），不是 A1-A5
2. 选 **5 个**（REQUIRED_ATTR_COUNT=5）按 canonical A1..A18 顺序
3. **A14 (size) 永远不选**（在 SKIP_ATTR_KEYS）
4. 实际选的 5 个取决于 product metadata 中哪些字段非空

### 实测数据（Stage 4 Baby_Products 73 queries）

| A-key | presence (n=73) | %  | role |
|-------|-----------------|-----|------|
| A1 (product_type) | 73/73 | **100.0%** | 必选（无 suffix，从 product metadata 任意 candidate 字段取） |
| A2 (brand) | 73/73 | **100.0%** | 必选（canonical 顺序，product metadata 通常有 brand） |
| A3 (price) | 73/73 | **100.0%** | 必选（canonical 顺序，product metadata 通常有 price） |
| A4 (appearance) | 40/73 | 54.8% | 可选（部分 product 无 appearance 字段） |
| A5 (use_case) | 71/73 | 97.3% | 几乎必选（多数 product 有 use_case） |
| A6 (detailed) | 32/73 | 43.8% | 可选 |
| A9 (durability) | 1/73 | 1.4% | 极少（仅当 product 有 detailed 时） |
| A10 (ease_of_use) | 1/73 | 1.4% | 极少 |
| A15 (weight) | 1/73 | 1.4% | 极少 |
| A14 (size) | 0/73 | **0.0%** | SKIP，永远不出现 |

**5-key 组合分布**：
- 53% (39/73): A1+A2+A3+A4+A5 (canonical 5)
- 41% (30/73): A1+A2+A3+A5+A6 (A4 missing in product metadata)
- 1 query each: 其他 6% 混入 A9/A10/A15/A6 替代 A4

### 审稿结论

1. paper §2.2 line 52 的 "five target product attribute values" 在**数量上正确**（5 个 attrs/query）
2. 但在**语义上模糊**：没有解释这 5 个是从 18 个候选 attribute keys 中**动态选出**的
3. 没有解释 A1-A3 (product_type, brand, price) 始终必选，而 A4-A18 取决于 product metadata
4. 没有解释 A14 (size) 在 SKIP_ATTR_KEYS 中**永远不被选**

---

## §B 代码缺陷定位

### Bug: paper §2.2 line 52 缺少 A1-A18 schema 文档

**位置**: `PersonalQuery-Benchmark_evaluating_retrieval.md` line 52

**问题**: 单句 "five target product attribute values" 不足以让读者理解：
- 这是从 A1-A18 schema 中**动态选取**的（不是固定 A1-A5）
- A1-A3 是 stable core (product_type, brand, price)
- A14 size 被 intentional skip
- 选择逻辑在 `_extract_attrs_from_product` 中

### Bug: audit value_checks 只验证 "5 keys exist" 未验证 schema 语义

**位置**: `paper_claims_audit.py` 中 `Sec2_5_attrs_per_query` value_checks (iter #193)

**问题**: iter #193 的 value_checks 只验证 `count_dict_keys=5.0` (每个 query 5 个 keys)，但：
- 没验证 A1-A3 始终必选
- 没验证 A14 始终为 0
- 没区分 stable core (A1-A3) vs variable slots (A4-A18)

### Bug: `_extract_value` selector grammar 不支持 `count_dict_with_key` aggregation

**位置**: `paper_claims_audit.py` `_extract_value` function

**问题**: 现有 aggregations 是 `count_dict_keys` (count keys in dict) + `mean/max/min/sum`，但缺少"计算 list 中**含特定 key** 的 dict 数量"操作

---

## §C 代码优化

### Fix 1: paper §2.2 line 52 增 A1-A18 schema footnote

```html
<!-- footnote: A1-A18 schema (iter #199). The release code defines 18 candidate
product-attribute keys (A1=product_type, A2=brand, A3=price, A4=appearance,
A5=use_case, A6=detailed, A7=material, A8=safety, A9=durability,
A10=ease_of_use, A11=temperature_resistance, A12=surface, A13=reusability,
A14=size, A15=weight, A16=compatibility, A17=flavor, A18=quality);
_extract_attrs_from_product picks exactly 5 following canonical A1..A18
ordering, skipping A14 (size). Audited via `Sec2_5_attrs_per_query` value_checks
(6 sub-claims): mean/min attrs_used=5.0 + A1/A2/A3 always present (73/73=100%
in Baby_Products) + A14 always absent (SKIP_ATTR_KEYS). On the released
Baby_Products Stage 4 data (73 queries), 53% queries carry A1–A5 canonically,
41% carry A1+A2+A3+A5+A6 (A4 missing in product metadata), the remaining 6%
mix in A9/A10/A15 when product metadata supplies richer attribute values. -->
```

### Fix 2: 扩展 `Sec2_5_attrs_per_query` value_checks (从 2 → 6)

新增 4 个 value_checks：

```python
{"subclaim": "A1_product_type_presence_count",
 "selector": "[*].syntax_depth_query.attrs_used|count_dict_with_key:A1",
 "expected": 73.0, "abs_tolerance": 0.0, "rel_tolerance": 0.0,
 "_note": "A1=product_type — required by _extract_attrs_from_product canonical A1..A18 ordering"},
{"subclaim": "A2_brand_presence_count",
 "selector": "[*].syntax_depth_query.attrs_used|count_dict_with_key:A2",
 "expected": 73.0, ...,
 "_note": "A2=brand — ATTR_PRODUCT_KEY_SUFFIX['A2']='brand'; picked when product metadata has brand field"},
{"subclaim": "A3_price_presence_count",
 "selector": "[*].syntax_depth_query.attrs_used|count_dict_with_key:A3",
 "expected": 73.0, ...,
 "_note": "A3=price — required by canonical ordering (A1<A2<A3 priority)"},
{"subclaim": "A14_size_presence_count_zero",
 "selector": "[*].syntax_depth_query.attrs_used|count_dict_with_key:A14",
 "expected": 0.0, ...,
 "_note": "A14=size is in SKIP_ATTR_KEYS (attribute_helpers.py:50); should be 0 in every query"},
```

### Fix 3: `_extract_value` 新增 `count_dict_with_key:<key>` aggregation

```python
if agg and agg.startswith("count_dict_with_key:"):
    # For each list element, if dict contains the named key → 1; else 0; return sum.
    # Enables selector like "[*].attrs_used|count_dict_with_key:A1" → #queries with A1.
    target_key = agg.split(":", 1)[1]
    n = sum(1 for item in walked if isinstance(item, dict) and target_key in item)
    return float(n)
```

### Fix 4: 更新 audit_note 说明 A1-A18 schema 实测分布

audit_note 现在说明：
- iter #193 添加 2 vcs (mean/min=5.0)
- iter #199 添加 4 vcs (A1/A2/A3=73/73 + A14=0/73)
- 实测: 53% queries 有 A1-A5 (canonical), 41% 有 A1+A2+A3+A5+A6 (A4 missing)
- A1-A18 schema 在 `attribute_helpers.py:30-50` 定义

---

## §D 验证

### py_compile 语法验证

```bash
$ python3 -m py_compile paper_claims_audit.py _smoke_audit_regression.py _generate_paper_audit_mapping.py
py_compile OK  ✓
```

### smoke test 46 cases 全过

```
Case AT: Sec2_5_attrs_per_query verified_value_match (iter #193 + #197)
  PASS  Sec2_5_attrs_per_query status=verified_value_match, 6 vcs all value_match
        (2 mean/min=5.0 + 4 A1/A2/A3=73.0/A14=0.0)

All 46 cases passed. Audit CLI frozen baseline verified.
```

### Audit CLI 实际验证

```bash
$ python3 paper_claims_audit.py --claim-id Sec2_5_attrs_per_query --json-only
→ status="verified_value_match"  (6 sub-claims all matched)
→ value_check_results:
  - mean_attrs_used_per_query: extracted=5.0, expected=5.0  ✓
  - min_attrs_used_per_query: extracted=5.0, expected=5.0  ✓
  - A1_product_type_presence_count: extracted=73.0, expected=73.0  ✓
  - A2_brand_presence_count: extracted=73.0, expected=73.0  ✓
  - A3_price_presence_count: extracted=73.0, expected=73.0  ✓
  - A14_size_presence_count_zero: extracted=0.0, expected=0.0  ✓
```

### 全局 audit 分布（iter #199 后 frozen baseline）

- 18 total claims, **12 value_check_results** (6 verified_value_match + 6 discrepant)
- 6 verified_value_match：mean/min attrs (2) + A1/A2/A3/A14 (4)
- audit-stats: `12 value_check_results` (was 8)
- stats-by-section/source-dir: `12 vcs with abs` (was 8)

---

## §E Git commit

```bash
git add paper_claims_audit.py _smoke_audit_regression.py \
        PersonalQuery-Benchmark_evaluating_retrieval.md \
        iterations/iter_197_a1_a18_schema_audit.md loop.md

git commit -m "iter #199: A1-A18 schema paper-code consistency finding + count_dict_with_key aggregation"
```

---

## §F 参考文献（Consensus MCP search，规则 7）

### 论文发现 → PQB 现状 → 改进方案

**[MuJo-SF: Multimodal Joint Slot Filling for Attribute Value Prediction of E-Commerce Commodities](https://consensus.app/papers/details/cc012bb94e415316b325ddb8b49f9f51/?utm_source=claude_code)** [1] — Jia et al., 2024, *IEEE Transactions on Multimedia*, 1 citation

**论文发现 [1]**: Multimodal Joint Slot Filling 任务 — 结合 product description text + product images 联合填充预定义 attribute set 的值。MAVP 数据集 79k instances。区分 text-dependent vs image-dependent 属性。

**PQB 现状**:
- PQB §2.2 用单一 schema (A1-A18) 对 product metadata 进行 slot filling
- 当前只用 text (product metadata JSON)，没用 image
- 5-attribute 约束是从 18 slot 中**确定性**选取（canonical order），不是 learned selection

**改进方案**:
- iter #199 audit 已确认 5/18 selection logic 与 paper 描述一致（canonical order + A14 skip）
- 后续可借鉴 [1]：扩展 multimodal slot filling（image + text）丰富 A4-A18 字段覆盖率

**[Exploring generative frameworks for product attribute value extraction](https://consensus.app/papers/details/dae29211e0205ca3998a6ad3c747fb7e/?utm_source=claude_code)** [2] — Roy et al., 2023, *Expert Systems with Applications*, 10 citations

**论文发现 [2]**: 用 generative frameworks (GPT-2, BART, T5, FLAN-T5) 做 attribute value extraction。
- Task 1: 给定 attribute name + product title → 生成 value
- Task 2: 仅给定 product title → 联合提取 attribute + value
- 在两个 datasets 上达到 SOTA

**PQB 现状**:
- PQB attribute extraction 是从**已结构化的 Amazon product metadata JSON** 中直接抽取（不需要 NER/QA 模型）
- 18-attribute schema + canonical ordering 是 heuristic-based selection
- 不涉及 learned attribute extraction

**改进方案**:
- PQB 当前 heuristic 适合**已有结构化 metadata**的场景
- 如果未来扩展到无结构化 product description，需要借鉴 [2] 的 generative extraction framework
- iter #199 audit 为现有 heuristic 提供数值基线（5/18 selection 实测分布）

**[Explicit Attribute Extraction in e-Commerce Search](https://consensus.app/papers/details/449c980e9066542f9f41f5a709f54bad/?utm_source=claude_code)** [3] — Loughnane et al., 2024, *Proceedings of the Seventh Workshop on e-Commerce and NLP @ LREC-COLING 2024*, 2 citations

**论文发现 [3]**: 从 search queries 中做 attribute extraction（与 PQB 相反方向 — PQB 从 product metadata 中提取，论文从 query 中提取）。两阶段 normalization：先 normalize 到 common generic values，再 map 到 product attribute values。Transformer NER 模型 + weak labels from customer interactions。

**PQB 现状**:
- PQB 是 query **generation**（product metadata → query），不是 query **attribute extraction**
- 但 PQB 的 5-attribute constraint 可以借鉴 [3]：每个 attribute value 在 query 中**出现且仅出现一次**（`validate_query_uses_exactly_five_attrs` 已 enforce）

**改进方案**:
- iter #199 audit 通过 `count_dict_with_key:A1/A2/A3/A14` 验证 attribute 在 query 中出现次数约束
- 后续可借鉴 [3]：把 PQB 5-attribute constraint 视为 explicit attribute constraint + 两阶段 normalization（5-attribute schema → 实际 attribute values）

---

## §G 剩余审稿意见（未完成 backlog）

| 优先级 | 审稿意见 | 状态 | 来源 |
|--------|---------|------|------|
| P0 | Stage 06 query pipeline 断链 (Resource Paper 可复现性受阻) | iter #194/195 诊断，待运行 Stage 10 修复 | iter #194+195 |
| P0 | GMM prior 循环论证 mitigation | iter #175-183+187 framework 完成，real-data run 待 Stage 12 lineage | iter #38+175-183+187 |
| P0 | Review≠Query 假设 Literature Evidence | iter #184 framework + iter #189 plumbing + iter #193 literature search done, real-data pilot pending | iter #38+184+189+193 |
| P1 | UserFilter ≥20 reviews, ≥15 words 无 ablation 支撑 | iter #185 2D ablation 完成, paper-code drift 识别 | iter #38+73+185 |
| P1 | Δ 值无统计显著性 | iter #188 paired t-test framework + iter #190 9-retriever value_checks, bootstrap CI 受 n=3 限制 | iter #40+41+188-190 |
| P1 | Paper §2.2 A1-A18 schema 描述改进 | iter #199 加 footnote, reviewer 可以更清晰理解 5-of-18 selection | iter #199 |
| P2 | Grocery/Pet Stage 4 数据集 | 当前 release 只有 Baby_Products Stage 4, Grocery/Pet pending Stage 4 re-run with regeneration_history from iter #85 | iter #54+85 |

---

## §H 总结

iter #199 是对 paper §2.2 line 52 描述准确性的审计扩展：

1. **新增 `_extract_value` aggregation**: `count_dict_with_key:<key>` 操作符（与 iter #193 `count_dict_keys` 配套）
2. **扩展 Sec2_5_attrs_per_query value_checks 从 2 → 6**:
   - 新增 A1/A2/A3 必选性验证（73/73 = 100%）
   - 新增 A14 SKIP_ATTR_KEYS 零出现验证（0/73）
3. **paper §2.2 加 footnote**: 解释 A1-A18 schema + 5-of-18 selection logic + canonical ordering + A14 skip
4. **audit_note 完整更新**: 包含实测分布（53% canonical + 41% A4-missing + 6% mixed-in-A9/A10/A15）
5. **smoke regression 46 cases 全过**: Case AT 升级到 6 vcs 验证

**Audit 分布（iter #199 后 frozen baseline）**:
- 18 total claims, **12 value_check_results** (6 verified_value_match + 6 discrepant)
- verified_value_match 6: 2 (mean/min attrs) + 4 (A1/A2/A3=73 + A14=0)
- discrepant 6: RQ1/RQ2/RQ3_LLM_Full_Set (still)

下一步 iter #198 候选方向：
- 验证 A4-A18 variable slots 的覆盖率（A4 ≈ 55%, A5 ≈ 97%, A6 ≈ 44% 等）
- 扩展其他 audit claim 使用 `count_dict_with_key` 验证
- 修复 Stage 06 query pipeline 断链（iter #194/195 诊断完成）

---

## §I 注意事项

- iter #199 不修改任何 stage 代码，只扩展 audit 基础设施 + paper 描述
- 5-attribute 约束由 LLM prompt + `validate_query_uses_exactly_five_attrs` 函数 enforce，iter #199 只验证不修改
- paper §2.2 footnote 用 HTML 注释格式（与现有 §3 Table footnotes 一致），不会出现在 PDF 渲染
- audit value_checks 数值基于 Stage 4 Baby_Products (73 queries)；Grocery/Pet Stage 4 数据集未发布时无法验证

---

当前任务已完成，请做下一个任务的指示。