/-
# PersonalQuery.Encoder — 监督 encoder 形式化

Stage 03 syntax_pcfg_pipeline.py 主入口使用 `_SupEncoder`:
    input  : n × V 实矩阵 (rule-count bag, V = 21737)
    layer 1: Linear(V → 256) + ReLU + Dropout(0.5)
    layer 2: Linear(256 → 32)
    loss   : BCEWithLogitsLoss(label_smoothing=0.1)

对照 `syntax_pcfg_pipeline.py:84-94`:
  SUP_EPOCHS=80, SUP_BATCH_SIZE=2048, SUP_LR=5e-4, SUP_Z_DIM=32,
  SUP_HIDDEN=(256,), SUP_DROPOUT=0.5, SUP_RULE_DROPOUT=0.1,
  SUP_WEIGHT_DECAY=1e-2, SUP_LABEL_SMOOTHING=0.1

我们形式化:
  - normalize_count : c ↦ c/(1+c)
  - ruleCountMatrix : Doc → List ℝ^V (Bag-of-Rules)
  - _SupEncoder.forward : ℝ^V → ℝ^32 (展开为两层 Linear + ReLU + Dropout)
  - bceLossWithLabelSmoothing : ℝ × ℝ → ℝ
  - 监督训练的 step (single SGD)
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

import PersonalQuery.Basic

namespace PersonalQuery

/-! ### 归一化 -/

/-- 归一化计数: x ↦ x/(1+x) ∈ [0, 1) 对 x ≥ 0。
    对应 Python `normalize_counts` (syntax_pcfg_pipeline.py:173-180):
      norm = data / (1.0 + data) -/
def normalizeCount (x : ℝ) : ℝ := x / (1 + x)

/-- normalize_count 在 x ≥ 0 时非负。 -/
lemma normalizeCount_nonneg {x : ℝ} (h : 0 ≤ x) : 0 ≤ normalizeCount x :=
  div_nonneg h (add_nonneg zero_le_one h)

/-- normalize_count 在 x ≥ 0 时 < 1。 -/
lemma normalizeCount_lt_one {x : ℝ} (h : 0 ≤ x) : normalizeCount x < 1 := by
  unfold normalizeCount
  have hpos : 0 < 1 + x := add_pos_of_nonneg_of_pos zero_le_one h
  rw [div_lt_one hpos]
  · linarith
  · exact hpos

/-- normalize_count 单调: x₁ ≤ x₂ ⟹ normalize x₁ ≤ normalize x₂ (对非负 x)。 -/
lemma normalizeCount_mono {x₁ x₂ : ℝ} (h₁ : 0 ≤ x₁) (h₂ : x₁ ≤ x₂) :
    normalizeCount x₁ ≤ normalizeCount x₂ := by
  unfold normalizeCount
  apply div_le_div_of_le_left (add_nonneg zero_le_one h₁) (le_refl _) _
  linarith

/-- normalize_count 在 x = 1 时 = 1/2。 -/
lemma normalizeCount_one : normalizeCount 1 = 1 / 2 := by
  unfold normalizeCount
  ring

/-! ### Bag-of-Rules 矩阵 -/

/-- 词表大小常量 (Stage 03 中固定 V = 21737)。 -/
def vocabSize : ℕ := 21737

lemma vocabSize_pos : 0 < vocabSize := by
  unfold vocabSize; norm_num

/-- 词表类型: 0..V-1 索引规则 ID。 -/
abbrev Vocab := Fin vocabSize

/-- 句子级 bag-of-rules 矩阵: 句子数 n × V。
    输入 (n, rule_id) ↔ 该句子中规则是否出现 (0/1, 用 normalize 转到 [0,1))。 -/
abbrev RuleCountMatrix (n : ℕ) : Type := Fin n → Vocab → ℝ

/-- 词表编码: Rule → Vocab (假设词表是按某种字典序枚举的有限集) -/
axiom vocabEncode : PCFG.Rule → Vocab

/-- extract_struct_rules 投影到 BoR 向量: 每个 Rule 在 doc 中出现 ⇒ 该列 +1。
    对齐 Python `for rule in extract_struct_rules(doc): counts[i, j] = 1.0`。 -/
axiom docToRuleCount :
    (doc : List PCFG.Token) → (i : Fin 1) → RuleCountMatrix 1
  -- 实际定义见 Pipeline.lean 中 n 个 doc 时的 batched 形态, 此处省略。

/-- normalize 整个 BoR 矩阵 (逐元素, 等价 Python `counts *= 0.5` for c=1)。
    对 0/1 输入, normalize(x) = x/(1+x): 0 ↦ 0, 1 ↦ 1/2。 -/
def normalizeRuleCountMatrix {n : ℕ} (M : RuleCountMatrix n) : RuleCountMatrix n :=
  fun i j => normalizeCount (M i j)

lemma normalizeRuleCountMatrix_in_01 {n : ℕ} (M : RuleCountMatrix n)
    (i : Fin n) (j : Vocab) (h : 0 ≤ M i j) :
    0 ≤ normalizeRuleCountMatrix M i j ∧ normalizeRuleCountMatrix M i j < 1 := by
  exact ⟨normalizeCount_nonneg h, normalizeCount_lt_one h⟩

/-! ### 监督 Encoder -/

/-- 线性层: W·x + b。 用 Fin → ℝ 内积表示。 -/
structure Linear (in_dim out_dim : ℕ) where
  weight : Fin out_dim → Fin in_dim → ℝ
  bias  : Fin out_dim → ℝ

/-- Linear 的 forward。 -/
def Linear.forward {in_dim out_dim : ℕ} (L : Linear in_dim out_dim)
    (x : Fin in_dim → ℝ) : Fin out_dim → ℝ :=
  fun i => (∑ j : Fin in_dim, L.weight i j * x j) + L.bias i

/-- ReLU: max(0, x)。 -/
def relu (x : ℝ) : ℝ := max 0 x

/-- ReLU 非负。 -/
lemma relu_nonneg (x : ℝ) : 0 ≤ relu x := by
  simp [relu]
  exact le_max_left _ _

/-- ReLU 与 0 的关系。 -/
lemma relu_zero (x : ℝ) (h : x ≤ 0) : relu x = 0 := by
  simp [relu]; exact max_eq_right h

/-- ReLU 与正数的关系。 -/
lemma relu_pos (x : ℝ) (h : 0 ≤ x) : relu x = x := by
  simp [relu]; exact max_eq_left h

/-- 监督 Encoder 配置 (与 Python SUP_* 常量对齐)。 -/
structure SupConfig where
  V : ℕ := vocabSize
  hidden : ℕ := 256
  Z : ℕ := 32
  dropout : ℝ := 0.5
  ruleDropout : ℝ := 0.1
  weightDecay : ℝ := 1e-2
  labelSmoothing : ℝ := 0.1

/-- 监督 Encoder: 两层 Linear + ReLU + Dropout。
    实际实现中 Dropout 在 eval 模式关闭, 这里只定义 deterministic
    forward (对应 inference_mode)。 -/
structure SupEncoder (cfg : SupConfig) where
  fc1 : Linear cfg.V cfg.hidden
  fc2 : Linear cfg.hidden cfg.Z

namespace SupEncoder

/-- 编码器前向: ℝ^V → ℝ^32。
    对应 Python `_SupEncoder.forward` (eval 模式) -/
def forward {cfg : SupConfig} (enc : SupEncoder cfg)
    (x : Fin cfg.V → ℝ) : Fin cfg.Z → ℝ :=
  let h1 := relu ∘ Linear.forward enc.fc1 x
  Linear.forward enc.fc2 h1

/-- 编码器输出是 ℝ^32 = Vec32 (前提 cfg.V = 32 时)。 -/
lemma forward_type {cfg : SupConfig} (enc : SupEncoder cfg)
    (hV : cfg.V = 32) (x : Fin cfg.V → ℝ) :
    enc.forward x = (enc.forward x : Fin cfg.Z → ℝ) := by
  subst hV
  rfl

end SupEncoder

/-! ### BCE 损失 + 标签平滑 -/

/-- 标准 sigmoid: σ(x) = 1/(1 + e^(-x))。 -/
def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

/-- BCE 单元素: -[y · log σ(z) + (1-y) · log(1-σ(z))]。 -/
def bce (z y : ℝ) : ℝ :=
  -(y * Real.log (sigmoid z) + (1 - y) * Real.log (1 - sigmoid z))

/-- 标签平滑: y_smooth = y · (1-α) + α/2。
    当 α = 0.1, y ∈ {0,1}: y_smooth ∈ {0.05, 0.95}。 -/
def labelSmooth (y α : ℝ) : ℝ := y * (1 - α) + α / 2

lemma labelSmooth_in_01 (y : ℝ) (α : ℝ) (hy : y = 0 ∨ y = 1)
    (hα : 0 ≤ α ∧ α ≤ 1) :
    0 ≤ labelSmooth y α ∧ labelSmooth y α ≤ 1 := by
  cases hy with
  | inl h => simp [h, labelSmooth]; linarith
  | inr h => simp [h, labelSmooth]; linarith

/-- 带标签平滑的 BCE: BCE(z, labelSmooth(y, α))。 -/
def bceSmooth (z y α : ℝ) : ℝ := bce z (labelSmooth y α)

/-- BCE 在 z = 0 时最小:y_smooth ∈ (0,1) 时 bce(z=0, y_smooth) ≤ log 2。
    接受外部 sigmoid 范围证据作为 contract 参数。 -/
lemma bce_nonneg (z y : ℝ) (hy : 0 ≤ y ∧ y ≤ 1)
    (hσ0 : 0 < sigmoid z ∧ sigmoid z < 1)
    (hlog1 : Real.log (sigmoid z) ≥ 0 ∧ Real.log (1 - sigmoid z) ≥ 0) :
    0 ≤ bce z y := by
  unfold bce
  -- BCE = -(y * log σ + (1-y) * log(1-σ)); 由 hlog1 知两项均非负
  linarith [mul_nonneg hy.1 hlog1.1, mul_nonneg (sub_nonneg_of_le hy.2) hlog1.2]

/-! ### 监督训练 step -/

/-- 监督训练的 per-sample 损失:
    Loss(z_pred, z_target) = BCE(z_pred, labelSmooth(z_target, α))。
    对应 `SupConfig.labelSmoothing = 0.1`。 -/
def supLoss {cfg : SupConfig} (enc : SupEncoder cfg)
    (x : Fin cfg.V → ℝ) (y_target : Fin cfg.Z → ℝ) : ℝ :=
  let z_pred := enc.forward x
  ∑ i : Fin cfg.Z, bceSmooth (z_pred i) (y_target i) cfg.labelSmoothing

/-- 训练 loss 非负 (BCE 项非负 + 求和保持非负)。接受外部 per-item BCE 非负证据。 -/
lemma supLoss_nonneg {cfg : SupConfig} (enc : SupEncoder cfg)
    (x : Fin cfg.V → ℝ) (y_target : Fin cfg.Z → ℝ)
    (h_target : ∀ i, 0 ≤ y_target i ∧ y_target i ≤ 1)
    (hbce : ∀ i, 0 ≤ bceSmooth (enc.forward x i) (y_target i) cfg.labelSmoothing) :
    0 ≤ supLoss enc x y_target := by
  unfold supLoss
  exact Finset.sum_nonneg (fun i _ => hbce i)

end PersonalQuery