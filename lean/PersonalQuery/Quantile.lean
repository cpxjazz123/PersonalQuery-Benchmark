/-
# PersonalQuery.Quantile — 经验分位数 + Q95 Gate

Stage 04 fit_per_user_gaussian.py:46 把 GATE_QUANTILE = 0.95 硬编码,
对每个用户的 val D² 序列取 q95 作为 gate_T:

    d2_q50 = quantile(d2_val, 0.50)
    d2_q75 = quantile(d2_val, 0.75)
    d2_q95 = quantile(d2_val, 0.95)   ← gate_T
    d2_max = d2_val.max()

Stage 05 audit 验证 gate_T 的合理性 (Stage 05 raw_cov_validity.py),
Stage 08 用 gate_T 作为马氏距离门控阈值。

形式化目标:
  1. 定义 sorted list 上的 quantile 函数
  2. 形式化 q95 对应 Python `np.quantile(d2_val, 0.95)` 的「线性插值」语义
  3. 定义 gate_T = quantile_q95(D²_val_u)
  4. 证明 gate_T 的基本性质 (非负, 单调于 D² 值)
  5. 形式化 Stage 04 的 q50/q75/q95/max 聚合
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Order.Interval.Set.Basic

import PersonalQuery.Basic
import PersonalQuery.Mahalanobis

namespace PersonalQuery

/-! ### Q95 常量 -/

/-- Stage 04 门控分位数: 0.95。 -/
def FILTER_Q : ℝ := 0.95

lemma FILTER_Q_in_unit : 0 ≤ FILTER_Q ∧ FILTER_Q ≤ 1 := by
  unfold FILTER_Q; norm_num

/-! ### 经验分位数 -/

/-- 经验分位数 (简化: type-7, R 默认)。
    对长度 n 的 list X 与 q ∈ [0,1], quantile_q(X) = X[idx]
    其中 idx = floor(q * (n - 1))。
    Python `np.quantile` 默认 linear interpolation, 这里用更简化的索引语义。
    完整形式化需 List 排序 + 边界处理, 见下。 -/
noncomputable def quantileIdx (n : ℕ) (q : ℝ) (hn : n > 0) : ℕ :=
  -- index = ⌊q * (n - 1)⌋
  Int.toNat ⌊q * ((n : ℝ) - 1)⌋

lemma quantileIdx_in_range (n : ℕ) (q : ℝ) (hn : n > 0)
    (hq : 0 ≤ q ∧ q ≤ 1) :
    quantileIdx n q hn < n := by
  unfold quantileIdx
  -- ⌊q * (n - 1)⌋ ≤ q * (n - 1) ≤ n - 1 < n
  have hn_real : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hn_m1_nonneg : (0 : ℝ) ≤ (n : ℝ) - 1 := by linarith
  have hq_mul : q * ((n : ℝ) - 1) ≤ (n : ℝ) - 1 := by nlinarith [hq.1, hq.2, hn_m1_nonneg]
  have hfloor_le_R : (⌊q * ((n : ℝ) - 1)⌋ : ℝ) ≤ (n : ℝ) - 1 :=
    (Int.floor_le _).trans hq_mul
  have hfloor_lt_R : (⌊q * ((n : ℝ) - 1)⌋ : ℝ) < (n : ℝ) := by linarith
  -- ⌊_⌋ ≥ 0 因为 q ≥ 0 ∧ (n-1) ≥ 0
  have hfloor_nonneg : (0 : ℤ) ≤ ⌊q * ((n : ℝ) - 1)⌋ :=
    Int.floor_nonneg.mpr (by nlinarith [hq.1, hn_m1_nonneg])
  -- 转 Int 不等式, 然后经 toNat_of_nonneg 保留 < n
  have hfloor_lt_Z : (⌊q * ((n : ℝ) - 1)⌋ : ℤ) < (n : ℤ) := by exact_mod_cast hfloor_lt_R
  have hcast : (⌊q * ((n : ℝ) - 1)⌋.toNat : ℤ) = ⌊q * ((n : ℝ) - 1)⌋ :=
    Int.toNat_of_nonneg hfloor_nonneg
  have hkey : (⌊q * ((n : ℝ) - 1)⌋.toNat : ℤ) < (n : ℤ) := by
    rw [hcast]
    exact hfloor_lt_Z
  exact Int.ofNat_lt.mp hkey

/-- 经验分位数 (按排序后 list 取 idx): quantile_q(X_sorted) = X_sorted[idx]。 -/
noncomputable def empiricalQuantile {n : ℕ} (X_sorted : Fin n → ℝ) (q : ℝ)
    (hn : n > 0) (hq : 0 ≤ q ∧ q ≤ 1) : ℝ :=
  X_sorted ⟨quantileIdx n q hn, quantileIdx_in_range n q hn hq⟩

/-- q = 0 时的 quantile 是最小值。 -/
lemma empiricalQuantile_zero {n : ℕ} (X_sorted : Fin n → ℝ)
    (hn : n > 0) (hXmin : ∀ i, X_sorted ⟨0, by omega⟩ ≤ X_sorted i) :
    empiricalQuantile X_sorted 0 hn ⟨by norm_num, by norm_num⟩ = X_sorted ⟨0, by omega⟩ := by
  unfold empiricalQuantile quantileIdx
  simp [Int.floor_zero, Int.toNat_zero]

/-- q = 1 时的 quantile 是最大值。 -/
lemma empiricalQuantile_one {n : ℕ} (X_sorted : Fin n → ℝ)
    (hn : n > 0) (hXmax : ∀ i, X_sorted i ≤ X_sorted ⟨n - 1, by omega⟩) :
    empiricalQuantile X_sorted 1 hn ⟨by norm_num, by norm_num⟩ = X_sorted ⟨n - 1, by omega⟩ := by
  -- 直接证明 quantileIdx n 1 hn = n - 1
  have hidx_eq : quantileIdx n 1 hn = n - 1 := by
    unfold quantileIdx
    -- ⌊1 * (n - 1)⌋ = ↑(n - 1) : ℤ
    have hfloor : (⌊1 * ((n : ℝ) - 1)⌋ : ℤ) = (n : ℤ) - 1 := by
      rw [one_mul, Int.floor_sub_one]
      norm_cast
      rw [Int.subNatNat_eq_coe]
      simp [hn]
    rw [hfloor]
    -- 现在目标: ((↑n - 1) : ℤ).toNat = n - 1 : ℕ
    simp [Int.subNatNat_eq_coe, Int.toNat_natCast]
  unfold empiricalQuantile
  exact congrArg X_sorted (Fin.eq_of_val_eq hidx_eq)

/-! ### 单调性 -/

/-- 经验分位数单调性: q₁ ≤ q₂ ⟹ quantile_q₁ ≤ quantile_q₂ 对固定排序输入。 -/
theorem empiricalQuantile_mono {n : ℕ} (X_sorted : Fin n → ℝ)
    (q₁ q₂ : ℝ) (hn : n > 0)
    (hq₁ : 0 ≤ q₁ ∧ q₁ ≤ 1) (hq₂ : 0 ≤ q₂ ∧ q₂ ≤ 1)
    (hqq : q₁ ≤ q₂)
    (hmono : ∀ i j : Fin n, i ≤ j → X_sorted i ≤ X_sorted j) :
    empiricalQuantile X_sorted q₁ hn hq₁ ≤
    empiricalQuantile X_sorted q₂ hn hq₂ := by
  -- 排序输入 ⇒ idx₁ ≤ idx₂ ⇒ X_sorted[idx₁] ≤ X_sorted[idx₂]
  have hn_real : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hn_m1_nonneg : (0 : ℝ) ≤ (n : ℝ) - 1 := by linarith
  have hqq_mul : q₁ * ((n : ℝ) - 1) ≤ q₂ * ((n : ℝ) - 1) := by
    nlinarith [hq₁.1, hq₂.1, hqq, hn_m1_nonneg]
  have hfloor_le : ⌊q₁ * ((n : ℝ) - 1)⌋ ≤ ⌊q₂ * ((n : ℝ) - 1)⌋ :=
    Int.floor_mono hqq_mul
  have hidx : quantileIdx n q₁ hn ≤ quantileIdx n q₂ hn := by
    unfold quantileIdx
    exact Int.toNat_le_toNat hfloor_le
  have hkey :
      (⟨quantileIdx n q₁ hn, quantileIdx_in_range n q₁ hn hq₁⟩ : Fin n) ≤
      ⟨quantileIdx n q₂ hn, quantileIdx_in_range n q₂ hn hq₂⟩ := by
    exact Fin.le_def.mpr hidx
  exact hmono _ _ hkey

/-- 经验分位数对输入单调: X_sorted ⊑ Y_sorted ⟹ quantile_q X ≤ quantile_q Y。 -/
theorem empiricalQuantile_input_mono {n : ℕ}
    (X_sorted Y_sorted : Fin n → ℝ)
    (q : ℝ) (hn : n > 0) (hq : 0 ≤ q ∧ q ≤ 1)
    (hXY : ∀ i, X_sorted i ≤ Y_sorted i) :
    empiricalQuantile X_sorted q hn hq ≤ empiricalQuantile Y_sorted q hn hq :=
  hXY _

/-- q95 quantile 是 val D² 的上分位: q95 ≥ median (q50)。 -/
theorem quantile_q95_above_q50 {n : ℕ} (X_sorted : Fin n → ℝ)
    (hn : n > 0) (hmono : ∀ i j : Fin n, i ≤ j → X_sorted i ≤ X_sorted j) :
    empiricalQuantile X_sorted 0.5 hn ⟨by norm_num, by norm_num⟩ ≤
    empiricalQuantile X_sorted 0.95 hn ⟨by norm_num, by norm_num⟩ :=
  empiricalQuantile_mono X_sorted 0.5 0.95 hn ⟨by norm_num, by norm_num⟩ ⟨by norm_num, by norm_num⟩
    (by norm_num) hmono

/-! ### Gate_T -/

/-- Stage 04 用户 gate_T = Q_q95(D²_val)。 -/
noncomputable def gateT {n_val : ℕ} (D2_val_sorted : Fin n_val → ℝ)
    (hn_val : n_val > 0) : ℝ :=
  empiricalQuantile D2_val_sorted 0.95 hn_val FILTER_Q_in_unit

/-- gate_T 非负: empiricalQuantile 在索引 idx 处取值, 由 hnonneg 知 ≥ 0。 -/
theorem gateT_nonneg {n_val : ℕ} (D2_val_sorted : Fin n_val → ℝ)
    (hn_val : n_val > 0) (hnonneg : ∀ i, 0 ≤ D2_val_sorted i) :
    0 ≤ gateT D2_val_sorted hn_val := by
  unfold gateT empiricalQuantile
  exact hnonneg ⟨quantileIdx n_val 0.95 hn_val,
                quantileIdx_in_range n_val 0.95 hn_val FILTER_Q_in_unit⟩

/-- gate_T 单调于 D² 值 (与单调性定理组合)。 -/
theorem gateT_mono {n_val : ℕ}
    (D2_val_sorted₁ D2_val_sorted₂ : Fin n_val → ℝ)
    (hn_val : n_val > 0)
    (hle : ∀ i, D2_val_sorted₁ i ≤ D2_val_sorted₂ i) :
    gateT D2_val_sorted₁ hn_val ≤ gateT D2_val_sorted₂ hn_val := by
  unfold gateT
  exact empiricalQuantile_input_mono _ _ _ hn_val FILTER_Q_in_unit hle

/-- gate_T ≥ D²_val 中位数 (q50 ≤ q95)。 -/
theorem gateT_above_median {n_val : ℕ} (D2_val_sorted : Fin n_val → ℝ)
    (hn_val : n_val > 0)
    (hmono : ∀ i j : Fin n_val, i ≤ j → D2_val_sorted i ≤ D2_val_sorted j) :
    empiricalQuantile D2_val_sorted 0.5 hn_val ⟨by norm_num, by norm_num⟩ ≤
    gateT D2_val_sorted hn_val :=
  quantile_q95_above_q50 _ hn_val hmono

/-! ### Stage 04 聚合 -/

/-- Stage 04 用户聚合: q50/q75/q95/max 四个标量。 -/
structure D2Quantiles where
  q50 : ℝ
  q75 : ℝ
  q95 : ℝ  -- = gate_T
  max : ℝ

/-- 单调性: q50 ≤ q75 ≤ q95 ≤ max。
    结构体内部默认有序 (与 Stage 04 fit_per_user_gaussian.py 假定一致); 此处
    接受外部单调性证据 h5075/h7595/h95max 作为 contract 参数。 -/
theorem D2Quantiles_chain (q : D2Quantiles)
    (h5075 : q.q50 ≤ q.q75) (h7595 : q.q75 ≤ q.q95) (h95max : q.q95 ≤ q.max) :
    q.q50 ≤ q.q75 ∧ q.q75 ≤ q.q95 ∧ q.q95 ≤ q.max :=
  ⟨h5075, h7595, h95max⟩

end PersonalQuery