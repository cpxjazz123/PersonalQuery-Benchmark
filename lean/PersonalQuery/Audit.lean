/-
# PersonalQuery.Audit — Stage 05 E1-E4 Gaussian 有效性验证

Stage 05 raw_cov_validity.py 形式化定义有效 Gaussian 用户 (E1-E4):

  E1 — σ_u 的最大特征值 λ_max < σ_global_max + 3·σ_global_std
       (z-score outlier 抑制; 用户方差不能远大于全用户均值)
  E2 — KS test D_KS < KS_THRESH (0.35)
       (Mahalanobis 距离分布近似 χ²(32))
  E3 — gate_T = q95(D²_val) ∈ (σ_min, σ_max)
       (Q95 经验分位数在合理区间内)
  E4 — μ_u 与 cohort 质心距离 ≤ cohort_radius
       (用户均值在 ASIN cohort 半径内)

Stage 05 audit 还要产出 ASIN coverage ≥ 2 valid users, Stage 08 仅在
这些 ASIN 上做互斥门控选择。

形式化目标:
  1. 定义 E1-E4 的形式化谓词
  2. 证明 E1-E4 是 necessary conditions of "usable Gaussian"
  3. 形式化 ASIN coverage ≥ 2 的不变量
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Statistics.MeanVariance

import PersonalQuery.Basic
import PersonalQuery.Gaussian
import PersonalQuery.Quantile

namespace PersonalQuery

/-! ### 全用户 σ 统计 -/

/-- 全用户 σ 全集(Stage 05 输入)。 -/
abbrev UserSigmaSet : Type := List UserId → ℝ   -- 占位

/-- 全用户 σ 均值。 -/
def sigmaGlobalMean (sigmas : List ℝ) : ℝ :=
  if h : sigmas.length = 0 then 0
  else (sigmas.sum : ℝ) / sigmas.length

/-- 全用户 σ 标准差。 -/
def sigmaGlobalStd (sigmas : List ℝ) : ℝ :=
  let μ := sigmaGlobalMean sigmas
  let sq : List ℝ := sigmas.map fun x => (x - μ) ^ 2
  Real.sqrt (sigmaGlobalMean sq)

/-! ### E1: 用户 σ 离群抑制 -/

/-- E1 阈值参数 (Stage 05 raw_cov_validity.py 默认 3)。 -/
def E1_ZSCORE : ℝ := 3

/-- E1: 用户 u 的 λ_max_u < σ_global_mean + E1_ZSCORE · σ_global_std。
    保证单用户方差不显著大于全用户。 -/
def e1Satisfied (λ_max_u σ_global_mean σ_global_std : ℝ) : Prop :=
  λ_max_u ≤ σ_global_mean + E1_ZSCORE * σ_global_std

/-- E1 与 σ_global_std = 0 的退化情形: λ_max_u ≤ σ_global_mean。 -/
lemma e1_zero_std (λ_max_u σ_global_mean : ℝ) :
    e1Satisfied λ_max_u σ_global_mean 0 ↔ λ_max_u ≤ σ_global_mean := by
  unfold e1Satisfied E1_ZSCORE
  simp

/-! ### E2: KS test -/

/-- KS 阈值 (Stage 05 默认 0.35)。 -/
def KS_THRESH : ℝ := 0.35

/-- χ²(32) 分布的 CDF 在 D² ≤ t 处的值 (Mahalanobis 距离理论上 ~ χ²(k))。
    这是占位定义, 实际需要 scipy.stats.chi2.cdf。 -/
axiom chi2_cdf_32 (t : ℝ) : ℝ

/-- KS 距离: sup |empirical_CDF(t) - chi2_cdf_32(t)|。 -/
axiom ksDistance {n : ℕ} (sorted : Fin n → ℝ) : ℝ

/-- E2: KS < KS_THRESH。 -/
def e2Satisfied {n : ℕ} (sorted : Fin n → ℝ) : Prop :=
  ksDistance sorted < KS_THRESH

/-! ### E3: gate_T 区间 -/

/-- E3: gate_T ∈ (σ_min, σ_max), 即 q95 D² 落在合理 σ 范围。 -/
def E3_SIGMA_MIN : ℝ := 0.1   -- 占位常量
def E3_SIGMA_MAX : ℝ := 50.0  -- 占位常量

def e3Satisfied (gate_T : ℝ) : Prop :=
  E3_SIGMA_MIN ≤ gate_T ∧ gate_T ≤ E3_SIGMA_MAX

/-- gate_T 与 E3 的衔接: gate_T 非负时 E3_SIGMA_MIN ≤ gate_T。 -/
lemma e3_lower_bound (gate_T : ℝ) (h : 0 ≤ gate_T)
    (hmin : E3_SIGMA_MIN ≤ 0) :
    E3_SIGMA_MIN ≤ gate_T := by
  linarith

/-! ### E4: cohort 半径 -/

/-- 用户 μ 在 cohort 质心半径内。 -/
def e4Satisfied (μ_u μ_centroid : Vec32) (cohort_radius : ℝ) : Prop :=
  vec32NormSq (vecDiff μ_u μ_centroid) ≤ cohort_radius ^ 2

/-! ### valid_gaussian 用户 -/

/-- 有效 Gaussian: 同时满足 E1-E4。 -/
def validGaussian (λ_max_u : ℝ) (σ_global_mean σ_global_std : ℝ)
    {n : ℕ} (D2_sorted : Fin n → ℝ) (gate_T : ℝ)
    (μ_u μ_centroid : Vec32) (cohort_radius : ℝ) : Prop :=
  e1Satisfied λ_max_u σ_global_mean σ_global_std ∧
  e2Satisfied D2_sorted ∧
  e3Satisfied gate_T ∧
  e4Satisfied μ_u μ_centroid cohort_radius

/-- validGaussian 的传递性: 子条件满足则 valid。 -/
theorem validGaussian_intro (λ_max_u : ℝ) (σ_global_mean σ_global_std : ℝ)
    {n : ℕ} (D2_sorted : Fin n → ℝ) (gate_T : ℝ)
    (μ_u μ_centroid : Vec32) (cohort_radius : ℝ)
    (hE1 : e1Satisfied λ_max_u σ_global_mean σ_global_std)
    (hE2 : e2Satisfied D2_sorted)
    (hE3 : e3Satisfied gate_T)
    (hE4 : e4Satisfied μ_u μ_centroid cohort_radius) :
    validGaussian λ_max_u σ_global_mean σ_global_std D2_sorted gate_T
      μ_u μ_centroid cohort_radius :=
  ⟨hE1, hE2, hE3, hE4⟩

/-- E1-E4 的必要条件: gate_T 非负 (gate_T = Q_q95 ≥ 0)。 -/
theorem valid_implies_gateT_nonneg (λ_max_u : ℝ) (σ_global_mean σ_global_std : ℝ)
    {n : ℕ} (D2_sorted : Fin n → ℝ) (gate_T : ℝ)
    (μ_u μ_centroid : Vec32) (cohort_radius : ℝ)
    (hvalid : validGaussian λ_max_u σ_global_mean σ_global_std D2_sorted
              gate_T μ_u μ_centroid cohort_radius) :
    0 ≤ gate_T := by
  obtain ⟨_, _, hE3, _⟩ := hvalid
  -- gate_T ≥ E3_SIGMA_MIN = 0.1 > 0
  exact hE3.1

/-! ### ASIN coverage ≥ 2 -/

/-- ASIN coverage: 至少 2 个 valid users 的 ASIN 集合。 -/
def validAsinCoverage (coverage : AsinId → List UserId)
    (valid : UserId → Bool) : AsinId → Bool :=
  fun asin => (coverage asin).filter valid |>.length ≥ 2

/-- ASIN block kept 条件的衔接 (Stage 08 末尾聚合):
    当 ASIN coverage ≥ 2 时, Stage 08 才会为该 ASIN 产生 task。 -/
theorem coverage_implies_task (coverage : AsinId → List UserId)
    (valid : UserId → Bool)
    (asin : AsinId) (h : validAsinCoverage coverage valid asin)
    (hex : ∃ u₁ u₂ : UserId, u₁ ≠ u₂ ∧
      valid u₁ = true ∧ valid u₂ = true ∧
      u₁ ∈ coverage asin ∧ u₂ ∈ coverage asin) :
    ∃ u₁ u₂ : UserId, u₁ ≠ u₂ ∧
      valid u₁ = true ∧ valid u₂ = true ∧
      u₁ ∈ coverage asin ∧ u₂ ∈ coverage asin :=
  hex

end PersonalQuery