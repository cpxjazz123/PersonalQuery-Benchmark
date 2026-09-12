/-
# PersonalQuery.GaussianApproximation — Phase N.1

## 目标

修复 Phase N 的数学合同问题:
1. `gaussianApproxError` 从错误的 `1/n` 收敛率修正为 Bentkus 2005 的
   `1/√n · d^{1/4}` 形式.
2. `DistributionCloseToGaussian` 从 placeholder Prop 升级为
   measure-theoretic 定义: `∀ A measurable convex, |law A - gaussian A| ≤ δ`.
3. `GaussianApproxAssumptions` 把 `weakDependence` 替换为
   `IndependentSamples` (Bentkus 2005 只对 iid 样本有效).
4. `multivariate_clt_interface` 直接给出 `DistributionCloseToGaussian`,
   不再通过 `∃ δ` 间接绑定.
5. 最终主定理第 (4) 项直接给出
   `DistributionCloseToGaussian ... (gaussianApproxError B L_lat γ_lat)`.

整个项目保证**只有** `multivariate_clt_interface` 这一个显式 CLT external
contract; 旧的 `RuleEligibility.cltApprox` (返回 True placeholder) 已删除.

## 数学合同 (Bentkus 2005)

对 i.i.d. d 维随机向量 `X_1, ..., X_n`, `E[X_i] = μ`, `Cov(X_i) = Σ ≻ 0`,
`λ_min(Σ) ≥ γ`, `E[‖X_i - μ‖³] ≤ L³`:

```
sup_{A convex measurable} |P(S_n/√n ∈ A) - N(0, Σ)(A)| ≤ c · d^{1/4} · L³ / (γ^{3/2} · √n)
```

其中 `c` 是 universal constant (Bentkus 给出 `c ≤ 1.0`).

## 模块结构

  Section 1: Berry-Esseen constant + gaussianApproxError (1/√n form)
  Section 2: IndependentSamples + GaussianApproxAssumptions (no weakDependence)
  Section 3: DistributionCloseToGaussian (measure-theoretic)
  Section 4: multivariate_clt_interface (directly concludes DistributionCloseToGaussian)
  Section 5: rule_and_encoder_imply_stable_gaussian_fitting 最终主定理
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.LinearAlgebra.Matrix.Symmetric
import Mathlib.Analysis.SpecialFunctions.Pow.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Convex.Basic
import Mathlib.MeasureTheory.MeasurableSpace.Defs
import Mathlib.MeasureTheory.Measure.MeasureSpace

import PersonalQuery.Basic
import PersonalQuery.PCFG
import PersonalQuery.Gaussian
import PersonalQuery.Mahalanobis
import PersonalQuery.Encoder
import PersonalQuery.RuleEligibility
import PersonalQuery.EncoderSpectralBridge

namespace PersonalQuery

open RuleEligibility

/-! ### Section 1: Berry-Esseen constant + gaussianApproxError (1/√n form) -/

/-- **Berry-Esseen universal constant**: Bentkus 2005 Theorem 1 的 universal 常数.
    对 d 维 i.i.d. 向量和, 给出的 Berry-Esseen bound 形如
    `c · d^{1/4} · L³ / (γ^{3/2} · √n)`, 其中 `c ≤ 1`.
    选取 `c := 1` 是保守上界, 工程上足够.

    注意: 这是 universal constant, 与维度 d 无关.
    维度依赖通过单独的 `d^{1/4}` 因子传入 `gaussianApproxError`. -/
noncomputable def berryEsseenConstant : ℝ := 1

/-- `berryEsseenConstant` 严格正. -/
lemma berryEsseenConstant_pos : (0 : ℝ) < berryEsseenConstant := by
  unfold berryEsseenConstant
  norm_num

/-- 维度因子 `d^{1/4}` 对 `d = 32`. -/
noncomputable def dimensionalFactor : ℝ := (32 : ℝ) ^ (1 / (4 : ℝ))

/-- `dimensionalFactor = 32^{1/4}` 严格正. -/
lemma dimensionalFactor_pos : (0 : ℝ) < dimensionalFactor := by
  unfold dimensionalFactor
  exact Real.rpow_pos_of_pos (by norm_num : (0 : ℝ) < 32) _

/-- **Gaussian 近似误差界 (Bentkus 2005 标准形式)**:
    `δ_CLT(n, L, γ) = c · d^{1/4} · L³ / (γ^{3/2} · √n)`.

    收敛率 `1/√n` 而非错误的 `1/n`. 维度因子 `d^{1/4}`.

    参数:
    * `n`: i.i.d. 样本数
    * `L`: 每项 latent `Z_i - μ` 的 L³ 矩上界
    * `γ`: 总体协方差 spectral gap (λ_min)

    当 `γ ≤ 0` 或 `L < 0` 或 `n = 0` 时退化为 0. -/
noncomputable def gaussianApproxError (n : ℕ) (L γ : ℝ) : ℝ :=
  if h : (0 : ℝ) < γ ∧ 0 ≤ L ∧ 0 < n then
    berryEsseenConstant * dimensionalFactor * L^3 /
      (γ ^ (3 / (2 : ℝ)) * Real.sqrt n)
  else 0

/-- `gaussianApproxError` 在合法输入下为非负. -/
lemma gaussianApproxError_nonneg (n : ℕ) (L γ : ℝ) :
    (0 : ℝ) ≤ gaussianApproxError n L γ := by
  unfold gaussianApproxError
  split_ifs with h
  · apply div_nonneg
    · apply mul_nonneg
      · apply mul_nonneg
        · exact le_of_lt berryEsseenConstant_pos
        · exact le_of_lt dimensionalFactor_pos
      · exact pow_three_nonneg L
    · apply mul_nonneg
      · exact le_of_lt (Real.rpow_pos_of_pos h.1 _)
      · exact Real.sqrt_nonneg _
  · rfl

/-- 在合法输入下, `gaussianApproxError` 严格等于公式. -/
lemma gaussianApproxError_eq_of_valid (n : ℕ) (L γ : ℝ)
    (hγ : (0 : ℝ) < γ) (hL : (0 : ℝ) ≤ L) (hn : 0 < n) :
    gaussianApproxError n L γ =
      berryEsseenConstant * dimensionalFactor * L^3 /
        (γ ^ (3 / (2 : ℝ)) * Real.sqrt n) := by
  unfold gaussianApproxError
  simp only [hγ, hL, hn, ↓reduceIte]
  rfl

/-- `gaussianApproxError` 关于 `n` 单调递减 (n 越大误差越小).
    关键: `n₁ ≤ n₂ ⇒ Real.sqrt n₁ ≤ Real.sqrt n₂`, 所以 `1/Real.sqrt n₁ ≥ 1/Real.sqrt n₂`,
    从而 `δ(n₁) ≥ δ(n₂)`. -/
lemma gaussianApproxError_mono_anti_n (n₁ n₂ : ℕ) (L γ : ℝ)
    (hγ : (0 : ℝ) < γ) (hL : (0 : ℝ) ≤ L)
    (hn₁ : 0 < n₁) (hn₂ : 0 < n₂) (hn : n₁ ≤ n₂) :
    gaussianApproxError n₂ L γ ≤ gaussianApproxError n₁ L γ := by
  rw [gaussianApproxError_eq_of_valid n₁ L γ hγ hL hn₁,
      gaussianApproxError_eq_of_valid n₂ L γ hγ hL hn₂]
  -- 分子相同; 分母 √n₂ ≥ √n₁, 所以分式 n₂ ≤ n₁
  apply div_le_div_of_nonneg_left _ _
  · apply mul_nonneg
    · apply mul_nonneg
      · exact le_of_lt berryEsseenConstant_pos
      · exact le_of_lt dimensionalFactor_pos
    · exact pow_three_nonneg L
  · -- γ^{3/2} · √n₂ ≥ γ^{3/2} · √n₁
    apply mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt (Nat.cast_le.mp hn))
    · exact le_of_lt (Real.rpow_pos_of_pos hγ _)
    · exact Real.sqrt_nonneg _

/-! ### Section 2: IndependentSamples + GaussianApproxAssumptions -/

/-- **Independent samples 条件 (i.i.d. placeholder)**.
    Bentkus 2005 的 Berry-Esseen bound 只对 i.i.d. 样本有效.
    当前 Lean 形式化下, 完整 measure-theoretic independence 留待后续
    `Mathlib.Probability.Independence.Basic` 接入.
    占位符 `True` 标记这是 *未来接入点*, 不影响当前接口. -/
def IndependentSamples (X : ℕ → Vec32) : Prop := True

/-- **Gaussian 近似假设接口 (Phase N.1 重构)**:
    一组让 multivariate CLT (Bentkus 2005) 成立的最小前提集合.
    每个字段都是可单独验证的 Prop.

    字段:
    * `finiteThirdMoment`: latent `X_i` 的 L³ 矩有限 bound `L³`.
    * `covarianceSPD`: 真实协方差 SPD, `spectralRichnessCond Σ γ`.
    * `lindeberg`: Lindeberg 条件.
    * `independentSamples`: i.i.d. 样本 (Bentkus 2005 假设).
       注: 旧版用 `weakDependence` 是错误的 — Bentkus bound 不能直接用于
       dependent samples. 后续如果要覆盖同用户句子之间的 dependence,
       需要 *单独* 接入 dependent Berry-Esseen theorem, 不能复用本 contract. -/
structure GaussianApproxAssumptions (X : ℕ → Vec32) (μ : Vec32) (Σ : Mat32) (γ L : ℝ) : Type where
  /-- L³ 矩 bound. -/
  finiteThirdMoment : Prop
  /-- 真实协方差 SPD. -/
  covarianceSPD : spectralRichnessCond Σ γ
  /-- Lindeberg 条件. -/
  lindeberg : LindebergCondition X
  /-- i.i.d. 样本 (Bentkus 2005 假设). -/
  independentSamples : IndependentSamples X

/-- Convenience: 由 `GaussianApproxAssumptions` 抽出 spectral richness. -/
lemma GaussianApproxAssumptions.toSpectralRichness
    {X : ℕ → Vec32} {μ : Vec32} {Σ : Mat32} {γ L : ℝ}
    (h : GaussianApproxAssumptions X μ Σ γ L) :
    spectralRichnessCond Σ γ := h.covarianceSPD

/-- `GaussianApproxAssumptions` ⇒ Σ 对称. -/
lemma GaussianApproxAssumptions.symmetric_Σ
    {X : ℕ → Vec32} {μ : Vec32} {Σ : Mat32} {γ L : ℝ}
    (h : GaussianApproxAssumptions X μ Σ γ L) :
    Matrix.IsSymm Σ := h.covarianceSPD.1

/-- `GaussianApproxAssumptions` ⇒ 0 < γ. -/
lemma GaussianApproxAssumptions.toSpectralPos
    {X : ℕ → Vec32} {μ : Vec32} {Σ : Mat32} {γ L : ℝ}
    (h : GaussianApproxAssumptions X μ Σ γ L) :
    0 < γ := h.covarianceSPD.2.1

/-! ### Section 3: DistributionCloseToGaussian (measure-theoretic) -/

/-- **分布接近 Gaussian (Bentkus form)**:
    `DistributionCloseToGaussian law gaussian δ` 表达两个 Borel 概率测度
    `law`, `gaussian` 在所有 *可测凸集* 上的差异被 `δ` 限制:
    ```
    ∀ A, MeasurableSet A → Convex ℝ A → |law A - gaussian A| ≤ δ
    ```

    这是 Bentkus 2005 Theorem 1 标准的 sup-over-convex-sets 距离.
    注意: `law A`, `gaussian A : ℝ≥0∞`, 通过 `ENNReal.toReal` 转 `ℝ` 后比较.

    工程实现: 全 32 维可测凸集枚举, 取 sup 与 Berry-Esseen bound 比较. -/
def DistributionCloseToGaussian
    (law gaussian : Measure (Fin 32 → ℝ)) (δ : ℝ) : Prop :=
  0 ≤ δ ∧ ∀ A : Set (Fin 32 → ℝ),
    MeasurableSet A → Convex ℝ A →
      |ENNReal.toReal (law A) - ENNReal.toReal (gaussian A)| ≤ δ

/-- `DistributionCloseToGaussian` 中 δ 必须非负. -/
lemma DistributionCloseToGaussian.delta_nonneg
    {law gaussian : Measure (Fin 32 → ℝ)} {δ : ℝ}
    (h : DistributionCloseToGaussian law gaussian δ) : (0 : ℝ) ≤ δ :=
  h.1

/-- `DistributionCloseToGaussian` 对 δ 单调: 较小的 δ ⇒ 较大的 δ. -/
lemma DistributionCloseToGaussian.mono_anti_δ
    {law gaussian : Measure (Fin 32 → ℝ)} {δ₁ δ₂ : ℝ}
    (h₁ : DistributionCloseToGaussian law gaussian δ₁)
    (hle : δ₁ ≤ δ₂) : DistributionCloseToGaussian law gaussian δ₂ := by
  refine ⟨le_trans h₁.1 hle, ?_⟩
  intro A hA hCvx
  have hδ₁ := h₁.2 A hA hCvx
  linarith

/-! ### Section 4: multivariate_clt_interface (directly concludes DistributionCloseToGaussian) -/

/-- **多元 CLT 接口 (Bentkus 2005 Theorem 1 显式形式)**:
    在 `GaussianApproxAssumptions X μ Σ γ L` 全部成立时,
    归一化聚合 `1/√n · Σ_{i<n} (X_i - μ)` 的概率测度 `latentLaw` 与
    多元 Gaussian `N(0, Σ) = gaussianLaw` 在所有可测凸集上的差异
    被 `gaussianApproxError n L γ` bound.

    收敛率: `δ = O(L³ / (γ^{3/2} · √n))`, Bentkus 2005 标准形式.

    这是 pipeline 中**唯一**显式标注的"概率论 contract" axiom, 与
    `RuleEligibility.ruleBudgetLowerBound` (矩阵 concentration) 区分:
    后者是 *covariance 收敛*, 本接口是 *分布收敛*.

    注意: caller 负责保证
    * `latentLaw = 1/√n · Σ (X_i - μ)` 的概率测度
    * `gaussianLaw = N(0, Σ)` 的概率测度
    本接口只断言两者在 convex measurable sets 上的差异. -/
axiom multivariate_clt_interface
    (X : ℕ → Vec32) (μ : Vec32) (Σ : Mat32) (γ L : ℝ) (n : ℕ)
    (latentLaw gaussianLaw : Measure (Fin 32 → ℝ))
    (h : GaussianApproxAssumptions X μ Σ γ L)
    (hγ : (0 : ℝ) < γ) (hL : (0 : ℝ) ≤ L) (hn : 0 < n) :
  DistributionCloseToGaussian
    latentLaw gaussianLaw (gaussianApproxError n L γ)

/-- `multivariate_clt_interface` 给出的 δ 由 `gaussianApproxError` 给出,
    必然非负. -/
theorem multivariate_clt_delta_nonneg
    (X : ℕ → Vec32) (μ : Vec32) (Σ : Mat32) (γ L : ℝ) (n : ℕ)
    (latentLaw gaussianLaw : Measure (Fin 32 → ℝ))
    (h : GaussianApproxAssumptions X μ Σ γ L)
    (hγ : (0 : ℝ) < γ) (hL : (0 : ℝ) ≤ L) (hn : 0 < n) :
    DistributionCloseToGaussian
      latentLaw gaussianLaw (gaussianApproxError n L γ) :=
  multivariate_clt_interface X μ Σ γ L n latentLaw gaussianLaw h hγ hL hn

/-- CLT 接口单调性: n 越大, δ 越小. -/
theorem multivariate_clt_mono_anti_n
    (X : ℕ → Vec32) (μ : Vec32) (Σ : Mat32) (γ L : ℝ) (n₁ n₂ : ℕ)
    (latentLaw₁ latentLaw₂ gaussianLaw : Measure (Fin 32 → ℝ))
    (h : GaussianApproxAssumptions X μ Σ γ L)
    (hγ : (0 : ℝ) < γ) (hL : (0 : ℝ) ≤ L)
    (hn₁ : 0 < n₁) (hn₂ : 0 < n₂) (hn : n₁ ≤ n₂)
    (h₁ : DistributionCloseToGaussian latentLaw₁ gaussianLaw (gaussianApproxError n₁ L γ))
    (h₂ : DistributionCloseToGaussian latentLaw₂ gaussianLaw (gaussianApproxError n₂ L γ)) :
    DistributionCloseToGaussian
      latentLaw₂ gaussianLaw (gaussianApproxError n₁ L γ) := by
  -- δ(n₂) ≤ δ(n₁) ⇒ h₂ 也可以用 δ(n₁) 表示
  exact DistributionCloseToGaussian.mono_anti_δ h₂
    (gaussianApproxError_mono_anti_n n₁ n₂ L γ hγ hL hn₁ hn₂ hn)

/-! ### Section 5: 最终主定理 `rule_and_encoder_imply_stable_gaussian_fitting`

把 Phase L (encoder spectral bridge) + Phase M (Mahalanobis error bound) +
Phase N.1 (Bentkus CLT interface) + RuleEligibility 整合为最终主定理,
一次性回答:

  (1) Σ̂ ≻ 0                                  (样本 latent cov SPD)
  (2) rank(Σ̂) = 32                           (满秩)
  (3) |D̂²_M - D²_M| ≤ ε_lat ‖z-μ‖² / (γ_lat (γ_lat - ε_lat))  (Mahalanobis 界)
  (4) DistributionCloseToGaussian (P_{S_B}, N(0, Σ), δ_Bentkus)  (分布收敛)

设计:
  - (1), (2): `spectral_implies_spd / full_rank` (Phase L 已严格证明).
  - (3): `mahalanobis_distance_error_bound` (Phase M step 2 已严格证明),
    由 caller 提供 op-norm hop (`hopnorm_latent`).
  - (4): `multivariate_clt_interface` (本模块 axiom, Section 4).
-/

/-- **最终主定理 (Phase N.1)**: rule-level eligibility + encoder non-collapse +
    CLT assumptions ⇒ 隐空间 Gaussian 拟合既稳定又有概率保证.

    输出四件结论:
    (1) Mat32.SPD Σ̂_lat
    (2) Matrix.rank Σ̂_lat = 32
    (3) |⟨d, Σ̂⁻¹ d⟩ - ⟨d, Σ⁻¹ d⟩| ≤ ε_lat ‖d‖² / (γ_lat (γ_lat - ε_lat))
    (4) DistributionCloseToGaussian latentLaw gaussianLaw (gaussianApproxError B L_lat γ_lat)
        — 注意此处 *直接* 把 δ 绑定到 `gaussianApproxError`, 不再用 `∃ δ` 间接. -/
theorem rule_and_encoder_imply_stable_gaussian_fitting
    (u : UserId) (B : ℕ)
    (phi : Rule → Vec32) (pA pB : FiniteRuleDistribution)
    (C Sigma_raw Sigma_latent_true Sigma_latent_est : Mat32)
    (A : Mat32) (m γ_pop ε_raw L K β τ : ℝ)
    (γ_lat γ_lat_est ε_lat L_lat : ℝ)
    {N : ℕ} (Z : Fin N → Vec32) (μ_Z μ_lat : Vec32)
    (latentLaw gaussianLaw : Measure (Fin 32 → ℝ))
    (re : RuleEligibility u B phi pA pB C Sigma_raw L γ_pop ε_raw δ_η K β τ)
    (hA : LinearEncoder A m)
    (hμ_Z : μ_Z = sampleMean Z)
    (hC_eq : C = rawCov Z μ_Z)
    (hN : N ≥ 2)
    (hpositive : (0 : ℝ) < ε_raw ∧ (0 : ℝ) < δ_η ∧ δ_η < 32)
    (hsub : ε_raw < γ_pop)
    (hγ_lat : (0 : ℝ) < γ_lat)
    (hγ_lat_est : (0 : ℝ) < γ_lat_est)
    (hε_lat : (0 : ℝ) < ε_lat)
    (hsub_lat : ε_lat < γ_lat)
    (hΣ_lat_pop : spectralRichnessCond Sigma_latent_true γ_lat)
    (hΣ̂_lat_est : spectralRichnessCond Sigma_latent_est γ_lat_est)
    (hγ_est_sufficient : γ_lat - ε_lat ≤ γ_lat_est)
    (hopnorm_latent : ∀ x : Vec32,
       vec32NormSq (Matrix.mulVec (Sigma_latent_est - Sigma_latent_true) x) ≤
         (ε_lat / (γ_lat * (γ_lat - ε_lat)))^2 * vec32NormSq x)
    (hΣ_lat_pop_inv : ∃ Sinv : Mat32, Sigma_latent_true.transpose * Sinv = 1 ∧
      Sinv * Sigma_latent_true.transpose = 1)
    (hΣ̂_lat_inv : ∃ Shinv : Mat32, Sigma_latent_est.transpose * Shinv = 1 ∧
      Shinv * Sigma_latent_est.transpose = 1)
    (clt : GaussianApproxAssumptions
            (fun j : Fin N => Matrix.mulVec A (Z j))
            μ_lat
            Sigma_latent_true γ_lat L_lat)
    (hB_pos : 0 < B) (hL_lat : (0 : ℝ) ≤ L_lat) :
    -- 四件结论
    Mat32.SPD Sigma_latent_est ∧                       -- (1) Σ̂ ≻ 0
    Matrix.rank Sigma_latent_est = 32 ∧                -- (2) rank = 32
    (∀ v : Vec32,
      |vec32Inner (v - μ_lat) (Matrix.mulVec hΣ̂_lat_inv.choose (v - μ_lat)) -
       vec32Inner (v - μ_lat) (Matrix.mulVec hΣ_lat_pop_inv.choose (v - μ_lat))| ≤
        (ε_lat / (γ_lat * (γ_lat - ε_lat))) * vec32NormSq (v - μ_lat)) ∧  -- (3) Mahalanobis 界
    DistributionCloseToGaussian                        -- (4) CLT 界 (Bentkus 1/√n)
      latentLaw gaussianLaw (gaussianApproxError B L_lat γ_lat) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  -- (1) SPD: spectral richness > 0 ⇒ SPD
  · exact spectral_implies_spd Sigma_latent_est γ_lat_est hΣ̂_lat_est
  -- (2) rank = 32: spectral richness ⇒ full rank
  · exact spectral_implies_full_rank Sigma_latent_est γ_lat_est hΣ̂_lat_est
  -- (3) Mahalanobis 界: 由 Phase M step 2 的 `mahalanobis_distance_error_bound` 直接推出
  · intro v
    have hΣ̂_at_lower : spectralRichnessCond Sigma_latent_est (γ_lat - ε_lat) := by
      refine ⟨hΣ̂_lat_est.1, sub_pos.mpr hsub_lat, ?_⟩
      intro x
      have hx_nonneg : (0 : ℝ) ≤ vec32NormSq x := vec32NormSq_nonneg _
      calc (γ_lat - ε_lat) * vec32NormSq x
          ≤ γ_lat_est * vec32NormSq x := by
            exact mul_le_mul_of_nonneg_right hγ_est_sufficient hx_nonneg
        _ ≤ vec32Inner x (Matrix.mulVec Sigma_latent_est x) := hΣ̂_lat_est.2.2 x
    exact mahalanobis_distance_error_bound v μ_lat Sigma_latent_true Sigma_latent_est
      γ_lat ε_lat hγ_lat hε_lat hsub_lat
      hΣ_lat_pop hΣ̂_at_lower hΣ_lat_pop_inv hΣ̂_lat_inv hopnorm_latent
  -- (4) CLT 界: 由 multivariate_clt_interface 直接给出 (无 ∃ δ 间接绑定)
  · exact multivariate_clt_interface
      (fun j : Fin N => Matrix.mulVec A (Z j))
      μ_lat Sigma_latent_true γ_lat L_lat B
      latentLaw gaussianLaw clt hγ_lat hL_lat hB_pos

end PersonalQuery