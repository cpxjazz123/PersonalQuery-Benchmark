/-
# PersonalQuery.PCAFitting — PCA 投影后的 Gaussian 拟合误差界

本模块把 Stage 03 的全局 PCA + per-user Gaussian 拟合严格形式化。

## 数学设计

设 raw 空间维数 `|ι|`, PCA 基 `W : Matrix ι (Fin 32) ℝ` 满足 `Wᵀ W = 1`
（列正交归一）。对任一用户 `u`：

* raw mean     `μ_u`
* raw covariance  `S_u`
* raw sample covariance  `S_u_hat`

严格恒等式（不是近似）：

* PCA 投影后的 mean:     `μ_u^{PCA} = Wᵀ (μ_u - μ_global)`
* PCA 投影后的 covariance: `C_u = Wᵀ S_u W`
* PCA 投影后的 sample covariance: `Ĉ_u = Wᵀ S_u_hat W`

## 关键观测

由于 `Wᵀ W = 1`, 对所有 `y : Fin 32 → ℝ`:

  `yᵀ C_u y = (Wy)ᵀ S_u (Wy)`  (transposed quadratic form)
  `‖Wy‖ = ‖y‖`                  (norm preservation)

所以 population spectral richness 不需要 raw 空间"满秩"，只需要用户在 PCA 选出的
32 个方向上具有足够 variation。

## 误差链

由 `‖Ĉ_u - C_u‖ ≤ ε` 与 `λ_min(C_u) ≥ γ`, 推出:

* `λ_min(Ĉ_u) ≥ γ - ε`        ⇒ SPD + rank 32 + Cholesky
* `‖Ĉ_u⁻¹ - C_u⁻¹‖ ≤ ε / (γ(γ-ε))`    (inverse covariance error)
* `|D̂² - D²| ≤ ‖v‖² · ε / (γ(γ-ε))`   (Mahalanobis error)

## 实现说明

本模块只引入新的 PCA 形式化, 严格证明投影恒等式与到 spectral richness 的桥接。
`spectralRichnessCond` / `cov_close_implies_spectral` 在这里**就地定义**，
与 `PersonalQuery.RuleEligibility` 中对应接口保持一致但本模块不依赖之，
以避免模块间传递错误的累积。本模块不实现 Cholesky / LDL 那部分。
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Diagonal
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Matrix.Symmetric
import Mathlib.Analysis.SpecialFunctions.Log.Basic

import PersonalQuery.Basic

namespace PersonalQuery

variable {I : Type*} [Fintype I] [DecidableEq I]

/-! ### 局部 SPD 接口（与 Gaussian.Mat32.SPD 一致） -/

/-- Mat32 SPD 谓词：对称 + 严格正定（∀ x ≠ 0, 0 < xᵀ M x）。
    这是 Stage 04 Gaussian 拟合成功拟合的最终判定条件。 -/
def Mat32.SPD (M : Mat32) : Prop :=
  Matrix.IsSymm M ∧ ∀ x : Vec32, x ≠ 0 → 0 < vec32Inner x (Matrix.mulVec M x)

/-! ### 局部 spectral richness 接口（与 RuleEligibility 一致） -/

/-- `λ_min(C) ≥ γ` 的二次型表达：C 对称且所有方向的 Rayleigh 二次型至少为 `γ ||x||²`。 -/
def spectralRichnessCond (C : Mat32) (γ : ℝ) : Prop :=
  Matrix.IsSymm C ∧ 0 < γ ∧
    ∀ x : Vec32, γ * vec32NormSq x ≤ vec32Inner x (Matrix.mulVec C x)

/-- 逐二次型定义的 covariance operator error。 -/
def covarianceClose (SigmaHat Sigma : Mat32) (ε : ℝ) : Prop :=
  ∀ x : Vec32,
    |vec32Inner x (Matrix.mulVec (SigmaHat - Sigma) x)| ≤ ε * vec32NormSq x

/-- `covarianceClose + spectralRichnessCond + ε < γ` ⇒ `spectralRichnessCond C' (γ - ε)`。
    与 RuleEligibility.cov_close_implies_spectral 一致。 -/
theorem cov_close_implies_spectral
    (C Sigma : Mat32) (γ ε : ℝ)
    (hsym : Matrix.IsSymm C)
    (hpop : spectralRichnessCond Sigma γ)
    (hclose : covarianceClose C Sigma ε)
    (hsub : ε < γ) :
    spectralRichnessCond C (γ - ε) := by
  refine ⟨hsym, sub_pos.mpr hsub, ?_⟩
  intro x
  have hpop_x : γ * vec32NormSq x ≤ vec32Inner x (Matrix.mulVec Sigma x) := hpop.2.2 x
  have hclose_x : |vec32Inner x (Matrix.mulVec (C - Sigma) x)| ≤ ε * vec32NormSq x := hclose x
  -- 拆 C = Sigma + (C - Sigma) (entry-wise)
  have hCdecomp : C = Sigma + (C - Sigma) := by
    ext i j
    show C i j = Sigma i j + (C i j - Sigma i j)
    rw [Matrix.sub_apply]
    ring
  rw [hCdecomp, Matrix.add_mulVec, vec32Inner_add_left]
  have hlower : -(ε * vec32NormSq x) ≤ vec32Inner x (Matrix.mulVec (C - Sigma) x) :=
    (abs_le.mp hclose_x).1
  have hlower' : -ε * vec32NormSq x ≤ vec32Inner x (Matrix.mulVec (C - Sigma) x) := by
    linarith [hlower]
  -- 用 hpop_x (给出下界 γ‖x‖²) + hlower' (给出 (C-Sigma)x 部分的下界 -ε‖x‖²)
  have : γ * vec32NormSq x + -ε * vec32NormSq x =
         (γ - ε) * vec32NormSq x := by ring
  linarith

/-! ### Raw 空间与 PCA 基 -/

/-- Raw 空间 PSD 谓词：∀ x : RawVec, 0 ≤ ⟨x, M x⟩。 -/
def RawMat.PSD (M : Matrix I I ℝ) : Prop :=
  ∀ x : I → ℝ, 0 ≤ (∑ i, x i * (Matrix.mulVec M x) i)

/-- **PCA 基**：`W : Matrix I (Fin 32) ℝ`，列正交归一（`Wᵀ * W = 1 : Mat32`）。
    这是全局共享的 fixed PCA，从 pooled covariance 一次性训练得到，
    不随用户变化。 -/
structure PCABasis where
  W : Matrix I (Fin 32) ℝ
  orthonormal : W.transpose * W = (1 : Mat32)

namespace PCABasis

/-- PCA 投影：z = Wᵀ x。 -/
def project [inst : Fintype I] (basis : @PCABasis I inst) (x : I → ℝ) : Vec32 :=
  Matrix.mulVec basis.W.transpose x

/-- PCA 投影后的 per-user mean: μ_u^{PCA} = Wᵀ (μ_u - μ_global)。 -/
def pcaMean [inst : Fintype I] (basis : @PCABasis I inst) (muG muU : I → ℝ) : Vec32 :=
  project basis (muU - muG)

/-- PCA 投影后的 per-user covariance: C = Wᵀ S W (32×32)。 -/
def pcaCov [inst : Fintype I] (basis : @PCABasis I inst) (S : Matrix I I ℝ) : Mat32 :=
  basis.W.transpose * S * basis.W

/-- PCA 投影后的 per-user sample covariance: Ĉ = Wᵀ Ŝ W (32×32)。 -/
def pcaSampleCov [inst : Fintype I] (basis : @PCABasis I inst) (Sh : Matrix I I ℝ) : Mat32 :=
  basis.W.transpose * Sh * basis.W

/-! ### 严格恒等式 -/

/-- `pcaMean` 严格等于 `Wᵀ (μ_u - μ_global)`（直接由定义）。 -/
@[simp]
theorem pcaMean_eq (basis : PCABasis) (muG muU : I → ℝ) :
    pcaMean basis muG muU = Matrix.mulVec (basis.W.transpose) ((muU - muG)) := rfl

/-- `pcaCov` 严格等于 `Wᵀ S W`（直接由定义）。 -/
@[simp]
theorem pcaCov_eq (basis : PCABasis) (S : Matrix I I ℝ) :
    pcaCov basis S = basis.W.transpose * S * basis.W := rfl

/-- **transposed quadratic form**:
    `yᵀ (Wᵀ S W) y = (Wy)ᵀ S (Wy)`。 -/
theorem quadratic_form (basis : PCABasis) (S : Matrix I I ℝ) (y : Vec32) :
    vec32Inner y (Matrix.mulVec (pcaCov basis S) y) =
      ∑ j : I, (Matrix.mulVec basis.W y) j * (Matrix.mulVec S (Matrix.mulVec basis.W y)) j := by
  -- 关键步骤: dotProduct y (Wᵀ v) = (y ⬝ Wᵀ) v = (W y) ⬝ v
  unfold pcaCov vec32Inner
  rw [Matrix.mulVec_mulVec (A := basis.W.transpose * S) (B := basis.W) (v := y)]
  rw [Matrix.mulVec_mulVec (A := basis.W.transpose) (B := S) (v := Matrix.mulVec basis.W y)]
  rw [Matrix.dotProduct_mulVec y basis.W.transpose (Matrix.mulVec S (Matrix.mulVec basis.W y))]
  rw [Matrix.vecMul_transpose y basis.W]
  unfold Matrix.dotProduct
  rfl

/-- **norm preservation**: `Wᵀ W = I` 推出 `‖Wy‖² = ‖y‖²`。 -/
theorem preserves_norm (basis : PCABasis) (y : Vec32) :
    vec32NormSq (Matrix.mulVec basis.W y) = vec32NormSq y := by
  unfold vec32NormSq vec32Inner Matrix.mulVec
  simp only [dotProduct]
  rw [Finset.sum_mul, Finset.sum_mul]
  rw [Finset.sum_comm]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro k' _
  rw [← Finset.sum_mul, ← Finset.sum_mul]
  show y k * y k' * ∑ i, basis.W i k * basis.W i k' =
       y k * y k' * (basis.W.transpose * basis.W) k k'
  rw [Matrix.mul_apply, Matrix.transpose_apply]
  rw [basis.orthonormal]
  simp only [Matrix.one_apply]
  rw [Finset.sum_ite_eq _ k (fun _ => (1 : ℝ))]
  simp

end PCABasis

open PCABasis

/-! ### 对称性 -/

/-- `pcaCov basis S` 在 `S` 对称时对称: `(Wᵀ S W)ᵀ = Wᵀ Sᵀ W = Wᵀ S W`。 -/
theorem PCABasis.pcaCov_symmetric (basis : PCABasis) (S : Matrix I I ℝ)
    (hS : Matrix.IsSymm S) :
    Matrix.IsSymm (basis.pcaCov S) := by
  intro i j
  show ((basis.W.transpose * S * basis.W)ᵀ) i j = (basis.W.transpose * S * basis.W) i j
  rw [Matrix.transpose_mul (A := basis.W.transpose * S) (B := basis.W)]
  rw [Matrix.transpose_mul (A := basis.W.transpose) (B := S)]
  rw [Matrix.mul_apply (A := basis.W.transpose * S.transpose) (B := basis.W)]
  rw [Matrix.mul_apply (A := basis.W.transpose) (B := S.transpose)]
  rw [Matrix.transpose_apply]
  rw [Matrix.transpose_apply]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro l _
  rw [hS l k]
  ring

/-! ### PCA 投影后的 spectral richness 桥 -/

/-- **PCA 子空间 spectral richness 条件**：在 PCA 选出的 32 个方向上，
    `S` 至少提供 γ 的 Rayleigh 二次型下界。
    等价于 raw S 在 col(W) 上是 spectral richness 的. -/
def pcaSubspaceRichness (basis : PCABasis) (S : Matrix I I ℝ) (γ : ℝ) : Prop :=
  0 < γ ∧ ∀ y : Vec32,
    γ * vec32NormSq y ≤
      ∑ j : I, (Matrix.mulVec basis.W y) j * (Matrix.mulVec S (Matrix.mulVec basis.W y)) j

/-- PCA 子空间 spectral richness 直接推出 32-dim spectral richness (假设 S 对称)。 -/
theorem pca_subspace_implies_spectral (basis : PCABasis) (S : Matrix I I ℝ)
    (hS : Matrix.IsSymm S) (γ : ℝ)
    (h : pcaSubspaceRichness basis S γ) :
    spectralRichnessCond (basis.pcaCov S) γ := by
  refine ⟨?_, h.1, ?_⟩
  · exact basis.pcaCov_symmetric S hS
  · intro y
    have h_quad : vec32Inner y (Matrix.mulVec (basis.pcaCov S) y) =
                  ∑ j : I, (Matrix.mulVec basis.W y) j * (Matrix.mulVec S (Matrix.mulVec basis.W y)) j :=
      basis.quadratic_form S y
    rw [h_quad]
    exact h.2 y

/-! ### PCA 投影后的 covariance close 桥 -/

/-- 投影后 covariance 的 operator-norm 误差: `‖Ĉ - C‖ ≤ ε`。 -/
def pcaCovClose (basis : PCABasis) (Sh S : Matrix I I ℝ) (ε : ℝ) : Prop :=
  ∀ y : Vec32,
    |vec32Inner y ((basis.pcaSampleCov Sh - basis.pcaCov S) y)|
      ≤ ε * vec32NormSq y

/-- PCA 投影后的谱桥: `pcaSubspaceRichness + pcaCovClose + ε < γ` ⇒
    `spectralRichnessCond Ĉ (γ - ε)`。 -/
theorem pca_cov_close_implies_spectral
    (basis : PCABasis) (Sh S : Matrix I I ℝ) (γ ε : ℝ)
    (hS : Matrix.IsSymm S) (hSh : Matrix.IsSymm Sh)
    (hpop : pcaSubspaceRichness basis S γ)
    (hclose : pcaCovClose basis Sh S ε)
    (hsub : ε < γ) :
    spectralRichnessCond (basis.pcaSampleCov Sh) (γ - ε) := by
  -- Step 1: 把 pcaSubspaceRichness 桥接为 spectralRichnessCond (pcaCov S) γ
  have hC_spec : spectralRichnessCond (basis.pcaCov S) γ :=
    pca_subspace_implies_spectral basis S hS γ hpop
  have hC_symm : Matrix.IsSymm (basis.pcaCov S) :=
    basis.pcaCov_symmetric S hS
  have hCh_symm : Matrix.IsSymm (basis.pcaSampleCov Sh) :=
    basis.pcaCov_symmetric Sh hSh
  -- Step 2: 直接套用 cov_close_implies_spectral (pcaCovClose = covarianceClose 形式)
  exact cov_close_implies_spectral (basis.pcaSampleCov Sh) (basis.pcaCov S) γ ε
    hCh_symm hC_spec hclose hsub

/-! ### 主定理: PCA richness → SPD 整链 -/

/-- **主定理**: 对称 S 上的 PCA subspace richness + 协方差误差 ε < γ ⇒
    投影 sample covariance Ĉ 是 SPD。 -/
theorem pca_fitting_implies_spd
    (basis : PCABasis) (Sh S : Matrix I I ℝ) (γ ε : ℝ)
    (hS : Matrix.IsSymm S) (hSh : Matrix.IsSymm Sh)
    (hpop : pcaSubspaceRichness basis S γ)
    (hclose : pcaCovClose basis Sh S ε)
    (hsub : ε < γ) :
    Mat32.SPD (basis.pcaSampleCov Sh) := by
  -- Step 1: 获得 spectralRichnessCond Ĉ (γ - ε)
  have hspec : spectralRichnessCond (basis.pcaSampleCov Sh) (γ - ε) :=
    pca_cov_close_implies_spectral basis Sh S γ ε hS hSh hpop hclose hsub
  -- Step 2: spectralRichnessCond ⟹ SPD (局部证明)
  refine ⟨hspec.1, ?_⟩
  intro x hx
  have hnorm : 0 < vec32NormSq x := by
    unfold vec32NormSq vec32Inner
    have hex : ∃ i : Fin 32, x i ≠ 0 := by
      by_contra h; apply hx; funext i; by_contra hi; exact h ⟨i, hi⟩
    obtain ⟨i, hi⟩ := hex
    apply Finset.sum_pos' (fun j _ => mul_self_nonneg (x j))
    exact ⟨i, Finset.mem_univ _, mul_self_pos.mpr hi⟩
  have hprod : 0 < (γ - ε) * vec32NormSq x := mul_pos (sub_pos.mpr hsub) hnorm
  have hineq : (γ - ε) * vec32NormSq x ≤ vec32Inner x ((basis.pcaSampleCov Sh) x) :=
    hspec.2.2 x
  exact lt_of_lt_of_le hprod hineq

end PersonalQuery
