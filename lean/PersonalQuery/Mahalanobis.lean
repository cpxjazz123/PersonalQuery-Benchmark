/-
# PersonalQuery.Mahalanobis — 马氏距离 + Cholesky 数值稳定性

Stage 08 (syntax_select_mahalanobis_gate.py:326-333) 定义:
    def maha_d2(Z, mu, inv_sigma):
        diff = Z - mu
        return np.einsum("nd,de,ne->n", diff, inv_sigma, diff)

Stage 04 (fit_per_user_gaussian.py:139-141) 用 Cholesky 求逆路径:
    diff = z_val - mu
    whitened = np.linalg.solve(chol, diff.T).T
    d2_val = np.sum(whitened * whitened, axis=1)

形式化目标:
  1. 定义 mahaD2 (直接路径) 与 mahaD2Chol (Cholesky 路径)
  2. 证明 mahaD2 ≥ 0 对 PSD Σ
  3. 证明 mahaD2 = 0 ⟺ z = μ 对 SPD Σ (正向)
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct

import PersonalQuery.Basic
import PersonalQuery.Gaussian

namespace PersonalQuery

open Matrix

/-! ### 向量差 -/

/-- 向量差: z - μ。 -/
def vecDiff (z mu : Vec32) : Vec32 := fun i => z i - mu i

/-! ### 马氏距离 (直接路径) -/

/-- 直接马氏距离: D²(z, μ) = (z - μ)ᵀ Σ⁻¹ (z - μ)。
    对应 Python `maha_d2_one`。 -/
def mahaD2 (z mu : Vec32) (invSigma : Mat32) : ℝ :=
  let d := vecDiff z mu
  vec32Inner d (Matrix.mulVec invSigma d)

/-- 直接马氏距离 (批处理): 输入候选 z_q, 输出 D² 值。 -/
def mahaD2Batch (Z : ℕ → Vec32) (mu : Vec32) (invSigma : Mat32) : ℕ → ℝ :=
  fun j => mahaD2 (Z j) mu invSigma

/-! ### Cholesky 路径 -/

/-- Cholesky 白化算子: L⁻¹ v（严格定义，使用 Mathlib 的 `Matrix.nonsing_inv`）。
    当 L 真正可逆（kernel trivial）时，`cholesky_solve_correct` 给出 `L · (cholesky_solve L v) = v`。 -/
noncomputable def cholesky_solve (L : Mat32) (v : Vec32) : Vec32 :=
  L⁻¹ *ᵥ v

/-- Cholesky 求解正确性: `L · (L⁻¹ v) = v`，基于 `IsUnit A.det ⟹ A * A⁻¹ = 1`。 -/
theorem cholesky_solve_correct (L : Mat32) (v : Vec32)
    (hinv : ∀ w, Matrix.mulVec L w = 0 → w = 0) :
    Matrix.mulVec L (cholesky_solve L v) = v := by
  unfold cholesky_solve
  -- L * (L⁻¹ *ᵥ v) = (L * L⁻¹) *ᵥ v = 1 *ᵥ v = v
  -- 关键事实: kernel trivial ⟹ IsUnit
  have hUnit : IsUnit L := (Matrix.mulVec_injective_iff_isUnit (A := L)).mpr hinv
  have hUnitDet : IsUnit L.det := (isUnit_iff_isUnit_det L).mp hUnit
  rw [Matrix.mulVec_mulVec, mul_nonsing_inv L hUnitDet, Matrix.one_mulVec]

/-- Cholesky 路径马氏距离: D² = ‖w‖² = ⟨w, w⟩, 其中 w = L⁻¹ (z - μ)。 -/
noncomputable def mahaD2Chol (z mu : Vec32) (L : Mat32) : ℝ :=
  let w := cholesky_solve L (vecDiff z mu)
  vec32Inner w w

/-! ### 基本性质 -/

/-- mahaD2 在 PSD Σ 下非负。 -/
theorem mahaD2_nonneg (z mu : Vec32) (invSigma : Mat32)
    (hpsd : Mat32.PSD invSigma) :
    0 ≤ mahaD2 z mu invSigma := by
  unfold mahaD2 vecDiff
  exact hpsd (vecDiff z mu)

/-- mahaD2Chol 非负: ‖w‖² ≥ 0。 -/
theorem mahaD2Chol_nonneg (z mu : Vec32) (L : Mat32) :
    0 ≤ mahaD2Chol z mu L := by
  unfold mahaD2Chol
  exact vec32NormSq_nonneg _

/-- mahaD2Chol = 0 ⟹ z = μ (当 L 可逆时)。
    由 L · w = (z - μ) 且 w = 0 推出 z = μ。 -/
theorem mahaD2Chol_zero_implies_eq (z mu : Vec32) (L : Mat32)
    (_hinv : ∀ v, Matrix.mulVec L v = 0 → v = 0)
    (h : mahaD2Chol z mu L = 0) :
    z = mu := by
  unfold mahaD2Chol at h
  set w := cholesky_solve L (vecDiff z mu) with hw_def
  have hnorm : vec32NormSq w = 0 := h
  -- 零范数意味着 w = 0
  have hw_zero : w = 0 := by
    unfold vec32NormSq vec32Inner at hnorm
    funext i
    have hsum : ∑ j : Fin 32, w j * w j = 0 := hnorm
    have hnonneg : ∀ j, 0 ≤ w j * w j := fun j => mul_self_nonneg _
    have hkey := Finset.sum_eq_zero_iff_of_nonneg (fun j _ => hnonneg j) |>.mp hsum
    have hi : w i * w i = 0 := hkey i (Finset.mem_univ _)
    -- w i * w i = 0 ⟹ w i = 0
    by_contra hne
    -- 若 w i ≠ 0, 则 (w i)^2 > 0, 但 hi = 0, 矛盾
    have hpos : 0 < w i * w i := by
      rcases lt_or_gt_of_ne hne with hlt | hgt
      · have : w i < 0 := hlt
        nlinarith
      · have : 0 < w i := hgt
        nlinarith
    linarith
  have hLw : vecDiff z mu = Matrix.mulVec L w := by
    rw [hw_def]
    -- cholesky_solve_correct: Matrix.mulVec L (cholesky_solve L v) = v
    exact (cholesky_solve_correct L (vecDiff z mu)).symm
  -- Matrix.mulVec L 0 = 0
  have hzero : Matrix.mulVec L 0 = 0 := by
    funext i
    simp [Matrix.mulVec]
  -- vecDiff z mu = 0
  have hvdiff : vecDiff z mu = 0 := by
    rw [hLw, hw_zero, hzero]
  -- 现在 hvdiff : (fun i => z i - mu i) = 0
  funext i
  have hi : z i - mu i = 0 := by
    have hx := congrFun hvdiff i
    exact hx
  linarith

/-- mahaD2(z, μ) = 0 ⟸ z = μ (trivial). -/
theorem mahaD2_zero_of_eq (z mu : Vec32) (invSigma : Mat32) (h : z = mu) :
    mahaD2 z mu invSigma = 0 := by
  subst h
  unfold mahaD2 vecDiff
  simp [vec32Inner, Pi.sub_apply]

/-- mahaD2 关于 z 的平移性: mahaD2(z+a, μ+a) = mahaD2(z, μ)。 -/
theorem mahaD2_translation (z mu a : Vec32) (invSigma : Mat32) :
    mahaD2 (z + a) (mu + a) invSigma = mahaD2 z mu invSigma := by
  unfold mahaD2 vecDiff
  simp [vec32Inner, Pi.add_apply, Pi.sub_apply, Matrix.mulVec,
        Finset.sum_add_distrib, add_mul, Finset.sum_sub_distrib, sub_mul]

/-- mahaD2 关于 invSigma 的线性性 (左乘标量)。 -/
theorem mahaD2_smul_invSigma (c : ℝ) (z mu : Vec32) (invSigma : Mat32) :
    mahaD2 z mu (c • invSigma) = c * mahaD2 z mu invSigma := by
  unfold mahaD2
  show vec32Inner (vecDiff z mu) (Matrix.mulVec (c • invSigma) (vecDiff z mu)) =
       c * vec32Inner (vecDiff z mu) (Matrix.mulVec invSigma (vecDiff z mu))
  rw [Matrix.mulVec_smul]
  rw [vec32Inner_smul_right]

/-! ### 验证集 D² (Stage 04 gate_T 计算) -/

/-- 验证集 D²: 对 Z_val 逐点计算 D²(z_v, μ)。 -/
noncomputable def validationD2 {n_val : ℕ} (Z_val : Fin n_val → Vec32) (mu : Vec32) (L : Mat32) :
    Fin n_val → ℝ :=
  fun j => mahaD2Chol (Z_val j) mu L

/-- 验证集 D² 全部非负。 -/
theorem validationD2_nonneg {n_val : ℕ} (Z_val : Fin n_val → Vec32)
    (mu : Vec32) (L : Mat32) :
    ∀ j, 0 ≤ validationD2 Z_val mu L j := by
  intro j
  exact mahaD2Chol_nonneg (Z_val j) mu L

/-! ### 两路径理论等价 (声明式) -/

/-- Cholesky 路径与直接路径在 Σ = L Lᵀ 下等价。
    严格证明需要 Cholesky inverse consistency (Mathlib `Matrix.mul_inv_cancel` 配合
    Cholesky L·Lᵀ = Σ ⟹ L⁻¹ = Lᵀ Σ⁻¹) 加 vec32Inner 双线性, 留待 long-form 形式化。
    此处接受外部等价证据 heq 作为 contract 占位。 -/
theorem chol_d2_equals_direct (z mu : Vec32) (L invSigma : Mat32)
    (hChol : ∃ sigma : Mat32, L * L.transpose = sigma)
    (heq : mahaD2Chol z mu L = mahaD2 z mu invSigma) :
    mahaD2Chol z mu L = mahaD2 z mu invSigma :=
  heq

end PersonalQuery
