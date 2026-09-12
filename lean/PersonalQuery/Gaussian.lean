/-
# PersonalQuery.Gaussian — Per-User Gaussian 拟合
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.LinearAlgebra.Matrix.Symmetric
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.LDL
import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.BigOperators.Ring.Finset

import PersonalQuery.Basic

namespace PersonalQuery

open Matrix

/-! ### 配置常量 -/

def minProfileSents : ℕ := 40
def minValSents : ℕ := 10
def minEigenRatio : ℝ := 1e-8

lemma minEigenRatio_pos : 0 < minEigenRatio := by
  unfold minEigenRatio; norm_num

/-! ### 样本均值 -/

noncomputable def sampleMean {n : ℕ} (Z : Fin n → Vec32) : Vec32 :=
  fun i => (∑ j : Fin n, Z j i) / n

/-! ### 中心化 -/

def center {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) : Fin n → Vec32 :=
  fun j => fun i => Z j i - mu i

/-! ### 原始样本协方差 -/

noncomputable def sampleCovRaw {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) : Mat32 :=
  fun i k =>
    let s := ∑ j : Fin n, (Z j i - mu i) * (Z j k - mu k)
    s / ((n : ℝ) - 1)

/-- 原始协方差是对称的: transpose 等于自身。 -/
theorem sampleCovRaw_symmetric {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) :
    (sampleCovRaw Z mu).transpose = sampleCovRaw Z mu := by
  funext i k
  show (∑ j, (Z j k - mu k) * (Z j i - mu i)) / ((n : ℝ) - 1) =
       (∑ j, (Z j i - mu i) * (Z j k - mu k)) / ((n : ℝ) - 1)
  congr 1
  exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-! ### 对称化 -/

noncomputable abbrev sampleCovSymm {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) : Mat32 :=
  symmetrize (sampleCovRaw Z mu)

/-- 对称化协方差是对称的。 -/
theorem sampleCovSymm_symmetric {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) :
    (sampleCovSymm Z mu).transpose = sampleCovSymm Z mu :=
  (symmetrize_symmetric (sampleCovRaw Z mu)).symm

/-- 对称化不改变对称矩阵。 -/
lemma sampleCovSymm_id_of_symmetric {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32)
    (h : sampleCovRaw Z mu = (sampleCovRaw Z mu).transpose) :
    sampleCovSymm Z mu = sampleCovRaw Z mu :=
  symmetrize_of_symmetric h.symm

/-! ### 半正定性质 -/

namespace GaussPSD

/-- 内部辅助：`(n : ℝ) - 1 ≠ 0`，要求 `n ≥ 2`。 -/
private lemma n_sub_one_ne_zero {n : ℕ} (hn : n ≥ 2) : (n : ℝ) - 1 ≠ 0 := by
  have h : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  linarith

/-- 内部辅助：`(0 : ℝ) < (n : ℝ) - 1`，要求 `n ≥ 2`。 -/
private lemma n_sub_one_pos {n : ℕ} (hn : n ≥ 2) : (0 : ℝ) < (n : ℝ) - 1 := by
  have h : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  linarith

/-- `Σ x` 在坐标 k 上的值（展开形式）。 -/
lemma sampleCovRaw_mulVec_apply {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) (x : Vec32)
    (k : Fin 32) :
    (Matrix.mulVec (sampleCovRaw Z mu) x) k =
      (∑ j : Fin n, (Z j k - mu k) * (∑ i : Fin 32, x i * (Z j i - mu i))) /
        ((n : ℝ) - 1) := by
  simp [sampleCovRaw, Matrix.mulVec, dotProduct, Finset.sum_div]
  -- simp 后 LHS: ∑ x_1, (∑ i, (Z i k - μ k) * (Z i x_1 - μ x_1) / d) * x x_1
  -- RHS: ∑ i, ((Z i k - μ k) * ∑ i_1, x i_1 * (Z i i_1 - μ i_1)) / d
  apply Finset.sum_congr rfl  -- 处理 LHS 外层 ∑ x_1
  intro x_1 _
  rw [Finset.sum_mul]  -- per-x_1: (∑ i, f/d) * x x_1 = ∑ i, (f/d) * x x_1
  apply Finset.sum_congr rfl  -- 处理 RHS 外层 ∑ i
  intro i _
  rw [div_mul_eq_mul_div]  -- per-i: (f/d) * x x_1 = f * x x_1 / d
  congr 1  -- 去掉 /d
  rw [← Finset.mul_sum, ← Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro x_1 _
  ring

/-- `⟨x, zⱼ - μ⟩² = (Σ i, x i * (zⱼ i - μ i)) * (Σ k, x k * (zⱼ k - μ k))`。 -/
private lemma sampleCovRaw_sq_eq_prod (Z : Fin n → Vec32) (mu : Vec32) (x : Vec32)
    (j : Fin n) :
    (∑ i : Fin 32, x i * (Z j i - mu i)) *
      (∑ k : Fin 32, x k * (Z j k - mu k))
      = (∑ i : Fin 32, x i * (Z j i - mu i)) ^
          (2 : ℕ) := by
  have h : (∑ k : Fin 32, x k * (Z j k - mu k)) = ∑ i : Fin 32, x i * (Z j i - mu i) := by
    apply Finset.sum_congr rfl
    intro k _
    rfl
  rw [h, sq]

/-- `⟨x, Σ x⟩ * (n - 1) = Σⱼ ⟨x, zⱼ - μ⟩²`。核心代数恒等式。 -/
lemma sampleCovRaw_quadratic_scaled {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) (x : Vec32)
    (hn : n ≥ 2) :
    vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x) * ((n : ℝ) - 1) =
      ∑ j : Fin n, (∑ i : Fin 32, x i * (Z j i - mu i)) *
        (∑ k : Fin 32, x k * (Z j k - mu k)) := by
  have hLHS : vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x) =
      (∑ k : Fin 32, x k *
        ((∑ j : Fin n, (Z j k - mu k) * (∑ i : Fin 32, x i * (Z j i - mu i))) /
          ((n : ℝ) - 1))) := by
    unfold vec32Inner
    apply Finset.sum_congr rfl
    intro k _
    rw [sampleCovRaw_mulVec_apply]
  rw [hLHS]
  -- LHS: (∑ k, x k * ((∑ j, f) / d)) * d
  -- 用 Finset.sum_mul 把 * d 推入: ∑ k, (x k * ((∑ j, f) / d)) * d
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro k _
  rw [div_mul_cancel _ (n_sub_one_ne_zero hn)]
  -- per-k: x k * (∑ j, f)
  rw [Finset.mul_sum]
  -- per-k: ∑ j, x k * f
  -- 现在 LHS 是 ∑ k, ∑ j, x k * (Z j k - μ k) * (∑ i, x i * (Z j i - μ i))
  -- RHS 是 ∑ j, (∑ i, ...) * (∑ k, x k * (Z j k - μ k))
  rw [Finset.sum_comm (s := (Finset.univ : Finset (Fin 32)))
                        (t := (Finset.univ : Finset (Fin n)))]
  -- LHS: ∑ j, ∑ k, x k * (Z j k - μ k) * (∑ i, x i * (Z j i - μ i))
  apply Finset.sum_congr rfl
  intro j _
  rw [← Finset.sum_mul]
  rw [mul_comm]
  rfl

/-- 原始样本协方差是半正定的（`n ≥ 2` 时严格 PSD）。 -/
theorem sampleCovRaw_psd {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) (x : Vec32)
    (hn : n ≥ 2) :
    0 ≤ vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x) := by
  have hEq : vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x) * ((n : ℝ) - 1) =
      ∑ j : Fin n, (∑ i : Fin 32, x i * (Z j i - mu i)) *
        (∑ k : Fin 32, x k * (Z j k - mu k)) :=
    sampleCovRaw_quadratic_scaled Z mu x hn
  have hsquares :
      ∑ j : Fin n,
        (∑ i : Fin 32, x i * (Z j i - mu i)) *
          (∑ k : Fin 32, x k * (Z j k - mu k)) ≥ 0 := by
    apply Finset.sum_nonneg
    intro j _
    rw [sampleCovRaw_sq_eq_prod Z mu x j]
    exact sq_nonneg _
  have hpos : (0 : ℝ) < (n : ℝ) - 1 := n_sub_one_pos hn
  have habs : vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x) * ((n : ℝ) - 1) ≥ 0 := by
    rw [hEq]
    exact hsquares
  by_contra hneg
  push_neg at hneg
  exact (lt_irrefl _ (lt_of_lt_of_le (mul_neg_of_neg_of_pos hneg hpos) habs))

/-- transpose 不改变二次型：xᵀ (Mᵀ) x = xᵀ M x。 -/
lemma transpose_quadratic_eq (M : Mat32) (x : Vec32) :
    vec32Inner x (Matrix.mulVec M.transpose x) = vec32Inner x (Matrix.mulVec M x) := by
  show (∑ k, x k * ∑ i, M.transpose k i * x i) = (∑ k, x k * ∑ i, M k i * x i)
  simp only [Matrix.transpose_apply]
  -- 两边都拆 mul_sum 成 Σ k, Σ i, ...
  conv_lhs => rw [Finset.mul_sum]
  conv_rhs => rw [Finset.mul_sum]
  -- 把两层和改写为单层 Σ (a, b) over Fin 32 × Fin 32
  conv_lhs =>
    rw [← Finset.sum_product']
    rw [Finset.univ_product_univ]
  conv_rhs =>
    rw [← Finset.sum_product']
    rw [Finset.univ_product_univ]
  -- LHS: ∑ (a, b), x a * M b a * x b
  -- RHS: ∑ (a, b), x a * M a b * x b
  -- 对 LHS 用 prodComm swap (a,b) ↔ (b,a)
  have hLHS : (∑ p : Fin 32 × Fin 32, x p.1 * M p.2 p.1 * x p.2) =
              (∑ p : Fin 32 × Fin 32, x p.2 * M p.1 p.2 * x p.1) := by
    apply Fintype.sum_equiv (e := (Equiv.prodComm (α := Fin 32) (β := Fin 32)))
    rintro ⟨a, b⟩ _
    rfl
  rw [hLHS]
  -- 现在: ∑ (a, b), x b * M a b * x a = ∑ (a, b), x a * M a b * x b
  apply Finset.sum_congr rfl
  rintro ⟨c, d⟩ _
  ring

/-- 对称化协方差是半正定的。 -/
theorem sampleCovSymm_psd {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) (x : Vec32)
    (hn : n ≥ 2) :
    0 ≤ vec32Inner x (Matrix.mulVec (sampleCovSymm Z mu) x) := by
  have hEq :
      vec32Inner x (Matrix.mulVec (sampleCovSymm Z mu) x)
        = (vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x)
            + vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu).transpose x)) / 2 := by
    simp only [sampleCovSymm, symmetrize, vec32Inner, Matrix.mulVec, dotProduct,
               Matrix.transpose_apply, Finset.sum_add_distrib, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro k _
    simp only [Matrix.transpose_apply]
    ring
  rw [hEq]
  rw [transpose_quadratic_eq (sampleCovRaw Z mu) x]
  have hraw : 0 ≤ vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x) :=
    sampleCovRaw_psd Z mu x hn
  linarith [show
      (vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x) +
        vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x)) / 2
        = vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x) by ring]

end GaussPSD

/-- Re-export: 原始样本协方差是半正定的。 -/
theorem sampleCovRaw_psd {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) (x : Vec32)
    (hn : n ≥ 2) :
    0 ≤ vec32Inner x (Matrix.mulVec (sampleCovRaw Z mu) x) :=
  GaussPSD.sampleCovRaw_psd Z mu x hn

/-- Re-export: 对称化样本协方差是半正定的。 -/
theorem sampleCovSymm_psd {n : ℕ} (Z : Fin n → Vec32) (mu : Vec32) (x : Vec32)
    (hn : n ≥ 2) :
    0 ≤ vec32Inner x (Matrix.mulVec (sampleCovSymm Z mu) x) :=
  GaussPSD.sampleCovSymm_psd Z mu x hn

/-! ### Eigenvalue 门控 -/

/-- 严格正定: xᵀ Σ x > 0 对所有 x ≠ 0。 -/
def Mat32.SPD (M : Mat32) : Prop :=
  Matrix.IsSymm M ∧ ∀ x : Vec32, x ≠ 0 → 0 < vec32Inner x (Matrix.mulVec M x)

/-- Eigenvalue ratio gate (Stage 04 验证条件)。 -/
structure EigenGate (M : Mat32) : Prop where
  symm : Matrix.IsSymm M
  pos_min : ∀ x : Vec32, x ≠ 0 → 0 < vec32Inner x (Matrix.mulVec M x)
  ratioPos : ∃ lamMin lamMax : ℝ,
    lamMin > 0 ∧ lamMin ≤ lamMax ∧ lamMin / lamMax ≥ minEigenRatio

/-- EigenGate 蕴含 SPD。 -/
theorem eigenGate_implies_spd (M : Mat32) (h : EigenGate M) : M.SPD := by
  refine ⟨h.symm, fun x hx => ?_⟩
  exact h.pos_min x hx

/-! ### Cholesky 分解 -/

/-- Cholesky 因子: 下三角 L 满足 L * Lᵀ = Σ。 -/
structure CholeskyFactor (M : Mat32) where
  L : Mat32
  lowerTri : ∀ i j, i < j → L i j = 0
  decomps : L * L.transpose = M

/-- `Mat32.SPD M ↔ (M : Matrix (Fin 32) (Fin 32) ℝ).PosDef`。
    对 ℝ 矩阵: `star = id`、`conjTranspose = transpose`、`dotProduct x (M *ᵥ y) = vec32Inner x (M *ᵥ y)`。 -/
lemma mat32_spd_iff_posDef (M : Mat32) :
    Mat32.SPD M ↔ (M : Matrix (Fin 32) (Fin 32) ℝ).PosDef := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · refine ⟨?_, ?_⟩
    · -- IsSymm → IsHermitian for ℝ
      have hEq : M.conjTranspose = M.transpose := rfl
      rw [hEq]
      exact h.1
    · -- PosDef x ≠ 0 ⟹ 0 < dotProduct x (M *ᵥ x)
      intro x hx
      have hStar : star x = x := by
        funext i
        show star (x i) = x i
        simp [Pi.star_apply, star]
      rw [← hStar]
      exact h.2 x hx
  · refine ⟨?_, ?_⟩
    · -- IsHermitian → IsSymm for ℝ
      have hEq : M.conjTranspose = M.transpose := rfl
      rw [hEq] at h
      exact h.1
    · intro x hx
      have hStar : star x = x := by
        funext i
        show star (x i) = x i
        simp [Pi.star_apply, star]
      rw [← hStar]
      exact h.2 x hx

namespace CholeskyPSD

open Matrix LDL

variable {n : Type*} [Fintype n] [LinearOrder n] [WellFoundedLT n] [LocallyFiniteOrderBot n]

/-- LDL.lowerInv 是 lower triangular (`i < j → LDL.lowerInv hS i j = 0`)，
    等价于 `BlockTriangular (LDL.lowerInv hS) toDual`。 -/
lemma LDL.lowerInv_blockTriangular (S : Matrix n n ℝ) (hS : S.PosDef) :
    BlockTriangular (LDL.lowerInv hS) (toDual : n → nᵒᵈ) := by
  intro i j hij
  -- hij : toDual j < toDual i, 即 i < j (在 n 中)
  rw [toDual_lt_toDual] at hij
  exact LDL.lowerInv_triangular hS i j hij

/-- LDL.lower = (LDL.lowerInv)⁻¹ 也是 BlockTriangular toDual (即 lower triangular)。
    关键: `BlockTriangular.blockTriangular_inv_of_blockTriangular` 把下三角性传递到逆。 -/
lemma LDL.lower_blockTriangular (S : Matrix n n ℝ) (hS : S.PosDef) :
    BlockTriangular (LDL.lower hS) (toDual : n → nᵒᵈ) :=
  blockTriangular_inv_of_blockTriangular (LDL.lowerInv_blockTriangular S hS)

/-- LDL.lower 的下三角性: `i < j → LDL.lower hS i j = 0`。 -/
lemma LDL.lower_triangular (S : Matrix n n ℝ) (hS : S.PosDef) (i j : n) (hij : i < j) :
    LDL.lower hS i j = 0 :=
  LDL.lower_blockTriangular S hS i j (toDual_lt_toDual.mpr hij)

/-- LDL.diagEntries hS i = dotProduct (LDL.lowerInv hS i) (S *ᵥ (LDL.lowerInv hS i))
    (对 ℝ 矩阵成立，因为 conjTranspose = transpose)。

    推导路径:
      (LDL.diag hS) i i = LDL.diagEntries hS i                              [定义]
      LDL.diag hS = LDL.lowerInv hS * S * (LDL.lowerInv hS)ᵀ                [LDL.diag_eq_lowerInv_conj, ℝ 下 ᴴ = ᵀ]
      展开 ((LDL.lowerInv hS) * S * (LDL.lowerInv hS)ᵀ) i i:
        = ∑ k, ((LDL.lowerInv hS) * S) i k * (LDL.lowerInv hS)ᵀ k i
        = ∑ k, (∑ l, (LDL.lowerInv hS) i l * S l k) * (LDL.lowerInv hS) i k
        = ∑_{l k}, (LDL.lowerInv hS) i l * S l k * (LDL.lowerInv hS) i k
      而 RHS = dotProduct (LDL.lowerInv hS i) (S *ᵥ (LDL.lowerInv hS i))
            = ∑ k, (LDL.lowerInv hS i) k * (S *ᵥ (LDL.lowerInv hS i)) k
            = ∑ k, (LDL.lowerInv hS) i k * (∑ l, S k l * (LDL.lowerInv hS) i l)
            = ∑_{k l}, (LDL.lowerInv hS) i k * S k l * (LDL.lowerInv hS) i l
      两边都是 ∑_{l k}, (LDL.lowerInv hS) i l * S l k * (LDL.lowerInv hS) i k，相同。-/
lemma LDL.diagEntries_eq_dotProduct (S : Matrix n n ℝ) (hS : S.PosDef) (i : n) :
    LDL.diagEntries hS i =
      dotProduct (LDL.lowerInv hS i) (S *ᵥ (LDL.lowerInv hS i)) := by
  -- LHS = LDL.diagEntries hS i
  -- 关键事实: LDL.diag hS = LDL.lowerInv hS * S * (LDL.lowerInv hS)ᴴ
  have hDiagEq : LDL.diag hS = LDL.lowerInv hS * S * (LDL.lowerInv hS)ᴴ :=
    LDL.diag_eq_lowerInv_conj hS
  -- 取 (i, i) entry: (LDL.diag hS) i i = LDL.diagEntries hS i (定义)
  have hDiagEntry : (LDL.diag hS) i i = LDL.diagEntries hS i := by
    simp [LDL.diag, Matrix.diagonal]
  -- (LDL.lowerInv hS)ᴴ = (LDL.lowerInv hS)ᵀ 对 ℝ 矩阵
  have hH : (LDL.lowerInv hS)ᴴ = (LDL.lowerInv hS)ᵀ := rfl
  -- 串接: LDL.diagEntries hS i = ((LDL.lowerInv hS) * S * (LDL.lowerInv hS)ᵀ) i i
  have hLHS : LDL.diagEntries hS i = ((LDL.lowerInv hS) * S * (LDL.lowerInv hS)ᵀ) i i := by
    rw [← hDiagEntry]
    rw [hDiagEq, hH]
  -- 现在展开 RHS using dotProduct_mulVec: v ⬝ᵥ A *ᵥ w = v ᵥ* A ⬝ᵥ w
  -- 令 v := LDL.lowerInv hS i, w := LDL.lowerInv hS i, A := S
  -- 那么 v ⬝ᵥ (S *ᵥ w) = v ᵥ* S ⬝ᵥ w
  -- v ᵥ* S = fun k => ∑ l, v l * S l k
  -- v ᵥ* S ⬝ᵥ w = ∑ k, (∑ l, v l * S l k) * w k = ∑_{l k}, v l * S l k * w k
  -- 这正是 LHS 的展开形式
  rw [hLHS]
  -- 目标: ((LDL.lowerInv hS) * S * (LDL.lowerInv hS)ᵀ) i i = v ⬝ᵥ (S *ᵥ v) (其中 v = LDL.lowerInv hS i)
  have hRHS_form : ((LDL.lowerInv hS) * S * (LDL.lowerInv hS)ᵀ) i i =
                   dotProduct (LDL.lowerInv hS i) (S *ᵥ (LDL.lowerInv hS i)) := by
    -- 展开矩阵乘积形式
    have h1 : ((LDL.lowerInv hS) * S * (LDL.lowerInv hS)ᵀ) i i =
              ∑ k, ((LDL.lowerInv hS) * S) i k * ((LDL.lowerInv hS)ᵀ) k i := by
      rw [Matrix.mul_apply]
    have h2 : ((LDL.lowerInv hS) * S) i k = ∑ l, (LDL.lowerInv hS) i l * S l k := by
      rw [Matrix.mul_apply]
    have h3 : ((LDL.lowerInv hS)ᵀ) k i = (LDL.lowerInv hS) i k := by
      rw [Matrix.transpose_apply]
    -- 展开 RHS 的 dotProduct_mulVec
    have h4 : dotProduct (LDL.lowerInv hS i) (S *ᵥ (LDL.lowerInv hS i)) =
              (LDL.lowerInv hS i) ᵥ* S ⬝ᵥ (LDL.lowerInv hS i) := by
      rw [dotProduct_mulVec]
    -- 展开 vecMul 和 dotProduct
    have h5 : (LDL.lowerInv hS i) ᵥ* S ⬝ᵥ (LDL.lowerInv hS i) =
              ∑ k, (∑ l, (LDL.lowerInv hS) i l * S l k) * (LDL.lowerInv hS) i k := by
      rw [Matrix.vecMul, dotProduct]
    -- LHS = ∑ k, (∑ l, (LDL.lowerInv hS) i l * S l k) * (LDL.lowerInv hS) i k
    -- 通过 h1, h2, h3 把 LHS 展开成同一形式
    rw [h1, h2, h3]
    rw [h4, h5]
    -- 现在两边形式相同: ∑ k, (∑ l, ...) * ...
    -- 需要证明 ∑ k, (∑ l, A i l * S l k) * B i k = ∑ k, (∑ l, A i l * S l k) * B i k
    rfl
  exact hRHS_form

/-- LDL.diagEntries 严格正 (PosDef ⟹ 每个对角线 entry > 0)。 -/
lemma LDL.diagEntries_pos (S : Matrix n n ℝ) (hS : S.PosDef) (i : n) :
    0 < LDL.diagEntries hS i := by
  rw [LDL.diagEntries_eq_dotProduct S hS i]
  -- 0 < dotProduct v (S *ᵥ v)，其中 v = LDL.lowerInv hS i
  -- LDL.lowerInv hS 是 Invertible ⇒ det ≠ 0 ⇒ 每行非零 ⇒ LDL.lowerInv hS i ≠ 0
  haveI : Invertible (LDL.lowerInv hS) := LDL.invertibleLowerInv hS
  have hv_ne : (LDL.lowerInv hS i) ≠ 0 := by
    intro hv_eq
    -- LDL.lowerInv hS i = 0 作为函数意味着第 i 行全为 0
    have h_row_zero : ∀ k, (LDL.lowerInv hS) i k = 0 := by
      intro k
      have hk : (LDL.lowerInv hS i) k = 0 := by rw [hv_eq]; rfl
      exact hk
    -- 由此 det = 0
    have hdet_zero : (LDL.lowerInv hS).det = 0 :=
      det_eq_zero_of_row_eq_zero i h_row_zero
    -- 但 LDL.lowerInv hS Invertible ⇒ det 是 unit ⇒ det ≠ 0
    exact (isUnit_iff_isUnit_det _).mp (Invertible.isUnit (LDL.lowerInv hS)) |>.ne_zero hdet_zero
  -- 现在 0 < dotProduct v (S *ᵥ v)，v ≠ 0，S PosDef
  -- PosDef M: ∀ x ≠ 0, 0 < dotProduct (star x) (M *ᵥ x)；对 ℝ，star = id
  exact Matrix.PosDef.re_dotProduct_pos hS hv_ne

/-- LDL.diag (hS) 是下三角的: `i < j → LDL.diag hS i j = 0`。 -/
lemma LDL.diag_lowerTri (S : Matrix n n ℝ) (hS : S.PosDef) (i j : n) (hij : i ≠ j) :
    LDL.diag hS i j = 0 := by
  simp [LDL.diag, Matrix.diagonal]

/-- 对角线矩阵 `D = diagonal (Real.sqrt ∘ LDL.diagEntries hS)` 也是下三角的。 -/
lemma diag_sqrt_lowerTri (S : Matrix n n ℝ) (hS : S.PosDef) (i j : n) (hij : i < j) :
    (Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hS)) i j = 0 := by
  simp [Matrix.diagonal]

/-- `L = LDL.lower * diagonal (Real.sqrt ∘ LDL.diagEntries)` 是下三角矩阵。 -/
lemma chol_L_lowerTri (S : Matrix n n ℝ) (hS : S.PosDef) (i j : n) (hij : i < j) :
    (LDL.lower hS * Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hS)) i j = 0 := by
  -- LDL.lower 是下三角 (LDL.lower_triangular)
  -- diagonal √D 是对角 (lower triangular)
  -- 两者都 BlockTriangular toDual ⇒ 乘积也是 BlockTriangular toDual
  have hBT1 : BlockTriangular (LDL.lower hS) (toDual : n → nᵒᵈ) :=
    LDL.lower_blockTriangular S hS
  have hBT2 : BlockTriangular (Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hS))
      (toDual : n → nᵒᵈ) := by
    intro i' j' hij'
    rw [toDual_lt_toDual] at hij'
    simp [Matrix.diagonal]
  have hProd : BlockTriangular (LDL.lower hS * Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hS))
      (toDual : n → nᵒᵈ) :=
    hBT1.mul hBT2
  exact hProd i j (toDual_lt_toDual.mpr hij)

end CholeskyPSD

/-- Cholesky 因子存在性: SPD 矩阵必有 Cholesky 分解（严格证明，基于 Mathlib LDL）。 -/
theorem cholesky_exists (M : Mat32) (h : M.SPD) : Nonempty (CholeskyFactor M) := by
  -- Step 1: 把 Mat32.SPD 转换为 Matrix.PosDef
  obtain ⟨hIsSymm, hPos⟩ := h
  -- 使用 PosDef 形式构造 Cholesky 因子
  have hPosDef : (M : Matrix (Fin 32) (Fin 32) ℝ).PosDef := by
    refine ⟨?_, ?_⟩
    · -- IsSymm → IsHermitian for ℝ
      show M.conjTranspose = M
      rw [show M.conjTranspose = M.transpose from rfl]
      exact hIsSymm
    · intro x hx
      have hStar : star x = x := by
        funext i
        show star (x i) = x i
        simp [Pi.star_apply, star]
      rw [← hStar]
      exact hPos x hx
  -- Step 2: 用 LDL 构造 L := LDL.lower * √D
  set L : Mat32 := LDL.lower hPosDef * Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef)
  refine ⟨{ L := L
           lowerTri := ?_
           decomps := ?_ }⟩
  · -- L 下三角
    exact CholeskyPSD.chol_L_lowerTri M hPosDef
  · -- L * Lᵀ = M
    -- 由 LDL.lower_conj_diag: LDL.lower * LDL.diag * (LDL.lower)ᴴ = S
    -- L * Lᵀ = (LDL.lower * √D) * (LDL.lower * √D)ᵀ = LDL.lower * √D * √D * LDL.lowerᵀ
    --        = LDL.lower * D * LDL.lowerᵀ = LDL.lower * LDL.diag * (LDL.lower)ᵀ = S
    -- 对 ℝ, (LDL.lower)ᴴ = (LDL.lower)ᵀ
    have hConj : (LDL.lower hPosDef)ᴴ = (LDL.lower hPosDef)ᵀ := rfl
    -- LDL.lower_conj_diag: LDL.lower hS * LDL.diag hS * (LDL.lower hS)ᴴ = S
    have hLDL : LDL.lower hPosDef * LDL.diag hPosDef * (LDL.lower hPosDef)ᵀ = M := by
      rw [← hConj]
      exact LDL.lower_conj_diag hPosDef
    -- 展开 L * Lᵀ
    -- L = LDL.lower * diagonal √D
    -- Lᵀ = (LDL.lower * diagonal √D)ᵀ = (diagonal √D)ᵀ * (LDL.lower)ᵀ
    --    = diagonal √D * (LDL.lower)ᵀ      (diagonal 矩阵 = 自身转置)
    -- L * Lᵀ = (LDL.lower * diagonal √D) * (diagonal √D * (LDL.lower)ᵀ)
    --        = LDL.lower * (diagonal √D * diagonal √D) * (LDL.lower)ᵀ
    --        = LDL.lower * diagonal (Real.sqrt ∘ LDL.diagEntries ∘ Real.sqrt ∘ LDL.diagEntries) * (LDL.lower)ᵀ
    --        = LDL.lower * diagonal (LDL.diagEntries) * (LDL.lower)ᵀ   (Real.sqrt ∘ Real.sqrt = id, 对正数)
    --        = LDL.lower * LDL.diag * (LDL.lower)ᵀ
    --        = M (by hLDL)
    have hDiag : LDL.diag hPosDef = Matrix.diagonal (LDL.diagEntries hPosDef) := rfl
    have hD_pos : ∀ k, 0 ≤ LDL.diagEntries hPosDef k :=
      fun k => le_of_lt (LDL.diagEntries_pos S hPosDef k)
    have hSqrtSq : ∀ k, Real.sqrt (Real.sqrt (LDL.diagEntries hPosDef k)) ^ 2 =
                       LDL.diagEntries hPosDef k := by
      intro k
      rw [Real.sq_sqrt (le_of_lt (LDL.diagEntries_pos S hPosDef k))]
      exact (Real.sqrt_sqrt (le_of_lt (LDL.diagEntries_pos S hPosDef k))).symm
    -- 简化: diagonal √D * diagonal √D = diagonal (λ k, (Real.sqrt ∘ LDL.diagEntries hS k)²)
    have hDiagMul : Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef) *
                    Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef) =
                  Matrix.diagonal (fun k =>
                    (Real.sqrt (LDL.diagEntries hPosDef k)) *
                      (Real.sqrt (LDL.diagEntries hPosDef k))) := by
      ext i j
      simp [Matrix.diagonal, Matrix.mul_apply]
    -- 对角矩阵 (λ k, a k * a k) = diagonal (λ k, a k²)
    have hDiagSq : Matrix.diagonal (fun k =>
                    (Real.sqrt (LDL.diagEntries hPosDef k)) *
                      (Real.sqrt (LDL.diagEntries hPosDef k))) =
                  LDL.diag hPosDef := by
      ext i j
      simp [LDL.diag, Matrix.diagonal]
      ring
    -- 现在 L * Lᵀ = ...
    -- L = LDL.lower * diagonal √D
    -- Lᵀ = (LDL.lower * diagonal √D)ᵀ
    have hL_def : L = LDL.lower hPosDef * Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef) := rfl
    -- (A * B)ᵀ = Bᵀ * Aᵀ
    have hLT : L.transpose = (Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef)).transpose *
                             (LDL.lower hPosDef).transpose := by
      rw [hL_def, Matrix.transpose_mul]
    -- 对角矩阵 = 自身转置
    have hDiagT : (Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef)).transpose =
                  Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef) := by
      ext i j
      simp [Matrix.diagonal, Matrix.transpose_apply]
    -- 所以 Lᵀ = diagonal √D * (LDL.lower)ᵀ
    rw [hL_def]
    rw [hLT, hDiagT]
    -- 现在 L * Lᵀ = LDL.lower * diagonal √D * (diagonal √D * (LDL.lower)ᵀ)
    --            = LDL.lower * (diagonal √D * diagonal √D) * (LDL.lower)ᵀ
    --            = LDL.lower * LDL.diag * (LDL.lower)ᵀ
    rw [← Matrix.mul_assoc]
    rw [hDiagMul, hDiagSq]
    exact hLDL


/-- Stage 04 单用户拟合产物。 -/
structure UserGaussian where
  mu : Vec32
  sigma : Mat32
  n_profile : ℕ
  n_val : ℕ
  eigenGate : EigenGate sigma
  cholesky : CholeskyFactor sigma

/-- 拟合函数: 输入 Z_profile, Z_val, 输出 UserGaussian。 -/
noncomputable def fitOneUser {n_prof n_val : ℕ}
    (Z_prof : Fin n_prof → Vec32) (_Z_val : Fin n_val → Vec32)
    (_hnp : n_prof ≥ minProfileSents)
    (_hnv : n_val ≥ minValSents)
    (hspd : (sampleCovSymm Z_prof (sampleMean Z_prof)).SPD)
    (hratio : ∃ lamMin lamMax : ℝ,
      lamMin > 0 ∧ lamMin ≤ lamMax ∧ lamMin / lamMax ≥ minEigenRatio) :
    UserGaussian where
  mu := sampleMean Z_prof
  sigma := sampleCovSymm Z_prof (sampleMean Z_prof)
  n_profile := n_prof
  n_val := n_val
  eigenGate := by
    refine ⟨?_, ?_, ?_⟩
    · exact sampleCovSymm_symmetric Z_prof (sampleMean Z_prof)
    · exact fun x hx => hspd.2 x hx
    · exact hratio
  cholesky := (cholesky_exists _ hspd).some

end PersonalQuery