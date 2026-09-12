/-
# PersonalQuery.Basic — 向量 / 矩阵基础
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct
import Mathlib.LinearAlgebra.Matrix.Symmetric
import Mathlib.Analysis.InnerProductSpace.PiL2

namespace PersonalQuery

/-- 32 维实向量。 -/
abbrev Vec32 : Type := Fin 32 → ℝ

/-- 32 × 32 实矩阵。 -/
abbrev Mat32 : Type := Matrix (Fin 32) (Fin 32) ℝ

/-- n × 32 实矩阵。 -/
abbrev SentMatrix (n : ℕ) : Type := Fin n → Vec32

/-- 标准内积。 -/
def vec32Inner (u v : Vec32) : ℝ := ∑ i : Fin 32, u i * v i

/-- 平方范数。 -/
def vec32NormSq (u : Vec32) : ℝ := vec32Inner u u

notation "⟪" x ", " y "⟫" => vec32Inner x y

/-! ### 基础性质 -/

lemma vec32Inner_comm (u v : Vec32) : ⟪u, v⟫ = ⟪v, u⟫ := by
  unfold vec32Inner
  exact Finset.sum_congr rfl (fun x _ => mul_comm _ _)

lemma vec32Inner_add_left (u v w : Vec32) :
    ⟪u + v, w⟫ = ⟪u, w⟫ + ⟪v, w⟫ := by
  unfold vec32Inner
  simp [Pi.add_apply, Finset.sum_add_distrib, add_mul]

lemma vec32Inner_add_right (u v w : Vec32) :
    ⟪u, v + w⟫ = ⟪u, v⟫ + ⟪u, w⟫ := by
  unfold vec32Inner
  simp [Pi.add_apply, Finset.sum_add_distrib, mul_add]

lemma vec32Inner_smul_left (a : ℝ) (u v : Vec32) :
    ⟪a • u, v⟫ = a * ⟪u, v⟫ := by
  unfold vec32Inner
  simp only [Pi.smul_apply, smul_eq_mul, mul_assoc]
  rw [Finset.sum_congr rfl fun x _ => mul_comm _ _]
  rw [← Finset.sum_mul, mul_comm]

/-- 标量在右参数 (由对称性得来)。 -/
lemma vec32Inner_smul_right (a : ℝ) (u v : Vec32) :
    ⟪u, a • v⟫ = a * ⟪u, v⟫ := by
  rw [vec32Inner_comm, vec32Inner_smul_left, vec32Inner_comm]

/-- 左参数减去：⟪u - v, w⟫ = ⟪u, w⟫ - ⟪v, w⟫。 -/
lemma vec32Inner_sub_left (u v w : Vec32) :
    vec32Inner (u - v) w = vec32Inner u w - vec32Inner v w := by
  show ∑ i, (u i - v i) * w i = (∑ i, u i * w i) - ∑ i, v i * w i
  conv_lhs => enter [2, i]; rw [sub_mul]
  conv_lhs => enter [2, i]; rw [sub_eq_add_neg]
  rw [Finset.sum_add_distrib]
  rw [Finset.sum_neg_distrib]
  rw [sub_eq_add_neg]

/-- 右参数减去：⟪u, v - w⟫ = ⟪u, v⟫ - ⟪u, w⟫。 -/
lemma vec32Inner_sub_right (u v w : Vec32) :
    vec32Inner u (v - w) = vec32Inner u v - vec32Inner u w := by
  show ∑ i, u i * (v i - w i) = (∑ i, u i * v i) - ∑ i, u i * w i
  conv_lhs => enter [2, i]; rw [mul_sub]
  conv_lhs => enter [2, i]; rw [sub_eq_add_neg]
  rw [Finset.sum_add_distrib]
  rw [Finset.sum_neg_distrib]
  rw [sub_eq_add_neg]

/-- Matrix.smul 的 mulVec 线性: (c • A) *ᵥ v = c • (A *ᵥ v)。 -/
lemma Matrix.mulVec_smul (c : ℝ) (A : Mat32) (v : Vec32) :
    Matrix.mulVec (c • A) v = c • Matrix.mulVec A v := by
  funext i
  show (Matrix.mulVec (c • A) v) i = c * (Matrix.mulVec A v) i
  simp [Matrix.mulVec, dotProduct]
  -- 目标: ∑ x, c * A i x * v x = c * ∑ x, A i x * v x
  rw [show c * ∑ x, A i x * v x = (∑ x, A i x * v x) * c from mul_comm _ _]
  rw [Finset.sum_mul]
  rw [Finset.sum_congr rfl (fun x _ => mul_comm _ _)]
  apply Finset.sum_congr rfl
  intro x _
  ring

/-- 平方范数非负。 -/
lemma vec32NormSq_nonneg (u : Vec32) : 0 ≤ vec32NormSq u := by
  unfold vec32NormSq vec32Inner
  exact Finset.sum_nonneg fun _ _ => mul_self_nonneg _

/-- 零向量的平方范数为 0。 -/
lemma vec32NormSq_zero : vec32NormSq 0 = 0 := by
  unfold vec32NormSq vec32Inner
  simp

/-- 平方范数在向量加法下相容。 -/
lemma vec32NormSq_add (u v : Vec32) :
    vec32NormSq (u + v) = vec32NormSq u + vec32NormSq v + 2 * ⟪u, v⟫ := by
  unfold vec32NormSq vec32Inner
  simp [Pi.add_apply, Finset.sum_add_distrib, add_mul, mul_add, two_mul]
  have hswap : ∑ x, v x * u x = ∑ x, u x * v x :=
    Finset.sum_congr rfl (fun x _ => mul_comm _ _)
  rw [hswap]
  ring

/-! ### 矩阵转置 / 对称化 -/

/-- 矩阵对称化: (M + Mᵀ)/2。 -/
noncomputable def symmetrize (M : Mat32) : Mat32 :=
  fun i j => (M i j + M j i) / 2

/-- 对称化算子保持对称。 -/
lemma symmetrize_symmetric (M : Mat32) :
    symmetrize M = (symmetrize M).transpose := by
  funext i j
  simp only [symmetrize, Matrix.transpose_apply]
  ring

/-- 对称矩阵经对称化保持不变。 -/
lemma symmetrize_of_symmetric {M : Mat32} (h : M.transpose = M) :
    symmetrize M = M := by
  funext i j
  simp only [symmetrize]
  have h' : M j i = M.transpose i j := rfl
  rw [h', h]
  linarith

/-- 半正定类型。 -/
def Mat32.PSD (sigma : Mat32) : Prop :=
  ∀ x : Vec32, 0 ≤ vec32Inner x (Matrix.mulVec sigma x)

/-! ### Pipeline 公共对象 ID -/

abbrev UserId : Type := String
abbrev AsinId : Type := String
abbrev TaskId : Type := AsinId × UserId

end PersonalQuery