/-
# PersonalQuery.EncoderSpectralBridge — Encoder 不塌缩假设 + 谱桥

## 设计动机

Stage 03 / 04 流水线使用 `_SupEncoder` (Linear V→256 → ReLU → Linear 256→32)
将 raw rule 频次 bag (Vec21737) 映射到 32 维 latent. Lean 端要严格证明
"raw 谱信息被 encoder 保留到 latent", 必须把 "encoder 不塌缩" 这个直觉
形式化为最小数学条件, 不对 nonlinear encoder 做不现实的全局假设.

## 数学结构

设 raw 谱信息已通过 PCA 投到 32 维子空间 (见 `PCAFitting.lean`).
令 `Z : Fin n → Vec32` 为 parametrized raw 输入,
`f : Vec32 → Vec32` 为 effective encoder (PCA 投影 + 主 encoder 复合).

### Step 1: encoderNonCollapse

`EncoderNonCollapse f m support`:
  - `m > 0`
  - 对 support 上任意两点 x, y: `m² ‖x - y‖² ≤ ‖f(x) - f(y)‖²`

### Step 2: rawCov / latentCov

- `rawCov Z μ = sampleCovRaw Z μ`
- `latentCov f Z μ = sampleCovRaw (f ∘ Z) μ`

本模块**就地定义** `sampleMean` + `sampleCovRaw` + 对称 + PSD, 不依赖
`Gaussian.lean` (该文件在本仓库 untracked 且含 pre-existing 编译错误).

### Step 3: spectral bridge (pairwise variance identity)

**Pairwise trace identity**: `Σ_j ‖Y_j - μ_Y‖² = (1/(2n)) · Σ_{j,k} ‖Y_j - Y_k‖²`

**Trace bridge**: 任意 f 满足下 Lipschitz ⇒ `tr(latentCov) ≥ m² · tr(rawCov)`.

### Step 4: linear encoder eigenvalue bridge

对**对称线性** encoder `f(x) = A x`, `A = Aᵀ`, `m² ‖x‖² ≤ ‖A x‖²`,
若 `rawCov Z μ` 满足 `spectralRichnessCond γ`, 则
`latentCov f Z (A μ)` 满足 `spectralRichnessCond (m²γ)`.

### Step 5: 全链 raw_rules_and_encoder_imply_valid_fitting

依赖链:
  spectralRichnessCond γ' > 0 → Mat32.SPD (spectral_implies_spd)
    → rank = 32 (spectral_implies_full_rank)
      → CholeskyFactor 存在 (cholesky_exists_of_spd)
        → Mahalanobis D² via Cholesky (mahaD2Chol_nonneg)
          → valid_fitting

### Step 6: 误差界 (Phase M 目标)

- `inverse_covariance_error_bound`: `‖Σ̂⁻¹ - Σ⁻¹‖_op ≤ ε/(γ(γ-ε))`
- `mahalanobis_distance_error_bound`: `|D̂² - D²| ≤ ‖v‖² · ε/(γ(γ-ε))`

经典逆矩阵扰动定理 (Higham 2002). 此两步是仅剩的 axiom,
Phase M step 2 将用 resolvent identity + operator-norm submultiplicativity
彻底消掉.
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Diagonal
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.LinearAlgebra.Matrix.Symmetric
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.LDL
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.Algebra.Order.Rearrangement

import PersonalQuery.Basic

namespace PersonalQuery

open Matrix

/-! ## 局部定义: sampleMean / sampleCovRaw (独立于 Gaussian.lean) -/

/-- 样本均值: μ̂ = (1/n) Σ_j Y_j. -/
noncomputable def sampleMean {n : ℕ} (Y : Fin n → Vec32) : Vec32 :=
  fun i => (∑ j : Fin n, Y j i) / (n : ℝ)

/-- 样本协方差 (1/(n-1) 归一化, unbiased estimator). -/
noncomputable def sampleCovRaw {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32) : Mat32 :=
  fun i k =>
    (∑ j : Fin n, (Z j i - μ i) * (Z j k - μ k)) / ((n : ℝ) - 1)

/-- `sampleCovRaw` 对称. -/
theorem sampleCovRaw_symmetric {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32) :
    (sampleCovRaw Z μ).transpose = sampleCovRaw Z μ := by
  funext i k
  show (∑ j, (Z j k - μ k) * (Z j i - μ i)) / ((n : ℝ) - 1) =
       (∑ j, (Z j i - μ i) * (Z j k - μ k)) / ((n : ℝ) - 1)
  congr 1
  exact Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- `sampleCovRaw` 半正定 (`n ≥ 2`):
    关键代数: `xᵀ Σ x · (n-1) = Σ_j (⟨x, Z_j - μ⟩)² ≥ 0`,
    因此 `xᵀ Σ x ≥ 0`. -/
theorem sampleCovRaw_psd {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32) (x : Vec32)
    (hn : n ≥ 2) :
    0 ≤ vec32Inner x (Matrix.mulVec (sampleCovRaw Z μ) x) := by
  have hLHS : vec32Inner x (Matrix.mulVec (sampleCovRaw Z μ) x) =
              (1 / ((n : ℝ) - 1)) *
                  ∑ j : Fin n, (vec32Inner x (fun k => Z j k - μ k))^2 := by
    unfold vec32Inner sampleCovRaw Matrix.mulVec
    simp only [Finset.sum_div, Finset.sum_mul]
    rw [Finset.sum_mul, Finset.sum_mul]
    rw [← Finset.sum_mul]
    rw [Finset.sum_comm (s := (Finset.univ : Finset (Fin 32)))
                        (t := (Finset.univ : Finset (Fin n)))]
    rw [← Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro j _
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro i _
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro k _
    ring
  rw [hLHS]
  have h1 : (0 : ℝ) ≤ (1 / ((n : ℝ) - 1)) := by
    apply div_nonneg (by norm_num)
    linarith [show (2 : ℝ) ≤ (n : ℝ) by exact_mod_cast hn]
  apply mul_nonneg h1
  apply Finset.sum_nonneg
  intro j _
  exact sq_nonneg _

/-- Mat32 SPD 谓词. -/
def Mat32.SPD (M : Mat32) : Prop :=
  Matrix.IsSymm M ∧ ∀ x : Vec32, x ≠ 0 → 0 < vec32Inner x (Matrix.mulVec M x)

/-- spectralRichnessCond: `λ_min(M) ≥ γ`. -/
def spectralRichnessCond (C : Mat32) (γ : ℝ) : Prop :=
  Matrix.IsSymm C ∧ 0 < γ ∧
    ∀ x : Vec32, γ * vec32NormSq x ≤ vec32Inner x (Matrix.mulVec C x)

/-- spectral richness 蕴含 SPD. -/
theorem spectral_implies_spd (C : Mat32) (γ : ℝ)
    (hspec : spectralRichnessCond C γ) : Mat32.SPD C := by
  refine ⟨hspec.1, ?_⟩
  intro x hx
  have hnorm : 0 < vec32NormSq x := by
    unfold vec32NormSq vec32Inner
    have hex : ∃ i : Fin 32, x i ≠ 0 := by
      by_contra h; apply hx; funext i; by_contra hi; exact h ⟨i, hi⟩
    obtain ⟨i, hi⟩ := hex
    apply Finset.sum_pos' (fun j _ => mul_self_nonneg (x j))
    exact ⟨i, Finset.mem_univ _, mul_self_pos.mpr hi⟩
  have hγ_pos : 0 < γ := hspec.2.1
  have hprod : 0 < γ * vec32NormSq x := mul_pos hγ_pos hnorm
  have hineq : γ * vec32NormSq x ≤ vec32Inner x (Matrix.mulVec C x) := hspec.2.2 x
  exact lt_of_lt_of_le hprod hineq

/-- **spectral_implies_spd_inv** (Phase S 桥接 lemma):

    若 S 满足 `spectralRichnessCond S γ` 且存在逆矩阵 `Sᵀ · S⁻¹ = 1`,
    则 `S⁻¹` 也是 SPD.

    Proof 思路 (4 步, 无 axiom):
      1. `spectral_implies_spd` 推出 `Mat32.SPD S` (= `S.IsSymm` ∧ `S` PosDef).
      2. `mat32_spd_iff_posDef` 桥接 `Mat32.SPD S ↔ (S : Matrix ...).PosDef`.
      3. `Matrix.PosDef.inv` 推出 `(S⁻¹ : Matrix ...).PosDef`.
      4. `mat32_spd_iff_posDef` 反向桥接得 `Mat32.SPD S⁻¹`.

    Phase S 动机: 让 `mahalanobis_mean_error_bound` 用 SPD 接口调用
    `mahalanobis_cauchy_schwarz_strict`, 而 caller 通过本 lemma
    从 `spectralRichnessCond` 自动给 SPD 假设。 -/
theorem spectral_implies_spd_inv (S : Mat32) (γ : ℝ)
    (hspec : spectralRichnessCond S γ)
    (hinv : ∃ Sinv : Mat32, S.transpose * Sinv = 1 ∧ Sinv * S.transpose = 1) :
    Mat32.SPD hinv.choose := by
  -- Step 1: S 是 SPD
  have hS_spd : Mat32.SPD S := spectral_implies_spd S γ hspec
  -- Step 2: Mat32.SPD S → PosDef S (作为 Matrix (Fin 32) (Fin 32) ℝ)
  have hS_symm : Matrix.IsSymm S := hspec.1
  -- 取出逆
  obtain ⟨Sinv, hinv_l, hinv_r⟩ := hinv
  -- 由对称性 Sᵀ = S, 故 S · S⁻¹ = 1, S⁻¹ · S = 1
  have hSinv_r : S * Sinv = 1 := by
    rw [hS_symm] at hinv_l; exact hinv_l
  have hSinv_l : Sinv * S = 1 := by
    rw [hS_symm] at hinv_r; exact hinv_r
  -- PosDef 桥接
  have hPosDef_S : (S : Matrix (Fin 32) (Fin 32) ℝ).PosDef := by
    refine ⟨?_, ?_⟩
    · show S.conjTranspose = S
      rw [show S.conjTranspose = S.transpose from rfl]
      exact hS_spd.1
    · intro x hx
      have hStar : star x = x := by
        funext i; show star (x i) = x i; simp [Pi.star_apply, star]
      rw [← hStar]
      exact hS_spd.2 x hx
  -- Step 3: PosDef.inv 推出 Sinv PosDef
  have hPosDef_Sinv : (Sinv : Matrix (Fin 32) (Fin 32) ℝ).PosDef := by
    -- PosDef.inv 需要 DecidableEq n, 通过 inferInstance 自动获得
    refine ⟨?_, ?_⟩
    · show Sinv.conjTranspose = Sinv
      -- 由 inv_symm 推出 Sinvᵀ = Sinv
      have hSinv_symm : Sinv.transpose = Sinv :=
        inv_symm S Sinv hS_symm hSinv_r hSinv_l
      rw [show Sinv.conjTranspose = Sinv.transpose from rfl]
      exact hSinv_symm
    · intro x hx
      have hStar : star x = x := by
        funext i; show star (x i) = x i; simp [Pi.star_apply, star]
      rw [← hStar]
      -- PosDef 包含 ⟨x, M x⟩ > 0 for x ≠ 0, 由 PosDef.inv 推出
      exact (Matrix.PosDef.inv hPosDef_S).2 x hx
  -- Step 4: PosDef Sinv → Mat32.SPD Sinv
  refine ⟨?_, ?_⟩
  · -- 对称: Sinv.transpose = Sinv
    have hSinv_symm' : Sinv.transpose = Sinv :=
      inv_symm S Sinv hS_symm hSinv_r hSinv_l
    exact hSinv_symm'
  · -- x ≠ 0 → 0 < ⟨x, Sinv x⟩
    intro x hx
    have hStar : star x = x := by
      funext i; show star (x i) = x i; simp [Pi.star_apply, star]
    rw [← hStar]
    exact hPosDef_Sinv.2 x hx

/-! ## Step 1: encoder non-collapse -/

/-- 编码器下 Lipschitz 条件 (lower-Lipschitz). -/
structure EncoderNonCollapse (f : Vec32 → Vec32) (m : ℝ) (support : Vec32 → Prop) where
  m_pos : 0 < m
  lowerLipschitz : ∀ ⦃x y : Vec32⦄, support x → support y →
    m^2 * vec32NormSq (x - y) ≤ vec32NormSq (f x - f y)

/-! ## Step 2: raw / latent covariance -/

/-- Raw 协方差 (parametrized 32 维 raw 空间). -/
noncomputable def rawCov {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32) : Mat32 :=
  sampleCovRaw Z μ

/-- Raw 协方差对称. -/
theorem rawCov_symmetric {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32) :
    (rawCov Z μ).transpose = rawCov Z μ :=
  sampleCovRaw_symmetric Z μ

/-- Raw 协方差半正定 (`n ≥ 2`). -/
theorem rawCov_psd {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32) (x : Vec32) (hn : n ≥ 2) :
    0 ≤ vec32Inner x (Matrix.mulVec (rawCov Z μ) x) :=
  sampleCovRaw_psd Z μ x hn

/-- Latent 协方差. -/
noncomputable def latentCov {n : ℕ} (f : Vec32 → Vec32) (Z : Fin n → Vec32) (μ : Vec32) : Mat32 :=
  sampleCovRaw (fun j => f (Z j)) μ

/-- Latent 协方差对称. -/
theorem latentCov_symmetric {n : ℕ} (f : Vec32 → Vec32) (Z : Fin n → Vec32) (μ : Vec32) :
    (latentCov f Z μ).transpose = latentCov f Z μ :=
  sampleCovRaw_symmetric (fun j => f (Z j)) μ

/-- Latent 协方差半正定 (`n ≥ 2`). -/
theorem latentCov_psd {n : ℕ} (f : Vec32 → Vec32) (Z : Fin n → Vec32) (μ : Vec32) (x : Vec32)
    (hn : n ≥ 2) :
    0 ≤ vec32Inner x (Matrix.mulVec (latentCov f Z μ) x) :=
  sampleCovRaw_psd (fun j => f (Z j)) μ x hn

/-! ## Step 3a: pairwise variance identity (证明) -/

/-- 辅助: `(∑_j Y_j i) = n · μ_Y i` 当 `μ_Y i = (∑_j Y_j i) / n`. -/
private lemma sum_eq_n_mul_mu {n : ℕ} (Y : Fin n → Vec32) (μ_Y : Vec32)
    (hμ : ∀ i, μ_Y i = (∑ j, Y j i) / (n : ℝ)) (i : Fin 32) :
    (∑ j, Y j i) = (n : ℝ) * μ_Y i := by
  rw [hμ i]
  rw [div_mul_cancel₀ _ (Nat.cast_ne_zero.mpr (by omega : n ≠ 0))]

/-- 内部辅助: `(∑_j Y_j i)² = (n μ_Y i)²`. -/
private lemma sum_Y_sq_eq_n_sq_mu_sq {n : ℕ} (Y : Fin n → Vec32) (μ_Y : Vec32)
    (hμ : ∀ i, μ_Y i = (∑ j, Y j i) / (n : ℝ)) (i : Fin 32) :
    (∑ j, Y j i) ^ 2 = (n : ℝ) ^ 2 * μ_Y i ^ 2 := by
  rw [show (∑ j, Y j i) = (n : ℝ) * μ_Y i from sum_eq_n_mul_mu Y μ_Y hμ i]
  ring

/-- **Pairwise trace identity** (经典统计学事实). -/
theorem pairwise_variance_identity
    {n : ℕ} (Y : Fin n → Vec32) (μ_Y : Vec32)
    (hμ : ∀ i, μ_Y i = (∑ j, Y j i) / (n : ℝ))
    (hn : 0 < n) :
    ∑ j, vec32NormSq (fun i => Y j i - μ_Y i) =
      (1 / (2 * (n : ℝ))) * ∑ j, ∑ k,
        vec32NormSq (fun i => Y j i - Y k i) := by
  unfold vec32NormSq vec32Inner
  have hLHS : ∑ j, ∑ i, (Y j i - μ_Y i) ^ 2 =
              (∑ j, ∑ i, (Y j i) ^ 2) - (n : ℝ) * ∑ i, μ_Y i ^ 2 := by
    have hA : ∑ j, ∑ i, -(2 : ℝ) * Y j i * μ_Y i = -(2 : ℝ) * (n : ℝ) * ∑ i, μ_Y i ^ 2 := by
      have h1 : ∑ j, ∑ i, Y j i * μ_Y i = (n : ℝ) * ∑ i, μ_Y i ^ 2 := by
        have hinner : ∑ j, ∑ i, Y j i * μ_Y i = ∑ i, (∑ j, Y j i) * μ_Y i := by
          rw [Finset.sum_comm]
          apply Finset.sum_congr rfl
          intro i _
          rw [Finset.sum_mul]
        rw [hinner]
        apply Finset.sum_congr rfl
        intro i _
        rw [show (∑ j, Y j i) * μ_Y i = (n : ℝ) * μ_Y i * μ_Y i from by
              rw [sum_eq_n_mul_mu Y μ_Y hμ i]; ring]
        ring
      rw [h1]
      ring
    have hB : ∑ j, ∑ i, μ_Y i ^ 2 = (n : ℝ) * ∑ i, μ_Y i ^ 2 := by
      rw [←Finset.sum_mul, Finset.card_univ]
      simp
    simp_rw [sq]
    simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib]
    rw [hA, hB]
    ring
  rw [hLHS]
  have hRHS_pre : (1 / (2 * (n : ℝ))) * ∑ j, ∑ k, ∑ i, (Y j i - Y_k i) ^ 2 =
                  (∑ j, ∑ i, (Y j i) ^ 2) - (n : ℝ) * ∑ i, μ_Y i ^ 2 := by
    have hk_pair : ∑ j, ∑ k, ∑ i, (Y j i - Y_k i) ^ 2 =
                   2 * (n : ℝ) * (∑ j, ∑ i, (Y j i) ^ 2) -
                   2 * ∑ i, ((∑ j, Y j i) * (∑ k, Y_k i)) := by
      simp_rw [sq]
      simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.sum_neg_distrib]
      have h1 : ∑ j, ∑ k, ∑ i, (Y j i) ^ 2 = (n : ℝ) * ∑ j, ∑ i, (Y j i) ^ 2 := by
        rw [Finset.sum_comm (s := Finset.univ) (t := Finset.univ)]
        rw [←Finset.sum_mul, Finset.card_univ, Finset.sum_const]
        simp
      have h3 : ∑ j, ∑ k, ∑ i, (Y_k i) ^ 2 = (n : ℝ) * ∑ k, ∑ i, (Y_k i) ^ 2 := by
        rw [Finset.sum_comm (s := Finset.univ) (t := Finset.univ)]
        rw [←Finset.sum_mul, Finset.card_univ, Finset.sum_const]
        simp
      have h2 : ∑ j, ∑ k, ∑ i, (2 : ℝ) * Y j i * Y_k i =
                2 * ∑ i, ((∑ j, Y j i) * (∑ k, Y_k i)) := by
        rw [Finset.sum_comm (s := Finset.univ) (t := Finset.univ)]
        simp only [Finset.sum_mul]
        rw [Finset.sum_comm (s := Finset.univ) (t := Finset.univ)]
        simp only [Finset.sum_mul, Pi.add_apply]
        rw [←Finset.sum_mul]
        apply Finset.sum_congr rfl
        intro i _
        rw [Finset.sum_mul]
        ring
      rw [h1, h3, h2]
      ring
    rw [hk_pair]
    have hnorm_sq : ∑ i, ((∑ j, Y j i) * (∑ k, Y_k i)) = (n : ℝ) ^ 2 * ∑ i, μ_Y i ^ 2 := by
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro i _
      rw [sum_Y_sq_eq_n_sq_mu_sq Y μ_Y hμ i]
    rw [hnorm_sq]
    have hn_ne : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega : n ≠ 0)
    rw [show (1 / (2 * (n : ℝ))) * (2 * (n : ℝ) * (∑ j, ∑ i, (Y j i) ^ 2) -
                                    2 * (n : ℝ) ^ 2 * ∑ i, μ_Y i ^ 2) =
                (∑ j, ∑ i, (Y j i) ^ 2) - (n : ℝ) * ∑ i, μ_Y i ^ 2 from by
          rw [div_mul_eq_mul_div]
          field_simp [hn_ne]
          ring]
  rw [hRHS_pre]

/-! ## Step 3b: trace bridge (证明) -/

/-- **Trace bridge**: encoder 非塌缩 ⇒ latent trace ≥ m² · raw trace. -/
theorem encoder_noncollapse_implies_latent_trace
    {n : ℕ}
    (f : Vec32 → Vec32) (Z : Fin n → Vec32) (μ_Z μ_f : Vec32)
    (hμ_Z : ∀ i, μ_Z i = (∑ j, Z j i) / (n : ℝ))
    (hμ_f : ∀ i, μ_f i = (∑ j, f (Z j) i) / (n : ℝ))
    (enc : EncoderNonCollapse f m support)
    (hsupport : ∀ j, support (Z j))
    (hn : n ≥ 2) :
    ∑ i : Fin 32, (latentCov f Z μ_f) i i ≥
      m^2 * ∑ i : Fin 32, (rawCov Z μ_Z) i i := by
  have hn_pos : (0 : n) < n := by omega
  have hLHS : ∑ i : Fin 32, (latentCov f Z μ_f) i i =
              (1 / ((n : ℝ) - 1)) * ∑ j, vec32NormSq (fun k => f (Z j) k - μ_f k) := by
    unfold latentCov sampleCovRaw
    rw [show ∀ i, (sampleCovRaw (fun j => f (Z j)) μ_f) i i =
                  (∑ j, (f (Z j) i - μ_f i) ^ 2) / ((n : ℝ) - 1) from by
          intro i; show (∑ j, (f (Z j) i - μ_f i) * (f (Z j) i - μ_f i)) / ((n : ℝ) - 1) = _
          rw [←sq]; rfl]
    simp only [Finset.sum_div, Finset.sum_mul, Matrix.diag]
    rw [Finset.sum_comm]
    rw [←Finset.sum_mul]
  have hRHS : ∑ i : Fin 32, (rawCov Z μ_Z) i i =
              (1 / ((n : ℝ) - 1)) * ∑ j, vec32NormSq (fun k => Z j k - μ_Z k) := by
    unfold rawCov sampleCovRaw
    rw [show ∀ i, (sampleCovRaw Z μ_Z) i i =
                  (∑ j, (Z j i - μ_Z i) ^ 2) / ((n : ℝ) - 1) from by
          intro i; rw [←sq]; rfl]
    simp only [Finset.sum_div, Finset.sum_mul, Matrix.diag]
    rw [Finset.sum_comm]
    rw [←Finset.sum_mul]
  rw [hLHS, hRHS]
  have hpw_L : ∑ j, vec32NormSq (fun k => f (Z j) k - μ_f k) =
               (1 / (2 * (n : ℝ))) * ∑ j, ∑ k,
                 vec32NormSq (fun i => f (Z j) i - f (Z k) i) :=
    pairwise_variance_identity (fun j => f (Z j)) μ_f hμ_f hn_pos
  have hpw_R : ∑ j, vec32NormSq (fun k => Z j k - μ_Z k) =
               (1 / (2 * (n : ℝ))) * ∑ j, ∑ k,
                 vec32NormSq (fun i => Z j i - Z k i) :=
    pairwise_variance_identity Z μ_Z hμ_Z hn_pos
  rw [hpw_L, hpw_R]
  have hLP : ∀ j k, m^2 * vec32NormSq (Z j - Z k) ≤
                     vec32NormSq (f (Z j) - f (Z k)) := by
    intro j k
    exact enc.lowerLipschitz (Z j) (Z k) (hsupport j) (hsupport k)
  have hkey : ∑ j, ∑ k,
      (vec32NormSq (fun i => f (Z j) i - f (Z k) i) -
        m^2 * vec32NormSq (fun i => Z j i - Z k i)) ≥ 0 := by
    apply Finset.sum_nonneg
    intro j _
    apply Finset.sum_nonneg
    intro k _
    have h := hLP j k
    linarith
  have hc_pos : (0 : ℝ) ≤ (1 / ((n : ℝ) - 1)) * (1 / (2 * (n : ℝ))) := by positivity
  have hdiff : (1 / ((n : ℝ) - 1)) *
                 ((1 / (2 * (n : ℝ))) *
                   ∑ j, ∑ k,
                     vec32NormSq (fun i => f (Z j) i - f (Z k) i)) -
               m^2 * ((1 / ((n : ℝ) - 1)) *
                 ((1 / (2 * (n : ℝ))) *
                   ∑ j, ∑ k,
                     vec32NormSq (fun i => Z j i - Z k i))) =
               (1 / ((n : ℝ) - 1)) * (1 / (2 * (n : ℝ))) *
                 ∑ j, ∑ k,
                   (vec32NormSq (fun i => f (Z j) i - f (Z k) i) -
                     m^2 * vec32NormSq (fun i => Z j i - Z k i)) := by
    rw [←Finset.sum_sub_distrib]
    ring
  rw [hdiff]
  have hprod : (1 / ((n : ℝ) - 1)) * (1 / (2 * (n : ℝ))) *
               ∑ j, ∑ k,
                 (vec32NormSq (fun i => f (Z j) i - f (Z k) i) -
                   m^2 * vec32NormSq (fun i => Z j i - Z k i)) ≥ 0 := by
    exact mul_nonneg hc_pos hkey
  linarith [hdiff, hprod]

/-! ## Step 4: linear encoder eigenvalue bridge -/

/-- 线性 encoder 假设. -/
structure LinearEncoder (A : Mat32) (m : ℝ) : Prop where
  m_pos : 0 < m
  isSymm : Matrix.IsSymm A
  lowerSV : ∀ x : Vec32, m^2 * vec32NormSq x ≤ vec32NormSq (Matrix.mulVec A x)

/-- 对称 A 蕴含 Aᵀ 与 A 在 mulVec 上等价. -/
lemma symmetric_A_mulVec_eq {A : Mat32} (hA : Matrix.IsSymm A) (z : Vec32) :
    Matrix.mulVec A.transpose z = Matrix.mulVec A z := by
  funext i
  show (Matrix.mulVec A.transpose z) i = (Matrix.mulVec A z) i
  unfold Matrix.mulVec Matrix.transpose
  simp only [Finset.sum_mul, Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  rw [hA k i]
  ring

/-- vecMul z Aᵀ entry-wise = Aᵀ.mulVec z. -/
private lemma vecMul_At_eq_mulVec_At (A : Mat32) (z : Vec32) :
    Matrix.vecMul z A.transpose = Matrix.mulVec A.transpose z := by
  funext i
  show (Matrix.vecMul z Aᵀ) i = (Aᵀ.mulVec z) i
  unfold Matrix.vecMul Matrix.mulVec
  simp only [Matrix.transpose_apply]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- **Bilinear identity** (严格证明):
    `zᵀ (A · M · Aᵀ) z = (Aᵀ z)ᵀ M (Aᵀ z)`. -/
theorem quad_form_congr {A M : Mat32} (z : Vec32) :
    vec32Inner z (Matrix.mulVec (A * M * A.transpose) z) =
    vec32Inner (Matrix.mulVec A.transpose z)
               (Matrix.mulVec M (Matrix.mulVec A.transpose z)) := by
  -- Step 1: (A * M * Aᵀ).mulVec z = A.mulVec ((M * Aᵀ).mulVec z)
  rw [show A * M * Aᵀ = A * (M * Aᵀ) from (Matrix.mul_assoc A M Aᵀ).symm]
  rw [Matrix.mulVec_mulVec]
  -- Step 2: ((M * Aᵀ)).mulVec z = M.mulVec (Aᵀ.mulVec z)
  rw [Matrix.mulVec_mulVec]
  -- Step 3: 展开 dotProduct_mulVec 形式
  rw [Matrix.dotProduct_mulVec]
  -- Step 4: vecMul z A = Aᵀ.mulVec z (entry-wise)
  rw [vecMul_At_eq_mulVec_At A z]
  -- 现在 LHS = (Aᵀ.mulVec z).dotProduct (M.mulVec (Aᵀ.mulVec z))
  --       = vec32Inner (Aᵀ.mulVec z) (M.mulVec (Aᵀ.mulVec z))
  rfl

/-- latentCov (A·Z) (A·μ_Z) 的 entry-wise 等式: = A · rawCov Z μ_Z · Aᵀ. -/
lemma latentCov_eq_A_raw_cov_At
    {n : ℕ} (A : Mat32) (Z : Fin n → Vec32) (μ_Z : Vec32) :
    latentCov (fun x => Matrix.mulVec A x) Z
      (sampleMean (fun j => Matrix.mulVec A (Z j))) =
    A * (rawCov Z μ_Z) * A.transpose := by
  ext i k
  unfold latentCov rawCov sampleCovRaw
  simp only [Matrix.mul_apply, Matrix.transpose_apply]
  simp only [Finset.sum_div, Finset.sum_mul]
  rw [Finset.sum_comm (s := Finset.univ) (t := Finset.univ)]
  rw [Finset.sum_comm (s := Finset.univ) (t := Finset.univ)]
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro p _
  apply Finset.sum_congr rfl
  intro q _
  ring

/-- **Eigenvalue bridge** (linear encoder). -/
theorem linear_encoder_implies_latent_spectral
    {n : ℕ} (A : Mat32) (m γ : ℝ) (Z : Fin n → Vec32) (μ_Z : Vec32)
    (hA : LinearEncoder A m)
    (hμ_Z : μ_Z = sampleMean Z)
    (hpop : spectralRichnessCond (rawCov Z μ_Z) γ)
    (hn : n ≥ 2) :
    spectralRichnessCond
      (latentCov (fun x => Matrix.mulVec A x) Z
        (sampleMean (fun j => Matrix.mulVec A (Z j))))
      (m^2 * γ) := by
  refine ⟨?_, mul_pos (sq_pos_of_pos hA.m_pos) hpop.2.1, ?_⟩
  · -- 对称性
    intro i j
    rw [show (A * (rawCov Z μ_Z) * Aᵀ)ᵀ i j = (Aᵀᵀ * (rawCov Z μ_Z)ᵀ * Aᵀ) i j from by
          rw [Matrix.transpose_mul]]
    rw [Matrix.transpose_transpose]
    rw [hpop.1]
    show (Aᵀ * rawCov Z μ_Z * Aᵀ) i j = (A * rawCov Z μ_Z * Aᵀ) i j
    rw [Matrix.mul_apply, Matrix.mul_apply]
    apply Finset.sum_congr rfl
    intro p _
    rw [Matrix.mul_apply, Matrix.mul_apply]
    apply Finset.sum_congr rfl
    intro q _
    rw [hA.isSymm p q, hA.isSymm q p, hA.isSymm i p]
    ring
  · -- spectral richness
    intro z
    rw [latentCov_eq_A_raw_cov_At A Z μ_Z]
    rw [quad_form_congr A (rawCov Z μ_Z) z]
    have hpop_z : γ * vec32NormSq (Aᵀ.mulVec z) ≤
                  vec32Inner (Aᵀ.mulVec z)
                    (Matrix.mulVec (rawCov Z μ_Z) (Aᵀ.mulVec z)) :=
      hpop.2.2 (Aᵀ.mulVec z)
    have hAt_eq : vec32NormSq (Aᵀ.mulVec z) = vec32NormSq (A.mulVec z) := by
      rw [symmetric_A_mulVec_eq hA.isSymm z]
    have hA_lower : vec32NormSq (A.mulVec z) ≥ m^2 * vec32NormSq z := hA.lowerSV z
    have hγ_pos : (0 : ℝ) < γ := hpop.2.1
    nlinarith [hpop_z, hAt_eq, hA_lower, hγ_pos]

/-! ## Step 5a: spectral richness → rank = 32 (严格证明) -/

/-- spectral richness 蕴含 cov 矩阵无核 (injective). -/
theorem spectral_implies_injective (C : Mat32) (γ : ℝ)
    (hspec : spectralRichnessCond C γ) :
    ∀ x : Vec32, Matrix.mulVec C x = 0 → x = 0 := by
  intro x hzero
  by_contra hx
  -- 用 SPD 二次型: x ≠ 0 ⟹ 0 < xᵀ C x; 但 xᵀ C x = 0 ⟸ Cx = 0
  have hspd := spectral_implies_spd C γ hspec
  have hpos : 0 < vec32Inner x (Matrix.mulVec C x) := hspd.2 x hx
  rw [hzero] at hpos
  simp only [vec32Inner] at hpos
  linarith

/-- Mat32 上 spectral richness ⇒ rank = 32.

    严格证明路径:
      SPD ⇒ mulVec C injective ⇒ ker = ⊥
      ker = ⊥ ⇒ finrank_range = finrank_domain = 32
      finrank_range = 32 ⇒ Matrix.rank C = 32 (Mathlib 的 rank = finrank_toLin) -/
theorem spectral_implies_full_rank (C : Mat32) (γ : ℝ)
    (hspec : spectralRichnessCond C γ) : Matrix.rank C = 32 := by
  -- Step 1: 获得 injective
  have hinj : ∀ x, Matrix.mulVec C x = 0 → x = 0 :=
    spectral_implies_injective C γ hspec
  -- Step 2: injective 推出 ker = ⊥
  have hker : LinearMap.ker (Matrix.toLin' C) = ⊥ := by
    apply LinearMap.ker_eq_bot'.mpr
    intro v hv
    simpa [Matrix.toLin'_apply] using hinj v hv
  -- Step 3: rank-nullity: finrank_range + finrank_ker = 32
  have hrangefinrank :
      finrank ℝ (LinearMap.range (Matrix.toLin' C))
        + finrank ℝ (LinearMap.ker (Matrix.toLin' C))
      = finrank ℝ (Fin 32 → ℝ) :=
    LinearMap.finrank_range_add_finrank_ker (f := Matrix.toLin' C)
  have hdomain : finrank ℝ (Fin 32 → ℝ) = 32 :=
    Module.finrank_pi (R := ℝ) (ι := Fin 32)
  have hker_fin : finrank ℝ (LinearMap.ker (Matrix.toLin' C)) = 0 := by
    rw [hker, finrank_bot]
  have hrangefinrank_eq : finrank ℝ (LinearMap.range (Matrix.toLin' C)) = 32 := by
    linarith [hrangefinrank, hdomain, hker_fin]
  -- Step 4: finrank_range = 32 ⇒ Matrix.rank C = 32
  rw [Matrix.rank_eq_finrank_range_toLin C (Pi.basisFun ℝ (Fin 32)) (Pi.basisFun ℝ (Fin 32))]
  exact hrangefinrank_eq

/-! ## Step 5b: Cholesky 因子存在性 (基于 Mathlib LDL) -/

/-- 下三角谓词: `∀ i j, i < j → M i j = 0`. -/
def LowerTriangular (M : Mat32) : Prop :=
  ∀ i j : Fin 32, i < j → M i j = 0

/-- 下三角矩阵乘积仍下三角 (直接证明, 不依赖 BlockTriangular). -/
lemma lower_triangular_mul (A B : Mat32) (hA : LowerTriangular A) (hB : LowerTriangular B) :
    LowerTriangular (A * B) := by
  intro i j hij
  show (A * B) i j = 0
  unfold Matrix.mul Matrix.mul_apply
  apply Finset.sum_eq_zero
  intro k _
  show A i k * B k j = 0
  by_cases hki : i < k
  · rw [hA i k hki]
    ring
  · push_neg at hki
    have hkj : k < j := lt_of_lt_of_le hij hki
    rw [hB k j hkj]
    ring

/-- Cholesky 因子: 下三角 L 满足 L * Lᵀ = Σ. -/
structure CholeskyFactor (Σ : Mat32) where
  L : Mat32
  lowerTri : LowerTriangular L
  decomps : L * L.transpose = Σ

/-- 通过 Cholesky 计算 Mahalanobis 距离: `D²_Σ(v, μ) = ‖Lᵀ (v - μ)‖²`. -/
def mahaD2Chol (C : CholeskyFactor Σ) (v μ : Vec32) : ℝ :=
  vec32NormSq (Matrix.mulVec C.L.transpose (v - μ))

/-- Cholesky 路径的 Mahalanobis D² ≥ 0. -/
theorem mahaD2Chol_nonneg (C : CholeskyFactor Σ) (v μ : Vec32) :
    0 ≤ mahaD2Chol C v μ :=
  vec32NormSq_nonneg _

/-- Mat32.SPD 等价于 Matrix.PosDef (实数情形, `conjTranspose = transpose`). -/
lemma mat32_spd_iff_posDef (M : Mat32) :
    Mat32.SPD M ↔ (M : Matrix (Fin 32) (Fin 32) ℝ).PosDef := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · refine ⟨?_, ?_⟩
    · show M.conjTranspose = M
      rw [show M.conjTranspose = M.transpose from rfl]
      exact h.1
    · intro x hx
      have hStar : star x = x := by
        funext i
        show star (x i) = x i
        simp [Pi.star_apply, star]
      rw [← hStar]
      exact h.2 x hx
  · refine ⟨?_, ?_⟩
    · have hEq : M.conjTranspose = M.transpose := rfl
      rw [hEq] at h
      exact h.1
    · intro x hx
      have hStar : star x = x := by
        funext i
        show star (x i) = x i
        simp [Pi.star_apply, star]
      rw [← hStar]
      exact h.2 x hx

namespace CholeskyPSDLocal

open Matrix LDL

/-- LDL.diagEntries 严格正 (PosDef ⟹ 每个对角线 entry > 0). -/
lemma LDL.diagEntries_pos (S : Mat32) (hS : (S : Mat32).PosDef) (i : Fin 32) :
    0 < LDL.diagEntries hS i := by
  rw [LDL.diagEntries_eq_dotProduct S hS i]
  haveI : Invertible (LDL.lowerInv hS) := LDL.invertibleLowerInv hS
  have hv_ne : (LDL.lowerInv hS i) ≠ 0 := by
    intro hv_eq
    have h_row_zero : ∀ k, (LDL.lowerInv hS) i k = 0 := by
      intro k
      have hk : (LDL.lowerInv hS i) k = 0 := by rw [hv_eq]; rfl
      exact hk
    have hdet_zero : (LDL.lowerInv hS).det = 0 :=
      det_eq_zero_of_row_eq_zero i h_row_zero
    exact (isUnit_iff_isUnit_det _).mp (Invertible.isUnit (LDL.lowerInv hS)) |>.ne_zero hdet_zero
  exact Matrix.PosDef.re_dotProduct_pos hS hv_ne

/-- LDL.diag 是对角的: `i ≠ j → LDL.diag hS i j = 0`. -/
lemma LDL.diag_diagonal (S : Mat32) (hS : (S : Mat32).PosDef) (i j : Fin 32)
    (hij : i ≠ j) :
    LDL.diag hS i j = 0 := by
  simp [LDL.diag, Matrix.diagonal]

/-- 对角线矩阵 `D = diagonal (Real.sqrt ∘ LDL.diagEntries hS)` 是对角的. -/
lemma diag_sqrt_diagonal (S : Mat32) (hS : (S : Mat32).PosDef) (i j : Fin 32)
    (hij : i ≠ j) :
    (Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hS)) i j = 0 := by
  simp [Matrix.diagonal]

/-- `L = LDL.lower * diagonal (Real.sqrt ∘ LDL.diagEntries)` 是下三角矩阵.

    证明: LDL.lower 下三角, diagonal √D 对角(亦下三角); 乘积下三角. -/
lemma chol_L_lowerTri (S : Mat32) (hS : (S : Mat32).PosDef) (i j : Fin 32)
    (hij : i < j) :
    LowerTriangular
      (LDL.lower hS * Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hS)) := by
  intro i' j' hij'
  -- LDL.lower 下三角: LDL.lower_triangular
  -- diagonal √D 对角: i' < j' 时为 0
  have hBT1 : LowerTriangular (LDL.lower hS : Mat32) := by
    intro a b hab
    rw [LDL.lower_eq_lowerInv hS, LDL.lowerInv_triangular hS _ _ hab]
  have hBT2 : LowerTriangular (Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hS) : Mat32) := by
    intro a b hab
    simp [Matrix.diagonal]
  exact lower_triangular_mul _ _ hBT1 hBT2 i' j' hij'

end CholeskyPSDLocal

/-- **Cholesky 因子存在性** (严格证明): SPD 矩阵必有 Cholesky 分解.

    构造: `L := LDL.lower hS * √D`, 其中 `D = LDL.diag hS` 为正对角矩阵.
    验证:
      1. L 下三角 (chol_L_lowerTri)
      2. L * Lᵀ = (LDL.lower * √D) * (√D)ᵀ * (LDL.lower)ᵀ
                = LDL.lower * D * (LDL.lower)ᵀ
                = hS (LDL.lower_conj_diag) -/
theorem cholesky_exists_of_spd (Σ : Mat32) (hspd : Mat32.SPD Σ) :
    Nonempty (CholeskyFactor Σ) := by
  -- Step 1: Mat32.SPD → PosDef
  obtain ⟨hIsSymm, hPos⟩ := hspd
  have hPosDef : (Σ : Mat32).PosDef := by
    refine ⟨?_, ?_⟩
    · show Σ.conjTranspose = Σ
      rw [show Σ.conjTranspose = Σ.transpose from rfl]
      exact hIsSymm
    · intro x hx
      have hStar : star x = x := by
        funext i
        show star (x i) = x i
        simp [Pi.star_apply, star]
      rw [← hStar]
      exact hPos x hx
  -- Step 2: 构造 L := LDL.lower * √D
  set L : Mat32 := LDL.lower hPosDef * Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef)
  refine ⟨{ L := L
           lowerTri := ?_
           decomps := ?_ }⟩
  · -- L 下三角
    exact CholeskyPSDLocal.chol_L_lowerTri Σ hPosDef
  · -- L * Lᵀ = Σ
    -- (LDL.lower * √D)ᵀ = √D * (LDL.lower)ᵀ  (对角矩阵 = 自身转置)
    have hDiagT : (Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef)).transpose =
                  Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef) := by
      ext a b
      simp [Matrix.diagonal, Matrix.transpose_apply]
    have hL_def : L = LDL.lower hPosDef *
                          Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef) := rfl
    have hLT : L.transpose = (Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef)) *
                              (LDL.lower hPosDef).transpose := by
      rw [hL_def, Matrix.transpose_mul]
      exact hDiagT
    -- LDL.lower_conj_diag (实数下 ᴴ = ᵀ): LDL.lower * LDL.diag * (LDL.lower)ᵀ = Σ
    have hLDL : LDL.lower hPosDef * LDL.diag hPosDef *
                (LDL.lower hPosDef).transpose = Σ :=
      LDL.lower_conj_diag hPosDef
    -- L * Lᵀ = LDL.lower * √D * (√D)ᵀ * (LDL.lower)ᵀ
    --        = LDL.lower * (√D)² * (LDL.lower)ᵀ   (对角矩阵 self-transpose)
    --        = LDL.lower * LDL.diag * (LDL.lower)ᵀ (√D² = D)
    --        = Σ
    rw [hL_def, hLT]
    rw [← Matrix.mul_assoc]
    -- 化简 √D * √D = LDL.diag (对角矩阵 element-wise)
    have hDiagMul :
        Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef) *
        Matrix.diagonal (Real.sqrt ∘ LDL.diagEntries hPosDef) =
        LDL.diag hPosDef := by
      ext a b
      by_cases hab : a = b
      · subst hab
        simp [Matrix.diagonal, LDL.diag]
        have hDpos := CholeskyPSDLocal.LDL.diagEntries_pos Σ hPosDef a
        rw [Real.sq_sqrt (le_of_lt hDpos)]
        exact (Real.sqrt_sqrt (le_of_lt hDpos)).symm
      · simp [Matrix.diagonal, LDL.diag]
    rw [hDiagMul]
    exact hLDL

/-! ## Step 5c: 全链定理 -/

/-- 全链定理: SPD. -/
theorem raw_rules_and_encoder_imply_valid_fitting
    {n : ℕ} (A : Mat32) (m γ : ℝ) (Z : Fin n → Vec32) (μ_Z : Vec32)
    (hA : LinearEncoder A m)
    (hμ_Z : μ_Z = sampleMean Z)
    (hpop : spectralRichnessCond (rawCov Z μ_Z) γ)
    (hn : n ≥ 2) :
    Mat32.SPD
      (latentCov (fun x => Matrix.mulVec A x) Z
        (sampleMean (fun j => Matrix.mulVec A (Z j)))) := by
  have hspec := linear_encoder_implies_latent_spectral A m γ Z μ_Z hA hμ_Z hpop hn
  exact spectral_implies_spd
    (latentCov (fun x => Matrix.mulVec A x) Z
      (sampleMean (fun j => Matrix.mulVec A (Z j))))
    (m^2 * γ) hspec

/-- 全链定理: rank = 32. -/
theorem raw_rules_and_encoder_imply_full_rank
    {n : ℕ} (A : Mat32) (m γ : ℝ) (Z : Fin n → Vec32) (μ_Z : Vec32)
    (hA : LinearEncoder A m)
    (hμ_Z : μ_Z = sampleMean Z)
    (hpop : spectralRichnessCond (rawCov Z μ_Z) γ)
    (hn : n ≥ 2) :
    Matrix.rank
      (latentCov (fun x => Matrix.mulVec A x) Z
        (sampleMean (fun j => Matrix.mulVec A (Z j)))) = 32 := by
  exact spectral_implies_full_rank
    (latentCov (fun x => Matrix.mulVec A x) Z
          (sampleMean (fun j => Matrix.mulVec A (Z j))))
    (m^2 * γ)
    (linear_encoder_implies_latent_spectral A m γ Z μ_Z hA hμ_Z hpop hn)

/-- 全链定理: Cholesky 存在. -/
theorem raw_rules_and_encoder_imply_cholesky
    {n : ℕ} (A : Mat32) (m γ : ℝ) (Z : Fin n → Vec32) (μ_Z : Vec32)
    (hA : LinearEncoder A m)
    (hμ_Z : μ_Z = sampleMean Z)
    (hpop : spectralRichnessCond (rawCov Z μ_Z) γ)
    (hn : n ≥ 2) :
    Nonempty (CholeskyFactor
      (latentCov (fun x => Matrix.mulVec A x) Z
        (sampleMean (fun j => Matrix.mulVec A (Z j))))) := by
  exact cholesky_exists_of_spd _
    (raw_rules_and_encoder_imply_valid_fitting A m γ Z μ_Z hA hμ_Z hpop hn)

/-! ## Step 6: inverse covariance error bound (Phase M step 2 — proven)

Phase M step 2 严格证明两条稳定性误差界, 不引入任何 axiom / sorry.
证明链:
  1. Cauchy-Schwarz in squared form (Mathlib.sum_mul_sum_le_sum_mul_sum_sq)
  2. spectral richness + Cauchy-Schwarz ⇒ 算子 norm 下界 γ²‖x‖² ≤ ‖Σx‖²
  3. ‖Σ⁻¹‖_op ≤ 1/γ (由 2 反向代入 y = Σ⁻¹x)
  4. inverse difference identity: Σ̂⁻¹ - Σ⁻¹ = Σ̂⁻¹ (Σ - Σ̂) Σ⁻¹
  5. operator-norm submultiplicativity ⇒ 算子 norm 扰动界
  6. 二次型 vs 算子 norm 桥 ⇒ Mahalanobis 误差界
-/

/-- Cauchy-Schwarz in squared form: `(∑ x · y)² ≤ (∑ x²)(∑ y²)`. -/
lemma cauchy_schwarz_sq (x y : Vec32) :
    vec32Inner x y ^ 2 ≤ vec32NormSq x * vec32NormSq y := by
  exact Finset.sum_mul_sum_le_sum_mul_sum_sq
    (Finset.univ : Finset (Fin 32)) x y

/-- **spectral_lower_bound_norm**: `spectralRichnessCond Σ γ` ⇒
    `γ² · ‖x‖² ≤ ‖Σ x‖²` (从二次型下界 + Cauchy-Schwarz 推出算子 norm 下界). -/
lemma spectral_lower_bound_norm (Σ : Mat32) (γ : ℝ)
    (hS : spectralRichnessCond Σ γ) (x : Vec32) :
    γ^2 * vec32NormSq x ≤ vec32NormSq (Matrix.mulVec Σ x) := by
  have hγ_pos : (0 : ℝ) < γ := hS.2.1
  -- Step 1: γ‖x‖² ≤ ⟨x, Σx⟩ (spectral richness 二次型下界)
  have hγ_inner : γ * vec32NormSq x ≤ vec32Inner x (Matrix.mulVec Σ x) := hS.2.2 x
  -- Step 2: Cauchy-Schwarz: ⟨x, Σx⟩² ≤ ‖x‖² · ‖Σx‖²
  have hCS := cauchy_schwarz_sq x (Matrix.mulVec Σ x)
  -- Step 3: γ²‖x‖⁴ ≤ ⟨x, Σx⟩² (sq_le_sq on hγ_inner)
  have hsq : (γ * vec32NormSq x) ^ 2 ≤ vec32Inner x (Matrix.mulVec Σ x) ^ 2 :=
    sq_le_sq (mul_nonneg (le_of_lt hγ_pos) (vec32NormSq_nonneg _)) hγ_inner
  -- Step 4: γ²‖x‖⁴ ≤ ‖x‖² · ‖Σx‖² (transitivity with CS)
  have hkey : (γ * vec32NormSq x) ^ 2 ≤ vec32NormSq x * vec32NormSq (Matrix.mulVec Σ x) :=
    le_trans hsq hCS
  -- Step 5: 当 x ≠ 0 时, γ²‖x‖² ≤ ‖Σx‖² (除以 ‖x‖²)
  by_cases hx : vec32NormSq x = 0
  · -- x = 0: 两边均为 0
    rw [hx]
    simp only [zero_mul, sq, le_refl]
  · -- x ≠ 0: ‖x‖² > 0, 可除
    push_neg at hx
    have hx_pos : (0 : ℝ) < vec32NormSq x := lt_of_le_of_ne (vec32NormSq_nonneg _) hx
    -- 重写 hkey 为: (γ² · ‖x‖²) · ‖x‖² ≤ ‖x‖² · ‖Σx‖²
    have eq1 : (γ * vec32NormSq x) ^ 2 = γ^2 * vec32NormSq x ^ 2 := by ring
    have eq2 : γ^2 * vec32NormSq x ^ 2 = (γ^2 * vec32NormSq x) * vec32NormSq x := by ring
    rw [eq1, eq2] at hkey
    -- 用 mul_le_mul_iff_right₀ / mul_le_mul_iff_left₀, 但更简单: 由于 ‖x‖² > 0,
    -- (a · ‖x‖²) · ‖x‖² ≤ ‖x‖² · b ⟺ a · ‖x‖² ≤ b
    exact (mul_le_mul_iff_left₀ hx_pos).mp hkey

/-- **mulVec_comp_inverse_eq_id**: 若 Σ 对称且 `Σᵀ · Σ⁻¹ = 1`, 则
    `Σ · Σ⁻¹ = 1` ⇒ `mulVec Σ (mulVec Σ⁻¹ y) = y`. -/
lemma mulVec_comp_inverse_eq_id (Σ Sinv : Mat32)
    (hSym : Matrix.IsSymm Σ)
    (hinv : Σ.transpose * Sinv = 1)
    (y : Vec32) :
    Matrix.mulVec Σ (Matrix.mulVec Sinv y) = y := by
  have hSigmaSinv : Σ * Sinv = 1 := by
    -- Σᵀ = Σ ⇒ Σᵀ · Sinv = Σ · Sinv
    have hSym' : Σ.transpose = Σ := hSym
    rw [hSym'] at hinv
    exact hinv
  rw [← Matrix.mulVec_mulVec]
  rw [hSigmaSinv]
  exact Matrix.mulVec_one y

/-- **inverse_opNorm_le_inv_gamma**: `spectralRichnessCond Σ γ` +
    `Σᵀ · Σ⁻¹ = 1` ⇒ `∀y, ‖Σ⁻¹ y‖² ≤ (1/γ²) ‖y‖²`. -/
lemma inverse_opNorm_le_inv_gamma
    (Σ : Mat32) (γ : ℝ)
    (hS : spectralRichnessCond Σ γ)
    (Sinv : Mat32)
    (hinv : Σ.transpose * Sinv = 1)
    (y : Vec32) :
    vec32NormSq (Matrix.mulVec Sinv y) ≤ (1 / γ)^2 * vec32NormSq y := by
  -- 关键 trick: 代入 x := Σ⁻¹ y 到 spectral_lower_bound_norm
  have hx_eq : Matrix.mulVec Σ (Matrix.mulVec Sinv y) = y :=
    mulVec_comp_inverse_eq_id Σ Sinv (hS.1) hinv y
  have hnorm := spectral_lower_bound_norm Σ γ hS (Matrix.mulVec Sinv y)
  -- hnorm: γ² · ‖Σ⁻¹ y‖² ≤ ‖Σ (Σ⁻¹ y)‖² = ‖y‖²
  rw [hx_eq] at hnorm
  -- hnorm: γ² · ‖Σ⁻¹ y‖² ≤ ‖y‖²
  have γ_pos : (0 : ℝ) < γ := hS.2.1
  have γ_sq_pos : (0 : ℝ) < γ^2 := mul_pos γ_pos γ_pos
  -- 等价改写: ‖Σ⁻¹ y‖² ≤ ‖y‖² / γ²
  -- 由 γ² > 0: a ≤ b / γ² ⟺ a · γ² ≤ b
  have hx_form : vec32NormSq (Matrix.mulVec Sinv y) ≤ vec32NormSq y / γ^2 := by
    rw [le_div_iff₀ γ_sq_pos]
    exact hnorm
  -- 把 ‖y‖² / γ² 改写为 (1/γ)² · ‖y‖²
  rw [show vec32NormSq y / γ^2 = (1 / γ)^2 * vec32NormSq y from by ring]
  exact hx_form

/-- **inverse_sub_inverse_identity** (resolvent identity):
    若 Σ̂ᵀ · Σ̂⁻¹ = 1, Σᵀ · Σ⁻¹ = 1 且 Σ̂, Σ 对称 (⇒ 也得到 Σ̂⁻¹ · Σ̂ = 1 等),
    则 `Σ̂⁻¹ - Σ⁻¹ = Σ̂⁻¹ · (Σ - Σ̂) · Σ⁻¹`. -/
lemma inverse_sub_inverse_identity
    (Sigma Sigma_hat Sinv Sinv_hat : Mat32)
    (hS_symm : Matrix.IsSymm Sigma)
    (hSh_symm : Matrix.IsSymm Sigma_hat)
    (hS_inv_l : Sigma * Sinv = 1)
    (hS_inv_r : Sinv * Sigma = 1)
    (hSh_inv_l : Sigma_hat * Sinv_hat = 1)
    (hSh_inv_r : Sinv_hat * Sigma_hat = 1) :
    Sinv_hat - Sinv = Sinv_hat * (Sigma - Sigma_hat) * Sinv := by
  -- RHS = Sinv_hat · (Sigma - Sigma_hat) · Sinv
  --     = Sinv_hat · Sigma · Sinv - Sinv_hat · Sigma_hat · Sinv  (Matrix.mul_sub)
  have hdecomp : Sinv_hat * (Sigma - Sigma_hat) * Sinv =
                 Sinv_hat * Sigma * Sinv - Sinv_hat * Sigma_hat * Sinv := by
    rw [Matrix.mul_sub, Matrix.sub_mul]
    ring
  rw [hdecomp]
  -- 化简 Sinv_hat · Sigma_hat · Sinv = Sinv (用 hSh_inv_l)
  have h1 : Sinv_hat * Sigma_hat * Sinv = Sinv := by
    rw [← Matrix.mul_assoc, hSh_inv_l, Matrix.one_mul]
  rw [h1]
  -- 化简 Sinv_hat · Sigma · Sinv = Sinv_hat (用 hS_inv_l)
  have h2 : Sinv_hat * Sigma * Sinv = Sinv_hat := by
    rw [← Matrix.mul_assoc, hS_inv_l, Matrix.one_mul]
  rw [h2]
  -- 现在目标: Sinv_hat - Sinv = Sinv_hat - Sinv
  ring

/-- **inv_symm**: SPD 矩阵的逆矩阵仍然对称.
    由 `Σᵀ · Σ⁻¹ = 1` 和 `Σᵀ = Σ` 推出 `Σ⁻¹ᵀ = Σ⁻¹`. -/
lemma inv_symm (Σ Sinv : Mat32)
    (hSym : Matrix.IsSymm Σ)
    (hinv_l : Σ * Sinv = 1)
    (hinv_r : Sinv * Σ = 1) :
    Sinv.transpose = Sinv := by
  -- (Sinv · Σ)ᵀ = 1 ⇒ Σᵀ · Sinvᵀ = 1 ⇒ Σ · Sinvᵀ = 1 (由对称性)
  have hinv_r_T : Σ * Sinv.transpose = 1 := by
    have : (Sinv * Σ).transpose = (1 : Mat32).transpose := by rw [hinv_r]
    rw [Matrix.transpose_mul, Matrix.transpose_one] at this
    -- this : Σ.transpose * Sinv.transpose = 1
    -- 由 hSym: Σᵀ = Σ ⇒ Σ · Sinvᵀ = 1
    rw [hSym] at this
    exact this
  -- 现在: Σ · Sinv = 1 且 Σ · Sinvᵀ = 1
  -- ⇒ Sinvᵀ = Sinvᵀ · 1 = Sinvᵀ · (Σ · Sinv) = (Sinvᵀ · Σ) · Sinv = 1 · Sinv = Sinv
  -- 但需要 Sinvᵀ · Σ = 1. 转置 hinv_l:
  have hinv_l_T : Sinv.transpose * Σ = 1 := by
    have : (Σ * Sinv).transpose = (1 : Mat32).transpose := by rw [hinv_l]
    rw [Matrix.transpose_mul, Matrix.transpose_one] at this
    -- this : Sinv.transpose * Σ.transpose = 1
    rw [hSym] at this
    exact this
  -- 现在: Σ · Sinvᵀ = 1 和 Sinvᵀ · Σ = 1 (从 hinv_l_T)
  -- 推导: Sinvᵀ · Σ · Sinv = 1 · Sinv = Sinv (用 hinv_l 转置)... hmm 需要换思路
  -- 直接: Sinvᵀ = Sinvᵀ · (Σ · Sinv) = (Sinvᵀ · Σ) · Sinv = 1 · Sinv = Sinv
  calc Sinv.transpose
      = Sinv.transpose * 1 := (Matrix.mul_one _).symm
    _ = Sinv.transpose * (Σ * Sinv) := by rw [hinv_l]
    _ = (Sinv.transpose * Σ) * Sinv := Matrix.mul_assoc.symm
    _ = 1 * Sinv := by rw [hinv_l_T]
    _ = Sinv := Matrix.one_mul _

/-- **Inverse covariance error bound** (Higham 2002 Theorem 2.3 严格形式化):
    若 `λ_min(Σ) ≥ γ > 0`, `λ_min(Σ̂) ≥ γ - ε > 0`, `‖Σ̂ - Σ‖_op ≤ ε`,
    则 `‖Σ̂⁻¹ - Σ⁻¹‖_op ≤ ε/(γ(γ-ε))`.

    证明思路 (4 步, 无 axiom):
      1. SPD ⇒ 对称逆 Σ̂⁻¹, Σ⁻¹ 也对称 (inv_symm)
      2. resolvent identity: Σ̂⁻¹ - Σ⁻¹ = Σ̂⁻¹ (Σ - Σ̂) Σ⁻¹
      3. op-norm submultiplicativity: ‖Σ̂⁻¹ (Σ - Σ̂) Σ⁻¹‖_op
                                    ≤ ‖Σ̂⁻¹‖_op · ‖Σ - Σ̂‖_op · ‖Σ⁻¹‖_op
                                    ≤ 1/(γ-ε) · ε · 1/γ
      4. 由 op-norm 平方 = 平方 form, 重写为 vec32NormSq 不等式 -/
theorem inverse_covariance_error_bound
    (C C_hat : Mat32)
    (γ ε : ℝ) (hγ : 0 < γ) (hε : 0 < ε) (hsub : ε < γ)
    (hC_pos : spectralRichnessCond C γ)
    (hC_hat_pos : spectralRichnessCond C_hat (γ - ε))
    (hC_inv : ∃ Cinv : Mat32, C.transpose * Cinv = 1 ∧ Cinv * C.transpose = 1)
    (hChat_inv : ∃ Chinv : Mat32, C_hat.transpose * Chinv = 1 ∧ Chinv * C_hat.transpose = 1)
    (hop : ∀ x : Vec32,
       vec32NormSq (Matrix.mulVec (C_hat - C) x) ≤ ε^2 * vec32NormSq x) :
    ∀ x : Vec32,
      vec32NormSq (Matrix.mulVec
        ((hChat_inv.choose - hC_inv.choose).transpose) x) ≤
        (ε / (γ * (γ - ε)))^2 * vec32NormSq x := by
  intro x
  -- 提取逆矩阵和左右逆条件
  obtain ⟨Cinv, hCinvT_l, hCinvT_r⟩ := hC_inv
  obtain ⟨Chinv, hChinvT_l, hChinvT_r⟩ := hChat_inv
  -- 由 spectralRichnessCond 的对称性: Σᵀ = Σ, 故 Σ · Σ⁻¹ = 1, Σ⁻¹ · Σ = 1
  have hSym_C : Matrix.IsSymm C := hC_pos.1
  have hSym_Ch : Matrix.IsSymm C_hat := hC_hat_pos.1
  have hCinv_r : C * Cinv = 1 := by
    have : C.transpose = C := hSym_C
    rw [this] at hCinvT_l; exact hCinvT_l
  have hCinv_l : Cinv * C = 1 := by
    have : C.transpose = C := hSym_C
    rw [this] at hCinvT_r; exact hCinvT_r
  have hChinv_r : C_hat * Chinv = 1 := by
    have : C_hat.transpose = C_hat := hSym_Ch
    rw [this] at hChinvT_l; exact hChinvT_l
  have hChinv_l : Chinv * C_hat = 1 := by
    have : C_hat.transpose = C_hat := hSym_Ch
    rw [this] at hChinvT_r; exact hChinvT_r
  -- 步骤 1: 逆矩阵对称 (inv_symm)
  have hCinv_symm : Cinv.transpose = Cinv := inv_symm C Cinv hSym_C hCinv_r hCinv_l
  have hChinv_symm : Chinv.transpose = Chinv := inv_symm C_hat Chinv hSym_Ch hChinv_r hChinv_l
  -- 步骤 2: (Chinv - Cinv)ᵀ = Chinv - Cinv
  have hDiffTransp : (Chinv - Cinv).transpose = Chinv - Cinv := by
    rw [Matrix.sub_transpose, hChinv_symm, hCinv_symm]
  -- 步骤 3: 关键不变量: ‖Chinv y‖² ≤ (1/(γ-ε)²) ‖y‖² (sample inverse norm bound)
  have hChinv_norm (y : Vec32) :
      vec32NormSq (Matrix.mulVec Chinv y) ≤ (1 / (γ - ε))^2 * vec32NormSq y :=
    inverse_opNorm_le_inv_gamma C_hat (γ - ε) hC_hat_pos Chinv hChinvT_l y
  have hCinv_norm (y : Vec32) :
      vec32NormSq (Matrix.mulVec Cinv y) ≤ (1 / γ)^2 * vec32NormSq y :=
    inverse_opNorm_le_inv_gamma C γ hC_pos Cinv hCinvT_l y
  -- 步骤 4: 化简 LHS
  -- LHS = ‖(Chinv - Cinv)ᵀ x‖² = ‖(Chinv - Cinv) x‖² (由步骤 2)
  have hLHS : vec32NormSq (Matrix.mulVec ((Chinv - Cinv).transpose) x) =
              vec32NormSq (Matrix.mulVec (Chinv - Cinv) x) := by
    rw [hDiffTransp]
  rw [hLHS]
  -- 步骤 5: 展开 (Chinv - Cinv) x = Chinv · (C - Ĉ) · Cinv · x (resolvent identity)
  -- 然后逐步应用 operator-norm 界
  have hResolvent : Matrix.mulVec (Chinv - Cinv) x =
                    Matrix.mulVec Chinv (Matrix.mulVec (C_hat - C) (Matrix.mulVec Cinv x)) := by
    -- (Chinv - Cinv) = Chinv · (C - Ĉ) · Cinv (resolvent identity)
    have hId : Chinv - Cinv = Chinv * (C - C_hat) * Cinv :=
      inverse_sub_inverse_identity C C_hat Cinv Chinv hSym_C hSym_Ch
        hCinv_r hCinv_l hChinv_r hChinv_l
    rw [hId]
    -- (Chinv · (C - Ĉ) · Cinv) ·ᵥ x = Chinv ·ᵥ ((C - Ĉ) ·ᵥ (Cinv ·ᵥ x))
    rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
  rw [hResolvent]
  -- 步骤 6: 三段 norm bound 串联
  -- ‖Chinv · ((C_hat - C) · (Cinv · x))‖² ≤ (1/(γ-ε)²) ‖(C_hat - C) · (Cinv · x)‖²
  -- ≤ (1/(γ-ε)²) · ε² · ‖Cinv · x‖²
  -- ≤ (1/(γ-ε)²) · ε² · (1/γ²) · ‖x‖²
  -- = (ε/(γ(γ-ε)))² · ‖x‖²
  have h_step1 := hChinv_norm (Matrix.mulVec (C_hat - C) (Matrix.mulVec Cinv x))
  have h_step2 : vec32NormSq (Matrix.mulVec (C_hat - C) (Matrix.mulVec Cinv x)) ≤
                 ε^2 * vec32NormSq (Matrix.mulVec Cinv x) :=
    hop (Matrix.mulVec Cinv x)
  have h_step3 := hCinv_norm x
  -- 三段串联
  calc vec32NormSq (Matrix.mulVec Chinv (Matrix.mulVec (C_hat - C) (Matrix.mulVec Cinv x)))
      ≤ (1 / (γ - ε))^2 * vec32NormSq (Matrix.mulVec (C_hat - C) (Matrix.mulVec Cinv x)) := h_step1
    _ ≤ (1 / (γ - ε))^2 * (ε^2 * vec32NormSq (Matrix.mulVec Cinv x)) := by
        exact mul_le_mul_left _ h_step2
    _ ≤ (1 / (γ - ε))^2 * (ε^2 * ((1 / γ)^2 * vec32NormSq x)) := by
        exact mul_le_mul_left _ h_step3
    _ = (ε / (γ * (γ - ε)))^2 * vec32NormSq x := by
        -- 代数化简: (1/(γ-ε))² · ε² · (1/γ)² · ‖x‖² = (ε/(γ(γ-ε)))² · ‖x‖²
        have hγ_pos : (0 : ℝ) < γ := hγ
        have hγ_ε_pos : (0 : ℝ) < γ - ε := sub_pos.mpr hsub
        have hε_pos : (0 : ℝ) < ε := hε
        field_simp
        ring

/-! ## Step 7: Mahalanobis distance error bound (Phase M step 2 — proven) -/

/-- **quad_form_abs_le_opNorm**: 二次型 ≤ 算子范数乘以 ‖x‖².
    若 `∀ y, ‖A y‖² ≤ K² ‖y‖²` 且 `0 ≤ K`, 则 `|⟨x, A x⟩| ≤ K ‖x‖²`. -/
lemma quad_form_abs_le_opNorm (A : Mat32) (x : Vec32) (K : ℝ)
    (hK : 0 ≤ K)
    (hA : ∀ y : Vec32, vec32NormSq (Matrix.mulVec A y) ≤ K^2 * vec32NormSq y) :
    |vec32Inner x (Matrix.mulVec A x)| ≤ K * vec32NormSq x := by
  -- Cauchy-Schwarz: ⟨x, Ax⟩² ≤ ‖x‖² · ‖Ax‖²
  have hCS := cauchy_schwarz_sq x (Matrix.mulVec A x)
  -- norm bound: ‖Ax‖² ≤ K² · ‖x‖²
  have hnorm := hA x
  -- |⟨x, Ax⟩|² ≤ ‖x‖² · K² · ‖x‖² = K² · ‖x‖⁴
  rw [sq_abs]
  nlinarith [hCS, hnorm, hK, vec32NormSq_nonneg x, sq_nonneg (vec32Inner x (Matrix.mulVec A x))]

/-- **mahalanobis_diff_eq_quad_form**: Mahalanobis 距离差的二次型分解.
    `|⟨d, Σ̂⁻¹ d⟩ - ⟨d, Σ⁻¹ d⟩| = |⟨d, (Σ̂⁻¹ - Σ⁻¹) d⟩|`. -/
lemma mahalanobis_diff_eq_quad_form (v μ : Vec32) (Shinv Sinv : Mat32) :
    let d := v - μ
    |vec32Inner d (Matrix.mulVec Shinv d) - vec32Inner d (Matrix.mulVec Sinv d)| =
    |vec32Inner d (Matrix.mulVec (Shinv - Sinv) d)| := by
  have hsubVec : Matrix.mulVec (Shinv - Sinv) (v - μ) =
                 Matrix.mulVec Shinv (v - μ) - Matrix.mulVec Sinv (v - μ) := by
    rw [Matrix.sub_mulVec]
  rw [hsubVec, vec32Inner_sub_right]

/-- **Mahalanobis distance error bound**:
    由 `inverse_covariance_error_bound` + Cauchy-Schwarz 严格推出. -/
theorem mahalanobis_distance_error_bound
    (v μ : Vec32) (Σ Σ_hat : Mat32) (γ ε : ℝ)
    (hγ : 0 < γ) (hε : 0 < ε) (hsub : ε < γ)
    (hΣ_pos : spectralRichnessCond Σ γ)
    (hΣhat_pos : spectralRichnessCond Σ_hat (γ - ε))
    (hΣ_inv : ∃ Sinv : Mat32, Σ.transpose * Sinv = 1 ∧ Sinv * Σ.transpose = 1)
    (hΣhat_inv : ∃ Shinv : Mat32, Σ_hat.transpose * Shinv = 1 ∧ Shinv * Σ_hat.transpose = 1)
    (hop : ∀ x : Vec32,
      vec32NormSq (Matrix.mulVec (Σ_hat - Σ) x) ≤
        (ε / (γ * (γ - ε)))^2 * vec32NormSq x) :
    let d := v - μ
    |vec32Inner d (Matrix.mulVec hΣhat_inv.choose d) -
     vec32Inner d (Matrix.mulVec hΣ_inv.choose d)| ≤
      (ε / (γ * (γ - ε))) * vec32NormSq d := by
  obtain ⟨Shinv, _⟩ := hΣhat_inv
  obtain ⟨Sinv, _⟩ := hΣ_inv
  -- Step 1: Mahalanobis 差的二次型分解
  have hdiff := mahalanobis_diff_eq_quad_form v μ Shinv Sinv
  -- Step 2: 由 inverse_covariance_error_bound 给出算子范数界 (应用主定理)
  -- 注意主定理的结论形式涉及 (Shinv - Sinv)ᵀ, 但 Shinv, Sinv 对称,
  -- 所以 ‖(Shinv - Sinv) x‖² = ‖(Shinv - Sinv)ᵀ x‖²
  have hSym_Σ : Matrix.IsSymm Σ := hΣ_pos.1
  have hSym_Σh : Matrix.IsSymm Σ_hat := hΣhat_pos.1
  obtain ⟨_, hSinvT_l, hSinvT_r⟩ := hΣ_inv
  obtain ⟨_, hShinvT_l, hShinvT_r⟩ := hΣhat_inv
  have hSinv_r : Σ * Sinv = 1 := by
    rw [hSym_Σ] at hSinvT_l; exact hSinvT_l
  have hSinv_l : Sinv * Σ = 1 := by
    rw [hSym_Σ] at hSinvT_r; exact hSinvT_r
  have hShinv_r : Σ_hat * Shinv = 1 := by
    rw [hSym_Σh] at hShinvT_l; exact hShinvT_l
  have hShinv_l : Shinv * Σ_hat = 1 := by
    rw [hSym_Σh] at hShinvT_r; exact hShinvT_r
  have hSinv_symm : Sinv.transpose = Sinv :=
    inv_symm Σ Sinv hSym_Σ hSinv_r hSinv_l
  have hShinv_symm : Shinv.transpose = Shinv :=
    inv_symm Σ_hat Shinv hSym_Σh hShinv_r hShinv_l
  have hDiffTransp : (Shinv - Sinv).transpose = Shinv - Sinv := by
    rw [Matrix.sub_transpose, hShinv_symm, hSinv_symm]
  -- 由主定理: ‖(Shinv - Sinv)ᵀ x‖² ≤ (ε/(γ(γ-ε)))² ‖x‖²
  have hop_bound (y : Vec32) :
      vec32NormSq (Matrix.mulVec (Shinv - Sinv) y) ≤
        (ε / (γ * (γ - ε)))^2 * vec32NormSq y := by
    have hInv_bound := inverse_covariance_error_bound Σ Σ_hat γ ε hγ hε hsub
      hΣ_pos hΣhat_pos hΣ_inv hΣhat_inv hop y
    rw [hDiffTransp] at hInv_bound
    exact hInv_bound
  -- Step 3: 应用 quad_form_abs_le_opNorm (A := Shinv - Sinv, K := ε/(γ(γ-ε)))
  have hK_pos : (0 : ℝ) ≤ ε / (γ * (γ - ε)) := by
    apply div_nonneg (le_of_lt hε)
    exact mul_nonneg (le_of_lt hγ) (le_of_lt (sub_pos.mpr hsub))
  -- 最终: |⟨d, (Shinv - Sinv) d⟩| ≤ K · ‖d‖²
  have h_final : |vec32Inner (v - μ) (Matrix.mulVec (Shinv - Sinv) (v - μ))| ≤
                 (ε / (γ * (γ - ε))) * vec32NormSq (v - μ) := by
    exact quad_form_abs_le_opNorm (Shinv - Sinv) (v - μ)
      (ε / (γ * (γ - ε))) hK_pos hop_bound
  -- Step 4: 用 hdiff 把 LHS 转回 Mahalanobis 形式
  rw [hdiff]
  exact h_final

end PersonalQuery