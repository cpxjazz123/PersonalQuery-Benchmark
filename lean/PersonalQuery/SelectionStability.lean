/-
# PersonalQuery.SelectionStability — Phase O

Query-selection ranking stability theorem: 把 Phase M 的 Mahalanobis 误差界
桥接到 PQB 个性化 query 选择 (Stage 08 `selectByMinD2`).

## 核心定理

对两个 candidates `q_a`, `q_b` 假设
* 真值 Mahalanobis 距离 `D²(q) = (q - μ)ᵀ Σ⁻¹ (q - μ)` (用真实 Σ)
* 估计 Mahalanobis 距离 `D̂²(q) = (q - μ)ᵀ Σ̂⁻¹ (q - μ)` (用估计 Σ̂)
* 估计误差上界: `|D̂²(q) - D²(q)| ≤ Δ_q`

则
```
   D²(q_b) - D²(q_a) > Δ_a + Δ_b
⇒ D̂²(q_b) - D̂²(q_a) > 0
```

即 estimated Σ̂ 不会把 ranking 翻过来.

## Pipeline 集成

  Stage 04: 拟合 Σ̂ (spectral richness)
  Stage 08: 选 D̂² 最小的 candidate (`selectByMinD2`)
  Phase M: 给出 `|D̂²(q) - D²(q)| ≤ ε/(γ(γ-ε)) · ‖q - μ‖²`
  Phase O (本模块): 只要任何两个 candidate 的真值 gap 超过 2Δ,
                    estimated ranking 与真实 ranking 一致.

## 后续 (Phase P+)

把 `Δ_q = ε/(γ(γ-ε)) · ‖q - μ‖²` 代入, 得到关于 user-specific
covariance error ε / cohort spectral gap γ 的显式 ranking margin bound.

-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Basic

import PersonalQuery.Basic
import PersonalQuery.Mahalanobis
import PersonalQuery.EncoderSpectralBridge

namespace PersonalQuery

open Matrix

/-! ### Section 1: 实数版 ranking-stability 引理 -/

/-- Ranking stability (uniform Δ):
    若真值 gap `D²(q_b) - D²(q_a) > 2Δ` 且两个 candidate 的估计误差都 ≤ Δ,
    则 estimated ordering `D̂²(q_b) > D̂²(q_a)` 保持.

    这是 Phase O 的 *核心引理*: 由估计误差的均匀上界推 ranking 一致性.
    proof strategy:
      D̂²(q_b) - D̂²(q_a)
      = D²(q_b) - D²(q_a) + (D̂²(q_b) - D²(q_b)) - (D̂²(q_a) - D²(q_a))
      ≥ D²(q_b) - D²(q_a) - 2Δ
      > 0. -/
theorem ranking_stability_uniform_Δ
    (D2_a D2_b Dhat2_a Dhat2_b Δ : ℝ)
    (hgap : D2_b - D2_a > 2 * Δ)
    (hbound_a : |Dhat2_a - D2_a| ≤ Δ)
    (hbound_b : |Dhat2_b - D2_b| ≤ Δ) :
    Dhat2_b - Dhat2_a > 0 := by
  -- 关键代数恒等式
  have hkey : Dhat2_b - Dhat2_a =
    (D2_b - D2_a) + (Dhat2_b - D2_b) - (Dhat2_a - D2_a) := by ring
  rw [hkey]
  -- 拆上界: (Dhat2_b - D2_b) ≥ -|...| ≥ -Δ, 类似 a
  have hlb_a : -(Dhat2_a - D2_a) ≥ -Δ := by
    rw [neg_sub]
    linarith [abs_le.mp hbound_a]
  have hlb_b : (Dhat2_b - D2_b) ≥ -Δ := by
    linarith [abs_le.mp hbound_b]
  -- 由 hgap + hlb_a + hlb_b 推出
  linarith

/-- Ranking stability (per-candidate Δ): 允许两个 candidate 有不同误差上界.
    若 `D²(q_b) - D²(q_a) > Δ_a + Δ_b`, 则 estimated ordering 保持. -/
theorem ranking_stability_per_candidate_Δ
    (D2_a D2_b Dhat2_a Dhat2_b Δ_a Δ_b : ℝ)
    (hgap : D2_b - D2_a > Δ_a + Δ_b)
    (hbound_a : |Dhat2_a - D2_a| ≤ Δ_a)
    (hbound_b : |Dhat2_b - D2_b| ≤ Δ_b) :
    Dhat2_b - Dhat2_a > 0 := by
  have hkey : Dhat2_b - Dhat2_a =
    (D2_b - D2_a) + (Dhat2_b - D2_b) - (Dhat2_a - D2_a) := by ring
  rw [hkey]
  have hlb_a : -(Dhat2_a - D2_a) ≥ -Δ_a := by
    rw [neg_sub]
    linarith [abs_le.mp hbound_a]
  have hlb_b : (Dhat2_b - D2_b) ≥ -Δ_b := by
    linarith [abs_le.mp hbound_b]
  linarith

/-! ### Section 2: Mahalanobis 距离的 ranking-stability 实例化 -/

/-- Mahalanobis 距离的 *估计* 与 *真值* 形式.
    `mahaD2D` 是直接路径, 用 `invSigma` 即可计算. -/
def mahaD2D (z mu : Vec32) (invSigma : Mat32) : ℝ :=
  mahaD2 z mu invSigma

/-- 真值协方差下, `q_a` 的 Mahalanobis 距离比 `q_b` 小 (等价于 `D²(q_a) ≤ D²(q_b)`). -/
def isTrueCloser (q_a q_b mu : Vec32) (invΣ : Mat32) : Prop :=
  mahaD2 q_a mu invΣ ≤ mahaD2 q_b mu invΣ

/-- 估计协方差下, `q_a` 的 Mahalanobis 距离比 `q_b` 小. -/
def isEstimatedCloser (q_a q_b mu : Vec32) (invΣ̂ : Mat32) : Prop :=
  mahaD2 q_a mu invΣ̂ ≤ mahaD2 q_b mu invΣ̂

/-- **Phase O 主定理 (uniform Δ 版)**:

    给定 Phase M 的 per-candidate Mahalanobis 误差界
    `|D̂²(q) - D²(q)| ≤ Δ` for q ∈ {q_a, q_b},

    若真值 gap 严格大于 `2Δ`
    (即 `mahaD2 q_b μ Σ⁻¹ - mahaD2 q_a μ Σ⁻¹ > 2Δ`),
    则 estimated ordering 与真值 ranking 一致:
    `mahaD2 q_a μ Σ̂⁻¹ < mahaD2 q_b μ Σ̂⁻¹`. -/
theorem selection_ranking_stability_uniform
    (q_a q_b μ : Vec32)
    (Σ_inv Σ̂_inv : Mat32)
    (Δ : ℝ) (hΔ : 0 ≤ Δ)
    (hgap : mahaD2 q_b μ Σ_inv - mahaD2 q_a μ Σ_inv > 2 * Δ)
    (hbound_a : |mahaD2 q_a μ Σ̂_inv - mahaD2 q_a μ Σ_inv| ≤ Δ)
    (hbound_b : |mahaD2 q_b μ Σ̂_inv - mahaD2 q_b μ Σ_inv| ≤ Δ) :
    mahaD2 q_a μ Σ̂_inv < mahaD2 q_b μ Σ̂_inv := by
  apply ranking_stability_uniform_Δ
    (D2_a := mahaD2 q_a μ Σ_inv)
    (D2_b := mahaD2 q_b μ Σ_inv)
    (Dhat2_a := mahaD2 q_a μ Σ̂_inv)
    (Dhat2_b := mahaD2 q_b μ Σ̂_inv)
  · exact hgap
  · exact hbound_a
  · exact hbound_b

/-- **Phase O 主定理 (per-candidate Δ 版)**:
    更精细的版本, 允许 `q_a`, `q_b` 各自有不同的误差上界 `Δ_a`, `Δ_b`.
    若 `D²(q_b) - D²(q_a) > Δ_a + Δ_b`, 则 estimated ranking 保持.

    适用于两个 candidate 在 latent 空间中距离 μ 远近差异显著时
    (远处的 `‖q - μ‖²` 大, `Δ` 也大; 近处的 Δ 小). -/
theorem selection_ranking_stability_per_candidate
    (q_a q_b μ : Vec32)
    (Σ_inv Σ̂_inv : Mat32)
    (Δ_a Δ_b : ℝ) (hΔ_a : 0 ≤ Δ_a) (hΔ_b : 0 ≤ Δ_b)
    (hgap : mahaD2 q_b μ Σ_inv - mahaD2 q_a μ Σ_inv > Δ_a + Δ_b)
    (hbound_a : |mahaD2 q_a μ Σ̂_inv - mahaD2 q_a μ Σ_inv| ≤ Δ_a)
    (hbound_b : |mahaD2 q_b μ Σ̂_inv - mahaD2 q_b μ Σ_inv| ≤ Δ_b) :
    mahaD2 q_a μ Σ̂_inv < mahaD2 q_b μ Σ̂_inv := by
  apply ranking_stability_per_candidate_Δ
    (D2_a := mahaD2 q_a μ Σ_inv)
    (D2_b := mahaD2 q_b μ Σ_inv)
    (Dhat2_a := mahaD2 q_a μ Σ̂_inv)
    (Dhat2_b := mahaD2 q_b μ Σ̂_inv)
  · exact hgap
  · exact hbound_a
  · exact hbound_b

/-! ### Section 3: 与 Phase M 误差界的 pipeline 接口 -/

/-- `selection_ranking_stability_uniform` 与 Phase M
    `mahalanobis_distance_error_bound` 的直接组合:

    给定
    * spectral richness `λ_min(Σ) ≥ γ`, `λ_min(Σ̂) ≥ γ - ε`, `ε < γ`
    * operator-norm 误差 `∀x, ‖(Σ̂ - Σ)x‖ ≤ (ε/(γ(γ-ε)))·‖x‖`
    * 真值 ranking gap `D²(q_b) - D²(q_a) > 2Δ`

    若 `Δ ≥ ε/(γ(γ-ε)) · max(‖q_a - μ‖², ‖q_b - μ‖²)`, 则 estimated ranking 保持. -/
theorem selection_ranking_stability_via_phase_M
    (q_a q_b μ : Vec32)
    (Σ Σ̂ Σ_inv Σ̂_inv : Mat32)
    (γ ε Δ : ℝ)
    (hγ : 0 < γ) (hε : 0 < ε) (hsub : ε < γ)
    (hΣ_pos : spectralRichnessCond Σ γ)
    (hΣ̂_pos : spectralRichnessCond Σ̂ (γ - ε))
    (hΣ_inv : ∃ Sinv : Mat32, Σ.transpose * Sinv = 1 ∧ Sinv * Σ.transpose = 1)
    (hΣ̂_inv : ∃ Shinv : Mat32, Σ̂.transpose * Shinv = 1 ∧ Shinv * Σ̂.transpose = 1)
    (hop : ∀ x : Vec32,
      vec32NormSq (Matrix.mulVec (Σ̂ - Σ) x) ≤ (ε / (γ * (γ - ε)))^2 * vec32NormSq x)
    (hgap : mahaD2 q_b μ Σ_inv - mahaD2 q_a μ Σ_inv > 2 * Δ)
    (hΔ_nonneg : 0 ≤ Δ)
    (hΔ_a : Δ ≥ (ε / (γ * (γ - ε))) * vec32NormSq (vecDiff q_a μ))
    (hΔ_b : Δ ≥ (ε / (γ * (γ - ε))) * vec32NormSq (vecDiff q_b μ)) :
    mahaD2 q_a μ Σ̂_inv < mahaD2 q_b μ Σ̂_inv := by
  -- Step 1: 抽取 Phase M 给出的 per-candidate 误差界
  have hmaha_a := mahalanobis_distance_error_bound
    q_a μ Σ Σ̂ γ ε hγ hε hsub hΣ_pos hΣ̂_pos hΣ_inv hΣ̂_inv hop
  have hmaha_b := mahalanobis_distance_error_bound
    q_b μ Σ Σ̂ γ ε hγ hε hsub hΣ_pos hΣ̂_pos hΣ_inv hΣ̂_inv hop
  -- Step 2: 把 Δ_a, Δ_b 的下界翻译为 |D̂² - D²| ≤ Δ
  have hbound_a : |mahaD2 q_a μ Σ̂_inv - mahaD2 q_a μ Σ_inv| ≤ Δ := by
    apply le_trans hmaha_a
    -- Phase M bound 是 `(ε/(γ(γ-ε))) * vec32NormSq d`, 且该值 ≤ Δ
    simpa using hΔ_a
  have hbound_b : |mahaD2 q_b μ Σ̂_inv - mahaD2 q_b μ Σ_inv| ≤ Δ := by
    apply le_trans hmaha_b
    simpa using hΔ_b
  -- Step 3: 直接套用 selection_ranking_stability_uniform
  exact selection_ranking_stability_uniform q_a q_b μ Σ_inv Σ̂_inv Δ hΔ_nonneg hgap hbound_a hbound_b

/-! ### Section 4: selection 语义: pick smaller D² -/

/-- **Selection-correctness**: 在 estimated ranking 与真值 ranking 一致的前提下,
    用 Σ̂ 选出的 (D̂² 最小的) candidate 与用 Σ 选出的 (D² 最小的) 一致.

    对两个 candidates, 这意味着:
    若 D̂²(q_a) < D̂²(q_b) 且 estimated ranking 与真值一致,
    则 D²(q_a) ≤ D²(q_b) (所以 q_a 在真值下也是更近的). -/
theorem selection_correctness_under_stability
    (q_a q_b μ : Vec32)
    (Σ_inv Σ̂_inv : Mat32)
    (hest : mahaD2 q_a μ Σ̂_inv < mahaD2 q_b μ Σ̂_inv)
    (hgap_pos : mahaD2 q_b μ Σ_inv - mahaD2 q_a μ Σ_inv > 0) :
    mahaD2 q_a μ Σ_inv ≤ mahaD2 q_b μ Σ_inv := by
  -- 由 hgap_pos 直接: D²(q_b) > D²(q_a) ⟹ D²(q_a) < D²(q_b)
  linarith

/-- **Phase O → Stage 08 selectByMinD2 不变量**:

    给定 Phase M 误差上界 Δ, 若真值 gap `D²(q_b) - D²(q_a) > 2Δ`,
    则 estimated ordering `D̂²(q_a) < D̂²(q_b)` 保持 — 也就是说
    用 Σ̂ 选出的 D̂² 最小 candidate (Stage 08 selectByMinD2) 与
    用真值 Σ 选出的 ranking 一致.

    注: 这是*单方向*结论. 反方向 `D̂²(q_a) ≤ D̂²(q_b) ⟹ D²(q_a) ≤ D²(q_b)`
    需要更强的 gap 假设 `D²(q_b) - D²(q_a) > 2Δ` 的 *绝对值* 形式
    (而非单边). 工程上, Stage 08 总是从 estimated D̂² 选择,
    所以单方向 (真值 ranking 一致 ⟹ estimated 一致) 即可保证不变量. -/
theorem selectByMinD2_preserves_ranking
    (q_a q_b μ : Vec32)
    (Σ Σ̂ Σ_inv Σ̂_inv : Mat32)
    (γ ε Δ : ℝ)
    (hγ : 0 < γ) (hε : 0 < ε) (hsub : ε < γ)
    (hΣ_pos : spectralRichnessCond Σ γ)
    (hΣ̂_pos : spectralRichnessCond Σ̂ (γ - ε))
    (hΣ_inv : ∃ Sinv : Mat32, Σ.transpose * Sinv = 1 ∧ Sinv * Σ.transpose = 1)
    (hΣ̂_inv : ∃ Shinv : Mat32, Σ̂.transpose * Shinv = 1 ∧ Shinv * Σ̂.transpose = 1)
    (hop : ∀ x : Vec32,
      vec32NormSq (Matrix.mulVec (Σ̂ - Σ) x) ≤ (ε / (γ * (γ - ε)))^2 * vec32NormSq x)
    (hgap : mahaD2 q_b μ Σ_inv - mahaD2 q_a μ Σ_inv > 2 * Δ)
    (hΔ_nonneg : 0 ≤ Δ)
    (hΔ_a : Δ ≥ (ε / (γ * (γ - ε))) * vec32NormSq (vecDiff q_a μ))
    (hΔ_b : Δ ≥ (ε / (γ * (γ - ε))) * vec32NormSq (vecDiff q_b μ)) :
    mahaD2 q_a μ Σ̂_inv < mahaD2 q_b μ Σ̂_inv :=
  selection_ranking_stability_via_phase_M q_a q_b μ Σ Σ̂ Σ_inv Σ̂_inv
    γ ε Δ hγ hε hsub hΣ_pos hΣ̂_pos hΣ_inv hΣ̂_inv hop hgap hΔ_nonneg hΔ_a hΔ_b

/-- **estimated ranking 一致 ⟹ 真值 ranking 一致** (反向):

    若 D̂²(q_a) ≤ D̂²(q_b) 且 Δ ≥ Phase M 误差上界,
    则 D²(q_a) ≤ D²(q_b) + 2Δ, 即 ranking 接近但不完全一致.
    本定理给出弱保证: 真值 q_a 的 D² 最多比 q_b 大 2Δ. -/
theorem estimated_ranking_implies_truth_within_2Δ
    (q_a q_b μ : Vec32)
    (Σ Σ̂ Σ_inv Σ̂_inv : Mat32)
    (γ ε Δ : ℝ)
    (hγ : 0 < γ) (hε : 0 < ε) (hsub : ε < γ)
    (hΣ_pos : spectralRichnessCond Σ γ)
    (hΣ̂_pos : spectralRichnessCond Σ̂ (γ - ε))
    (hΣ_inv : ∃ Sinv : Mat32, Σ.transpose * Sinv = 1 ∧ Sinv * Σ.transpose = 1)
    (hΣ̂_inv : ∃ Shinv : Mat32, Σ̂.transpose * Shinv = 1 ∧ Shinv * Σ̂.transpose = 1)
    (hop : ∀ x : Vec32,
      vec32NormSq (Matrix.mulVec (Σ̂ - Σ) x) ≤ (ε / (γ * (γ - ε)))^2 * vec32NormSq x)
    (hest : mahaD2 q_a μ Σ̂_inv ≤ mahaD2 q_b μ Σ̂_inv)
    (hΔ_nonneg : 0 ≤ Δ)
    (hΔ_a : Δ ≥ (ε / (γ * (γ - ε))) * vec32NormSq (vecDiff q_a μ))
    (hΔ_b : Δ ≥ (ε / (γ * (γ - ε))) * vec32NormSq (vecDiff q_b μ)) :
    mahaD2 q_a μ Σ_inv ≤ mahaD2 q_b μ Σ_inv + 2 * Δ := by
  have hmaha_a := mahalanobis_distance_error_bound
    q_a μ Σ Σ̂ γ ε hγ hε hsub hΣ_pos hΣ̂_pos hΣ_inv hΣ̂_inv hop
  have hmaha_b := mahalanobis_distance_error_bound
    q_b μ Σ Σ̂ γ ε hγ hε hsub hΣ_pos hΣ̂_pos hΣ_inv hΣ̂_inv hop
  -- |D̂²(q_a) - D²(q_a)| ≤ Δ ⟹ D²(q_a) ≤ D̂²(q_a) + Δ
  -- |D̂²(q_b) - D²(q_b)| ≤ Δ ⟹ D²(q_b) ≥ D̂²(q_b) - Δ
  -- 由 D̂²(q_a) ≤ D̂²(q_b), 推出 D²(q_a) ≤ D̂²(q_a) + Δ ≤ D̂²(q_b) + Δ ≤ D²(q_b) + 2Δ
  have htight_a : mahaD2 q_a μ Σ_inv ≤ mahaD2 q_a μ Σ̂_inv + Δ := by
    rw [show mahaD2 q_a μ Σ̂_inv + Δ = Δ + mahaD2 q_a μ Σ̂_inv by ring]
    linarith [abs_le.mp hmaha_a]
  have htight_b : mahaD2 q_b μ Σ_inv ≥ mahaD2 q_b μ Σ̂_inv - Δ := by
    linarith [abs_le.mp hmaha_b]
  linarith

/-! ### Section 5: argminD2 over Fin n candidates (Phase P) -/

/-- **Phase P — argmin of D² over `Fin n` candidates**.

    返回 candidates 中 `mahaD2 (Z i) μ invΣ` 最小的 `i : Fin n`.

    实现策略: 用 `Finset.argmax (fun i => -mahaD2 ...)` 把 argmin 转化为 argmax,
    因为 Mathlib 的 `Finset.argmax` 找最大值; 取负翻转单调方向.
    需要 `ℝ` 是 `LinearOrder + InfSet` (默认有),
    与 `Finset.univ_nonempty` (由 `hn : 0 < n` 推出). -/
noncomputable def argminD2 {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32) (invΣ : Mat32) (hn : 0 < n) : Fin n :=
  (Finset.univ : Finset (Fin n)).argmax (fun i => -(mahaD2 (Z i) μ invΣ))
    (Finset.univ_nonempty.mpr ⟨⟨0, hn⟩⟩)

/-- `argminD2` 返回的 index 在 candidates 中. -/
theorem argminD2_mem {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32) (invΣ : Mat32) (hn : 0 < n) :
    argminD2 Z μ invΣ hn ∈ (Finset.univ : Finset (Fin n)) :=
  Finset.argmax_mem _ _ _

/-- **核心性质**: `argminD2` 给出的 D² 不超过任何其他 candidate 的 D².

    Proof: 由 `argmax` 定义, `-D²(k) ≤ -D²(argmax)`, 取负后即得 `D²(argmax) ≤ D²(k)`. -/
theorem argminD2_le {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32) (invΣ : Mat32) (hn : 0 < n)
    (k : Fin n) :
    mahaD2 (Z (argminD2 Z μ invΣ hn)) μ invΣ ≤ mahaD2 (Z k) μ invΣ := by
  unfold argminD2
  have hle := Finset.le_argmax (f := fun i => -(mahaD2 (Z i) μ invΣ))
    (s := (Finset.univ : Finset (Fin n))) (H := _) k (Finset.mem_univ k)
  -- hle : -(mahaD2 (Z k) μ invΣ) ≤ -(mahaD2 (Z (argmax ...)) μ invΣ)
  -- 由 neg 翻转得到 mahaD2 (Z (argmax ...)) ≤ mahaD2 (Z k)
  have hneg : -(mahaD2 (Z (argmax (fun i => -(mahaD2 (Z i) μ invΣ))
      (Finset.univ_nonempty.mpr ⟨⟨0, hn⟩⟩)) μ invΣ) ≤ -(mahaD2 (Z k) μ invΣ) := hle
  linarith

/-- **Phase O → argmin endpoint**: 若任意两个 candidates 的真值 D² gap > 2Δ,
    则 `argminD2 Z μ Σ̂_inv hn = argminD2 Z μ Σ_inv hn`.

    Proof by contradiction:
    * 设 `i := argminD2 Z μ Σ̂_inv hn` (Σ̂ argmin), `j := argminD2 Z μ Σ_inv hn` (Σ argmin).
    * 反设 `i ≠ j`. 由 `hgaps`, `|D²_Σ(i) - D²_Σ(j)| > 2Δ`.
    * 由 `argminD2_le` (Σ), `D²_Σ(j) ≤ D²_Σ(i)` ⇒ `D²_Σ(i) - D²_Σ(j) ≥ 0`,
      结合 `hgaps` 得 `D²_Σ(i) - D²_Σ(j) > 2Δ`.
    * 由 Phase M: `D̂²(i) ≥ D²(i) - Δ`, `D̂²(j) ≤ D²(j) + Δ`.
    * 故 `D̂²(i) - D̂²(j) ≥ (D²(i) - D²(j)) - 2Δ > 0`, 即 `D̂²(i) > D̂²(j)`.
    * 但 `argminD2_le` (Σ̂) 要求 `D̂²(i) ≤ D̂²(j)`, 矛盾.
    * 所以 `i = j`.

    工程意义: 在所有 pairwise gap 严格超过 `2Δ` 时, 用 Σ̂ 选出的 candidate
    与用真值 Σ 选出的 candidate 完全一致 — Stage 08 selectByMinD2 的 ranking
    在 Phase O + Phase M 联合保证下是 *稳定* 的. -/
theorem argminD2_agrees_under_stability
    {n : ℕ} (Z : Fin n → Vec32) (μ : Vec32)
    (Σ Σ̂ Σ_inv Σ̂_inv : Mat32)
    (γ ε Δ : ℝ)
    (hγ : 0 < γ) (hε : 0 < ε) (hsub : ε < γ)
    (hΣ_pos : spectralRichnessCond Σ γ)
    (hΣ̂_pos : spectralRichnessCond Σ̂ (γ - ε))
    (hΣ_inv : ∃ Sinv : Mat32, Σ.transpose * Sinv = 1 ∧ Sinv * Σ.transpose = 1)
    (hΣ̂_inv : ∃ Shinv : Mat32, Σ̂.transpose * Shinv = 1 ∧ Shinv * Σ̂.transpose = 1)
    (hop : ∀ x : Vec32,
      vec32NormSq (Matrix.mulVec (Σ̂ - Σ) x) ≤ (ε / (γ * (γ - ε)))^2 * vec32NormSq x)
    (hn : 0 < n)
    (hΔ_nonneg : 0 ≤ Δ)
    (hΔ_a : ∀ i : Fin n, Δ ≥ (ε / (γ * (γ - ε))) * vec32NormSq (vecDiff (Z i) μ))
    (hgaps : ∀ k l : Fin n, k ≠ l →
      |mahaD2 (Z k) μ Σ_inv - mahaD2 (Z l) μ Σ_inv| > 2 * Δ) :
    argminD2 Z μ Σ̂_inv hn = argminD2 Z μ Σ_inv hn := by
  set i := argminD2 Z μ Σ̂_inv hn with hi_def
  set j := argminD2 Z μ Σ_inv hn with hj_def
  -- Phase M 误差界 (uniform Δ): 每个 candidate 的 |D̂² - D²| ≤ Δ
  have hmaha : ∀ k : Fin n,
      |mahaD2 (Z k) μ Σ̂_inv - mahaD2 (Z k) μ Σ_inv| ≤ Δ := by
    intro k
    have hM := mahalanobis_distance_error_bound
      (Z k) μ Σ Σ̂ γ ε hγ hε hsub hΣ_pos hΣ̂_pos hΣ_inv hΣ̂_inv hop
    exact le_trans hM (hΔ_a k)
  -- Σ̂ argmin: D̂²(i) ≤ D̂²(j)
  have hest_le : mahaD2 (Z i) μ Σ̂_inv ≤ mahaD2 (Z j) μ Σ̂_inv :=
    argminD2_le Z μ Σ̂_inv hn j
  -- Σ argmin: D²(j) ≤ D²(i)
  have htrue_le : mahaD2 (Z j) μ Σ_inv ≤ mahaD2 (Z i) μ Σ_inv :=
    argminD2_le Z μ Σ_inv hn i
  -- 反证
  by_contra hneq
  -- |D²_Σ(i) - D²_Σ(j)| > 2Δ
  have hgap_ij := hgaps i j hneq
  -- case 1: D²_Σ(i) ≥ D²_Σ(j) (case 2 与 htrue_le 矛盾)
  by_cases hge : mahaD2 (Z i) μ Σ_inv ≥ mahaD2 (Z j) μ Σ_inv
  · -- D²(i) - D²(j) ≥ 0, 所以 |D²(i) - D²(j)| = D²(i) - D²(j) > 2Δ
    have hdiff_nonneg : mahaD2 (Z i) μ Σ_inv - mahaD2 (Z j) μ Σ_inv ≥ 0 := by linarith
    rw [abs_of_nonneg hdiff_nonneg] at hgap_ij
    have hdiff_big : mahaD2 (Z i) μ Σ_inv - mahaD2 (Z j) μ Σ_inv > 2 * Δ := hgap_ij
    -- D̂²(i) ≥ D²(i) - Δ
    have hi_lower : mahaD2 (Z i) μ Σ̂_inv ≥ mahaD2 (Z i) μ Σ_inv - Δ := by
      linarith [abs_le.mp (hmaha i)]
    -- D̂²(j) ≤ D²(j) + Δ
    have hj_upper : mahaD2 (Z j) μ Σ̂_inv ≤ mahaD2 (Z j) μ Σ_inv + Δ := by
      linarith [abs_le.mp (hmaha j)]
    -- D̂²(i) - D̂²(j) > 0
    have hest_diff_pos : mahaD2 (Z i) μ Σ̂_inv - mahaD2 (Z j) μ Σ̂_inv > 0 := by linarith
    -- 与 hest_le 矛盾
    linarith [hest_le, hest_diff_pos]
  · -- case 2: D²(i) < D²(j), 但 htrue_le 说 D²(j) ≤ D²(i), 矛盾
    linarith [htrue_le]

end PersonalQuery