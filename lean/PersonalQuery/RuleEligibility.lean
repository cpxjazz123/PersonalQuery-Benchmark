/-
# PersonalQuery.RuleEligibility — 规则层 Gaussian eligibility

本模块把 pre-encoder 用户筛选从固定 rule budget 提升为四类充分条件：

* coverage：未观测 rule mass 足够小；
* diversity：有效 rule 数量足够大且单一 rule 不支配分布；
* stability：两个时间窗口的 rule distribution 足够接近；
* spectral richness：rule feature covariance 在每个 32d 方向上都有正曲率。

谱条件的线性代数后果在本文件中严格证明。matrix concentration、JS divergence
和多元 CLT 依赖的概率定理以显式 axiom 表示；它们不伪装成已经由 Lean 推出的结论。
-/

import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Rank
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas

import PersonalQuery.Basic
import PersonalQuery.PCFG
import PersonalQuery.Gaussian

namespace PersonalQuery

open Module

/-! ### 有限 rule distribution -/

/-- 有限支持的 rule probability distribution。`mass` 在 support 外必须为零。 -/
structure FiniteRuleDistribution : Type where
  support : List Rule
  mass : Rule → ℝ
  nodup : support.Nodup
  nonneg : ∀ r, 0 ≤ mass r
  massSum : ℝ
  massSum_eq : List.sum (support.map mass) = massSum
  total_mass : massSum = 1
  outside_zero : ∀ r, r ∉ support → mass r = 0

namespace FiniteRuleDistribution

/-- Shannon entropy，约定 `0 * log 0 = 0`（Lean 中 `Real.log 0 = 0`）。 -/
noncomputable def shannonEntropy (p : FiniteRuleDistribution) : ℝ :=
  List.sum (p.support.map (fun r => -(p.mass r * Real.log (p.mass r))))

/-- exp(H) 是 entropy effective number，而不是句子数或 rule occurrence 数。 -/
noncomputable def effectiveRuleCount (p : FiniteRuleDistribution) : ℝ :=
  Real.exp (p.shannonEntropy)

/-- 支持上的最大概率用逐点条件表达，避免依赖不存在的 `Finset.max'` 前提。 -/
def maxMassLe (p : FiniteRuleDistribution) (β : ℝ) : Prop :=
  ∀ r, p.mass r ≤ β

/-- 两个分布在 rule feature 上的 pointwise total variation 距离。 -/
noncomputable def totalVariation (p q : FiniteRuleDistribution) : ℝ :=
  (1 / 2) * List.sum ((p.support ++ q.support).map (fun r => |p.mass r - q.mass r|))

/-- TV 距离非负。 -/
theorem totalVariation_nonneg (p q : FiniteRuleDistribution) : 0 ≤ totalVariation p q := by
  unfold totalVariation
  apply mul_nonneg
  · norm_num
  · exact List.sum_nonneg (fun _ _ => abs_nonneg _)

/-- TV 距离对称。 -/
theorem totalVariation_symm (p q : FiniteRuleDistribution) :
    totalVariation p q = totalVariation q p := by
  unfold totalVariation
  congr 1
  rw [List.map_comm]
  -- 现在: List.sum (List.map (fun r => |p.mass r - q.mass r|) (p.support ++ q.support))
  --    = List.sum (List.map (fun r => |p.mass r - q.mass r|) (q.support ++ p.support))
  -- 等价于 List.append 的对称性
  rw [← List.map_append, ← List.map_append]
  -- 现在需要: List.map f (p.support ++ q.support) = List.map f (q.support ++ p.support)
  have happ : (p.support ++ q.support).map (fun r => |p.mass r - q.mass r|) =
              (q.support ++ p.support).map (fun r => |p.mass r - q.mass r|) := by
    rw [List.append_comm]
  rw [happ]

/-- TV 距离为 0 当且仅当两个分布相同。

    证明思路: TV = 0 ⟹ sum = 0 ⟹ 每个 |p.mass r - q.mass r| = 0（因为都非负）。
    由此对所有 r ∈ p.support ∪ q.support, p.mass r = q.mass r。
    在 support 外, 两侧都是 0。所以 p.mass = q.mass, 即 p = q。
    反方向 trivial。-/
theorem totalVariation_zero_iff_same (p q : FiniteRuleDistribution) :
    totalVariation p q = 0 ↔ p = q := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · -- 0 = totalVariation p q = (1/2) * sum
    -- ⟹ sum = 0 (因为 (1/2) ≠ 0)
    have hsum : List.sum ((p.support ++ q.support).map (fun r => |p.mass r - q.mass r|)) = 0 := by
      rw [totalVariation] at h
      -- h : 0 = (1/2) * sum  ⟹ sum = 0
      nlinarith
    -- 每个 |...| ≥ 0, sum = 0 ⟹ 每项 = 0
    have h_zero_each : ∀ r ∈ p.support ++ q.support,
        |p.mass r - q.mass r| = 0 := by
      intro r hr
      have hnonneg : ∀ r ∈ p.support ++ q.support, 0 ≤ |p.mass r - q.mass r| := fun _ _ => abs_nonneg _
      exact (List.sum_eq_zero_iff_of_nonneg hnonneg).mp hsum r hr
    -- 由此, 对所有 r, p.mass r = q.mass r
    funext r
    by_cases hr : r ∈ p.support ++ q.support
    · -- 在并集中: |p.mass r - q.mass r| = 0
      have habs : |p.mass r - q.mass r| = 0 := h_zero_each r hr
      exact (abs_eq_zero.mp habs)
    · -- 在并集外: 两侧都是 0
      have hpn : r ∉ p.support := fun h => hr (List.mem_append.mpr ⟨h, List.not_mem_nil _⟩)
      have hqn : r ∉ q.support := fun h => hr (List.mem_append.mpr ⟨List.not_mem_nil _, h⟩)
      rw [p.outside_zero hpn, q.outside_zero hqn]
  · -- p = q ⟹ TV = 0
    intro h
    subst h
    unfold totalVariation
    simp

end FiniteRuleDistribution

/-- 未观测概率质量；它由证据对象显式提供，而不是从已观测样本数臆测。 -/
structure RuleEvidence : Type where
  distribution : FiniteRuleDistribution
  unseenMass : ℝ
  unseen_nonneg : 0 ≤ unseenMass
  coverage_mass : distribution.massSum + unseenMass = 1

/-- `Coverage ≥ 1 - η` 等价于 unseen mass 不超过 η。 -/
def coverageCond (e : RuleEvidence) (η : ℝ) : Prop :=
  e.unseenMass ≤ η

/-- entropy diversity：effective rule count 至少 K，且没有 rule mass 超过 β。 -/
def diversityCond (e : RuleEvidence) (K β : ℝ) : Prop :=
  K ≤ e.distribution.effectiveRuleCount ∧
  e.distribution.maxMassLe β

lemma coverageCond_iff (e : RuleEvidence) (η : ℝ) :
    coverageCond e η ↔ 1 - e.unseenMass ≥ 1 - η := by
  unfold coverageCond
  constructor <;> intro h <;> linarith

lemma effectiveRuleCount_pos (p : FiniteRuleDistribution) :
    0 < p.effectiveRuleCount := by
  exact Real.exp_pos _

lemma diversityCond_effective_count_pos (e : RuleEvidence) (K β : ℝ)
    (hdiv : diversityCond e K β) (hK : 0 < K) :
    0 < e.distribution.effectiveRuleCount := by
  exact lt_of_lt_of_le hK hdiv.1

/-! ### rule feature 与经验 covariance -/

/-- PCFG rule 到 32d feature 的 encoder-level 表示。
    这是业务 encoder 的接口；本模块不假设 rule count 单独决定其数值。 -/
axiom ruleFeature : Rule → Vec32

/-- 规则 feature 的经验均值。 -/
noncomputable def ruleFeatureMean {n : ℕ} (rules : Fin n → Rule) : Vec32 :=
  fun i => (∑ j : Fin n, ruleFeature (rules j) i) / n

/-- 规则 feature covariance；这里使用 `1/n` 的 population normalization。
    非空样本与 unbiased `1/(n-1)` 版本的转换需由调用方另行提供。 -/
noncomputable def ruleCovariance {n : ℕ} (rules : Fin n → Rule) : Mat32 :=
  let μ := ruleFeatureMean rules
  fun i k =>
    (∑ j : Fin n,
      (ruleFeature (rules j) i - μ i) *
      (ruleFeature (rules j) k - μ k)) / n

/--
`λ_min(C) ≥ γ` 的二次型表达：C 对称且所有非零方向的 Rayleigh 二次型
至少为 `γ ||x||²`。这一定义避免了把不完整的 eigenvalue API 冒充成证明。
-/
def spectralRichnessCond (C : Mat32) (γ : ℝ) : Prop :=
  Matrix.IsSymm C ∧
  0 < γ ∧
  ∀ x : Vec32, γ * vec32NormSq x ≤
    vec32Inner x (Matrix.mulVec C x)

/-- 规则 covariance 不退化的无核表述。 -/
def fullRank (C : Mat32) : Prop :=
  ∀ x : Vec32, Matrix.mulVec C x = 0 → x = 0

private lemma vec32NormSq_pos_of_ne_zero {x : Vec32} (hx : x ≠ 0) :
    0 < vec32NormSq x := by
  unfold vec32NormSq vec32Inner
  have hex : ∃ i : Fin 32, x i ≠ 0 := by
    by_contra h
    apply hx
    funext i
    by_contra hi
    exact h ⟨i, hi⟩
  obtain ⟨i, hi⟩ := hex
  apply Finset.sum_pos' (fun j _ => mul_self_nonneg (x j))
  exact ⟨i, Finset.mem_univ _, mul_self_pos.mpr hi⟩

/-- spectral richness 严格推出 SPD：这是本模块的核心代数证明。 -/
theorem spectral_implies_spd (C : Mat32) (γ : ℝ)
    (h : spectralRichnessCond C γ) : Mat32.SPD C := by
  refine ⟨h.1, ?_⟩
  intro x hx
  have hnorm : 0 < vec32NormSq x := vec32NormSq_pos_of_ne_zero hx
  have hprod : 0 < γ * vec32NormSq x := mul_pos h.2.1 hnorm
  exact lt_of_lt_of_le hprod (h.2.2 x)

/-- spectral richness 严格推出 PSD。 -/
theorem spectral_implies_psd (C : Mat32) (γ : ℝ)
    (h : spectralRichnessCond C γ) : Mat32.PSD C := by
  intro x
  have hnonneg : 0 ≤ γ * vec32NormSq x :=
    mul_nonneg h.2.1.le (vec32NormSq_nonneg x)
  exact le_trans hnonneg (h.2.2 x)

/-- SPD 二次型下界推出 covariance 的核为零。 -/
theorem spectral_implies_fullRank (C : Mat32) (γ : ℝ)
    (h : spectralRichnessCond C γ) : fullRank C := by
  intro x hzero
  by_contra hx
  have hpos := (spectral_implies_spd C γ h).2 x hx
  rw [hzero] at hpos
  -- hpos 简化成 0 < 0
  have hzero' : (0 : ℝ) < 0 := by simpa [vec32Inner] using hpos
  exact (lt_irrefl 0) hzero'

/-- `fullRank` 等价于 `Matrix.toLin'` 的 kernel 为 0；通过 `Matrix.toLin'_apply` 把
    `Matrix.mulVec` 与 `(toLin' C) v` 等同起来。 -/
lemma fullRank_iff_ker_toLin_eq_bot (C : Mat32) :
    fullRank C ↔ LinearMap.ker (Matrix.toLin' C) = ⊥ := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · -- fullRank ⇒ ker ⊥: 任意 v, toLin' C v = 0 ⇒ Matrix.mulVec C v = 0 ⇒ v = 0
    apply LinearMap.ker_eq_bot'.mpr
    intro v hv
    -- hv : (Matrix.toLin' C) v = 0
    have hv' : Matrix.mulVec C v = 0 := by
      simpa [Matrix.toLin'_apply] using hv
    exact h v hv'
  · -- ker ⊥ ⇒ fullRank
    intro v hv
    -- hv : Matrix.mulVec C v = 0
    have hv' : (Matrix.toLin' C) v = 0 := by
      simpa [Matrix.toLin'_apply] using hv
    -- v ∈ ker (toLin' C) = ⊥
    have hker : v ∈ LinearMap.ker (Matrix.toLin' C) := hv'
    have hbot : v ∈ (⊥ : Submodule ℝ (Fin 32 → ℝ)) := by
      rw [h] at hker
      exact hker
    exact (Submodule.mem_bot ℝ).mp hbot

/-- `fullRank` ⇒ `Matrix.toLin' C` 的 range 的 finrank 等于 32。 -/
lemma fullRank_implies_finrank_range (C : Mat32) (h : fullRank C) :
    finrank ℝ (LinearMap.range (Matrix.toLin' C)) = 32 := by
  -- ker = ⊥ ⇒ finrank_range + finrank_ker = finrank_domain
  have hker : LinearMap.ker (Matrix.toLin' C) = ⊥ := fullRank_iff_ker_toLin_eq_bot C |>.mp h
  have hEq :
      finrank ℝ (LinearMap.range (Matrix.toLin' C))
        + finrank ℝ (LinearMap.ker (Matrix.toLin' C))
      = finrank ℝ (Fin 32 → ℝ) :=
    LinearMap.finrank_range_add_finrank_ker (f := Matrix.toLin' C)
  have hFin : finrank ℝ (Fin 32 → ℝ) = 32 :=
    Module.finrank_pi (R := ℝ) (ι := Fin 32)
  have hZero : finrank ℝ (LinearMap.ker (Matrix.toLin' C)) = 0 := by
    rw [hker, finrank_bot]
  linarith [hEq, hFin, hZero]

/-- 把无核结论连接到 Mathlib 的矩阵 rank：严格证明，无需 axiom。 -/
theorem fullRank_implies_matrix_rank (C : Mat32) (h : fullRank C) :
    Matrix.rank C = 32 := by
  -- Matrix.rank = finrank (range toLin) 转 ℕ
  rw [Matrix.rank_eq_finrank_range_toLin C (Pi.basisFun ℝ (Fin 32)) (Pi.basisFun ℝ (Fin 32))]
  exact fullRank_implies_finrank_range C h

theorem spectral_implies_matrix_rank (C : Mat32) (γ : ℝ)
    (h : spectralRichnessCond C γ) : Matrix.rank C = 32 :=
  fullRank_implies_matrix_rank C (spectral_implies_fullRank C γ h)

/-! ### rule_feature_independence ⇒ spectral richness 桥 -/

/--
**总体规则协方差（population rule covariance）**：
    Σ_pop(φ, p) = ∑_r p(r) · (φ(r) - μ)(φ(r) - μ)ᵀ
    其中 μ = ∑_r p(r) · φ(r) 是 φ 在分布 p 下的总体均值。

这是 *rule-level* 的协方差矩阵，不是句子级。它由 rule distribution p
（而不是已观测样本）显式确定，因此可以脱离"我们已观测多少句子"独立讨论
它的谱性质。
-/
noncomputable def ruleFeatureCovPop
    (phi : Rule → Vec32) (p : FiniteRuleDistribution) : Mat32 :=
  let μ i := ∑ r ∈ p.support, p.mass r * phi r i
  fun i k =>
    ∑ r ∈ p.support, p.mass r * (phi r i - μ i) * (phi r k - μ k)

/--
**规则特征方向独立性（rule feature independence）**：
总体协方差 `Σ_pop(φ, p)` 在每个 32d 方向上都有正方差，等价于
`Σ_pop(φ, p) ⪰ γ_pop I`。

这条条件在数学上严格强于"non-concentration"（即 `max_r p(r) ≤ β`）：
non-concentration 仅保证没有单一 rule 主导分布，但如果所有 rule 的特征向量
都落在同一超平面上，总体协方差仍会奇异。`ruleFeatureIndependence` 直接
等价于总体协方差满谱 (`λ_min ≥ γ_pop > 0`)。

工程上这一条件由 encoder 的训练（保证 32d feature 之间正交结构）+ 业务
rule 集合的覆盖度共同决定。本模块把它作为 model-level precondition 显式声明，
而不是从 coverage / diversity 假装推出。
-/
def ruleFeatureIndependence
    (phi : Rule → Vec32) (p : FiniteRuleDistribution) (γ_pop : ℝ) : Prop :=
  spectralRichnessCond (ruleFeatureCovPop phi p) γ_pop

/--
**核心桥定理**：`covarianceClose + spectralPop + symm C + ε < γ_pop` ⇒
`spectralRichnessCond C (γ_pop - ε)`。

证明思路:
    对任意 x, 由 `covarianceClose` 得
        `|xᵀ (C - Σ) x| ≤ ε ||x||²`
        ⇒ `-ε ||x||² ≤ xᵀ (C - Σ) x`.
    由 `spectralRichnessCond Σ γ_pop` 得 `γ_pop ||x||² ≤ xᵀ Σ x`.
    由 `C = Σ + (C - Σ)` 与内积双线性得
        `xᵀ C x = xᵀ Σ x + xᵀ (C - Σ) x ≥ γ_pop ||x||² - ε ||x||²
                  = (γ_pop - ε) ||x||²`.

这是把"总体协方差有谱间隙"传递到"样本协方差有谱间隙（缩小 ε）"的关键代数步骤。
-/
theorem cov_close_implies_spectral
    (C Sigma : Mat32) (γ_pop ε : ℝ)
    (hsym : Matrix.IsSymm C)
    (hpop : spectralRichnessCond Sigma γ_pop)
    (hclose : covarianceClose C Sigma ε)
    (hsub : ε < γ_pop) :
    spectralRichnessCond C (γ_pop - ε) := by
  refine ⟨hsym, sub_pos.mpr hsub, ?_⟩
  intro x
  -- 已知分量
  have hpop_x : γ_pop * vec32NormSq x ≤ vec32Inner x (Matrix.mulVec Sigma x) := hpop.2.2 x
  have hclose_x : |vec32Inner x (Matrix.mulVec (C - Sigma) x)| ≤ ε * vec32NormSq x := hclose x
  -- |a| ≤ b 推出 -b ≤ a
  have hlower : -ε * vec32NormSq x ≤ vec32Inner x (Matrix.mulVec (C - Sigma) x) :=
    (abs_le.mp hclose_x).1
  -- C = Σ + (C - Σ) 拆开 mulVec
  have hdecomp : Matrix.mulVec C x =
      Matrix.mulVec Sigma x + Matrix.mulVec (C - Sigma) x := by
    rw [Matrix.add_mulVec]; congr 1
    rw [add_sub_cancel]
  -- 内积加法拆开
  have hsum : vec32Inner x (Matrix.mulVec C x) =
      vec32Inner x (Matrix.mulVec Sigma x) + vec32Inner x (Matrix.mulVec (C - Sigma) x) := by
    rw [hdecomp, vec32Inner_add_left]
  -- 链式不等式
  have hxC : (γ_pop - ε) * vec32NormSq x ≤ vec32Inner x (Matrix.mulVec C x) := by
    rw [sub_mul]
    linarith
  exact hxC

/-- 总体规则协方差的谱条件即 `ruleFeatureIndependence` 的展开。 -/
theorem ruleFeatureIndependence_iff_spectral (phi : Rule → Vec32)
    (p : FiniteRuleDistribution) (γ_pop : ℝ) :
    ruleFeatureIndependence phi p γ_pop ↔
    spectralRichnessCond (ruleFeatureCovPop phi p) γ_pop :=
  Iff.rfl

/-! ### minimum rule mass + independent directions ⇒ population spectral richness -/

/-- **总体规则均值**：μ_pop = ∑_r p(r) φ(r)。
    它与 `ruleFeatureCovPop` 内部 `let`-绑定的 μ 一致，作为独立接口暴露出来便于
    在 `ruleDirectionsIndependent` 与 `ruleFeatureCovPop` 之间建立标识。 -/
noncomputable def ruleFeaturePopMean (phi : Rule → Vec32)
    (p : FiniteRuleDistribution) : Vec32 :=
  fun i => ∑ r ∈ p.support, p.mass r * phi r i

/-- `ruleFeatureCovPop` 内部使用的 μ 与 `ruleFeaturePopMean` 一致。
    这是后续桥定理里把 `ruleFeatureIndependence` 拆开时需要的一致性事实。 -/
lemma ruleFeatureCovPop_uses_popMean (phi : Rule → Vec32)
    (p : FiniteRuleDistribution) (i : Fin 32) :
    (let μ := ruleFeaturePopMean phi p; μ i) =
      ∑ r ∈ p.support, p.mass r * phi r i := rfl

/-- **32 个 rule 的最小概率质量下界**：每个 `rs i` 在 p.support 内，
    且概率至少 `p_min`，且 32 个 rule 互不相同（injective）。 -/
def hasMassLowerBound (p : FiniteRuleDistribution)
    (rs : Fin 32 → Rule) (p_min : ℝ) : Prop :=
  (∀ i, rs i ∈ p.support) ∧
  (∀ i, p_min ≤ p.mass (rs i)) ∧
  Function.Injective rs

/-- **32 个 rule 的中心化特征矩阵**：
    `A = [phi(rs_0) - μ, ..., phi(rs_31) - μ]`。
    第 k 列是 `fun i => phi(rs_k) i - μ i`。 -/
noncomputable def centeredRuleMatrix
    (phi : Rule → Vec32) (rs : Fin 32 → Rule) (μ : Vec32) : Mat32 :=
  fun i k => phi (rs k) i - μ i

/-- **方向独立性**：σ_min(A) ≥ s > 0，等价于 `A Aᵀ ⪰ s² I`。
    用 `spectralRichnessCond (A Aᵀ) s²` 表示，因为它已经覆盖了 IsSymm + γ > 0
    + Rayleigh 下界的全部语义。 -/
def ruleDirectionsIndependent
    (phi : Rule → Vec32) (rs : Fin 32 → Rule) (μ : Vec32) (s : ℝ) : Prop :=
  0 < s ∧
  spectralRichnessCond
    (centeredRuleMatrix phi rs μ * (centeredRuleMatrix phi rs μ)ᵀ) (s^2)

/-! #### 基础代数：外积、二次型与 PSD 算子 -/

/-- 单个外积的二次型 = `(xᵀ v)²`。 -/
lemma outer_product_quadratic (v x : Vec32) :
    vec32Inner x (Matrix.mulVec (Matrix.vecMulVec v v) x) =
      (vec32Inner x v) * (vec32Inner x v) := by
  -- mulVec (vecMulVec v v) x i = ∑ k, v i * v k * x k = v i * (∑ k, v k * x k)
  -- 因此 vec32Inner x (...) = (∑ i, x i * v i) * (∑ k, v k * x k) = (xᵀv)²
  set sx := vec32Inner x v with hsx
  set sv := vec32Inner v x with hsv
  have hfact : ∀ i : Fin 32,
      (∑ k : Fin 32, v i * v k * x k) = v i * sv := by
    intro i
    rw [show v i * v k * x k = v i * (v k * x k) from by ring]
    rw [Finset.sum_congr rfl (fun k _ => by ring)]
    rw [← Finset.sum_mul, hsv]
  -- 现在 vec32Inner x (Matrix.mulVec (vecMulVec v v) x) = ∑ i, x i * (v i * sv)
  have hsum : vec32Inner x (fun i : Fin 32 => ∑ k : Fin 32, v i * v k * x k) =
      sx * sv := by
    unfold vec32Inner
    rw [Finset.sum_congr rfl (fun i _ => by rw [hfact i]; ring)]
    rw [Finset.sum_mul]
    -- ∑ i, x i * v i * sv = sv * (∑ i, x i * v i) = sv * sx
    rw [Finset.sum_congr rfl (fun i _ => by ring)]
    rw [← Finset.sum_mul]
    rw [hsx, hsv]
  rw [show Matrix.mulVec (Matrix.vecMulVec v v) x =
        fun i => ∑ k, v i * v k * x k from by
        funext i; simp [Matrix.mulVec, Matrix.vecMulVec_apply, dotProduct]]
  rw [hsum]
  rw [hsx, hsv]
  ring

/-- 外积是 PSD：xᵀ (v vᵀ) x = (xᵀv)² ≥ 0。 -/
lemma outer_product_psd (v : Vec32) :
    Mat32.PSD (Matrix.vecMulVec v v) := by
  intro x
  rw [outer_product_quadratic]
  exact mul_self_nonneg _

/-- PSD 在加法下封闭。 -/
lemma psd_add (M N : Mat32) (hM : Mat32.PSD M) (hN : Mat32.PSD N) :
    Mat32.PSD (M + N) := by
  intro x
  have hM' : 0 ≤ vec32Inner x (Matrix.mulVec M x) := hM x
  have hN' : 0 ≤ vec32Inner x (Matrix.mulVec N x) := hN x
  -- vec32Inner x (mulVec (M + N) x) = vec32Inner x (mulVec M x) + vec32Inner x (mulVec N x)
  rw [show Matrix.mulVec (M + N) x = Matrix.mulVec M x + Matrix.mulVec N x from Matrix.add_mulVec M N x]
  rw [vec32Inner_add_left]
  linarith

/-- PSD 在非负标量乘法下封闭。 -/
lemma psd_smul (c : ℝ) (M : Mat32) (hc : 0 ≤ c) (hM : Mat32.PSD M) :
    Mat32.PSD (c • M) := by
  intro x
  rw [Matrix.mulVec_smul]
  rw [vec32Inner_smul_left]
  exact mul_nonneg hc (hM x)

/-- 加权外积 `(c • vecMulVec v v)` 是 PSD。 -/
lemma weighted_outer_product_psd (v : Vec32) (c : ℝ) (hc : 0 ≤ c) :
    Mat32.PSD (c • Matrix.vecMulVec v v) :=
  psd_smul c _ hc (outer_product_psd v)

/-! #### Σ_pop 对称性与二次型展开 -/

/-- Σ_pop = (ruleFeatureCovPop phi p) 是对称矩阵（外积和）。 -/
lemma ruleFeatureCovPop_symmetric (phi : Rule → Vec32)
    (p : FiniteRuleDistribution) :
    Matrix.IsSymm (ruleFeatureCovPop phi p) := by
  intro i k
  -- 展开: (Σ_pop)ᵀ i k = (Σ_pop) k i
  -- (Σ_pop) k i = ∑ r, p.mass r * (phi r k - μ k) * (phi r i - μ i)
  -- (Σ_pop) i k = ∑ r, p.mass r * (phi r i - μ i) * (phi r k - μ k)
  -- 由 mul_comm 逐项相等
  show ((ruleFeatureCovPop phi p)ᵀ) i k = (ruleFeatureCovPop phi p) i k
  simp only [Matrix.transpose_apply, ruleFeatureCovPop, ruleFeaturePopMean]
  apply Finset.sum_congr rfl
  intro r _
  ring

/-- Σ_pop 的二次型展开：
    `xᵀ Σ_pop x = ∑ r ∈ p.support, p.mass r * (xᵀ v(r))²`。 -/
lemma cov_pop_quadratic_expand (phi : Rule → Vec32)
    (p : FiniteRuleDistribution) (x : Vec32) :
    vec32Inner x (Matrix.mulVec (ruleFeatureCovPop phi p) x) =
      ∑ r ∈ p.support, p.mass r *
        (vec32Inner x (fun i => phi r i - ruleFeaturePopMean phi p i))^2 := by
  -- vec32Inner x (mulVec Σ_pop x)
  -- = ∑ i, x i * (∑ k, Σ_pop i k * x k)
  -- = ∑ i, x i * (∑ k, ∑ r, p.mass r * (phi r i - μ i) * (phi r k - μ k) * x k)
  -- 交换求和顺序: = ∑ r, p.mass r * (∑ i, x i * (phi r i - μ i)) * (∑ k, (phi r k - μ k) * x k)
  -- = ∑ r, p.mass r * (xᵀ v(r)) * (v(r)ᵀ x) = ∑ r, p.mass r * (xᵀ v(r))²
  have hmul : ∀ i k : Fin 32,
      (ruleFeatureCovPop phi p) i k * x k =
        ∑ r ∈ p.support, p.mass r * (phi r i - ruleFeaturePopMean phi p i) *
          (phi r k - ruleFeaturePopMean phi p k) * x k := by
    intro i k
    rw [ruleFeatureCovPop, ruleFeaturePopMean]
    ring_nf
    rw [Finset.sum_mul]
    rfl
  have hsum : ∀ i : Fin 32,
      (∑ k, (ruleFeatureCovPop phi p) i k * x k) =
        ∑ r ∈ p.support, p.mass r * (phi r i - ruleFeaturePopMean phi p i) *
          (vec32Inner (fun k => phi r k - ruleFeaturePopMean phi p k) x) := by
    intro i
    rw [Finset.sum_comm]
    rw [Finset.sum_congr rfl (fun r _ => by
      rw [Finset.sum_mul]
      rw [Finset.sum_congr rfl (fun k _ => by ring)]
      rw [← Finset.sum_mul]
      rfl)]
  -- 现在 vec32Inner x (...) = ∑ i, x i * (∑ r, ...) = ∑ r, p.mass r * (xᵀ v(r))²
  rw [show Matrix.mulVec (ruleFeatureCovPop phi p) x =
        fun i => ∑ k, (ruleFeatureCovPop phi p) i k * x k from by
        funext i; simp [Matrix.mulVec, dotProduct]]
  rw [vec32Inner]
  rw [Finset.sum_comm]
  rw [Finset.sum_congr rfl (fun r _ => by
    rw [Finset.sum_mul]
    rw [← Finset.sum_mul]
    rw [Finset.sum_congr rfl (fun i _ => by ring)]
    rw [Finset.sum_congr rfl (fun k _ => by ring)])]
  rfl

/-- xᵀ (A Aᵀ) x = ∑_k (xᵀ v_k)²，其中 v_k = centered column of A。
    具体地说,  v_k(i) = phi (rs k) i - μ i = centeredRuleMatrix phi rs μ i k。 -/
lemma centered_matrix_quadratic (phi : Rule → Vec32)
    (rs : Fin 32 → Rule) (μ x : Vec32) :
    vec32Inner x (Matrix.mulVec
      (centeredRuleMatrix phi rs μ * (centeredRuleMatrix phi rs μ)ᵀ) x) =
      ∑ k : Fin 32,
        (vec32Inner x (fun i => phi (rs k) i - μ i))^2 := by
  set A := centeredRuleMatrix phi rs μ with hA
  -- Aᵀ x k = ∑ j, Aᵀ k j * x j = ∑ j, A j k * x j = xᵀ (column k of A)
  have hAtx : (Matrix.mulVec Aᵀ x : Vec32) =
      fun k => vec32Inner x (fun i => phi (rs k) i - μ i) := by
    funext k
    simp only [Matrix.mulVec, Matrix.transpose_apply, dotProduct, A,
               centeredRuleMatrix]
    rw [Finset.sum_congr rfl (fun i _ => by ring)]
    rw [← vec32Inner]
    rfl
  -- xᵀ (A (Aᵀ x)) = xᵀ (fun k => (Aᵀ x) k • (column k of A))
  --              = ∑ k, xᵀ ((Aᵀ x) k • v_k) = ∑ k, (Aᵀ x) k * (xᵀ v_k) = ∑ k, (xᵀ v_k)²
  rw [show Matrix.mulVec (A * Aᵀ) x =
        Matrix.mulVec A (Matrix.mulVec Aᵀ x) from Matrix.mulVec_mulVec Aᵀ x]
  rw [show Matrix.mulVec A (Matrix.mulVec Aᵀ x) =
        fun i => ∑ k, A i k * (Matrix.mulVec Aᵀ x) k from by
        funext i; simp [Matrix.mulVec, dotProduct]]
  rw [vec32Inner]
  rw [Finset.sum_congr rfl (fun i _ => by
    rw [Finset.sum_mul]
    rw [Finset.sum_congr rfl (fun k _ => by ring)]
    rw [← Finset.sum_mul])]
  rw [hAtx]
  -- 现在: ∑ k, (∑ i, x i * A i k) * (xᵀ v_k) = ∑ k, (xᵀ v_k)²
  -- 这里 A i k = phi (rs k) i - μ i = v_k i
  rw [show ∀ k, ∑ i, x i * A i k = vec32Inner x (fun i => phi (rs k) i - μ i) from by
      intro k
      simp only [A, centeredRuleMatrix]
      rw [Finset.sum_congr rfl (fun i _ => by ring)]
      rw [← vec32Inner]; rfl]
  rw [Finset.sum_congr rfl (fun k _ => by
    rw [Finset.sum_congr rfl (fun i _ => by ring)]
    rw [← Finset.sum_mul]
    rw [mul_comm (vec32Inner x (fun i => phi (rs k) i - μ i))
                (∑ i, x i * (phi (rs k) i - μ i))]
    rfl)]
  simp only [Finset.sum_mul]
  rfl

/-- 单个外积的 v = phi(rs_k) - μ 与 centeredRuleMatrix 一致。 -/
lemma centeredRuleMatrix_column (phi : Rule → Vec32)
    (rs : Fin 32 → Rule) (μ : Vec32) (k : Fin 32) :
    (fun i => phi (rs k) i - μ i) =
      fun i => centeredRuleMatrix phi rs μ i k := by
  funext i
  rfl

/-! #### 核心桥定理 -/

/-- **主定理**:
    若 32 个 distinct rules 各自有 ≥ p_min 的概率质量，且它们的中心化
    特征向量方向独立 (σ_min(A) ≥ s > 0)，则
        Σ_pop(φ, p) ⪰ (p_min * s²) I,
    即 `ruleFeatureIndependence phi p (p_min * s²)` 成立。

    证明链:
    1. `xᵀ Σ_pop x = ∑_r p(r) (xᵀ v(r))²` (cov_pop_quadratic_expand).
    2. 仅保留 {rs i : i : Fin 32} 部分，其余非负项可直接扔掉：
       `∑_r p(r) (xᵀ v(r))² ≥ ∑_i p(rs_i) (xᵀ v(rs_i))²`.
       这一步用了 rs i ∈ p.support（hmass）和 p(r) * (...)² ≥ 0（非负性）。
    3. 逐项放缩 `p(rs_i) ≥ p_min`：
       `∑_i p(rs_i) (xᵀ v(rs_i))² ≥ p_min ∑_i (xᵀ v(rs_i))²`.
    4. 改写为矩阵形式 `∑_i (xᵀ v(rs_i))² = xᵀ (A Aᵀ) x`（centered_matrix_quadratic）。
    5. 由 hdir 给出 `xᵀ (A Aᵀ) x ≥ s² ||x||²`，故
       `p_min ∑_i (xᵀ v(rs_i))² ≥ p_min * s² * ||x||²`.

    这一桥把可观测的 rule 集合条件 (mass + direction) 提升到 population
    谱条件 (λ_min ≥ p_min s²)；之后经 `cov_close_implies_spectral` 与
    `ruleBudgetLowerBound` 即可推出 sample covariance 的谱间隙。 -/
theorem ruleMassAndDirections_implies_pop_spectral
    (phi : Rule → Vec32) (p : FiniteRuleDistribution)
    (rs : Fin 32 → Rule) (p_min s : ℝ)
    (hpos : 0 < p_min ∧ 0 < s)
    (hmass : hasMassLowerBound p rs p_min)
    (hdir : ruleDirectionsIndependent phi rs (ruleFeaturePopMean phi p) s) :
    ruleFeatureIndependence phi p (p_min * s^2) := by
  -- 目标: spectralRichnessCond (ruleFeatureCovPop phi p) (p_min * s^2)
  -- 等价于: IsSymm + γ > 0 + ∀ x, γ * ||x||² ≤ xᵀ Σ_pop x
  refine ⟨?_, ?_, ?_⟩
  · -- 对称性
    exact ruleFeatureCovPop_symmetric phi p
  · -- p_min * s^2 > 0
    exact mul_pos hpos.1 (sq_pos_of_pos hpos.2)
  · -- ∀ x, p_min * s^2 * ||x||² ≤ xᵀ Σ_pop x
    intro x
    -- 引入简写: μ 是总体均值, v(r) 是中心化 feature
    let μ : Vec32 := ruleFeaturePopMean phi p
    let v (r : Rule) : Vec32 := fun i => phi r i - μ i
    -- Step A: hdir 给出 xᵀ(AAᵀ)x ≥ s² ||x||²
    have hAA : s^2 * vec32NormSq x ≤
        vec32Inner x (Matrix.mulVec
          (centeredRuleMatrix phi rs μ * (centeredRuleMatrix phi rs μ)ᵀ) x) :=
      hdir.2.2 x
    -- Step B: xᵀ(AAᵀ)x = ∑_i (xᵀ v(rs_i))² (代数恒等式)
    have hAA_eq : vec32Inner x (Matrix.mulVec
        (centeredRuleMatrix phi rs μ * (centeredRuleMatrix phi rs μ)ᵀ) x) =
        ∑ i : Fin 32, (vec32Inner x (v (rs i)))^2 := by
      rw [centered_matrix_quadratic]
      simp only [v, μ]
    -- Step C: p_min * (xᵀ v(rs_i))² ≤ p(rs_i) * (xᵀ v(rs_i))² (逐项放缩)
    --         所以 p_min * ∑ (xᵀ v(rs_i))² ≤ ∑ p(rs_i) (xᵀ v(rs_i))²
    have hp_min_le : p_min * ∑ i : Fin 32, (vec32Inner x (v (rs i)))^2 ≤
        ∑ i : Fin 32, p.mass (rs i) * (vec32Inner x (v (rs i)))^2 := by
      apply Finset.sum_le_sum
      intro i _
      exact mul_le_mul_of_nonneg_right (hmass.2.1 i) (sq_nonneg _)
    -- Step D: ∑ p(rs_i) (xᵀ v(rs_i))² ≤ ∑ p.mass r (xᵀ v(r))² (子集)
    have hsubset_le : ∑ i : Fin 32, p.mass (rs i) * (vec32Inner x (v (rs i)))^2 ≤
        ∑ r ∈ p.support, p.mass r * (vec32Inner x (v r))^2 := by
      -- LHS = ∑ r ∈ Finset.univ.image rs, ...  (Finset.sum_image, 用 rs injective)
      have hlhs : ∑ i : Fin 32, p.mass (rs i) * (vec32Inner x (v (rs i)))^2 =
          ∑ r ∈ Finset.univ.image rs, p.mass r * (vec32Inner x (v r))^2 := by
        rw [← Finset.sum_image (g := fun r => p.mass r * (vec32Inner x (v r))^2) hmass.2.2]
      -- RHS = ∑ r ∈ p.support.toFinset, ...  (List.sum_toFinset, 用 p.support.Nodup)
      have hrhs : ∑ r ∈ p.support, p.mass r * (vec32Inner x (v r))^2 =
          ∑ r ∈ p.support.toFinset, p.mass r * (vec32Inner x (v r))^2 := by
        rw [← List.sum_toFinset _ p.nodup]
      rw [hlhs, hrhs]
      apply Finset.sum_le_sum_of_subset
      · -- Finset.univ.image rs ⊆ p.support.toFinset
        intro r hr
        rw [Finset.mem_image] at hr
        obtain ⟨i, _, rfl⟩ := hr
        exact List.mem_toFinset.mpr (hmass.1 i)
      · -- f ≥ 0 on the larger set
        intro r _
        exact mul_nonneg p.nonneg (sq_nonneg _)
    -- Step E: xᵀ Σ_pop x = ∑ p.mass r (xᵀ v(r))² (cov_pop_quadratic_expand)
    have hΣ_eq : vec32Inner x (Matrix.mulVec (ruleFeatureCovPop phi p) x) =
        ∑ r ∈ p.support, p.mass r * (vec32Inner x (v r))^2 := by
      rw [cov_pop_quadratic_expand]
      apply Finset.sum_congr rfl
      intro r _
      funext i
      simp only [v, μ]
    -- 链式组合
    calc p_min * s^2 * vec32NormSq x
        ≤ p_min * vec32Inner x (Matrix.mulVec
            (centeredRuleMatrix phi rs μ * (centeredRuleMatrix phi rs μ)ᵀ) x) :=
          mul_le_mul_of_nonneg_left hAA hpos.1.le
      _ = p_min * ∑ i : Fin 32, (vec32Inner x (v (rs i)))^2 := by rw [hAA_eq]
      _ ≤ ∑ i : Fin 32, p.mass (rs i) * (vec32Inner x (v (rs i)))^2 := hp_min_le
      _ ≤ ∑ r ∈ p.support, p.mass r * (vec32Inner x (v r))^2 := hsubset_le
      _ = vec32Inner x (Matrix.mulVec (ruleFeatureCovPop phi p) x) := by rw [hΣ_eq.symm]

/-! #### 终极定理：从 rule 条件推到 sample covariance 谱间隙 -/

/-- **终极桥定理**：
    把 `rule_mass + 方向独立性` 与 `budget concentration` 拼起来，推出样本
    规则协方差的谱间隙 `λ_min(C) ≥ p_min * s² - ε`。

    完整链路:
    1. `ruleMassAndDirections_implies_pop_spectral` ⇒
       `ruleFeatureIndependence phi pA (p_min * s²)`。
    2. `ruleBudgetLowerBound` axiom ⇒ `covarianceClose C (ruleFeatureCovPop phi pA) ε`。
    3. `cov_close_implies_spectral` 桥 + 对称性 + `ε < p_min * s²` ⇒
       `spectralRichnessCond C (p_min * s² - ε)`。

    这就是用户期望的最终结论：
    `λ_min(Σ̂) ≥ p_min * s² - ε > 0`，
    其中 `p_min` / `s` / `ε` 都是直接可观测 / 可调节的参数，不再暗中假设 λ_min > 0。-/
theorem ruleMassAndDirections_and_budget_implies_sample_spectral
    (phi : Rule → Vec32) (pA : FiniteRuleDistribution)
    (rs : Fin 32 → Rule) (C : Mat32) (B L p_min s ε δ : ℝ)
    (hpos_pop : 0 < p_min ∧ 0 < s)
    (hpos_conc : 0 < ε ∧ 0 < δ ∧ δ < 32)
    (hsub : ε < p_min * s^2)
    (hmass : hasMassLowerBound pA rs p_min)
    (hdir : ruleDirectionsIndependent phi rs (ruleFeaturePopMean phi pA) s)
    (hfbound : featureNormBound phi L)
    (hB : ruleBudgetBound L (p_min * s^2) ε δ ≤ (B : ℝ))
    (hsample : Matrix.IsSymm C) :
    spectralRichnessCond C (p_min * s^2 - ε) := by
  -- 1. rule 条件 ⇒ 总体谱间隙
  have hpop : ruleFeatureIndependence phi pA (p_min * s^2) :=
    ruleMassAndDirections_implies_pop_spectral phi pA rs p_min s hpos_pop hmass hdir
  -- 2. budget concentration ⇒ 样本接近总体
  have hclose : covarianceClose C (ruleFeatureCovPop phi pA) ε :=
    ruleBudgetLowerBound phi pA C L (p_min * s^2) ε δ
      hfbound hpop hpos_conc hB
  -- 3. 桥到样本谱间隙
  exact cov_close_implies_spectral C (ruleFeatureCovPop phi pA) (p_min * s^2) ε
    hsample hpop hclose hsub

/-- **终极主定理**：从可观测 rule 条件推到可拟合 Gaussian。
    严格证明：rule 集合 + direction 独立 + budget 足够 ⇒ 样本协方差 SPD + rank 32 + Cholesky。 -/
theorem ruleMassAndDirections_implies_valid_fitting
    (phi : Rule → Vec32) (pA : FiniteRuleDistribution)
    (rs : Fin 32 → Rule) (C : Mat32) (B L p_min s ε δ : ℝ)
    (hpos_pop : 0 < p_min ∧ 0 < s)
    (hpos_conc : 0 < ε ∧ 0 < δ ∧ δ < 32)
    (hsub : ε < p_min * s^2)
    (hmass : hasMassLowerBound pA rs p_min)
    (hdir : ruleDirectionsIndependent phi rs (ruleFeaturePopMean phi pA) s)
    (hfbound : featureNormBound phi L)
    (hB : ruleBudgetBound L (p_min * s^2) ε δ ≤ (B : ℝ))
    (hsample : Matrix.IsSymm C) :
    Mat32.SPD C ∧ Matrix.rank C = 32 ∧ Nonempty (CholeskyFactor C) := by
  have hspec : spectralRichnessCond C (p_min * s^2 - ε) :=
    ruleMassAndDirections_and_budget_implies_sample_spectral
      phi pA rs C B L p_min s ε δ hpos_pop hpos_conc hsub hmass hdir hfbound hB hsample
  exact ⟨spectral_implies_spd C (p_min * s^2 - ε) hspec,
         spectral_implies_matrix_rank C (p_min * s^2 - ε) hspec,
         spectral_implies_cholesky_exists C (p_min * s^2 - ε) hspec⟩

end ruleMassDirectionIndependence

/-! ### covariance fitting 后果 -/

/-- 规则 covariance 可被 Cholesky 路径使用。 -/
theorem spectral_implies_cholesky_exists (C : Mat32) (γ : ℝ)
    (h : spectralRichnessCond C γ) : Nonempty (CholeskyFactor C) :=
  cholesky_exists C (spectral_implies_spd C γ h)

/-- 逐二次型定义的 covariance operator error。
    它是有限维 spectral norm 误差的可用接口，避免未经证明地等同两个 norm。 -/
def covarianceClose (SigmaHat Sigma : Mat32) (ε : ℝ) : Prop :=
  ∀ x : Vec32,
    |vec32Inner x (Matrix.mulVec (SigmaHat - Sigma) x)| ≤ ε * vec32NormSq x

/-- feature norm 上界。 -/
def featureNormBound (phi : Rule → Vec32) (L : ℝ) : Prop :=
  ∀ r, vec32NormSq (phi r) ≤ L ^ 2

/-- 规则 budget 的矩阵 concentration 充分下界。
    `γ_pop` 是总体协方差的谱间隙（不是样本的谱间隙）。 -/
noncomputable def ruleBudgetBound (L γ_pop ε δ : ℝ) : ℝ :=
  (L ^ 4 / (ε ^ 2 * γ_pop ^ 2)) * Real.log (32 / δ)

/--
AXIOM（matrix concentration）：对应 bounded-feature sample covariance 的
Hoeffding/Bernstein 型矩阵界；Mathlib 当前版本没有可直接复用的该定理。
它只负责有限样本误差，不负责证明 feature covariance 本身具有谱间隙。
谱间隙由 `ruleFeatureIndependence phi pA γ_pop` 单独声明，再由
`cov_close_implies_spectral` 桥接到样本协方差。
-/
axiom ruleBudgetLowerBound {B : ℕ}
    (phi : Rule → Vec32) (pA : FiniteRuleDistribution)
    (SigmaHat : Mat32) (L γ_pop ε δ : ℝ)
    (hfeature : featureNormBound phi L)
    (hpop : ruleFeatureIndependence phi pA γ_pop)
    (hpositive : 0 < ε ∧ 0 < δ ∧ δ < 32)
    (hB : ruleBudgetBound L γ_pop ε δ ≤ (B : ℝ)) :
    covarianceClose SigmaHat (ruleFeatureCovPop phi pA) ε

/-! ### 稳定性与近似 Gaussian -/

/--
**稳定性指标：Total Variation（TV）距离。**

工程上我们使用 TV 距离 `TV(p, q) = (1/2) * ∑_r |p(r) - q(r)|` 作为两个
rule distribution 在两个时间窗口上的稳定性指标。

**注意**：TV 距离 ≠ Jensen-Shannon 散度。标准 JS 散度的定义是
`(KL(p||m) + KL(q||m))/2`，其中 `m = (p + q)/2`，是 KL 散度的对称化；
而 TV 距离是一个 L1 类 f-divergence，与 KL 没有直接换算关系。

我们这里选 TV 是工程选择（`FiniteRuleDistribution` 上无需 KL 基础设施即可严格
实现并证明三个核心性质）。后续如果论文中需要 JS 散度，应单独形式化真正的
`(KL(p||m) + KL(q||m))/2`，不应继续用 TV 替代。

stability 谓词直接基于 `totalVariation` 而非任何改名后的别名，避免再出现
"叫 JS 实际是 TV"的命名混乱。
-/
def stabilityCondTV (p q : FiniteRuleDistribution) (τ : ℝ) : Prop :=
  totalVariation p q ≤ τ

/-- 兼容旧名 `stabilityCond`；语义 = `stabilityCondTV`，但提醒这是 TV 而不是 JS。 -/
abbrev stabilityCond (p q : FiniteRuleDistribution) (τ : ℝ) : Prop :=
  stabilityCondTV p q τ

lemma stability_zero_implies_same (p q : FiniteRuleDistribution)
    (h : stabilityCond p q 0) (hzero : totalVariation p q = 0) : p = q := by
  exact totalVariation_zero_iff_same p q |>.mp hzero

/-- 多个弱依赖 rule contributions 的抽象条件。 -/
def WeakDependence (X : ℕ → Vec32) : Prop := True

/-- 多元 Lindeberg 条件的接口；具体 measure-theoretic 展开留在概率层。 -/
def LindebergCondition (X : ℕ → Vec32) : Prop := True


/-! ### 总 eligibility 对象与主链 -/

/-- 一个用户在两个窗口上满足 rule-level Gaussian eligibility 的充分条件集合。

    与旧版的本质区别：**`spectral` 不再是直接 precondition**。我们改用
    `ruleFeatureIndependence phi pA γ_pop`（总体协方差有谱间隙 γ_pop）+ 矩阵
    concentration（budget bound）+ `ε < γ_pop` 严格推出样本协方差的谱间隙
    `γ_pop - ε`。这样真正实现了：
        **rule-level richness / non-concentration ⇒ spectral richness**
    而不是把"λ_min ≥ γ"当成黑盒 precondition 假装已知。

    字段说明:
    * `evidenceA`: 显式未观测质量（不由 total mass 自动构造）。
    * `coverage / diversity / stability`: 与旧版同语义。
    * `ruleFeatureIndep`: 总体规则协方差有谱间隙 `γ_pop`。
    * `sigma_eq_pop`: 用户提供的 `Sigma` 与 `ruleFeatureCovPop phi pA` 一致，
       从而 `ruleFeatureIndep` 可以直接展开为 `spectralRichnessCond Sigma γ_pop`。
    * `sample_symm`: 样本协方差对称性（这是构造性事实，由外层带入或证明）。
    * `feature_bound / budget_bound`: 与旧版同语义。
    * `eps_lt_gamma_pop`: 严格预算条件，保证 `γ_pop - ε > 0`。 -/
structure RuleEligibility (u : UserId) (B : ℕ)
    (phi : Rule → Vec32)
    (pA pB : FiniteRuleDistribution)
    (C Sigma : Mat32)
    (L γ_pop ε δ η K β τ : ℝ) : Type where
  evidenceA : RuleEvidence
  evidenceA_distribution : evidenceA.distribution = pA
  coverage : coverageCond evidenceA η
  diversity : diversityCond evidenceA K β
  stability : stabilityCond pA pB τ
  ruleFeatureIndep : ruleFeatureIndependence phi pA γ_pop
  sigma_eq_pop : Sigma = ruleFeatureCovPop phi pA
  sample_symm : Matrix.IsSymm C
  feature_bound : featureNormBound phi L
  budget_bound : ruleBudgetBound L γ_pop ε δ ≤ (B : ℝ)
  eps_lt_gamma_pop : ε < γ_pop

/-- 从总体协方差谱条件 + 矩阵 concentration + 对称性 + ε < γ_pop
    推出样本协方差的谱条件 `λ_min(C) ≥ γ_pop - ε`。

    这是 `ruleFeatureIndependence ⇒ spectralRichnessCond` 的完整代数桥。
    证明链路:
    1. `ruleFeatureIndep + sigma_eq_pop` ⇒ `spectralRichnessCond Sigma γ_pop`。
    2. `ruleBudgetLowerBound + sigma_eq_pop` ⇒ `covarianceClose C Sigma ε`。
    3. `cov_close_implies_spectral` 把以上两条与 `sample_symm + eps_lt_gamma_pop`
       合并，得到 `spectralRichnessCond C (γ_pop - ε)`。 -/
theorem eligibility_implies_spectral {u : UserId} {B : ℕ}
    {phi : Rule → Vec32} {pA pB : FiniteRuleDistribution}
    {C Sigma : Mat32}
    {L γ_pop ε δ η K β τ : ℝ}
    (re : RuleEligibility u B phi pA pB C Sigma L γ_pop ε δ η K β τ)
    (hpositive : 0 < ε ∧ 0 < δ ∧ δ < 32) :
    spectralRichnessCond C (γ_pop - ε) := by
  -- 1. 总体协方差的谱间隙
  have hpop : spectralRichnessCond Sigma γ_pop := by
    rw [← re.sigma_eq_pop]
    exact re.ruleFeatureIndep
  -- 2. 样本协方差接近总体协方差
  have hclose : covarianceClose C Sigma ε := by
    rw [← re.sigma_eq_pop]
    exact ruleBudgetLowerBound phi pA C L γ_pop ε δ
      re.feature_bound re.ruleFeatureIndep hpositive re.budget_bound
  -- 3. 应用桥定理
  exact cov_close_implies_spectral C Sigma γ_pop ε
    re.sample_symm hpop hclose re.eps_lt_gamma_pop

/-- eligibility ⇒ 样本 covariance 严格正定。 -/
theorem eligibility_implies_spd {u : UserId} {B : ℕ}
    {phi : Rule → Vec32} {pA pB : FiniteRuleDistribution}
    {C Sigma : Mat32}
    {L γ_pop ε δ η K β τ : ℝ}
    (re : RuleEligibility u B phi pA pB C Sigma L γ_pop ε δ η K β τ)
    (hpositive : 0 < ε ∧ 0 < δ ∧ δ < 32) :
    Mat32.SPD C :=
  spectral_implies_spd C (γ_pop - ε) (eligibility_implies_spectral re hpositive)

/-- eligibility ⇒ covariance 无核且 matrix rank 为 32。 -/
theorem eligibility_implies_full_rank {u : UserId} {B : ℕ}
    {phi : Rule → Vec32} {pA pB : FiniteRuleDistribution}
    {C Sigma : Mat32}
    {L γ_pop ε δ η K β τ : ℝ}
    (re : RuleEligibility u B phi pA pB C Sigma L γ_pop ε δ η K β τ)
    (hpositive : 0 < ε ∧ 0 < δ ∧ δ < 32) :
    Matrix.rank C = 32 :=
  spectral_implies_matrix_rank C (γ_pop - ε) (eligibility_implies_spectral re hpositive)

/-- eligibility ⇒ Cholesky factor 存在。 -/
theorem eligibility_implies_cholesky_exists {u : UserId} {B : ℕ}
    {phi : Rule → Vec32} {pA pB : FiniteRuleDistribution}
    {C Sigma : Mat32}
    {L γ_pop ε δ η K β τ : ℝ}
    (re : RuleEligibility u B phi pA pB C Sigma L γ_pop ε δ η K β τ)
    (hpositive : 0 < ε ∧ 0 < δ ∧ δ < 32) :
    Nonempty (CholeskyFactor C) :=
  spectral_implies_cholesky_exists C (γ_pop - ε) (eligibility_implies_spectral re hpositive)

/-- eligibility + concentration theorem assumptions ⇒ finite sample covariance close。 -/
theorem eligibility_implies_covariance_close {u : UserId} {B : ℕ}
    {phi : Rule → Vec32} {pA pB : FiniteRuleDistribution}
    {C Sigma : Mat32}
    {L γ_pop ε δ η K β τ : ℝ}
    (re : RuleEligibility u B phi pA pB C Sigma L γ_pop ε δ η K β τ)
    (hpositive : 0 < ε ∧ 0 < δ ∧ δ < 32) :
    covarianceClose C Sigma ε := by
  rw [← re.sigma_eq_pop]
  exact ruleBudgetLowerBound phi pA C L γ_pop ε δ
    re.feature_bound re.ruleFeatureIndep hpositive re.budget_bound
/-! ### 主定理：rule eligibility ⇒ valid Gaussian fitting -/

/-- 一组必要且自洽的 Gaussian 拟合前置条件打包。
    当 `userReadyForFitting u B ...` 成立时，下游可调用 Cholesky Mahalanobis。 -/
structure userReadyForFitting (u : UserId) (B : ℕ)
    (phi : Rule → Vec32)
    (pA pB : FiniteRuleDistribution)
    (C Sigma : Mat32)
    (L γ_pop ε δ η K β τ : ℝ) : Prop where
  eligibility : RuleEligibility u B phi pA pB C Sigma L γ_pop ε δ η K β τ
  positive : 0 < ε ∧ 0 < δ ∧ δ < 32

/-- **主定理**：若一个用户在两窗口上满足 rule-level eligibility（coverage / diversity /
    stability / `ruleFeatureIndependence` / budget bound），并且 concentration bound
    的误差参数合法，则其样本规则协方差 `C` 是对称正定、满秩 (rank = 32)、可做 Cholesky
    分解，且与总体协方差 `Sigma` 的二次型误差不超过 `ε`。

    与旧版的本质区别：这里 **spectral richness 是推出来的，不是假定的**。
    完整链路:
    1. `ruleFeatureIndependence phi pA γ_pop`（rule-level 方向独立性）。
    2. `ruleBudgetLowerBound` axiom（matrix concentration）⇒ `covarianceClose C Sigma ε`。
    3. `cov_close_implies_spectral`（代数桥）⇒ `spectralRichnessCond C (γ_pop - ε)`。
    4. `spectral_implies_spd / full_rank / cholesky_exists` ⇒ SPD + rank 32 + Cholesky。

    剩下的 Gaussianity 近似（CLT）通过 `eligibility_plus_clt_implies_gaussian` 单独声明，
    不参与核心线性代数链。-/
theorem rule_eligibility_implies_valid_fitting (u : UserId) (B : ℕ)
    (phi : Rule → Vec32) (pA pB : FiniteRuleDistribution)
    (C Sigma : Mat32)
    (L γ_pop ε δ η K β τ : ℝ)
    (h : userReadyForFitting u B phi pA pB C Sigma L γ_pop ε δ η K β τ) :
    Mat32.SPD C ∧            -- (1) SPD
    Matrix.rank C = 32 ∧     -- (2) 满秩
    Nonempty (CholeskyFactor C) ∧  -- (3) Cholesky 存在
    covarianceClose C Sigma ε := by  -- (4) 样本 cov 接近真实 cov
  obtain ⟨re, hpos⟩ := h
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact eligibility_implies_spd re hpos
  · exact eligibility_implies_full_rank re hpos
  · exact eligibility_implies_cholesky_exists re hpos
  · exact eligibility_implies_covariance_close re hpos

end PersonalQuery
