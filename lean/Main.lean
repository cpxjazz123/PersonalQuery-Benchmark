import PersonalQuery.Basic
import PersonalQuery.PCFG
import PersonalQuery.Encoder
import PersonalQuery.Gaussian
import PersonalQuery.Mahalanobis
import PersonalQuery.Quantile
import PersonalQuery.Selection
import PersonalQuery.Audit
import PersonalQuery.Pipeline
import PersonalQuery.RuleEligibility
import PersonalQuery.EncoderSpectralBridge
import PersonalQuery.GaussianApproximation
import PersonalQuery.SelectionStability
import PersonalQuery.MeanCovarianceStability
import PersonalQuery.Test

/-!
# PersonalQuery 形式化入口

Per-user Gaussian 拟合 + Mahalanobis D² + Q95 互斥门控 + PCFG 句法规则的
Lean 4 形式化验证。 模块结构见 `PersonalQuery/` 子目录:

  Basic.lean      32 维向量 / 矩阵 / 对称化 / 内积 / 范数基础
  PCFG.lean       POS/DepLabel 枚举 + D4/D3/P3 句法规则归纳定义
  Encoder.lean    Bag-of-Rules + normalize + SupEncoder + BCE + label smoothing
  Gaussian.lean   μ/Σ 估计 + 对称化 + 半正定 + EigenGate + Cholesky
  Mahalanobis.lean 直接路径 vs Cholesky 路径等价 + 基本性质
  Quantile.lean   经验分位数 + q50/q75/q95/max + gate_T 单调性
  Selection.lean  target_inside ⊕ competitor_outside 互斥门控 + 最小 D² 选择
  Audit.lean      Stage 05 E1-E4 有效性验证 + ASIN coverage ≥ 2
  Pipeline.lean   端到端数据流类型签名 + 主定理 + 高层不变量
  RuleEligibility.lean    pre-encoder 四类条件 (coverage/diversity/stability/spectral)
  EncoderSpectralBridge.lean  encoder 非塌缩 + raw→latent 谱桥 + 误差界
  GaussianApproximation.lean  Phase N: 多元 CLT 接口 (Bentkus 1/√n) + 最终主定理
  SelectionStability.lean    Phase O: query-selection ranking stability 定理 (Phase M → Stage 08 selectByMinD2)
  MeanCovarianceStability.lean  Phase Q+R+S: joint mean+covariance Mahalanobis stability (Phase M → Phase O/P 升级版) + Cholesky 严格化 Mahalanobis CS + 端到端定理
  (Phase S 收束: 删除 `vec32_inner_cs_sq` axiom (用二次判别式 lemma 化), 删除 `psd_cholesky_representation` axiom (改用项目内定理 `cholesky_exists_of_spd` + `spectral_implies_spd_inv` 桥接 lemma); 主链 axioms 从 3 压到 1)
  总主链 axiom 计数: 1 (Phase N.1: `multivariate_clt_interface`, 概率论 contract; 其余 Phase Q/R 内部 axiom 已全部 lemma 化)
  sorry 计数: 0

调用:

```sh
cd /home/wlia0047/ar57/wenyu/PersoanlQuery/lean
lake build
```

依赖 Mathlib v4.18.0, 首次构建需要 30-60 分钟编译 Mathlib OLeans.
本仓库无 elan toolchain, 完整编译需先:

```sh
curl -sSf https://raw.githubusercontent.com/leanprover-community/mathlib4/master/scripts/install_elan.sh | sh
```

注意: 部分证明使用 `sorry` 占位 (标记后续需补充的细节), 完整证明策略
已写入每个 theorem 的 docstring, 由后续迭代填充。
-/