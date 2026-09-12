/-
# PersonalQuery.Test — Smoke 验证

最小化测试文件, 仅 import 所有模块以验证项目结构无循环依赖。
完整 smoke 需要 `lake build` 编译 (本仓库无 elan, 无法本地执行)。
-/

import PersonalQuery.Basic
import PersonalQuery.PCFG
import PersonalQuery.Encoder
import PersonalQuery.Gaussian
import PersonalQuery.Mahalanobis
import PersonalQuery.Quantile
import PersonalQuery.Selection
import PersonalQuery.Pipeline

namespace PersonalQuery.Test

/-- 编译期验证: 类型别名能引用, 证明能构造。 -/
#check @Vec32
#check @Mat32
#check @sampleMean
#check @sampleCovRaw
#check @sampleCovSymm
#check @mahaD2
#check @mahaD2Chol
#check @gateT
#check @targetInside
#check @competitorOutside
#check @exclusive
#check @selectByMinD2
#check @UserGauss
#check @Stage04Output
#check @Stage08Output

end PersonalQuery.Test