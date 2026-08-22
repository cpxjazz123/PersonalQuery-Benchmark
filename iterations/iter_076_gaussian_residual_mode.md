# Iter 076: Gaussian VADES `diagonal_residual` 模式 (318d 中性 residual 拟合)

## Context

之前 Gaussian VADES 用 `StandardScaler.fit_transform(raw_features)` 直接训 per-user Gaussian。
但 `StandardScaler` 之后 scaled_features 已经是 mean=0,所以"global mean residual"在 scaled 空间无意义(≈ 不做)。

Phase 13/14 `e23_style_vector.py` + `phase14_q_*.py` 已经在 Qwen hidden 空间用过 residual = user - neutral 路线,Phase 14.F SOTA 用 global neutral 重排。

**目标**: 把"中性 residual"思想搬到 **318d 句法特征空间**(在 raw 空间算 residual),新增 `diagonal_residual` mode。

## 改动 (gaussian/gaussian_vades.py, +50/-1)

### 1. 新 mode (line 132)
```python
"diagonal_residual",  # 318d raw space global neutral residual
```

### 2. 新 env vars (line 134-138)
```python
RESIDUAL_NEUTRAL_MODE = os.environ.get("VADES_RESIDUAL_NEUTRAL", "global_mean")
RESIDUAL_NEUTRAL_CACHE = os.environ.get(
    "VADES_RESIDUAL_NEUTRAL_CACHE",
    "/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/global_neutral.npy",
)
```

### 3. 新 helper 函数 (line 1189-1203)
```python
def compute_global_neutral_ref(feature_matrix: np.ndarray) -> np.ndarray:
    """raw 空间 per-dim mean (因 StandardScaler 后 mean=0 残差无意义)。"""
    return feature_matrix.mean(axis=0).astype(np.float64)

def apply_residual(features, neutral): ...
def restore_from_residual(residuals, neutral): ...
```

### 4. main_train() 分支 (line 3455-3477)
```python
if COVARIANCE_MODE == "diagonal_residual":
    log("[diagonal_residual] 计算 global neutral reference...")
    feature_matrix_raw = dataset["feature_matrix"]
    cache_path = Path(RESIDUAL_NEUTRAL_CACHE)
    if cache_path.exists():
        global_neutral = np.load(cache_path).astype(np.float64)
    else:
        global_neutral = compute_global_neutral_ref(feature_matrix_raw)
        np.save(cache_path, global_neutral)
    # raw 空间 residual, 然后 StandardScaler
    residual = apply_residual(feature_matrix_raw, global_neutral)
    residual_scaler = StandardScaler()
    scaled_residual = residual_scaler.fit_transform(residual).astype(np.float64)
    # 替换 scaled_features
    dataset["scaled_features"] = scaled_residual
    dataset["scaler"] = residual_scaler
    dataset["global_neutral"] = global_neutral
    dataset["residual_mode"] = True
```

## 数据流对比

### 默认 (diagonal_gmm 等)
```
raw features (num_sentences, 318d)
 ↓ StandardScaler.fit_transform
scaled_features (mean=0, std=1)
 ↓ encoder
(mu, logvar, recon) ← recon_target = raw features
```

### 新增 diagonal_residual
```
raw features (num_sentences, 318d)
 ↓ - global_neutral (per-dim mean of raw)
residual (raw - global_neutral)
 ↓ StandardScaler.fit_transform  
scaled_residual (mean=0, std=1)
 ↓ encoder
(mu, logvar, recon) ← recon_target = raw features (same as default)
```

**注意**: Decoder reconstruction target 仍为 `feature_matrix_raw`(沿用默认行为)。Encoder 输入从 scaled_features 换成 scaled_residual,让 VAE 学"风格偏差分布"而非"风格绝对分布"。

## 关键设计权衡

| 决策 | 理由 |
|------|------|
| **raw 空间减 neutral** | StandardScaler 后 mean=0,residual 无意义 |
| **缓存 global_neutral** | 首次计算后复用,避免不同 run 漂移 |
| **保留 feature_matrix_raw** | v4 prototype teacher 仍需 raw;decoder target 也用 raw |
| **新 dataset["residual_mode"] flag** | 下游消费者(若有)可识别模式 |
| **不与 prototype 模式叠加** | 减少 scope,先验证基本假设 |
| **不复用新 UserDistributionTable** | 沿用 `UserDistributionTable`(单 Gaussian),与 `diagonal` 一致 |

## 与其他模式的关系

| Mode | 输入空间 | User 表 | 适合场景 |
|------|---------|---------|---------|
| `diagonal` | scaled raw 318d | 单 Gaussian | baseline |
| `diagonal_gmm` | scaled raw 318d | GMM | 多风格 |
| `diagonal_prototype` | scaled raw 318d | 单 Gaussian + cluster anchor | 跨用户对比 |
| **`diagonal_residual`(新)** | **scaled residual** | **单 Gaussian** | **个人风格偏差**(理论上更 discriminative) |

## 验证 (待执行)

1. **py_compile**: ✅ 通过
2. **smoke test**: 用现有 data 跑新 mode,验证:
   - log 出现 "[diagonal_residual] 计算 global neutral reference..."
   - encoder / user_table 正常 training loop
   - summary json 写出
3. **对比 baseline (diagonal_gmm) vs diagonal_residual**:
   - user_holdout_mse(residual 应该更低)
   - per-user Gaussian cluster silhouette(residual 应该更紧)
   - 接入 query_gen pipeline,看 rank-1 提升

## 已知风险

| 风险 | 状态 |
|------|------|
| 318d 句法 raw space 中性化未必有意义 | Smoke test 跑 norm 对比 |
| Decoder target 仍是 raw,scale mismatch 存在 | 沿用默认行为,decoder 自适应 |
| 现有下游消费者期待 scaled_features = raw / scaled | `infer_user_sentence_distributions` 不需改(读 scaled_features) |
| 与 prototype 模式叠加未实现 | 后续可加 `diagonal_residual_prototype` |

## 后续可扩展 (本次不做)

1. `diagonal_residual_llm`: LLM neutral rewrite(像 e23 StyleVector)
2. `diagonal_residual_cluster`: per-cluster mean(KMeans)
3. `diagonal_residual_prototype`: residual + cluster prototype + InfoNCE

## 文件清单

| Path | 用途 |
|------|------|
| `gaussian/gaussian_vades.py` | 主修改文件 (+50/-1 行) |
| `/home/wlia0047/hj82_scratch2/wenyu/gaussian_vades/global_neutral.npy` | 首次运行时缓存 |