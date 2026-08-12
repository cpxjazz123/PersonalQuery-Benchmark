# Iteration 179 — BIC/AIC Stub Demo with Synthetic VAE Latent

**日期**: 2026-07-21
**角色**: NLP/IR 专业审稿人
**scope**: §3.4 GMM BIC/AIC framework numerical sanity check without Stage 12 outputs

## §A 审稿意见（尖锐批评）

### 问题: iter #71/86 阻塞在 Stage 12 outputs 缺失 — BIC/AIC framework 数值未验证

**严重程度**: Minor (framework-level), but blocks Major §3.4 GMM-best empirical claim

**iter #71 audit findings**: 
- Stage 12 输出目录 `result/personal_query/12_complexity_analysis_clause_features/<cat>/` 完全空
- `compute_prior_bic_aic.py` 只能 load `None` + 打 WARNING + return (no actual BIC computation)
- iter #41 设计的 BIC/AIC framework 从未在真实 latent 上跑过

**iter #86 audit refinement**:
- paper_claims_audit RQ4 entry 状态: blocked → unverified (after iter #86 文档化 fallback path)
- 真实 Stage 10 → Stage 12 重跑需 ~45-135 min/cat × 3 ≈ 9-21 h GPU infra

**iter #175 param count fix**:
- GMM/Laplace/Logistic 参数计数修正 (`2*K*d+K` for GMM, `2*d` for single-component priors)
- 但 framework 本身在数值上能跑通吗? 没验证过

**iter #179 目标**:
- 不依赖 Stage 12 (9-21 h GPU infra)
- 用 synthetic GMM-sampled latent (100 users × 10 sents = 1000 points) 验证 BIC/AIC framework 端到端
- 验证 4 件事: (a) param count math, (b) BIC formula, (c) log-likelihood calculation, (d) output schema

## §B 缺陷定位

| 问题 | 文件 | 行 | 缺陷 |
|------|------|----|------|
| Stub-only BIC computation | `compute_prior_bic_aic.py` | L142-156 | `if data is None: return` 后只 print param counts, 没有任何 actual BIC value |
| 无 per-user distribution fitting | 同上 | 全文 | 缺 fit_user_distribution(GMM/Laplace/Logistic) |
| 无 Stage 12 stub generator | `10_complexity_analysis/common/` | - | 完全没替代 source for VAE latent |

## §C 本轮代码优化

### C.1 新增 `_stub_synthetic_vae_latent.py` (~125 行)

合成 latent data 写入 `12_complexity_analysis_clause_features/<cat>/synthetic_iter179/{user_profiles,sentences}.jsonl`：
- N users = 100, D = 8 (LATENT_DIM), K = 2 (GMM_COMPONENTS)
- mu ~ N(0, 1), logvar ~ U(-1, 0), mix_logits ~ N(0, 1) — per user
- 10 latent samples per user drawn from user's GMM
- Schema matches train_vades_lite_sentence_latent_threshold.py expected format

### C.2 扩展 `compute_prior_bic_aic.py` (+ ~115 行)

新增 `fit_user_distribution(profile, sentences)`:
- **GMM log-L** (per user): `logsumexp_k(log weight_k + Σ_d log N(x_d | mu_kd, sigma_kd^2))`
- **Laplace log-L**: `Σ_{i,d} [-log(2 b_d) - |x_{id} - mu_d| / b_d]`, b_d = std_d / √2
- **Logistic log-L**: `Σ_{i,d} [-log(s_d) - z - 2 log(1 + exp(-z))]`, s_d = std_d · √3/π
- t-distribution 未拟合（scope 限制，需独立 fit function）

新增 main() numerical output:
```
Prior                k        Σ log L        BIC          AIC
GMM                  34       -9947.08       20129.02     19962.16
Laplace              16       -12186.42      24483.36     24404.84
Logistic             16       -11579.63       23269.79     23191.26
```

新增 GMM advantage 计算 + win/lose 判定:
- BIC penalty diff (GMM - Laplace) = 124.34
- Observed log-L diff = 2239.34 (GMM much better)
- → GMM wins on BIC ✓

修复 broadcast bug (log_phi 计算时算符优先级错误):
```python
# before:
log_phi = -0.5 * np.log(2 * np.pi) - logvar - 0.5 * (diff_sq) / np.exp(logvar)
# (a - b) is [K, D]; (a - b) - [K, n, D] fails

# after:
log_phi = (-0.5 * np.log(2*np.pi) - 0.5 * logvar[:, None, :] - 0.5 * diff_sq / np.exp(logvar)[:, None, :])
```

## §D 验证

### D.1 py_compile
```bash
$ python3 -m py_compile _stub_synthetic_vae_latent.py
OK
$ python3 -m py_compile compute_prior_bic_aic.py
OK
```

### D.2 End-to-end numerical demo
```bash
$ python3 _stub_synthetic_vae_latent.py --category Baby_Products --n-users 100
Wrote 100 profiles + 1000 sentences → 12_complexity_analysis_clause_features/Baby_Products/synthetic_iter179/

$ python3 compute_prior_bic_aic.py --category Baby_Products --latent-dim 8 --n-components 2
================================================================================
BIC/AIC Comparison for Baby_Products
Latent dim: 8, GMM components: 2
================================================================================
Prior                k (params)
--------------------------------
GMM                  34
t-distribution       17
Laplace              16
Logistic             16

Fitted 100 users, 1000 total latent points

Prior                k        Σ log L        BIC          AIC
------------------------------------------------------------------
GMM                  34       -9947.08       20129.02     19962.16
Laplace              16       -12186.42      24483.36     24404.84
Logistic             16       -11579.63      23269.79     23191.26

GMM advantage in BIC penalty: 124.34
Observed log-L difference GMM - Laplace: 2239.34
→ GMM wins on BIC (better likelihood overcomes higher param count)
```

### D.3 数学 sanity check

- **GMM k = 34**: 2*K*d + K = 2*2*8 + 2 = 34 ✓
- **Laplace k = 16**: 2*d = 2*8 = 16 ✓
- **Logistic k = 16**: 2*d = 16 ✓
- **BIC = k·log(n) - 2·log(L)**: 
  - GMM: 34·log(1000) - 2·(-9947.08) = 234.86 + 19894.16 = 20129.02 ✓
  - Laplace: 16·log(1000) - 2·(-12186.42) = 110.52 + 24372.84 = 24483.36 ✓
- **AIC = 2k - 2·log(L)**:
  - GMM: 2·34 - 2·(-9947.08) = 68 + 19894.16 = 19962.16 ✓
  - Laplace: 32 + 24372.84 = 24404.84 ✓
- All numbers internally consistent.

## §E 局限与下一步

### E.1 局限

- **Synthetic data is GMM-generated** → BIC 必然偏 GMM（这是 BIC 的预期行为，不代表 paper claim 已被实证）
- **没有真实 Stage 12 latent** → paper §3.4 GMM-best claim 仍未真正实证
- **t-distribution 未拟合** → 与 GMM/Laplace/Logistic 不完整比较
- **User-marginal 拟合**（当前）vs **joint-marginal 拟合**（paper §3.4 Table 3 描述） — 当前 per-user log-likelihood sum 是 user-conditional，不是 paper Table 3 报告的 marginal GMM
- **Synthetic samples 太少 (1000)** → BIC penalty 优势低, 实证需更多数据才稳

### E.2 下一步

- **iter #180**: 用真实 Stage 12 outputs (待 GPU infra, 9-21 h cost)
- **iter #181**: t-distribution fitting 补完 (类似 fit_user_distribution)
- **iter #182**: paper §5 Limitations #4 文档化「BIC/AIC framework 数值 framework sanity check 通过 (iter #179) but real-data verification pending Stage 12 outputs (≈9-21 h)」

### E.3 对 §11 backlog P0 的影响

iter #41 P0「§3.4 GMM 比较无 BIC/AIC 校正」:
- iter #175 修了 param count formula ✓
- iter #179 验证了 framework 端到端可跑通 (synthetic) ✓
- ❌ 真实 empirical 验证 pending Stage 12 (9-21 h GPU)

loop.md §11 backlog 状态应更新:
```
P0 | [Major] §3.4 GMM BIC/AIC 校正 | iter #41 + #175 (param count) + #179 (framework sanity)
     → framework-level 已通过 (synthetic BIC numbers consistent), 真实 empirical 验证 pending Stage 12 GPU
```

## §F Git Commit

- iter #179: BIC/AIC framework numerical sanity check via synthetic VAE latent (compute_prior_bic_aic.py: add fit_user_distribution for GMM/Laplace/Logistic + numerical output + GMM advantage verdict; new _stub_synthetic_vae_latent.py generator)