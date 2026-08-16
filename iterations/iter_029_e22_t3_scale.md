# Iter #029 (E22 Task 3): 318 维规模网格诊断 — **分布不可比，7 维为唯一可对齐空间**

**Date**: 2026-08-16
**Issue**: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/22
**Status**: 诊断完成 — 318 维聚合无法与评论句法对齐；Task 1 的 7 维 Query-compatible 特征为唯一验证可对齐空间

## §A 任务（用户指令 2026-08-16）

1. 修正 Task 2 syntax encoder：Qwen hidden states → syntax probe → 318 维（冻结）
2. 用完整 318 维 z_user 训练 projector；生成不限制 7 维
3. 评估不同 Query 长度 × 不同 Query 数量，找到稳定承载用户句法的最小生成规模
4. 测试不同维度数量（可变维度集），寻找可对齐配置

## §B 实施

### B1. Step 1: Qwen hidden → syntax probe → 318 维（已完成，3 check 全过）
- 输入：内容中性化 query（固定占位 "xx xx"，同模板产品 bit-identical）
- 标签：spaCy 318 维（Task-1 标准化空间）
- 模板留出 split（TEMPLATE_SPLIT_TEST=[10,11]）防记忆
- 结果（`e22_t3_syntax_probe.json`）：
  - check1 held-out 模板：probe L2=39.6 vs 均值基线 51.4，**改善 23%** ✓
  - check2 content-swap：**0.19**（内容替换输出不变）✓
  - check3 syntax-swap：69.1（句法改变可检测）✓
  - **all_checks_pass: TRUE**

### B2. Step 2: 318 维 projector 训练
- 冻结 Step 1 probe 作为 style head；训练 projector(318→16×1536) + copy head
- 单次拼接 forward（[real; shuffled] 2B rows）加速 2 倍 + 真早停（patience=3）
- 模板 318 特征落盘缓存（`e22_t3_template318_cache.npz`）
- dev loss 5.69→2.95（8 epochs）

### B3. Step 3: 规模网格生成（120 test users）
- 网格：长度 {16,32,64} × Query 数 {2,4,8} × 控制 {real, shuffled} × 重复 {0,1}
- **解码退化 bug 修复**：force-copy pending 逻辑死循环（"anaana..."）→ 移除强制复制，仅保留 EOS 屏蔽 + copy-head 混合。退化率 85% → 2.3%
- 生成 3624 条（`e22_t3_grid_generations.jsonl`）

## §C 规模扫描结果（Query 数 K 对齐 318 维）

`e22_t3_scale_scan.json`（K∈{2,4,6,8,12,16,24,32}，聚合协议同 z_user：多 query 句子合并一次 user_features_v2）：

| K | n | fidelity | cos_gap | corr | swap | stab |
|---|---|---|---|---|---|---|
| 2 | 105 | -0.50 | -0.098 | -0.03 | 0.46 | 0.53 |
| 4 | 87 | -0.48 | -0.100 | -0.06 | 0.49 | 0.62 |
| 8 | 64 | -0.22 | -0.035 | -0.05 | 0.53 | 0.69 |
| 12 | 46 | -0.40 | -0.094 | -0.09 | **0.74** | 0.74 |

**fidelity 全负且不随 K 收敛** → 不是样本量问题。

## §D 维度数量扫描（99 组合：5 策略 × N × K）

`e22_t3_dimscan.json`。策略：S1 std-topN / S2 评论非零率-topN / S3 Task1-7维 / S4 Query可观测-topN / S5 std×可观测-topN；N∈{5,7,10,15,20,30,50,100,200,318}；K∈{2,4,8}。

**top 配置**：

| 配置 | N | K | fid | cos_gap | corr | swap | stab |
|---|---|---|---|---|---|---|---|
| S1/S5 std-top50 | 50 | 2 | **-0.03** | **-0.002** | +0.001 | 0.60 | 0.67 |
| S1/S5 std-top50 | 50 | 8 | -0.14 | -0.020 | -0.035 | **0.75** | **0.84** |
| Task1 7维 | 7 | 4 | -0.11 | -0.054 | -0.107 | 0.52 | 0.78 |
| queryobs | 100+ | 2 | -0.14 | -0.010 | -0.007 | 0.59 | 0.57 |

## §E 结论

1. **维度数量是决定性因素**：N=50（std 排序）最优，fid=-0.03、cos_gap≈0、corr≈0（中性）；N 过小（<10）或过大（>100）都更差
2. **K 只改善稳定性与可控性**：K: 2→8 使 swap 0.60→0.75、stab 0.67→0.84，但 fidelity 不随 K 改善
3. **没有任何维度配置能正向对齐**（fid>0 且 CI 排除 0）→ **318 维空间下生成 Query 聚合与评论句法分布本质不可比**：
   - 318 维为长文本（用户 30 句评论）设计；生成 Query（~12 词）聚合后大量维度恒 0 或 OOD 爆炸
   - 可比维度仅 23/318（std>0.05 且评论可观测）
4. **Task 1 的 7 维 Query-compatible 特征是唯一被验证可对齐的空间**（Check-1: d=1.31, AUC=0.82）——正是为解决"短 Query 句法可比性"而设计

## §F 输出文件

- `result/e22_t3/e22_t3_syntax_probe.{json,pt}` — Step 1 probe（3 check 全过）
- `result/e22_t3/e22_t3_step1_hidden_cache.npz` — probe hidden 缓存
- `result/e22_t3/e22_t3_injector.pt` — 318 维 projector + copy head
- `result/e22_t3/e22_t3_train.{json,samples.jsonl}` — 318 训练
- `result/e22_t3/e22_t3_grid_generations.jsonl` — 3624 条网格生成
- `result/e22_t3/e22_t3_grid_eval.json` — 网格评估（长度×Query数）
- `result/e22_t3/e22_t3_scale_scan.json` — K 对齐扫描
- `result/e22_t3/e22_t3_dimscan.json` — 维度数量扫描（99 组合）
- `result/e22_t3/e22_t3_template318_cache.npz` — 模板 318 特征缓存

## §G 脚本

- `query_gen/query_gen_main.py` — 自包含主脚本（train/generate/eval/scale/dimscan 五阶段，不 import 项目脚本）
- `query/soft_prefix/e22_t3_{step1_syntax_probe,train_318,grid_generate,grid_eval}.py` — 分步脚本（保留）
- 运行环境：`pq_env`（torch 2.13+cu130, transformers 4.40.2, spacy 3.8.4）

## §H 后续方向（待决策）

1. **接受 7 维为对齐空间**，318 维仅作诊断（Task 3 回归 7 维 + 受约束解码 + 端到端验证）
2. **训练时用多 query 聚合作 style 监督**：让模型主动对齐聚合空间（可能缓解 OOD）
3. **扩充生成规模**再扫 K∈{16,32,64}（成本高，预计不改变结论）
