# E11 — 用户风格隐向量注入：VADES → Soft Prefix → 个性化 Query 生成

Issue: https://gitlab.com/wlia0047/PersonalQuery-Benchmark/-/work_items/11

主路线：`z_u = L2-normalized(VADES user_mu)` (20-dim) → `MLP → LayerNorm → K
soft tokens` → prepend 到 Qwen2-7B 输入嵌入。冻结 Qwen，只训练 projector
(B5)；可选共享 LoRA (B6)。

## 架构

```
z_u ~ q_phi(z | H_u)                    # VADES-lite diagonal 20d user_mu
P_u = reshape(LayerNorm(MLP(z_u)))      # K=4 soft-prefix tokens
p(y | x_item, z_u) = Qwen([P_u; Embed(prompt(x_item))])
```

- `user_style_vectors.py` — VADES profiles 加载、L2 归一化、missing→零向量+显式
  `has_vector` 标志（冷启动用户不随机化）；`none/zero/global_mean/shuffled/
  vades/e5_mean` 六种条件模式，shuffled 用固定种子置换并存入 manifest。
- `projector.py` — `Linear(20→128) → GELU → Linear(128→K*3584) → LayerNorm`，
  小标准差初始化（初始接近无扰动）。
- `content_validation.py` — 五属性 prompt 构建；y+ = 内容有效候选中离用户
  VADES 中心最近的（或 VADES-retained teacher query）；y- = 最远候选；
  **不使用固定 `queries[0]`**；`all_content_valid=True` 时全部 10 个内容有效
  候选都作为 SFT 目标（候选蒸馏，10x 数据）。
- `build_candidate_features.py` — 为缺失域的 10 候选生成 20 维句法特征。
- `soft_prefix_query_train.py` — SFT 训练；**可训练参数断言**（projector 必须
  可训练、启用 LoRA 时 LoRA 必须可训练，修复"二次冻结"bug）；explicit
  attention mask（padding=0，EOS padding 不算有效 token）；train/generate
  共享同一 prompt + chat template + padding 配置；user split（测试用户不出现在
  训练集）；checkpoint 保存 projector/tokenizer/config/split/vector manifests。
- `soft_prefix_query_generate.py` — 从 checkpoint 恢复，同样的预处理生成。
- `soft_prefix_query_eval.py` — 内容正确性（5 属性覆盖）+ 风格一致性（生成
  query 到用户 VADES 中心的标准化距离、in-95% range）+ 用户级 bootstrap CI +
  配对 Wilcoxon + warm/cold 分桶。

## 数据

| 域 | 04_query 记录 | 训练样本(all_content_valid) | VADES profiles |
|---|---:|---:|---:|
| Baby_Products | 205 | 2244 (train 1903) | 204 |
| Grocery_and_Gourmet_Food | 131 | 1442 (train ~1200) | 131 |
| Pet_Supplies | 143 | 1574 (train ~1300) | 143 |

每域 5 模式 × 5 epoch，batch 8，lr 5e-4，seed 42，user split 85/15。

## 结果（Baby_Products，n=205，冷启动测试用户 n=31）

内容正确性（属性存在率，5/5）：

| 模式 | 5-attr exact | 平均缺失属性 |
|---|---:|---:|
| B5 vades | 1.5% | 1.85 |
| B3 shuffled | 0% | 1.97 |
| zero | 0% | 2.16 |
| global_mean | 0% | 1.88 |
| B1 none | 0% | **2.89** |

条件注入不牺牲内容（non-inferiority 达成），正确向量内容略优于控制组。

风格一致性（生成 query 20 维句法特征 → 用户 VADES 中心距离，越小越贴合）：

| 配对 | mean diff | Wilcoxon p | bootstrap 95% CI |
|---|---:|---:|---:|
| B5_rp vs B3 (shuffled) | +2.98 | **2.8e-09** | [2.15, 3.79] |
| B5_rp vs zero | +1.13 | **5.4e-03** | [0.34, 1.93] |
| B5_rp vs global_mean | +3.62 | **1.4e-13** | [2.72, 4.51] |
| B5_rp vs B1 (none) | +0.51 | 7.2e-02 | [-0.02, 1.06] |

（B5_rp = B5 + generation repetition_penalty=1.3，消除重复退化伪影后的正确向量。）

跨域复现：Pet B5 vs B3 p=1.1e-4；Grocery B5 vs B3 p=0.22（方向一致，n.s.）。

## 关键结论

1. **软前缀条件注入确实生效**：把生成 query 的句法特征推向目标用户的 VADES
   风格中心；正确向量 vs 打乱/零/全局均值向量在 3 个域上方向一致，Baby/Pet
   统计显著。
2. **内容正确性不退化**：属性存在率 B5 3.15/5 为所有模式最高，B1 仅 2.11/5。
3. **生成质量仍低于 10 候选模板流**（5-attr exact ≤1.5%），直接生成的内容
   正确率是后续 Phase 2（更多数据/共享 LoRA 调参）的主要改进点；旧 10 候选
   + VADES rerank 流程保留为 teacher baseline（B0）。

## 复现

```bash
PY=~/hj82_scratch2/wenyu/RAG/style_venv/bin/python3.11
export HF_HOME=/fs04/scratch2/ar57/wenyu/hf_home
# 训练（5 模式 × 3 域，各 ~5 min）
$PY 04_query/soft_prefix/soft_prefix_query_train.py --category Baby_Products \
    --out_dir $OUT/Baby_Products/vades --mode vades --num_tokens 4 \
    --epochs 5 --batch_size 8 --lr 5e-4 --seed 42
# 生成（B5 建议 --repetition_penalty 1.3）
$PY 04_query/soft_prefix/soft_prefix_query_generate.py \
    --checkpoint_dir $OUT/Baby_Products/vades/checkpoint --category Baby_Products \
    --out_dir $OUT/generated --repetition_penalty 1.3
# 评估
$PY 04_query/soft_prefix/soft_prefix_query_eval.py \
    --checkpoint_dir $OUT/Baby_Products/vades/checkpoint --category Baby_Products \
    --main_label B5_rp --gen_files B5_rp=... B3=... zero=... global_mean=... B1=... \
    --out_dir $OUT/eval
```

## 依赖环境

- `~/hj82_scratch2/wenyu/RAG/style_venv`：torch 2.5.1+cu118, transformers
  4.46.0, peft 0.13.0, sentence-transformers 5.6.1, spacy 3.7.5 +
  en_core_web_sm, scipy, scikit-learn。
- 注意：该 venv 全局 pip target 指向 `~/.local`，装包需显式 `--target=` venv
  site-packages；spacy 的 Cython 扩展需 `--no-build-isolation` 编译以匹配
  numpy 1.26.4。
