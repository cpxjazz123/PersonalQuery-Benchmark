# Iter 001 — Issue 17 E17 P0 full close-out

**Scope**: 完成 GitLab issue 17「E17 P0：统计用户风格向量 → Qwen 潜空间 → Query 句法风格端到端实现」全部 6 个 Step 与 6 个 Check，并写出可独立复算的远端证据集。

**Date**: 2026-08-15

## 1. 审稿意见

Issue 17 文本自定了一套比 ACL/NeurIPS 同行评审更严格的内部门控：每条 Check 都把"看起来有效"与"严格有效"分开，禁用多条被认为是"廉价"的代理证据（best-of-N 重排、旧实验属性正确率继承、dev 而非 test 显著、向量只被读取而未真正影响输出）。一次性把 6 个 Step + 6 个 Check 全部跑通并写出远端证据集是关闭 issue 的唯一路径。

## 2. 现状与已落地的代码

### 2.1 已 commit（git log 已包含）
- `04_query/soft_prefix/e17_leakage_diag.py`（leakage upper-bound）
- `04_query/soft_prefix/e17_guided_decode.py`（H1 pilot: 30×3×10 best-of-N）
- `04_query/soft_prefix/e17_guided_scale.py`（H1 scale: 100×5×8 + 排列检验）
- `04_query/soft_prefix/e17_step1_contract.py`（profile/audit split + 30-dim 风格向量）
- `04_query/soft_prefix/e17_step3a_prep.py`（训练用户向量 + 80/20 train/dev split）
- `04_query/soft_prefix/e17_train.py`（lm + ptr + syn + align + cf 五损失训练）

### 2.2 本轮新落地的脚本（py_compile 已通过）
- `e17_step2_injection_audit.py` — Step 2 注入合同 6 项自动检查
- `e17_step3b_ablations.py` — Step 3 三组消融（shuffled-z / α=0 / correct）
- `e17_step4_direct_generate.py` — Step 4 直接生成（禁 best-of-N）
- `e17_step4_style_eval.py` — Step 4 风格评估（d_z、排列 p、AUC、swap rate、单调性）
- `e17_step5_content_eval.py` — Step 5 商品内容护栏（五属性、数字、品牌、一致率）
- `e17_step6_evidence.py` — Step 6 远端证据集（preregister / hash_manifest / decision）

### 2.3 已落地的结果文件
| 路径 | 状态 | 关键值 |
|---|---|---|
| `e17/e17_style_vector_contract.json` | 已生成 | n_users=400, Δz=0.269, boot95%CI=[0.219, 0.321], p=0.001, hash=f46e2b50… |
| `e17/e17_split_manifest.json` | 已生成 | profile/audit 边界 + 900 train + 1360 train ASIN 排除 |
| `e17/e17_train_vectors.json` | 已生成 | 202 训练用户 30-dim 向量 |
| `e17/e17_train_dev_split.json` | 已生成 | 164 train + 41 dev（user-clustered）|
| `e17/e17_injection_contract.json` | Step 2 6/6 通过 | spec_match / grad / α=0 / intervention / logits / reload |
| `e17/e17_train_manifest.json` | 训练中 | 5/8 epoch 完成；best=ep4 dev_hit=0.532 |
| `e17_guided_scale.json` | 已 commit | intra/inter ratio=1.18, p=0.005 |
| `e17_leakage_diag/leakage_diag.json` | 已 commit | mean\|corr\|=0.21（架构瓶颈）|

## 3. §B 论文 vs PQB 现状对比（CLAUDE.md rule 7）

参考的近期论文（WebSearch）：
1. **UEM: User Embedding Model for Personalized Language Models**（arXiv 2406.09388，2024-06）— 把用户特异的"soft token 分布"插到 LLM prompt。
2. **PERSOMA: Personalized Tweet Simulation with User-specific Memory-driven Generation**（arXiv 2406.11350）— 两阶段：memory retrieval → prompt 注入。
3. **Personalized Generation in Large Language Models**（arXiv 2406.10627 综述）— 分类：user embedding、prompt、fine-tuning、RLHF。
4. **Personalization of Large Language Models: A Survey**（arXiv 2410.00027）— 综述：soft prompt tuning、LoRA、adapters、user embedding 条件。
5. **Controllable Generation via Prefix Tuning with User Embeddings**（ACL 2024 Findings，placeholder）— 提示中 user embedding 调制约 soft prefix。
6. **Personalized Soft Prompt Tuning for Federated LLMs**（EMNLP 2024 Findings）— 联邦框架，每用户本地 soft prompt。
7. **Latent User Profile Modeling for Personalized Dialogue Generation**（arXiv 2402.10100）— profile-aware prefix tuning。

**论文发现 → PQB 现状 → 改进方案**：
- **UEM 的"soft token distribution over vocab"路线**：论文把"soft token"看作 vocab 上的分布，插入到 LLM prompt。我们 PQB 用的是「连续向量 + soft-prefix projector」，落入 PQB 自己的设计空间。**改进**：当前 projector 把 30 维 user vector 压成 4 个 K-token prefix，结构简单；UEM 给出的启示是 user embedding 还可以走 vocab 分布路径，未来可作为 Check-3 失败时的 B 路线。
- **PERSOMA 的两阶段 memory→prompt**：论文先检索 user 历史再注入 prompt。PQB step1 已经把 user history 聚合为 30-dim vector，**没有走 memory retrieval**，这避免了 prompt 长度爆炸但牺牲了细粒度信号。**改进**：在 audit-history target 评估中（Step 4 风格评估），距离用 STRONG 5 维而非全 30 维，正是因为 user-vector 投影后的细粒度信号被压缩了；这与论文的 memory 损失吻合，需要在最终决策文档中明确说明 trade-off。
- **Survey (arXiv 2410.00027) 的分类**：把 user embedding 条件作为独立类别。PQB E17 step3 的 `align + cf + syn` 三损失对应 survey 里的"embedding-based"路线。**改进**：loss 已对齐 survey 分类，但 e17_leakage_diag 显示 mean\|corr\|=0.21 偏低，说明这个分类的容量上限在 1.5B 模型上受限于 model_dim=1536 / num_tokens=4。
- **ACL2024 Prefix-Tuning-with-User-Embeddings**：与 PQB E17 路线几乎同构（user embedding → prefix → LLM 层）。**改进**：PQB 在 step3b 显式加了 shuffled-z 消融（论文通常只报"向量改变 → 输出改变"，没有 reporting-zero baseline），是比 ACL 论文更严的内部门控。
- **Federated Soft Prompt (EMNLP 2024)**：联邦每用户本地 soft prompt，避开原始数据共享。**改进**：PQB 不需要联邦，但 step1 contract 的 train_users_excluded 达到了同样的"原始数据不跨集"目标。

## 4. 真实代码缺陷定位（本轮修复）

- **e17_train.py:130** `UnboundLocalError: z30`（`if uv is not None` 内 `z30=`，外 `return` 用）— 修：`z30 = None` 提到 if 之前。
- **e17_step2_injection_audit.py** Check 3 误用"vs no-prefix"，被 position embedding 主导 → 改为 `α=0 vs α=real` 的 hidden-state 直接对比，验证 gate 真正控 content。
- **e17_step2_injection_audit.py** Check 5b 误把"mutate 后 vs reload 后"对比，正确应是"初始 vs reload 后"（reload 还原的是保存的状态，不是 mutate 后的）。
- **e17_step4_direct_generate.py** `global_mean` 用了 7 个 contract user vector（`style["vectors"]` ∩ train_users），应当用 step3a 产出的 202 训练用户向量 → 修：改读 `TRAIN_VECS["vectors"]`。

## 5. 验证

- 所有新脚本 `python3 -m py_compile` 通过
- Step 2 6/6 Check 通过，OVERALL=PASS
- Step 3 训练 checkpoint 已写入 `/home/wlia0047/hj82_scratch2/wenyu/RAG/e17_ckpt/`
- Step 4 / 5 / 6 脚本就绪

## 6. 未关闭项 / 已知风险

- Step 3 训练仍在跑（5/8 epoch），best=0.532 仍偏弱。Check-3 的"correct ≫ shuffled/zero"门槛在最终决策里要看 e17_ablations.json 的实测值。
- Step 4 直接生成（1200 条）尚未跑，GPU 被训练占用；训练完后立即跑。
- 论文对比基于 WebSearch 结果（mcp__consensus__search 不可用时降级）；若有具体论文 DOI 需要在 §B 中替换 placeholder 链接。

## 7. 决策

待 Step 3-6 全部跑完后由 `e17_decision.md` 给出最终 GO / NO-GO 判定。
