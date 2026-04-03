# AGENTS.md - Python Execution Rules

## ⚠️ CRITICAL: 所有脚本必须使用 sbatch_wrapper 运行

### 执行方式判断

| 脚本类型 | 执行方式 | 命令格式 |
|---------|---------|---------|
| **不涉及 GPU** | sbatch_wrapper（无 --gpu） | `python3 sbatch_wrapper.py "conda env execute..."` |
| **GPU 训练/检索** | sbatch_wrapper --gpu | `python3 sbatch_wrapper.py --gpu "conda env execute..."` |

### 为什么必须使用 sbatch_wrapper？

- ✅ 任务追踪和日志记录
- ✅ 超时处理
- ✅ 集群基础设施集成
- ✅ 所有任务统一管理

---

## Critical Execution Rules

### Rule 1: 所有脚本必须通过 sbatch_wrapper 运行

**需要 sbatch_wrapper --gpu（GPU 训练）**：
- ✅ 模型训练（pytorch, tensorflow 等）
- ✅ GPU 依赖的检索/评估操作

**需要 sbatch_wrapper（无 --gpu 参数）**：
- ✅ LLM 调用（GLM, GPT, Claude 等）
- ✅ 模型推理（CPU 或通过 API）
- ✅ 向量化/嵌入生成（CPU）
- ✅ 数据预处理/清洗
- ✅ 文件 IO 操作
- ✅ JSON/CSV 处理
- ✅ 纯计算

### Rule 2: 正确使用执行方式

**不涉及 GPU 时——使用 sbatch_wrapper（无 --gpu）**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && python3 script.py"
```

**涉及 GPU 时——使用 sbatch_wrapper --gpu**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && python3 script.py"
```

### Rule 3: Required Environment Every Time

Every execution call MUST include:
1. Conda activation: `source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh`
2. Environment: `conda activate /home/wlia0047/ar57_scratch/wenyu/stark`
3. Working directory: `cd /fs04/ar57/wenyu` (or appropriate path)

### Rule 4: Task Completion & Summary
- ❌ **NEVER**: Create README files or documentation after task completion
- ❌ **NEVER**: Generate documentation files
- ✅ **ALWAYS**: Print task summary directly to output
- ✅ **ALWAYS**: Summarize what was accomplished in the response

When a task is complete, provide a concise summary of:
- What was done
- Results and outcomes
- Any important information
- Print it directly without creating files

### Rule 5: Final Completion Message
- ✅ **ALWAYS**: End every completed task with the final line:
  ```
  当前任务已完成，请做下一个任务的指示。
  ```
- This must be the last sentence in your response after task completion
- Use this in all task completion summaries

### Rule 6: ALWAYS Respond in Chinese (中文)
- ✅ **ALWAYS**: Every response MUST be in Chinese (中文)
- ❌ **NEVER**: Use English in replies to the user
- ❌ **NEVER**: Mix English and Chinese in explanations
- Code comments and technical identifiers (file paths, function names) may remain in English
- ✅ **Always**: Use clear, professional Chinese for all explanations and instructions
- **Exception**: When quoting exact file content or code, preserve original formatting

### Rule 7: 禁止使用 Fallback 逻辑
- ❌ **NEVER**: 在任何逻辑、任何代码生成中添加 fallback 逻辑
- ❌ **NEVER**: 使用默认值、回退方法、降级策略等 fallback 机制
- ✅ **ALWAYS**: 如果结果不符合预期，直接抛出错误并终止逻辑
- ✅ **ALWAYS**: 明确区分"预期内的缺失"和"预期外的失败"，前者应返回空值/错误，后者应直接 raise

**正确示例**：
```python
# 预期数据可能不存在时：返回 None 或特定标记
result = data.get("optional_field")
if result is None:
    return None  # 预期内的缺失

# 预期数据必须存在时：直接抛出错误
if required_data is None:
    raise ValueError(f"Required field 'id' is missing for user {user_id}")
```

**错误示例**：
```python
# ❌ 禁止：使用 fallback 默认值掩盖错误
result = data.get("field", "default_value")  # 掩盖了数据缺失问题

# ❌ 禁止：使用 fallback 方法回退
try:
    do_primary_action()
except:
    do_fallback_action()  # 不允许 fallback，应直接 raise

# ❌ 禁止：静默忽略错误
try:
    do_something()
except:
    pass  # 不允许，必须明确处理或 raise
```

---
name: personal_query    
description: 提取细粒度用户偏好并生成"接地气"的用户画像与个性化搜索查询。
---

# Personal Query Manager

此技能实现了一套完整的 **"Grounded" (落地/实操型)** 个性化 pipeline，通过 **14 个有序阶段**（Stage 0-13）将原始用户评论转化为高度差异化的画像及搜索查询，并进行多维度评估。

---

### Stage 0: 用户筛选与数据准备

**阶段描述**：从大量用户中筛选出高质量用户（有元数据、评论数在范围内），然后加载这些用户的评论数据。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python -u PersoanlQuery/00_data_preparation/00_batch_prepare_data.py \
         --review-file /fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz \
         --meta-file /fs04/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz \
         --min-reviews 100 \
         --max-reviews 400 \
         --max-users 10 \
         --output-dir result/personal_query/00_data_preparation"
```

---

### Stage 1: 偏好提取

**阶段描述**：使用LLM从评论中提取偏好，区分目标用户（target）和其他用户（other）。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python -u PersoanlQuery/01_preference_extraction/01_extract_preferences.py \
         --output-dir result/personal_query/01_preference_extraction \
         --max-workers 10"
```

### Stage 2: 数据处理与过滤

**阶段描述**：将 Stage 1 的偏好提取结果进行数据过滤，只保留满足属性阈值的商品作为查询集。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python -u PersoanlQuery/02_processing/run_stage2_pipeline.py"
```

---

### Stage 3: 画像描述生成

**阶段描述**：该阶段需要更新以适配新的 Stage 2 输出结构，当前脚本尚未适配。

**运行命令**：暂无

---

### Stage 4: 写作风格分析

**阶段描述**：分析用户的字符级拼写错误，进行详细的错误分类和统计。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python -u PersoanlQuery/04_writing_analysis/04_extract_all_user_errors.py"
```

---

### Stage 5: LINGCONV 模型训练

**阶段描述**：训练 LINGCONV 论文实现的语言嵌入层注入模型，实现复杂度可控的句子改写。

**配置文件**：`/fs04/ar57/wenyu/PersoanlQuery/05_syntactic_analysis/lingconv.py`（参数硬编码）

**运行环境**：`conda activate /home/wlia0047/ar57/wenyu/envs/lingconv`（transformers==4.46.0）

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57/wenyu/envs/lingconv && \
     cd /fs04/ar57/wenyu/PersoanlQuery/05_syntactic_analysis && \
     python -u 05_lingconv.py"
```

**当前配置**：
- `combine_method`: decoder_add_first
- `batch_size`: 64
- `gradient_accumulation`: 4（等效 batch_size=256）
- `epochs`: 2
- `early_stopping_patience`: 3
- 缓存目录：`/home/wlia0047/ar57_scratch/wenyu/ling_conversion_data_cache`

---

### Stage 6: 查询生成

**阶段描述**：使用 Stage 5 训练的 LINGCONV 模型进行推理，生成复杂度可控的个性化查询。

**运行环境**：`conda activate /home/wlia0047/ar57_scratch/wenyu/stark`

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python -u PersoanlQuery/06_query/06_generate_all_user_queries.py"
```

---

### Stage 7: 迭代式风格优化

**阶段描述**：使用 GLM-5 模型对查询进行迭代式优化，使其风格对齐到用户写作特征。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u PersoanlQuery/07_iterative_refinement/07_iterative_refinement.py \
         --query-dir /home/wlia0047/wenyu/result/personal_query/06_query \
         --linguistic-dir /home/wlia0047/wenyu/result/personal_query/05_syntactic_analysis \
         --output-dir /home/wlia0047/wenyu/result/personal_query/07_iterative_refinement \
         --max-rounds 5 \
         --candidates-per-round 3 \
         --feature-set style_only_16 \
         --max-workers 1"
```

---

### Stage 8: 拼写难度打分模型

**阶段描述**：训练认知拼写难度模型，为后续噪声注入提供精准靶点参考。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u PersoanlQuery/08_spelling_difficulty/08_train_spelling_model.py"
```

---

### Stage 9: 靶向噪声注入

**阶段描述**：根据用户真实拼写错误历史，精确注入个性化噪声到查询中。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python -u PersoanlQuery/09_targeted_noisy_query/09_generate_all_user_noisy_queries.py"
```

---

### Stage 10: 多维度评估

**阶段描述**：验证个性化提升效果，对比 Public Query vs Personalized Query。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /fs04/ar57/wenyu && \
     python -u PersoanlQuery/10_evaluation/10_evaluate_LLM_score.py \
         --input-file /fs04/ar57/wenyu/result/personal_query/06_query/queries_A13OFOB1394G31.json \
         --persona-dir /fs04/ar57/wenyu/result/personal_query/03_persona \
         --output-dir /fs04/ar57/wenyu/result/personal_query/10_evaluation \
         --workers 10"
```

---

### Stage 11: 人类评估

**阶段描述**：验证 LLM 自动评估与真人评估的对齐度，检测系统性偏见。

**运行命令**：
```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     python -u PersoanlQuery/11_human_evaluation/11_generate_human_eval_tasks.py \
         --stage10-dir /home/wlia0047/wenyu/result/personal_query/10_evaluation \
         --stage9-dir /home/wlia0047/wenyu/result/personal_query/09_targeted_noisy_query \
         --persona-dir /home/wlia0047/wenyu/result/personal_query/03_persona/results \
         --output-dir /home/wlia0047/wenyu/result/personal_query/11_human_evaluation/tasks"
```

---

### Stage 12: 检索评估

**阶段描述**：构建索引文件、查询缓存，并评估所有用户的查询在多种检索模型下的表现。

**运行命令**：
```bash
# 构建索引文件
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/PersoanlQuery/12_retrieval/evaluators && \
     python -u 12_build_retriever_indices.py"

# 构建查询缓存
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/PersoanlQuery/12_retrieval/evaluators && \
     python -u 12_generate_query_cache.py"

# 批量评估
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
    "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
     conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
     cd /home/wlia0047/ar57/wenyu/PersoanlQuery/12_retrieval && \
     python -u 12_evaluate_all_users_fullscale.py"
```

---
