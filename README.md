# PersonalQuery

本仓库实现了一套面向电商评论数据的个性化查询生成与评估流水线，目标是从用户历史评论中抽取偏好、分析写作与句法特征、生成个性化查询、注入用户特有噪声，并在检索任务上做评估。

注意：

- 目录名当前为 `PersoanlQuery/`，这是仓库内现有拼写，脚本路径请按实际目录使用。
- 大部分结果默认写入 `result/personal_query/`。
- 当前仓库只保留项目代码与必要配置；本地工具目录、文档工作区和部分数据目录已被配置为不再推送到远程。

## 目录结构

核心代码位于 `PersoanlQuery/`：

- `00_data_preparation/`：按域筛选用户并准备评论数据
- `01_preference_extraction/`：从评论中抽取商品属性与用户偏好
- `04_writing_analysis/`：抽取拼写/句法相关错误
- `05_syntactic_analysis/`：统计 ACL、CCOMP、属性密度等句法复杂度特征
- `06_query/`：基于用户画像生成正确查询
- `07_inject_noisy/`：基于用户错误模式生成 noisy query
- `08_retrieval/`：构建索引、缓存并进行检索评估
- `09_noisy_retrieval/`：评估 noisy query 的检索表现
- `10_compare_all_domain/`：跨域对比分析
- `11_query_dataset/`：构建和上传 query dataset

其它常用目录：

- `result/personal_query/`：各阶段输出结果
- `data/`：原始与处理中数据
- `logs/`：`sbatch_wrapper` 运行日志
- `bin/`：本地辅助脚本

## 运行要求

本项目主要依赖：

- Python
- Conda 环境：`/home/wlia0047/ar57_scratch/wenyu/stark`
- 集群提交包装器：`/home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py`

推荐工作目录：

```bash
cd /fs04/ar57/wenyu
```

## 运行方式

所有脚本统一通过 `sbatch_wrapper` 提交。

通用模板：

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py \
  "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
   conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
   cd /fs04/ar57/wenyu && \
   python -u <script.py>"
```

如果任务需要 GPU，则改为：

```bash
python3 /home/wlia0047/ar57/wenyu/.cursor/hooks/sbatch_wrapper.py --gpu \
  "source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh && \
   conda activate /home/wlia0047/ar57_scratch/wenyu/stark && \
   cd /fs04/ar57/wenyu && \
   python -u <script.py>"
```

## 典型流水线

下面是当前仓库中最常用的阶段顺序：

1. `00_data_preparation`
2. `01_preference_extraction`
3. `04_writing_analysis`
4. `05_syntactic_analysis`
5. `06_query`
6. `07_inject_noisy`
7. `08_retrieval`
8. `09_noisy_retrieval`
9. `10_compare_all_domain`
10. `11_query_dataset`

## 代表性脚本

按域生成正确查询：

- `PersoanlQuery/06_query/06_generate_by_persona_placeholder_Baby_Products.py`
- `PersoanlQuery/06_query/06_generate_by_persona_placeholder_Grocery_and_Gourmet_Food.py`
- `PersoanlQuery/06_query/06_generate_by_persona_placeholder_Pet_Supplies.py`

按域生成 noisy query：

- `PersoanlQuery/07_inject_noisy/07_generate_noisy_queries_by_llm_Baby_Products.py`
- `PersoanlQuery/07_inject_noisy/07_generate_noisy_queries_by_llm_Grocery_and_Gourmet_Food.py`
- `PersoanlQuery/07_inject_noisy/07_generate_noisy_queries_by_llm_Pet_Supplies.py`

按域执行 retrieval：

- `PersoanlQuery/08_retrieval/08_fast_fullscale_eval_Baby_Products.py`
- `PersoanlQuery/08_retrieval/08_fast_fullscale_eval_Grocery_and_Gourmet_Food.py`
- `PersoanlQuery/08_retrieval/08_fast_fullscale_eval_Pet_Supplies.py`

## 输出说明

常见输出位置：

- 正确查询：`result/personal_query/06_query/<Domain>/query.json`
- 错误查询：`result/personal_query/07_inject_noisy/<Domain>/noisy_query.json`
- 检索汇总：`result/personal_query/08_retrieval/<Domain>/retrieval_all_summary.json`
- noisy 检索对比：`result/personal_query/09_noisy_retrieval/<Domain>/correct_vs_noisy_results.json`

## 当前状态

当前仓库已经包含：

- 三个域的 query 生成脚本
- 三个域的 noisy query 生成脚本
- noisy retrieval 和跨域比较脚本
- query dataset 构建与 Hugging Face 上传脚本

如果要继续扩展新域，建议直接参考 `Baby_Products`、`Grocery_and_Gourmet_Food`、`Pet_Supplies` 三套现有脚本命名与输出结构。
