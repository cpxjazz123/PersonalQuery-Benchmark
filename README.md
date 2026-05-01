# PersonalQuery

This repository implements a personalized query generation and evaluation pipeline for e-commerce review data. The goal is to extract user preferences from historical reviews, analyze writing and syntactic traits, generate personalized queries, inject user-specific noise, and evaluate downstream retrieval performance.

## Repository Layout

The main code lives under `PersoanlQuery/`:

- `00_data_preparation/`: user filtering and review data preparation by domain
- `01_preference_extraction/`: product attribute and preference extraction from reviews
- `04_writing_analysis/`: user error extraction and writing-style analysis
- `05_syntactic_analysis/`: ACL, CCOMP, and attribute-density complexity analysis
- `06_query/`: correct personalized query generation
- `07_inject_noisy/`: noisy query generation based on user error patterns
- `08_retrieval/`: index building, query caching, and retrieval evaluation
- `09_noisy_retrieval/`: retrieval evaluation for noisy queries
- `10_compare_all_domain/`: cross-domain comparison
- `11_query_dataset/`: query dataset packaging and upload utilities

Other commonly used directories:

- `result/personal_query/`: stage outputs
- `data/`: raw and processed datasets
- `logs/`: execution logs
- `bin/`: local helper scripts

## Typical Pipeline

The most common stage order in the current repository is:

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

## Representative Scripts

Correct query generation by domain:

- `PersoanlQuery/06_query/06_generate_by_persona_placeholder_Baby_Products.py`
- `PersoanlQuery/06_query/06_generate_by_persona_placeholder_Grocery_and_Gourmet_Food.py`
- `PersoanlQuery/06_query/06_generate_by_persona_placeholder_Pet_Supplies.py`

Noisy query generation by domain:

- `PersoanlQuery/07_inject_noisy/07_generate_noisy_queries_by_llm_Baby_Products.py`
- `PersoanlQuery/07_inject_noisy/07_generate_noisy_queries_by_llm_Grocery_and_Gourmet_Food.py`
- `PersoanlQuery/07_inject_noisy/07_generate_noisy_queries_by_llm_Pet_Supplies.py`

Retrieval evaluation by domain:

- `PersoanlQuery/08_retrieval/08_fast_fullscale_eval_Baby_Products.py`
- `PersoanlQuery/08_retrieval/08_fast_fullscale_eval_Grocery_and_Gourmet_Food.py`
- `PersoanlQuery/08_retrieval/08_fast_fullscale_eval_Pet_Supplies.py`

## Output Files

Common output locations:

- Correct queries: `result/personal_query/06_query/<Domain>/query.json`
- Noisy queries: `result/personal_query/07_inject_noisy/<Domain>/noisy_query.json`
- Retrieval summary: `result/personal_query/08_retrieval/<Domain>/retrieval_all_summary.json`
- Correct vs. noisy retrieval comparison: `result/personal_query/09_noisy_retrieval/<Domain>/correct_vs_noisy_results.json`

## Current Scope

The repository currently includes:

- correct query generation scripts for three domains
- noisy query generation scripts for three domains
- noisy retrieval and cross-domain comparison scripts
- query dataset construction and Hugging Face upload utilities

If you want to extend the pipeline to a new domain, the existing `Baby_Products`, `Grocery_and_Gourmet_Food`, and `Pet_Supplies` implementations are the best templates for naming, stage structure, and output layout.
