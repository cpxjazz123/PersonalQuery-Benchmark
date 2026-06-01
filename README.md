# PersonalQuery

PersonalQuery is a personalized product-search pipeline built from user review history. The project focuses on turning raw review behavior into user-grounded search queries, modeling user-specific writing patterns, and evaluating how personalized and noisy queries affect downstream retrieval.

## Pipeline Overview

The pipeline is organized as a sequential personalized-query construction workflow.

**Stage 0: User filtering and review preparation**

The pipeline first selects qualified users and collects their review history. This stage prepares the user-level review corpus that will be used in all later steps.

**Stage 1: Preference extraction**

The system extracts product attributes and user preferences from historical reviews. These preferences provide the semantic grounding for later query generation.

**Stage 2: Query-set filtering**

The extracted preferences are filtered so that only products and attributes that satisfy the required constraints are kept for query construction.

**Stage 3: Persona description generation**

This stage is intended to convert filtered preference signals into textual persona descriptions. In the current codebase, this stage is not the main operational focus of the released pipeline.

**Stage 4: Writing-pattern analysis**

The system analyzes user writing behavior, especially user-specific error patterns. These signals are later used to create realistic noisy query variants instead of generic synthetic noise.

**Stage 5: Syntactic complexity analysis**

The pipeline estimates user-level linguistic complexity, including broad and deeper syntactic patterns. These complexity signals are used to control the style and complexity level of generated queries.

**Stage 6: Personalized clean-query generation**

Given user preferences and complexity signals, the system generates personalized clean queries for target products. Each product can produce different query styles, including broader and deeper formulations.

**Stage 7: Personalized noisy-query generation**

The clean queries are transformed into noisy queries using user-specific writing-error patterns. This stage creates error-aware query variants for robustness analysis.

**Stage 8: Retrieval evaluation**

The generated queries are evaluated with retrieval models to measure ranking quality and robustness. This stage supports comparison between clean queries and noisy queries under the same product-search setting.

Overall, the pipeline maps:

`user reviews -> preferences -> linguistic profile -> clean queries -> noisy queries -> retrieval evaluation`

## Dataset Overview

The current released dataset is a clustered user-product query dataset with noisy query variants. It is built from:

- Stage 06: clean personalized queries and `attrs_used`
- Stage 07: noisy query variants with user-specific writing-error patterns
- Stage 12: `strict5550_query_gmm_user_profiles.jsonl` cluster assignments

### Included Categories

- `Baby_Products`
- `Grocery_and_Gourmet_Food`
- `Pet_Supplies`

### Dataset Files

The generated dataset is stored under `result/personal_query/11_query_dataset/`:

- `result/personal_query/11_query_dataset/Baby_Products/data.jsonl`
- `result/personal_query/11_query_dataset/Grocery_and_Gourmet_Food/data.jsonl`
- `result/personal_query/11_query_dataset/Pet_Supplies/data.jsonl`

Each file is JSONL. Each row corresponds to one user-product query instance.

### What Each Record Contains

Each dataset record includes:

- `category`: product domain
- `uuid`: user identifier
- `asin`: target product identifier
- `cluster_index`: integer query-cluster index
- `correct_query`: the correct personalized query
- `noisy_query`: the noisy query variant (may be identical to `correct_query` if no error was injected)
- `error_pattern`: the user's writing error pattern (object with `original` and `corrected` fields), or `null` if no noise injected

### Dataset Statistics

| Category | Total | With Noisy | Without Noisy |
|----------|-------|------------|--------------|
| Baby_Products | 5,537 | 471 | 5,066 |
| Grocery_and_Gourmet_Food | 6,803 | 594 | 6,209 |
| Pet_Supplies | 17,112 | 1,262 | 15,850 |
| **Total** | **29,452** | **2,327** | **27,125** |

### Example Record (with noise injection)

```json
{
  "category": "Baby_Products",
  "uuid": "AE27EZJGURITRHDXGP6RODDKD7PA",
  "asin": "B0891R8DT2",
  "cluster_index": 0,
  "correct_query": "I am looking for a Small Food Storage unit that is produced by PandaEar and costs 19.98 for Storage.",
  "noisy_query": "I am laying for a Small Food Storage unit that is produced by PandaEar and costs 19.98 for Storage.",
  "error_pattern": {
    "original": "laying",
    "corrected": "lying"
  }
}
```

In this example, the user commonly writes "laying" instead of "lying". This personal writing error was detected in Stage 04 and used in Stage 07 to create a realistic noisy query variant for robustness testing.
