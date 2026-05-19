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

The current released dataset is a user-product query dataset derived from the syntax-depth pipeline. Stage 08 provides the correct-query retrieval records, and Stage 09 provides paired writing-typo records for robustness analysis.

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
- `complexity_group`: syntax-depth group, one of `low`, `medium`, or `high`
- `depth`: target syntactic depth used for query construction
- `correct_query`: the correct personalized query used downstream
- `attrs_used`: product attributes or writing-typo metadata used during query construction
- `has_error_query`: whether a user-specific noisy query is available
- `error_query`: the noisy query when available
- `injected_errors`: structured description of injected syntax-depth typo anchors

Example record:

```json
{
  "category": "Baby_Products",
  "uuid": "AE23NZUELB4BYWLHKWJXAS73PTSQ",
  "asin": "B08R5BZNSP",
  "complexity_group": "medium",
  "depth": 5,
  "correct_query": "I want Small KK BETO Ties for Baby lumber whne I dres them in matching outfits, because they stay in place, and if they are made of soft fabric, the price is 8.99.",
  "attrs_used": {
    "noise_type": "clause_boundary_error",
    "correct_text": "lumber",
    "noisy_text": "wood",
    "anchor_replaced_text": "lumber"
  },
  "has_error_query": true,
  "error_query": "I want Small KK BETO Ties for Baby wood whne I dres them in matching outfits, because they stay in place, and if they are made of soft fabric, the price is 8.99.",
  "injected_errors": [
    {
      "target_token_depth": 5
    }
  ]
}
```
