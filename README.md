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

The current released dataset is a flattened query dataset derived from the later stages of the pipeline, especially Stage 5, Stage 6, and Stage 7.

### Included Categories

- `Baby_Products`
- `Grocery_and_Gourmet_Food`
- `Pet_Supplies`

### Dataset Files

The dataset is stored under `dataset/` as three category-specific JSON files:

- `dataset/Baby_Products_query.json`
- `dataset/Grocery_and_Gourmet_Food_query.json`
- `dataset/Pet_Supplies_query.json`

Each file is a JSON array. Each record corresponds to one personalized query instance for a specific user-product pair and query style.

### What Each Record Contains

Each dataset record includes:

- `category`: product domain
- `uuid`: user identifier
- `asin`: target product identifier
- `query_category`: query style label
- `complexity_level`: complexity level of the generated query
- `profile_complexity_level`: user-level complexity profile
- `correct_query`: the clean personalized query
- `correct_word_count`: number of words in the clean query
- `idf`: average inverse document frequency score of the query tokens
- `attrs_used`: product attributes used during query construction
- `has_error_query`: whether a user-specific noisy query is available
- `error_query`: the noisy query when available
- `injected_errors`: structured description of injected user-specific errors

Example record:

```json
{
  "category": "Baby_Products",
  "uuid": "AG3FZXXT5BHVQW3FTLL4EF42ULRA",
  "asin": "B00195W9MM",
  "query_category": "wide",
  "complexity_level": 1,
  "profile_complexity_level": 1,
  "correct_query": "I need baby bottles from Bottle Burper priced at 9.99 which are easy to use",
  "correct_word_count": 15,
  "idf": 3.892145224737435,
  "attrs_used": {
    "A1": "Bottles",
    "A2": "Bottle Burper",
    "A3": "9.99",
    "A5": "Baby",
    "A10": "Easy"
  },
  "has_error_query": false,
  "error_query": null,
  "injected_errors": []
}
```
