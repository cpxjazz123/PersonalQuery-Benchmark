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

The current released dataset is a user-product query dataset derived from the later stages of the pipeline, especially Stage 5, Stage 6, and Stage 7. Stage 6 provides the base clean queries, and Stage 7 can revise a clean query when an anchor must be inserted before personalized noisy injection.

### Included Categories

- `Baby_Products`
- `Grocery_and_Gourmet_Food`
- `Pet_Supplies`

### Dataset Files

The dataset is stored under `dataset/` as three category-specific JSON files:

- `dataset/Baby_Products_query.json`
- `dataset/Grocery_and_Gourmet_Food_query.json`
- `dataset/Pet_Supplies_query.json`

Each file is a JSON array. Each record corresponds to one user-product pair and contains both query styles for that pair.

### What Each Record Contains

Each dataset record includes:

- `category`: product domain
- `uuid`: user identifier
- `asin`: target product identifier
- `attrs_used`: shared product attributes used during query construction
- `queries`: the paired query records for this user-product pair

Each item in `queries` includes:

- `query_category`: query style label, either `wide` or `deep`
- `complexity_level`: complexity level of the generated query
- `correct_query`: the final clean personalized query used downstream; this may be the original Stage 6 query or a Stage 7 revised query
- `correct_word_count`: number of words in the final clean query
- `idf`: average inverse document frequency score of the query tokens
- `has_error_query`: whether a user-specific noisy query is available
- `error_query`: the noisy query when available
- `injected_errors`: structured description of injected user-specific errors

Example record:

```json
{
  "category": "Grocery_and_Gourmet_Food",
  "uuid": "AGU552MYZNZ7ENJ432YPOT7MISTA",
  "asin": "B09WMP8TJ7",
  "attrs_used": {
    "A1": "Baking Syrups",
    "A2": "ONETANG",
    "A3": "7.99",
    "A5": "Baking",
    "A16": "Kosher Certified"
  },
  "queries": [
    {
      "query_category": "wide",
      "complexity_level": 2,
      "correct_query": "I need Baking Syrups from ONETANG priced at 7.99 for Baking which is Kosher Certified which is vanilla flavored, nothing overkill",
      "correct_word_count": 21,
      "idf": 4.447411123507031,
      "has_error_query": true,
      "error_query": "I need Baking Syrups from ONETANG priced at 7.99 for Baking which is Kosher Certified which is vanilla flavored, nothing over",
      "injected_errors": [
        {
          "correct": "overkill",
          "error": "over",
          "error_type": "modifier_typo"
        }
      ]
    },
    {
      "query_category": "deep",
      "complexity_level": 2,
      "correct_query": "I need Baking Syrups by ONETANG priced at 7.99 for Baking that has a rich caramel flavor with hints of licorice that works well as a vanilla extract substitute",
      "correct_word_count": 29,
      "idf": 4.176467214404339,
      "has_error_query": true,
      "error_query": "I need Baking Syrups by ONETANG priced at 7.99 for Baking that has a rich caramel flavor with hints of liquorish that works well as a vanilla extract substitute",
      "injected_errors": [
        {
          "correct": "licorice",
          "error": "liquorish",
          "error_type": "clause_boundary_error"
        }
      ]
    }
  ]
}
```
