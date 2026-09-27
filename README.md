# PersonalQuery

PersonalQuery is a personalized product-search pipeline built from user review history. The project focuses on turning raw review behavior into user-grounded search queries, modeling user-specific writing patterns, and evaluating how personalized and noisy queries affect downstream retrieval.

This document describes the conceptual architecture of the pipeline and the structure of its released dataset. Operational details (code, scripts, configuration, run instructions) live in their respective subdirectories; this README is intentionally free of execution references.

## Pipeline Architecture

The pipeline is organized as a sequential personalized-query construction workflow. Each stage transforms a representation into a richer one, and the full chain maps raw user history into retrieval-ready queries and their noisy variants.

<img src="pipeline.png" width="100%" alt="Personalized Query Construction and Retrieval Evaluation Pipeline">

The seven conceptual stages are:

**User filtering and review preparation**

The pipeline first selects qualified users and collects their review history. This stage prepares the user-level review corpus that will be used in all later steps.

**Preference extraction**

The system extracts product attributes and user preferences from historical reviews. These preferences provide the semantic grounding for later query generation.

**Writing-pattern analysis**

The system analyzes user writing behavior, especially user-specific error patterns. These signals are later used to create realistic noisy query variants instead of generic synthetic noise.

**Syntactic complexity analysis**

The pipeline estimates user-level linguistic complexity, including broad and deeper syntactic patterns. These complexity signals are used to control the style and complexity level of generated queries.

**Personalized query generation**

Given user preferences and complexity signals, the system generates personalized queries for target products. Each product can produce different query styles, including broader and deeper formulations.

**Personalized noisy-query generation**

These personalized queries are transformed into noisy queries using user-specific writing-error patterns. This stage creates error-aware query variants for robustness analysis.

**Retrieval evaluation**

The generated queries are evaluated with retrieval models to measure ranking quality and robustness. This stage supports comparison between personalized queries and their noisy variants under the same product-search setting.

Overall, the pipeline maps:

`user reviews -> preferences -> linguistic profile -> personalized queries -> noisy queries -> retrieval evaluation`

## Dataset Architecture

The released dataset is a clustered user-product query dataset with optional noisy query variants. It is structured around three product categories and exposes a single record envelope that consumers can rely on regardless of which category they target.

### Categories

The dataset covers three product domains:

- `Baby_Products`
- `Grocery_and_Gourmet_Food`
- `Pet_Supplies`

Each category has one corresponding dataset file in the release. Consumers index into the file by category name; downstream processing is otherwise identical across categories.

### Record Envelope

Each grouped record corresponds to one user-product pair. A grouped record contains one query entry per cluster that the pair appears in.

The fixed envelope of every grouped record is:

- `category`: product domain
- `uuid`: user identifier
- `asin`: target product identifier
- `queries`: list of query entries (see below)

Each query entry has a variable shape:

- `cluster`: integer query-cluster index
- `correct_query`: the correct personalized query

If a writing error was injected for this entry, two additional fields are present:

- `noisy_query`: the noisy query variant (always non-empty when present)
- `error_pattern`: the writing error injected into the query (object with `correct_word` and `error_word` fields)
  - `correct_word`: the correct word from `correct_query` that was replaced
  - `error_word`: the error word that replaced it in `noisy_query`

When no error was injected, the entry has only `cluster` and `correct_query`; the `noisy_query` and `error_pattern` fields are omitted. This is the dominant case: most user-product pairs in the dataset do not have a personalized noise variant.

### Dataset Statistics

| Category | Total | With Noise | Without Noise |
|----------|-------|------------|---------------|
| Baby_Products | 6,535 | 540 | 5,995 |
| Grocery_and_Gourmet_Food | 6,141 | 536 | 5,605 |
| Pet_Supplies | 13,933 | 1,166 | 12,767 |
| **Total** | **26,609** | **2,242** | **24,367** |

### Example Record (with noise injection)

```json
{
  "category": "Baby_Products",
  "uuid": "AE27EZJGURITRHDXGP6RODDKD7PA",
  "asin": "B0891R8DT2",
  "queries": [
    {
      "cluster": 0,
      "correct_query": "I am looking for a Small Food Storage unit that is produced by PandaEar and costs 19.98 for Storage.",
      "noisy_query": "I am laying for a Small Food Storage unit that is produced by PandaEar and costs 19.98 for Storage.",
      "error_pattern": {
        "correct_word": "looking",
        "error_word": "laying"
      }
    }
  ]
}
```

In this example, the word "looking" in the personalized query was replaced with "laying" to create a realistic noisy query variant. This writing error was detected from the user's historical writing patterns in the writing-pattern analysis stage.

## Documentation Layout

The repository's top-level documentation is organized as:

- `README.md` (this file) — conceptual architecture and dataset schema
- `pipeline.png` — end-to-end pipeline diagram
- Subdirectory READMEs — per-stage architectural notes (where applicable)

Consumers of the dataset and readers of the paper should treat this file as the canonical entry point. Code-level and run-level documentation live next to their implementation rather than being duplicated here.