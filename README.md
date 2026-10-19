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

The released dataset groups users by target product and pairs each user with a clean personalized query and a noisy personalized variant derived from the user's own writing-error profile. The dataset is structured around three product categories and exposes a fixed per-record schema that consumers can rely on regardless of which category they target.

### Categories

The dataset covers three product domains:

- `Baby_Products`
- `Musical_Instruments`
- `Video_Games`

Each category has one corresponding dataset file in the release. Consumers index into the file by category name; downstream processing is otherwise identical across categories.

### File Layout

Each dataset file is a JSON object keyed by ASIN (the target product identifier). The value at each key is a list of all users who contributed a personalized query for that product.

```
<ASIN>: [
  { <user record 1> },
  { <user record 2> },
  ...
]
```

### Record Schema

Every user record inside an ASIN's list has a fixed four-field shape:

- `asin`: target product identifier (mirrors the surrounding key)
- `uid`: user identifier
- `syntax_query`: the clean personalized query generated for this user–product pair (always non-empty)
- `typo_query`: the noisy variant of `syntax_query`, produced by injecting writing errors sampled from this user's writing-error profile; the empty string when no writing error was sampled for this pair (see Dataset Statistics for the non-empty rate)

When `typo_query` is non-empty, the clean and noisy queries form a within-record comparison unit for robustness evaluation. There is no per-record error annotation beyond the visible `typo_query` text; the underlying error pattern is recoverable by diffing the two strings if needed.

### Dataset Statistics

| Category | ASINs | User–Product Pairs | `syntax_query` | `typo_query` |
|----------|-------|--------------------|----------------|--------------|
| Baby_Products | 199 | 398 | 398 | 298 |
| Musical_Instruments | 371 | 742 | 742 | 573 |
| Video_Games | 1,257 | 2,514 | 2,514 | 1,991 |
| **Total** | **1,827** | **3,654** | **3,654** | **2,862** |

`typo_query` is non-empty for the majority of pairs (about 78% overall) and empty for the remainder: when no writing error was sampled from the user's error profile for that pair, `typo_query` is the empty string and `syntax_query` should be used as the only query. Consumers should treat `syntax_query` as always present and `typo_query` as present only when non-empty.

### Example Record

```json
{
  "B09BHWCH2M": [
    {
      "asin": "B09BHWCH2M",
      "uid": "AERYYQ2LT6B427JFUK626KQY3ESA",
      "syntax_query": "I'm looking for a child door mount made of white foam, where the brand is Wittle and the style is child.",
      "typo_query": "I'm looking for a child door moun made of white foam, where the brand is Wittle and the style is child."
    },
    {
      "asin": "B09BHWCH2M",
      "uid": "AG4APLODR5CG5TCC5NVIICMQRHNQ",
      "syntax_query": "I'm after a white foam door mount from Wittle, with a child's style and a door that requires mounting on a wall.",
      "typo_query": "I'm after a white foam door munt frm Wittle, with a child's style and a door that reqires mounting on a wall."
    }
  ]
}
```

Each retained ASIN carries two distinct users with their own query texts. In the first record, the word "mount" in the clean query was corrupted to "moun" to create the noisy variant; the second record shows several such edits ("mount" to "munt", "from" to "frm", "requires" to "reqires"). These writing errors were detected from each user's historical writing patterns in the writing-pattern analysis stage.

## Documentation Layout

The repository's top-level documentation is organized as:

- `README.md` (this file) — conceptual architecture and dataset schema
- `pipeline.png` — end-to-end pipeline diagram
- Subdirectory READMEs — per-stage architectural notes (where applicable)

Consumers of the dataset and readers of the paper should treat this file as the canonical entry point. Code-level and run-level documentation live next to their implementation rather than being duplicated here.