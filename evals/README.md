# Retrieval evaluation dataset

`retrieval_questions.jsonl` contains the initial manually labeled retrieval set.

Each JSON line contains:

- `id`: stable evaluation-question identifier.
- `question`: query sent to the retrieval system.
- `query_type`: `semantic`, `exact`, or `mixed`.
- `filters`: metadata filters used for the query. `year` refers to `filing_date`.
- `relevant_chunks`: one or more acceptable relevance labels.
- `answer_hint`: a short fact used to manually verify that the label is grounded.

A relevance label is primarily identified by this stable tuple:

```text
(accession_number, item, section_title, chunk_index)
```

`current_chunk_id` is included for convenient evaluation against the current local
database, but it may change if the filing is deleted and re-ingested. Evaluators
should prefer the stable tuple when resolving labels.

