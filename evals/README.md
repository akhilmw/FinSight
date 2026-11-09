# Retrieval evaluation dataset

`retrieval_questions.jsonl` contains 50 manually labeled retrieval questions.

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

`current_chunk_id` is included for convenient inspection, but it can change when
a filing is processed again. The evaluator resolves the stable tuple against the
current database before each run, so reprocessing does not silently invalidate
the benchmark.

Additional evaluation artifacts:

- `prompt_injection_cases.jsonl`: adversarial instructions embedded in evidence.
- `results/retrieval_summary.json`: four-mode retrieval comparison.
- `results/answer_evaluation.json`: groundedness and citation-quality results.
- `results/prompt_injection_evaluation.json`: adversarial pass/fail details.
- `results/mlflow_trace_summary.json`: aggregated tokens, model cost, and trace
  duration from the answer evaluation.

