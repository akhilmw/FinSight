# FinSight Week 2 Retrieval Evaluation

Date: October 16, 2026

## Objective

Evaluate vector, PostgreSQL full-text, and hybrid reciprocal-rank-fusion
retrieval over SEC filing chunks before adding answer generation or reranking.

## Evaluation set

- 25 manually authored and labeled questions.
- 34 relevant chunk labels.
- Source: NVIDIA fiscal 2026 Form 10-K.
- Accession number: `0001045810-26-000021`.
- Query types: exact, semantic, and mixed.
- Metadata filters: company, filing year, and filing type.
- Metrics: Recall@5, Recall@10, mean reciprocal rank, and end-to-end query
  latency.

The relevance labels were not changed between evaluation runs.

## Retrieval configurations

### Vector

- `text-embedding-3-small`, 1,536 dimensions.
- Cosine distance in pgvector.
- HNSW index using `vector_cosine_ops`.

### Keyword

- PostgreSQL English full-text search.
- GIN expression index over `to_tsvector('english', text)`.
- `ts_rank_cd` relevance ranking.

### Hybrid

- Vector and keyword candidate pools.
- Reciprocal rank fusion with rank constant 60.
- Candidate pool size: `max(limit * 4, 20)`.

## Baseline failure and correction

The first keyword implementation passed each full natural-language question to
`websearch_to_tsquery`. PostgreSQL connected the remaining terms with AND, so a
chunk effectively needed to contain every meaningful query term. Keyword
retrieval found relevant evidence for only two of 25 questions.

The corrected implementation extracts searchable terms, connects them with OR,
and lets `ts_rank_cd` reward chunks matching more terms. This changed only query
construction; the evaluation questions and labels remained fixed.

| Run | Mode | Recall@5 | Recall@10 | MRR | Mean latency |
|---|---|---:|---:|---:|---:|
| Baseline v1 | Vector | 0.540 | 0.600 | 0.470 | 447.6 ms |
| Baseline v1 | Keyword | 0.080 | 0.080 | 0.080 | 69.0 ms |
| Baseline v1 | Hybrid | 0.540 | 0.600 | 0.470 | 458.1 ms |
| Keyword OR v2 | Vector | 0.540 | 0.600 | 0.470 | 328.6 ms |
| Keyword OR v2 | Keyword | **0.620** | **0.760** | 0.456 | **120.3 ms** |
| Keyword OR v2 | Hybrid | 0.560 | 0.700 | **0.531** | 512.0 ms |

Latency differences between runs include external embedding-service and local
system variability and should not be interpreted as a controlled performance
improvement.

## Results by query type

| Query type | Mode | Recall@5 | Recall@10 | MRR |
|---|---|---:|---:|---:|
| Exact | Vector | 0.500 | 0.562 | 0.416 |
| Exact | Keyword | **0.625** | **0.750** | **0.396** |
| Exact | Hybrid | 0.500 | 0.625 | 0.312 |
| Semantic | Vector | 0.545 | 0.591 | 0.417 |
| Semantic | Keyword | **0.636** | **0.773** | 0.488 |
| Semantic | Hybrid | 0.545 | 0.682 | **0.616** |
| Mixed | Vector | 0.583 | 0.667 | 0.639 |
| Mixed | Keyword | 0.583 | 0.750 | 0.477 |
| Mixed | Hybrid | **0.667** | **0.833** | **0.667** |

## Interpretation

- Keyword retrieval produced the best overall Recall@5 and Recall@10 and the
  lowest mean latency.
- Hybrid retrieval produced the best overall MRR, meaning its first relevant
  result appeared earliest on average.
- Hybrid was strongest across all three metrics for mixed questions.
- Hybrid substantially improved semantic-query MRR to 0.616.
- Equal-weight RRF sometimes pushed keyword-only relevant evidence below the
  top-ten cutoff, particularly for exact questions.
- The results support using a larger hybrid candidate pool followed by a
  learned reranker rather than repeatedly tuning RRF weights on this small set.

## Recorded failure cases

### Vector zero-recall-at-10 cases

`fin-006`, `fin-012`, `fin-013`, `fin-015`, `fin-016`, `fin-017`, `fin-022`

These include manufacturing descriptions and exact financial facts such as the
H20 charge, investment amounts, liquidity, repurchases, expense drivers, and
data-center resource constraints.

### Keyword zero-recall-at-10 cases

`fin-003`, `fin-007`, `fin-009`, `fin-012`, `fin-022`

OR-based matching restored broad coverage but can still rank common filing
language above a precise answer when many question terms occur throughout the
document.

### Hybrid zero-recall-at-10 cases

`fin-007`, `fin-012`, `fin-013`, `fin-015`, `fin-016`, `fin-022`

Some relevant keyword candidates entered the fusion pool but fell below the
final top-ten cutoff after equal-weight rank fusion.

## Indexing benchmark

The benchmark used a temporary database clone of the downloaded NVIDIA filing.
It exercised parsing, section persistence, chunk creation, embedding generation,
vector persistence, and database index maintenance. The temporary filing was
deleted afterward, leaving the labeled filing and chunk IDs unchanged.

| Measurement | Result |
|---|---:|
| Sections | 23 |
| Chunks | 162 |
| Embedding batches | 3 |
| HTML processing and section persistence | 350.2 ms |
| Chunking and chunk persistence | 580.6 ms |
| Embedding generation and vector storage | 5,089.1 ms |
| Total indexing time | **6,019.9 ms** |
| End-to-end throughput | **26.9 chunks/second** |

Embedding generation and vector storage represented approximately 85% of total
indexing time. This is a single local measurement and includes external API
latency; it is an initial engineering baseline rather than a statistically
stable benchmark.

## Limitations

- The dataset covers one company and one 10-K.
- The 25 questions and relevance labels were manually created.
- Query-type subsets are small.
- Timing results come from a single local run.
- Retrieval metrics measure evidence discovery, not answer groundedness or
  citation correctness.
- Multiple chunking strategies were not benchmarked; the selected strategy is
  section-aware, token-bounded natural-boundary chunking with a 600-token maximum
  and 100-token overlap.

## Week 2 conclusion

FinSight now provides vector, keyword, and hybrid retrieval; metadata filtering;
indexed PostgreSQL search; a labeled evaluation dataset; automated retrieval
metrics; saved baselines; failure-case analysis; and an indexing benchmark.
Cross-encoder reranking is the next planned retrieval improvement.
