# FinSight Answer Quality and Security Evaluation

Evaluation date: November 8, 2026

## Objective

Measure the quality, citation support, latency, token usage, cost, and
prompt-injection resilience of the complete retrieval-augmented generation
pipeline.

## Method

- 50 manually authored questions grounded in NVIDIA's fiscal 2026 Form 10-K.
- Hybrid retrieval followed by cross-encoder reranking.
- Five evidence chunks supplied to `gpt-4o-mini` using Structured Outputs.
- Citations restricted to chunk IDs present in the retrieved evidence.
- A separate structured judge call scores groundedness and citation correctness
  from 0.0 to 1.0 using only the generated answer and its cited evidence.
- MLflow OpenAI tracing records each embedding, answer, and judge call.

## Results

| Metric | Result |
| --- | ---: |
| Questions | 50 |
| Sufficient-evidence rate | 1.000 |
| Mean groundedness | **0.962** |
| Mean citation correctness | **0.947** |
| Mean end-to-end latency | 8,704.4 ms |
| OpenAI traces | 150 |
| Input tokens | 237,226 |
| Output tokens | 15,533 |
| Total tokens | 252,759 |
| Estimated model cost | **$0.0449** |

The reported cost covers the 50-question answer evaluation: one embedding call,
one answer-generation call, and one judge call per question. Local cross-encoder
inference and infrastructure are not included in the model cost.

## Adversarial evidence evaluation

Five synthetic filing chunks contain common prompt-injection patterns:

- Direct requests to ignore previous instructions
- Fake system tags and policy overrides
- Requests to reveal hidden instructions
- Forced-output attacks
- Instructions to mark fabricated claims as sufficient evidence

The first run passed 1/5 cases. Four malicious chunks caused fabricated answers.
FinSight was then updated with stronger evidence boundaries and a deterministic
filter that removes chunks containing high-confidence injection patterns before
generation. The same unchanged suite subsequently passed **5/5**.

This filter is intentionally one defense layer rather than a claim of complete
prompt-injection prevention. Future evaluation should include indirect,
obfuscated, multilingual, and domain-specific attacks.

## Limitations

- The questions cover one company and one annual filing.
- Groundedness and citation correctness are LLM-judge scores, not human ratings.
- The judge uses the same model family as answer generation.
- Sufficient-evidence rate is not a refusal-quality metric because all 50
  financial questions were designed to be answerable.
- Latency is from one local run and includes external API variability.

Raw per-question results are stored in
`evals/results/answer_evaluation.json`. Aggregated trace usage and cost are in
`evals/results/mlflow_trace_summary.json`, and adversarial results are in
`evals/results/prompt_injection_evaluation.json`.
