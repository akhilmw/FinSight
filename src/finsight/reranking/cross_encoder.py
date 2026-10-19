from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        if not documents:
            return []

        pairs = [(query, document) for document in documents]

        scores = self._model.predict(
            pairs,
            show_progress_bar=False,
        )

        return [float(score) for score in scores]