from typing import Protocol


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Return one relevance score per document."""
        ...