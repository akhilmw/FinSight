from bisect import bisect_left
from dataclasses import dataclass

import tiktoken
from tiktoken import Encoding


@dataclass(frozen=True)
class SectionChunk:
    index: int
    text: str
    token_count: int
    start_token: int
    end_token: int


class SectionChunker:
    def __init__(
        self,
        max_tokens: int = 600,
        overlap_tokens: int = 100,
        encoding: Encoding | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must not be negative")

        if overlap_tokens >= max_tokens:
            raise ValueError(
                "overlap_tokens must be smaller than max_tokens"
            )

        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._encoding = encoding or tiktoken.get_encoding(
            "cl100k_base"
        )

    def chunk(self, text: str) -> list[SectionChunk]:
        normalized_text = text.strip()

        if not normalized_text:
            return []

        tokens = self._encoding.encode(normalized_text)
        chunks: list[SectionChunk] = []
        start_token = 0

        while start_token < len(tokens):
            hard_end = min(
                start_token + self._max_tokens,
                len(tokens),
            )

            end_token = self._find_natural_boundary(
                tokens=tokens,
                start_token=start_token,
                hard_end=hard_end,
            )

            chunk_text = self._encoding.decode(
                tokens[start_token:end_token]
            )

            if chunk_text.strip():
                chunks.append(
                    SectionChunk(
                        index=len(chunks),
                        text=chunk_text,
                        token_count=len(
                            self._encoding.encode(chunk_text)
                        ),
                        start_token=start_token,
                        end_token=end_token,
                    )
                )

            if end_token == len(tokens):
                break

            proposed_start = max(
                end_token - self._overlap_tokens,
                start_token + 1,
            )

            start_token = self._find_overlap_start(
                tokens=tokens,
                proposed_start=proposed_start,
                previous_end=end_token,
            )

        return chunks

    def _find_natural_boundary(
        self,
        *,
        tokens: list[int],
        start_token: int,
        hard_end: int,
    ) -> int:
        if hard_end == len(tokens):
            return hard_end

        candidate, token_offsets = (
            self._encoding.decode_with_offsets(
                tokens[start_token:hard_end]
            )
        )

        minimum_position = int(len(candidate) * 0.6)

        for separator in ("\n\n", "\n", ". "):
            position = candidate.rfind(separator)

            if position < minimum_position:
                continue

            boundary_character = position

            if separator == ". ":
                boundary_character += 1

            relative_end = bisect_left(
                token_offsets,
                boundary_character,
            )

            if relative_end > 0:
                return start_token + relative_end

        return hard_end

    def _find_overlap_start(
        self,
        *,
        tokens: list[int],
        proposed_start: int,
        previous_end: int,
    ) -> int:
        search_end = min(
            proposed_start + 10,
            previous_end,
        )

        for index in range(proposed_start, search_end):
            token_bytes = self._encoding.decode_single_token_bytes(
                tokens[index]
            )

            if token_bytes.startswith(
                (b" ", b"\n", b"\t", b"\r")
            ):
                return index

        return proposed_start