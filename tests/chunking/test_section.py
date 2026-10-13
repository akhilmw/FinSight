from itertools import pairwise

import pytest
import tiktoken

from finsight.chunking.section import SectionChunk, SectionChunker


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens", "message"),
    [
        (0, 0, "max_tokens must be at least 1"),
        (10, -1, "overlap_tokens must not be negative"),
        (10, 10, "overlap_tokens must be smaller than max_tokens"),
        (10, 11, "overlap_tokens must be smaller than max_tokens"),
    ],
)
def test_chunker_rejects_invalid_configuration(
    max_tokens: int,
    overlap_tokens: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SectionChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_chunk_returns_empty_list_for_blank_text(text: str) -> None:
    assert SectionChunker().chunk(text) == []


def test_short_text_produces_one_chunk() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")
    text = "NVIDIA designs accelerated computing platforms."
    tokens = encoding.encode(text)

    chunks = SectionChunker(
        max_tokens=100,
        overlap_tokens=20,
        encoding=encoding,
    ).chunk(text)

    assert chunks == [
        SectionChunk(
            index=0,
            text=text,
            token_count=len(tokens),
            start_token=0,
            end_token=len(tokens),
        )
    ]


def test_long_text_uses_natural_boundaries_and_overlap() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")
    text = " ".join(f"sentence-{index}." for index in range(300))
    source_tokens = encoding.encode(text)

    chunks = SectionChunker(
        max_tokens=80,
        overlap_tokens=15,
        encoding=encoding,
    ).chunk(text)

    assert len(chunks) > 1
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].start_token == 0
    assert chunks[-1].end_token == len(source_tokens)

    for chunk in chunks:
        assert chunk.token_count <= 80
        assert chunk.text.lstrip().startswith("sentence-")
        assert chunk.text.rstrip().endswith(".")
        assert chunk.start_token < chunk.end_token

    for previous, current in pairwise(chunks):
        assert previous.start_token < current.start_token
        assert current.start_token < previous.end_token
        assert previous.end_token < current.end_token


def test_text_without_natural_boundaries_uses_hard_token_limit() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")
    text = "abcdefghijklmnopqrstuvwxyz" * 200

    chunks = SectionChunker(
        max_tokens=50,
        overlap_tokens=10,
        encoding=encoding,
    ).chunk(text)

    assert len(chunks) > 1
    assert chunks[0].end_token == 50
    assert all(chunk.token_count <= 50 for chunk in chunks)


def test_zero_overlap_produces_adjacent_source_ranges() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 200
    chunks = SectionChunker(
        max_tokens=50,
        overlap_tokens=0,
    ).chunk(text)

    for previous, current in pairwise(chunks):
        assert current.start_token == previous.end_token


def test_boundary_whitespace_does_not_exceed_token_limit() -> None:
    chunks = SectionChunker(
        max_tokens=1,
        overlap_tokens=0,
    ).chunk("alpha improvements")

    assert all(chunk.token_count <= 1 for chunk in chunks)