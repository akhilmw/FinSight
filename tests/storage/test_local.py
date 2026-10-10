from hashlib import sha256
from pathlib import Path

import pytest

from finsight.storage import LocalDocumentStore


def test_save_writes_exact_content_and_returns_metadata(
    tmp_path: Path,
) -> None:
    content = b"<html><body>SEC filing</body></html>"
    store = LocalDocumentStore(tmp_path / "raw")

    result = store.save(
        cik="0001045810",
        accession_number="0001045810-26-000021",
        content=content,
    )

    expected_path = tmp_path / "raw" / "0001045810" / "0001045810-26-000021.html"

    assert result.path == expected_path.as_posix()
    assert result.sha256 == sha256(content).hexdigest()
    assert expected_path.read_bytes() == content
    assert not expected_path.with_suffix(".html.tmp").exists()


@pytest.mark.parametrize(
    ("cik", "accession_number"),
    [
        ("invalid", "0001045810-26-000021"),
        ("1045810", "0001045810-26-000021"),
        ("0001045810", "invalid"),
        ("0001045810", "12--34"),
    ],
)
def test_save_rejects_invalid_identifiers(
    tmp_path: Path,
    cik: str,
    accession_number: str,
) -> None:
    store = LocalDocumentStore(tmp_path / "raw")

    with pytest.raises(ValueError):
        store.save(
            cik=cik,
            accession_number=accession_number,
            content=b"content",
        )
