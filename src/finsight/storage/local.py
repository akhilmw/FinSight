from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from re import fullmatch


@dataclass(frozen=True)
class StoredDocument:
    path: str
    sha256: str


class LocalDocumentStore:
    def __init__(self, root_directory: Path) -> None:
        self._root_directory = root_directory

    def save(
        self,
        *,
        cik: str,
        accession_number: str,
        content: bytes,
    ) -> StoredDocument:
        if len(cik) != 10 or not cik.isdigit():
            raise ValueError("CIK must contain exactly 10 digits")

        if fullmatch(r"\d{10}-\d{2}-\d{6}", accession_number) is None:
            raise ValueError("Accession number has an invalid format")

        company_directory = self._root_directory / cik
        company_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = company_directory / f"{accession_number}.html"
        temporary_destination = destination.with_suffix(".html.tmp")

        temporary_destination.write_bytes(content)
        temporary_destination.replace(destination)

        content_hash = sha256(content).hexdigest()

        return StoredDocument(
            path=destination.as_posix(),
            sha256=content_hash,
        )