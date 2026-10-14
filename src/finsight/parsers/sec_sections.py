from dataclasses import dataclass
from re import IGNORECASE, compile, sub

from finsight.parsers.sec_html import ParsedFiling

ITEM_HEADING_PATTERN = compile(
    r"^item\s+(?P<item>\d{1,2}[a-z]?)\.?\s*(?P<title>.*)$",
    IGNORECASE,
)


@dataclass(frozen=True)
class SectionHeading:
    item: str
    title: str
    start_line: int
    heading_end_line: int


@dataclass(frozen=True)
class FilingSection:
    item: str
    title: str
    text: str
    start_line: int
    end_line: int


class SecSectionDetector:
    def find_candidates(
        self,
        filing: ParsedFiling,
    ) -> list[SectionHeading]:
        lines = filing.text.splitlines()
        headings: list[SectionHeading] = []

        for index, line in enumerate(lines):
            match = ITEM_HEADING_PATTERN.fullmatch(line)

            if match is None:
                continue

            item = match.group("item").upper()
            title = match.group("title").strip()
            heading_end_line = index

            # SEC sometimes puts the title on the following line.
            if not title and index + 1 < len(lines):
                title = lines[index + 1].strip()
                heading_end_line = index + 1

            if not title or title.isdigit():
                continue

            headings.append(
                SectionHeading(
                    item=item,
                    title=title,
                    start_line=index,
                    heading_end_line=heading_end_line,
                )
            )

        return headings

    def extract_sections(
        self,
        filing: ParsedFiling,
    ) -> list[FilingSection]:
        lines = filing.text.splitlines()
        candidates = self.find_candidates(filing)
        headings = self._remove_duplicates(candidates)

        sections: list[FilingSection] = []

        for index, heading in enumerate(headings):
            if index + 1 < len(headings):
                end_line = headings[index + 1].start_line
            else:
                end_line = len(lines)

            content_start = heading.heading_end_line + 1
            section_text = "\n".join(lines[content_start:end_line]).strip()

            if not section_text:
                continue

            sections.append(
                FilingSection(
                    item=heading.item,
                    title=heading.title,
                    text=section_text,
                    start_line=heading.start_line,
                    end_line=end_line,
                )
            )

        return sections

    @classmethod
    def _remove_duplicates(
        cls,
        candidates: list[SectionHeading],
    ) -> list[SectionHeading]:
        latest_headings: dict[tuple[str, str], SectionHeading] = {}

        for heading in candidates:
            key = (
                heading.item,
                cls._normalize_title(heading.title),
            )
            latest_headings[key] = heading

        return sorted(
            latest_headings.values(),
            key=lambda heading: heading.start_line,
        )

    @staticmethod
    def _normalize_title(title: str) -> str:
        normalized = title.casefold()
        normalized = sub(r"[^a-z0-9]+", " ", normalized)
        return " ".join(normalized.split())
