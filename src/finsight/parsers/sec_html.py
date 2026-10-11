from dataclasses import dataclass
from re import sub
from warnings import catch_warnings, simplefilter

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


class InvalidSecHtmlError(Exception):
    pass


@dataclass(frozen=True)
class ParsedFiling:
    title: str | None
    text: str


class SecHtmlParser:
    def parse(self, content: bytes) -> ParsedFiling:
        if not content:
            raise InvalidSecHtmlError("Filing HTML must not be empty")

        with catch_warnings():
            simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(content, "lxml")

        title = None
        if soup.title is not None:
            title = self._normalize_text(soup.title.get_text(" ", strip=True))

        for element in soup.select(
            "script, style, noscript, svg, ix\\:header, ix\\:hidden"
        ):
            element.decompose()

        lines: list[str] = []

        for text_node in soup.stripped_strings:
            normalized_line = self._normalize_text(text_node)

            if normalized_line:
                lines.append(normalized_line)

        text = "\n".join(lines)

        if not text:
            raise InvalidSecHtmlError(
                "Filing HTML did not contain readable text"
            )

        return ParsedFiling(
            title=title or None,
            text=text,
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        value = value.replace("\xa0", " ")
        return sub(r"\s+", " ", value).strip()
