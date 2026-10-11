from pathlib import Path

from finsight.parsers.sec_html import SecHtmlParser
from finsight.parsers.sec_sections import SecSectionDetector

path = Path(
    "data/raw/0001045810/0001045810-26-000075.html"
)

filing = SecHtmlParser().parse(path.read_bytes())
headings = SecSectionDetector().find_candidates(filing)

for heading in headings:
    print(
        heading.start_line,
        heading.item,
        heading.title,
    )


sections = SecSectionDetector().extract_sections(filing)

for section in sections:
    print(
        section.item,
        section.title,
        section.start_line,
        section.end_line,
        len(section.text),
    )