import pytest

from finsight.parsers.sec_html import InvalidSecHtmlError, SecHtmlParser


def test_parse_extracts_readable_text_and_removes_noise() -> None:
    content = b"""
    <html>
      <head>
        <title>NVIDIA&nbsp; 10-K</title>
        <style>.hidden { display: none; }</style>
        <script>window.secret = "not filing text";</script>
      </head>
      <body>
        <ix:header>
          <ix:hidden>Hidden XBRL metadata</ix:hidden>
          <ix:resources>
            <xbrli:unit><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>
          </ix:resources>
        </ix:header>
        <h1>Item 1A. Risk Factors</h1>
        <p>Competition&nbsp; may increase.</p>
        <ix:nonNumeric>Visible inline XBRL text</ix:nonNumeric>
        <noscript>Browser fallback text</noscript>
        <svg><text>Chart decoration</text></svg>
      </body>
    </html>
    """

    result = SecHtmlParser().parse(content)

    assert result.title == "NVIDIA 10-K"
    assert "Item 1A. Risk Factors" in result.text
    assert "Competition may increase." in result.text
    assert "Visible inline XBRL text" in result.text
    assert "Hidden XBRL metadata" not in result.text
    assert "iso4217:USD" not in result.text
    assert "window.secret" not in result.text
    assert "Browser fallback text" not in result.text
    assert "Chart decoration" not in result.text


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"<html><body><script>only noise</script></body></html>",
    ],
)
def test_parse_rejects_content_without_readable_text(content: bytes) -> None:
    with pytest.raises(InvalidSecHtmlError):
        SecHtmlParser().parse(content)


def test_parse_normalizes_whitespace_while_preserving_lines() -> None:
    content = b"""
    <html><body>
      <h1>  Item   7.  Management Discussion  </h1>
      <p>Revenue\n increased\t during the year.</p>
    </body></html>
    """

    result = SecHtmlParser().parse(content)

    assert result.text == (
        "Item 7. Management Discussion\n"
        "Revenue increased during the year."
    )
