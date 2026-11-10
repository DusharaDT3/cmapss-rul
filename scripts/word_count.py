"""Count words in the report drafts using the brief's exclusions.

Usage: python scripts/word_count.py
Excluded: HTML comments, the title block (lines before the first '## ' heading), tables, figures and
their captions, URLs, the References section and anything after it, appendices, and equation lines (Q1 lines
that are mostly arithmetic). Headings in the body are counted.
"""
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def is_equation(line):
    text = re.sub(r"[*`]", "", line)
    symbols = sum(ch in "=+−-×·/()[]^ᵀ½²∂θδ0123456789.," for ch in text if not ch.isspace())
    letters = sum(ch.isalpha() for ch in text)
    return "=" in text and symbols >= letters


def body_words(path, include_references=False):
    s = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S)
    s = s[s.index("\n## "):]
    if include_references:
        # CW1 brief excludes only tables and captions, so the reference list is counted
        s = re.sub(r"\n## Appendix.*?(?=\n## References)", "\n", s, flags=re.S).replace("\n## References", "\nReferences")
    s = re.split(r"\n## (References|Appendix)", s)[0]
    words = 0
    for line in s.splitlines():
        stripped = line.strip()
        if (not stripped or stripped.startswith(("|", "![", "---")) or re.match(r"^\*(Fig|Table)\.? ?[0-9.]", stripped)
                or is_equation(stripped)):
            continue
        stripped = re.sub(r"<?https?://[^\s>]+>?", "", stripped)
        words += len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.\-]*", stripped))
    return words


def main():
    cw1 = DOCS / "CW1_technical_record_DRAFT.md"
    print(f"CW1_technical_record_DRAFT.md: {body_words(cw1, include_references=True)} words "
          f"(including references; {body_words(cw1)} without)")
    print(f"CW2_report_DRAFT.md: {body_words(DOCS / 'CW2_report_DRAFT.md')} words")


if __name__ == "__main__":
    main()
