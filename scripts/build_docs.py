"""Convert the Markdown drafts in docs/ to PDF and Word via HTML and LibreOffice (headless).

Usage: python scripts/build_docs.py
Outputs: docs/build/<name>.pdf and docs/build/<name>.docx
"""
import base64
import re
import subprocess
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parents[1] / "docs"
SOURCES = ["CW1_technical_record_DRAFT", "CW2_report_DRAFT", "AI_use_declaration"]
CSS = """
body { font-family: 'Liberation Serif', 'Times New Roman', serif; font-size: 11pt; line-height: 1.4; }
h1 { font-size: 17pt; } h2 { font-size: 14pt; margin-top: 18pt; } h3 { font-size: 12pt; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #888; padding: 3pt 5pt; vertical-align: top; }
img { width: 16cm; }
code { font-family: 'Liberation Mono', monospace; font-size: 9.5pt; }
"""


def main():
    out = DOCS / "build"
    out.mkdir(exist_ok=True)
    for name in SOURCES:
        body = markdown.markdown((DOCS / f"{name}.md").read_text(), extensions=["tables", "fenced_code"])
        # Embed images (so the Word file is self-contained), size them to the page, and border the tables.
        body = re.sub(r'<img alt="([^"]*)" src="([^"]+)"', lambda m: (
            f'<img width="620" alt="{m.group(1)}" src="data:image/png;base64,'
            f'{base64.b64encode((DOCS / m.group(2)).read_bytes()).decode()}"'), body)
        body = body.replace("<table>", '<table border="1" cellpadding="4" cellspacing="0">')
        html = out / f"{name}.html"
        html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
        subprocess.run(["soffice", "--headless", "--infilter=HTML (StarWriter)", "--convert-to", "docx:MS Word 2007 XML",
                        "--outdir", str(out), str(html)], check=True, capture_output=True)
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf:writer_pdf_Export", "--outdir", str(out),
                        str(out / f"{name}.docx")], check=True, capture_output=True)
        html.unlink()
        print("built", name)


if __name__ == "__main__":
    main()
