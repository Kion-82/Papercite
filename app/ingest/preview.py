"""Parse one PDF, print a summary and write every block to JSONL for manual checking.

Usage (from the project root, with .venv active):
    python -m app.ingest.preview data/papers/smith2021.pdf
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from app.ingest.pdf_extract import extract

OUT_DIR = Path("data/parsed")


def main(pdf_path: str) -> None:
    path = Path(pdf_path)
    doc = extract(str(path), drop_fragments=True)

    kinds = Counter(b.kind for b in doc.blocks)
    words = [len(b.text.split()) for b in doc.blocks if b.kind == "para"]
    sections = list(dict.fromkeys(b.section for b in doc.blocks if b.section))
    cross_page = sum(1 for b in doc.blocks if b.page_end > b.page)

    print(f"File:        {path.name}")
    print(f"Title:       {doc.title or '(not found)'}")
    print(f"Pages:       {doc.n_pages}, pages without text: {doc.empty_pages or 'none'}")
    print(f"Body size:   {doc.body_size}pt")
    print(f"Blocks:      {dict(kinds)}, {cross_page} paragraph(s) span a page break")
    if words:
        print(f"Para words:  median {statistics.median(words):.0f}, max {max(words)}, "
              f"{sum(w < 30 for w in words)} under 30 words")
    print(f"Sections:    {sections}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for b in doc.blocks:
            f.write(json.dumps(b.to_dict(), ensure_ascii=False) + "\n")
    print(f"Written:     {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m app.ingest.preview <path to pdf>")
    main(sys.argv[1])
