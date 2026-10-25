"""Split a parsed paper into retrieval chunks.

Each chunk is a run of consecutive paragraphs from one section, merged up to
roughly 300-500 words. A chunk keeps its paragraphs as separate passages, each
with its own page number, instead of joining them into one string. Later, when
a chunk is sent to Claude as a search_result block, every passage becomes its
own text block; Claude's citations point at block indices, so a citation can be
traced back to the exact page of the passage it quotes, even when the chunk
spans a page break.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .pdf_extract import Block, Document, extract

TARGET_WORDS = 400   # close a chunk once it reaches this size
MAX_WORDS = 500      # never let a chunk grow past this by adding a paragraph
MIN_WORDS = 120      # a smaller leftover at the end of a section is merged into the previous chunk
OVERLAP_WORDS = 40   # carry up to this many words (whole sentences) into the next chunk

# Sections that are noise for question answering. The reference list alone is
# often a quarter of a paper's words, and its author names and journal titles
# would match many keyword queries.
_SKIP_SECTION_RE = re.compile(
    r"^(references|acknowledge?ments?|author\s+contributions?|author\s+information|"
    r"funding|declarations?|competing\s+interests?|conflicts?\s+of\s+interest|ethics|"
    r"additional\s+information)\b", re.I)

# A real figure/table caption, as opposed to labels and formula fragments inside a figure
_CAPTION_START_RE = re.compile(r"^(fig(ure)?|table|tab|scheme|plate)\s*\.?\s*\d+", re.I)

# Front matter before the first heading: the author list and affiliations.
# "Yann LeCun, Yoshua Bengio & Geoffrey Hinton", "A. Smith, B. Jones and C. Lee"
_AUTHORS_RE = re.compile(r"^[A-Z][\w.'’\- ]+(?:,\s*[A-Z][\w.'’\- ]+)*,?\s+(?:&|and)\s+[A-Z][\w.'’\- ]+$")
_AFFILIATION_RE = re.compile(
    r"\b(university|universidad|universit[éäà]|institute|instituto|department|school of|"
    r"faculty of|centre|center|laboratory|college)\b", re.I)

# Abbreviations that end with a period but don't end a sentence
_ABBREV_RE = re.compile(
    r"\b(?:et al|e\.g|i\.e|cf|vs|approx|ca|Fig|Figs|Eq|Eqs|Ref|Refs|No|Dr|Prof|"
    r"sp|spp|var|subsp|[A-Z])\.$")


@dataclass
class Passage:
    text: str
    page: int


@dataclass
class Chunk:
    chunk_id: str        # e.g. "smith2021#c007"; stable as long as the PDF and the code don't change
    paper_id: str        # file name without extension, e.g. "smith2021"
    paper_title: str
    section: str
    kind: str            # "text" (body paragraphs) or "figure" (captions)
    passages: list[Passage] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(p.text for p in self.passages)

    @property
    def n_words(self) -> int:
        return sum(len(p.text.split()) for p in self.passages)

    @property
    def page_start(self) -> int:
        return min(p.page for p in self.passages)

    @property
    def page_end(self) -> int:
        return max(p.page for p in self.passages)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(page_start=self.page_start, page_end=self.page_end, n_words=self.n_words)
        return d


def split_sentences(text: str) -> list[str]:
    """Split on ". ", "? ", "! " before a capital letter, except after known abbreviations.

    Rule-based and imperfect (e.g. a sentence ending in a species abbreviation),
    but good enough for choosing overlap and splitting over-long paragraphs.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    sentences: list[str] = []
    for part in parts:
        if sentences and _ABBREV_RE.search(sentences[-1]):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences


def _overlap_tail(passages: list[Passage]) -> list[Passage]:
    """The last 1-2 sentences of a chunk, as long as they fit in OVERLAP_WORDS."""
    last = passages[-1]
    tail: list[str] = []
    for sent in reversed(split_sentences(last.text)):
        if tail and sum(len(s.split()) for s in tail) + len(sent.split()) > OVERLAP_WORDS:
            break
        if len(sent.split()) > OVERLAP_WORDS:
            break
        tail.insert(0, sent)
        if len(tail) == 2:
            break
    return [Passage(" ".join(tail), last.page)] if tail else []


def _split_long(block: Block) -> list[Passage]:
    """Split a paragraph longer than MAX_WORDS into sentence groups that fit."""
    pieces: list[Passage] = []
    current: list[str] = []
    for sent in split_sentences(block.text):
        if current and sum(len(s.split()) for s in current) + len(sent.split()) > MAX_WORDS:
            pieces.append(Passage(" ".join(current), block.page))
            current = []
        current.append(sent)
    if current:
        pieces.append(Passage(" ".join(current), block.page))
    return pieces


def _group(paragraphs: list[Passage]) -> list[list[Passage]]:
    """Greedily merge consecutive paragraphs into groups of about TARGET_WORDS."""
    groups: list[list[Passage]] = []
    current: list[Passage] = []
    n = 0
    for para in paragraphs:
        w = len(para.text.split())
        if current and n + w > MAX_WORDS:
            groups.append(current)
            current, n = [], 0
        current.append(para)
        n += w
        if n >= TARGET_WORDS:
            groups.append(current)
            current, n = [], 0
    if current:
        prev_words = sum(len(p.text.split()) for p in groups[-1]) if groups else 0
        if groups and n < MIN_WORDS and prev_words + n <= MAX_WORDS + MIN_WORDS:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def _is_front_matter(b: Block) -> bool:
    """Author list or affiliation block before the first heading.

    Only blocks with no section are checked, so a body paragraph that happens to
    mention a university is never dropped. Affiliation blocks are short and list
    several institutions, so at least two institution words are required.
    """
    if b.section or len(b.text.split()) > 120:
        return False
    if len(b.text.split()) <= 40 and _AUTHORS_RE.match(b.text):
        return True
    return len(_AFFILIATION_RE.findall(b.text)) >= 2


def _runs(doc: Document) -> list[tuple[str, str, list[Block]]]:
    """Group the blocks into runs of (section, kind, blocks). Chunks never cross a run.

    Body text and figure captions go into separate runs. A caption sits wherever
    the figure was placed, often between two halves of the same paragraph;
    mixing them would make neither cleanly retrievable. So a body-text run
    continues past an interrupting caption, and each caption gets its own run.
    Captions are recognised by their opening ("Figure 2 |", "Table 1.") plus any
    following smaller-than-body blocks in the caption's own font size, which
    continue it. Other small text (figure labels, formula fragments) is dropped.
    """
    runs: list[tuple[str, str, list[Block]]] = []
    text_run: tuple[str, str, list[Block]] | None = None
    caption_run: tuple[str, str, list[Block]] | None = None
    for b in doc.blocks:
        if b.kind in ("title", "heading") or _SKIP_SECTION_RE.match(b.section) \
                or _is_front_matter(b):
            caption_run = None
            continue
        small = b.size < doc.body_size - 0.5
        if _CAPTION_START_RE.match(b.text):
            caption_run = (b.section, "figure", [b])
            runs.append(caption_run)
        elif small and caption_run is not None \
                and abs(b.size - caption_run[2][0].size) <= 0.5:
            caption_run[2].append(b)
        elif small or b.kind == "caption":
            continue                      # labels and fragments inside figures
        else:
            caption_run = None
            if text_run is None or text_run[0] != b.section:
                text_run = (b.section, "text", [])
                runs.append(text_run)
            text_run[2].append(b)
    return runs


def chunk_document(doc: Document, paper_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, kind, blocks in _runs(doc):
        paragraphs = [p for b in blocks for p in
                      (_split_long(b) if len(b.text.split()) > MAX_WORDS else [Passage(b.text, b.page)])]
        prev: Chunk | None = None
        for group in _group(paragraphs):
            passages = list(group)
            # Overlap only between body-text chunks of the same run (same section)
            if prev is not None and kind == "text":
                passages = _overlap_tail(prev.passages) + passages
            prev = Chunk(chunk_id=f"{paper_id}#c{len(chunks):03d}", paper_id=paper_id,
                         paper_title=doc.title or paper_id, section=section,
                         kind=kind, passages=passages)
            chunks.append(prev)
    return chunks


def chunk_pdf(path: str | Path) -> list[Chunk]:
    path = Path(path)
    doc = extract(str(path), drop_fragments=True)
    return chunk_document(doc, paper_id=path.stem)
