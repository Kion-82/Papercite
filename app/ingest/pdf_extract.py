"""Turn a PDF into paragraphs in reading order, each with its page and section.

Reused from the Eng_Zh-translator project, keeping only the pdfminer.six
backend. Academic papers come in many layouts: single-column books,
two-column journals, three-column reviews, and pages with a full-width
figure in the middle. The problems this module handles:

  1. Columns -- left and right columns must not be interleaved line by line.
  2. Running heads, footers and page numbers -- repeated noise on every page.
  3. Line breaks and hyphenation -- "un-" + "naturalness" -> "unnaturalness".
  4. Citation superscripts -- the digits in "recognition1-4" must be removed,
     otherwise they pollute retrieval text and model input.
  5. Paragraphs split across columns or pages -- the unfinished part must be
     joined back on.
  6. Ligatures and unmapped glyphs -- fi/fl ligatures, (cid:123) and so on.

pdfminer.six does its own layout analysis and groups each page into text
boxes, which never span a column gutter. A box-level XY-cut then sets the
reading order. Every character has an exact font size and position, which is
what makes it possible to detect superscripts as "smaller and raised".
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass, field

from . import pdf_extract_pdfminer as _pdfminer_backend


@dataclass
class Block:
    """A block of text: a paragraph, heading or figure/table caption."""

    text: str
    # Pages are always physical PDF pages (page 1 = first page of the file),
    # not the page numbers printed by the journal.
    page: int  # first page, 1-based
    kind: str = "para"  # para | heading | caption | title
    size: float = 10.0
    page_end: int = 0  # last page; can exceed `page` after merging across pages; 0 means same as `page`
    section: str = ""  # heading of the enclosing section ("References" for the reference list)

    def __post_init__(self):
        if not self.page_end:
            self.page_end = self.page

    def to_dict(self) -> dict:
        return {"text": self.text, "page": self.page, "page_end": self.page_end,
                "kind": self.kind, "size": self.size, "section": self.section}


@dataclass
class Document:
    blocks: list[Block] = field(default_factory=list)
    n_pages: int = 0
    empty_pages: list[int] = field(default_factory=list)  # pages with no text layer (scans)
    title: str = ""
    body_size: float = 0.0  # font size of the body text, used for heading/caption rules
    backend: str = "pdfminer"


# --------------------------------------------------------------------------
# Line-level helpers
# --------------------------------------------------------------------------

_PAGENO_RE = re.compile(r"^[\s\-–—|]*(?:page\s*)?[ivxlcdm\d]{1,6}[\s\-–—|]*$", re.I)
_BOILERPLATE_RE = re.compile(
    r"^(downloaded from|this content downloaded|©|copyright|doi:|https?://|arxiv:)", re.I)

# Running heads and footers only appear near the top and bottom edges. The
# page-number / short-line / all-caps rules are applied only inside this
# margin band. Applying them everywhere used to delete all-caps section
# headings (INTRODUCTION, REFERENCES) and short last lines such as "it.".
_MARGIN_BAND = 0.10   # top and bottom 10% of the page
_REPEAT_BAND = 0.15   # repetition across pages is checked in the top and bottom 15%


def _in_band(ln: dict, height: float, frac: float) -> bool:
    return ln["top"] < height * frac or ln["bottom"] > height * (1 - frac)


def _repeat_key(ln: dict) -> tuple[str, int] | None:
    """Key for spotting running heads: text without digits + vertical position (~10pt buckets).

    A running head differs only by its page number and always sits at the same
    height. The position is needed because lines in a two-column paper are often
    under 70 characters; comparing text alone could drop a body line that happens
    to appear on two pages.
    """
    text = re.sub(r"[\d\s]+", " ", ln["text"].lower()).strip()
    return (text, int(round(ln["top"] / 10.0))) if text else None


def _isolated(ln: dict) -> bool:
    """True if the line's text box has at most 2 lines."""
    return ln.get("_nbox", 1) <= 2


def _looks_like_running_head(ln: dict, height: float, body_size: float) -> bool:
    t = ln["text"].strip()
    if not t:
        return True
    if len(t) > 70:
        return False
    if _BOILERPLATE_RE.match(t):
        return True
    if not _in_band(ln, height, _MARGIN_BAND):
        return False
    if _PAGENO_RE.match(t):
        return True
    # Very short lines of symbols or digits; keep ones with lowercase letters
    # (e.g. a paragraph's last line "it.")
    if len(t) <= 3 and not re.search(r"[a-z]", t):
        return True
    # A short all-caps line in the margin band that is not bold, not larger than
    # body text and sits in its own box is usually a running head. Real headings
    # are normally bold or larger, and a line inside a paragraph box doesn't count.
    letters = [c for c in t if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.9 \
            and not ln["_bold"] and ln["_size"] <= body_size + 0.3 and _isolated(ln):
        return True
    return False


def _filter_running_heads(pages: list[tuple[float, list[list[dict]]]],
                          body_size: float) -> list[list[list[dict]]]:
    """pages: [(page height, [lines per region...])]. Drop heads/footers, keep the structure.

    Besides the single-line rules, this also detects repetition across pages:
    the same line (ignoring page numbers) in the margin band on enough pages is
    a running head, whether or not it is in capitals.
    """
    from collections import Counter

    seen: Counter = Counter()
    n_live = 0
    for height, regions in pages:
        if not regions:
            continue
        n_live += 1
        keys = {_repeat_key(ln) for lines in regions for ln in lines
                if len(ln["text"]) <= 70 and _isolated(ln) and _in_band(ln, height, _REPEAT_BAND)}
        seen.update(k for k in keys if k)
    threshold = max(2, -(-3 * n_live // 10)) if n_live >= 3 else None
    # Neighbouring vertical buckets count as the same position, so a running head
    # that sits right on a bucket boundary isn't split into two groups.
    repeated = {k for k in seen if threshold and
                sum(seen.get((k[0], k[1] + d), 0) for d in (-1, 0, 1)) >= threshold}

    out = []
    for height, regions in pages:
        new_regions = []
        for lines in regions:
            kept = []
            for ln in lines:
                if _looks_like_running_head(ln, height, body_size):
                    continue
                if repeated and len(ln["text"]) <= 70 and _isolated(ln) \
                        and _in_band(ln, height, _REPEAT_BAND) \
                        and _repeat_key(ln) in repeated:
                    continue
                kept.append(ln)
            new_regions.append(kept)
        out.append(new_regions)
    return out


def _rect_gap(boxes: list[dict], axis: str, min_gap: float) -> float | None:
    """Project boxes onto one axis and return the midpoint of the widest gap (or None)."""
    lo_key, hi_key = ("x0", "x1") if axis == "x" else ("top", "bottom")
    spans = sorted(((b[lo_key], b[hi_key]) for b in boxes), key=lambda t: t[0])
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    best = None
    for a, b in zip(merged, merged[1:]):
        gap = b[0] - a[1]
        if gap >= min_gap and (best is None or gap > best[1]):
            best = ((a[1] + b[0]) / 2, gap)
    return best[0] if best else None


def _order_boxes(boxes: list[dict], depth: int = 0) -> list[list[dict]]:
    """Set reading order with a box-level XY-cut: split columns first, then horizontal bands.

    pdfminer's text boxes are the atomic unit. They are already the output of
    layout analysis and never span a column, so column gutters are clean. This
    is far more stable than guessing from character coordinates.
    """
    if len(boxes) <= 1 or depth >= 8:
        return [sorted(boxes, key=lambda b: (b["top"], b["x0"]))]

    cut = _rect_gap(boxes, "x", 8.0)
    if cut is not None:
        left = [b for b in boxes if b["x1"] <= cut]
        right = [b for b in boxes if b["x0"] >= cut]
        if left and right and len(left) + len(right) == len(boxes):
            return _order_boxes(left, depth + 1) + _order_boxes(right, depth + 1)

    cut = _rect_gap(boxes, "y", 4.0)
    if cut is not None:
        upper = [b for b in boxes if b["bottom"] <= cut]
        lower = [b for b in boxes if b["top"] >= cut]
        if upper and lower and len(upper) + len(lower) == len(boxes):
            return _order_boxes(upper, depth + 1) + _order_boxes(lower, depth + 1)

    return [sorted(boxes, key=lambda b: (b["top"], b["x0"]))]


_END_PUNCT = tuple(".!?\"')”’]:;")
# End of sentence: the closing punctuation may be followed by a footnote number ("...mouths.42")
_ENDS_SENTENCE = re.compile(r"""[.!?…"'”’)\]:;]\s*\d{0,3}$""")
# Note: a bare letter "o" used to count as a bullet, so continuation lines
# starting with "of ..." or "on ..." were treated as new paragraphs. Every
# bullet symbol now has to be followed by whitespace.
_BULLET_RE = re.compile(
    r"^\s*(?:[•·▪◦∙‣]\s*|o\s+|[-–—]\s+|\(?\d{1,2}[.)]\s+|[a-z]\)\s+|[IVXLC]{1,5}[.)]\s+)")


_NOTE_MARK_RE = re.compile(r"^[\d\s.,·*†‡]{1,6}$")


def _group_lines(lines: list[dict], body_size: float) -> list[tuple[str, str, float]]:
    """Merge lines into (text, kind, size) paragraphs."""
    # Footnote/endnote markers sometimes come out as a line of their own in a
    # smaller font. Left in, they split a paragraph in two, so drop them first.
    lines = [ln for ln in lines
             if not (ln["_size"] < body_size - 0.5 and _NOTE_MARK_RE.match(ln["text"]))]
    if not lines:
        return []

    gaps = []
    for a, b in zip(lines, lines[1:]):
        g = b["top"] - a["bottom"]
        if -5 < g < 40:
            gaps.append(g)
    med_gap = statistics.median(gaps) if gaps else 2.0

    right_edge = max(ln["x1"] for ln in lines)
    left_edge = min(ln["x0"] for ln in lines)

    paras: list[list[dict]] = [[lines[0]]]
    for prev, cur in zip(lines, lines[1:]):
        newpara = False
        gap = cur["top"] - prev["bottom"]
        if gap > med_gap + max(3.0, med_gap * 1.6):
            newpara = True
        if abs(cur["_size"] - prev["_size"]) > 0.6:
            newpara = True
        if cur["_bold"] != prev["_bold"] and gap > med_gap * 0.6:
            newpara = True
        # The previous line stops short of the right edge and ends with sentence punctuation
        if prev["text"].endswith(_END_PUNCT) and prev["x1"] < right_edge - (right_edge - left_edge) * 0.12:
            newpara = True
        if _BULLET_RE.match(cur["text"]):
            newpara = True
        # First-line indent
        if cur["x0"] > left_edge + 8 and gap >= med_gap * 0.8:
            newpara = True
        if newpara:
            paras.append([cur])
        else:
            paras[-1].append(cur)

    out = []
    for p in paras:
        text = _join_lines([ln["text"] for ln in p])
        if not text:
            continue
        size = round(statistics.median([ln["_size"] for ln in p]), 1)
        bold = sum(1 for ln in p if ln["_bold"]) > len(p) / 2
        kind = "para"
        words = len(text.split())
        # Captions are checked first: "Figure 1 | ..." is often bold and short,
        # and must not be mistaken for a section heading.
        if re.match(r"^(fig(ure)?|table|tab|chart|scheme|eq(uation)?|plate)\s*\.?\s*\d+\s*[.:|)]", text, re.I):
            kind = "caption"
        elif _looks_like_heading(text, size, bold, body_size):
            kind = "heading"
        elif size < body_size - 0.8 and words < 60:
            kind = "caption"
        out.append((text, kind, size))
    return out


def _looks_like_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    """A heading is short, set larger than the body text, and reads as a complete label.

    Size: bold text must be at least 0.5pt larger than the body; non-bold text
    at least 1.5pt larger. Bold at body size is never enough. Nature-style
    papers set the first line of some paragraphs in bold at body size, and
    figures contain labels slightly larger than the body (but not bold);
    counting either as headings mislabels every block that follows.
    The trade-off: journals whose subheadings are bold *at body size* lose those
    subheadings, and their blocks keep the enclosing (larger) heading instead.
    A missing subheading is a smaller error than a wrong one.
    """
    if size < body_size + (0.5 if bold else 1.5):
        return False
    if len(text.split()) > 30:
        return False
    # Unfinished text (a line cut mid-sentence or mid-word) is not a heading
    if text.endswith((".", ",", "-", "‐", "‑")):
        return False
    # Author lists: "A. Smith, B. Jones & C. Lee", or many commas
    if text.count(",") > 2 or ("," in text and " & " in text):
        return False
    if not (text[:1].isupper() or text[:1].isdigit()):
        return False
    return True


def _body_size(lines: list[dict]) -> float:
    """Font size of the body text: the size that carries the most characters.

    The median over *lines* is not reliable: a reference list and figure labels
    produce many short lines in small fonts. In lecun2015 they pull the median
    line size down to 8.3pt while the body is 9.3pt.
    """
    chars: dict[float, int] = {}
    for ln in lines:
        chars[ln["_size"]] = chars.get(ln["_size"], 0) + len(ln["text"])
    return max(chars, key=chars.get) if chars else 10.0


def _normalize(text: str) -> str:
    """Normalise typographic characters: split ﬁ/ﬂ ligatures, full-width to ASCII, odd spaces.

    PDFs often encode fi and fl as single ligature characters (U+FB01/FB02).
    Left as they are, keyword search fails to match ("ﬁsh" != "fish").
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace(" ", " ").replace("﻿", "")
    # pdfminer emits (cid:123) for glyphs it cannot map; drop them
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Symbol fonts map glyphs (author asterisks, maths symbols) into the Unicode
    # Private Use Area, e.g. "". They have no meaning as text; drop them.
    text = re.sub(r"[-]", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines: handle end-of-line hyphens, otherwise join with a space."""
    buf = ""
    for raw in lines:
        ln = raw.strip()
        if not ln:
            continue
        if not buf:
            buf = ln
            continue
        if re.search(r"[A-Za-z]([-‐‑–])$", buf):
            # Hyphen at line end: if the next line starts lowercase, it is a split word
            if re.match(r"^[a-z]", ln):
                buf = buf[:-1] + ln
            else:
                buf = buf + ln
        else:
            buf = buf + " " + ln
    return re.sub(r"\s{2,}", " ", buf).strip()


_JUNK_RE = re.compile(r"^[\s\d\W_]+$")


def _clean_title(raw: str) -> str:
    """Titles in PDF metadata are often junk: (anonymous), "Microsoft Word - xx.doc"."""
    t = (raw or "").strip()
    if not t or len(t) < 4:
        return ""
    if re.match(r"^\(?(anonymous|untitled|unknown)\)?$", t, re.I):
        return ""
    t = re.sub(r"^Microsoft Word\s*-\s*", "", t, flags=re.I)
    t = re.sub(r"\.(docx?|pdf|tex|indd|qxd|qxp)$", "", t, flags=re.I)
    t = t.strip()
    # Internal file names left by typesetting software, e.g. "2.5 comm mx", "jan04 final"
    if len(t) < 15 or not t[:1].isalpha():
        return ""
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < 3:
        return ""
    if sum(1 for w in words if re.fullmatch(r"[A-Za-z][A-Za-z\-’\']+", w)) < len(words) * 0.6:
        return ""
    return t


def _pdf_meta_title(path: str) -> str:
    """Try to read the title from the PDF metadata; give up quietly if it isn't there."""
    try:
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfparser import PDFParser
        with open(path, "rb") as fh:
            info = PDFDocument(PDFParser(fh)).info
        if info:
            val = info[0].get("Title")
            if isinstance(val, bytes):
                return val.decode("utf-8", "ignore").strip("\x00")
            return str(val or "")
    except Exception:
        pass
    return ""


_REFS_RE = re.compile(
    r"^\s*(?:\d{1,2}\.?|[IVX]{1,4}\.)?\s*(references(?:\s+and\s+notes)?|reference\s+list|"
    r"bibliography|literature\s+cited|works\s+cited|cited\s+literature)\s*:?\s*$",
    re.I)
# Some journals (e.g. Nature) have no "References" heading at all: the list just
# starts, in a smaller font, with "1. Surname, A. B., ...". Only entry 1 is matched,
# because a reference list always starts there.
_FIRST_REF_RE = re.compile(r"^\[?1[.\]]\s+[A-Z][A-Za-z'’\-]+,\s+(?:[A-Z]\.\s*-?)+")
# Sections that can follow the reference list. When one starts (as a heading, or
# run-in at the start of a paragraph: "Acknowledgements The authors ..."), the
# reference list is over.
_BACKMATTER_RE = re.compile(
    r"^\s*(?:[A-Z]\.?\s*)?(appendix|appendices|supplementary(?:\s+information)?|"
    r"supporting\s+information|supplemental|acknowledge?ments?|author\s+contributions?|"
    r"author\s+information|funding|declarations?|competing\s+interests?|"
    r"conflicts?\s+of\s+interest|ethics|additional\s+information|data\s+availability)\b", re.I)
_CAPTION_RE = re.compile(
    r"^(fig(ure)?|table|tab|chart|scheme|eq(uation)?|plate)\s*\.?\s*\d+", re.I)


def extract(path: str, drop_refs: bool = False, progress=None,
            page_range: tuple | None = None,
            drop_fragments: bool = False) -> Document:
    """Parse a PDF and return its text blocks in reading order.

    progress:       optional callback progress(done_pages, total_pages).
    page_range:     (first, last), 1-based and inclusive; None means the whole file.
                    Pages outside the range are not laid out.
    drop_refs:      skip the reference list (resumes at an appendix/supplementary heading).
                    If kept, reference blocks have section == "References" and can be
                    filtered later.
    drop_fragments: drop stray figure labels (non-heading blocks under 4 words that
                    aren't captions). Used when building retrieval chunks.
    """
    def in_range(idx0: int) -> bool:      # idx0 is 0-based
        if not page_range:
            return True
        a, b = page_range
        return (a is None or idx0 + 1 >= a) and (b is None or idx0 + 1 <= b)

    doc = Document()
    raw_pages: list[tuple[float, list[list[dict]]]] = []  # per page: (height, regions -> lines)

    try:
        from pdfminer.pdfpage import PDFPage
        with open(path, "rb") as fh:
            total_pages = sum(1 for _ in PDFPage.get_pages(fh))
    except Exception:
        total_pages = 0
    doc.title = _clean_title(_pdf_meta_title(path))

    if total_pages and page_range:
        wanted = [i for i in range(total_pages) if in_range(i)]
        layouts = zip(wanted, _pdfminer_backend.iter_page_layouts_boxes(
            path, page_numbers=set(wanted)))
        raw_pages = [(0.0, []) for _ in range(total_pages)]
    else:
        wanted = None
        layouts = enumerate(_pdfminer_backend.iter_page_layouts_boxes(path))

    done = 0
    for i, layout in layouts:
        if not in_range(i):
            raw_pages.append((0.0, []))
            continue
        boxes, width, height = _pdfminer_backend.page_boxes(layout)
        top_cut = height * 0.055
        bot_cut = height * 0.945
        boxes = [b for b in boxes
                 if b["bottom"] > top_cut and b["top"] < bot_cut]
        regions = [[ln for b in group for ln in b["lines"]]
                   for group in _order_boxes(boxes)]
        if wanted is None:
            raw_pages.append((height, regions))
        else:
            raw_pages[i] = (height, regions)
        done += 1
        if progress:
            progress(done, len(wanted) if wanted is not None else max(total_pages, i + 1))
    doc.n_pages = max(total_pages, len(raw_pages))

    # Heads/footers are filtered only after every page is read: this needs the
    # body font size of the whole document and repetition across pages.
    body_size = _body_size([ln for _, regions in raw_pages for lines in regions for ln in lines])
    per_page_lines = _filter_running_heads(raw_pages, body_size)
    for pno, ((height, regions), kept) in enumerate(zip(raw_pages, per_page_lines), start=1):
        if height and not any(kept):
            doc.empty_pages.append(pno)
    kept_lines = [ln for regions in per_page_lines for lines in regions for ln in lines]
    if kept_lines:
        body_size = _body_size(kept_lines)
    doc.body_size = body_size

    section = ""
    in_refs = False
    for pno, page_regions in enumerate(per_page_lines, start=1):
        for lines in page_regions:
            for text, kind, size in _group_lines(lines, body_size):
                if _JUNK_RE.match(text) or len(text) < 2:
                    continue
                clean = _normalize(text)
                if not clean:
                    continue
                if _REFS_RE.match(clean):
                    # Regardless of whether it was detected as a heading (some journals
                    # set the References heading in body size and weight)
                    kind, in_refs, section = "heading", True, "References"
                elif not in_refs and size < body_size and _FIRST_REF_RE.match(clean):
                    # A reference list with no heading
                    in_refs, section = True, "References"
                elif in_refs and (m := _BACKMATTER_RE.match(clean)):
                    in_refs = False
                    section = clean if kind == "heading" else m.group(1).capitalize()
                elif kind == "heading" and not in_refs:
                    section = clean
                if in_refs and drop_refs:
                    continue
                if drop_fragments and kind != "heading" and len(clean.split()) < 4 \
                        and not _CAPTION_RE.match(clean):
                    continue
                doc.blocks.append(Block(text=clean, page=pno, kind=kind, size=size,
                                        section=section))

    _mark_title(doc, body_size)
    doc.blocks = _merge_continuations(doc.blocks)
    return doc


def _mark_title(doc: Document, body_size: float) -> None:
    """Find the paper title: the largest heading on page 1, in display-size type.

    PDF metadata often has no usable title, but the title is needed for citations.
    The title is not a section, so blocks before the first real heading (usually
    the abstract) get an empty section instead of the title text.
    """
    candidates = [b for b in doc.blocks
                  if b.page == 1 and b.kind == "heading" and b.size >= body_size * 1.5]
    if not candidates:
        return
    title = max(candidates, key=lambda b: b.size)
    title.kind = "title"
    for b in doc.blocks:
        if b.section == title.text:
            b.section = ""
    if not doc.title:
        doc.title = title.text


def _merge_continuations(blocks: list[Block]) -> list[Block]:
    """Join paragraphs that were cut across a column or page break.

    This is deliberately conservative: blocks are merged only if the previous one
    is clearly unfinished (it doesn't end with sentence punctuation, optionally
    followed by a footnote number) and the next one starts lowercase. Otherwise
    they stay separate -- a wrong merge pulls a whole page of text onto the
    previous page, which is much worse than one extra split.

    After a cross-page merge, `page` stays the first page and `page_end` records
    the last, so citations can show a page range.
    """
    out: list[Block] = []
    for blk in blocks:
        if out:
            prev = out[-1]
            same_kind = prev.kind == blk.kind == "para"
            prev_text = prev.text.rstrip()
            prev_open = not _ENDS_SENTENCE.search(prev_text)
            starts_lower = bool(re.match(r"^[a-z,;)]", blk.text))
            hyphen_open = bool(re.search(r"[A-Za-z][-‐‑]$", prev_text))
            if same_kind and prev.section == blk.section and abs(prev.size - blk.size) < 0.4 and \
                    ((prev_open and starts_lower) or hyphen_open):
                prev.text = _join_lines([prev.text, blk.text])
                prev.page_end = max(prev.page_end, blk.page_end)
                continue
        out.append(blk)
    return out
