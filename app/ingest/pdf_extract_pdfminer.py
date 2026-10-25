"""PDF extraction backend built on pdfminer.six.

pdfminer does its own layout analysis and groups characters into text boxes
by proximity. These boxes never span a column gutter, so they make a much
more reliable unit for ordering than guessing from coordinates. Every
character also carries an exact font size and position, which is how
citation superscripts can be detected as "smaller and raised".

Only two functions are public: iter_page_layouts_boxes (yields one layout per
page) and page_boxes (splits a layout into text boxes). Ordering and paragraph
grouping happen in pdf_extract.
"""

from __future__ import annotations

import re
import statistics

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTAnno, LTChar, LTTextContainer, LTTextLine


def _walk_chars(element):
    stack = [element]
    while stack:
        el = stack.pop()
        if isinstance(el, LTChar):
            yield el
        elif hasattr(el, "__iter__"):
            stack.extend(list(el))


_SUP_CHARS = set("0123456789,.-–—*†‡§¶")


def _line_from_line(line_obj, height: float) -> dict | None:
    """Convert a pdfminer text line into our line dict, dropping citation superscripts.

    Citation markers in academic papers (recognition1–4, molecules8, disease12,13)
    end up glued to the preceding word in plain text. They add noise to keyword
    search and to the text the model reads, so they are removed here: characters
    that are smaller than the line's body size, raised above its baseline, and
    made only of digits and punctuation.

    pdfminer's inferred spaces (LTAnno) are kept while iterating. Some PDFs have
    no space characters at all and encode word gaps purely as positions, so
    rebuilding the text from characters alone would run every word together.
    """
    items = list(line_obj)
    chars = [c for c in items if isinstance(c, LTChar) and c.get_text().strip()]
    if not chars:
        return None

    sizes = [float(getattr(c, "size", 10.0) or 10.0) for c in chars]
    body_size = statistics.median(sizes)
    baseline = statistics.median([c.y0 for c in chars
                                  if abs(float(getattr(c, "size", 10.0)) - body_size) < 0.6]
                                 or [c.y0 for c in chars])

    # First flag every character that looks like a superscript
    sup_flag = {}
    for c in chars:
        size = float(getattr(c, "size", 10.0) or 10.0)
        raised = c.y0 - baseline
        sup_flag[id(c)] = (size <= body_size * 0.86
                           and raised >= body_size * 0.10
                           and c.get_text() in _SUP_CHARS)
    # A lone raised symbol is not a citation (it may be maths);
    # only runs that contain at least one digit count
    groups: list[list] = []
    for c in chars:
        if sup_flag[id(c)]:
            if groups and groups[-1][-1] is not None and c.x0 - groups[-1][-1].x1 < body_size:
                groups[-1].append(c)
            else:
                groups.append([c])
    for g in groups:
        if not any(ch.get_text().isdigit() for ch in g):
            for ch in g:
                sup_flag[id(ch)] = False

    parts = []
    kept = []
    for obj in items:
        if isinstance(obj, LTAnno):
            parts.append(obj.get_text())
        elif isinstance(obj, LTChar):
            if sup_flag.get(id(obj)):
                continue
            parts.append(obj.get_text())
            if obj.get_text().strip():
                kept.append(obj)

    text = re.sub(r"\s+", " ", "".join(parts)).strip()
    if not text or not kept:
        return None

    names = [str(getattr(c, "fontname", "")) for c in kept]
    bold = sum(1 for n in names if re.search(r"bold|black|heavy", n, re.I))
    ksizes = [float(getattr(c, "size", 10.0) or 10.0) for c in kept]
    return {
        "text": text,
        "x0": min(c.x0 for c in kept),
        "x1": max(c.x1 for c in kept),
        "top": height - max(c.y1 for c in kept),
        "bottom": height - min(c.y0 for c in kept),
        "_size": round(statistics.median(ksizes), 1),
        "_bold": bold > len(names) / 2 if names else False,
    }


def _line_from_chars(chars, height: float, text: str | None = None) -> dict | None:
    """Fallback when there is no line object: build the line straight from characters."""
    items = [c for c in chars if isinstance(c, LTChar)]
    if not items:
        return None
    if text is None:
        text = "".join(c.get_text() for c in items)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    sizes = [float(getattr(c, "size", 10.0) or 10.0)
             for c in items if not c.get_text().isspace()]
    names = [str(getattr(c, "fontname", "")) for c in items if not c.get_text().isspace()]
    bold = sum(1 for n in names if re.search(r"bold|black|heavy", n, re.I))
    return {
        "text": text,
        "x0": min(c.x0 for c in items),
        "x1": max(c.x1 for c in items),
        "top": height - max(c.y1 for c in items),
        "bottom": height - min(c.y0 for c in items),
        "_size": round(statistics.median(sizes), 1) if sizes else 10.0,
        "_bold": bold > len(names) / 2 if names else False,
    }


def page_boxes(layout) -> tuple[list[dict], float, float]:
    """Return (text boxes, page width, page height).

    Each text box comes from pdfminer's layout analysis, which clusters text by
    proximity and never crosses a column gutter. That makes boxes a more
    reliable unit for ordering than raw coordinates.
    """
    width = float(layout.width)
    height = float(layout.height)
    boxes: list[dict] = []
    for element in layout:
        if not isinstance(element, LTTextContainer):
            continue
        lines = []
        for child in element:
            if isinstance(child, LTTextLine):
                ln = _line_from_line(child, height)
                if ln:
                    lines.append(ln)
        if not lines:
            ln = _line_from_chars(list(_walk_chars(element)), height, element.get_text())
            if ln:
                lines = [ln]
        if not lines:
            continue
        lines.sort(key=lambda l: (l["top"], l["x0"]))
        for ln in lines:
            # Number of lines in the enclosing box: heads and footers almost always
            # sit in a box of their own, while body lines belong to multi-line boxes
            ln["_nbox"] = len(lines)
        boxes.append({
            "x0": min(l["x0"] for l in lines),
            "x1": max(l["x1"] for l in lines),
            "top": min(l["top"] for l in lines),
            "bottom": max(l["bottom"] for l in lines),
            "lines": lines,
        })
    return boxes, width, height


def iter_page_layouts_boxes(path: str, page_numbers=None):
    """Layout iterator for page_boxes, with pdfminer's layout analysis turned on."""
    laparams = LAParams(detect_vertical=False, boxes_flow=0.5)
    return extract_pages(path, laparams=laparams, page_numbers=page_numbers)
