"""Unit tests for chunking. They build Documents by hand, so no PDF is needed."""

from app.ingest.chunk import MAX_WORDS, OVERLAP_WORDS, chunk_document, split_sentences
from app.ingest.pdf_extract import Block, Document


def para(n_words: int, page: int, section: str, word: str = "word") -> Block:
    """A paragraph of n_words ending in a full stop."""
    text = " ".join([word] * (n_words - 1)) + " end."
    return Block(text=text, page=page, kind="para", size=10.0, section=section)


def make_doc(blocks: list[Block]) -> Document:
    return Document(blocks=blocks, n_pages=max(b.page for b in blocks),
                    title="Test paper", body_size=10.0)


def test_chunks_never_cross_sections():
    doc = make_doc([para(200, 1, "Methods"), para(200, 1, "Methods"),
                    para(200, 2, "Results"), para(200, 2, "Results")])
    chunks = chunk_document(doc, "test2020")
    assert [c.section for c in chunks] == ["Methods", "Results"]


def test_chunk_size_is_bounded():
    doc = make_doc([para(150, p, "Results") for p in range(1, 11)])
    chunks = chunk_document(doc, "test2020")
    assert len(chunks) > 1
    assert all(c.n_words <= MAX_WORDS + OVERLAP_WORDS for c in chunks)


def test_long_paragraph_is_split_by_sentence():
    sentence = "This sentence has exactly ten words in it, no more."
    block = Block(text=" ".join([sentence] * 80), page=3, section="Discussion")
    chunks = chunk_document(make_doc([block]), "test2020")
    assert len(chunks) >= 2
    assert all(c.n_words <= MAX_WORDS + OVERLAP_WORDS for c in chunks)


def test_pages_are_tracked_per_passage():
    doc = make_doc([para(150, 4, "Results"), para(150, 5, "Results")])
    (chunk,) = chunk_document(doc, "test2020")
    assert [p.page for p in chunk.passages] == [4, 5]
    assert (chunk.page_start, chunk.page_end) == (4, 5)


def sentences(n: int, page: int, section: str, word: str) -> Block:
    """A paragraph of n ten-word sentences: "Alpha alpha ... alpha." """
    sentence = " ".join([word.capitalize()] + [word] * 9) + "."
    return Block(text=" ".join([sentence] * n), page=page, section=section)


def test_overlap_repeats_end_of_previous_chunk():
    doc = make_doc([sentences(40, 1, "Results", "alpha"), sentences(40, 2, "Results", "beta")])
    first, second = chunk_document(doc, "test2020")
    overlap = second.passages[0]
    assert first.passages[-1].text.endswith(overlap.text)
    assert overlap.page == 1                               # keeps the page it came from
    assert 0 < len(overlap.text.split()) <= OVERLAP_WORDS  # whole sentences, within budget


def test_no_overlap_when_last_sentence_is_too_long():
    # One 400-word "sentence" can't fit in the overlap budget, so nothing is repeated
    doc = make_doc([para(400, 1, "Results", word="alpha"), para(400, 2, "Results", word="beta")])
    first, second = chunk_document(doc, "test2020")
    assert second.passages[0].text.startswith("beta")


def test_references_and_back_matter_are_skipped():
    doc = make_doc([para(200, 1, "Results"),
                    para(300, 2, "References"),
                    para(50, 2, "Acknowledgements")])
    chunks = chunk_document(doc, "test2020")
    assert {c.section for c in chunks} == {"Results"}


def test_figure_captions_are_separate_chunks():
    doc = make_doc([para(150, 1, "Results"),
                    Block(text="Figure 1 | Shell outlines of two clades.", page=1,
                          kind="caption", size=8.0, section="Results"),
                    Block(text="tiny axis label here", page=1, kind="caption",
                          size=6.0, section="Results"),
                    para(150, 1, "Results")])
    chunks = chunk_document(doc, "test2020")
    assert [c.kind for c in chunks] == ["text", "figure"]
    assert chunks[0].n_words == 300          # the caption doesn't split the text run
    assert "axis label" not in chunks[1].text  # figure labels are dropped


def test_author_list_and_affiliations_are_dropped():
    doc = make_doc([Block(text="Ann Smith, Bo Li & Cara Jones", page=1, section=""),
                    Block(text="Department of Zoology, University of Oxford, UK.", page=1, section=""),
                    para(150, 1, "")])
    (chunk,) = chunk_document(doc, "test2020")
    assert "Smith" not in chunk.text and "Oxford" not in chunk.text


def test_split_sentences_keeps_abbreviations():
    text = "Shells were measured (Smith et al. 2020). Size varied, e.g. by site. Fig. 2 shows this."
    assert split_sentences(text) == ["Shells were measured (Smith et al. 2020).",
                                     "Size varied, e.g. by site.",
                                     "Fig. 2 shows this."]
