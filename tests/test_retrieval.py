"""Tests for hybrid retrieval. A fake embedder replaces the real model, so no download is needed."""

import json
import zlib

import numpy as np
import pytest

from app.retrieval import Retriever, build_embeddings, rrf, tokenize

DIMS = 256


class FakeEmbedder:
    """Bag-of-words vectors: each token adds 1 to a fixed slot. Deterministic and instant.

    Texts sharing words get similar vectors, which is all the tests need. The
    real model's quality is measured by the eval set, not by unit tests.
    """
    name = "fake-bow"

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(DIMS, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % DIMS] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text):
        return self._embed(text)


def make_chunk(paper_id: str, n: int, section: str, text: str) -> dict:
    return {"chunk_id": f"{paper_id}#c{n:03d}", "paper_id": paper_id,
            "paper_title": f"Paper {paper_id}", "section": section, "kind": "text",
            "passages": [{"text": text, "page": 1}], "page_start": 1, "page_end": 1,
            "n_words": len(text.split())}


CHUNKS = [
    make_chunk("smith2020", 0, "Methods", "We photographed 412 limpet shells with a macro lens."),
    make_chunk("smith2020", 1, "Results", "The classifier separated the two clades with high accuracy."),
    make_chunk("jones2021", 0, "Methods", "Butterflies were sampled across twelve sites in Wales."),
    make_chunk("jones2021", 1, "Results", "Aricia agestis expanded its range northwards after 1990."),
    make_chunk("lee2019", 0, "Methods", "DNA barcodes used the COI gene region for every specimen."),
]


@pytest.fixture
def index_dir(tmp_path):
    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in CHUNKS:
            f.write(json.dumps(c) + "\n")
    build_embeddings(FakeEmbedder(), tmp_path)
    return tmp_path


@pytest.fixture
def retriever(index_dir):
    return Retriever(index_dir, embedder=FakeEmbedder())


def test_rrf_rewards_agreement():
    # Chunk 2 is 2nd in both rankings; chunks 0 and 1 are 1st in only one ranking each.
    fused = rrf([[0, 2, 3], [1, 2, 4]])
    assert fused[0][0] == 2
    assert fused[0][1] == pytest.approx(2 / 62)


def test_bm25_finds_exact_species_name(retriever):
    hits = retriever.search("Aricia agestis", k=1, mode="bm25")
    assert hits[0].chunk["chunk_id"] == "jones2021#c001"


def test_bm25_ignores_chunks_without_query_terms(retriever):
    hits = retriever.search("COI", k=5, mode="bm25")
    assert [h.chunk["chunk_id"] for h in hits] == ["lee2019#c000"]


@pytest.mark.parametrize("mode", ["hybrid", "vector", "bm25"])
def test_modes_return_ranked_hits(retriever, mode):
    hits = retriever.search("how many limpet shells were photographed", k=3, mode=mode)
    assert hits[0].chunk["chunk_id"] == "smith2020#c000"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_hybrid_reports_both_component_ranks(retriever):
    top = retriever.search("limpet shells photographed", k=1)[0]
    assert top.vector_rank == 1 and top.bm25_rank == 1


def test_paper_filter(retriever):
    hits = retriever.search("methods results", k=5, paper_id="jones2021")
    assert hits and {h.chunk["paper_id"] for h in hits} == {"jones2021"}


def test_unknown_paper_is_an_error(retriever):
    with pytest.raises(ValueError):
        retriever.search("anything", paper_id="nobody1999")


def test_stale_embeddings_are_detected(index_dir):
    # Rebuild chunks.jsonl without re-embedding: the vectors no longer line up with the rows
    with (index_dir / "chunks.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(make_chunk("new2024", 0, "Intro", "A new chunk.")) + "\n")
    with pytest.raises(SystemExit, match="out of date"):
        Retriever(index_dir, embedder=FakeEmbedder())


def test_tokenize_keeps_hyphenated_terms_and_drops_stopwords():
    assert tokenize("The F1-score of the CNN") == ["f1", "score", "cnn", "f1-score"]
