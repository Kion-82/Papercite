"""Hybrid retrieval over paper chunks: dense vectors + BM25, merged with reciprocal rank fusion.

Dense (embedding) search finds passages that mean the same thing in different
words. BM25 keyword search finds exact terms that embeddings handle poorly:
species names (Aricia agestis), gene names (COI), acronyms (CNN, XAI) and
numbers. RRF combines the two rankings without having to calibrate their
scores against each other.

Usage (from the project root):
    python -m app.retrieval "how many specimens were imaged?"
    python -m app.retrieval "COI barcode" --mode bm25 --k 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

INDEX_DIR = Path("data/index")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained to embed short queries with this instruction in front,
# and passages without it. Leaving it out lowers retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60          # standard constant from the original RRF paper (Cormack et al., 2009)
CANDIDATES = 50     # how deep each ranking goes before fusion

# Common English words carry no signal for BM25. Kept short on purpose:
# words like "not", "no" or "between" can matter in scientific questions.
_STOPWORDS = set("""a an and are as at be by for from has have in into is it its of on or
that the their then there these this those to was were which with""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; hyphenated terms are also kept whole.

    No stemming: it would merge "specimens" and "specimen" (good) but also
    mangle species and gene names (bad). Whether stemming helps can be tested
    later with the eval set.
    """
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    hyphenated = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", text)   # "f1-score", "co-occurrence"
    return [w for w in words + hyphenated if w not in _STOPWORDS]


def chunk_search_text(chunk: dict) -> str:
    """The text that gets indexed: paper title and section, then the chunk itself.

    The header gives every chunk context it may not contain on its own, e.g. a
    Methods paragraph that never repeats the species studied in the title.
    """
    return f"{chunk['paper_title']}\n{chunk['section']}\n" + \
        " ".join(p["text"] for p in chunk["passages"])


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Local embedding model. Loaded lazily, because importing torch takes seconds."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True makes every vector unit length, so cosine
        # similarity is just a dot product
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=True).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(QUERY_PREFIX + text, normalize_embeddings=True).astype(np.float32)

    def count_truncated(self, texts: list[str]) -> int:
        """How many texts are longer than the model's input limit (the rest is silently cut off)."""
        limit = self.model.max_seq_length
        # verbose=False: we tokenize over-long texts on purpose here, so skip the length warning
        return sum(len(self.model.tokenizer(t, verbose=False)["input_ids"]) > limit for t in texts)


def _fingerprint(chunks: list[dict]) -> str:
    """Hash of chunk ids and texts, to detect embeddings that no longer match chunks.jsonl."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["chunk_id"].encode())
        h.update(chunk_search_text(c).encode())
    return h.hexdigest()[:16]


def load_chunks(index_dir: Path = INDEX_DIR) -> list[dict]:
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m app.ingest.build")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_embeddings(embedder: Embedder, index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    """Embed every chunk and save the vectors next to chunks.jsonl. Returns (n_chunks, dims)."""
    chunks = load_chunks(index_dir)
    vectors = embedder.embed_documents([chunk_search_text(c) for c in chunks])
    np.save(index_dir / "embeddings.npy", vectors)
    meta = {"model": embedder.name, "n_chunks": len(chunks), "dims": int(vectors.shape[1]),
            "fingerprint": _fingerprint(chunks)}
    (index_dir / "embeddings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return len(chunks), int(vectors.shape[1])


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

@dataclass
class Hit:
    chunk: dict
    rank: int                  # 1-based, after fusion
    score: float               # RRF score in hybrid mode; cosine or BM25 score otherwise
    vector_rank: int | None    # where each retriever placed this chunk (None = not in its top CANDIDATES)
    bm25_rank: int | None


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank of d).

    Only ranks are used, never raw scores, so a cosine similarity of 0.8 and a
    BM25 score of 14.2 never have to be put on the same scale. A large k (60)
    flattens the curve, so a chunk ranked well by both retrievers beats one
    ranked first by only one of them.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda t: -t[1])


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR, embedder: Embedder | None = None):
        self.chunks = load_chunks(index_dir)
        self.embedder = embedder or SentenceTransformerEmbedder()

        meta_path = index_dir / "embeddings.json"
        if not meta_path.exists():
            raise SystemExit("Embeddings not built yet. Run: python -m app.ingest.embed")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Stale index check: vectors are matched to chunks by row number, so if
        # chunks.jsonl was rebuilt since, every result would point at the wrong text.
        if meta["fingerprint"] != _fingerprint(self.chunks) or meta["model"] != self.embedder.name:
            raise SystemExit("Embeddings are out of date with chunks.jsonl (or a different model). "
                             "Run: python -m app.ingest.embed")
        self.vectors = np.load(index_dir / "embeddings.npy")

        # BM25 is cheap to build for a few thousand chunks, so it isn't saved to disk
        self.bm25 = BM25Okapi([tokenize(chunk_search_text(c)) for c in self.chunks])
        self.paper_ids = sorted({c["paper_id"] for c in self.chunks})

    def _vector_ranking(self, query: str, allowed: np.ndarray) -> tuple[list[int], np.ndarray]:
        sims = self.vectors @ self.embedder.embed_query(query)   # cosine, since vectors are unit length
        sims = np.where(allowed, sims, -np.inf)
        order = np.argsort(-sims)[:CANDIDATES]
        return [int(i) for i in order if allowed[i]], sims

    def _bm25_ranking(self, query: str, allowed: np.ndarray) -> tuple[list[int], np.ndarray]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        scores = np.where(allowed, scores, -np.inf)
        order = np.argsort(-scores)[:CANDIDATES]
        # A chunk with BM25 score 0 shares no query term; it isn't a keyword match at all
        return [int(i) for i in order if allowed[i] and scores[i] > 0], scores

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               paper_id: str | None = None) -> list[Hit]:
        """Top-k chunks for a query. mode: "hybrid" | "vector" | "bm25"."""
        if mode not in ("hybrid", "vector", "bm25"):
            raise ValueError(f"unknown mode {mode!r}")
        if paper_id is not None and paper_id not in self.paper_ids:
            raise ValueError(f"unknown paper_id {paper_id!r}")
        allowed = np.array([paper_id is None or c["paper_id"] == paper_id for c in self.chunks])

        vec_rank, sims = self._vector_ranking(query, allowed) if mode != "bm25" else ([], None)
        bm_rank, bm_scores = self._bm25_ranking(query, allowed) if mode != "vector" else ([], None)
        vec_pos = {idx: r for r, idx in enumerate(vec_rank, start=1)}
        bm_pos = {idx: r for r, idx in enumerate(bm_rank, start=1)}

        if mode == "hybrid":
            ranked = rrf([vec_rank, bm_rank])
        elif mode == "vector":
            ranked = [(i, float(sims[i])) for i in vec_rank]
        else:
            ranked = [(i, float(bm_scores[i])) for i in bm_rank]

        return [Hit(chunk=self.chunks[i], rank=r, score=s,
                    vector_rank=vec_pos.get(i), bm25_rank=bm_pos.get(i))
                for r, (i, s) in enumerate(ranked[:k], start=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the paper index.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    parser.add_argument("--paper", default=None, help="restrict to one paper_id, e.g. lecun2015")
    args = parser.parse_args()

    retriever = Retriever()
    for h in retriever.search(args.query, k=args.k, mode=args.mode, paper_id=args.paper):
        c = h.chunk
        pages = f"p{c['page_start']}" + (f"-{c['page_end']}" if c["page_end"] != c["page_start"] else "")
        print(f"#{h.rank}  {c['chunk_id']}  {pages}  [{c['section'][:40]}]  "
              f"score={h.score:.4f}  vector#{h.vector_rank}  bm25#{h.bm25_rank}")
        print(f"    {c['passages'][0]['text'][:200]}\n")


if __name__ == "__main__":
    main()
