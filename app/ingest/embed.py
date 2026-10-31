"""Embed every chunk in data/index/chunks.jsonl and save the vectors for retrieval.

Run after app.ingest.build, whenever the chunks change:
    python -m app.ingest.embed

The first run downloads the embedding model (about 130 MB) from Hugging Face.
"""

import time

from app.retrieval import INDEX_DIR, SentenceTransformerEmbedder, build_embeddings, chunk_search_text, load_chunks


def main() -> None:
    embedder = SentenceTransformerEmbedder()
    texts = [chunk_search_text(c) for c in load_chunks()]

    t0 = time.perf_counter()
    n, dims = build_embeddings(embedder)
    print(f"\nEmbedded {n} chunks with {embedder.name} ({dims} dims) "
          f"in {time.perf_counter() - t0:.0f}s -> {INDEX_DIR / 'embeddings.npy'}")

    # The model reads at most max_seq_length tokens; anything after that is ignored
    # without warning. BM25 still sees the whole chunk.
    limit = embedder.model.max_seq_length
    truncated = embedder.count_truncated(texts)
    print(f"Chunks longer than the model's {limit}-token limit: {truncated} of {n} "
          f"({truncated / n:.0%}); their tails are not embedded")


if __name__ == "__main__":
    main()
