"""Retrieval evaluation: recall@k and MRR for each search mode, on hand-labelled questions.

No LLM is called, so this is free and fast; run it after every change to
parsing, chunking or retrieval.

Usage (from the project root):
    python -m evals.eval_retrieval
    python -m evals.eval_retrieval --k 5 --show-misses
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from app.retrieval import Hit, Retriever

QUESTIONS_PATH = Path("evals/questions.jsonl")
RESULTS_DIR = Path("evals/results")
MODES = ["vector", "bm25", "hybrid"]
KS = [1, 3, 5, 10]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """Questions with at least one gold source. Unanswerable and injection
    questions have no gold page, so they don't count towards retrieval recall."""
    with path.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    for q in questions:
        for s in q.get("sources", []):
            assert {"paper_id", "pages"} <= s.keys(), f"{q['id']}: each source needs paper_id and pages"
    return [q for q in questions if q.get("sources")]


def source_found(hit: Hit, source: dict) -> bool:
    """A hit counts for a gold source if it is from that paper and one of its
    passages is on one of the gold pages."""
    chunk = hit.chunk
    return chunk["paper_id"] == source["paper_id"] and \
        any(p["page"] in source["pages"] for p in chunk["passages"])


def score_question(hits: list[Hit], sources: list[dict]) -> dict:
    """Per-question metrics.

    recall@k: fraction of gold sources found in the top k. For a single-source
              question this is 0 or 1; a question that needs two papers scores
              0.5 if only one of them is retrieved.
    first_rank: rank of the first hit that matches any gold source (None if not retrieved).
    """
    result = {}
    for k in KS:
        top = hits[:k]
        result[f"recall@{k}"] = sum(any(source_found(h, s) for h in top) for s in sources) / len(sources)
    result["first_rank"] = next((h.rank for h in hits if any(source_found(h, s) for s in sources)), None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="k used for the headline recall and misses")
    parser.add_argument("--show-misses", action="store_true",
                        help="print what was retrieved for questions hybrid search missed")
    args = parser.parse_args()

    questions = load_questions()
    retriever = Retriever()
    # A typo in a gold paper_id (or a renamed PDF) would silently score 0, so fail loudly instead
    unknown = sorted({s["paper_id"] for q in questions for s in q["sources"]} - set(retriever.paper_ids))
    if unknown:
        raise SystemExit(f"Gold sources refer to papers not in the index: {unknown}")
    depth = max(KS)
    print(f"{len(questions)} questions with gold sources, {len(retriever.chunks)} chunks\n")

    per_question: dict[str, dict[str, dict]] = {q["id"]: {} for q in questions}
    retrieved: dict[str, list[Hit]] = {}
    for mode in MODES:
        for q in questions:
            hits = retriever.search(q["question"], k=depth, mode=mode)
            per_question[q["id"]][mode] = score_question(hits, q["sources"])
            if mode == "hybrid":
                retrieved[q["id"]] = hits

    # Summary: mean recall@k per mode, and MRR (mean of 1/first_rank, 0 if not retrieved)
    summary = {}
    header = "mode    " + "".join(f"{'recall@' + str(k):>11}" for k in KS) + f"{'MRR':>8}"
    print(header)
    for mode in MODES:
        rows = [per_question[q["id"]][mode] for q in questions]
        summary[mode] = {f"recall@{k}": statistics.mean(r[f"recall@{k}"] for r in rows) for k in KS}
        summary[mode]["mrr"] = statistics.mean(1 / r["first_rank"] if r["first_rank"] else 0 for r in rows)
        print(f"{mode:<8}" + "".join(f"{summary[mode][f'recall@{k}']:>11.2f}" for k in KS)
              + f"{summary[mode]['mrr']:>8.2f}")

    # Per question: rank of the first correct chunk in each mode. This is where
    # the interesting cases show up, e.g. found by BM25 but missed by vectors.
    print(f"\n{'id':<6}{'type':<8}" + "".join(f"{m:>8}" for m in MODES) + "   question")
    for q in questions:
        ranks = [per_question[q["id"]][m]["first_rank"] for m in MODES]
        cells = "".join(f"{(str(r) if r else '-'):>8}" for r in ranks)
        print(f"{q['id']:<6}{q.get('type', ''):<8}{cells}   {q['question'][:60]}")
    print("(numbers = rank of the first chunk from a gold page; - = not in top 10)")

    if args.show_misses:
        for q in questions:
            if per_question[q["id"]]["hybrid"][f"recall@{args.k}"] < 1:
                gold = ", ".join(f"{s['paper_id']} p{s['pages']}" for s in q["sources"])
                print(f"\nMISS {q['id']}: {q['question']}\n  gold: {gold}")
                for h in retrieved[q["id"]][:args.k]:
                    c = h.chunk
                    pages = sorted({p["page"] for p in c["passages"]})
                    print(f"  #{h.rank} {c['chunk_id']} pages {pages}  {c['passages'][0]['text'][:90]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"n_questions": len(questions), "n_chunks": len(retriever.chunks),
                               "summary": summary, "per_question": per_question}, indent=2),
                   encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
