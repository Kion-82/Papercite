"""Parse and chunk every PDF in data/papers/ and write all chunks to one JSONL file.

Usage (from the project root, with .venv active):
    python -m app.ingest.build

Each PDF is parsed in its own child process. PDFs are untrusted input: a
malformed file can make a parsing library crash the whole interpreter (an
access violation, which Python cannot catch) or loop forever. Isolating each
file means one bad PDF is reported as failed and the build carries on.
"""

import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

from app.ingest.chunk import chunk_pdf

PAPERS_DIR = Path("data/papers")
OUT_PATH = Path("data/index/chunks.jsonl")
TIMEOUT_S = 120  # per PDF; the slowest normal paper takes a few seconds

# Windows exit codes for crashes, to make failure messages readable
_CRASH_CODES = {
    -1073741819: "access violation (0xC0000005)",
    -1073741571: "stack overflow (0xC00000FD)",
    3221225477: "access violation (0xC0000005)",
    3221225725: "stack overflow (0xC00000FD)",
    2147483651: "breakpoint / heap corruption detected (0x80000003)",
}


def _worker(path: str, conn) -> None:
    """Runs in the child process: chunk one PDF and send the result back."""
    import faulthandler
    faulthandler.enable()  # if this child crashes, print where (to the console)
    try:
        chunks = chunk_pdf(path)
        conn.send(("ok", [c.to_dict() for c in chunks]))
    except Exception as exc:
        conn.send(("error", repr(exc)))
    finally:
        conn.close()


def chunk_isolated(pdf: Path) -> tuple[str, object]:
    """Chunk one PDF in a child process. Returns ("ok", chunks) or ("error", message)."""
    ctx = mp.get_context("spawn")  # the only start method on Windows; use it everywhere for consistency
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(str(pdf), child_conn))
    proc.start()
    child_conn.close()  # the parent only reads; closing our copy lets us see EOF if the child dies

    try:
        if not parent_conn.poll(TIMEOUT_S):
            proc.terminate()
            proc.join()
            return "error", f"timed out after {TIMEOUT_S}s"
        try:
            return parent_conn.recv()
        except EOFError:  # the child died before sending anything
            proc.join()
            code = proc.exitcode
            return "error", f"parser crashed: {_CRASH_CODES.get(code, f'exit code {code}')}"
    finally:
        proc.join(timeout=5)
        parent_conn.close()


def main() -> None:
    pdfs = sorted(p for p in PAPERS_DIR.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PAPERS_DIR}/")
    print(f"Found {len(pdfs)} PDFs in {PAPERS_DIR}/\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_words: list[int] = []
    n_chunks = 0
    failed: list[str] = []
    print(f"{'paper':<22}{'pages':>6}{'chunks':>8}{'figure':>8}{'median words':>14}  title")
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for pdf in pdfs:
            # Print the name before parsing (flushed), so a hang shows which file it is
            print(f"{pdf.stem:<22}", end="", flush=True)
            t0 = time.perf_counter()
            status, payload = chunk_isolated(pdf)
            if status != "ok" and str(payload).startswith("parser crashed"):
                # Interpreter crashes were rare and random on one dev machine (not tied to any
                # file), so one retry is a cheap safety net. Parse errors and timeouts are
                # deterministic and are not retried.
                print("(crashed, retrying) ", end="", flush=True)
                status, payload = chunk_isolated(pdf)
            if status != "ok":
                print(f"  FAILED: {payload}")
                failed.append(pdf.name)
                continue
            chunks = payload
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            words = [c["n_words"] for c in chunks if c["kind"] == "text"]
            all_words += words
            n_chunks += len(chunks)
            pages = max((c["page_end"] for c in chunks), default=0)
            n_fig = sum(c["kind"] == "figure" for c in chunks)
            med = f"{statistics.median(words):.0f}" if words else "-"
            title = chunks[0]["paper_title"] if chunks else ""
            print(f"{pages:>6}{len(chunks):>8}{n_fig:>8}{med:>14}  "
                  f"{title[:50]}  ({time.perf_counter() - t0:.1f}s)")

    print(f"\n{n_chunks} chunks from {len(pdfs) - len(failed)} of {len(pdfs)} PDFs -> {OUT_PATH}")
    if all_words:
        print(f"Text chunk words: min {min(all_words)}, median {statistics.median(all_words):.0f}, "
              f"max {max(all_words)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
