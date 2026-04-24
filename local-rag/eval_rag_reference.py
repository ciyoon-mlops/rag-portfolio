"""
Evaluate local RAG against gold Q&A in .cursor/rules/reference.json.

Reuses the same stack as rag_sayno_full.py (embeddings, FAISS, RetrievalQA).
Run from project root after building faiss_index and with Ollama serving the LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from rag_sayno_full import embeddings, qa_chain

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE = SCRIPT_DIR / "reference.json"


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _answer_similarity(ref_text: str, gen_text: str) -> float:
    """Cosine similarity between embedding vectors of reference vs generated answers."""
    ref_vec = embeddings.embed_query(ref_text)
    gen_vec = embeddings.embed_query(gen_text)
    return _cosine(ref_vec, gen_vec)


def _load_tests(reference_path: Path) -> list[dict]:
    with reference_path.open(encoding="utf-8") as f:
        data = json.load(f)
    perf = data.get("performance_evaluation") or {}
    tests = perf.get("test_questions") or []
    if not tests:
        raise ValueError(f"No test_questions in {reference_path}")
    return tests


def run_eval(
    reference_path: Path,
    top_k_sources: int,
    verbose: bool,
) -> list[dict]:
    results: list[dict] = []
    tests = _load_tests(reference_path)

    for i, item in enumerate(tests, start=1):
        question = item["question"]
        ref_answer = item.get("generated_answer") or ""
        ref_perf = item.get("performance") or {}

        t0 = time.perf_counter()
        out = qa_chain.invoke({"query": question})
        latency_s = time.perf_counter() - t0

        gen_answer = (out.get("result") or "").strip()
        source_docs = out.get("source_documents") or []

        sim = _answer_similarity(ref_answer, gen_answer) if ref_answer and gen_answer else 0.0

        snippets = []
        for doc in source_docs[:top_k_sources]:
            text = getattr(doc, "page_content", "") or ""
            snippets.append(text[:400])

        row = {
            "index": i,
            "question": question,
            "latency_s": round(latency_s, 3),
            "answer_cosine_vs_reference": round(sim, 4),
            "reference_performance": ref_perf,
            "generated_answer_preview": gen_answer[:500] + ("…" if len(gen_answer) > 500 else ""),
        }
        results.append(row)

        print(f"\n--- [{i}/{len(tests)}] ---")
        print(f"Q: {question}")
        print(f"latency: {row['latency_s']}s | answer_cosine_vs_ref: {row['answer_cosine_vs_reference']}")
        print(f"reference.json performance: {ref_perf}")
        if verbose:
            print("\n[reference answer]\n", ref_answer)
            print("\n[RAG answer]\n", gen_answer)
            print("\n[top retrieved snippets]")
            for j, snip in enumerate(snippets, 1):
                print(f"  {j}. {snip[:300]}…" if len(snip) > 300 else f"  {j}. {snip}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG vs reference.json gold answers.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Path to reference.json (default: .cursor/rules/reference.json)",
    )
    parser.add_argument(
        "--top-k-preview",
        type=int,
        default=3,
        help="Number of retrieved source snippets to show in verbose mode.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print full answers and retrieval snippets.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="If set, write a JSON array of summary rows to this path.",
    )
    args = parser.parse_args()

    if not args.reference.is_file():
        print(f"Reference file not found: {args.reference}", file=sys.stderr)
        return 1

    rows = run_eval(args.reference, args.top_k_preview, args.verbose)

    sims = [r["answer_cosine_vs_reference"] for r in rows]
    lats = [r["latency_s"] for r in rows]
    print("\n======== SUMMARY ========")
    print(f"cases: {len(rows)}")
    print(f"mean answer_cosine_vs_reference: {np.mean(sims):.4f}")
    print(f"min / max: {np.min(sims):.4f} / {np.max(sims):.4f}")
    print(f"mean latency_s: {np.mean(lats):.3f}")

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
