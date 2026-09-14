"""RAG core: combines retrieval with Claude to produce an answer."""

from . import llm
from .retrieval import search


def build_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        if r["type"] == "record":
            m = r["metadata"]
            parts.append(
                f"- Parameter '{m['name']}' (used in {m['plugin_count']} plugin(s)): "
                f"{m['description'] or '(no description)'} — Category: {m['main_data_type']} / {m['data_type']}"
            )
        elif r["type"] == "audit":
            parts.append(f"[Privacy policy audit] {r['text']}")
        else:
            parts.append(f"[Analysis finding — {r['metadata']['filename']}]\n{r['text']}")
    return "\n\n".join(parts)


def answer(question: str, top_k: int = 5) -> dict:
    results = search(question, top_k=top_k)
    context = build_context(results)
    response_text = llm.ask(question, context)
    return {"answer": response_text, "sources": results}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What kinds of data are collected the most, and which are sensitive?"
    result = answer(question)
    print("QUESTION:", question)
    print("\nANSWER:", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  [{s['score']:.3f}] ({s['type']}) {s['text'][:100]}")
