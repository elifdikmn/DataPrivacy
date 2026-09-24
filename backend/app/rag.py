"""RAG core: combines retrieval with Claude to produce an answer."""

import logging

from . import llm
from .retrieval import search

logger = logging.getLogger("chatbot.rag")


def build_context(results: list[dict]) -> str:
    # FACTS tablosu burada değil, system prompt'ta (llm.system_blocks) — her istekte
    # aynı olduğu için orada önbelleğe alınıyor. Burada yalnızca soruya göre
    # değişen, retrieval'dan gelen kanıtlar var.
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
    return "\n\n".join(parts) if parts else "(no retrieved evidence)"


def retrieval_query(question: str, history: list[dict] | None = None) -> str:
    """Follow-up questions ("and its confidence interval?") are short and vague on
    their own, so the previous user question is added to the search query."""
    previous = [t.get("text", "").strip() for t in (history or []) if t.get("role") == "user"]
    previous = [text for text in previous if text]
    return f"{previous[-1]}\n{question}" if previous else question


def answer(question: str, top_k: int = 5, audience: str = "general",
           history: list[dict] | None = None) -> dict:
    results = search(retrieval_query(question, history), top_k=top_k)
    context = build_context(results)
    response_text = llm.ask(question, context, audience=audience, history=history)

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
