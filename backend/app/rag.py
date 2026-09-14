"""RAG core: combines retrieval with Claude to produce an answer."""

import logging

from . import llm
from .facts import format_facts_block, verify_answer_numbers
from .retrieval import search

logger = logging.getLogger("chatbot.rag")


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

    # FACTS tablosu her zaman en sonda eklenir — küçük olduğu için tamamı,
    # sorudan bağımsız. Amaç: LLM'in sayıları ham kayıtlardan kendi kafasında
    # hesaplaması yerine, buradan birebir kopyalaması.
    parts.append(format_facts_block())

    return "\n\n".join(parts)


def answer(question: str, top_k: int = 5) -> dict:
    results = search(question, top_k=top_k)
    context = build_context(results)
    response_text = llm.ask(question, context)

    unknown_numbers = verify_answer_numbers(response_text)
    if unknown_numbers:
        logger.warning("Question: %r — unverified numbers in answer: %s", question, unknown_numbers)

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
