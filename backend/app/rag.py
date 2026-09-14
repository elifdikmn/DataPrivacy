"""RAG çekirdeği: retrieval + Claude ile cevap üretimini birleştirir."""

from . import llm
from .retrieval import search


def build_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        if r["type"] == "record":
            m = r["metadata"]
            parts.append(
                f"- Parametre '{m['name']}' ({m['plugin_count']} eklentide kullanılıyor): "
                f"{m['description'] or '(açıklama yok)'} — Kategori: {m['main_data_type']} / {m['data_type']}"
            )
        else:
            parts.append(f"[Analiz bulgusu — {r['metadata']['filename']}]\n{r['text']}")
    return "\n\n".join(parts)


def answer(question: str, top_k: int = 5) -> dict:
    results = search(question, top_k=top_k)
    context = build_context(results)
    response_text = llm.ask(question, context)
    return {"answer": response_text, "sources": results}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Hangi tür veriler en çok toplanıyor ve hangileri hassas?"
    result = answer(question)
    print("SORU:", question)
    print("\nCEVAP:", result["answer"])
    print("\nKAYNAKLAR:")
    for s in result["sources"]:
        print(f"  [{s['score']:.3f}] ({s['type']}) {s['text'][:100]}")
