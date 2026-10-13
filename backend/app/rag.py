"""RAG core: combines retrieval with Claude to produce an answer."""

import logging
import re

from . import llm
from .chart_mapping import get_chart_filename
from .facts import load_facts
from .researcher_answers import answer_general_question, answer_researcher_question
from .retrieval import search

logger = logging.getLogger("chatbot.rag")

_OTHER_GENERAL_QUESTIONS = {
    'can the model suggest categories for "other" records',
    'can mislabeled "other" records be identified automatically',
}
_OTHER_SENSITIVE_QUESTION = 'are there hidden sensitive parameters mislabeled as "other"'


def curated_password_answer(question: str, audience: str) -> dict | None:
    """Keep the password answer aligned with the manually reviewed chart."""
    if get_chart_filename(question) != "password_use_cases.png":
        return None

    facts = load_facts()["password_records"]
    groups = facts["purpose_group_counts"]
    summary = (
        f"Across **{facts['unique_plugins']} distinct GPT Actions**, the dataset lists "
        f"**{facts['total_records']} password-labeled fields**: "
        f"{groups['Database access']} for database access, "
        f"{groups['Content or resume protection']} for content or resume protection, "
        f"{groups['SMPP messaging']} for SMPP messaging, "
        f"{groups['Host connection']} for host connection, "
        f"{groups['Article-posting account']} for an article-posting account, "
        f"and {groups['General or unclear']} with unclear purposes."
    )
    if audience == "general":
        answer = summary + " These declared fields do not prove that passwords were sent."
    else:
        answer = (
            summary + " The groups were assigned conservatively from schema descriptions "
            "and count parameter records, not Actions; one record may appear in several "
            "Actions. The dataset does not show whether "
            "any password was actually transmitted at runtime."
        )
    return {
        "answer": answer,
        "sources": [{
            "text": "Manually reviewed purposes of Password-labeled schema fields and distinct Action counts.",
            "type": "finding",
            "metadata": {"filename": "project_facts.json"},
            "score": 1.0,
        }],
    }


def curated_other_answer(question: str, audience: str) -> dict | None:
    """Ground the fixed RQ5 suggestions in measured facts, without an LLM claim of proof."""
    normalized = re.sub(r"\s+", " ", question.strip().lower()).rstrip(" ?!.")
    if normalized not in _OTHER_GENERAL_QUESTIONS | {_OTHER_SENSITIVE_QUESTION}:
        return None

    facts = load_facts()["other_reclassification"]
    total = facts["total_other_data_type_records"]
    candidates = facts["confident_subset_count"]
    share = facts["confident_subset_pct"]
    sensitive = facts["sensitive_flagged_in_confident_subset"]
    if normalized == _OTHER_SENSITIVE_QUESTION:
        if audience == "general":
            answer = (f"The model suggested a sensitive category for **{sensitive}** 'Other' records. "
                      "A person would need to check them before we could call them mislabeled.")
        else:
            answer = (f"Among **{total:,}** records labeled 'Other', **{candidates}** exceeded an "
                      f"uncalibrated 0.5 prediction-score threshold. **{sensitive}** of those were "
                      "assigned to the project's selected sensitive categories: "
                      f"{facts['sensitive_flagged_breakdown']['Security credentials']} to Security "
                      f"credentials and {facts['sensitive_flagged_breakdown']['Personal information']} "
                      "to Personal information. These are unverified review candidates, not "
                      "confirmed mislabeled or hidden sensitive records. Human labels are needed "
                      "to assess correctness on the 'Other' population.")
    elif audience == "general":
        answer = (f"The model suggested categories for **{candidates} of {total:,}** 'Other' records "
                  f"({share}%). These are suggestions to check, not confirmed labeling mistakes.")
    else:
        answer = (f"Of **{total:,}** 'Other' records, **{candidates} ({share}%)** exceeded the "
                  f"uncalibrated 0.5 prediction-score threshold. **{sensitive}** of those were "
                  "assigned to the project's selected sensitive categories. This is a priority "
                  "list for human review, not automatic relabeling: none of these predictions "
                  "has been verified, and test-set accuracy on known labels does not establish "
                  "accuracy on the 'Other' records.")
    return {
        "answer": answer,
        "sources": [{
            "text": "RQ5: 'Other' reclassification results and limitations.",
            "type": "finding",
            "metadata": {"filename": "rq5_other_siniflandirma.md"},
            "score": 1.0,
        }],
    }


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
    curated = curated_password_answer(question, audience)
    if curated is not None:
        return curated
    curated = answer_researcher_question(question, audience)
    if curated is not None:
        return curated
    curated = answer_general_question(question, audience)
    if curated is not None:
        return curated
    curated = curated_other_answer(question, audience)
    if curated is not None:
        return curated
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
