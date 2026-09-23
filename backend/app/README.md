# Backend modules

- `domain.py`: shared unique-Action counting, audit-label priorities and unambiguous taxonomy mapping.
- `main.py`: FastAPI routes, bounded requests (including recent chat history), configurable CORS origins, liveness/readiness and provider error responses.
- `indexing.py`: record/knowledge/audit documents and FAISS indices.
- `index_state.py`: source fingerprints invalidate stale indices after analysis changes (re-hashed only when a source file changes).
- `retrieval.py`: source-aware index loading and searches.
- `llm.py`: system prompt (non-specialist audience, short answers, key terms in bold), FACTS sent as a cached system block, chat history, tolerant legacy JSON handling.
- `facts.py`: complete reference facts and optional legacy rendering; numeric diagnostics (percentage and Turkish decimal forms accepted) only log and never block chat.
- `rag.py`: retrieval and answer orchestration; follow-up questions are searched together with the previous question.
- `chart_mapping.py`: suggested question → prebuilt chart (case, spacing, quote style and trailing punctuation ignored), including conditional confidence intervals.

See the repository README for installation, analysis regeneration, index rebuilding and limitations. Index files are local generated artifacts; rebuilding requires the configured embedding model. Live answers require a configured API key.
