# Backend modules

- `domain.py`: shared unique-Action counting, audit-label priorities and unambiguous taxonomy mapping.
- `main.py`: FastAPI routes, bounded requests, liveness/readiness and provider error responses.
- `indexing.py`: record/knowledge/audit documents and FAISS indices.
- `index_state.py`: source fingerprints invalidate stale indices after analysis changes.
- `retrieval.py`: source-aware index loading and searches.
- `llm.py`: Anthropic response contract; qualitative explanation plus fact IDs.
- `facts.py`: renders numbers with their own fact labels, rejects invalid payloads; numeric membership is only a diagnostic.
- `rag.py`: retrieval and answer orchestration.
- `chart_mapping.py`: suggested question → prebuilt chart, including conditional confidence intervals.

See the repository README for installation, analysis regeneration, index rebuilding and limitations. Index files are local generated artifacts; rebuilding requires the configured embedding model. Live answers require a configured API key.
