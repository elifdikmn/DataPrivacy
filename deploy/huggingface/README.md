---
title: DataPrivacy API
emoji: 🔒
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8080
pinned: false
short_description: RAG API for the GPT Plugin Privacy Assistant
---

# GPT Plugin Privacy Assistant — API

Backend of the GPT Plugin Privacy Assistant (FastAPI + FAISS + Claude). This Space is
deployed automatically from the GitHub repository `elifdikmn/DataPrivacy` by the
`deploy-backend-hf` workflow; edit the code on GitHub, not here.

- `GET /ready` — reports whether the retrieval index and the API key are configured
- `POST /ask` — answers a question (used by the React frontend)

Space settings needed: secret `ANTHROPIC_API_KEY`, variable `CORS_ORIGINS` (the frontend's origin).
