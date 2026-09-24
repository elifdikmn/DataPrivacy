# Deploy on Render

This repository needs two Render services from the same GitHub repository: a Python Web Service for the API and a React Static Site for the interface. Deploy the API first, because the Static Site needs its public URL at build time.

## 1. API (Web Service)

Create a Render **Web Service** connected to the repository's `main` branch:

| Setting | Value |
| --- | --- |
| Language | Python 3 |
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt && python -m app.indexing` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

Add these environment variables in Render, **not** in GitHub:

| Variable | Value |
| --- | --- |
| `ANTHROPIC_API_KEY` | Your secret Anthropic API key |
| `PYTHON_VERSION` | `3.13.5` (the version used for local verification) |
| `CORS_ORIGINS` | The exact frontend origin, e.g. `https://your-site.onrender.com` |

You can initially leave `CORS_ORIGINS` at its local-only default while deploying the API, then set it after the Static Site URL is known. Do not include a path or trailing slash. To allow multiple frontend origins, separate them with commas.

The FAISS indices are generated during the build, because `backend/app/index_store/` is intentionally not committed. The first build downloads the embedding model and may take time. If dependency installation or indexing exceeds the chosen plan's memory or build limits, use a plan with more resources. Do not move index generation to a pre-deploy command: its filesystem changes are not carried into the running service.

After deploy, open `https://YOUR-API.onrender.com/ready`. It should report `status: ready`, `index_current: true`, and `llm_key_configured: true`. `/health` alone does not check the index or key.

## 2. Interface (Static Site)

Create a Render **Static Site** from the same repository and branch:

| Setting | Value |
| --- | --- |
| Root Directory | `frontend` |
| Build Command | `npm ci && npm run build` |
| Publish Directory | `build` |

Set `REACT_APP_API_BASE` to the API's public HTTPS URL, for example `https://your-api.onrender.com`. Do not add a trailing slash. This is a public address embedded into the React build, **not** a secret. If you change it later, redeploy the Static Site; changing the variable does not alter an already-built bundle.

Once the Static Site has a URL, return to the API service and set `CORS_ORIGINS` to that exact origin. Redeploy the API if Render does not restart it automatically after the setting changes.

## 3. Smoke test

1. Check the API `/ready` endpoint, then open the Static Site on a phone or another computer.
2. Ask a General audience question and confirm an answer and its chart appear.
3. Switch to Researcher and ask a technical question.
4. If the browser reports a connection error, check `REACT_APP_API_BASE`, the API service logs, and `CORS_ORIGINS`. A frontend URL should never point to `localhost` in production.

## Before inviting broad public traffic

The `/ask` endpoint calls a paid LLM. CORS restricts which browser origins can call it, but **CORS is not authentication or a rate limit**; direct clients can still call the API. Set an Anthropic spending limit and add server-side or edge rate limiting before sharing the site widely. Keep `ANTHROPIC_API_KEY` only in Render's secret environment settings. Render's free Web Service can sleep when idle and its filesystem is ephemeral, so cold starts may take time; indices must therefore be rebuilt as part of each deploy.

References: [Render FastAPI deployment](https://render.com/docs/deploy-fastapi), [monorepo root directories](https://render.com/docs/monorepo-support), [Create React App static sites](https://render.com/docs/deploy-create-react-app), [free service limitations](https://render.com/docs/free).
