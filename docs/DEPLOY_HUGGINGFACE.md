# Deploy the backend on Hugging Face Spaces (free)

Recommended free setup: the React frontend stays a **Render Static Site**
(`https://dataprivacy-1.onrender.com`) and only the Python API moves to a **Hugging Face Docker
Space**. Render's free web service has 512 MB of memory, which the embedding model does not fit
in; the free Hugging Face CPU hardware has much more memory and needs no credit card.
Free Spaces go to sleep after a period without traffic; the first request afterwards waits for
the container to start (about a minute).

The repository-root `Dockerfile` builds the API, downloads the embedding model and generates the
FAISS indices inside the image. The `deploy-backend-hf` GitHub Actions workflow uploads the
Dockerfile and `backend/` to the Space on every push to `main` that touches them; Hugging Face
then rebuilds the image.

## 1. Hugging Face token

1. Create a free account at huggingface.co.
2. **Settings → Access Tokens → Create new token**, type **Write**. Copy it once; never commit it.

## 2. Connect GitHub to the Space

In the GitHub repository: **Settings → Secrets and variables → Actions**

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `HF_TOKEN` | the Hugging Face write token |
| Variable | `HF_SPACE` | `<your-hf-username>/dataprivacy-api` |

Then **Actions → deploy-backend-hf → Run workflow** (or push to `main`). The workflow creates the
Space if it does not exist, uploads the backend and prints the API URL:
`https://<your-hf-username>-dataprivacy-api.hf.space`.

## 3. Space settings

On the Space page: **Settings → Variables and secrets**

| Kind | Name | Value |
| --- | --- | --- |
| Secret | `ANTHROPIC_API_KEY` | your Anthropic API key |
| Variable | `CORS_ORIGINS` | `https://dataprivacy-1.onrender.com` (no trailing slash) |

Optional variables: `RATE_LIMIT_PER_MINUTE` (default 8), `RATE_LIMIT_PER_DAY` (100 per client),
`GLOBAL_DAILY_LIMIT` (500 questions per day for everyone). The Space restarts after a change.

The first build takes several minutes (CPU PyTorch, the embedding model, 12,811 records to
embed); follow it in the Space's **Logs** tab. Then open
`https://<your-hf-username>-dataprivacy-api.hf.space/ready` — it must report
`"status": "ready"`, `index_current: true` and `llm_key_configured: true`.

## 4. Point the frontend at the new API

Render → the `DataPrivacy-1` **Static Site** → **Environment**: set `REACT_APP_API_BASE` to the
Space URL (no trailing slash), then **Manual Deploy → Deploy latest commit**. Create React App
embeds this value at build time, so a rebuild is required.

Test one General audience question with its chart and one Researcher question. When both work,
delete or suspend the old Render web service for the API so it no longer runs.

## Costs and safety

Hosting is free; each question calls the paid Anthropic API (roughly $0.004–0.01 with
Claude Haiku 4.5). `/ask` is rate-limited per client and has a global daily cap, but the API is
public: also set a monthly spending limit in the Anthropic Console. The rate limiter keeps its
counters in memory, so they reset when the Space restarts.
