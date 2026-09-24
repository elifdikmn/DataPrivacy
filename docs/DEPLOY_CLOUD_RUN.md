# Move the backend to Google Cloud Run

The React frontend can stay at `https://dataprivacy-1.onrender.com`. Only the Python API moves. This is a deployment option for the current sentence-transformers/FAISS backend, which exceeded the Render Free instance's 512 MB memory limit.

Cloud Run requires a Google Cloud project with a billing account, even when usage stays within its free tier. **Do not create a service or enable billing until you are comfortable with possible charges.** The 2 GiB setting below is a starting point, not a guarantee; verify actual memory usage after the first question. Cloud Build, Artifact Registry, Secret Manager, network traffic, and Anthropic API calls may also have costs. Set budgets/alerts and review the service's usage.

## What is already prepared

The repository-root `Dockerfile` builds the existing API, downloads the multilingual embedding model, and generates the three FAISS indices *inside the container image*. The `.dockerignore`, `.gcloudignore`, and explicit `COPY` statements keep `backend/.env` and local virtual environments out of the build and source upload. Cloud Run therefore does not need to rebuild indices or download the model on every cold start.

## Create the backend service

1. In Google Cloud, select or create a project and attach a billing account. Enable Cloud Run, Cloud Build, Artifact Registry, and Secret Manager when prompted.
2. In Secret Manager, create a secret for the Anthropic key. Enter the key there yourself; never paste it into GitHub, a build variable, a support message, or the React frontend.
3. In Cloud Run, choose **Deploy from source repository / continuously deploy from a repository**, connect the GitHub repository `elifdikmn/DataPrivacy`, choose branch `main`, and select the repository-root `Dockerfile` as the build configuration. Alternatively, from a local checkout with Google Cloud CLI configured, deploy from the repository root with `gcloud run deploy dataprivacy-api --source .` (the root Dockerfile is selected automatically).
4. Configure the service with **1 CPU, 2 GiB memory, concurrency 1, minimum instances 0, maximum instances 1** to start. Allow unauthenticated access because the public React site calls the API directly. This makes the API public; it is not a security control. Keep the default request timeout unless real tests show that it is too short.
5. Add an ordinary environment variable `CORS_ORIGINS=https://dataprivacy-1.onrender.com`. Expose the Secret Manager secret as the `ANTHROPIC_API_KEY` environment variable. Do not set `ANTHROPIC_API_KEY` as a literal source/build variable.
6. Deploy. The image build may take time because it installs CPU-only PyTorch and embeds 12,811 records. Open the resulting `https://...run.app/ready` URL. It must report `index_current: true` and `llm_key_configured: true`.

## Connect the existing frontend

In Render, open the `DataPrivacy-1` **Static Site** → **Environment** and change `REACT_APP_API_BASE` to the new Cloud Run HTTPS URL (without a trailing slash). **Rebuild and deploy** the static site; Create React App embeds this value at build time. The backend's `CORS_ORIGINS` must remain exactly `https://dataprivacy-1.onrender.com`.

Test one General audience question and its chart, then one Researcher question. Only after both work should you retire the old Render backend. Keep the Anthropic spending limit and add API rate limiting before broad public sharing; CORS alone does not prevent direct calls to a public API.

## References

- [Deploy Cloud Run continuously from GitHub](https://docs.cloud.google.com/run/docs/continuous-deployment)
- [Deploy from source with a Dockerfile](https://docs.cloud.google.com/run/docs/deploying-source-code)
- [Cloud Run memory settings](https://docs.cloud.google.com/run/docs/configuring/services/memory-limits)
- [Use Secret Manager with Cloud Run](https://docs.cloud.google.com/run/docs/configuring/services/secrets)
- [Google Cloud free-tier conditions](https://docs.cloud.google.com/free/docs/free-cloud-features)
