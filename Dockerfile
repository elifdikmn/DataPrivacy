FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# FAISS/PyTorch need the OpenMP runtime. Install a CPU-only PyTorch wheel so
# the image does not pull CUDA libraries onto a CPU-only Cloud Run service.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Explicit copies keep the local backend/.env and unrelated repo files out of
# the image. The model cache and FAISS indices become immutable image assets.
COPY backend/app ./app
COPY backend/data ./data
COPY backend/knowledge ./knowledge
COPY backend/final_results ./final_results
COPY backend/static ./static
RUN python -m app.indexing

# Runtime must use the model baked into the image, not download it on a cold start.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
