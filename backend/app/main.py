"""FastAPI giriş noktası: RAG çekirdeğini HTTP üzerinden sunar."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, rag
from .chart_mapping import get_chart_filename

app = FastAPI(title="GPT Plugin Privacy RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local geliştirme için; production'da spesifik origin(ler) ile değiştirin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    text: str
    type: str
    metadata: dict
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    # Sabit soru→görsel eşleştirme tablosundan gelen dosyanın public URL'i
    # (örn. "/static/charts/rq1_category_distribution.png"); eşleşme yoksa None.
    chart_image: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question boş olamaz.")
    try:
        result = rag.answer(request.question, top_k=request.top_k)
    except RuntimeError as e:
        # Örn: index kurulmamış, API anahtarı eksik.
        raise HTTPException(status_code=503, detail=str(e))

    filename = get_chart_filename(request.question)
    result["chart_image"] = f"/static/charts/{filename}" if filename else None
    return result
