"""FastAPI giriş noktası: RAG çekirdeğini HTTP üzerinden sunar."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import rag
from .viz import sources_category_chart

app = FastAPI(title="GPT Plugin Privacy RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local geliştirme için; production'da spesifik origin(ler) ile değiştirin
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    chart: str | None = None  # Plotly figure JSON (fig.to_json()); frontend Plotly.js ile çizer


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
    result["chart"] = sources_category_chart(result["sources"])
    return result
