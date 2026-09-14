"""FastAPI giriş noktası: RAG çekirdeğini HTTP üzerinden sunar."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import rag
from .retrieval import search
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


class ChartDatum(BaseModel):
    category: str
    count: int
    sensitive: bool


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    chart: list[ChartDatum] | None = None  # frontend Recharts ile çizer


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question boş olamaz.")
    try:
        result = rag.answer(request.question, top_k=request.top_k)
        # Grafik, LLM'in cevap için kullandığı dar bağlamdan (top_k=5) bağımsız,
        # daha geniş bir kayıt örneklemine dayanır — böylece geniş kapsamlı
        # sorularda da anlamlı bir kategori dağılımı gösterebilir.
        chart_sources = search(request.question, top_k=50, top_k_knowledge=0, top_k_audit=0)
    except RuntimeError as e:
        # Örn: index kurulmamış, API anahtarı eksik.
        raise HTTPException(status_code=503, detail=str(e))
    result["chart"] = sources_category_chart(chart_sources)
    return result
