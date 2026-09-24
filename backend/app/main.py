"""FastAPI giriş noktası: RAG çekirdeğini HTTP üzerinden sunar."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
import anthropic

from . import config, rag
from .chart_mapping import get_chart_filename

app = FastAPI(title="GPT Plugin Privacy RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    audience: Literal["general", "researcher"] = "general"


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


@app.get("/ready")
def ready():
    from .index_state import index_is_current
    checks = {"index_current": index_is_current(), "llm_key_configured": bool(config.ANTHROPIC_API_KEY)}
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ready", **checks}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question boş olamaz.")
    try:
        result = rag.answer(request.question, top_k=request.top_k, audience=request.audience)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=503, detail="The answer service is not configured correctly.")
    except anthropic.APITimeoutError:
        raise HTTPException(status_code=504, detail="The answer service timed out. Please try again.")
    except anthropic.APIError:
        raise HTTPException(status_code=502, detail="The answer service is temporarily unavailable.")
    except RuntimeError as e:
        # Örn: index kurulmamış, API anahtarı eksik.
        raise HTTPException(status_code=503, detail=str(e))

    filename = get_chart_filename(request.question)
    result["chart_image"] = f"/static/charts/{filename}" if filename else None
    return result
