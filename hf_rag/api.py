"""
FastAPI — точка входа для RAG-сервиса.

Endpoints:
  GET  /health          — статус сервисов
  POST /search          — только поиск (E5 + BGE), с кэшем
  POST /ask             — полный RAG-пайплайн, с кэшем
  GET  /cache/stats     — статистика Redis
  POST /cache/clear     — очистить кэш
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hf_rag.cache import get_cache
from hf_rag.config import settings
from hf_rag import pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="HuggingFace Docs RAG API",
    description=(
        "RAG-система по документации HuggingFace. "
        "Поиск: E5-large + Qdrant + BGE Reranker Large. "
        f"Генерация: {settings.groq_model} (Groq)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── схемы ─────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000,
                       example="How do I fine-tune a model with Trainer?")
    top_k: Optional[int] = Field(None, ge=1, le=20,
                                 description="Кол-во результатов (по умолч. из конфига)")


class ContextItem(BaseModel):
    text: str
    source: str
    library: str
    reranker_score: float
    doc_id: Optional[int]
    chunk_idx: Optional[int]


class SearchResponse(BaseModel):
    query: str
    contexts: List[ContextItem]
    cache_hit: bool
    elapsed_ms: float


class AskResponse(BaseModel):
    query: str
    answer: str
    contexts: List[ContextItem]
    answer_cache_hit: bool
    context_cache_hit: bool
    elapsed_ms: float
    model: str


class CacheStats(BaseModel):
    hits: int
    misses: int
    keys: int
    redis_ok: bool


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    cache = get_cache()
    try:
        cols = [c.name for c in pipeline.get_qdrant().get_collections().collections]
        qdrant_ok = settings.qdrant_collection in cols
    except Exception:
        qdrant_ok = False
    return {
        "status": "ok",
        "model": settings.groq_model,
        "qdrant": qdrant_ok,
        "redis": cache.ping(),
        "collection": settings.qdrant_collection,
    }


@app.post("/search", response_model=SearchResponse, tags=["RAG"])
def search(req: QueryRequest):
    """
    Поиск по документации HuggingFace.
    E5-large → Qdrant → BGE Reranker Large.
    Результаты кэшируются в Redis.
    """
    t0 = time.perf_counter()
    cache = get_cache()

    cached = cache.get_contexts(req.query)
    cache_hit = cached is not None
    if cache_hit:
        contexts = cached
    else:
        try:
            contexts = pipeline.search(req.query)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        cache.set_contexts(req.query, contexts)

    # обрезаем если передан top_k
    if req.top_k:
        contexts = contexts[: req.top_k]

    return SearchResponse(
        query=req.query,
        contexts=[ContextItem(**c) for c in contexts],
        cache_hit=cache_hit,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(req: QueryRequest):
    """
    Полный RAG-пайплайн: поиск + генерация ответа.
    Контексты и ответ кэшируются отдельно.
    """
    t0 = time.perf_counter()
    cache = get_cache()

    # 1. контексты
    cached_ctx = cache.get_contexts(req.query)
    ctx_hit = cached_ctx is not None
    if ctx_hit:
        contexts = cached_ctx
    else:
        try:
            contexts = pipeline.search(req.query)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        cache.set_contexts(req.query, contexts)

    if req.top_k:
        contexts = contexts[: req.top_k]

    # 2. ответ
    cached_ans = cache.get_answer(req.query)
    ans_hit = cached_ans is not None
    if ans_hit:
        answer = cached_ans
    else:
        try:
            answer = pipeline.generate_answer(req.query, contexts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        cache.set_answer(req.query, answer)

    return AskResponse(
        query=req.query,
        answer=answer,
        contexts=[ContextItem(**c) for c in contexts],
        answer_cache_hit=ans_hit,
        context_cache_hit=ctx_hit,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        model=settings.groq_model,
    )


@app.get("/cache/stats", response_model=CacheStats, tags=["Cache"])
def cache_stats():
    cache = get_cache()
    s = cache.stats()
    return CacheStats(redis_ok=cache.ping(), **s)


@app.post("/cache/clear", tags=["Cache"])
def cache_clear():
    cache = get_cache()
    try:
        cache._r.flushdb()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
