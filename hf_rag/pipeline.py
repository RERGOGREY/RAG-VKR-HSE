"""
RAG-пайплайн: E5-large → Qdrant → BGE Reranker Large → Qwen3-32B (Groq)
"""
from __future__ import annotations

import re
import uuid
from functools import lru_cache
from typing import List, Dict

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import CrossEncoder, SentenceTransformer

from hf_rag.config import settings

# ── промпт ────────────────────────────────────────────────────────────────────
RAG_PROMPT = """\
You are an expert assistant on the Hugging Face documentation (Transformers, Diffusers, Datasets, etc.).
Answer the question using ONLY the provided context. Be concise and factual.
If the context does not contain the answer, say so explicitly.
Do NOT include your internal reasoning or thinking process in the answer — output only the final answer.

Context:
{context}

Question: {question}

Answer:\
"""

# регулярка для очистки <think>...</think> блоков (Qwen3 chain-of-thought)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ── синглтоны (загружаются один раз при старте) ───────────────────────────────
@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model, max_length=512)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return ChatGroq(
        model_name=settings.groq_model,
        groq_api_key=settings.groq_api_key,
        temperature=0.0,
        max_retries=3,
    )


# ── поиск ─────────────────────────────────────────────────────────────────────
def vector_search(query: str, top_k: int | None = None) -> List[Dict]:
    """Векторный поиск через E5-large + Qdrant."""
    k = top_k or settings.top_k_retrieve
    embedder = get_embedder()
    qdrant = get_qdrant()

    vec = embedder.encode([f"query: {query}"], normalize_embeddings=True)[0].tolist()
    response = qdrant.query_points(
        collection_name=settings.qdrant_collection,
        query=vec,
        limit=k,
        with_payload=True,
    )
    return [
        {
            "text": p.payload.get("text", ""),
            "score": p.score,
            "source": p.payload.get("source", ""),
            "library": p.payload.get("library", ""),
            "doc_id": p.payload.get("doc_id"),
            "chunk_idx": p.payload.get("chunk_idx"),
        }
        for p in response.points
    ]


def rerank(query: str, candidates: List[Dict], top_k: int | None = None) -> List[Dict]:
    """BGE Reranker Large — переранжирует кандидатов."""
    k = top_k or settings.top_k_final
    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [{"reranker_score": float(s), **c} for s, c in ranked[:k]]


def search(query: str) -> List[Dict]:
    """Полный поиск: E5 → Qdrant → BGE."""
    candidates = vector_search(query)
    return rerank(query, candidates)


# ── генерация ─────────────────────────────────────────────────────────────────
def generate_answer(query: str, contexts: List[Dict]) -> str:
    """Генерирует ответ через Qwen3-32B (Groq) на основе контекстов."""
    llm = get_llm()
    ctx_text = "\n\n".join(
        f"[{i+1}] ({c.get('source', '')})\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    prompt = RAG_PROMPT.format(context=ctx_text, question=query)
    raw = llm.invoke(prompt).content
    # метод 2: вырезаем <think>...</think> блоки на случай если промт не сработал
    return _THINK_RE.sub("", raw).strip()


# ── индексация ────────────────────────────────────────────────────────────────
def index_corpus(jsonl_path: str, batch_size: int = 100) -> tuple[int, int]:
    """Индексирует JSONL-корпус в Qdrant. Возвращает (n_docs, n_chunks)."""
    import json
    from tqdm import tqdm

    embedder = get_embedder()
    qdrant = get_qdrant()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # создаём/пересоздаём коллекцию
    cols = [c.name for c in qdrant.get_collections().collections]
    if settings.qdrant_collection in cols:
        qdrant.delete_collection(settings.qdrant_collection)
    qdrant.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(
            size=embedder.get_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )

    docs, all_points = [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    for doc_idx, doc in enumerate(tqdm(docs, desc="Indexing")):
        text = doc.get("text") or doc.get("content") or ""
        if not text:
            continue
        chunks = splitter.split_text(text)
        passages = [f"passage: {c}" for c in chunks]
        vecs = embedder.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        meta = {k: v for k, v in doc.items() if k != "text"}
        for ci, (chunk, vec) in enumerate(zip(chunks, vecs)):
            all_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={"text": chunk, "chunk_idx": ci,
                             "total_chunks": len(chunks), "doc_id": doc_idx, **meta},
                )
            )

    for i in range(0, len(all_points), batch_size):
        qdrant.upsert(collection_name=settings.qdrant_collection,
                      points=all_points[i: i + batch_size])

    return len(docs), len(all_points)
