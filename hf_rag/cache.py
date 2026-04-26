"""
Redis-кэш для поисковых результатов и сгенерированных ответов.
Ключ = md5(query). TTL задаётся в settings.cache_ttl.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import List, Dict, Optional

import redis

from hf_rag.config import settings

log = logging.getLogger(__name__)


class RAGCache:
    def __init__(self) -> None:
        self._r = redis.from_url(settings.redis_url, decode_responses=True)
        self.ttl = settings.cache_ttl

    # ── вспомогательные ───────────────────────────────────────────────────────
    def _key(self, prefix: str, query: str) -> str:
        digest = hashlib.md5(query.strip().lower().encode()).hexdigest()
        return f"rag:{prefix}:{digest}"

    def ping(self) -> bool:
        try:
            return self._r.ping()
        except Exception:
            return False

    # ── контексты (результаты поиска) ─────────────────────────────────────────
    def get_contexts(self, query: str) -> Optional[List[Dict]]:
        try:
            raw = self._r.get(self._key("ctx", query))
            return json.loads(raw) if raw else None
        except Exception as e:
            log.warning("Cache get_contexts error: %s", e)
            return None

    def set_contexts(self, query: str, contexts: List[Dict]) -> None:
        try:
            self._r.setex(self._key("ctx", query), self.ttl, json.dumps(contexts))
        except Exception as e:
            log.warning("Cache set_contexts error: %s", e)

    # ── ответы ────────────────────────────────────────────────────────────────
    def get_answer(self, query: str) -> Optional[str]:
        try:
            return self._r.get(self._key("ans", query))
        except Exception as e:
            log.warning("Cache get_answer error: %s", e)
            return None

    def set_answer(self, query: str, answer: str) -> None:
        try:
            self._r.setex(self._key("ans", query), self.ttl, answer)
        except Exception as e:
            log.warning("Cache set_answer error: %s", e)

    # ── статистика ────────────────────────────────────────────────────────────
    def stats(self) -> Dict:
        try:
            info = self._r.info("stats")
            return {
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "keys": self._r.dbsize(),
            }
        except Exception:
            return {"hits": 0, "misses": 0, "keys": 0}


# синглтон
_cache: Optional[RAGCache] = None


def get_cache() -> RAGCache:
    global _cache
    if _cache is None:
        _cache = RAGCache()
    return _cache
