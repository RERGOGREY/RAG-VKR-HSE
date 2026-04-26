"""
Индексирует corpus.jsonl в Qdrant.
Использует index_corpus() из hf_rag/pipeline.py.

Запуск:
  python data_pipeline/ingest.py
  python data_pipeline/ingest.py --corpus data_pipeline/corpus.jsonl --qdrant-url http://localhost:6333
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# добавляем корень проекта в путь (на случай запуска не из корня)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CORPUS    = "data_pipeline/corpus.jsonl"
DEFAULT_QDRANT    = "http://localhost:6333"
DEFAULT_BATCH     = 100


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексация corpus.jsonl в Qdrant"
    )
    parser.add_argument(
        "--corpus", default=DEFAULT_CORPUS,
        help=f"Путь к JSONL-корпусу (по умолч: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", DEFAULT_QDRANT),
        help=f"URL Qdrant (по умолч: {DEFAULT_QDRANT})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH,
        help=f"Размер батча для upsert (по умолч: {DEFAULT_BATCH})",
    )
    args = parser.parse_args()

    corpus_path = ROOT / args.corpus
    if not corpus_path.exists():
        print(f"Корпус не найден: {corpus_path}")
        print(f"Сначала запустите: python data_pipeline/build_corpus.py")
        sys.exit(1)

    # патчим env-переменную до импорта settings
    os.environ["QDRANT_URL"] = args.qdrant_url

    # импортируем после установки env (settings читается при импорте)
    from hf_rag.config import settings
    from hf_rag.pipeline import index_corpus

    print("Индексация корпуса в Qdrant...")
    print(f"Корпус    : {corpus_path}")
    print(f"Qdrant    : {args.qdrant_url}")
    print(f"Коллекция : {settings.qdrant_collection}")
    print(f"Батч      : {args.batch_size}\n")

    t0 = time.time()
    n_docs, n_chunks = index_corpus(str(corpus_path), batch_size=args.batch_size)
    elapsed = time.time() - t0

    print(f"\n{'='*55}")
    print(f"Документов проиндексировано : {n_docs}")
    print(f"Чанков загружено            : {n_chunks}")
    print(f"Коллекция                   : {settings.qdrant_collection}")
    print(f"Время                       : {elapsed:.1f}с")
    print(f"{'='*55}")
    print("\nСледующий шаг:")
    print("docker-compose up -d   # (если сервис ещё не запущен)")


if __name__ == "__main__":
    main()
