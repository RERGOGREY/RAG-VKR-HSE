# HuggingFace Docs RAG

RAG-система для поиска и ответов на вопросы по документации HuggingFace.

## Установка

```bash
git clone https://github.com/RERGOGREY/RAG-VKR-HSE.git
cd RAG-VKR-HSE
pip install -e .
```

## Конфигурация

Создаём `.env` по примеру `.env.example`:

```bash
cp .env.example .env
```

| Переменная | Описание | По умолчанию |
|-----------|----------|-------------|
| `GROQ_API_KEY` | API-ключ [Groq](https://console.groq.com/) | — |
| `GROQ_MODEL` | Модель генерации | `qwen/qwen3-32b` |
| `EMBEDDING_MODEL` | Модель эмбеддингов | `intfloat/multilingual-e5-large` |
| `RERANKER_MODEL` | Модель переранжирования | `BAAI/bge-reranker-large` |
| `QDRANT_URL` | URL векторной БД | `http://qdrant:6333` |
| `CHUNK_SIZE` | Размер чанка (символов) | `600` |
| `TOP_K_RETRIEVE` | Кандидатов из Qdrant | `40` |
| `TOP_K_FINAL` | Финальных чанков после rerank | `5` |

## Загрузка данных

Документация скачивается и индексируется через пайплайн:

```bash
# Всё сразу — все 10 библиотек HuggingFace (~15-30 мин)
python data_pipeline/run_all.py
```

Или по шагам:

```bash
# 1. Скачать .md файлы (параллельно, git sparse-checkout)
python data_pipeline/download.py

# 2. Очистить и собрать corpus.jsonl
python data_pipeline/build_corpus.py

# 3. Проиндексировать в Qdrant
python data_pipeline/ingest.py
```

Поддерживаемые библиотеки: `transformers` · `diffusers` · `datasets` · `accelerate` · `peft` · `trl` · `evaluate` · `tokenizers` · `optimum` · `hub-docs`

## Использование

### 1. Запустить сервис

```bash
docker-compose up -d
```

| Сервис | URL |
|--------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000/docs |
| Qdrant Dashboard | http://localhost:6333/dashboard |

### 2. Задать вопрос через API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I apply LoRA fine-tuning with PEFT?"}'
```

### 3. Только векторный поиск

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "gradient checkpointing", "top_k": 5}'
```

### 4. Статистика кэша

```bash
curl http://localhost:8000/cache/stats
```

## Структура проекта

```
hf_rag/
  pipeline.py      # E5-large → Qdrant → BGE Reranker → Qwen3-32B
  api.py           # FastAPI эндпоинты (/ask, /search, /health)
  ui.py            # Streamlit интерфейс
  cache.py         # Redis кэш (TTL 1 час)
  config.py        # Pydantic Settings

data_pipeline/
  download.py      # Параллельное скачивание 10 HF-библиотек
  build_corpus.py  # Очистка .md → corpus.jsonl
  ingest.py        # Индексация в Qdrant
  run_all.py       # Оркестратор: download → build → ingest

docker/
  Dockerfile       # Python 3.10 + CPU PyTorch
  supervisord.conf # FastAPI + Streamlit в одном контейнере
docker-compose.yml # app + qdrant + redis
pyproject.toml     # Зависимости
```

## Требования

- Python 3.10+
- Docker & Docker Compose
- [Groq API key](https://console.groq.com/) (бесплатный тир)
- ~4 GB RAM (E5-large + BGE Reranker)
