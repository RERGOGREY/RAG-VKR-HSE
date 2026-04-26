FROM python:3.10-slim

# системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential supervisor curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-версия PyTorch (без CUDA) — иначе pip тянет ~3 GB NVIDIA-пакетов
RUN pip install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# зависимости Python (сначала — для кэширования слоёв)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# исходный код
COPY hf_rag/ ./hf_rag/
COPY scripts/ ./scripts/
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# корпус (добавь в .dockerignore если хочешь монтировать через volume)
COPY rag_corpus_cleaned_v2/ ./rag_corpus_cleaned_v2/

# порты: FastAPI + Streamlit
EXPOSE 8000 8501

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
