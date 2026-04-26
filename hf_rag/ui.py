"""
Streamlit UI для HuggingFace Docs RAG.
Общается с FastAPI через HTTP.
"""
import time

import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── конфиг страницы ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HuggingFace Docs RAG",
    page_icon="🤗",
    layout="wide",
)

# ── боковая панель ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Настройки")
    top_k = st.slider("Контекстов для ответа", min_value=1, max_value=10, value=5)
    mode = st.radio("Режим", ["💬 Ответ + контекст", "🔍 Только поиск"])

    st.divider()
    st.subheader("📊 Статус сервисов")
    if st.button("Обновить статус"):
        try:
            h = requests.get(f"{API_URL}/health", timeout=3).json()
            st.success("API: ✅")
            col1, col2 = st.columns(2)
            col1.metric("Qdrant", "✅" if h["qdrant"] else "❌")
            col2.metric("Redis", "✅" if h["redis"] else "❌")
            st.caption(f"Модель: `{h['model']}`")
        except Exception as e:
            st.error(f"API недоступен: {e}")

    st.divider()
    st.subheader("🗄️ Кэш Redis")
    if st.button("Статистика кэша"):
        try:
            s = requests.get(f"{API_URL}/cache/stats", timeout=3).json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Hits", s["hits"])
            c2.metric("Misses", s["misses"])
            c3.metric("Keys", s["keys"])
        except Exception as e:
            st.error(str(e))
    if st.button("Очистить кэш", type="secondary"):
        try:
            requests.post(f"{API_URL}/cache/clear", timeout=3)
            st.success("Кэш очищен")
        except Exception as e:
            st.error(str(e))

# ── основной контент ──────────────────────────────────────────────────────────
st.title("🤗 HuggingFace Docs RAG")
st.caption(
    "RAG-система по документации **Transformers**, **Diffusers**, **Datasets** и других библиотек HuggingFace.  \n"
    "Поиск: `multilingual-E5-large` → `Qdrant` → `BGE Reranker Large`  \n"
    "Генерация: `Qwen3-32B` (Groq)"
)

st.divider()

# примеры вопросов
with st.expander("💡 Примеры вопросов"):
    examples = [
        "How do I fine-tune a model with the Trainer API?",
        "What is the difference between pipeline() and AutoModel?",
        "How to use LoRA for parameter-efficient fine-tuning?",
        "How does the Diffusers pipeline work for image generation?",
        "What tokenizer should I use for multilingual tasks?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

query = st.text_area(
    "Задайте вопрос по документации HuggingFace:",
    value=st.session_state.get("query_input", ""),
    height=80,
    placeholder="How do I fine-tune BERT for text classification?",
    key="query_input",
)

ask_btn = st.button("🚀 Найти и ответить", type="primary", use_container_width=True)

if ask_btn and query.strip():
    endpoint = "/ask" if "Ответ" in mode else "/search"

    with st.spinner("Ищу в документации..."):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{API_URL}{endpoint}",
                json={"query": query.strip(), "top_k": top_k},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("❌ Не удаётся подключиться к API. Убедитесь, что сервис запущен.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
            st.stop()

    elapsed = round((time.perf_counter() - t0) * 1000, 0)

    # ── ответ модели ──────────────────────────────────────────────────────────
    if "answer" in data:
        cache_badge = "🟢 из кэша" if data.get("answer_cache_hit") else "🔵 сгенерирован"
        st.subheader(f"💬 Ответ  {cache_badge}")
        st.markdown(data["answer"])
        st.caption(
            f"⏱ {elapsed} мс  •  модель: `{data.get('model', '—')}`  •  "
            f"контекст: {'🟢 кэш' if data.get('context_cache_hit') else '🔵 поиск'}"
        )

    # ── контексты ─────────────────────────────────────────────────────────────
    st.divider()
    ctx_label = "🔍 Найденные фрагменты" if "answer" not in data else "📄 Использованный контекст"
    cache_ctx = "🟢 из кэша" if data.get("cache_hit", data.get("context_cache_hit")) else "🔵 поиск"
    st.subheader(f"{ctx_label}  {cache_ctx}")

    for i, ctx in enumerate(data["contexts"], 1):
        score = ctx.get("reranker_score", 0)
        source = ctx.get("source", "—")
        library = ctx.get("library", "—")

        with st.expander(
            f"[{i}] {source}  •  `{library}`  •  score: `{score:.3f}`",
            expanded=(i == 1),
        ):
            st.markdown(ctx["text"])
            st.caption(
                f"doc_id: `{ctx.get('doc_id', '—')}`  •  "
                f"chunk: `{ctx.get('chunk_idx', '—')}`"
            )

elif ask_btn:
    st.warning("Введите вопрос.")
