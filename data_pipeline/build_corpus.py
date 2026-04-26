"""
Сборка и очистка корпуса из скачанных .md файлов документации HuggingFace.

Читает из data_pipeline/raw/<library>/ (или --raw-dir).
Пишет JSONL в data_pipeline/corpus.jsonl (или --out).

Запуск:
  python data_pipeline/build_corpus.py
  python data_pipeline/build_corpus.py --out data_pipeline/corpus_test.jsonl --min-chars 200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# импортируем словарь всех библиотек из соседнего download.py
sys.path.insert(0, str(Path(__file__).parent))
from download import HF_LIBRARIES  # noqa: E402

# ── настройки по умолчанию ─────────────────────────────────────────────────────
DEFAULT_RAW_DIR = "data_pipeline/raw"
DEFAULT_OUT     = "data_pipeline/corpus.jsonl"
MIN_CHARS       = 150
MIN_LINES       = 3

# файлы без смысловой нагрузки
SKIP_SOURCES = {
    "_toctree.yml", "_config.py", "_redirects.yml",
    "index.md",
}


# ── очистка (идентична scripts/build_corpus.py) ───────────────────────────────
def clean(text: str) -> str:
    # 1. лицензионные заголовки (HTML-комментарии)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # 2. [[autodoc]] директивы
    text = re.sub(r'\[\[autodoc\]\].*?(\n|$)', '', text)
    # 3. YAML фронтматтер (--- ... ---)
    text = re.sub(r'^---.*?---\s*', '', text, flags=re.DOTALL)
    # 4. HTML-теги
    text = re.sub(r'<[^>]+>', '', text)
    # 5. markdown-ссылки → только текст
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # 6. изображения / бейджи
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # 7. множественные пустые строки → одна
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 8. строки из одних спецсимволов (разделители)
    lines = [l for l in text.splitlines()
             if not re.match(r'^[\s\-\*\_\=\#]{0,3}$', l)]
    return '\n'.join(lines).strip()


def is_useful(text: str, min_chars: int, min_lines: int) -> bool:
    if len(text) < min_chars:
        return False
    real_lines = [l for l in text.splitlines() if len(l.strip()) > 10]
    if len(real_lines) < min_lines:
        return False
    # >85% строк — чистый код → не несёт пользы для RAG
    code_lines = sum(
        1 for l in text.splitlines()
        if l.startswith('    ') or l.startswith('```')
    )
    total_lines = len(text.splitlines())
    if total_lines > 0 and code_lines / total_lines > 0.85:
        return False
    return True


# ── сбор документов ───────────────────────────────────────────────────────────
def build_corpus(
    raw_dir: Path,
    out_path: Path,
    min_chars: int,
    min_lines: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_raw       = 0
    total_kept      = 0
    skipped_short   = 0
    skipped_code    = 0
    stats: dict[str, int] = {}

    with open(out_path, "w", encoding="utf-8") as out_f:
        for library, cfg in HF_LIBRARIES.items():
            lib_raw_dir = raw_dir / library
            if not lib_raw_dir.exists():
                print(f"  [{library}] папка {lib_raw_dir} не найдена — пропускаю")
                continue

            # пробуем искать в подпапке docs_path, иначе по всему raw/<lib>
            docs_dir = lib_raw_dir / cfg["docs_path"]
            if not docs_dir.exists():
                docs_dir = lib_raw_dir
                print(f"  [{library}] docs_path не найден, ищу в {docs_dir}")

            md_files = sorted(docs_dir.rglob("*.md"))
            lib_kept = 0

            for md_file in md_files:
                if md_file.name in SKIP_SOURCES:
                    continue

                # пропускаем сломанные симлинки (git sparse-checkout их создаёт)
                if not md_file.is_file():
                    continue

                total_raw += 1
                try:
                    raw = md_file.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                cleaned = clean(raw)

                if not is_useful(cleaned, min_chars, min_lines):
                    if len(cleaned) < min_chars:
                        skipped_short += 1
                    else:
                        skipped_code += 1
                    continue

                source = str(md_file.relative_to(docs_dir))
                record = {
                    "text":    cleaned,
                    "source":  source,
                    "library": library,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_kept += 1
                lib_kept   += 1

            stats[library] = lib_kept
            print(f"  [{library}] сохранено {lib_kept} документов из {len(md_files)} .md")

    # итоговая статистика
    print(f"\n{'='*55}")
    print(f"  .md файлов обработано    : {total_raw}")
    print(f"  Документов сохранено     : {total_kept}")
    print(f"  Отфильтровано (кор.)     : {skipped_short}")
    print(f"  Отфильтровано (код)      : {skipped_code}")
    print(f"  Корпус записан в         : {out_path}")
    print(f"{'='*55}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сборка корпуса из скачанной документации HF"
    )
    parser.add_argument(
        "--raw-dir", default=DEFAULT_RAW_DIR,
        help=f"Папка с raw-данными (по умолч: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Путь к выходному JSONL (по умолч: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--min-chars", type=int, default=MIN_CHARS,
        help=f"Мин. символов после очистки (по умолч: {MIN_CHARS})",
    )
    parser.add_argument(
        "--min-lines", type=int, default=MIN_LINES,
        help=f"Мин. строк после очистки (по умолч: {MIN_LINES})",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    raw_dir  = base_dir / args.raw_dir
    out_path = base_dir / args.out

    print("🔍 Сборка корпуса из данных HuggingFace...")
    print(f"   Источник : {raw_dir}")
    print(f"   Выход    : {out_path}")
    print(f"   Мин. симв: {args.min_chars}\n")

    build_corpus(
        raw_dir   = raw_dir,
        out_path  = out_path,
        min_chars = args.min_chars,
        min_lines = args.min_lines,
    )


if __name__ == "__main__":
    main()