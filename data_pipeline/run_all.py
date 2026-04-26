"""
Полный pipeline: скачать документацию → собрать корпус → проиндексировать в Qdrant.

Запуск:
  python data_pipeline/run_all.py
  python data_pipeline/run_all.py --libs transformers diffusers --force
  python data_pipeline/run_all.py --qdrant-url http://localhost:6333
  python data_pipeline/run_all.py --skip-download   # если raw уже скачан
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent


def run_step(name: str, cmd: list[str]) -> bool:
    """Запускает шаг, возвращает True при успехе."""
    print(f"\n{'='*55}")
    print(f"  ▶  {name}")
    print(f"     {' '.join(cmd)}")
    print(f"{'='*55}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "ok" if ok else "not ok"
    print(f"\n  {status}  {name} завершён за {elapsed:.1f}с  (код {result.returncode})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Полный data pipeline: download → build_corpus → ingest"
    )
    parser.add_argument(
        "--libs", nargs="*", default=None,
        help="Список библиотек для скачивания (по умолч: все)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Перезаписать уже скачанные данные",
    )
    parser.add_argument(
        "--qdrant-url", default="http://localhost:6333",
        help="URL Qdrant (по умолч: http://localhost:6333)",
    )
    parser.add_argument(
        "--corpus", default="data_pipeline/corpus.jsonl",
        help="Путь к JSONL-корпусу (по умолч: data_pipeline/corpus.jsonl)",
    )
    parser.add_argument(
        "--raw-dir", default="data_pipeline/raw",
        help="Папка raw-данных (по умолч: data_pipeline/raw)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Пропустить шаг скачивания (использовать уже имеющийся raw)",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Пропустить сборку корпуса (использовать готовый corpus.jsonl)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Размер батча для Qdrant upsert (по умолч: 100)",
    )
    args = parser.parse_args()

    py = sys.executable
    steps_ok: list[str] = []
    steps_fail: list[str] = []

    total_start = time.time()

    # ── Шаг 1: Скачивание ────────────────────────────────────────────────────
    if not args.skip_download:
        cmd = [py, str(HERE / "download.py"), "--out", args.raw_dir]
        if args.libs:
            cmd += ["--libs"] + args.libs
        if args.force:
            cmd += ["--force"]

        if run_step("1/3 · Скачивание документации", cmd):
            steps_ok.append("download")
        else:
            steps_fail.append("download")
            print("\n Остановлено на шаге download. Проверьте ошибки выше.")
            sys.exit(1)
    else:
        print("\n Шаг 1 (download) пропущен (--skip-download)")

    # ── Шаг 2: Сборка корпуса ────────────────────────────────────────────────
    if not args.skip_build:
        cmd = [
            py, str(HERE / "build_corpus.py"),
            "--raw-dir", args.raw_dir,
            "--out",     args.corpus,
        ]

        if run_step("2/3 · Сборка и очистка корпуса", cmd):
            steps_ok.append("build_corpus")
        else:
            steps_fail.append("build_corpus")
            print("\n Остановлено на шаге build_corpus.")
            sys.exit(1)
    else:
        print("\n Шаг 2 (build_corpus) пропущен (--skip-build)")

    # ── Шаг 3: Индексация в Qdrant ───────────────────────────────────────────
    cmd = [
        py, str(HERE / "ingest.py"),
        "--corpus",     args.corpus,
        "--qdrant-url", args.qdrant_url,
        "--batch-size", str(args.batch_size),
    ]

    if run_step("3/3 · Индексация в Qdrant", cmd):
        steps_ok.append("ingest")
    else:
        steps_fail.append("ingest")
        print("\n Ошибка индексации. Убедитесь, что Qdrant запущен:")
        print("   docker-compose up -d qdrant")
        sys.exit(1)

    # ── Итог ─────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'='*55}")
    print(f"Pipeline завершён за {total_elapsed/60:.1f} мин")
    print(f"Успешно: {steps_ok}")
    if steps_fail:
        print(f"  Ошибки : {steps_fail}")
    print(f"{'='*55}")
    print("\nСервис можно запустить командой:")
    print("docker-compose up -d")
    print("API: http://localhost:8000/docs")
    print("UI : http://localhost:8501")


if __name__ == "__main__":
    main()
