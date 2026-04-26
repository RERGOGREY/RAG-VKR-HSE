"""
Скачивает документацию всех основных библиотек HuggingFace через git sparse-checkout.
Все библиотеки скачиваются ПАРАЛЛЕЛЬНО (ThreadPoolExecutor).
Каждая библиотека клонируется в data_pipeline/raw/<library>/.

Запуск:
  python data_pipeline/download.py
  python data_pipeline/download.py --libs transformers diffusers
  python data_pipeline/download.py --workers 5
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── все библиотеки HuggingFace с документацией ───────────────────────────────
HF_LIBRARIES = {
    "transformers": {
        "repo": "https://github.com/huggingface/transformers.git",
        "docs_path": "docs/source/en",
    },
    "diffusers": {
        "repo": "https://github.com/huggingface/diffusers.git",
        "docs_path": "docs/source/en",
    },
    "datasets": {
        "repo": "https://github.com/huggingface/datasets.git",
        "docs_path": "docs/source/en",
    },
    "accelerate": {
        "repo": "https://github.com/huggingface/accelerate.git",
        "docs_path": "docs/source/en",
    },
    "peft": {
        "repo": "https://github.com/huggingface/peft.git",
        "docs_path": "docs/source",
    },
    "trl": {
        "repo": "https://github.com/huggingface/trl.git",
        "docs_path": "docs/source",
    },
    "evaluate": {
        "repo": "https://github.com/huggingface/evaluate.git",
        "docs_path": "docs/source/en",
    },
    "tokenizers": {
        "repo": "https://github.com/huggingface/tokenizers.git",
        "docs_path": "docs/source/en",
    },
    "optimum": {
        "repo": "https://github.com/huggingface/optimum.git",
        "docs_path": "docs/source",
    },
    "hub-docs": {
        "repo": "https://github.com/huggingface/hub-docs.git",
        "docs_path": "docs/hub",
    },
}


# блокировка для безопасного вывода из нескольких потоков
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def run(cmd: list[str], cwd: Path | None = None, name: str = "") -> bool:
    """Запускает команду, возвращает True при успехе."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  [{name}] {' '.join(cmd[-2:])}: {result.stderr.strip()[:200]}")
        return False
    return True


def download_library(name: str, cfg: dict, raw_dir: Path, force: bool) -> bool:
    """
    Клонирует только папку с документацией через git sparse-checkout.
    Потокобезопасен: все print заменены на log().
    Возвращает True при успехе.
    """
    target = raw_dir / name

    if target.exists() and not force:
        log(f"  [{name}] уже скачан — пропускаю (--force для перезагрузки)")
        return True

    if target.exists() and force:
        shutil.rmtree(target)

    repo      = cfg["repo"]
    docs_path = cfg["docs_path"]

    log(f"  [{name}] -> клонирую {repo} (sparse: {docs_path})...")

    # 1. git init
    target.mkdir(parents=True)
    if not run(["git", "init"], cwd=target, name=name):
        return False

    # 2. remote add
    if not run(["git", "remote", "add", "origin", repo], cwd=target, name=name):
        return False

    # 3. sparse-checkout — только папка с docs
    if not run(["git", "sparse-checkout", "init", "--cone"], cwd=target, name=name):
        return False
    if not run(["git", "sparse-checkout", "set", docs_path], cwd=target, name=name):
        return False

    # 4. pull только последний коммит (depth=1 = быстро)
    if not run(["git", "pull", "--depth=1", "origin", "main"], cwd=target, name=name):
        # некоторые репо используют master
        if not run(["git", "pull", "--depth=1", "origin", "master"], cwd=target, name=name):
            return False

    n_md = len(list(target.rglob("*.md")))
    log(f"  [{name}] скачано {n_md} .md файлов → {target}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать документацию HF-библиотек (параллельно)")
    parser.add_argument(
        "--libs", nargs="*", default=list(HF_LIBRARIES.keys()),
        help="Список библиотек (по умолч. все)",
    )
    parser.add_argument(
        "--out", default="data_pipeline/raw",
        help="Папка для сырых данных (по умолч. data_pipeline/raw)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Перезаписать уже скачанные библиотеки",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Кол-во параллельных потоков (по умолч. 5)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    raw_dir  = base_dir / args.out
    raw_dir.mkdir(parents=True, exist_ok=True)

    libs = {k: v for k, v in HF_LIBRARIES.items() if k in args.libs}
    if not libs:
        print(f"Не найдено ни одной библиотеки из: {args.libs}")
        print(f"Доступные: {list(HF_LIBRARIES.keys())}")
        sys.exit(1)

    workers = min(args.workers, len(libs))
    print(f"Скачиваю документацию {len(libs)} библиотек HuggingFace (параллельно, {workers} потоков)...")
    print(f"Папка: {raw_dir}\n")

    ok: list[str] = []
    fail: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_library, name, cfg, raw_dir, args.force): name
            for name, cfg in libs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                success = future.result()
            except Exception as exc:
                log(f"  [{name}] неожиданная ошибка: {exc}")
                success = False
            (ok if success else fail).append(name)

    print(f"\n{'='*55}")
    print(f"  Успешно : {len(ok)}  → {sorted(ok)}")
    if fail:
        print(f"  Ошибки  : {len(fail)} → {fail}")
    print(f"{'='*55}")
    print("\nСледующий шаг:")
    print("  python data_pipeline/build_corpus.py")


if __name__ == "__main__":
    main()
