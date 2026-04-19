#!/usr/bin/env python
"""Build the pyfieldml JupyterLite site.

Copies the canonical tutorial notebooks from ``docs/notebooks/`` into a
staging directory, then invokes ``jupyter lite build`` to produce a static
site under ``docs/jupyterlite/_output/``.

Run from any working directory; paths are resolved relative to this file.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
NOTEBOOKS_SRC = REPO_ROOT / "docs" / "notebooks"
CONTENT_DIR = HERE / "content"
OUTPUT_DIR = HERE / "_output"


def stage_notebooks() -> list[Path]:
    """Copy ``.ipynb`` files from ``docs/notebooks/`` into ``content/``.

    Returns the list of staged notebook paths.
    """
    if not NOTEBOOKS_SRC.is_dir():
        raise SystemExit(f"notebook source not found: {NOTEBOOKS_SRC}")

    if CONTENT_DIR.exists():
        shutil.rmtree(CONTENT_DIR)
    CONTENT_DIR.mkdir(parents=True)

    staged: list[Path] = []
    for nb in sorted(NOTEBOOKS_SRC.glob("*.ipynb")):
        dst = CONTENT_DIR / nb.name
        shutil.copy2(nb, dst)
        staged.append(dst)

    if not staged:
        raise SystemExit(
            f"no notebooks found in {NOTEBOOKS_SRC}; did B2 land its notebook updates?"
        )

    print(f"staged {len(staged)} notebook(s) into {CONTENT_DIR}:")
    for p in staged:
        print(f"  - {p.name}")
    return staged


def run_jupyter_lite() -> None:
    """Invoke ``jupyter lite build`` with fail-loud semantics."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "lite",
        "build",
        "--contents",
        str(CONTENT_DIR),
        "--output-dir",
        str(OUTPUT_DIR),
    ]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=HERE, check=False)
    if result.returncode != 0:
        raise SystemExit(f"jupyter lite build failed with exit code {result.returncode}")

    index = OUTPUT_DIR / "index.html"
    if not index.is_file():
        raise SystemExit(f"jupyter lite build reported success but {index} is missing")
    print(f"built site at {OUTPUT_DIR} (index.html = {index.stat().st_size} bytes)")


def main() -> None:
    stage_notebooks()
    run_jupyter_lite()


if __name__ == "__main__":
    main()
