"""End-to-end CLI tests via subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

import pyfieldml as fml


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pyfieldml.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def test_inspect_prints_document_summary(fixtures_dir: Path) -> None:
    result = _run_cli("inspect", str(fixtures_dir / "minimal.fieldml"))
    assert result.returncode == 0
    assert "Region:" in result.stdout
    assert "Evaluators" in result.stdout


def test_validate_exits_zero_on_valid(fixtures_dir: Path) -> None:
    result = _run_cli("validate", str(fixtures_dir / "minimal.fieldml"))
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_validate_exits_nonzero_on_invalid(fixtures_dir: Path) -> None:
    result = _run_cli("validate", str(fixtures_dir / "invalid_no_name.fieldml"))
    assert result.returncode != 0
    assert "FAIL" in result.stdout


def test_lint_runs_without_error_on_clean_doc(fixtures_dir: Path, tmp_path: Path) -> None:
    # Write a fresh unit cube to tmp_path and lint it.
    from pyfieldml.builders.mesh import add_lagrange_mesh
    from pyfieldml.model.region import Region

    r = Region(name="cube")
    nodes = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    add_lagrange_mesh(
        r,
        name="m",
        nodes=nodes,
        elements=conn,
        topology="hex",
        order=1,
        coord_name="coords",
    )
    doc = fml.Document.from_region(r)
    p = tmp_path / "cube.fieldml"
    doc.write(p)

    result = _run_cli("lint", str(p))
    # Unit cube has no error-severity issues, so exit 0.
    assert result.returncode == 0


def test_diff_identical_documents_exits_zero(fixtures_dir: Path) -> None:
    result = _run_cli(
        "diff",
        str(fixtures_dir / "minimal.fieldml"),
        str(fixtures_dir / "minimal.fieldml"),
    )
    assert result.returncode == 0
    assert "identical" in result.stdout


def test_help_lists_all_subcommands() -> None:
    result = _run_cli("--help")
    assert result.returncode == 0
    for sub in ("bench", "inspect", "validate", "convert", "plot", "lint", "diff"):
        assert sub in result.stdout
