"""Dataset registry + loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import pyfieldml as fml
from pyfieldml import datasets
from pyfieldml.model.evaluators import ParameterEvaluator


def test_list_contains_unit_cube() -> None:
    assert "unit_cube" in datasets.list()


def test_info_returns_metadata() -> None:
    meta = datasets.info("unit_cube")
    assert meta["name"] == "unit_cube"
    assert meta["license"] == "CC0-1.0"
    assert meta["citation"]
    assert meta["origin"]


def test_load_unit_cube_offline() -> None:
    doc = datasets.load_unit_cube()
    assert isinstance(doc, fml.Document)
    # Unit cube has 8 nodes via add_lagrange_mesh -> coordinates parameter.
    coords = doc.evaluators["coordinates"]
    assert isinstance(coords, ParameterEvaluator)
    arr = coords.as_ndarray()
    assert arr.shape == (8, 3)


def test_load_dispatches_by_name() -> None:
    doc = datasets.load("unit_cube")
    assert isinstance(doc, fml.Document)


def test_cache_dir_respects_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYFIELDML_CACHE", str(tmp_path))
    # Force a re-import so the env var takes effect? Actually cache_dir reads
    # at call-time, so just call it.
    assert datasets.cache_dir() == tmp_path
