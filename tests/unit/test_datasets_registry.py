"""Dataset registry + loader tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_list_contains_all_bundled() -> None:
    names = datasets.list()
    for expected in ("unit_cube", "femur", "rectus_femoris"):
        assert expected in names


def test_load_femur_has_material_field() -> None:
    doc = datasets.load_femur()
    assert "coordinates" in doc.evaluators
    assert "youngs_modulus" in doc.evaluators
    coords = doc.evaluators["coordinates"]
    material = doc.evaluators["youngs_modulus"]
    assert isinstance(coords, ParameterEvaluator)
    assert isinstance(material, ParameterEvaluator)
    assert material.as_ndarray().shape == (coords.as_ndarray().shape[0],)


def test_load_rectus_femoris_has_fiber_field() -> None:
    doc = datasets.load_rectus_femoris()
    assert "coordinates" in doc.evaluators
    assert "fiber_direction" in doc.evaluators
    fibers = doc.evaluators["fiber_direction"]
    assert isinstance(fibers, ParameterEvaluator)
    arr = fibers.as_ndarray()
    assert arr.shape[1] == 3  # 3-vector per node
    # All fibers along +z
    np.testing.assert_allclose(arr[:, 2], 1.0, atol=1e-12)
