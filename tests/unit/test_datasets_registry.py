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
    for expected in (
        "unit_cube",
        "femur",
        "rectus_femoris",
        "bunny_stanford",
        "femur_bodyparts3d",
    ):
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


def test_load_bunny_stanford() -> None:
    doc = datasets.load_bunny_stanford()
    assert "coordinates" in doc.evaluators
    coords = doc.evaluators["coordinates"]
    assert isinstance(coords, ParameterEvaluator)
    # Bunny should have at least a few hundred points
    assert coords.as_ndarray().shape[0] >= 100


def test_bunny_has_triangle_mesh() -> None:
    doc = datasets.load_bunny_stanford()
    # It's a surface mesh, so the basis should be linear Lagrange triangle
    basis_names = [n for n in doc.evaluators if n.startswith("library.basis.")]
    assert any("triangle" in n for n in basis_names)


def test_load_femur_bodyparts3d_if_bundled() -> None:
    # May skip if the BodyParts3D fetch failed during generation
    if "femur_bodyparts3d" not in datasets.list():
        pytest.skip("femur_bodyparts3d not bundled (fetch failed during generation)")
    doc = datasets.load_femur_bodyparts3d()
    assert "coordinates" in doc.evaluators


def test_femur_is_anatomical_not_cylindrical() -> None:
    """The new anatomical femur should have non-constant axial radius."""
    doc = datasets.load_femur()
    coords = doc.evaluators["coordinates"]
    assert isinstance(coords, ParameterEvaluator)
    pts = coords.as_ndarray()
    # Split into 4 z-slices and measure radial extent in each
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    bins = np.linspace(z_min, z_max, 5)
    radii = []
    for i in range(4):
        mask = (pts[:, 2] >= bins[i]) & (pts[:, 2] < bins[i + 1])
        slab = pts[mask]
        if len(slab) > 3:
            r = np.sqrt(slab[:, 0] ** 2 + slab[:, 1] ** 2).max()
            radii.append(r)
    # Anatomical femur has wider proximal/distal than mid-shaft; a cylinder
    # would have uniform radius. Assert at least 20% variation.
    assert max(radii) / min(radii) > 1.2, f"Femur radii too uniform (cylinder-like): {radii}"
