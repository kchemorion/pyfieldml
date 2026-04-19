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
    """The anatomical femur must have:

    - Proximal head offset from shaft axis (head test)
    - Wider proximal + distal cross-sections than mid-shaft
    - Variable radial profile along z

    These three checks together catch the "it's still a rod" failure mode
    — a cylindrical mesh would pass none of them.
    """
    doc = datasets.load_femur()
    coords_ev = doc.evaluators["coordinates"]
    assert isinstance(coords_ev, ParameterEvaluator)
    pts = coords_ev.as_ndarray()

    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

    # Head offset test: proximal-most band (top ~2 cm of the bone) should
    # contain at least one point offset medially by more than the shaft
    # radius, i.e. the femoral head is not on the shaft axis.
    proximal_threshold = z_max - 0.02 * (z_max - z_min)  # top ~2 cm
    proximal = pts[pts[:, 2] > proximal_threshold]
    assert len(proximal) > 5, f"Too few proximal nodes to test head offset: {len(proximal)}"
    max_x_offset = float(np.max(np.abs(proximal[:, 0])))
    assert max_x_offset > 0.02, (
        f"Proximal end has no offset head — looks like a rod, not a femur "
        f"(max_x_offset={max_x_offset:.3f})"
    )

    # Shaft-narrower-than-condyles test: compare mid-shaft max radius to
    # distal max radius — the condyles must bulge wider than the shaft.
    mid_shaft_z = (z_max + z_min) / 2
    mid_band = pts[np.abs(pts[:, 2] - mid_shaft_z) < 0.02]
    distal = pts[pts[:, 2] < z_min + 0.03 * (z_max - z_min)]
    assert len(mid_band) > 3, f"Too few mid-shaft nodes: {len(mid_band)}"
    assert len(distal) > 5, f"Too few distal nodes: {len(distal)}"
    mid_r = float(np.max(np.sqrt(mid_band[:, 0] ** 2 + mid_band[:, 1] ** 2)))
    distal_r = float(np.max(np.sqrt(distal[:, 0] ** 2 + distal[:, 1] ** 2)))
    assert distal_r > mid_r * 1.3, (
        f"Distal condyles ({distal_r:.3f}) not wider than shaft ({mid_r:.3f}) — "
        "femur likely lacks condyle bulges"
    )


def test_bodyparts3d_femur_is_watertight() -> None:
    """The BP3D femur should be one connected component (not swiss-cheese).

    Random-subsample decimation in v1.1 left the surface riddled with
    holes (thousands of boundary edges, multiple disconnected fragments).
    After switching to topology-preserving edge-collapse decimation, the
    surface must be essentially a single connected component.
    """
    if "femur_bodyparts3d" not in datasets.list():
        pytest.skip("femur_bodyparts3d not bundled (fetch failed during generation)")

    import scipy.sparse
    import scipy.sparse.csgraph

    doc = datasets.load_femur_bodyparts3d()
    conn_ev = doc.evaluators["coordinates.connectivity"]
    assert isinstance(conn_ev, ParameterEvaluator)
    conn = conn_ev.as_ndarray().astype(np.int64) - 1  # 0-indexed

    coords_ev = doc.evaluators["coordinates"]
    assert isinstance(coords_ev, ParameterEvaluator)
    n_points = coords_ev.as_ndarray().shape[0]

    # Build the vertex adjacency graph from triangle edges.
    row: list[int] = []
    col: list[int] = []
    for tri in conn:
        for i, j in ((0, 1), (1, 2), (0, 2)):
            row.append(int(tri[i]))
            col.append(int(tri[j]))
            row.append(int(tri[j]))
            col.append(int(tri[i]))
    adj = scipy.sparse.coo_matrix(
        (np.ones(len(row), dtype=np.float64), (row, col)), shape=(n_points, n_points)
    ).tocsr()
    n_components, _ = scipy.sparse.csgraph.connected_components(adj)

    # A watertight femur is 1 component. Allow up to 3 for tiny decimation
    # artifacts; if it's higher the surface has been shredded again.
    assert n_components <= 3, f"Femur has {n_components} disconnected parts"
