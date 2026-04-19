"""Tests for the meshio interop bridge. Skipped if meshio is not installed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")

import pyfieldml as fml  # noqa: E402
from pyfieldml.builders.mesh import add_lagrange_mesh  # noqa: E402
from pyfieldml.model.evaluators import ParameterEvaluator  # noqa: E402
from pyfieldml.model.region import Region  # noqa: E402


def _coord_array(doc: fml.Document) -> np.ndarray:
    ev = doc.region.evaluators["coordinates"]
    assert isinstance(ev, ParameterEvaluator)
    return ev.as_ndarray()


def _unit_cube_doc() -> fml.Document:
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
    connectivity = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    add_lagrange_mesh(
        r,
        name="cube_mesh",
        nodes=nodes,
        elements=connectivity,
        topology="hex",
        order=1,
        coord_name="coordinates",
    )
    return fml.Document.from_region(r)


def test_to_meshio_preserves_points_and_cells() -> None:
    doc = _unit_cube_doc()
    m = doc.to_meshio()
    assert m.points.shape == (8, 3)
    assert len(m.cells) == 1
    assert m.cells[0].type == "hexahedron"
    assert m.cells[0].data.shape == (1, 8)
    # 1-indexed -> 0-indexed
    np.testing.assert_array_equal(m.cells[0].data, [[0, 1, 2, 3, 4, 5, 6, 7]])


def test_to_meshio_carries_coord_point_data() -> None:
    doc = _unit_cube_doc()
    m = doc.to_meshio()
    assert "coordinates" in m.point_data
    np.testing.assert_array_equal(
        m.point_data["coordinates"],
        _coord_array(doc),
    )


def test_from_meshio_round_trips_unit_cube() -> None:
    import meshio as _meshio

    points = np.array(
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
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    m = _meshio.Mesh(points=points, cells=[("hexahedron", conn)])
    doc = fml.Document.from_meshio(m, name="imported")
    np.testing.assert_array_equal(_coord_array(doc), points)


def test_meshio_register_plugin(tmp_path: Path) -> None:
    """meshio.read should dispatch .fieldml through pyfieldml after _register()."""
    import meshio as _meshio

    from pyfieldml.interop.meshio import _register

    _register()

    doc = _unit_cube_doc()
    fieldml_path = tmp_path / "cube.fieldml"
    doc.write(fieldml_path)

    m = _meshio.read(fieldml_path)
    assert m.points.shape == (8, 3)


def test_meshio_auto_registers_without_explicit_register_call(tmp_path: Path) -> None:
    """``meshio.read('x.fieldml')`` must work with no explicit ``_register()`` call.

    Deregisters the format first, then re-triggers the module-level
    auto-registration via ``importlib.reload`` — this mirrors the real-world
    flow for users who only do ``import pyfieldml`` (which transitively imports
    ``pyfieldml.interop.meshio``) and then reach for ``meshio.read(...)`` in a
    downstream notebook.
    """
    import importlib

    import meshio as _meshio
    from meshio._helpers import reader_map

    import pyfieldml.interop.meshio as _interop_meshio

    # Scrub any prior registration so we genuinely re-exercise the auto-register path.
    _meshio.deregister_format("fieldml")
    assert "fieldml" not in reader_map

    # Re-running the pyfieldml.interop.meshio module must re-install the plugin.
    importlib.reload(_interop_meshio)
    assert "fieldml" in reader_map

    doc = _unit_cube_doc()
    fieldml_path = tmp_path / "cube.fieldml"
    doc.write(fieldml_path)

    m = _meshio.read(fieldml_path)
    assert m.points.shape == (8, 3)
