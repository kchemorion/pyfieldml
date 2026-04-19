"""Tests for the PyVista interop bridge. Skipped if pyvista is not installed."""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

import pyfieldml as fml  # noqa: E402
from pyfieldml.builders.mesh import add_lagrange_mesh  # noqa: E402
from pyfieldml.interop.pyvista import to_pyvista  # noqa: E402
from pyfieldml.model.region import Region  # noqa: E402


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


def test_to_pyvista_produces_unstructured_grid() -> None:
    doc = _unit_cube_doc()
    grid = to_pyvista(doc)
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_points == 8
    assert grid.n_cells == 1
    assert "coordinates" in grid.point_data


def test_to_pyvista_cell_type_is_hexahedron() -> None:
    doc = _unit_cube_doc()
    grid = to_pyvista(doc)
    # VTK_HEXAHEDRON = 12
    assert grid.celltypes[0] == 12
