"""Spatial sampling tests."""

from __future__ import annotations

import numpy as np

import pyfieldml as fml
from pyfieldml.builders.mesh import add_lagrange_mesh
from pyfieldml.model.region import Region


def _make_unit_cube_doc() -> fml.Document:
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


def test_sample_on_unit_cube_returns_input_points() -> None:
    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    rng = np.random.default_rng(42)
    pts = rng.random((500, 3))
    vals = coords.sample(pts)
    np.testing.assert_allclose(vals, pts, atol=1e-10)


def test_sample_out_of_bounds_returns_nan() -> None:
    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    outside = np.array([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]])
    vals = coords.sample(outside)
    assert np.isnan(vals).all()
