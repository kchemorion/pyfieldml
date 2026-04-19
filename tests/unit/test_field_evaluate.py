"""End-to-end field evaluation on a unit-cube single-hex mesh."""

from __future__ import annotations

import numpy as np

import pyfieldml as fml
from pyfieldml.eval.field import Field


def _make_unit_cube_doc() -> fml.Document:
    """Hand-build a single-element trilinear-Lagrange unit cube with coord field."""
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


def test_field_evaluate_at_corners_returns_node_positions() -> None:
    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    assert isinstance(coords, Field)
    node_xi = np.array(
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
    for i in range(8):
        got = coords.evaluate(element=1, xi=node_xi[i])
        np.testing.assert_allclose(got, node_xi[i], atol=1e-12)


def test_field_evaluate_at_centroid_returns_midpoint() -> None:
    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    got = coords.evaluate(element=1, xi=(0.5, 0.5, 0.5))
    np.testing.assert_allclose(got, [0.5, 0.5, 0.5], atol=1e-12)


def test_field_evaluate_vectorized() -> None:
    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    rng = np.random.default_rng(0)
    xi = rng.random((1000, 3))
    elements = np.full(1000, 1)
    got = coords.evaluate(element=elements, xi=xi)
    np.testing.assert_allclose(got, xi, atol=1e-12)
