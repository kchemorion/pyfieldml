"""Jacobian tests - check against the identity for the unit cube."""

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


def test_jacobian_is_identity_on_unit_cube() -> None:
    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    j_mat = coords.jacobian(element=1, xi=(0.5, 0.5, 0.5))
    np.testing.assert_allclose(j_mat, np.eye(3), atol=1e-12)


def test_jacobian_on_stretched_cube_scales_correctly() -> None:
    """Stretch the cube to 2x in x; Jacobian should be diag(2, 1, 1)."""
    r = Region(name="stretched")
    nodes = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [2, 1, 0],
            [0, 0, 1],
            [2, 0, 1],
            [0, 1, 1],
            [2, 1, 1],
        ],
        dtype=np.float64,
    )
    connectivity = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    add_lagrange_mesh(
        r,
        name="m",
        nodes=nodes,
        elements=connectivity,
        topology="hex",
        order=1,
        coord_name="coords",
    )
    doc = fml.Document.from_region(r)
    coords = doc.field("coords")
    j_mat = coords.jacobian(element=1, xi=(0.5, 0.5, 0.5))
    np.testing.assert_allclose(j_mat, np.diag([2.0, 1.0, 1.0]), atol=1e-12)
