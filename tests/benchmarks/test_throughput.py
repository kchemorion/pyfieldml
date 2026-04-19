"""Throughput check — vectorized eval must be >=50x a scalar Python loop."""

from __future__ import annotations

import time

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


def test_vectorized_is_at_least_50x_scalar() -> None:
    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    rng = np.random.default_rng(0)
    xi = rng.random((10_000, 3))
    elements = np.ones(10_000, dtype=np.int64)

    t0 = time.perf_counter()
    vec = coords.evaluate(element=elements, xi=xi)
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    scalar = np.stack([coords.evaluate(element=1, xi=xi[i]) for i in range(10_000)])
    t_scalar = time.perf_counter() - t0

    np.testing.assert_allclose(vec, scalar, atol=1e-12)
    speedup = t_scalar / t_vec
    assert speedup >= 50.0, f"vectorized speedup only {speedup:.1f}x — should be >= 50x"
