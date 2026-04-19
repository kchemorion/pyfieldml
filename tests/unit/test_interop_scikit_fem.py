"""scikit-fem bridge tests. Skipped if scikit-fem is not installed."""

from __future__ import annotations

import numpy as np
import pytest

skfem = pytest.importorskip("skfem")

import pyfieldml as fml  # noqa: E402
from pyfieldml.builders.mesh import add_lagrange_mesh  # noqa: E402
from pyfieldml.interop.scikit_fem import to_scikit_fem  # noqa: E402
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


def test_to_scikit_fem_unit_cube_returns_mesh_and_basis() -> None:
    doc = _unit_cube_doc()
    m, basis = to_scikit_fem(doc)
    assert isinstance(m, skfem.MeshHex)
    assert m.nvertices == 8
    assert m.nelements == 1
    # Basis type check
    assert isinstance(basis, skfem.Basis)
