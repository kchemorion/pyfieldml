"""Mesh builders — one call produces a complete Lagrange mesh graph."""

from __future__ import annotations

import numpy as np

from pyfieldml.data.text import InlineTextBackend
from pyfieldml.model.evaluators import ExternalEvaluator, ParameterEvaluator
from pyfieldml.model.region import Region
from pyfieldml.model.types import ContinuousType, EnsembleType, MeshType

TOPOLOGY_BASIS_NAME: dict[tuple[str, int], str] = {
    ("line", 1): "library.basis.linear_lagrange.line",
    ("line", 2): "library.basis.quadratic_lagrange.line",
    ("triangle", 1): "library.basis.linear_lagrange.triangle",
    ("triangle", 2): "library.basis.quadratic_lagrange.triangle",
    ("quad", 1): "library.basis.linear_lagrange.quad",
    ("quad", 2): "library.basis.quadratic_lagrange.quad",
    ("tet", 1): "library.basis.linear_lagrange.tet",
    ("tet", 2): "library.basis.quadratic_lagrange.tet",
    ("hex", 1): "library.basis.linear_lagrange.hex",
    ("hex", 2): "library.basis.quadratic_lagrange.hex",
    ("wedge", 1): "library.basis.linear_lagrange.wedge",
    ("wedge", 2): "library.basis.quadratic_lagrange.wedge",
}

_TOPOLOGY_XI_DIM: dict[str, int] = {
    "line": 1,
    "triangle": 2,
    "quad": 2,
    "tet": 3,
    "hex": 3,
    "wedge": 3,
}


def add_lagrange_mesh(
    region: Region,
    *,
    name: str,
    nodes: np.ndarray,
    elements: np.ndarray,
    topology: str,
    order: int,
    coord_name: str = "coordinates",
) -> tuple[MeshType, ParameterEvaluator]:
    """Add a complete Lagrange mesh to ``region``.

    Produces types (ContinuousType for coord value type, two EnsembleTypes,
    ContinuousType for xi chart, MeshType), a ParameterEvaluator for node
    coords, a ParameterEvaluator for element→node connectivity (1-indexed),
    and an ExternalEvaluator naming the Lagrange basis.
    """
    d = nodes.shape[1]
    n_elems = elements.shape[0]
    basis_name = TOPOLOGY_BASIS_NAME[(topology, order)]

    vt = ContinuousType(
        name=f"{coord_name}.value_type",
        component_name=f"{coord_name}.component",
        component_count=d,
    )
    region.add_type(vt)

    elem_ens = EnsembleType(
        name=f"{name}.elements",
        members=range(1, n_elems + 1),  # type: ignore[arg-type]
    )
    region.add_type(elem_ens)
    node_ens = EnsembleType(
        name=f"{name}.nodes",
        members=range(1, nodes.shape[0] + 1),  # type: ignore[arg-type]
    )
    region.add_type(node_ens)
    chart_ct = ContinuousType(
        name=f"{name}.xi",
        component_name=f"{name}.xi.c",
        component_count=_TOPOLOGY_XI_DIM[topology],
    )
    region.add_type(chart_ct)
    mesh = MeshType(name=name, elements=elem_ens, chart=chart_ct)
    region.add_type(mesh)

    nodes_data = InlineTextBackend.from_ndarray(nodes)
    coords = ParameterEvaluator(name=coord_name, value_type=vt, data=nodes_data)
    region.add_evaluator(coords)

    conn_data = InlineTextBackend.from_ndarray(elements.astype(np.int64))
    conn_ct = ContinuousType(name=f"{coord_name}.connectivity.vt")
    region.add_type(conn_ct)
    conn = ParameterEvaluator(name=f"{coord_name}.connectivity", value_type=conn_ct, data=conn_data)
    region.add_evaluator(conn)

    region.add_evaluator(ExternalEvaluator(name=basis_name, value_type=vt))

    return mesh, coords
