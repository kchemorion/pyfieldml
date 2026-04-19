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


def add_hermite_mesh(
    region: Region,
    *,
    name: str,
    nodes: np.ndarray,
    elements: np.ndarray,
    derivatives: np.ndarray,
    scales: np.ndarray | None = None,
    topology: str = "line",
    coord_name: str = "coordinates",
) -> tuple[MeshType, ParameterEvaluator]:
    """Add a cubic-Hermite mesh to ``region``. Phase-3 line-only.

    Produces:
        - MeshType(name) with elements + xi chart
        - ParameterEvaluator(coord_name) for node coordinates - shape (N_nodes, D)
        - ParameterEvaluator(coord_name + '.derivatives') - shape (N_nodes, D)
        - ParameterEvaluator(coord_name + '.connectivity') - (n_elems, 2), 1-indexed
        - ParameterEvaluator(coord_name + '.scales') - (n_elems, 2); all-ones if scales is None
        - ExternalEvaluator("library.basis.cubic_hermite.line")

    Phase-3 simplification: one scalar scale per node per element (not per-DOF-pair-
    per-element as CMISS supports). Quad/hex Hermite builders are deferred and
    raise ``NotImplementedError``.
    """
    if topology != "line":
        raise NotImplementedError(
            f"Phase-3: add_hermite_mesh currently supports only topology='line', "
            f"got {topology!r}. Bicubic/tricubic Hermite builders are a later-phase task."
        )

    if nodes.ndim != 2:
        raise ValueError(f"nodes must be 2-D (N_nodes, D); got shape {nodes.shape}")
    if derivatives.shape != nodes.shape:
        raise ValueError(
            f"derivatives shape {derivatives.shape} must match nodes shape {nodes.shape}"
        )
    if elements.ndim != 2 or elements.shape[1] != 2:
        raise ValueError(
            f"elements must have shape (n_elems, 2) for a line topology; got {elements.shape}"
        )

    d = nodes.shape[1]
    n_elems = elements.shape[0]
    basis_name = "library.basis.cubic_hermite.line"

    if scales is None:
        scales = np.ones((n_elems, 2), dtype=np.float64)
    elif scales.shape != (n_elems, 2):
        raise ValueError(
            f"scales must have shape (n_elems, 2) = ({n_elems}, 2); got {scales.shape}"
        )

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

    nodes_data = InlineTextBackend.from_ndarray(nodes.astype(np.float64))
    coords = ParameterEvaluator(name=coord_name, value_type=vt, data=nodes_data)
    region.add_evaluator(coords)

    derivs_data = InlineTextBackend.from_ndarray(derivatives.astype(np.float64))
    derivs_ev = ParameterEvaluator(
        name=f"{coord_name}.derivatives", value_type=vt, data=derivs_data
    )
    region.add_evaluator(derivs_ev)

    conn_data = InlineTextBackend.from_ndarray(elements.astype(np.int64))
    conn_ct = ContinuousType(name=f"{coord_name}.connectivity.vt")
    region.add_type(conn_ct)
    conn = ParameterEvaluator(name=f"{coord_name}.connectivity", value_type=conn_ct, data=conn_data)
    region.add_evaluator(conn)

    scales_data = InlineTextBackend.from_ndarray(scales.astype(np.float64))
    scales_ct = ContinuousType(name=f"{coord_name}.scales.vt")
    region.add_type(scales_ct)
    scales_ev = ParameterEvaluator(
        name=f"{coord_name}.scales", value_type=scales_ct, data=scales_data
    )
    region.add_evaluator(scales_ev)

    region.add_evaluator(ExternalEvaluator(name=basis_name, value_type=vt))

    return mesh, coords
