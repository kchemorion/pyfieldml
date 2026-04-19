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


_HERMITE_TOPOLOGY: dict[str, tuple[str, int, int, int]] = {
    # topology -> (basis_name, n_nodes_per_elem, n_deriv_slots, n_dofs_per_elem)
    "line": ("library.basis.cubic_hermite.line", 2, 1, 4),
    "quad": ("library.basis.bicubic_hermite.quad", 4, 3, 16),
    "hex": ("library.basis.tricubic_hermite.hex", 8, 7, 64),
}


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
    """Add a cubic-Hermite mesh to ``region``.

    Supports three topologies:
        - ``line``: 2 DOF kinds/node (value, d/dxi1); 4 DOFs/elem.
        - ``quad`` (bicubic): 4 DOF kinds/node (value, d/dxi1, d/dxi2, d2/dxi1dxi2);
          16 DOFs/elem.
        - ``hex`` (tricubic): 8 DOF kinds/node (value, d/dxi1, d/dxi2, d/dxi3,
          d2/dxi1dxi2, d2/dxi1dxi3, d2/dxi2dxi3, d3/dxi1dxi2dxi3); 64 DOFs/elem.

    DOF convention (CMISS-style per-corner ordering):

        Quad per-corner basis DOF order: (value, d/dxi1, d/dxi2, d2/dxi1dxi2).
        User-supplied ``derivatives[n]`` has shape ``(3, D)`` with slot order
        (d/dxi1, d/dxi2, d2/dxi1dxi2) — matching the basis order minus the value.

        Hex per-corner basis DOF order (f_u + 2*f_v + 4*f_w, f in {0,1}):
            0: value, 1: d/dxi1, 2: d/dxi2, 3: d2/dxi1dxi2,
            4: d/dxi3, 5: d2/dxi1dxi3, 6: d2/dxi2dxi3, 7: d3/dxi1dxi2dxi3.
        User-supplied ``derivatives[n]`` has shape ``(7, D)`` with slot order
        (d/dxi1, d/dxi2, d/dxi3, d2/dxi1dxi2, d2/dxi1dxi3, d2/dxi2dxi3,
        d3/dxi1dxi2dxi3). The evaluator reorders to basis DOF layout at assembly.

    Produces:
        - MeshType(name) with elements + xi chart
        - ParameterEvaluator(coord_name) for node coordinates, shape (N_nodes, D)
        - ParameterEvaluator(coord_name + '.derivatives'), shape
          (N_nodes, n_deriv_slots * D) flattened (n_deriv_slots=1 for line, 3 for
          quad, 7 for hex)
        - ParameterEvaluator(coord_name + '.connectivity'), (n_elems, n_per_elem),
          1-indexed
        - ParameterEvaluator(coord_name + '.scales'), (n_elems, n_dofs_per_elem)
          for quad/hex (all-ones default); (n_elems, 2) for line (legacy layout)
        - ExternalEvaluator naming the basis

    For line, scales retain the legacy (n_elems, 2) layout (one scalar per node
    per element, applied to derivative slots only) for backward compatibility.
    """
    if topology not in _HERMITE_TOPOLOGY:
        raise ValueError(
            f"add_hermite_mesh: unsupported topology {topology!r}. "
            f"Supported: {sorted(_HERMITE_TOPOLOGY)}."
        )
    basis_name, n_per_elem, n_slots, n_dofs_per_elem = _HERMITE_TOPOLOGY[topology]

    if nodes.ndim != 2:
        raise ValueError(f"nodes must be 2-D (N_nodes, D); got shape {nodes.shape}")
    d = nodes.shape[1]
    n_elems = elements.shape[0]

    # Validate derivatives shape. Line uses the legacy (N_nodes, D) layout.
    if topology == "line":
        if derivatives.shape != nodes.shape:
            raise ValueError(
                f"derivatives shape {derivatives.shape} must match nodes shape "
                f"{nodes.shape} for topology='line'"
            )
        # Internal flat storage: (N_nodes, 1*D) == (N_nodes, D) — no change.
        derivs_flat = derivatives.astype(np.float64)
    else:
        expected = (nodes.shape[0], n_slots, d)
        if derivatives.shape != expected:
            raise ValueError(
                f"derivatives shape {derivatives.shape} must match "
                f"(N_nodes, {n_slots}, D) = {expected} for topology={topology!r}"
            )
        # Flatten per-node to (n_slots * D): slot-major within each node.
        derivs_flat = derivatives.reshape(nodes.shape[0], n_slots * d).astype(np.float64)

    if elements.ndim != 2 or elements.shape[1] != n_per_elem:
        raise ValueError(
            f"elements must have shape (n_elems, {n_per_elem}) for a {topology!r} "
            f"topology; got {elements.shape}"
        )

    if topology == "line":
        if scales is None:
            scales = np.ones((n_elems, 2), dtype=np.float64)
        elif scales.shape != (n_elems, 2):
            raise ValueError(
                f"scales must have shape (n_elems, 2) = ({n_elems}, 2); got {scales.shape}"
            )
    else:
        if scales is None:
            scales = np.ones((n_elems, n_dofs_per_elem), dtype=np.float64)
        elif scales.shape != (n_elems, n_dofs_per_elem):
            raise ValueError(
                f"scales must have shape (n_elems, {n_dofs_per_elem}) = "
                f"({n_elems}, {n_dofs_per_elem}); got {scales.shape}"
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

    # For quad/hex, build a derivatives ContinuousType sized to (n_slots * D) so
    # downstream loaders/writers see the flat slot-major layout as the value
    # type. For line we retain the original coord value_type for backward-compat.
    if topology == "line":
        derivs_vt: ContinuousType = vt
    else:
        derivs_vt = ContinuousType(
            name=f"{coord_name}.derivatives.value_type",
            component_name=f"{coord_name}.derivatives.component",
            component_count=n_slots * d,
        )
        region.add_type(derivs_vt)

    derivs_data = InlineTextBackend.from_ndarray(derivs_flat)
    derivs_ev = ParameterEvaluator(
        name=f"{coord_name}.derivatives", value_type=derivs_vt, data=derivs_data
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
