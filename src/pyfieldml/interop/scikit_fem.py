"""scikit-fem bridge — convert a FieldML Lagrange mesh to skfem.Mesh + Basis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import skfem  # noqa: F401


_SKFEM_MESH: dict[tuple[str, int], type] = {}
_SKFEM_ELEMENT: dict[tuple[str, int], type] = {}


def to_scikit_fem(doc: Any, *, mesh_name: str | None = None) -> tuple[Any, Any]:
    """Return (skfem.Mesh, skfem.Basis) for the named (or first) mesh.

    Phase-4: Lagrange linear meshes only. Raises NotImplementedError for
    unsupported topology+order combinations.
    """
    import skfem

    from pyfieldml.interop.meshio import (
        _basis_topology_order,
        _find_basis_name,
        _find_coord_evaluator,
    )
    from pyfieldml.model.evaluators import ParameterEvaluator

    _ensure_tables_loaded()

    region = doc.region
    if not region.meshes:
        raise ValueError("Document has no MeshType; cannot convert to scikit-fem.")

    mesh_name = mesh_name or next(iter(region.meshes))
    coord_name = _find_coord_evaluator(region)
    if coord_name is None:
        raise ValueError(f"No coordinate evaluator found for mesh {mesh_name!r}")

    coord_ev = region.evaluators[coord_name]
    if not isinstance(coord_ev, ParameterEvaluator):
        raise ValueError(f"{coord_name!r} is not a ParameterEvaluator")
    coords = coord_ev.as_ndarray().astype(np.float64)

    basis_name = _find_basis_name(region)
    topology, order = _basis_topology_order(basis_name)

    conn_name = f"{coord_name}.connectivity"
    conn_ev = region.evaluators[conn_name]
    if not isinstance(conn_ev, ParameterEvaluator):
        raise ValueError(f"{conn_name!r} is not a ParameterEvaluator")
    conn = conn_ev.as_ndarray().astype(np.int64) - 1  # 0-indexed

    mesh_cls = _SKFEM_MESH.get((topology, order))
    elem_cls = _SKFEM_ELEMENT.get((topology, order))
    if mesh_cls is None or elem_cls is None:
        raise NotImplementedError(
            f"scikit-fem bridge: (topology={topology!r}, order={order}) not yet supported. "
            f"Phase-4 supports linear Lagrange on tet/hex/tri/quad/line."
        )

    # scikit-fem conventions: points (D, n_points), elements (n_per, n_elems).
    m = mesh_cls(coords.T, conn.T)
    basis = skfem.Basis(m, elem_cls())
    return m, basis


def _build_tables() -> tuple[dict[tuple[str, int], type], dict[tuple[str, int], type]]:
    """Build the topology+order → MeshClass / ElementClass lookup at import time.

    Quadratic (order 2) entries use scikit-fem's P1 mesh classes combined
    with P2/S2 element classes: scikit-fem keeps mesh topology independent
    of basis order, so a linear mesh + quadratic basis is canonical.
    ``ElementHexP2`` / ``ElementQuadP2`` are not in scikit-fem; serendipity
    variants ``ElementHexS2`` / ``ElementQuadS2`` are the closest fit.
    """
    import skfem

    mesh_table: dict[tuple[str, int], type] = {
        ("tet", 1): skfem.MeshTet,
        ("hex", 1): skfem.MeshHex,
        ("triangle", 1): skfem.MeshTri,
        ("quad", 1): skfem.MeshQuad,
        ("line", 1): skfem.MeshLine,
        # Quadratic: keep a P1 mesh, let the Element do the P2 work.
        ("tet", 2): skfem.MeshTet1,
        ("hex", 2): skfem.MeshHex1,
        ("triangle", 2): skfem.MeshTri1,
        ("quad", 2): skfem.MeshQuad1,
    }
    elem_table: dict[tuple[str, int], type] = {
        ("tet", 1): skfem.ElementTetP1,
        ("hex", 1): skfem.ElementHex1,
        ("triangle", 1): skfem.ElementTriP1,
        ("quad", 1): skfem.ElementQuad1,
        ("line", 1): skfem.ElementLineP1,
        ("tet", 2): skfem.ElementTetP2,
        ("hex", 2): skfem.ElementHexS2,  # serendipity; no ElementHexP2 in skfem
        ("triangle", 2): skfem.ElementTriP2,
        ("quad", 2): skfem.ElementQuadS2,  # serendipity; no ElementQuadP2 in skfem
    }
    return mesh_table, elem_table


def _ensure_tables_loaded() -> None:
    """Populate the lookup tables on first call.

    Module-level lazy init so that importing this module doesn't require
    scikit-fem to be installed.
    """
    global _SKFEM_MESH, _SKFEM_ELEMENT
    if _SKFEM_MESH:
        return
    _SKFEM_MESH, _SKFEM_ELEMENT = _build_tables()
