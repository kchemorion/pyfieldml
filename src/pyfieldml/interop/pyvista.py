"""PyVista interop - convert FieldML documents to VTK UnstructuredGrid.

Only Lagrange meshes produced by ``add_lagrange_mesh`` are supported. For
Hermite meshes, sample the reconstructed geometry at node positions and
export a linear-Lagrange representation (loses basis fidelity, preserves
node positions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pyvista


# VTK cell type codes for topologies we support.
VTK_CELL_TYPE: dict[tuple[str, int], int] = {
    ("line", 1): 3,  # VTK_LINE
    ("line", 2): 21,  # VTK_QUADRATIC_EDGE
    ("triangle", 1): 5,  # VTK_TRIANGLE
    ("triangle", 2): 22,  # VTK_QUADRATIC_TRIANGLE
    ("quad", 1): 9,  # VTK_QUAD
    ("quad", 2): 28,  # VTK_BIQUADRATIC_QUAD
    ("tet", 1): 10,  # VTK_TETRA
    ("tet", 2): 24,  # VTK_QUADRATIC_TETRA
    ("hex", 1): 12,  # VTK_HEXAHEDRON
    ("hex", 2): 29,  # VTK_TRIQUADRATIC_HEXAHEDRON
    ("wedge", 1): 13,  # VTK_WEDGE
}


def to_pyvista(doc: Any) -> pyvista.UnstructuredGrid:
    """Convert a pyfieldml.Document to a PyVista UnstructuredGrid."""
    import pyvista as pv

    from pyfieldml.interop.meshio import (
        _basis_topology_order,
        _find_basis_name,
        _find_coord_evaluator,
    )
    from pyfieldml.model.evaluators import ParameterEvaluator

    region = doc.region
    if not region.meshes:
        raise ValueError("Document has no MeshType; cannot convert to pyvista.")

    mesh_name = next(iter(region.meshes))
    coord_name = _find_coord_evaluator(region)
    if coord_name is None:
        raise ValueError(f"No coordinate evaluator found for mesh {mesh_name!r}")

    coord_ev = region.evaluators[coord_name]
    if not isinstance(coord_ev, ParameterEvaluator):
        raise ValueError(f"{coord_name!r} is not a ParameterEvaluator")
    coords = coord_ev.as_ndarray().astype(np.float64)

    basis_name = _find_basis_name(region)
    topology, order = _basis_topology_order(basis_name)
    key = (topology, order)
    if key not in VTK_CELL_TYPE:
        raise NotImplementedError(
            f"PyVista bridge: no VTK mapping for (topology={topology!r}, order={order})"
        )

    conn_name = f"{coord_name}.connectivity"
    conn_ev = region.evaluators[conn_name]
    if not isinstance(conn_ev, ParameterEvaluator):
        raise ValueError(f"{conn_name!r} is not a ParameterEvaluator")
    conn = conn_ev.as_ndarray().astype(np.int64) - 1  # 1-indexed -> 0-indexed

    n_per_cell = conn.shape[1]
    # PyVista cells array is flattened: for each cell, [n_per_cell, node0, node1, ...].
    cells = np.hstack([np.full((conn.shape[0], 1), n_per_cell, dtype=np.int64), conn]).ravel()
    celltypes = np.full(conn.shape[0], VTK_CELL_TYPE[key], dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, coords)
    grid.point_data[coord_name] = coords

    # Extra ParameterEvaluators that look like per-node fields.
    reserved = {coord_name, conn_name, basis_name}
    # Also exclude connectivity + scales + derivatives if present.
    reserved |= {f"{coord_name}.scales", f"{coord_name}.derivatives"}
    for ev_name, ev in region.evaluators.items():
        if ev_name in reserved:
            continue
        if isinstance(ev, ParameterEvaluator):
            arr = ev.as_ndarray()
            if arr.shape and arr.shape[0] == coords.shape[0]:
                grid.point_data[ev_name] = arr

    return grid


def plot_doc(doc: Any, **kwargs: Any) -> Any:
    """Render a Document via PyVista. Returns whatever pyvista.plot returns."""
    grid = to_pyvista(doc)
    return grid.plot(**kwargs)


def plot_field(field: Any, doc: Any, **kwargs: Any) -> Any:
    """Shape-aware field plot - scalar/vector dispatch.

    ``doc`` is passed so we can build the full grid (Field doesn't carry a
    Document reference in Phase 2+).
    """
    import pyvista as pv

    grid = to_pyvista(doc)
    arr = field.as_ndarray() if hasattr(field, "as_ndarray") else None
    if arr is None:
        return grid.plot(**kwargs)
    if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[-1] == 1):
        grid.point_data[field.name] = arr
        return grid.plot(scalars=field.name, **kwargs)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        grid.point_data[field.name] = arr
        glyphs = grid.glyph(orient=field.name, scale=False, factor=0.1)
        plotter = pv.Plotter()
        plotter.add_mesh(grid, opacity=0.3)
        plotter.add_mesh(glyphs, color="red")
        return plotter.show(**kwargs)
    return grid.plot(**kwargs)
