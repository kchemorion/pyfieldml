"""meshio interop - two-way conversion between FieldML documents and meshio.Mesh.

Lossy in both directions: meshio has no evaluator graphs, and FieldML's
arbitrary-basis machinery doesn't map cleanly to meshio's fixed cell types.
For v0.4: only Lagrange meshes produced by add_lagrange_mesh are supported.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import meshio

FIELDML_TO_MESHIO: dict[tuple[str, int], str] = {
    ("line", 1): "line",
    ("line", 2): "line3",
    ("triangle", 1): "triangle",
    ("triangle", 2): "triangle6",
    ("quad", 1): "quad",
    ("quad", 2): "quad9",
    ("tet", 1): "tetra",
    ("tet", 2): "tetra10",
    ("hex", 1): "hexahedron",
    ("hex", 2): "hexahedron27",
    ("wedge", 1): "wedge",
}

MESHIO_TO_FIELDML: dict[str, tuple[str, int]] = {v: k for k, v in FIELDML_TO_MESHIO.items()}


def to_meshio(doc: Any) -> meshio.Mesh:
    """Convert a pyfieldml.Document to a meshio.Mesh (primary Lagrange mesh only)."""
    import meshio as _meshio

    region = doc.region
    if not region.meshes:
        raise ValueError("Document has no MeshType; cannot convert to meshio.")
    mesh_name, _mesh_type = next(iter(region.meshes.items()))

    coord_name = _find_coord_evaluator(region)
    if coord_name is None:
        raise ValueError(f"No coordinate evaluator found for mesh {mesh_name!r}")
    coords_arr = region.evaluators[coord_name].as_ndarray()

    conn_name = f"{coord_name}.connectivity"
    if conn_name not in region.evaluators:
        raise ValueError(f"No connectivity parameter {conn_name!r}")
    connectivity = region.evaluators[conn_name].as_ndarray().astype(np.int64) - 1

    basis_name = _find_basis_name(region)
    topology, order = _basis_topology_order(basis_name)
    cell_type = FIELDML_TO_MESHIO[(topology, order)]

    from pyfieldml.model.evaluators import ParameterEvaluator

    point_data: dict[str, np.ndarray] = {coord_name: coords_arr}
    for ev_name, ev in region.evaluators.items():
        if ev_name in {coord_name, conn_name, basis_name}:
            continue
        if isinstance(ev, ParameterEvaluator):
            arr = ev.as_ndarray()
            if arr.shape and arr.shape[0] == coords_arr.shape[0]:
                point_data[ev_name] = arr

    return _meshio.Mesh(
        points=coords_arr,
        cells=[(cell_type, connectivity)],
        point_data=point_data,
    )


def from_meshio(mesh: meshio.Mesh, *, name: str = "imported") -> Any:
    """Convert a meshio.Mesh into a pyfieldml.Document. First cell block only."""
    import pyfieldml as _fml
    from pyfieldml.builders.mesh import add_lagrange_mesh
    from pyfieldml.model.region import Region

    if not mesh.cells:
        raise ValueError("meshio mesh has no cell blocks")

    block = mesh.cells[0]
    cell_type = block.type
    if cell_type not in MESHIO_TO_FIELDML:
        raise NotImplementedError(
            f"meshio cell type {cell_type!r} not supported by the FieldML bridge"
        )
    topology, order = MESHIO_TO_FIELDML[cell_type]
    connectivity = block.data.astype(np.int64) + 1  # 0-indexed -> 1-indexed

    r = Region(name=name)
    add_lagrange_mesh(
        r,
        name=f"{name}_mesh",
        nodes=mesh.points.astype(np.float64),
        elements=connectivity,
        topology=topology,
        order=order,
        coord_name="coordinates",
    )
    return _fml.Document.from_region(r)


# ---------- helpers ----------


def _find_coord_evaluator(region: Any) -> str | None:
    """First ParameterEvaluator whose value_type has component_count == mesh.dimension."""
    from pyfieldml.model.evaluators import ParameterEvaluator
    from pyfieldml.model.types import ContinuousType

    if not region.meshes:
        return None
    mesh = next(iter(region.meshes.values()))
    for name, ev in region.evaluators.items():
        if (
            isinstance(ev, ParameterEvaluator)
            and isinstance(ev.value_type, ContinuousType)
            and ev.value_type.component_count == mesh.dimension
        ):
            return str(name)
    return None


def _find_basis_name(region: Any) -> str:
    for name in region.evaluators:
        if name.startswith("library.basis."):
            return str(name)
    raise ValueError("No library.basis.* evaluator in region")


def _basis_topology_order(basis_name: str) -> tuple[str, int]:
    # library.basis.<order-kind>_<family>.<topology>
    # e.g. library.basis.linear_lagrange.hex -> (hex, 1)
    parts = basis_name.split(".")
    # parts: ["library", "basis", "<order>_<family>", "<topology>"]
    order_tag = parts[2].split("_")[0]
    topology = parts[3]
    order_map = {"linear": 1, "quadratic": 2, "cubic": 3}
    return topology, order_map[order_tag]


# ---------- meshio plugin registration ----------


def _register() -> None:
    """Register pyfieldml as a meshio file format for '.fieldml' files."""
    import meshio as _meshio

    _meshio.register_format("fieldml", [".fieldml"], _reader, {"fieldml": _writer})


def _reader(path: Any) -> meshio.Mesh:
    import pyfieldml as _fml

    return to_meshio(_fml.read(path))


def _writer(path: Any, mesh: meshio.Mesh, **_kwargs: Any) -> None:
    import pyfieldml as _fml

    doc = _fml.Document.from_meshio(mesh, name="imported")
    doc.write(path)


# Auto-register with meshio on import so ``meshio.read("x.fieldml")`` works
# without requiring a manual ``_register()`` call. meshio's plugin mechanism
# (``register_format``) is not entry-point-driven, so the canonical hook-point
# is this module's import. Silent no-op if meshio isn't installed — the module
# is only reachable once the user has already imported it — and defensive
# against a partially-initialised meshio (never block ``import pyfieldml``).
with contextlib.suppress(Exception):  # pragma: no cover
    _register()
