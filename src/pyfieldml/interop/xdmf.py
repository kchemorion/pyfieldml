"""Write FieldML documents as XDMF3 + HDF5 for ParaView / VisIt consumption.

One-way export. Supports Lagrange meshes produced by ``add_lagrange_mesh``.
Emits:
- <path>.xdmf: XML descriptor (referenced datasets by relative path)
- <path>.h5:   HDF5 sidecar with /coords, /conn, and per-field datasets
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from lxml import etree

# XDMF TopologyType strings per FieldML topology (order=1 only for Phase-4).
XDMF_TOPOLOGY_TYPE: dict[tuple[str, int], str] = {
    ("line", 1): "Polyline",
    ("triangle", 1): "Triangle",
    ("quad", 1): "Quadrilateral",
    ("tet", 1): "Tetrahedron",
    ("hex", 1): "Hexahedron",
    ("wedge", 1): "Wedge",
}


def write_xdmf(doc: Any, path: Any) -> None:
    """Write ``doc`` as an XDMF3 descriptor + HDF5 sidecar at ``path``."""
    from pyfieldml.interop.meshio import (
        _basis_topology_order,
        _find_basis_name,
        _find_coord_evaluator,
    )
    from pyfieldml.model.evaluators import ParameterEvaluator

    path = Path(path)
    h5_path = path.with_suffix(".h5")
    region = doc.region

    if not region.meshes:
        raise ValueError("Document has no MeshType; cannot write XDMF.")
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
    topo_key = (topology, order)
    if topo_key not in XDMF_TOPOLOGY_TYPE:
        raise NotImplementedError(
            f"XDMF writer: no topology mapping for (topology={topology!r}, order={order})"
        )

    conn_name = f"{coord_name}.connectivity"
    conn_ev = region.evaluators[conn_name]
    if not isinstance(conn_ev, ParameterEvaluator):
        raise ValueError(f"{conn_name!r} is not a ParameterEvaluator")
    conn = conn_ev.as_ndarray().astype(np.int64) - 1  # 0-indexed for XDMF

    reserved = {coord_name, conn_name, basis_name}
    reserved |= {f"{coord_name}.scales", f"{coord_name}.derivatives"}

    # Write heavy data.
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("/coords", data=coords)
        f.create_dataset("/conn", data=conn)
        for ev_name, ev in region.evaluators.items():
            if ev_name in reserved:
                continue
            if isinstance(ev, ParameterEvaluator):
                arr = ev.as_ndarray()
                if arr.shape and arr.shape[0] == coords.shape[0]:
                    f.create_dataset(f"/{ev_name}", data=arr)

    # Write XDMF descriptor.
    xdmf = etree.Element("Xdmf", Version="3.0")
    domain = etree.SubElement(xdmf, "Domain")
    grid = etree.SubElement(domain, "Grid", Name=mesh_name, GridType="Uniform")

    topo_elem = etree.SubElement(
        grid,
        "Topology",
        TopologyType=XDMF_TOPOLOGY_TYPE[topo_key],
        NumberOfElements=str(conn.shape[0]),
    )
    etree.SubElement(
        topo_elem,
        "DataItem",
        Format="HDF",
        Dimensions=f"{conn.shape[0]} {conn.shape[1]}",
        NumberType="Int",
        Precision="8",
    ).text = f"{h5_path.name}:/conn"

    geom = etree.SubElement(grid, "Geometry", GeometryType="XYZ")
    etree.SubElement(
        geom,
        "DataItem",
        Format="HDF",
        Dimensions=f"{coords.shape[0]} {coords.shape[1]}",
        NumberType="Float",
        Precision="8",
    ).text = f"{h5_path.name}:/coords"

    for ev_name, ev in region.evaluators.items():
        if ev_name in reserved:
            continue
        if isinstance(ev, ParameterEvaluator):
            arr = ev.as_ndarray()
            if arr.shape and arr.shape[0] == coords.shape[0]:
                attr = etree.SubElement(
                    grid,
                    "Attribute",
                    Name=ev_name,
                    AttributeType="Scalar" if arr.ndim == 1 else "Vector",
                    Center="Node",
                )
                dims = " ".join(str(d) for d in arr.shape)
                etree.SubElement(
                    attr,
                    "DataItem",
                    Format="HDF",
                    Dimensions=dims,
                    NumberType="Float",
                    Precision="8",
                ).text = f"{h5_path.name}:/{ev_name}"

    etree.ElementTree(xdmf).write(
        str(path),
        pretty_print=True,
        xml_declaration=True,
        encoding="UTF-8",
    )
