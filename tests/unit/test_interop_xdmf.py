"""XDMF interop tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from lxml import etree

import pyfieldml as fml
from pyfieldml.builders.mesh import add_lagrange_mesh
from pyfieldml.interop.xdmf import write_xdmf
from pyfieldml.model.region import Region


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


def test_write_xdmf_produces_well_formed_xml_and_h5(tmp_path: Path) -> None:
    doc = _unit_cube_doc()
    xdmf = tmp_path / "cube.xdmf"
    write_xdmf(doc, xdmf)

    # XDMF file exists + parses
    assert xdmf.is_file()
    tree = etree.parse(str(xdmf))
    root = tree.getroot()
    assert root.tag == "Xdmf"

    # HDF5 sidecar exists with coords + conn
    h5 = tmp_path / "cube.h5"
    assert h5.is_file()
    with h5py.File(h5, "r") as f:
        assert "/coords" in f
        assert "/conn" in f
        coords = f["/coords"][()]
        conn = f["/conn"][()]
    assert coords.shape == (8, 3)
    assert conn.shape == (1, 8)


def test_write_xdmf_declares_hex_topology(tmp_path: Path) -> None:
    doc = _unit_cube_doc()
    xdmf = tmp_path / "hex.xdmf"
    write_xdmf(doc, xdmf)

    tree = etree.parse(str(xdmf))
    topo = tree.find(".//Topology")
    assert topo is not None
    assert topo.get("TopologyType") == "Hexahedron"
    assert topo.get("NumberOfElements") == "1"
