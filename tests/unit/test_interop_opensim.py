"""OpenSim-adjacent export tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")

import pyfieldml as fml  # noqa: E402
from pyfieldml.builders.mesh import add_lagrange_mesh  # noqa: E402
from pyfieldml.data.text import InlineTextBackend  # noqa: E402
from pyfieldml.interop.opensim import (  # noqa: E402
    export_fiber_lines,
    export_geometry,
    export_muscle_attachments,
)
from pyfieldml.model.evaluators import ParameterEvaluator  # noqa: E402
from pyfieldml.model.region import Region  # noqa: E402
from pyfieldml.model.types import ContinuousType  # noqa: E402


def _unit_cube_doc_with_fiber_and_landmarks() -> fml.Document:
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

    # Fiber field: 8 nodes, each pointing along +x.
    fiber_vectors = np.tile([1.0, 0.0, 0.0], (8, 1))
    fiber_vt = ContinuousType(
        name="fiber_direction.value_type",
        component_name="fiber_direction.component",
        component_count=3,
    )
    r.add_type(fiber_vt)
    r.add_evaluator(
        ParameterEvaluator(
            name="fiber_direction",
            value_type=fiber_vt,
            data=InlineTextBackend.from_ndarray(fiber_vectors),
        )
    )

    # Landmark set: 2 anatomical landmarks.
    lm_points = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]])
    lm_vt = ContinuousType(
        name="anatomical_landmarks.value_type",
        component_name="anatomical_landmarks.component",
        component_count=3,
    )
    r.add_type(lm_vt)
    r.add_evaluator(
        ParameterEvaluator(
            name="anatomical_landmarks",
            value_type=lm_vt,
            data=InlineTextBackend.from_ndarray(lm_points),
        )
    )
    return fml.Document.from_region(r)


def test_export_geometry_writes_vtu(tmp_path: Path) -> None:
    doc = _unit_cube_doc_with_fiber_and_landmarks()
    out = export_geometry(doc, tmp_path, filename="geom.vtp")
    assert out.is_file()
    # meshio can read it back
    m = meshio.read(out)
    assert m.points.shape == (8, 3)


def test_export_muscle_attachments_csv(tmp_path: Path) -> None:
    doc = _unit_cube_doc_with_fiber_and_landmarks()
    csv_path = export_muscle_attachments(doc, tmp_path / "landmarks.csv")
    lines = csv_path.read_text().splitlines()
    assert lines[0] == "label,x,y,z"
    assert "anatomical_landmarks_0" in lines[1]
    assert "0.5" in lines[1]


def test_export_fiber_lines_produces_vtp(tmp_path: Path) -> None:
    doc = _unit_cube_doc_with_fiber_and_landmarks()
    out = export_fiber_lines(doc, field="fiber_direction", out_path=tmp_path / "fibers.vtp")
    assert out.is_file()
    m = meshio.read(out)
    # 8 nodes x 2 endpoints = 16 points; 8 line cells.
    assert m.points.shape == (16, 3)
    assert m.cells[0].type == "line"
    assert m.cells[0].data.shape == (8, 2)
