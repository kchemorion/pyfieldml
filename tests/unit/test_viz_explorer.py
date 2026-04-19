"""Tests for the Jupyter graph-explorer widget. Skipped if ipywidgets/ipycytoscape not installed."""

from __future__ import annotations

import numpy as np
import pytest

widgets = pytest.importorskip("ipywidgets")
pytest.importorskip("ipycytoscape")

import pyfieldml as fml  # noqa: E402
from pyfieldml.builders.mesh import add_lagrange_mesh  # noqa: E402
from pyfieldml.model.region import Region  # noqa: E402
from pyfieldml.viz.explorer import explore  # noqa: E402


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


def test_explore_returns_widget() -> None:
    doc = _unit_cube_doc()
    w = explore(doc)
    # ipywidgets HBox is the outer container.
    assert isinstance(w, widgets.HBox)
    # Three children: tree, graph, detail.
    assert len(w.children) == 3


def test_document_explore_shortcut() -> None:
    doc = _unit_cube_doc()
    w = doc.explore()
    assert isinstance(w, widgets.HBox)
