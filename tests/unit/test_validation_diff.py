"""Tests for semantic Document diff."""

from __future__ import annotations

import numpy as np

import pyfieldml as fml
from pyfieldml.builders.mesh import add_lagrange_mesh
from pyfieldml.data.text import InlineTextBackend
from pyfieldml.model.evaluators import ParameterEvaluator
from pyfieldml.model.region import Region
from pyfieldml.validation.diff import diff_documents


def _build_cube_doc(coord_offset: float) -> fml.Document:
    r = Region(name="t")
    nodes = (
        np.array(
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
        + coord_offset
    )
    conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    add_lagrange_mesh(
        r,
        name="m",
        nodes=nodes,
        elements=conn,
        topology="hex",
        order=1,
        coord_name="c",
    )
    # Overwrite the coord parameter's data to ensure both docs
    # share the same evaluator names but have different parameter values.
    new_data = InlineTextBackend.from_ndarray(nodes)
    coords_ev = r.evaluators["c"]
    assert isinstance(coords_ev, ParameterEvaluator)
    coords_ev.data = new_data
    return fml.Document.from_region(r)


def test_diff_identical_documents_is_empty() -> None:
    a = _build_cube_doc(0.0)
    b = _build_cube_doc(0.0)
    d = diff_documents(a, b)
    assert not d.has_differences()


def test_diff_reports_parameter_value_differences() -> None:
    a = _build_cube_doc(0.0)
    b = _build_cube_doc(0.5)
    d = diff_documents(a, b)
    assert d.has_differences()
    kinds = [e.kind for e in d.entries]
    assert "parameter_values_differ" in kinds


def test_diff_reports_missing_evaluators() -> None:
    a = _build_cube_doc(0.0)

    r = Region(name="smaller")
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
    conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    add_lagrange_mesh(
        r,
        name="m",
        nodes=nodes,
        elements=conn,
        topology="hex",
        order=1,
        coord_name="c",
    )
    # Remove the basis external evaluator from `b`.
    basis_name = next(n for n in r.evaluators if n.startswith("library.basis."))
    del r.evaluators[basis_name]
    del r._objects[basis_name]

    b = fml.Document.from_region(r)
    d = diff_documents(a, b)
    assert any(e.kind == "missing_evaluator" and e.name == basis_name for e in d.entries)
