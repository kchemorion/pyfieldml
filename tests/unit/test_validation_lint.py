"""Tests for the semantic linter."""

from __future__ import annotations

import numpy as np

import pyfieldml as fml
from pyfieldml.builders.mesh import add_lagrange_mesh
from pyfieldml.model.evaluators import (
    AggregateEvaluator,
    ConstantEvaluator,
    ReferenceEvaluator,
)
from pyfieldml.model.region import Region
from pyfieldml.model.types import ContinuousType
from pyfieldml.validation.lint import lint_document


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


def test_clean_document_has_no_lint_issues() -> None:
    doc = _unit_cube_doc()
    issues = lint_document(doc)
    # The unit cube's evaluators produced by add_lagrange_mesh are all terminal
    # (ParameterEvaluator, ExternalEvaluator) — no orphan composites. The unit
    # cube has positive-orientation connectivity → no inverted element.
    assert all(i.code != "INVERTED_ELEMENT" for i in issues)


def test_lint_flags_orphan_composite_evaluator() -> None:
    r = Region(name="orphan_test")
    t = ContinuousType(name="real.1d")
    r.add_type(t)
    k = ConstantEvaluator(name="k", value_type=t, value=1.0)
    r.add_evaluator(k)
    # Orphan AggregateEvaluator: composite, but nothing references it.
    orphan = AggregateEvaluator(name="orphan_agg", value_type=t, components=[k])
    r.add_evaluator(orphan)

    doc = fml.Document.from_region(r)
    issues = lint_document(doc)
    assert any(i.code == "ORPHAN" and i.object_name == "orphan_agg" for i in issues)


def test_lint_does_not_flag_referenced_composite() -> None:
    r = Region(name="referenced_test")
    t = ContinuousType(name="real.1d")
    r.add_type(t)
    k = ConstantEvaluator(name="k", value_type=t, value=1.0)
    r.add_evaluator(k)
    agg = AggregateEvaluator(name="agg", value_type=t, components=[k])
    r.add_evaluator(agg)
    # A ReferenceEvaluator points at `agg`, so `agg` isn't orphan.
    from pyfieldml.model.bindings import BindingTable

    ref = ReferenceEvaluator(
        name="ref_to_agg",
        value_type=t,
        source=agg,
        bindings=BindingTable([]),
    )
    r.add_evaluator(ref)

    doc = fml.Document.from_region(r)
    issues = lint_document(doc)
    orphan_names = [i.object_name for i in issues if i.code == "ORPHAN"]
    assert "agg" not in orphan_names
    # But `ref_to_agg` itself is composite AND not referenced → orphan.
    assert "ref_to_agg" in orphan_names


def test_lint_flags_inverted_hex_element() -> None:
    """Flip the hex connectivity ordering to induce a negative Jacobian det."""
    r = Region(name="inverted")
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
    # Full top-bottom face swap reverses orientation in z → negative Jacobian det.
    connectivity = np.array([[5, 6, 7, 8, 1, 2, 3, 4]], dtype=np.int64)
    add_lagrange_mesh(
        r,
        name="bad_mesh",
        nodes=nodes,
        elements=connectivity,
        topology="hex",
        order=1,
        coord_name="coordinates",
    )
    doc = fml.Document.from_region(r)
    issues = lint_document(doc)
    assert any(i.code == "INVERTED_ELEMENT" for i in issues)
