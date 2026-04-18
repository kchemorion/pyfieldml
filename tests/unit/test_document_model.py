"""End-to-end tests exercising DOM → model loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pyfieldml as fml
from pyfieldml.model.evaluators import (
    AggregateEvaluator,
    ArgumentEvaluator,
    ConstantEvaluator,
    ParameterEvaluator,
    ReferenceEvaluator,
)


def test_parameter_inline_loads_with_correct_values(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "parameter_inline.fieldml")
    p = doc.evaluators["p"]
    assert isinstance(p, ParameterEvaluator)
    np.testing.assert_array_equal(p.as_ndarray(), [1.0, 2.0, 3.0, 4.0])


def test_reference_aggregate_loads_with_evaluator_subtypes(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "reference_aggregate.fieldml")
    assert isinstance(doc.evaluators["k"], ConstantEvaluator)
    assert isinstance(doc.evaluators["arg"], ArgumentEvaluator)
    assert isinstance(doc.evaluators["r"], ReferenceEvaluator)
    assert isinstance(doc.evaluators["agg"], AggregateEvaluator)


def test_document_exposes_region_views(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "reference_aggregate.fieldml")
    assert "real.1d" in doc.continuous
    assert "real.3d" in doc.continuous
    assert doc.continuous["real.3d"].component_count == 3
    assert set(doc.evaluators.keys()) == {"arg", "k", "r", "agg"}
