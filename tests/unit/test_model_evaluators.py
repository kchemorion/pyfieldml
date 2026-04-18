"""Tests for the Evaluator class hierarchy."""

from __future__ import annotations

import numpy as np

from pyfieldml.data.text import InlineTextBackend
from pyfieldml.model.bindings import ArgumentBinding, BindingTable
from pyfieldml.model.evaluators import (
    AggregateEvaluator,
    ArgumentEvaluator,
    ConstantEvaluator,
    Evaluator,
    ExternalEvaluator,
    ParameterEvaluator,
    PiecewiseEvaluator,
    ReferenceEvaluator,
)
from pyfieldml.model.types import ContinuousType


def test_parameter_evaluator_exposes_ndarray() -> None:
    value_type = ContinuousType(name="real.1d")
    data = InlineTextBackend(text="1 2 3 4", shape=(4,), dtype="float64")
    p = ParameterEvaluator(name="p", value_type=value_type, data=data)
    np.testing.assert_array_equal(p.as_ndarray(), [1, 2, 3, 4])
    assert p.shape == (4,)
    assert p.dtype == np.float64


def test_constant_evaluator_scalar() -> None:
    t = ContinuousType(name="real.1d")
    c = ConstantEvaluator(name="k", value_type=t, value=3.14)
    assert c.evaluate_constant() == 3.14


def test_constant_evaluator_vector() -> None:
    t = ContinuousType(name="real.3d", component_name="c", component_count=3)
    c = ConstantEvaluator(name="origin", value_type=t, value=(0.0, 0.0, 0.0))
    np.testing.assert_array_equal(c.evaluate_constant(), [0.0, 0.0, 0.0])


def test_evaluator_base_records_name_and_value_type() -> None:
    t = ContinuousType(name="real.1d")
    c = ConstantEvaluator(name="k", value_type=t, value=1.0)
    assert isinstance(c, Evaluator)
    assert c.name == "k"
    assert c.value_type is t


def test_reference_evaluator_stores_source_and_bindings() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    inner = ConstantEvaluator(name="k", value_type=t, value=1.0)
    src = ConstantEvaluator(name="src", value_type=t, value=7.0)
    ref = ReferenceEvaluator(
        name="r",
        value_type=t,
        source=inner,
        bindings=BindingTable([ArgumentBinding(argument=arg, source=src)]),
    )
    assert ref.source is inner
    assert len(ref.bindings) == 1


def test_aggregate_evaluator_records_components() -> None:
    vec = ContinuousType(name="real.3d", component_name="c", component_count=3)
    scalar = ContinuousType(name="real.1d")
    cs = [ConstantEvaluator(name=f"k{i}", value_type=scalar, value=float(i)) for i in range(3)]
    agg = AggregateEvaluator(name="v", value_type=vec, components=cs)
    assert len(agg.components) == 3


def test_piecewise_evaluator_maps_cases_to_sources() -> None:
    t = ContinuousType(name="real.1d")
    a = ConstantEvaluator(name="a", value_type=t, value=1.0)
    b = ConstantEvaluator(name="b", value_type=t, value=2.0)
    pw = PiecewiseEvaluator(name="pw", value_type=t, pieces={1: a, 2: b}, default=None)
    assert pw.pieces[1] is a
    assert pw.default is None


def test_external_evaluator_names_library_entry() -> None:
    t = ContinuousType(name="real.1d")
    ext = ExternalEvaluator(name="library.basis.linear_lagrange", value_type=t)
    assert ext.name.startswith("library.basis")
