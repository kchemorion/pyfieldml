"""Tests for the Evaluator class hierarchy."""

from __future__ import annotations

import numpy as np

from pyfieldml.data.text import InlineTextBackend
from pyfieldml.model.evaluators import ConstantEvaluator, Evaluator, ParameterEvaluator
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
