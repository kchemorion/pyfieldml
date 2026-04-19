"""Graph-walker tests."""

from __future__ import annotations

from pyfieldml.eval.context import EvaluationContext
from pyfieldml.eval.graph import evaluate_evaluator
from pyfieldml.model.bindings import ArgumentBinding, BindingTable
from pyfieldml.model.evaluators import (
    ArgumentEvaluator,
    ConstantEvaluator,
    ReferenceEvaluator,
)
from pyfieldml.model.types import ContinuousType


def test_constant_evaluator_returns_scalar() -> None:
    t = ContinuousType(name="real.1d")
    c = ConstantEvaluator(name="k", value_type=t, value=3.14)
    ctx = EvaluationContext()
    assert evaluate_evaluator(c, ctx) == 3.14


def test_reference_evaluator_with_binding() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    inner = ReferenceEvaluator(name="ref", value_type=t, source=arg, bindings=BindingTable([]))
    concrete = ConstantEvaluator(name="src", value_type=t, value=9.0)
    ref = ReferenceEvaluator(
        name="outer",
        value_type=t,
        source=inner,
        bindings=BindingTable([ArgumentBinding(argument=arg, source=concrete)]),
    )
    ctx = EvaluationContext()
    assert evaluate_evaluator(ref, ctx) == 9.0
