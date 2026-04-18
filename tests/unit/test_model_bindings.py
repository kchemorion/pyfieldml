"""Tests for ArgumentEvaluator and ArgumentBinding."""

from __future__ import annotations

import pytest

from pyfieldml.errors import UnboundArgumentError
from pyfieldml.model.bindings import ArgumentBinding, BindingTable
from pyfieldml.model.evaluators import ArgumentEvaluator, ConstantEvaluator
from pyfieldml.model.types import ContinuousType


def test_argument_evaluator_is_unbound_by_default() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    assert arg.name == "x"


def test_binding_table_resolve_uses_provided_value() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    src = ConstantEvaluator(name="src", value_type=t, value=7.0)
    table = BindingTable([ArgumentBinding(argument=arg, source=src)])
    assert table.resolve(arg) is src


def test_binding_table_raises_on_unbound() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    table = BindingTable([])
    with pytest.raises(UnboundArgumentError, match="x"):
        table.resolve(arg)
