"""Walk a FieldML evaluator graph and produce numeric values."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyfieldml.errors import EvaluationError, UnboundArgumentError
from pyfieldml.eval.context import EvaluationContext
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


def evaluate_evaluator(ev: Evaluator, ctx: EvaluationContext) -> Any:
    """Dispatch on evaluator subtype. Returns the numeric value of ``ev`` under ``ctx``."""
    if isinstance(ev, ConstantEvaluator):
        return ev.evaluate_constant()
    if isinstance(ev, ParameterEvaluator):
        return ev.as_ndarray()
    if isinstance(ev, ArgumentEvaluator):
        for table in reversed(ctx.bindings):
            try:
                bound = table.resolve(ev)
            except UnboundArgumentError:
                continue
            return evaluate_evaluator(bound, ctx)
        raise UnboundArgumentError(f"Argument {ev.name!r} unbound")
    if isinstance(ev, ReferenceEvaluator):
        pushed = False
        if ev.bindings is not None:
            ctx.bindings.append(ev.bindings)
            pushed = True
        try:
            return evaluate_evaluator(ev.source, ctx)
        finally:
            if pushed:
                ctx.bindings.pop()
    if isinstance(ev, AggregateEvaluator):
        parts = [np.asarray(evaluate_evaluator(c, ctx)) for c in ev.components]
        return np.stack(parts, axis=-1)
    if isinstance(ev, PiecewiseEvaluator):
        if ctx.element is None:
            raise EvaluationError(f"PiecewiseEvaluator {ev.name!r} requires an element in context")
        source = ev.pieces.get(ctx.element, ev.default)
        if source is None:
            raise EvaluationError(
                f"PiecewiseEvaluator {ev.name!r}: no case for element {ctx.element}"
            )
        return evaluate_evaluator(source, ctx)
    if isinstance(ev, ExternalEvaluator):
        raise EvaluationError(
            f"ExternalEvaluator {ev.name!r}: the eval engine dispatches externals "
            f"via the basis registry at call-sites, not standalone"
        )
    raise EvaluationError(f"Unknown evaluator type {type(ev).__name__}")
