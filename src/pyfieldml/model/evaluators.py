"""Evaluator hierarchy for the FieldML semantic model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from pyfieldml.data.base import DataSource
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)

if TYPE_CHECKING:
    from pyfieldml.model.bindings import BindingTable

ValueType = BooleanType | EnsembleType | ContinuousType | MeshType


@dataclass
class Evaluator:
    """Base class for all FieldML evaluators. Instantiated only via subclasses."""

    name: str
    value_type: ValueType


@dataclass
class ParameterEvaluator(Evaluator):
    """Evaluator backed by a concrete array of parameter values."""

    data: DataSource = field(default=None)  # type: ignore[assignment]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def as_ndarray(self) -> np.ndarray:
        return self.data.as_ndarray()


@dataclass
class ConstantEvaluator(Evaluator):
    """Evaluator that yields a fixed value (scalar or tuple of components)."""

    value: float | int | bool | Sequence[float] | Sequence[int] = 0.0

    def evaluate_constant(self) -> float | int | bool | np.ndarray:
        value = self.value
        if isinstance(value, list | tuple):
            return np.asarray(value)
        return cast("float | int | bool", value)


@dataclass
class ArgumentEvaluator(Evaluator):
    """Formal-parameter placeholder. Must be bound at call-sites."""


@dataclass
class ReferenceEvaluator(Evaluator):
    """Reference another evaluator, optionally with argument bindings."""

    source: Evaluator = field(default=None)  # type: ignore[assignment]
    bindings: BindingTable = field(default=None)  # type: ignore[assignment]


@dataclass
class AggregateEvaluator(Evaluator):
    """Assemble a value from per-component sub-evaluators."""

    components: list[Evaluator] = field(default_factory=list)


@dataclass
class PiecewiseEvaluator(Evaluator):
    """Dispatch to a per-ensemble-index sub-evaluator; optional default."""

    pieces: dict[int, Evaluator] = field(default_factory=dict)
    default: Evaluator | None = None


@dataclass
class ExternalEvaluator(Evaluator):
    """Reference to an object defined in an imported library (e.g. a basis function)."""
