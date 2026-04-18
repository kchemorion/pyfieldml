"""Evaluator hierarchy for the FieldML semantic model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from pyfieldml.data.base import DataSource
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)

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
