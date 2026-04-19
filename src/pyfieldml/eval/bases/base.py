"""Basis-function protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Basis(Protocol):
    """Protocol for any basis function usable by the evaluation engine.

    Implementations must be stateless. Shape arrays use the convention:

        xi:       shape (..., D)   where D = topology-dependent reference dimension
        φ(xi):    shape (..., N)   where N = number of nodes / DOFs
        ∂φ/∂xi:   shape (..., N, D)
    """

    topology: str
    order: int
    n_nodes: int

    def shape_functions(self, xi: np.ndarray) -> np.ndarray: ...
    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray: ...
