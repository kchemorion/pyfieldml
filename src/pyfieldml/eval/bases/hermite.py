"""Cubic Hermite basis functions. Registered on import."""

from __future__ import annotations

import numpy as np

from pyfieldml.eval.bases.registry import basis


def _hermite_1d(xi: np.ndarray) -> np.ndarray:
    """Cubic Hermite shape functions on xi in [0, 1].

    Returns shape (..., 4) in DOF order: H00, H01, H10, H11 corresponding to
    (value@0, derivative@0, value@1, derivative@1).
    """
    xi = np.asarray(xi, dtype=np.float64)
    x2 = xi * xi
    x3 = x2 * xi
    return np.stack(
        [
            2.0 * x3 - 3.0 * x2 + 1.0,  # H00
            x3 - 2.0 * x2 + xi,  # H01
            -2.0 * x3 + 3.0 * x2,  # H10
            x3 - x2,  # H11
        ],
        axis=-1,
    )


def _hermite_1d_derivative(xi: np.ndarray) -> np.ndarray:
    """d/dxi of cubic Hermite. Same DOF order. Returns shape (..., 4)."""
    xi = np.asarray(xi, dtype=np.float64)
    x2 = xi * xi
    return np.stack(
        [
            6.0 * x2 - 6.0 * xi,  # dH00
            3.0 * x2 - 4.0 * xi + 1.0,  # dH01
            -6.0 * x2 + 6.0 * xi,  # dH10
            3.0 * x2 - 2.0 * xi,  # dH11
        ],
        axis=-1,
    )


@basis("library.basis.cubic_hermite.line", topology="line", order=3)
class CubicHermiteLine:
    """Cubic Hermite basis on a line, 4 DOFs/element."""

    n_nodes = 4  # DOFs, not geometric nodes
    topology = "line"
    order = 3

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """φ at xi. Returns shape (..., 4)."""
        xi = np.asarray(xi, dtype=np.float64)
        return _hermite_1d(xi[..., 0])

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """dφ/dxi at xi. Returns shape (..., 4, 1)."""
        xi = np.asarray(xi, dtype=np.float64)
        d = _hermite_1d_derivative(xi[..., 0])  # (..., 4)
        return d[..., :, None]  # (..., 4, 1)
