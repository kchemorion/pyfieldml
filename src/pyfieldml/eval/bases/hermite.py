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


@basis("library.basis.bicubic_hermite.quad", topology="quad", order=3)
class BicubicHermiteQuad:
    """Bicubic Hermite basis on the unit quad, 16 DOFs/element."""

    n_nodes = 16
    topology = "quad"
    order = 3

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return 16 shape functions at xi. Shape (..., 16)."""
        xi = np.asarray(xi, dtype=np.float64)
        h_u = _hermite_1d(xi[..., 0])
        h_v = _hermite_1d(xi[..., 1])
        out = np.empty((*xi.shape[:-1], 16), dtype=np.float64)
        for corner_idx, (a, b) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            for f_u, f_v in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                ui = 2 * a + f_u
                vi = 2 * b + f_v
                dof = 4 * corner_idx + (f_u + 2 * f_v)
                out[..., dof] = h_u[..., ui] * h_v[..., vi]
        return out

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return 16 x 2 derivatives at xi. Shape (..., 16, 2)."""
        xi = np.asarray(xi, dtype=np.float64)
        h_u = _hermite_1d(xi[..., 0])
        h_v = _hermite_1d(xi[..., 1])
        dh_u = _hermite_1d_derivative(xi[..., 0])
        dh_v = _hermite_1d_derivative(xi[..., 1])
        out = np.empty((*xi.shape[:-1], 16, 2), dtype=np.float64)
        for corner_idx, (a, b) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
            for f_u, f_v in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                ui = 2 * a + f_u
                vi = 2 * b + f_v
                dof = 4 * corner_idx + (f_u + 2 * f_v)
                out[..., dof, 0] = dh_u[..., ui] * h_v[..., vi]
                out[..., dof, 1] = h_u[..., ui] * dh_v[..., vi]
        return out
