"""Linear and quadratic Lagrange basis functions on the standard topologies.

Reference element conventions:

- line:  xi in [0, 1]
- tri:   unit triangle, vertices (0,0) (1,0) (0,1)
- quad:  unit square [0,1]^2, CCW
- tet:   unit tet, vertices (0,0,0) (1,0,0) (0,1,0) (0,0,1)
- hex:   unit cube [0,1]^3; CMISS node order = lexicographic in (xi1, xi2, xi3)
- wedge: tri x line, i.e. triangle at xi3=0 and triangle at xi3=1
"""

from __future__ import annotations

import numpy as np

from pyfieldml.eval.bases.registry import basis

# ------------------------------------------------------------------ LINE, linear


@basis("library.basis.linear_lagrange.line", topology="line", order=1)
class LinearLagrangeLine:
    """Linear Lagrange basis on the reference line xi in [0, 1]."""

    n_nodes = 2
    topology = "line"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 2)."""
        xi = np.asarray(xi, dtype=np.float64)
        x = xi[..., 0]
        return np.stack([1.0 - x, x], axis=-1)

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 2, 1)."""
        xi = np.asarray(xi, dtype=np.float64)
        out = np.empty((*xi.shape[:-1], 2, 1), dtype=np.float64)
        out[..., 0, 0] = -1.0
        out[..., 1, 0] = 1.0
        return out


# ------------------------------------------------------------------ TRIANGLE, linear


@basis("library.basis.linear_lagrange.triangle", topology="triangle", order=1)
class LinearLagrangeTriangle:
    """Linear Lagrange basis on the unit triangle (0,0), (1,0), (0,1)."""

    n_nodes = 3
    topology = "triangle"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 3)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v = xi[..., 0], xi[..., 1]
        return np.stack([1.0 - u - v, u, v], axis=-1)

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 3, 2)."""
        xi = np.asarray(xi, dtype=np.float64)
        out = np.zeros((*xi.shape[:-1], 3, 2), dtype=np.float64)
        out[..., 0, 0] = -1.0
        out[..., 0, 1] = -1.0
        out[..., 1, 0] = 1.0
        out[..., 2, 1] = 1.0
        return out


# ------------------------------------------------------------------ QUAD, linear (bilinear)


@basis("library.basis.linear_lagrange.quad", topology="quad", order=1)
class LinearLagrangeQuad:
    """Bilinear Lagrange basis on the unit square [0,1]^2."""

    n_nodes = 4
    topology = "quad"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 4)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v = xi[..., 0], xi[..., 1]
        return np.stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v], axis=-1)

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 4, 2)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v = xi[..., 0], xi[..., 1]
        out = np.empty((*xi.shape[:-1], 4, 2), dtype=np.float64)
        out[..., 0, 0] = -(1 - v)
        out[..., 0, 1] = -(1 - u)
        out[..., 1, 0] = 1 - v
        out[..., 1, 1] = -u
        out[..., 2, 0] = -v
        out[..., 2, 1] = 1 - u
        out[..., 3, 0] = v
        out[..., 3, 1] = u
        return out


# ------------------------------------------------------------------ TET, linear


@basis("library.basis.linear_lagrange.tet", topology="tet", order=1)
class LinearLagrangeTet:
    """Linear Lagrange basis on the unit tetrahedron."""

    n_nodes = 4
    topology = "tet"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 4)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        return np.stack([1.0 - u - v - w, u, v, w], axis=-1)

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 4, 3)."""
        xi = np.asarray(xi, dtype=np.float64)
        out = np.zeros((*xi.shape[:-1], 4, 3), dtype=np.float64)
        out[..., 0, :] = -1.0
        out[..., 1, 0] = 1.0
        out[..., 2, 1] = 1.0
        out[..., 3, 2] = 1.0
        return out


# ------------------------------------------------------------------ HEX, linear (trilinear)


@basis("library.basis.linear_lagrange.hex", topology="hex", order=1)
class LinearLagrangeHex:
    """Trilinear Lagrange basis on the unit cube [0,1]^3."""

    n_nodes = 8
    topology = "hex"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 8)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        return np.stack(
            [
                (1 - u) * (1 - v) * (1 - w),
                u * (1 - v) * (1 - w),
                (1 - u) * v * (1 - w),
                u * v * (1 - w),
                (1 - u) * (1 - v) * w,
                u * (1 - v) * w,
                (1 - u) * v * w,
                u * v * w,
            ],
            axis=-1,
        )

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 8, 3)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        out = np.empty((*xi.shape[:-1], 8, 3), dtype=np.float64)
        out[..., 0, 0] = -(1 - v) * (1 - w)
        out[..., 0, 1] = -(1 - u) * (1 - w)
        out[..., 0, 2] = -(1 - u) * (1 - v)
        out[..., 1, 0] = (1 - v) * (1 - w)
        out[..., 1, 1] = -u * (1 - w)
        out[..., 1, 2] = -u * (1 - v)
        out[..., 2, 0] = -v * (1 - w)
        out[..., 2, 1] = (1 - u) * (1 - w)
        out[..., 2, 2] = -(1 - u) * v
        out[..., 3, 0] = v * (1 - w)
        out[..., 3, 1] = u * (1 - w)
        out[..., 3, 2] = -u * v
        out[..., 4, 0] = -(1 - v) * w
        out[..., 4, 1] = -(1 - u) * w
        out[..., 4, 2] = (1 - u) * (1 - v)
        out[..., 5, 0] = (1 - v) * w
        out[..., 5, 1] = -u * w
        out[..., 5, 2] = u * (1 - v)
        out[..., 6, 0] = -v * w
        out[..., 6, 1] = (1 - u) * w
        out[..., 6, 2] = (1 - u) * v
        out[..., 7, 0] = v * w
        out[..., 7, 1] = u * w
        out[..., 7, 2] = u * v
        return out


# ------------------------------------------------------------------ WEDGE, linear (tri x line)


@basis("library.basis.linear_lagrange.wedge", topology="wedge", order=1)
class LinearLagrangeWedge:
    """Linear Lagrange basis on the wedge/prism (triangle x line)."""

    n_nodes = 6
    topology = "wedge"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 6)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        tri = np.stack([1.0 - u - v, u, v], axis=-1)
        line = np.stack([1.0 - w, w], axis=-1)
        # Outer product: (..., 2, 3) -> flatten to (..., 6) in wedge node order
        # [bot-tri-0, bot-tri-1, bot-tri-2, top-tri-0, top-tri-1, top-tri-2].
        outer = line[..., :, None] * tri[..., None, :]
        result: np.ndarray = outer.reshape(*xi.shape[:-1], 6)
        return result

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 6, 3)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        d = np.empty((*xi.shape[:-1], 6, 3), dtype=np.float64)
        # Order: (bot-tri-0, bot-tri-1, bot-tri-2, top-tri-0, top-tri-1, top-tri-2).
        # d/du, d/dv, d/dw
        bot = 1.0 - w
        top = w
        # Node 0: (1-u-v)(1-w)
        d[..., 0, 0] = -bot
        d[..., 0, 1] = -bot
        d[..., 0, 2] = -(1.0 - u - v)
        # Node 1: u(1-w)
        d[..., 1, 0] = bot
        d[..., 1, 1] = 0.0
        d[..., 1, 2] = -u
        # Node 2: v(1-w)
        d[..., 2, 0] = 0.0
        d[..., 2, 1] = bot
        d[..., 2, 2] = -v
        # Node 3: (1-u-v)w
        d[..., 3, 0] = -top
        d[..., 3, 1] = -top
        d[..., 3, 2] = 1.0 - u - v
        # Node 4: u*w
        d[..., 4, 0] = top
        d[..., 4, 1] = 0.0
        d[..., 4, 2] = u
        # Node 5: v*w
        d[..., 5, 0] = 0.0
        d[..., 5, 1] = top
        d[..., 5, 2] = v
        return d
