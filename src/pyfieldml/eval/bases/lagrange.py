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

# ------------------------------------------------------------------ 1D quadratic helpers


def _lagrange_1d_quad(xi: np.ndarray) -> np.ndarray:
    """3 quadratic Lagrange polynomials at nodes {0, 0.5, 1}. Shape (..., 3)."""
    return np.stack(
        [
            (1.0 - xi) * (1.0 - 2.0 * xi),  # N0
            4.0 * xi * (1.0 - xi),  # N1
            xi * (2.0 * xi - 1.0),  # N2
        ],
        axis=-1,
    )


def _lagrange_1d_quad_d(xi: np.ndarray) -> np.ndarray:
    """d/dxi of 3 quadratic Lagrange. Shape (..., 3)."""
    return np.stack(
        [
            4.0 * xi - 3.0,
            4.0 - 8.0 * xi,
            4.0 * xi - 1.0,
        ],
        axis=-1,
    )


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


# ------------------------------------------------------------------ LINE, quadratic


@basis("library.basis.quadratic_lagrange.line", topology="line", order=2)
class QuadraticLagrangeLine:
    """Quadratic Lagrange basis on the reference line xi in [0, 1]."""

    n_nodes = 3
    topology = "line"
    order = 2

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 3)."""
        xi = np.asarray(xi, dtype=np.float64)
        return _lagrange_1d_quad(xi[..., 0])

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 3, 1)."""
        xi = np.asarray(xi, dtype=np.float64)
        d = _lagrange_1d_quad_d(xi[..., 0])
        return d[..., None]


# ------------------------------------------------------------------ TRIANGLE, quadratic


@basis("library.basis.quadratic_lagrange.triangle", topology="triangle", order=2)
class QuadraticLagrangeTriangle:
    """Quadratic Lagrange basis on the unit triangle (6 nodes)."""

    n_nodes = 6
    topology = "triangle"
    order = 2

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 6).

        Node order:
            0: vertex (0, 0)
            1: vertex (1, 0)
            2: vertex (0, 1)
            3: edge midpoint (0.5, 0)   -- between v0 and v1
            4: edge midpoint (0.5, 0.5) -- between v1 and v2
            5: edge midpoint (0,   0.5) -- between v2 and v0
        """
        xi = np.asarray(xi, dtype=np.float64)
        u, v = xi[..., 0], xi[..., 1]
        l0 = 1.0 - u - v
        l1 = u
        l2 = v
        return np.stack(
            [
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l0 * l2,
            ],
            axis=-1,
        )

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 6, 2)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v = xi[..., 0], xi[..., 1]
        l0 = 1.0 - u - v
        l1 = u
        l2 = v
        # dl0/du = -1, dl0/dv = -1; dl1/du = 1, dl1/dv = 0; dl2/du = 0, dl2/dv = 1.
        out = np.empty((*xi.shape[:-1], 6, 2), dtype=np.float64)
        # n0 = l0(2 l0 - 1) -> dn0 = (4 l0 - 1) * dl0
        f0 = 4.0 * l0 - 1.0
        out[..., 0, 0] = f0 * (-1.0)
        out[..., 0, 1] = f0 * (-1.0)
        # n1 = l1(2 l1 - 1) -> dn1 = (4 l1 - 1) * dl1
        f1 = 4.0 * l1 - 1.0
        out[..., 1, 0] = f1 * 1.0
        out[..., 1, 1] = 0.0
        # n2 = l2(2 l2 - 1) -> dn2 = (4 l2 - 1) * dl2
        f2 = 4.0 * l2 - 1.0
        out[..., 2, 0] = 0.0
        out[..., 2, 1] = f2 * 1.0
        # n3 = 4 l0 l1 -> dn3 = 4 (dl0 * l1 + l0 * dl1)
        out[..., 3, 0] = 4.0 * (-l1 + l0)
        out[..., 3, 1] = 4.0 * (-l1)
        # n4 = 4 l1 l2 -> dn4 = 4 (dl1 * l2 + l1 * dl2)
        out[..., 4, 0] = 4.0 * l2
        out[..., 4, 1] = 4.0 * l1
        # n5 = 4 l0 l2 -> dn5 = 4 (dl0 * l2 + l0 * dl2)
        out[..., 5, 0] = 4.0 * (-l2)
        out[..., 5, 1] = 4.0 * (-l2 + l0)
        return out


# ------------------------------------------------------------------ QUAD, quadratic (9 nodes)


@basis("library.basis.quadratic_lagrange.quad", topology="quad", order=2)
class QuadraticLagrangeQuad:
    """Biquadratic Lagrange basis on the unit square [0,1]^2 (9 nodes).

    Nodes arranged 3x3 in lexicographic order (xi1 inner, xi2 outer):
    k = i + 3*j where i, j in {0, 1, 2} correspond to positions {0, 0.5, 1}.
    """

    n_nodes = 9
    topology = "quad"
    order = 2

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 9)."""
        xi = np.asarray(xi, dtype=np.float64)
        n_u = _lagrange_1d_quad(xi[..., 0])  # (..., 3)
        n_v = _lagrange_1d_quad(xi[..., 1])  # (..., 3)
        # Outer product with v outer, u inner -> shape (..., 3, 3) then flatten.
        outer = n_v[..., :, None] * n_u[..., None, :]
        result: np.ndarray = outer.reshape(*xi.shape[:-1], 9)
        return result

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 9, 2)."""
        xi = np.asarray(xi, dtype=np.float64)
        n_u = _lagrange_1d_quad(xi[..., 0])
        n_v = _lagrange_1d_quad(xi[..., 1])
        dn_u = _lagrange_1d_quad_d(xi[..., 0])
        dn_v = _lagrange_1d_quad_d(xi[..., 1])
        # d/du = dn_u(u) * n_v(v); d/dv = n_u(u) * dn_v(v).
        dphi_du = (n_v[..., :, None] * dn_u[..., None, :]).reshape(*xi.shape[:-1], 9)
        dphi_dv = (dn_v[..., :, None] * n_u[..., None, :]).reshape(*xi.shape[:-1], 9)
        return np.stack([dphi_du, dphi_dv], axis=-1)


# ------------------------------------------------------------------ TET, quadratic (10 nodes)


@basis("library.basis.quadratic_lagrange.tet", topology="tet", order=2)
class QuadraticLagrangeTet:
    """Quadratic Lagrange basis on the unit tetrahedron (10 nodes).

    Node order:
        0-3: vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        4: mid(v0, v1) = (0.5, 0,   0  )
        5: mid(v1, v2) = (0.5, 0.5, 0  )
        6: mid(v0, v2) = (0,   0.5, 0  )
        7: mid(v0, v3) = (0,   0,   0.5)
        8: mid(v1, v3) = (0.5, 0,   0.5)
        9: mid(v2, v3) = (0,   0.5, 0.5)
    """

    n_nodes = 10
    topology = "tet"
    order = 2

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 10)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        l0 = 1.0 - u - v - w
        l1 = u
        l2 = v
        l3 = w
        return np.stack(
            [
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                l3 * (2.0 * l3 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l0 * l2,
                4.0 * l0 * l3,
                4.0 * l1 * l3,
                4.0 * l2 * l3,
            ],
            axis=-1,
        )

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 10, 3)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        l0 = 1.0 - u - v - w
        l1 = u
        l2 = v
        l3 = w
        # dl0/d{u,v,w} = (-1,-1,-1); dl1 = (1,0,0); dl2 = (0,1,0); dl3 = (0,0,1).
        out = np.empty((*xi.shape[:-1], 10, 3), dtype=np.float64)
        # Vertex nodes: n_i = l_i (2 l_i - 1), dn_i/dx = (4 l_i - 1) * dl_i/dx.
        f0 = 4.0 * l0 - 1.0
        out[..., 0, 0] = f0 * (-1.0)
        out[..., 0, 1] = f0 * (-1.0)
        out[..., 0, 2] = f0 * (-1.0)
        f1 = 4.0 * l1 - 1.0
        out[..., 1, 0] = f1 * 1.0
        out[..., 1, 1] = 0.0
        out[..., 1, 2] = 0.0
        f2 = 4.0 * l2 - 1.0
        out[..., 2, 0] = 0.0
        out[..., 2, 1] = f2 * 1.0
        out[..., 2, 2] = 0.0
        f3 = 4.0 * l3 - 1.0
        out[..., 3, 0] = 0.0
        out[..., 3, 1] = 0.0
        out[..., 3, 2] = f3 * 1.0
        # Edge nodes: n_ij = 4 l_i l_j, d/dx = 4 (dl_i/dx * l_j + l_i * dl_j/dx).
        # 4: 4 l0 l1
        out[..., 4, 0] = 4.0 * ((-1.0) * l1 + l0 * 1.0)
        out[..., 4, 1] = 4.0 * ((-1.0) * l1 + l0 * 0.0)
        out[..., 4, 2] = 4.0 * ((-1.0) * l1 + l0 * 0.0)
        # 5: 4 l1 l2
        out[..., 5, 0] = 4.0 * (1.0 * l2 + l1 * 0.0)
        out[..., 5, 1] = 4.0 * (0.0 * l2 + l1 * 1.0)
        out[..., 5, 2] = 4.0 * 0.0
        # 6: 4 l0 l2
        out[..., 6, 0] = 4.0 * ((-1.0) * l2 + l0 * 0.0)
        out[..., 6, 1] = 4.0 * ((-1.0) * l2 + l0 * 1.0)
        out[..., 6, 2] = 4.0 * ((-1.0) * l2 + l0 * 0.0)
        # 7: 4 l0 l3
        out[..., 7, 0] = 4.0 * ((-1.0) * l3 + l0 * 0.0)
        out[..., 7, 1] = 4.0 * ((-1.0) * l3 + l0 * 0.0)
        out[..., 7, 2] = 4.0 * ((-1.0) * l3 + l0 * 1.0)
        # 8: 4 l1 l3
        out[..., 8, 0] = 4.0 * (1.0 * l3 + l1 * 0.0)
        out[..., 8, 1] = 4.0 * 0.0
        out[..., 8, 2] = 4.0 * (0.0 * l3 + l1 * 1.0)
        # 9: 4 l2 l3
        out[..., 9, 0] = 4.0 * 0.0
        out[..., 9, 1] = 4.0 * (1.0 * l3 + l2 * 0.0)
        out[..., 9, 2] = 4.0 * (0.0 * l3 + l2 * 1.0)
        return out


# ------------------------------------------------------------------ HEX, quadratic (27 nodes)


@basis("library.basis.quadratic_lagrange.hex", topology="hex", order=2)
class QuadraticLagrangeHex:
    """Triquadratic Lagrange basis on the unit cube [0,1]^3 (27 nodes).

    Nodes arranged 3x3x3 in lexicographic order in (xi1, xi2, xi3):
    k = i + 3*j + 9*l where i, j, l in {0, 1, 2}.
    """

    n_nodes = 27
    topology = "hex"
    order = 2

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 27)."""
        xi = np.asarray(xi, dtype=np.float64)
        n_u = _lagrange_1d_quad(xi[..., 0])
        n_v = _lagrange_1d_quad(xi[..., 1])
        n_w = _lagrange_1d_quad(xi[..., 2])
        # Outer product: w outer, v middle, u inner -> (..., 3, 3, 3) -> (..., 27).
        outer = n_w[..., :, None, None] * n_v[..., None, :, None] * n_u[..., None, None, :]
        result: np.ndarray = outer.reshape(*xi.shape[:-1], 27)
        return result

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 27, 3)."""
        xi = np.asarray(xi, dtype=np.float64)
        n_u = _lagrange_1d_quad(xi[..., 0])
        n_v = _lagrange_1d_quad(xi[..., 1])
        n_w = _lagrange_1d_quad(xi[..., 2])
        dn_u = _lagrange_1d_quad_d(xi[..., 0])
        dn_v = _lagrange_1d_quad_d(xi[..., 1])
        dn_w = _lagrange_1d_quad_d(xi[..., 2])
        shape = (*xi.shape[:-1], 27)
        dphi_du = (
            n_w[..., :, None, None] * n_v[..., None, :, None] * dn_u[..., None, None, :]
        ).reshape(shape)
        dphi_dv = (
            n_w[..., :, None, None] * dn_v[..., None, :, None] * n_u[..., None, None, :]
        ).reshape(shape)
        dphi_dw = (
            dn_w[..., :, None, None] * n_v[..., None, :, None] * n_u[..., None, None, :]
        ).reshape(shape)
        return np.stack([dphi_du, dphi_dv, dphi_dw], axis=-1)


# ------------------------------------------------------------------ WEDGE, quadratic (18 nodes)


@basis("library.basis.quadratic_lagrange.wedge", topology="wedge", order=2)
class QuadraticLagrangeWedge:
    """Quadratic Lagrange basis on the wedge/prism (quadratic tri x quadratic line).

    Flattening: line outer, tri inner, so node k = l*6 + t where l in {0,1,2}
    indexes the line nodes at w in {0, 0.5, 1} and t in {0..5} indexes the
    quadratic triangle nodes (3 vertices + 3 edge midpoints).
    """

    n_nodes = 18
    topology = "wedge"
    order = 2

    def _tri_values(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        l0 = 1.0 - u - v
        l1 = u
        l2 = v
        return np.stack(
            [
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l0 * l2,
            ],
            axis=-1,
        )

    def _tri_derivs(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        l0 = 1.0 - u - v
        l1 = u
        l2 = v
        out = np.empty((*u.shape, 6, 2), dtype=np.float64)
        f0 = 4.0 * l0 - 1.0
        out[..., 0, 0] = f0 * (-1.0)
        out[..., 0, 1] = f0 * (-1.0)
        f1 = 4.0 * l1 - 1.0
        out[..., 1, 0] = f1 * 1.0
        out[..., 1, 1] = 0.0
        f2 = 4.0 * l2 - 1.0
        out[..., 2, 0] = 0.0
        out[..., 2, 1] = f2 * 1.0
        out[..., 3, 0] = 4.0 * (-l1 + l0)
        out[..., 3, 1] = 4.0 * (-l1)
        out[..., 4, 0] = 4.0 * l2
        out[..., 4, 1] = 4.0 * l1
        out[..., 5, 0] = 4.0 * (-l2)
        out[..., 5, 1] = 4.0 * (-l2 + l0)
        return out

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        """Return shape functions with shape (..., 18)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        tri = self._tri_values(u, v)  # (..., 6)
        line = _lagrange_1d_quad(w)  # (..., 3)
        # Line outer, tri inner: k = l*6 + t
        outer = line[..., :, None] * tri[..., None, :]
        result: np.ndarray = outer.reshape(*xi.shape[:-1], 18)
        return result

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        """Return shape-function derivatives with shape (..., 18, 3)."""
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        tri = self._tri_values(u, v)  # (..., 6)
        dtri = self._tri_derivs(u, v)  # (..., 6, 2)
        line = _lagrange_1d_quad(w)  # (..., 3)
        dline = _lagrange_1d_quad_d(w)  # (..., 3)
        shape = (*xi.shape[:-1], 18)
        # d/du = line * dtri/du ; d/dv = line * dtri/dv ; d/dw = dline * tri.
        dphi_du = (line[..., :, None] * dtri[..., None, :, 0]).reshape(shape)
        dphi_dv = (line[..., :, None] * dtri[..., None, :, 1]).reshape(shape)
        dphi_dw = (dline[..., :, None] * tri[..., None, :]).reshape(shape)
        return np.stack([dphi_du, dphi_dv, dphi_dw], axis=-1)
