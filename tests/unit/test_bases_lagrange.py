"""Numerical tests for Lagrange basis functions."""

from __future__ import annotations

import numpy as np
import pytest

from pyfieldml.eval.bases import get_basis


@pytest.mark.parametrize(
    ("name", "n"),
    [
        ("library.basis.linear_lagrange.line", 2),
        ("library.basis.linear_lagrange.triangle", 3),
        ("library.basis.linear_lagrange.quad", 4),
        ("library.basis.linear_lagrange.tet", 4),
        ("library.basis.linear_lagrange.hex", 8),
        ("library.basis.linear_lagrange.wedge", 6),
        ("library.basis.quadratic_lagrange.line", 3),
        ("library.basis.quadratic_lagrange.triangle", 6),
        ("library.basis.quadratic_lagrange.quad", 9),
        ("library.basis.quadratic_lagrange.tet", 10),
        ("library.basis.quadratic_lagrange.hex", 27),
        ("library.basis.quadratic_lagrange.wedge", 18),
    ],
)
def test_linear_partition_of_unity_at_random_xi(name: str, n: int) -> None:
    """Shape functions must sum to 1 at every interior xi."""
    b = get_basis(name)
    rng = np.random.default_rng(0)
    d = _ref_dim(name)
    xi = rng.random((200, d))
    # Restrict to the reference simplex for triangles/tets/wedges.
    xi = _clip_to_reference(name, xi)
    phi = b.shape_functions(xi)
    np.testing.assert_allclose(phi.sum(axis=-1), 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "n"),
    [
        ("library.basis.linear_lagrange.line", 2),
        ("library.basis.linear_lagrange.triangle", 3),
        ("library.basis.linear_lagrange.quad", 4),
        ("library.basis.linear_lagrange.tet", 4),
        ("library.basis.linear_lagrange.hex", 8),
        ("library.basis.linear_lagrange.wedge", 6),
        ("library.basis.quadratic_lagrange.line", 3),
        ("library.basis.quadratic_lagrange.triangle", 6),
        ("library.basis.quadratic_lagrange.quad", 9),
        ("library.basis.quadratic_lagrange.tet", 10),
        ("library.basis.quadratic_lagrange.hex", 27),
        ("library.basis.quadratic_lagrange.wedge", 18),
    ],
)
def test_kronecker_delta_at_nodes(name: str, n: int) -> None:
    """phi_i(node_j) = delta_ij."""
    b = get_basis(name)
    nodes = _reference_nodes(name)
    phi = b.shape_functions(nodes)
    np.testing.assert_allclose(phi, np.eye(n), rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "name",
    [
        "library.basis.linear_lagrange.line",
        "library.basis.linear_lagrange.triangle",
        "library.basis.linear_lagrange.quad",
        "library.basis.linear_lagrange.tet",
        "library.basis.linear_lagrange.hex",
        "library.basis.linear_lagrange.wedge",
        "library.basis.quadratic_lagrange.line",
        "library.basis.quadratic_lagrange.triangle",
        "library.basis.quadratic_lagrange.quad",
        "library.basis.quadratic_lagrange.tet",
        "library.basis.quadratic_lagrange.hex",
        "library.basis.quadratic_lagrange.wedge",
    ],
)
def test_lagrange_shape_derivatives_match_finite_difference(name: str) -> None:
    """Central FD of shape_functions must match analytic shape_derivatives."""
    b = get_basis(name)
    rng = np.random.default_rng(0)
    d = _ref_dim(name)
    n_samples = 10

    # Sample 10 interior xi points inside the reference element
    xi = rng.random((n_samples, d)) * 0.6 + 0.2  # stay away from boundaries
    # For simplex topologies, clip to u+v+w <= 0.8 (safely interior)
    if name.endswith(("triangle", "tet", "wedge")):
        # Simple rejection to get 10 interior samples
        samples: list[np.ndarray] = []
        while len(samples) < n_samples:
            x = rng.random(d) * 0.6 + 0.2
            if x[:2].sum() <= 0.8:  # works for triangle/tet/wedge
                samples.append(x)
        xi = np.array(samples)

    analytic = b.shape_derivatives(xi)  # (N, n_nodes, D)

    # Central FD
    h = 1e-6
    fd = np.zeros_like(analytic)
    for dim in range(d):
        offset = np.zeros(d)
        offset[dim] = h
        f_plus = b.shape_functions(xi + offset)
        f_minus = b.shape_functions(xi - offset)
        fd[..., dim] = (f_plus - f_minus) / (2 * h)

    # Tolerance: 1e-5 accounts for float round-off in h and polynomial orders.
    np.testing.assert_allclose(analytic, fd, atol=1e-5, rtol=1e-5)


def _ref_dim(name: str) -> int:
    if name.endswith("line"):
        return 1
    if name.endswith("triangle") or name.endswith("quad"):
        return 2
    return 3


def _reference_nodes(name: str) -> np.ndarray:
    table = {
        "library.basis.linear_lagrange.line": [[0.0], [1.0]],
        "library.basis.linear_lagrange.triangle": [[0, 0], [1, 0], [0, 1]],
        "library.basis.linear_lagrange.quad": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "library.basis.linear_lagrange.tet": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "library.basis.linear_lagrange.hex": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        "library.basis.linear_lagrange.wedge": [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        "library.basis.quadratic_lagrange.line": [[0.0], [0.5], [1.0]],
        "library.basis.quadratic_lagrange.triangle": [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ],
        "library.basis.quadratic_lagrange.quad": [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
        "library.basis.quadratic_lagrange.tet": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
        "library.basis.quadratic_lagrange.hex": [
            [i * 0.5, j * 0.5, k * 0.5] for k in range(3) for j in range(3) for i in range(3)
        ],
        "library.basis.quadratic_lagrange.wedge": [
            # Line-outer, tri-inner: k = l*6 + t.
            # l=0, w=0
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            # l=1, w=0.5
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5],
            # l=2, w=1
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.0, 1.0],
            [0.5, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ],
    }
    return np.array(table[name], dtype=np.float64)


def _clip_to_reference(name: str, xi: np.ndarray) -> np.ndarray:
    if name.endswith("triangle"):
        # Reject points outside the unit triangle — resample by rejection.
        mask = xi.sum(axis=-1) <= 1.0
        return xi[mask]
    if name.endswith("tet") or name.endswith("wedge"):
        mask = xi.sum(axis=-1) <= 1.0
        return xi[mask]
    return xi
