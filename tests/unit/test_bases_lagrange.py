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
    ],
)
def test_kronecker_delta_at_nodes(name: str, n: int) -> None:
    """phi_i(node_j) = delta_ij."""
    b = get_basis(name)
    nodes = _reference_nodes(name)
    phi = b.shape_functions(nodes)
    np.testing.assert_allclose(phi, np.eye(n), rtol=0, atol=1e-12)


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
