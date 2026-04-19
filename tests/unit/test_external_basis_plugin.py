"""Verify external user-registered basis functions work end-to-end."""

from __future__ import annotations

import numpy as np

from pyfieldml.eval.bases import get_basis
from pyfieldml.eval.bases.registry import basis


def test_external_basis_is_honored_by_evaluation() -> None:
    @basis("custom.funky_linear_line", topology="line", order=1)
    class _Funky:
        topology = "line"
        order = 1
        n_nodes = 2

        def shape_functions(self, xi: np.ndarray) -> np.ndarray:
            xi = np.asarray(xi)
            x = xi[..., 0]
            return np.stack([1.0 - x, x], axis=-1)

        def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
            xi = np.asarray(xi)
            out = np.empty((*xi.shape[:-1], 2, 1))
            out[..., 0, 0] = -1.0
            out[..., 1, 0] = 1.0
            return out

    b = get_basis("custom.funky_linear_line")
    np.testing.assert_array_almost_equal(b.shape_functions(np.array([0.25])), [0.75, 0.25])
