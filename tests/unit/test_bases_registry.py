"""Tests for the basis-function registry."""

from __future__ import annotations

import numpy as np
import pytest

from pyfieldml.errors import BasisNotFoundError
from pyfieldml.eval.bases import get_basis, list_bases  # noqa: F401
from pyfieldml.eval.bases.registry import basis


def test_get_unknown_basis_raises() -> None:
    with pytest.raises(BasisNotFoundError, match="not_a_real_basis"):
        get_basis("not_a_real_basis")


def test_register_custom_basis_and_resolve() -> None:
    @basis("tests.identity", topology="line", order=1)
    class Identity:
        topology = "line"
        order = 1
        n_nodes = 2

        def shape_functions(self, xi: np.ndarray) -> np.ndarray:
            xi = np.atleast_1d(xi)
            return np.stack([1.0 - xi[..., 0], xi[..., 0]], axis=-1)

        def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
            xi = np.atleast_1d(xi)
            shape = (*xi.shape[:-1], 2, 1)
            out = np.empty(shape)
            out[..., 0, 0] = -1.0
            out[..., 1, 0] = 1.0
            return out

    b = get_basis("tests.identity")
    assert b.topology == "line"
    assert b.order == 1
    np.testing.assert_array_almost_equal(b.shape_functions(np.array([0.25])), [0.75, 0.25])
