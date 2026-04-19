"""Cubic Hermite basis tests."""

from __future__ import annotations

import numpy as np

from pyfieldml.eval.bases import get_basis


def test_hermite_line_interpolates_values_at_endpoints() -> None:
    """At xi=0 the value@0 DOF contributes 1; at xi=1 the value@1 DOF contributes 1."""
    b = get_basis("library.basis.cubic_hermite.line")
    phi_left = b.shape_functions(np.array([0.0]))
    np.testing.assert_allclose(phi_left, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    phi_right = b.shape_functions(np.array([1.0]))
    np.testing.assert_allclose(phi_right, [0.0, 0.0, 1.0, 0.0], atol=1e-12)


def test_hermite_line_derivatives_at_endpoints() -> None:
    """At xi=0 the derivative@0 DOF gives 1; at xi=1 the derivative@1 DOF gives 1."""
    b = get_basis("library.basis.cubic_hermite.line")
    d = b.shape_derivatives(np.array([0.0]))
    np.testing.assert_allclose(d[..., :, 0], [0.0, 1.0, 0.0, 0.0], atol=1e-12)
    d = b.shape_derivatives(np.array([1.0]))
    np.testing.assert_allclose(d[..., :, 0], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_hermite_line_reproduces_cubic() -> None:
    """Cubic Hermite must interpolate any cubic polynomial exactly."""
    b = get_basis("library.basis.cubic_hermite.line")
    # f(xi) = 2*xi^3 + 3*xi^2 - xi + 5
    # DOFs: f(0)=5, f'(0)=-1, f(1)=9, f'(1)=11
    dofs = np.array([5.0, -1.0, 9.0, 11.0])
    xi_test = np.linspace(0, 1, 21).reshape(-1, 1)
    phi = b.shape_functions(xi_test)
    reconstructed = phi @ dofs
    expected = 2 * xi_test[:, 0] ** 3 + 3 * xi_test[:, 0] ** 2 - xi_test[:, 0] + 5
    np.testing.assert_allclose(reconstructed, expected, atol=1e-12)


def test_hermite_quad_value_at_corners() -> None:
    """Value@corner DOFs (index 0, 4, 8, 12) must be delta on corners (0,0),(1,0),(0,1),(1,1)."""
    b = get_basis("library.basis.bicubic_hermite.quad")
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    phi = b.shape_functions(corners)
    value_dofs = [0, 4, 8, 12]
    for i, _corner in enumerate(corners):
        for j, dof in enumerate(value_dofs):
            expected = 1.0 if i == j else 0.0
            assert (
                abs(phi[i, dof] - expected) < 1e-12
            ), f"phi[corner={i}, value-DOF-of-corner={j}] = {phi[i, dof]}, expected {expected}"


def test_hermite_quad_derivative_dofs_zero_at_corners() -> None:
    """All derivative DOFs give 0 at all corners (only value DOFs matter for position)."""
    b = get_basis("library.basis.bicubic_hermite.quad")
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    phi = b.shape_functions(corners)
    derivative_dofs = [k for k in range(16) if k not in (0, 4, 8, 12)]
    for dof in derivative_dofs:
        np.testing.assert_allclose(phi[:, dof], 0.0, atol=1e-12)
