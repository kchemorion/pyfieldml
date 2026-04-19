"""End-to-end Hermite line evaluation + C1 continuity with scaling."""

from __future__ import annotations

import numpy as np
import pytest

import pyfieldml as fml
from pyfieldml.builders.mesh import add_hermite_mesh
from pyfieldml.model.region import Region


def _build_hermite_line_doc(
    node_positions: np.ndarray,
    node_derivatives: np.ndarray,
    connectivity: np.ndarray,
    scales: np.ndarray | None = None,
) -> fml.Document:
    r = Region(name="hermite_line")
    add_hermite_mesh(
        r,
        name="line_mesh",
        nodes=node_positions,
        elements=connectivity,
        derivatives=node_derivatives,
        scales=scales,
        topology="line",
        coord_name="coordinates",
    )
    return fml.Document.from_region(r)


def test_single_element_hermite_line_reproduces_cubic() -> None:
    """A single Hermite element should reconstruct a cubic exactly when DOFs are set from it."""
    # Physical positions + derivatives for cubic f(s) = 2s**3 + 3s**2 - s + 5 on s in [0, 1].
    # Node 0: s=0 -> f=5, f'=-1
    # Node 1: s=1 -> f=9, f'=11
    nodes = np.array([[5.0], [9.0]])
    derivs = np.array([[-1.0], [11.0]])
    conn = np.array([[1, 2]], dtype=np.int64)
    doc = _build_hermite_line_doc(nodes, derivs, conn)
    coords = doc.field("coordinates")

    xi_test = np.linspace(0.0, 1.0, 21).reshape(-1, 1)
    got = np.array([coords.evaluate(element=1, xi=x)[0] for x in xi_test])
    expected = 2 * xi_test[:, 0] ** 3 + 3 * xi_test[:, 0] ** 2 - xi_test[:, 0] + 5
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_two_element_hermite_line_value_continuous_at_shared_node() -> None:
    """A 2-element line shares node 2; both elements should match at that node."""
    # Arbitrary DOFs.
    nodes = np.array([[0.0], [5.0], [11.0]])
    derivs = np.array([[1.0], [2.0], [1.5]])
    conn = np.array([[1, 2], [2, 3]], dtype=np.int64)
    doc = _build_hermite_line_doc(nodes, derivs, conn)
    coords = doc.field("coordinates")

    from_left = coords.evaluate(element=1, xi=1.0)
    from_right = coords.evaluate(element=2, xi=0.0)
    np.testing.assert_allclose(from_left, from_right, atol=1e-12)
    # Both must equal the node-2 value DOF (5.0).
    np.testing.assert_allclose(from_left, [5.0], atol=1e-12)


def test_hermite_line_scale_factors_affect_derivative_contribution() -> None:
    """With scale_factors = 2 at a node, the derivative contribution doubles."""
    # Single-element line with known DOFs. Compare evaluation with scale=1 vs
    # scale=2 at the right node's derivative.
    nodes = np.array([[0.0], [0.0]])  # both nodes at x=0 physically
    derivs = np.array([[1.0], [1.0]])  # both derivatives = 1
    conn = np.array([[1, 2]], dtype=np.int64)

    # Unscaled
    doc1 = _build_hermite_line_doc(nodes, derivs, conn)
    c1 = doc1.field("coordinates")
    v_unscaled = c1.evaluate(element=1, xi=0.75)[0]

    # Scale derivative@node-b by 2: element 1 has DOF slot index (1,3) for d@a, d@b; scale
    # format is (n_elems, 2) with [scale@a, scale@b].
    scales = np.array([[1.0, 2.0]])
    doc2 = _build_hermite_line_doc(nodes, derivs, conn, scales=scales)
    c2 = doc2.field("coordinates")
    v_scaled = c2.evaluate(element=1, xi=0.75)[0]

    # Reconstruct analytically: H11(0.75) = 0.75**3 - 0.75**2 = -0.140625
    # Unscaled contribution from d@b (DOF 3): 1.0 * H11(0.75) = -0.140625
    # Scaled: 2.0 * 1.0 * H11(0.75) = -0.28125
    # Difference: -0.140625. So v_scaled - v_unscaled should be -0.140625.
    diff_expected = -0.140625
    assert abs((v_scaled - v_unscaled) - diff_expected) < 1e-12, (
        f"expected diff {diff_expected}, got {v_scaled - v_unscaled}"
    )


def test_hermite_line_sample_raises_not_implemented() -> None:
    """Field.sample must refuse Hermite meshes cleanly in v1.0.x (no shape-mismatch crash)."""
    nodes = np.array([[0.0], [5.0], [11.0]])
    derivs = np.array([[1.0], [2.0], [1.5]])
    conn = np.array([[1, 2], [2, 3]], dtype=np.int64)
    doc = _build_hermite_line_doc(nodes, derivs, conn)
    coords = doc.field("coordinates")

    query = np.array([[2.5], [7.5]])
    with pytest.raises(NotImplementedError, match="Hermite"):
        coords.sample(query)
