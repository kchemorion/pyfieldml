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


# --------------------------------------------------------------- Quad / hex Hermite.
# TODO (v1.3+): C1 continuity tests for multi-element bicubic/tricubic Hermite
# under CMISS scaling — tricky because scale-factor convention is per-arc-length
# and assembly differs across CMISS/FieldML documents. Current tests cover
# single-element reproduction + Kronecker behaviour at corners.


def _build_hermite_quad_doc(
    node_positions: np.ndarray,
    node_derivatives: np.ndarray,
    connectivity: np.ndarray,
    scales: np.ndarray | None = None,
) -> fml.Document:
    r = Region(name="hermite_quad")
    add_hermite_mesh(
        r,
        name="quad_mesh",
        nodes=node_positions,
        elements=connectivity,
        derivatives=node_derivatives,
        scales=scales,
        topology="quad",
        coord_name="coordinates",
    )
    return fml.Document.from_region(r)


def _build_hermite_hex_doc(
    node_positions: np.ndarray,
    node_derivatives: np.ndarray,
    connectivity: np.ndarray,
    scales: np.ndarray | None = None,
) -> fml.Document:
    r = Region(name="hermite_hex")
    add_hermite_mesh(
        r,
        name="hex_mesh",
        nodes=node_positions,
        elements=connectivity,
        derivatives=node_derivatives,
        scales=scales,
        topology="hex",
        coord_name="coordinates",
    )
    return fml.Document.from_region(r)


def _bilinear(u: np.ndarray, v: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    # coeffs: (4, D) -> a, b, c, d such that f = a + b*u + c*v + d*u*v.
    a, b, c, d = coeffs
    return a + b * u[..., None] + c * v[..., None] + d * (u * v)[..., None]


def _trilinear(u: np.ndarray, v: np.ndarray, w: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    # coeffs: (8, D) -> a0..a7 for 1, u, v, w, uv, uw, vw, uvw.
    a0, a1, a2, a3, a4, a5, a6, a7 = coeffs
    return (
        a0
        + a1 * u[..., None]
        + a2 * v[..., None]
        + a3 * w[..., None]
        + a4 * (u * v)[..., None]
        + a5 * (u * w)[..., None]
        + a6 * (v * w)[..., None]
        + a7 * (u * v * w)[..., None]
    )


def test_hermite_quad_reproduces_bilinear() -> None:
    """Bicubic Hermite on a single [0,1]^2 element must reproduce any bilinear.

    f(u, v) = a + b*u + c*v + d*u*v.
        df/du = b + d*v, df/dv = c + d*u, d2f/(du dv) = d.
    """
    rng = np.random.default_rng(42)
    # D = 2 physical dims.
    coeffs = rng.standard_normal((4, 2))  # a, b, c, d

    # Four corner nodes of [0,1]^2 in basis order: (0,0), (1,0), (0,1), (1,1).
    corner_uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    u, v = corner_uv[:, 0], corner_uv[:, 1]
    nodes = _bilinear(u, v, coeffs)  # (4, D)

    # Derivatives at each node: (4, 3, D) = (N_nodes, slots, D)
    # slot 0: df/du = b + d*v;  slot 1: df/dv = c + d*u;  slot 2: d2f/(du dv) = d.
    b, c, d = coeffs[1], coeffs[2], coeffs[3]
    derivs = np.empty((4, 3, 2), dtype=np.float64)
    for i in range(4):
        ui, vi = u[i], v[i]
        derivs[i, 0] = b + d * vi  # d/du
        derivs[i, 1] = c + d * ui  # d/dv
        derivs[i, 2] = d  # d^2/(du dv)

    conn = np.array([[1, 2, 3, 4]], dtype=np.int64)
    doc = _build_hermite_quad_doc(nodes, derivs, conn)
    coords = doc.field("coordinates")

    # Ten random interior xi points.
    xi = rng.random((10, 2))
    expected = _bilinear(xi[:, 0], xi[:, 1], coeffs)
    got = np.array([coords.evaluate(element=1, xi=pt) for pt in xi])
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_hermite_hex_reproduces_trilinear() -> None:
    """Tricubic Hermite on a single [0,1]^3 element must reproduce any trilinear.

    f = a0 + a1*u + a2*v + a3*w + a4*uv + a5*uw + a6*vw + a7*uvw.
    """
    rng = np.random.default_rng(7)
    coeffs = rng.standard_normal((8, 3))  # D = 3

    # Eight corners in basis order: c outermost, b, a innermost:
    # (0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1).
    corners_uvw = np.array(
        [[a, bc, cc] for cc in (0, 1) for bc in (0, 1) for a in (0, 1)], dtype=np.float64
    )
    u = corners_uvw[:, 0]
    v = corners_uvw[:, 1]
    w = corners_uvw[:, 2]
    nodes = _trilinear(u, v, w, coeffs)  # (8, D)

    a1, a2, a3, a4, a5, a6, a7 = (
        coeffs[1],
        coeffs[2],
        coeffs[3],
        coeffs[4],
        coeffs[5],
        coeffs[6],
        coeffs[7],
    )
    # Derivatives at each node: (8, 7, D). Slot order:
    # 0: df/du = a1 + a4 v + a5 w + a7 vw
    # 1: df/dv = a2 + a4 u + a6 w + a7 uw
    # 2: df/dw = a3 + a5 u + a6 v + a7 uv
    # 3: d2f/(du dv) = a4 + a7 w
    # 4: d2f/(du dw) = a5 + a7 v
    # 5: d2f/(dv dw) = a6 + a7 u
    # 6: d3f/(du dv dw) = a7
    derivs = np.empty((8, 7, 3), dtype=np.float64)
    for i in range(8):
        ui, vi, wi = u[i], v[i], w[i]
        derivs[i, 0] = a1 + a4 * vi + a5 * wi + a7 * vi * wi
        derivs[i, 1] = a2 + a4 * ui + a6 * wi + a7 * ui * wi
        derivs[i, 2] = a3 + a5 * ui + a6 * vi + a7 * ui * vi
        derivs[i, 3] = a4 + a7 * wi
        derivs[i, 4] = a5 + a7 * vi
        derivs[i, 5] = a6 + a7 * ui
        derivs[i, 6] = a7

    conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    doc = _build_hermite_hex_doc(nodes, derivs, conn)
    coords = doc.field("coordinates")

    xi = rng.random((10, 3))
    expected = _trilinear(xi[:, 0], xi[:, 1], xi[:, 2], coeffs)
    got = np.array([coords.evaluate(element=1, xi=pt) for pt in xi])
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_hermite_quad_node_values_preserved() -> None:
    """Evaluating at the 4 corner xi's must return the value DOFs exactly."""
    rng = np.random.default_rng(3)
    nodes = rng.standard_normal((4, 2))
    # Arbitrary but non-zero derivative DOFs — they should not affect corner values.
    derivs = rng.standard_normal((4, 3, 2))
    conn = np.array([[1, 2, 3, 4]], dtype=np.int64)
    doc = _build_hermite_quad_doc(nodes, derivs, conn)
    coords = doc.field("coordinates")

    corner_uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    for i, uv in enumerate(corner_uv):
        got = coords.evaluate(element=1, xi=uv)
        np.testing.assert_allclose(got, nodes[i], atol=1e-12)


def test_hermite_hex_node_values_preserved() -> None:
    """Evaluating at the 8 corner xi's must return the value DOFs exactly."""
    rng = np.random.default_rng(11)
    nodes = rng.standard_normal((8, 3))
    derivs = rng.standard_normal((8, 7, 3))
    conn = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    doc = _build_hermite_hex_doc(nodes, derivs, conn)
    coords = doc.field("coordinates")

    corners_uvw = np.array(
        [[a, bc, cc] for cc in (0, 1) for bc in (0, 1) for a in (0, 1)], dtype=np.float64
    )
    for i, uvw in enumerate(corners_uvw):
        got = coords.evaluate(element=1, xi=uvw)
        np.testing.assert_allclose(got, nodes[i], atol=1e-12)
