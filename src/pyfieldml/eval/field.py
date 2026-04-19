"""User-facing Field wrapper over an evaluable Evaluator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pyfieldml.errors import EvaluationError
from pyfieldml.eval.bases import Basis, get_basis
from pyfieldml.model.evaluators import Evaluator, ParameterEvaluator
from pyfieldml.model.region import Region

ArrayLike = np.ndarray | Sequence[float] | Sequence[int] | float | int


class Field:
    """Ergonomic wrapper for evaluating a FieldML evaluator graph.

    Phase 2: supports Lagrange-only meshes where the region contains:
      - A ParameterEvaluator named ``coord_name`` with node coordinates.
      - A ParameterEvaluator named ``coord_name + ".connectivity"`` holding
        1-indexed element->node connectivity.
      - An ExternalEvaluator whose name begins with ``library.basis.`` naming
        the Lagrange basis (produced by ``add_lagrange_mesh``).

    Phase 3: also supports cubic-Hermite line meshes where the region additionally
    contains:
      - ``coord_name + ".derivatives"`` - per-node derivative DOFs (same shape as
        nodes).
      - ``coord_name + ".scales"`` - per-element per-end derivative scale factors,
        shape ``(n_elems, 2)``.
    """

    def __init__(
        self,
        *,
        evaluator: Evaluator,
        region: Region,
        name: str,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        basis: Basis,
        node_derivatives: np.ndarray | None = None,
        scale_factors: np.ndarray | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._region = region
        self._name = name
        self._nodes = np.asarray(node_coords, dtype=np.float64)
        self._conn = np.asarray(connectivity, dtype=np.int64)  # 1-indexed
        self._basis = basis
        self._node_derivatives = (
            None if node_derivatives is None else np.asarray(node_derivatives, dtype=np.float64)
        )
        self._scale_factors = (
            None if scale_factors is None else np.asarray(scale_factors, dtype=np.float64)
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> np.dtype:
        return self._nodes.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._nodes.shape

    def _is_hermite_line(self) -> bool:
        return (
            getattr(self._basis, "topology", None) == "line"
            and getattr(self._basis, "order", None) == 3
            and self._node_derivatives is not None
        )

    def _is_hermite_quad(self) -> bool:
        return (
            getattr(self._basis, "topology", None) == "quad"
            and getattr(self._basis, "order", None) == 3
            and self._node_derivatives is not None
        )

    def _is_hermite_hex(self) -> bool:
        return (
            getattr(self._basis, "topology", None) == "hex"
            and getattr(self._basis, "order", None) == 3
            and self._node_derivatives is not None
        )

    def _is_hermite(self) -> bool:
        return self._is_hermite_line() or self._is_hermite_quad() or self._is_hermite_hex()

    def _assemble_hermite_line_dofs(self, element_arr: np.ndarray) -> np.ndarray:
        """Build per-point (M, 4, D) DOF block with scales applied.

        DOF order within each element: [v@a, d@a, v@b, d@b]. Scales apply to
        derivative slots only: slot 1 gets ``scales[e, 0]``; slot 3 gets
        ``scales[e, 1]``. Value slots are multiplied by 1.0.
        """
        assert self._node_derivatives is not None
        conn = self._conn[element_arr - 1]  # (M, 2), 1-indexed node IDs
        n_a = conn[:, 0] - 1
        n_b = conn[:, 1] - 1
        v_a = self._nodes[n_a]  # (M, D)
        v_b = self._nodes[n_b]
        d_a = self._node_derivatives[n_a]
        d_b = self._node_derivatives[n_b]

        if self._scale_factors is not None:
            s = self._scale_factors[element_arr - 1]  # (M, 2)
            s_a = s[:, 0:1]
            s_b = s[:, 1:2]
        else:
            s_a = np.ones((element_arr.shape[0], 1), dtype=np.float64)
            s_b = s_a

        dofs = np.stack([v_a, d_a * s_a, v_b, d_b * s_b], axis=1)  # (M, 4, D)
        return dofs

    # User-supplied derivative slot order -> basis per-corner DOF index.
    # Slot 0 of a corner is always the 'value' DOF pulled from self._nodes.
    # Remaining basis per-corner DOFs are populated from self._node_derivatives
    # (stored flat as (N_nodes, n_slots * D), reshaped to (N_nodes, n_slots, D)).
    # Quad: user slots (d/dxi1, d/dxi2, d2/dxi1dxi2) -> basis per-corner indices
    # (1, 2, 3) — identity, since the basis orders per-corner DOFs as
    # f_u + 2*f_v.
    # Hex: user slots
    #   (d/dxi1, d/dxi2, d/dxi3, d2/dxi1dxi2, d2/dxi1dxi3, d2/dxi2dxi3,
    #    d3/dxi1dxi2dxi3)
    # map to basis per-corner indices (1, 2, 4, 3, 5, 6, 7) because the basis
    # orders per-corner DOFs as f_u + 2*f_v + 4*f_w.
    _HERMITE_QUAD_SLOT_TO_BASIS: tuple[int, ...] = (1, 2, 3)
    _HERMITE_HEX_SLOT_TO_BASIS: tuple[int, ...] = (1, 2, 4, 3, 5, 6, 7)

    def _assemble_hermite_tensor_dofs(
        self,
        element_arr: np.ndarray,
        *,
        n_corners: int,
        dofs_per_corner: int,
        slot_to_basis: tuple[int, ...],
    ) -> np.ndarray:
        """Assemble (M, n_corners * dofs_per_corner, D) DOF block for quad/hex.

        Value DOF at basis index 0 within each corner comes from ``self._nodes``.
        Derivative slots come from ``self._node_derivatives`` reshaped to
        ``(N_nodes, n_slots, D)`` and remapped via ``slot_to_basis`` into the
        basis per-corner DOF layout. ``self._scale_factors``, if present, has
        shape ``(n_elems, n_dofs_per_elem)`` and multiplies each DOF.
        """
        assert self._node_derivatives is not None
        d = self._nodes.shape[1]
        n_slots = dofs_per_corner - 1
        n_nodes_total = self._nodes.shape[0]
        # node_derivs_slots: (N_nodes, n_slots, D)
        node_derivs_slots = self._node_derivatives.reshape(n_nodes_total, n_slots, d)

        conn = self._conn[element_arr - 1]  # (M, n_corners)
        m = element_arr.shape[0]
        n_dofs_per_elem = n_corners * dofs_per_corner
        dofs = np.empty((m, n_dofs_per_elem, d), dtype=np.float64)

        for corner_idx in range(n_corners):
            node_ids = conn[:, corner_idx] - 1  # (M,)
            # value DOF -> basis per-corner index 0
            base = corner_idx * dofs_per_corner
            dofs[:, base + 0, :] = self._nodes[node_ids]
            # derivative slots
            for slot, basis_idx in enumerate(slot_to_basis):
                dofs[:, base + basis_idx, :] = node_derivs_slots[node_ids, slot, :]

        if self._scale_factors is not None:
            s = self._scale_factors[element_arr - 1]  # (M, n_dofs_per_elem)
            dofs = dofs * s[:, :, None]

        return dofs

    def _assemble_hermite_dofs(self, element_arr: np.ndarray) -> np.ndarray:
        """Dispatch to per-topology Hermite DOF assembly."""
        if self._is_hermite_line():
            return self._assemble_hermite_line_dofs(element_arr)
        if self._is_hermite_quad():
            return self._assemble_hermite_tensor_dofs(
                element_arr,
                n_corners=4,
                dofs_per_corner=4,
                slot_to_basis=self._HERMITE_QUAD_SLOT_TO_BASIS,
            )
        if self._is_hermite_hex():
            return self._assemble_hermite_tensor_dofs(
                element_arr,
                n_corners=8,
                dofs_per_corner=8,
                slot_to_basis=self._HERMITE_HEX_SLOT_TO_BASIS,
            )
        raise AssertionError("_assemble_hermite_dofs called on non-Hermite field")

    def evaluate(
        self,
        *,
        element: int | np.ndarray,
        xi: ArrayLike,
    ) -> np.ndarray:
        """Evaluate the field at parametric coordinate(s) within element(s)."""
        element_arr = np.atleast_1d(np.asarray(element, dtype=np.int64))
        xi_arr = np.atleast_2d(np.asarray(xi, dtype=np.float64))
        if xi_arr.shape[0] == 1 and element_arr.shape[0] > 1:
            xi_arr = np.broadcast_to(xi_arr, (element_arr.shape[0], xi_arr.shape[1])).copy()

        phi = self._basis.shape_functions(xi_arr)  # (M, N)

        if self._is_hermite():
            dofs = self._assemble_hermite_dofs(element_arr)  # (M, n_dofs, D)
            result: np.ndarray = np.einsum("mn,mnd->md", phi, dofs)
        else:
            # Elements are 1-indexed; node IDs inside conn are 1-indexed too.
            conn = self._conn[element_arr - 1]
            node_vals = self._nodes[conn - 1]  # (M, N, D)
            result = np.einsum("mn,mnd->md", phi, node_vals)

        if result.shape[0] == 1:
            out: np.ndarray = result[0]
            return out
        return result

    def jacobian(self, *, element: int, xi: ArrayLike) -> np.ndarray:
        """Return the d(field)/d(xi) Jacobian at the given element and xi.

        For a D-valued field on an R-dim reference element, returns shape ``(D, R)``.
        """
        xi_arr = np.atleast_2d(np.asarray(xi, dtype=np.float64))
        d_phi = self._basis.shape_derivatives(xi_arr)  # (M, N, R)
        element_arr = np.atleast_1d(np.asarray(element, dtype=np.int64))

        if self._is_hermite():
            dofs = self._assemble_hermite_dofs(element_arr)  # (1, n_dofs, D)
            # J[d, r] = sum over n of d_phi[0, n, r] * dofs[0, n, d]
            j = np.einsum("mnr,mnd->mdr", d_phi, dofs)
            result: np.ndarray = j[0]
            return result

        conn = self._conn[element_arr[0] - 1]
        node_vals = self._nodes[conn - 1]  # (N, D)
        j = np.einsum("mnr,nd->mdr", d_phi, node_vals)
        result = j[0]
        return result

    def sample(self, points: ArrayLike) -> np.ndarray:
        """Sample the field at physical points. Points outside the mesh return NaN."""
        if self._is_hermite():
            raise NotImplementedError(
                "Field.sample is not supported on Hermite meshes in v1.0.x. "
                "Use Field.evaluate(element=..., xi=...) with a known element + xi instead. "
                "A Hermite-aware spatial locator is planned for v1.1."
            )
        from pyfieldml.eval.locate import SpatialLocator

        points_arr = np.asarray(points, dtype=np.float64)
        locator = SpatialLocator(nodes=self._nodes, connectivity=self._conn, basis=self._basis)
        elems, xis = locator.locate(points_arr)
        out = np.full(
            (points_arr.shape[0], self._nodes.shape[1]),
            np.nan,
            dtype=np.float64,
        )
        inside = elems > 0
        if inside.any():
            out[inside] = self.evaluate(element=elems[inside], xi=xis[inside])
        return out

    def as_ndarray(self) -> np.ndarray:
        """Return the underlying node-value DOFs as an ndarray."""
        return self._nodes

    def plot(self, doc: Any = None, **kwargs: Any) -> Any:
        """Render this Field via PyVista. Requires a Document + pyfieldml[viz]."""
        if doc is None:
            raise ValueError(
                "Field.plot(doc=...) requires the parent Document; Phase-4 can't "
                "resolve the grid without it."
            )
        try:
            from pyfieldml.interop.pyvista import plot_field
        except ImportError as exc:
            raise ImportError(
                "field.plot() requires the [viz] extra: pip install pyfieldml[viz]"
            ) from exc
        return plot_field(self, doc, **kwargs)


def resolve_field(region: Region, *, name: str) -> Field:
    """Best-effort graph resolution from a FieldML evaluator to a Field.

    Phase-2 simplification documented on ``Field``. Users building graphs
    manually must follow the convention produced by ``add_lagrange_mesh``
    or ``add_hermite_mesh``.
    """
    ev = region.evaluators.get(name)
    if ev is None:
        raise EvaluationError(f"No evaluator named {name!r} in region {region.name!r}")
    if not isinstance(ev, ParameterEvaluator):
        raise EvaluationError(
            f"Field resolution requires a ParameterEvaluator for node "
            f"coordinates; {name!r} is {type(ev).__name__}"
        )

    conn_name = f"{name}.connectivity"
    conn_ev = region.evaluators.get(conn_name)
    if not isinstance(conn_ev, ParameterEvaluator):
        raise EvaluationError(f"No connectivity parameter {conn_name!r} found for {name!r}")

    basis_name = next(
        (e.name for e in region.evaluators.values() if e.name.startswith("library.basis.")),
        None,
    )
    if basis_name is None:
        raise EvaluationError(f"No library.basis.* evaluator found in region {region.name!r}")
    basis = get_basis(basis_name)

    # Optional Hermite-mesh parameters.
    derivs_ev = region.evaluators.get(f"{name}.derivatives")
    scales_ev = region.evaluators.get(f"{name}.scales")
    node_derivatives = derivs_ev.as_ndarray() if isinstance(derivs_ev, ParameterEvaluator) else None
    scale_factors = scales_ev.as_ndarray() if isinstance(scales_ev, ParameterEvaluator) else None

    return Field(
        evaluator=ev,
        region=region,
        name=name,
        node_coords=ev.as_ndarray(),
        connectivity=conn_ev.as_ndarray(),
        basis=basis,
        node_derivatives=node_derivatives,
        scale_factors=scale_factors,
    )
