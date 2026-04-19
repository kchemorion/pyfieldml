"""User-facing Field wrapper over an evaluable Evaluator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pyfieldml.errors import EvaluationError
from pyfieldml.eval.bases import Basis, get_basis
from pyfieldml.model.evaluators import Evaluator, ParameterEvaluator
from pyfieldml.model.region import Region

ArrayLike = np.ndarray | Sequence[float] | Sequence[int]


class Field:
    """Ergonomic wrapper for evaluating a FieldML evaluator graph.

    Phase 2: supports Lagrange-only meshes where the region contains:
      - A ParameterEvaluator named ``coord_name`` with node coordinates.
      - A ParameterEvaluator named ``coord_name + ".connectivity"`` holding
        1-indexed element→node connectivity.
      - An ExternalEvaluator whose name begins with ``library.basis.`` naming
        the Lagrange basis (produced by ``add_lagrange_mesh``).
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
    ) -> None:
        self._evaluator = evaluator
        self._region = region
        self._name = name
        self._nodes = np.asarray(node_coords, dtype=np.float64)
        self._conn = np.asarray(connectivity, dtype=np.int64)  # 1-indexed
        self._basis = basis

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> np.dtype:
        return self._nodes.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._nodes.shape

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
        # Elements are 1-indexed; node IDs inside conn are 1-indexed too.
        conn = self._conn[element_arr - 1]
        node_vals = self._nodes[conn - 1]  # (M, N, D)
        result: np.ndarray = np.einsum("mn,mnd->md", phi, node_vals)

        if result.shape[0] == 1:
            out: np.ndarray = result[0]
            return out
        return result


def resolve_field(region: Region, *, name: str) -> Field:
    """Best-effort graph resolution from a FieldML evaluator to a Field.

    Phase-2 simplification documented on ``Field``. Users building graphs
    manually must follow the convention produced by ``add_lagrange_mesh``.
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

    return Field(
        evaluator=ev,
        region=region,
        name=name,
        node_coords=ev.as_ndarray(),
        connectivity=conn_ev.as_ndarray(),
        basis=basis,
    )
