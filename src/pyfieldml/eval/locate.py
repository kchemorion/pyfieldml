"""Locate physical points in a Lagrange mesh — cKDTree + inverse-xi Newton."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pyfieldml.eval.bases import Basis

# Default starting xi for inverse-xi Newton per topology (interior of reference element).
_DEFAULT_XI: dict[str, np.ndarray] = {
    "line": np.array([0.5]),
    "triangle": np.array([1 / 3, 1 / 3]),
    "quad": np.array([0.5, 0.5]),
    "tet": np.array([1 / 4, 1 / 4, 1 / 4]),
    "hex": np.array([0.5, 0.5, 0.5]),
    "wedge": np.array([1 / 3, 1 / 3, 0.5]),
}


class SpatialLocator:
    """Find the (element, xi) for each physical point in a Lagrange mesh."""

    def __init__(
        self,
        *,
        nodes: np.ndarray,
        connectivity: np.ndarray,  # 1-indexed, shape (n_elems, n_per)
        basis: Basis,
    ) -> None:
        self._nodes = nodes
        self._conn = connectivity
        self._basis = basis
        centroids = nodes[connectivity - 1].mean(axis=1)  # (n_elems, D)
        self._tree = cKDTree(centroids)
        self._default_xi = _DEFAULT_XI[basis.topology]

    def locate(
        self,
        points: np.ndarray,
        *,
        k_candidates: int = 8,
        max_iter: int = 30,
        tol: float = 1e-10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (elements, xi). Points outside the mesh return element=-1, xi=NaN."""
        points = np.asarray(points, dtype=np.float64)
        n = points.shape[0]
        out_elem = np.full(n, -1, dtype=np.int64)
        out_xi = np.full(points.shape, np.nan, dtype=np.float64)

        k_eff = min(k_candidates, self._conn.shape[0])
        _, nbr_idx = self._tree.query(points, k=k_eff)
        # cKDTree returns shape (n,) when k=1 and (n, k) when k>1; normalise to (n, k).
        nbr_idx = np.asarray(nbr_idx).reshape(n, k_eff)

        for i in range(n):
            for cand in nbr_idx[i]:
                conn = self._conn[cand]
                node_vals = self._nodes[conn - 1]
                xi, ok = self._newton_inverse(points[i], node_vals, max_iter, tol)
                if ok and self._in_reference(xi):
                    out_elem[i] = cand + 1  # 1-indexed element id
                    out_xi[i] = xi
                    break
        return out_elem, out_xi

    def _newton_inverse(
        self,
        x: np.ndarray,
        node_vals: np.ndarray,
        max_iter: int,
        tol: float,
    ) -> tuple[np.ndarray, bool]:
        xi = self._default_xi.copy()
        for _ in range(max_iter):
            phi = self._basis.shape_functions(xi)
            pos = phi @ node_vals
            r = pos - x
            if np.linalg.norm(r) < tol:
                return xi, True
            d_phi = self._basis.shape_derivatives(xi)
            j_mat = np.einsum("nr,nd->dr", d_phi, node_vals)
            try:
                delta = np.linalg.solve(j_mat, r)
            except np.linalg.LinAlgError:
                return xi, False
            xi = xi - delta
        return xi, False

    def _in_reference(self, xi: np.ndarray) -> bool:
        t = self._basis.topology
        if t == "line":
            return bool(0.0 <= xi[0] <= 1.0)
        if t in {"quad", "hex"}:
            return bool(np.all((xi >= 0.0) & (xi <= 1.0)))
        if t == "triangle":
            return bool(xi[0] >= 0.0 and xi[1] >= 0.0 and xi[0] + xi[1] <= 1.0)
        if t == "tet":
            return bool(xi[0] >= 0.0 and xi[1] >= 0.0 and xi[2] >= 0.0 and xi.sum() <= 1.0)
        if t == "wedge":
            return bool(
                xi[0] >= 0.0 and xi[1] >= 0.0 and xi[0] + xi[1] <= 1.0 and 0.0 <= xi[2] <= 1.0
            )
        return False
