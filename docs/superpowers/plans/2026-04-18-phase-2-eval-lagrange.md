# pyfieldml Phase 2 — Evaluation Engine (Lagrange) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.
>
> **⚠ Forward-draft notice:** This plan assumes the types, method names, and module layout established in Phase 1. If Phase 1 shipped with divergent naming, update this plan before starting Phase 2.

**Goal:** Ship a working evaluation engine over the Phase-1 model graph. A user must be able to: `doc.field("coords").evaluate(element=5, xi=(0.5, 0.5, 0.5))` on a Lagrange mesh and get back the correct spatial position. Vectorized evaluation, Jacobians, and spatial `sample(points)` also work. Tag `v0.2.0`.

**Architecture:** New `eval/` subpackage. Basis-function registry pattern: built-in Lagrange bases (orders 1 + 2 on line/tri/quad/tet/hex/wedge) register themselves on import; external users register their own via `@pyfieldml.basis("name", topology="tet")`. The evaluator-graph walker consumes the Phase-1 model graph, resolves argument bindings, and dispatches to basis kernels. A lightweight `Field` wrapper exposes ergonomic evaluate/sample/jacobian on top of any evaluable evaluator.

**Tech Stack:** `numpy` (vectorized shape functions), `scipy.spatial.cKDTree` (point location), `scipy.optimize` (inverse-xi Newton).

**Spec reference:** spec §2.2 Pillar A (evaluation engine, Lagrange), §4.2 (evaluate/sample/jacobian API).

**Completion criteria:**
- `Field.evaluate(element, xi)` works for scalar and vector fields on all 12 Lagrange topology×order combinations.
- Vectorized evaluation `evaluate(element=ndarray, xi=ndarray)` is ≥50× faster than a Python loop over the scalar version on 10k points.
- `Field.sample(points)` locates each point in the mesh and returns interpolated values with accuracy < 1e-10 on a unit-cube Lagrange mesh.
- `Field.jacobian(element, xi)` returns the correct `(d_value, d_xi)` Jacobian.
- `@pyfieldml.basis` decorator successfully registers an external basis and the engine uses it.
- `v0.2.0` tagged + published.

---

## File structure created

| Path | Responsibility |
|------|----------------|
| `src/pyfieldml/eval/__init__.py` | Re-exports `Field`, `basis` decorator, registry accessors |
| `src/pyfieldml/eval/bases/__init__.py` | Re-exports protocols, built-ins |
| `src/pyfieldml/eval/bases/base.py` | `Basis` protocol (shape_functions, shape_derivatives, topology, order, n_nodes) |
| `src/pyfieldml/eval/bases/registry.py` | `BasisRegistry`, `@basis` decorator, `get_basis(name)` |
| `src/pyfieldml/eval/bases/lagrange.py` | Linear + quadratic Lagrange on 6 topologies — 12 entries |
| `src/pyfieldml/eval/graph.py` | Evaluator-graph walker — resolves References, Aggregates, Piecewise, Constants |
| `src/pyfieldml/eval/context.py` | `EvaluationContext` — carries BindingTable + element/xi scope |
| `src/pyfieldml/eval/field.py` | `Field` user-facing class |
| `src/pyfieldml/eval/locate.py` | `SpatialLocator` — cKDTree + inverse-xi Newton |
| `src/pyfieldml/cli/bench.py` | `pyfieldml bench` stub |
| `tests/unit/test_bases_registry.py` | Registry tests |
| `tests/unit/test_bases_lagrange.py` | Shape-function + derivative numerics |
| `tests/unit/test_eval_graph.py` | Graph walker tests |
| `tests/unit/test_field_evaluate.py` | End-to-end `Field.evaluate` |
| `tests/unit/test_field_sample.py` | Spatial sampling |
| `tests/unit/test_field_jacobian.py` | Jacobians |
| `tests/unit/test_external_basis_plugin.py` | External `@basis` plugin wiring |
| `tests/benchmarks/test_throughput.py` | Vectorized-evaluation throughput check |

---

## Task 1 — Basis protocol + registry

- [ ] **Step 1: Failing tests — `tests/unit/test_bases_registry.py`**

```python
import numpy as np
import pytest

from pyfieldml.eval.bases import Basis, get_basis, list_bases
from pyfieldml.eval.bases.registry import basis
from pyfieldml.errors import BasisNotFoundError


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
            shape = xi.shape[:-1] + (2, 1)
            out = np.empty(shape)
            out[..., 0, 0] = -1.0
            out[..., 1, 0] = 1.0
            return out

    b = get_basis("tests.identity")
    assert b.topology == "line"
    assert b.order == 1
    np.testing.assert_array_almost_equal(
        b.shape_functions(np.array([0.25])), [0.75, 0.25]
    )


def test_list_bases_includes_builtins() -> None:
    names = list_bases()
    # Built-ins registered on import of the module.
    assert any(n.startswith("library.basis.linear_lagrange") for n in names)
```

- [ ] **Step 2: Implement `src/pyfieldml/eval/bases/base.py`**

```python
"""Basis-function protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Basis(Protocol):
    """Protocol for any basis function usable by the evaluation engine.

    Implementations must be stateless. Shape arrays use the convention:

        xi:      shape (..., D)       where D = topology-dependent reference dimension
        φ(xi):   shape (..., N)       where N = number of nodes / DOFs
        ∂φ/∂xi: shape (..., N, D)
    """

    topology: str
    order: int
    n_nodes: int

    def shape_functions(self, xi: np.ndarray) -> np.ndarray: ...
    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray: ...
```

- [ ] **Step 3: Implement `src/pyfieldml/eval/bases/registry.py`**

```python
"""Runtime registry for basis functions."""

from __future__ import annotations

from typing import Callable, TypeVar

from pyfieldml.errors import BasisNotFoundError
from pyfieldml.eval.bases.base import Basis

_registry: dict[str, Basis] = {}

T = TypeVar("T")


def basis(
    name: str, *, topology: str, order: int
) -> Callable[[type[T]], type[T]]:
    """Class decorator: register a basis-function implementation under ``name``."""

    def wrap(cls: type[T]) -> type[T]:
        instance = cls()  # type: ignore[call-arg]
        setattr(instance, "topology", topology)
        setattr(instance, "order", order)
        _registry[name] = instance  # type: ignore[assignment]
        return cls

    return wrap


def get_basis(name: str) -> Basis:
    if name not in _registry:
        raise BasisNotFoundError(f"No basis registered under {name!r}")
    return _registry[name]


def list_bases() -> list[str]:
    return sorted(_registry.keys())


def _clear_for_tests() -> None:
    """Reset the registry — tests only."""
    _registry.clear()
```

- [ ] **Step 4: Implement `src/pyfieldml/eval/bases/__init__.py`**

```python
"""Basis-function registry and protocol."""

from pyfieldml.eval.bases.base import Basis
from pyfieldml.eval.bases.registry import basis, get_basis, list_bases

# Importing this triggers built-in basis registration.
from pyfieldml.eval.bases import lagrange  # noqa: F401

__all__ = ["Basis", "basis", "get_basis", "list_bases"]
```

- [ ] **Step 5: Write a stub `src/pyfieldml/eval/bases/lagrange.py`** so the
import above does not fail (real contents in Task 2):

```python
"""Lagrange basis functions. Registered on import.

Populated by subsequent tasks.
"""
```

- [ ] **Step 6: Run tests. The first two pass; `test_list_bases_includes_builtins` will fail until Lagrange is implemented (Task 2).**

- [ ] **Step 7: Commit**

```bash
git add src/pyfieldml/eval/bases/ tests/unit/test_bases_registry.py
git commit -m "feat(eval): basis-function protocol + runtime registry"
```

---

## Task 2 — Linear Lagrange on 6 topologies

**Topologies + node counts:**

| Topology | Linear n_nodes | Reference element |
|---|---|---|
| line  | 2 | `xi ∈ [0, 1]` |
| tri   | 3 | unit triangle, vertices `(0,0) (1,0) (0,1)` |
| quad  | 4 | unit square `[0,1]²`, CCW |
| tet   | 4 | unit tet, vertices `(0,0,0) (1,0,0) (0,1,0) (0,0,1)` |
| hex   | 8 | unit cube `[0,1]³`, CMISS node order |
| wedge | 6 | prism: linear tri × linear line |

- [ ] **Step 1: Failing tests — `tests/unit/test_bases_lagrange.py`**

```python
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
    """φ_i(node_j) = δ_ij."""
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
        "library.basis.linear_lagrange.tet": [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ],
        "library.basis.linear_lagrange.hex": [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ],
        "library.basis.linear_lagrange.wedge": [
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
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
```

- [ ] **Step 2: Implement Lagrange bases in `src/pyfieldml/eval/bases/lagrange.py`**

```python
"""Linear and quadratic Lagrange basis functions on the standard topologies.

Reference element conventions:

- line:  xi ∈ [0, 1]
- tri:   unit triangle, vertices (0,0) (1,0) (0,1)
- quad:  unit square [0,1]², CCW
- tet:   unit tet, vertices (0,0,0) (1,0,0) (0,1,0) (0,0,1)
- hex:   unit cube [0,1]³; CMISS node order = lexicographic in (xi1, xi2, xi3)
- wedge: tri × line, i.e. triangle at xi3=0 and triangle at xi3=1
"""

from __future__ import annotations

import numpy as np

from pyfieldml.eval.bases.registry import basis


# ------------------------------------------------------------------ LINE, linear

@basis("library.basis.linear_lagrange.line", topology="line", order=1)
class LinearLagrangeLine:
    n_nodes = 2
    topology = "line"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        x = xi[..., 0]
        return np.stack([1.0 - x, x], axis=-1)

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        out = np.empty(xi.shape[:-1] + (2, 1), dtype=np.float64)
        out[..., 0, 0] = -1.0
        out[..., 1, 0] = 1.0
        return out


# ------------------------------------------------------------------ TRIANGLE, linear

@basis("library.basis.linear_lagrange.triangle", topology="triangle", order=1)
class LinearLagrangeTriangle:
    n_nodes = 3
    topology = "triangle"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        u, v = xi[..., 0], xi[..., 1]
        return np.stack([1.0 - u - v, u, v], axis=-1)

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        out = np.zeros(xi.shape[:-1] + (3, 2), dtype=np.float64)
        out[..., 0, 0] = -1.0
        out[..., 0, 1] = -1.0
        out[..., 1, 0] = 1.0
        out[..., 2, 1] = 1.0
        return out


# ------------------------------------------------------------------ QUAD, linear (bilinear)

@basis("library.basis.linear_lagrange.quad", topology="quad", order=1)
class LinearLagrangeQuad:
    n_nodes = 4
    topology = "quad"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        u, v = xi[..., 0], xi[..., 1]
        return np.stack(
            [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v], axis=-1
        )

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        u, v = xi[..., 0], xi[..., 1]
        out = np.empty(xi.shape[:-1] + (4, 2), dtype=np.float64)
        out[..., 0, 0] = -(1 - v)
        out[..., 0, 1] = -(1 - u)
        out[..., 1, 0] = (1 - v)
        out[..., 1, 1] = -u
        out[..., 2, 0] = -v
        out[..., 2, 1] = (1 - u)
        out[..., 3, 0] = v
        out[..., 3, 1] = u
        return out


# ------------------------------------------------------------------ TET, linear

@basis("library.basis.linear_lagrange.tet", topology="tet", order=1)
class LinearLagrangeTet:
    n_nodes = 4
    topology = "tet"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        return np.stack([1.0 - u - v - w, u, v, w], axis=-1)

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        out = np.zeros(xi.shape[:-1] + (4, 3), dtype=np.float64)
        out[..., 0, :] = -1.0
        out[..., 1, 0] = 1.0
        out[..., 2, 1] = 1.0
        out[..., 3, 2] = 1.0
        return out


# ------------------------------------------------------------------ HEX, linear (trilinear)

@basis("library.basis.linear_lagrange.hex", topology="hex", order=1)
class LinearLagrangeHex:
    n_nodes = 8
    topology = "hex"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        N = np.stack(
            [
                (1 - u) * (1 - v) * (1 - w),
                u * (1 - v) * (1 - w),
                (1 - u) * v * (1 - w),
                u * v * (1 - w),
                (1 - u) * (1 - v) * w,
                u * (1 - v) * w,
                (1 - u) * v * w,
                u * v * w,
            ],
            axis=-1,
        )
        return N

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        shape = xi.shape[:-1] + (8, 3)
        out = np.empty(shape, dtype=np.float64)
        out[..., 0, 0] = -(1 - v) * (1 - w)
        out[..., 0, 1] = -(1 - u) * (1 - w)
        out[..., 0, 2] = -(1 - u) * (1 - v)
        out[..., 1, 0] = (1 - v) * (1 - w)
        out[..., 1, 1] = -u * (1 - w)
        out[..., 1, 2] = -u * (1 - v)
        out[..., 2, 0] = -v * (1 - w)
        out[..., 2, 1] = (1 - u) * (1 - w)
        out[..., 2, 2] = -(1 - u) * v
        out[..., 3, 0] = v * (1 - w)
        out[..., 3, 1] = u * (1 - w)
        out[..., 3, 2] = -u * v
        out[..., 4, 0] = -(1 - v) * w
        out[..., 4, 1] = -(1 - u) * w
        out[..., 4, 2] = (1 - u) * (1 - v)
        out[..., 5, 0] = (1 - v) * w
        out[..., 5, 1] = -u * w
        out[..., 5, 2] = u * (1 - v)
        out[..., 6, 0] = -v * w
        out[..., 6, 1] = (1 - u) * w
        out[..., 6, 2] = (1 - u) * v
        out[..., 7, 0] = v * w
        out[..., 7, 1] = u * w
        out[..., 7, 2] = u * v
        return out


# ------------------------------------------------------------------ WEDGE, linear (tri × line)

@basis("library.basis.linear_lagrange.wedge", topology="wedge", order=1)
class LinearLagrangeWedge:
    n_nodes = 6
    topology = "wedge"
    order = 1

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        tri = np.stack([1.0 - u - v, u, v], axis=-1)
        line = np.stack([1.0 - w, w], axis=-1)
        # Outer product: (..., 3, 2) -> flatten to (..., 6) in wedge node order.
        outer = tri[..., :, None] * line[..., None, :]
        return outer.reshape(*xi.shape[:-1], 6)

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        u, v, w = xi[..., 0], xi[..., 1], xi[..., 2]
        shape = xi.shape[:-1] + (6, 3)
        d = np.empty(shape, dtype=np.float64)
        # Order: (bot-tri-0, bot-tri-1, bot-tri-2, top-tri-0, top-tri-1, top-tri-2).
        # d/du, d/dv, d/dw
        bot = (1.0 - w)
        top = w
        # Node 0: (1-u-v)(1-w)
        d[..., 0, 0] = -bot
        d[..., 0, 1] = -bot
        d[..., 0, 2] = -(1.0 - u - v)
        # Node 1: u(1-w)
        d[..., 1, 0] = bot
        d[..., 1, 1] = 0.0
        d[..., 1, 2] = -u
        # Node 2: v(1-w)
        d[..., 2, 0] = 0.0
        d[..., 2, 1] = bot
        d[..., 2, 2] = -v
        # Node 3: (1-u-v)w
        d[..., 3, 0] = -top
        d[..., 3, 1] = -top
        d[..., 3, 2] = (1.0 - u - v)
        # Node 4: u·w
        d[..., 4, 0] = top
        d[..., 4, 1] = 0.0
        d[..., 4, 2] = u
        # Node 5: v·w
        d[..., 5, 0] = 0.0
        d[..., 5, 1] = top
        d[..., 5, 2] = v
        return d
```

- [ ] **Step 3: Run tests, pass**

Run: `uv run pytest tests/unit/test_bases_lagrange.py -v`
Expected: 12 parametrized cases pass.

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/eval/bases/lagrange.py tests/unit/test_bases_lagrange.py
git commit -m "feat(eval): linear Lagrange bases on line/tri/quad/tet/hex/wedge"
```

---

## Task 3 — Quadratic Lagrange on 6 topologies

Quadratic nodes follow CMISS / OpenCMISS convention for 2D and 3D elements.

| Topology | Quadratic n_nodes | Nodes |
|---|---|---|
| line  | 3  | 0, 0.5, 1.0 |
| tri   | 6  | 3 vertex + 3 edge midpoints |
| quad  | 9  | 3×3 grid |
| tet   | 10 | 4 vertex + 6 edge midpoints |
| hex   | 27 | 3×3×3 grid |
| wedge | 18 | quadratic tri × quadratic line |

- [ ] **Step 1: Add parametrized tests** (copy the template from Task 2's tests, add `quadratic_lagrange.*` entries with the correct `n_nodes`).

- [ ] **Step 2: Implement** — append to `lagrange.py`.

For each topology, the pattern is:

- line: `N_i = product of 1D Lagrange polynomials` for 3 nodes at 0, 0.5, 1.
- hex: tensor product of three 1D quadratic Lagrange sets.
- quad: tensor product of two 1D quadratic Lagrange sets.
- tri / tet: compute via barycentric coordinates — the canonical formulas are available in any FE textbook. Encode carefully; test against the Kronecker-delta property for the node table.

This task is mechanical but error-prone. Allocate a full day. Use SymPy once to derive the shape functions, copy the numeric expressions into numpy code.

Minimum acceptance: all 12 `test_linear_partition_of_unity` + 12 `test_kronecker_delta_at_nodes` pass for orders 1 and 2.

- [ ] **Step 3: Run the full basis-test suite**

Run: `uv run pytest tests/unit/test_bases_lagrange.py -v`
Expected: 24 tests pass (12 linear + 12 quadratic).

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/eval/bases/lagrange.py tests/unit/test_bases_lagrange.py
git commit -m "feat(eval): quadratic Lagrange bases on all 6 topologies"
```

---

## Task 4 — EvaluationContext + graph walker

**Files:**
- Create: `src/pyfieldml/eval/context.py`
- Create: `src/pyfieldml/eval/graph.py`
- Test: `tests/unit/test_eval_graph.py`

- [ ] **Step 1: Failing tests**

```python
"""Graph-walker tests."""

from __future__ import annotations

import numpy as np

from pyfieldml.eval.context import EvaluationContext
from pyfieldml.eval.graph import evaluate_evaluator
from pyfieldml.model.bindings import ArgumentBinding, BindingTable
from pyfieldml.model.evaluators import (
    ArgumentEvaluator,
    ConstantEvaluator,
    ReferenceEvaluator,
)
from pyfieldml.model.types import ContinuousType


def test_constant_evaluator_returns_scalar() -> None:
    t = ContinuousType(name="real.1d")
    c = ConstantEvaluator(name="k", value_type=t, value=3.14)
    ctx = EvaluationContext()
    assert evaluate_evaluator(c, ctx) == 3.14


def test_reference_evaluator_with_binding() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    inner = ReferenceEvaluator(name="ref", value_type=t, source=arg, bindings=BindingTable([]))
    concrete = ConstantEvaluator(name="src", value_type=t, value=9.0)
    ref = ReferenceEvaluator(
        name="outer",
        value_type=t,
        source=inner,
        bindings=BindingTable([ArgumentBinding(argument=arg, source=concrete)]),
    )
    ctx = EvaluationContext()
    assert evaluate_evaluator(ref, ctx) == 9.0
```

- [ ] **Step 2: Implement `src/pyfieldml/eval/context.py`**

```python
"""Evaluation-time scope: binding tables, element/xi, free-variable map."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyfieldml.model.bindings import BindingTable


@dataclass
class EvaluationContext:
    """Scope carried through a single evaluation call.

    Attributes
    ----------
    bindings
        Stack of binding tables. Outermost table resolves first; later pushes
        override earlier ones within a ReferenceEvaluator.
    element
        Active element id (for Piecewise evaluators).
    xi
        Parametric coordinate within the element.
    """

    bindings: list[BindingTable] = field(default_factory=list)
    element: int | None = None
    xi: np.ndarray | None = None
```

- [ ] **Step 3: Implement `src/pyfieldml/eval/graph.py`**

```python
"""Walk a FieldML evaluator graph and produce numeric values."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyfieldml.errors import EvaluationError, UnboundArgumentError
from pyfieldml.eval.context import EvaluationContext
from pyfieldml.model.evaluators import (
    AggregateEvaluator,
    ArgumentEvaluator,
    ConstantEvaluator,
    Evaluator,
    ExternalEvaluator,
    ParameterEvaluator,
    PiecewiseEvaluator,
    ReferenceEvaluator,
)


def evaluate_evaluator(ev: Evaluator, ctx: EvaluationContext) -> Any:
    """Dispatch on evaluator subtype. Returns the numeric value of ``ev`` under ``ctx``."""
    if isinstance(ev, ConstantEvaluator):
        return ev.evaluate_constant()
    if isinstance(ev, ParameterEvaluator):
        # For raw evaluation — return the full array. Per-element indexing
        # happens at the Field layer.
        return ev.as_ndarray()
    if isinstance(ev, ArgumentEvaluator):
        for table in reversed(ctx.bindings):
            try:
                bound = table.resolve(ev)
            except UnboundArgumentError:
                continue
            return evaluate_evaluator(bound, ctx)
        raise UnboundArgumentError(f"Argument {ev.name!r} unbound")
    if isinstance(ev, ReferenceEvaluator):
        ctx.bindings.append(ev.bindings)
        try:
            return evaluate_evaluator(ev.source, ctx)
        finally:
            ctx.bindings.pop()
    if isinstance(ev, AggregateEvaluator):
        parts = [np.asarray(evaluate_evaluator(c, ctx)) for c in ev.components]
        return np.stack(parts, axis=-1)
    if isinstance(ev, PiecewiseEvaluator):
        if ctx.element is None:
            raise EvaluationError(
                f"PiecewiseEvaluator {ev.name!r} requires an element in context"
            )
        source = ev.pieces.get(ctx.element, ev.default)
        if source is None:
            raise EvaluationError(
                f"PiecewiseEvaluator {ev.name!r}: no case for element {ctx.element}"
            )
        return evaluate_evaluator(source, ctx)
    if isinstance(ev, ExternalEvaluator):
        raise EvaluationError(
            f"ExternalEvaluator {ev.name!r}: the eval engine dispatches externals via "
            f"the basis registry at call-sites, not standalone"
        )
    raise EvaluationError(f"Unknown evaluator type {type(ev).__name__}")
```

- [ ] **Step 4: Run, pass**

Run: `uv run pytest tests/unit/test_eval_graph.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/eval/context.py src/pyfieldml/eval/graph.py tests/unit/test_eval_graph.py
git commit -m "feat(eval): evaluator-graph walker with binding-stack resolution"
```

---

## Task 5 — Field wrapper + `evaluate(element, xi)`

**Files:**
- Create: `src/pyfieldml/eval/field.py`
- Modify: `src/pyfieldml/document.py` (add `Document.field()`)
- Test: `tests/unit/test_field_evaluate.py`

The challenge: knitting together {mesh, node coordinates, element connectivity, basis per element, Parameter DOFs} so that `field.evaluate(element=e, xi=xi)` returns the right value.

Phase 2 scope: support **Lagrange-only** meshes. The field must be a reference/aggregate/parameter chain that ultimately points to (a) a connectivity parameter (nodes per element) and (b) a DOF parameter (values per node), plus an external evaluator naming the basis.

- [ ] **Step 1: Failing tests**

```python
"""End-to-end field evaluation on a unit-cube single-hex mesh."""

from __future__ import annotations

import numpy as np

import pyfieldml as fml
from pyfieldml.eval.field import Field


def _make_unit_cube_doc() -> fml.Document:
    """Hand-build a single-element trilinear-Lagrange unit cube with coord field."""
    from pyfieldml.builders.mesh import add_lagrange_mesh  # added Task 8
    from pyfieldml.model.region import Region

    r = Region(name="cube")
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float64,
    )
    connectivity = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)  # 1-indexed
    mesh, coords = add_lagrange_mesh(
        r, name="cube_mesh", nodes=nodes, elements=connectivity,
        topology="hex", order=1, coord_name="coordinates",
    )
    return fml.Document.from_region(r)


def test_field_evaluate_at_corners_returns_node_positions() -> None:
    doc = _make_unit_cube_doc()
    coords: Field = doc.field("coordinates")
    node_xi = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    for i, xi in enumerate(node_xi):
        got = coords.evaluate(element=1, xi=xi)
        expected = node_xi[i]  # geometry coincides with reference for a unit cube
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_field_evaluate_at_centroid_returns_midpoint() -> None:
    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    got = coords.evaluate(element=1, xi=(0.5, 0.5, 0.5))
    np.testing.assert_allclose(got, [0.5, 0.5, 0.5], atol=1e-12)


def test_field_evaluate_vectorized() -> None:
    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    rng = np.random.default_rng(0)
    xi = rng.random((1000, 3))
    elements = np.full(1000, 1)
    got = coords.evaluate(element=elements, xi=xi)
    # On a unit cube the parametric coords equal the physical coords.
    np.testing.assert_allclose(got, xi, atol=1e-12)
```

Note: this test depends on `add_lagrange_mesh` from Task 8. Either (a) write
a small inline graph-assembly helper inside the test, or (b) mark these
tests with `@pytest.mark.skipif` until Task 8 lands, or (c) reorder: do the
builder first. **Recommendation:** implement `add_lagrange_mesh` as Task 5
instead of Task 8, since the evaluation tests need a way to assemble a mesh.
Below we proceed assuming `add_lagrange_mesh` is co-developed in this task.

- [ ] **Step 2: Implement `src/pyfieldml/eval/field.py`**

```python
"""User-facing Field wrapper over an evaluable Evaluator."""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np

from pyfieldml.errors import EvaluationError
from pyfieldml.eval.bases import Basis, get_basis
from pyfieldml.eval.context import EvaluationContext
from pyfieldml.eval.graph import evaluate_evaluator
from pyfieldml.model.evaluators import Evaluator
from pyfieldml.model.region import Region

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


class Field:
    """Ergonomic wrapper for evaluating a FieldML evaluator graph.

    Phase 2: supports evaluator chains backed by:
      1. Node coordinate parameter (shape ``(n_nodes, D)``).
      2. Element-node connectivity parameter (shape ``(n_elements, n_nodes_per_elem)``, 1-indexed).
      3. A ``library.basis.*_lagrange.*`` external evaluator naming the basis.

    More complex chains (Hermite scaling, per-element basis choice, nested
    references) arrive in Phase 3.
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
        element: Union[int, np.ndarray],
        xi: ArrayLike,
    ) -> np.ndarray:
        """Evaluate the field at parametric coordinate(s) within element(s)."""
        element = np.atleast_1d(np.asarray(element, dtype=np.int64))
        xi = np.atleast_2d(np.asarray(xi, dtype=np.float64))
        if xi.shape[0] == 1 and element.shape[0] > 1:
            xi = np.broadcast_to(xi, (element.shape[0], xi.shape[1])).copy()

        # Shape functions at all xi — shape (M, N)
        phi = self._basis.shape_functions(xi)  # (M, N)

        # Per-point connectivity — shape (M, N)
        conn = self._conn[element - 1]  # 1-indexed → 0-indexed

        # Per-point node coordinates — shape (M, N, D)
        node_vals = self._nodes[conn - 1]

        # Contract shape functions with DOFs: phi[..., None] * node_vals summed over nodes
        result = np.einsum("mn,mnd->md", phi, node_vals)

        # If called with a scalar element + xi, return a flat vector.
        if result.shape[0] == 1:
            return result[0]
        return result
```

- [ ] **Step 3: Add `Document.field()`**

In `src/pyfieldml/document.py`:

```python
    def field(self, name: str) -> "Field":
        """Return a Field wrapper for the evaluator named ``name``.

        Phase-2 resolution is best-effort: it requires the evaluator graph
        to be a simple Lagrange aggregate with known connectivity + nodes.
        A clearer resolution layer arrives in Phase 3.
        """
        from pyfieldml.eval.field import resolve_field

        return resolve_field(self.region, name=name)
```

And add `resolve_field` to `src/pyfieldml/eval/field.py`:

```python
def resolve_field(region: Region, *, name: str) -> Field:
    """Best-effort graph resolution from a FieldML evaluator to a Field.

    Phase-2 simplification: we expect the region to contain
      - A ParameterEvaluator "<name>" whose data is the node-coordinate array.
      - A ParameterEvaluator "<name>.connectivity" whose data is 1-indexed
        element-node connectivity.
      - An ExternalEvaluator whose name begins with "library.basis." naming
        the basis.

    In the Phase-5 builder API this layout is produced automatically by
    ``add_lagrange_mesh``. Users building graphs manually may need to follow
    the same convention.
    """
    ev = region.evaluators.get(name)
    if ev is None:
        raise EvaluationError(f"No evaluator named {name!r} in region {region.name!r}")

    nodes_ev = region.evaluators.get(name)
    conn_ev = region.evaluators.get(f"{name}.connectivity")
    if conn_ev is None:
        raise EvaluationError(
            f"No connectivity parameter {name!r}.connectivity found for {name!r}"
        )

    basis_name = next(
        (e.name for e in region.evaluators.values()
         if e.name.startswith("library.basis.")),
        None,
    )
    if basis_name is None:
        raise EvaluationError(
            f"No library.basis.* evaluator found in region {region.name!r}"
        )
    basis = get_basis(basis_name)

    from pyfieldml.model.evaluators import ParameterEvaluator

    if not isinstance(nodes_ev, ParameterEvaluator) or not isinstance(
        conn_ev, ParameterEvaluator
    ):
        raise EvaluationError(
            "Field resolution requires ParameterEvaluators for both nodes and connectivity"
        )

    return Field(
        evaluator=ev,
        region=region,
        name=name,
        node_coords=nodes_ev.as_ndarray(),
        connectivity=conn_ev.as_ndarray(),
        basis=basis,
    )
```

- [ ] **Step 4: Add a thin `add_lagrange_mesh` that satisfies the resolver convention**

Create `src/pyfieldml/builders/__init__.py` (empty re-export) and
`src/pyfieldml/builders/mesh.py`:

```python
"""High-level builders for common construction patterns."""

from __future__ import annotations

import numpy as np

from pyfieldml.data.text import InlineTextBackend
from pyfieldml.model.evaluators import ExternalEvaluator, ParameterEvaluator
from pyfieldml.model.region import Region
from pyfieldml.model.types import ContinuousType, EnsembleType, MeshType

TOPOLOGY_BASIS_NAME: dict[tuple[str, int], str] = {
    ("line", 1): "library.basis.linear_lagrange.line",
    ("line", 2): "library.basis.quadratic_lagrange.line",
    ("triangle", 1): "library.basis.linear_lagrange.triangle",
    ("triangle", 2): "library.basis.quadratic_lagrange.triangle",
    ("quad", 1): "library.basis.linear_lagrange.quad",
    ("quad", 2): "library.basis.quadratic_lagrange.quad",
    ("tet", 1): "library.basis.linear_lagrange.tet",
    ("tet", 2): "library.basis.quadratic_lagrange.tet",
    ("hex", 1): "library.basis.linear_lagrange.hex",
    ("hex", 2): "library.basis.quadratic_lagrange.hex",
    ("wedge", 1): "library.basis.linear_lagrange.wedge",
    ("wedge", 2): "library.basis.quadratic_lagrange.wedge",
}


def add_lagrange_mesh(
    region: Region,
    *,
    name: str,
    nodes: np.ndarray,
    elements: np.ndarray,
    topology: str,
    order: int,
    coord_name: str = "coordinates",
) -> tuple[MeshType, ParameterEvaluator]:
    """Add a complete Lagrange mesh to ``region``.

    Produces:
        - ContinuousType(coord_name.value_type, N-d)
        - EnsembleType(name + '.elements') and (name + '.nodes')
        - MeshType(name, elements, chart)
        - ParameterEvaluator(coord_name) — node coordinates (N_nodes, D)
        - ParameterEvaluator(coord_name + '.connectivity') — element-node connectivity (N_elems, N_per_elem), 1-indexed
        - ExternalEvaluator(basis_name) — names the Lagrange basis
    """
    d = nodes.shape[1]
    n_elems, n_per = elements.shape
    basis_name = TOPOLOGY_BASIS_NAME[(topology, order)]

    # Types
    vt = ContinuousType(
        name=f"{coord_name}.value_type",
        component_name=f"{coord_name}.component",
        component_count=d,
    )
    region.add_type(vt)

    elem_ens = EnsembleType(name=f"{name}.elements", members=range(1, n_elems + 1))
    region.add_type(elem_ens)
    node_ens = EnsembleType(name=f"{name}.nodes", members=range(1, nodes.shape[0] + 1))
    region.add_type(node_ens)
    chart_ct = ContinuousType(
        name=f"{name}.xi",
        component_name=f"{name}.xi.c",
        component_count={"line": 1, "triangle": 2, "quad": 2, "tet": 3, "hex": 3, "wedge": 3}[topology],
    )
    region.add_type(chart_ct)
    mesh = MeshType(name=name, elements=elem_ens, chart=chart_ct)
    region.add_type(mesh)

    # Parameters: nodes + connectivity
    nodes_data = InlineTextBackend.from_ndarray(nodes)
    coords = ParameterEvaluator(name=coord_name, value_type=vt, data=nodes_data)
    region.add_evaluator(coords)

    conn_data = InlineTextBackend.from_ndarray(elements.astype(np.int64))
    # Give the connectivity its own integer ContinuousType to satisfy FieldML's
    # semantic rule that all parameters have a value type.
    conn_ct = ContinuousType(name=f"{coord_name}.connectivity.vt")
    region.add_type(conn_ct)
    conn = ParameterEvaluator(
        name=f"{coord_name}.connectivity", value_type=conn_ct, data=conn_data
    )
    region.add_evaluator(conn)

    # Basis external
    region.add_evaluator(ExternalEvaluator(name=basis_name, value_type=vt))

    return mesh, coords
```

- [ ] **Step 5: Run, pass**

Run: `uv run pytest tests/unit/test_field_evaluate.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/eval/field.py src/pyfieldml/builders/ \
        src/pyfieldml/document.py tests/unit/test_field_evaluate.py
git commit -m "feat(eval): Field wrapper + resolve_field + add_lagrange_mesh builder"
```

---

## Task 6 — Jacobians

**Files:**
- Modify: `src/pyfieldml/eval/field.py` (add `jacobian`)
- Test: `tests/unit/test_field_jacobian.py`

- [ ] **Step 1: Failing tests**

```python
"""Jacobian tests — check against the identity for the unit cube."""

from __future__ import annotations

import numpy as np


def test_jacobian_is_identity_on_unit_cube() -> None:
    from tests.unit.test_field_evaluate import _make_unit_cube_doc  # reuse helper

    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    J = coords.jacobian(element=1, xi=(0.5, 0.5, 0.5))
    np.testing.assert_allclose(J, np.eye(3), atol=1e-12)


def test_jacobian_on_stretched_cube_scales_correctly() -> None:
    """Stretch the cube to 2x in x; Jacobian should be diag(2,1,1)."""
    from pyfieldml.builders.mesh import add_lagrange_mesh
    from pyfieldml.model.region import Region
    import pyfieldml as fml

    r = Region(name="stretched")
    nodes = np.array(
        [
            [0, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0],
            [0, 0, 1], [2, 0, 1], [0, 1, 1], [2, 1, 1],
        ],
        dtype=np.float64,
    )
    connectivity = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    add_lagrange_mesh(
        r, name="m", nodes=nodes, elements=connectivity,
        topology="hex", order=1, coord_name="coords",
    )
    doc = fml.Document.from_region(r)
    coords = doc.field("coords")
    J = coords.jacobian(element=1, xi=(0.5, 0.5, 0.5))
    np.testing.assert_allclose(J, np.diag([2.0, 1.0, 1.0]), atol=1e-12)
```

- [ ] **Step 2: Implement `Field.jacobian`**

Append to `src/pyfieldml/eval/field.py` in the `Field` class:

```python
    def jacobian(
        self, *, element: int, xi: ArrayLike
    ) -> np.ndarray:
        """Return the ∂(field)/∂(xi) Jacobian at the given element and xi.

        For a D-valued field on an R-dim reference element, returns (D, R).
        """
        xi_arr = np.atleast_2d(np.asarray(xi, dtype=np.float64))
        d_phi = self._basis.shape_derivatives(xi_arr)  # (1, N, R)
        conn = self._conn[element - 1]
        node_vals = self._nodes[conn - 1]  # (N, D)
        # J[d, r] = sum over n of d_phi[0, n, r] * node_vals[n, d]
        J = np.einsum("mnr,nd->mdr", d_phi, node_vals)
        return J[0]
```

- [ ] **Step 3: Run, pass**

Run: `uv run pytest tests/unit/test_field_jacobian.py -v`

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/eval/field.py tests/unit/test_field_jacobian.py
git commit -m "feat(eval): Field.jacobian(element, xi) for Lagrange meshes"
```

---

## Task 7 — Spatial locate (cKDTree + inverse-xi Newton) + `Field.sample`

**Files:**
- Create: `src/pyfieldml/eval/locate.py`
- Modify: `src/pyfieldml/eval/field.py` (add `sample`)
- Test: `tests/unit/test_field_sample.py`

- [ ] **Step 1: Failing tests**

```python
"""Spatial sampling tests."""

from __future__ import annotations

import numpy as np


def test_sample_on_unit_cube_returns_input_points() -> None:
    from tests.unit.test_field_evaluate import _make_unit_cube_doc

    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    rng = np.random.default_rng(42)
    pts = rng.random((500, 3))
    vals = coords.sample(pts)
    np.testing.assert_allclose(vals, pts, atol=1e-10)


def test_sample_out_of_bounds_returns_nan() -> None:
    from tests.unit.test_field_evaluate import _make_unit_cube_doc

    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    outside = np.array([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]])
    vals = coords.sample(outside)
    assert np.isnan(vals).all()
```

- [ ] **Step 2: Implement `src/pyfieldml/eval/locate.py`**

```python
"""Locate physical points in a Lagrange mesh — cKDTree + inverse-xi Newton."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pyfieldml.eval.bases import Basis

# Per-topology sensible starting xi and element-side xi ranges.
_DEFAULT_XI = {
    "line":     np.array([0.5]),
    "triangle": np.array([1/3, 1/3]),
    "quad":     np.array([0.5, 0.5]),
    "tet":      np.array([1/4, 1/4, 1/4]),
    "hex":      np.array([0.5, 0.5, 0.5]),
    "wedge":    np.array([1/3, 1/3, 0.5]),
}


class SpatialLocator:
    """Find the (element, xi) for each physical point in a Lagrange mesh."""

    def __init__(
        self,
        *,
        nodes: np.ndarray,
        connectivity: np.ndarray,   # 1-indexed, (n_elems, n_per)
        basis: Basis,
    ) -> None:
        self._nodes = nodes
        self._conn = connectivity
        self._basis = basis
        # Centroid tree for candidate selection.
        centroids = nodes[(connectivity - 1)].mean(axis=1)  # (n_elems, D)
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

        _, nbr_idx = self._tree.query(points, k=min(k_candidates, self._conn.shape[0]))
        nbr_idx = np.atleast_2d(nbr_idx)

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
        self, x: np.ndarray, node_vals: np.ndarray, max_iter: int, tol: float
    ) -> tuple[np.ndarray, bool]:
        xi = self._default_xi.copy()
        for _ in range(max_iter):
            phi = self._basis.shape_functions(xi)
            pos = phi @ node_vals
            r = pos - x
            if np.linalg.norm(r) < tol:
                return xi, True
            d_phi = self._basis.shape_derivatives(xi)
            # Jacobian J[d, r] = sum_n d_phi[n, r] * node_vals[n, d]
            J = np.einsum("nr,nd->dr", d_phi, node_vals)
            try:
                delta = np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                return xi, False
            xi = xi - delta
        return xi, False

    def _in_reference(self, xi: np.ndarray) -> bool:
        t = self._basis.topology
        if t == "line":
            return 0.0 <= xi[0] <= 1.0
        if t == "quad":
            return all(0.0 <= c <= 1.0 for c in xi)
        if t == "triangle":
            return xi[0] >= 0.0 and xi[1] >= 0.0 and xi[0] + xi[1] <= 1.0
        if t == "hex":
            return all(0.0 <= c <= 1.0 for c in xi)
        if t == "tet":
            return (
                xi[0] >= 0.0 and xi[1] >= 0.0 and xi[2] >= 0.0
                and xi.sum() <= 1.0
            )
        if t == "wedge":
            return (
                xi[0] >= 0.0 and xi[1] >= 0.0 and xi[0] + xi[1] <= 1.0
                and 0.0 <= xi[2] <= 1.0
            )
        return False
```

- [ ] **Step 3: Add `Field.sample`**

In `src/pyfieldml/eval/field.py`, in the `Field` class:

```python
    def sample(self, points: ArrayLike) -> np.ndarray:
        """Sample the field at physical points. Points outside the mesh return NaN."""
        from pyfieldml.eval.locate import SpatialLocator

        points = np.asarray(points, dtype=np.float64)
        locator = SpatialLocator(
            nodes=self._nodes, connectivity=self._conn, basis=self._basis
        )
        elems, xis = locator.locate(points)
        out = np.full((points.shape[0], self._nodes.shape[1]), np.nan)
        inside = elems > 0
        if inside.any():
            out[inside] = self.evaluate(element=elems[inside], xi=xis[inside])
        return out
```

- [ ] **Step 4: Run, pass**

Run: `uv run pytest tests/unit/test_field_sample.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/eval/locate.py src/pyfieldml/eval/field.py tests/unit/test_field_sample.py
git commit -m "feat(eval): SpatialLocator (cKDTree + inverse-xi Newton) + Field.sample"
```

---

## Task 8 — External basis plugin end-to-end test

**Files:**
- Test: `tests/unit/test_external_basis_plugin.py`

- [ ] **Step 1: Test**

```python
"""Verify external user-registered basis functions work end-to-end."""

from __future__ import annotations

import numpy as np

import pyfieldml as fml
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
            out = np.empty(xi.shape[:-1] + (2, 1))
            out[..., 0, 0] = -1.0
            out[..., 1, 0] = 1.0
            return out

    b = fml.eval.bases.get_basis("custom.funky_linear_line")
    np.testing.assert_array_almost_equal(
        b.shape_functions(np.array([0.25])), [0.75, 0.25]
    )
```

- [ ] **Step 2: Run, pass**

Run: `uv run pytest tests/unit/test_external_basis_plugin.py -v`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_external_basis_plugin.py
git commit -m "test(eval): @basis plugin registers and resolves via get_basis"
```

---

## Task 9 — Benchmark + `pyfieldml bench` CLI

**Files:**
- Create: `src/pyfieldml/cli/__init__.py`
- Create: `src/pyfieldml/cli/main.py`
- Create: `src/pyfieldml/cli/bench.py`
- Test: `tests/benchmarks/test_throughput.py`
- Modify: `pyproject.toml` (add CLI entry point)

- [ ] **Step 1: Benchmark test**

```python
"""Throughput check — vectorized eval must be ≥50× scalar Python loop."""

from __future__ import annotations

import time

import numpy as np


def test_vectorized_is_at_least_50x_scalar() -> None:
    from tests.unit.test_field_evaluate import _make_unit_cube_doc

    doc = _make_unit_cube_doc()
    coords = doc.field("coordinates")
    rng = np.random.default_rng(0)
    xi = rng.random((10_000, 3))
    elements = np.ones(10_000, dtype=np.int64)

    t0 = time.perf_counter()
    vec = coords.evaluate(element=elements, xi=xi)
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    scalar = np.stack([coords.evaluate(element=1, xi=xi[i]) for i in range(10_000)])
    t_scalar = time.perf_counter() - t0

    np.testing.assert_allclose(vec, scalar, atol=1e-12)
    assert t_scalar / t_vec >= 50.0, (
        f"vectorized speedup only {t_scalar / t_vec:.1f}× — should be ≥50×"
    )
```

- [ ] **Step 2: Implement CLI entry point + `bench` subcommand**

`src/pyfieldml/cli/main.py`:

```python
"""Top-level `pyfieldml` CLI."""

from __future__ import annotations

import argparse
import sys

from pyfieldml.cli import bench as bench_cmd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pyfieldml")
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("bench", help="Benchmark field evaluation throughput.")
    bp.add_argument("path", help="FieldML document to benchmark.")
    bp.add_argument("--field", required=True, help="Evaluator name to use.")
    bp.add_argument("--n", type=int, default=10_000, help="Point count.")

    args = p.parse_args(argv)
    if args.cmd == "bench":
        return bench_cmd.run(path=args.path, field=args.field, n=args.n)
    return 2
```

`src/pyfieldml/cli/bench.py`:

```python
"""`pyfieldml bench` — measure field evaluation throughput."""

from __future__ import annotations

import time

import numpy as np

import pyfieldml as fml


def run(*, path: str, field: str, n: int) -> int:
    doc = fml.read(path)
    f = doc.field(field)
    rng = np.random.default_rng(0)
    xi = rng.random((n, 3))
    elements = np.ones(n, dtype=np.int64)

    t0 = time.perf_counter()
    f.evaluate(element=elements, xi=xi)
    dt = time.perf_counter() - t0
    print(f"evaluate: {n:,} points in {dt*1000:.2f} ms ({n/dt:.0f} pts/s)")
    return 0
```

- [ ] **Step 3: Wire up entry point in `pyproject.toml`**

```toml
[project.scripts]
pyfieldml = "pyfieldml.cli.main:main"
```

- [ ] **Step 4: Run benchmark**

```bash
uv run pytest tests/benchmarks/ -v
```

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/cli/ tests/benchmarks/ pyproject.toml
git commit -m "feat(cli,bench): pyfieldml bench + throughput regression guard"
```

---

## Task 10 — Phase-2 closeout + v0.2.0

- [ ] **Step 1: Full sanity gauntlet** (same as Phase 1 Task 16 Step 1).
- [ ] **Step 2: CHANGELOG entry for 0.2.0:**

```markdown
## [0.2.0] - YYYY-MM-DD

### Added
- `pyfieldml.eval`: evaluator-graph walker, evaluation context, `Field` wrapper.
- `pyfieldml.eval.bases`: runtime plugin registry (`@basis`), linear + quadratic
  Lagrange on line/tri/quad/tet/hex/wedge (12 built-ins).
- `Field.evaluate(element, xi)`, `Field.sample(points)`, `Field.jacobian`.
- Spatial point location (`cKDTree` + inverse-xi Newton).
- `pyfieldml.builders.add_lagrange_mesh`.
- `pyfieldml bench` CLI subcommand.
```

- [ ] **Step 3: Bump version to 0.2.0, update CITATION.cff, commit, tag `v0.2.0`, push.**

## Phase-2 done-definition

- [ ] All 24 basis tests (linear + quadratic × 6 topologies) pass Kronecker + partition-of-unity.
- [ ] `Field.evaluate` correct on unit-cube at corners + centroid + random xi.
- [ ] Vectorized evaluation ≥50× scalar.
- [ ] `Field.sample` correct on the unit cube (interior points) + NaN outside.
- [ ] `Field.jacobian` identity on unit cube, diag(2,1,1) on stretched cube.
- [ ] External `@basis` plugin round-trips.
- [ ] `pyfieldml bench` runs on the unit-cube fixture.
- [ ] `v0.2.0` on PyPI.

Next: **Phase 3 — Hermite bases + legacy read** (cubic Hermite with per-node scaling; 0.3/0.4 → 0.5 up-converter).
