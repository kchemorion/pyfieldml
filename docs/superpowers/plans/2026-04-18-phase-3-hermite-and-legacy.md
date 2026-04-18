# pyfieldml Phase 3 — Hermite Bases + Legacy Read Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.
>
> **⚠ Forward-draft notice:** Depends on Phase 1 (model) + Phase 2 (eval). Update this plan if earlier phases shipped divergent naming.

**Goal:** Two things, which pair naturally: (1) cubic Hermite bases with per-node scaling on line / bicubic quad / tricubic hex — the bases required by the Auckland cardiac models and several legacy musculoskeletal assets — and (2) up-convert FieldML 0.3 / 0.4 documents to 0.5 on read so the legacy Physiome corpus becomes evaluable. Tag `v0.3.0`.

**Architecture:** `eval/bases/hermite.py` adds three Hermite basis entries plus a scaling-resolution helper. The evaluation engine consumes Hermite DOFs (a per-node tuple of value + derivative components + scale factor). A new `dom/upconverter.py` transforms legacy DOM trees into 0.5-shape trees before model loading; the parser auto-detects version and dispatches. Writing stays 0.5-only.

**Tech Stack:** same as Phase 2. No new runtime deps.

**Spec reference:** §2.2 (Hermite, legacy), §7.7 P3.

**Completion criteria:**
- Cubic Hermite: `Field.evaluate` correct at node parametric positions (φ_i=δ_ij for value DOFs) and at interior xi vs. a Sympy-verified reference.
- Per-node scaling modifies derivative DOFs correctly; C¹ continuity test on a two-element Hermite line passes.
- A FieldML 0.3 fixture parses, up-converts in memory, and the resulting model has the same semantic content as a hand-written 0.5 version of the same asset.
- A FieldML 0.4 fixture round-trips similarly.
- `v0.3.0` tagged + published.

---

## File structure created

| Path | Responsibility |
|------|----------------|
| `src/pyfieldml/eval/bases/hermite.py` | Cubic Hermite bases (line, quad, hex) + scaling helpers |
| `src/pyfieldml/dom/upconverter.py` | 0.3 → 0.5 and 0.4 → 0.5 tree transformations |
| `src/pyfieldml/builders/mesh.py` | Add `add_hermite_mesh(...)` builder |
| `tests/fixtures/legacy_0_3_minimal.fieldml` | Simplest possible 0.3 document |
| `tests/fixtures/legacy_0_3_parameter.fieldml` | 0.3 doc with a ParameterEvaluator (tests data-resource up-conversion) |
| `tests/fixtures/legacy_0_4_minimal.fieldml` | Simplest 0.4 doc |
| `tests/fixtures/hermite_line.fieldml` | Two-element Hermite line with per-node scaling |
| `tests/fixtures/hermite_hex.fieldml` | Single-element tricubic Hermite hex |
| `tests/unit/test_bases_hermite.py` | Hermite numerics |
| `tests/unit/test_hermite_scaling.py` | Scaling + C¹-continuity tests |
| `tests/unit/test_dom_upconverter.py` | Up-converter unit tests |
| `tests/unit/test_legacy_read.py` | End-to-end legacy read |

---

## Task 1 — Cubic Hermite on line (1D)

Cubic Hermite on a line uses 4 DOFs per element: value + first derivative at
each endpoint. The shape functions (for xi ∈ [0, 1]) are:

```
H00(xi) =  2xi³ - 3xi² + 1
H01(xi) =    xi³ - 2xi² + xi     # derivative DOF at xi=0
H10(xi) = -2xi³ + 3xi²
H11(xi) =    xi³ -   xi²         # derivative DOF at xi=1
```

- [ ] **Step 1: Test `tests/unit/test_bases_hermite.py`**

```python
"""Cubic Hermite basis tests."""

from __future__ import annotations

import numpy as np
import pytest

from pyfieldml.eval.bases import get_basis


def test_hermite_line_interpolates_values_at_endpoints() -> None:
    b = get_basis("library.basis.cubic_hermite.line")
    # φ at xi=0: value DOF at xi=0 should contribute 1, all others 0.
    phi_left = b.shape_functions(np.array([0.0]))
    np.testing.assert_allclose(phi_left, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    phi_right = b.shape_functions(np.array([1.0]))
    np.testing.assert_allclose(phi_right, [0.0, 0.0, 1.0, 0.0], atol=1e-12)


def test_hermite_line_derivatives_at_endpoints() -> None:
    b = get_basis("library.basis.cubic_hermite.line")
    d = b.shape_derivatives(np.array([0.0]))
    # d/dxi at xi=0: derivative DOF at xi=0 should give 1; all others give 0.
    np.testing.assert_allclose(d[..., :, 0], [0.0, 1.0, 0.0, 0.0], atol=1e-12)
    d = b.shape_derivatives(np.array([1.0]))
    np.testing.assert_allclose(d[..., :, 0], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_hermite_line_reproduces_cubic() -> None:
    """Cubic Hermite must interpolate a cubic polynomial exactly."""
    b = get_basis("library.basis.cubic_hermite.line")
    # f(xi) = 2xi³ + 3xi² - xi + 5
    # DOFs: f(0)=5, f'(0)=-1, f(1)=9, f'(1)=11
    dofs = np.array([5.0, -1.0, 9.0, 11.0])
    xi_test = np.linspace(0, 1, 21).reshape(-1, 1)
    phi = b.shape_functions(xi_test)
    reconstructed = phi @ dofs
    expected = 2 * xi_test[:, 0]**3 + 3 * xi_test[:, 0]**2 - xi_test[:, 0] + 5
    np.testing.assert_allclose(reconstructed, expected, atol=1e-12)
```

- [ ] **Step 2: Implement `src/pyfieldml/eval/bases/hermite.py`**

```python
"""Cubic Hermite basis functions on line, quad (bicubic), hex (tricubic)."""

from __future__ import annotations

import numpy as np

from pyfieldml.eval.bases.registry import basis


# -------------------------------------------------------------------------- 1D

def _hermite_1d(xi: np.ndarray) -> np.ndarray:
    """Cubic Hermite shape functions on xi ∈ [0, 1].

    Returns array shape (..., 4) in DOF order: H00, H01, H10, H11
    corresponding to (value@0, derivative@0, value@1, derivative@1).
    """
    xi = np.asarray(xi, dtype=np.float64)
    x2 = xi * xi
    x3 = x2 * xi
    return np.stack(
        [
            2.0 * x3 - 3.0 * x2 + 1.0,   # H00
            x3 - 2.0 * x2 + xi,          # H01
            -2.0 * x3 + 3.0 * x2,        # H10
            x3 - x2,                     # H11
        ],
        axis=-1,
    )


def _hermite_1d_derivative(xi: np.ndarray) -> np.ndarray:
    """d/dxi of cubic Hermite. Same DOF order. Returns shape (..., 4)."""
    xi = np.asarray(xi, dtype=np.float64)
    x2 = xi * xi
    return np.stack(
        [
            6.0 * x2 - 6.0 * xi,
            3.0 * x2 - 4.0 * xi + 1.0,
            -6.0 * x2 + 6.0 * xi,
            3.0 * x2 - 2.0 * xi,
        ],
        axis=-1,
    )


@basis("library.basis.cubic_hermite.line", topology="line", order=3)
class CubicHermiteLine:
    n_nodes = 4  # DOFs, not geometric nodes
    topology = "line"
    order = 3

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        return _hermite_1d(xi[..., 0])

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        d = _hermite_1d_derivative(xi[..., 0])  # (..., 4)
        return d[..., :, None]  # (..., 4, 1)
```

- [ ] **Step 3: Run, pass.**

Run: `uv run pytest tests/unit/test_bases_hermite.py::test_hermite_line_interpolates_values_at_endpoints -v`
(Expand to all three tests — 3 pass.)

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/eval/bases/hermite.py tests/unit/test_bases_hermite.py
git commit -m "feat(eval): cubic Hermite line basis"
```

---

## Task 2 — Bicubic Hermite on quad

Bicubic Hermite = tensor product of 1D Hermite. 16 DOFs per element: for
each of 4 corners, `(f, df/dxi1, df/dxi2, d²f/dxi1dxi2)`. Node order: 4
corners × 4 DOF kinds = 16 functions. Use the CMISS convention: the DOFs at
each node are listed in order `(value, ∂/∂xi1, ∂/∂xi2, ∂²/∂xi1∂xi2)`, and
nodes are ordered `(0,0) (1,0) (0,1) (1,1)`.

- [ ] **Step 1: Test**

```python
def test_hermite_quad_value_at_corners() -> None:
    b = get_basis("library.basis.bicubic_hermite.quad")
    xi = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    phi = b.shape_functions(xi)
    # The "value at corner i" DOF should be 1 at corner i, 0 elsewhere.
    value_columns = [0, 4, 8, 12]  # in our DOF layout
    for i, col in enumerate(value_columns):
        for j, corner in enumerate(xi):
            expected = 1.0 if i == j else 0.0
            assert abs(phi[j, col] - expected) < 1e-12
```

- [ ] **Step 2: Implement — append to `hermite.py`**

```python
@basis("library.basis.bicubic_hermite.quad", topology="quad", order=3)
class BicubicHermiteQuad:
    n_nodes = 16
    topology = "quad"
    order = 3

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        h_u = _hermite_1d(xi[..., 0])            # (..., 4)
        h_v = _hermite_1d(xi[..., 1])            # (..., 4)
        # DOF ordering at each node: (f, ∂/∂u, ∂/∂v, ∂²/∂u∂v)
        # At node (a, b), the 4 DOF basis functions are h[2a+f_u](u)·h[2b+f_v](v)
        # where f_u ∈ {0,1} selects value vs derivative, similarly f_v.
        # Layout corners in the order (0,0), (1,0), (0,1), (1,1).
        out = np.empty(xi.shape[:-1] + (16,), dtype=np.float64)
        corner_idx = 0
        for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            for (f_u, f_v) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                ui = 2 * a + f_u
                vi = 2 * b + f_v
                out[..., 4 * corner_idx + (f_u + 2 * f_v)] = h_u[..., ui] * h_v[..., vi]
            corner_idx += 1
        return out

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        h_u = _hermite_1d(xi[..., 0])
        h_v = _hermite_1d(xi[..., 1])
        dh_u = _hermite_1d_derivative(xi[..., 0])
        dh_v = _hermite_1d_derivative(xi[..., 1])
        out = np.empty(xi.shape[:-1] + (16, 2), dtype=np.float64)
        corner_idx = 0
        for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            for (f_u, f_v) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                ui = 2 * a + f_u
                vi = 2 * b + f_v
                out[..., 4 * corner_idx + (f_u + 2 * f_v), 0] = dh_u[..., ui] * h_v[..., vi]
                out[..., 4 * corner_idx + (f_u + 2 * f_v), 1] = h_u[..., ui] * dh_v[..., vi]
            corner_idx += 1
        return out
```

**Note:** this indexing is fiddly — verify with the Kronecker-delta tests
before continuing. If the value-at-corner test fails, inspect the `h[2a+f_u]`
indexing carefully.

- [ ] **Step 3: Run, pass.**

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/eval/bases/hermite.py tests/unit/test_bases_hermite.py
git commit -m "feat(eval): bicubic Hermite quad basis"
```

---

## Task 3 — Tricubic Hermite on hex

Tensor product of three 1D Hermite. 64 DOFs per element = 8 corners × 8 DOF
kinds (value, 3 first derivatives, 3 mixed second derivatives, one mixed
third). Implementation mirrors Task 2 scaled to 3D.

- [ ] **Step 1: Add Kronecker-delta test for the 8 "value at corner" DOFs.**

- [ ] **Step 2: Implement `TricubicHermiteHex`. Follow the same product pattern.**

Skeleton:

```python
@basis("library.basis.tricubic_hermite.hex", topology="hex", order=3)
class TricubicHermiteHex:
    n_nodes = 64
    topology = "hex"
    order = 3

    def shape_functions(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        h_u = _hermite_1d(xi[..., 0])
        h_v = _hermite_1d(xi[..., 1])
        h_w = _hermite_1d(xi[..., 2])
        out = np.empty(xi.shape[:-1] + (64,), dtype=np.float64)
        idx = 0
        for (a, b, c) in [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)]:
            for (f_u, f_v, f_w) in [
                (x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)
            ]:
                ui = 2 * a + f_u
                vi = 2 * b + f_v
                wi = 2 * c + f_w
                out[..., idx] = h_u[..., ui] * h_v[..., vi] * h_w[..., wi]
                idx += 1
        return out

    def shape_derivatives(self, xi: np.ndarray) -> np.ndarray:
        # Three partial derivatives; mirror shape_functions with one factor
        # replaced by its derivative. Code elided for brevity — follow the
        # bicubic template.
        raise NotImplementedError("flesh out following the quad template")
```

Note the `NotImplementedError` — do implement it. The skeleton exists to
show the loop structure.

- [ ] **Step 3: Run, pass** (Kronecker for values + one C¹ test at a random xi using SymPy-generated reference).

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/eval/bases/hermite.py tests/unit/test_bases_hermite.py
git commit -m "feat(eval): tricubic Hermite hex basis"
```

---

## Task 4 — Per-node scaling

CMISS / OpenCMISS convention: to maintain geometric continuity across
elements of different physical sizes, the derivative DOFs of Hermite bases
are multiplied by scale factors. In FieldML 0.5, scale factors are stored as
ParameterEvaluators indexed per-node per-derivative.

Simplification for Phase 3: support **per-node scalar scaling** — one scale
factor per node applied uniformly to all derivative DOFs at that node.
Derivative-pair-specific scaling (value, ∂/∂u, ∂/∂v, ∂²) arrives in a later
phase if users request it.

**Files:**
- Modify: `src/pyfieldml/eval/field.py` (apply scale factors when resolving Hermite fields)
- Create: `tests/unit/test_hermite_scaling.py`
- Create: `tests/fixtures/hermite_line.fieldml` (two-element 1D Hermite with scaling)

- [ ] **Step 1: Failing test `tests/unit/test_hermite_scaling.py`**

```python
"""Scale-factor + C¹-continuity tests on a two-element Hermite line."""

from __future__ import annotations

import numpy as np


def test_two_element_hermite_line_is_c1_at_shared_node() -> None:
    """Construct two 1D Hermite elements sharing node 2 with matched derivative DOFs.

    If per-node scaling is honored, the derivative at xi=1 in element 1 should
    match the derivative at xi=0 in element 2 when evaluated in the physical
    domain, modulo scale factors.
    """
    from pyfieldml.builders.mesh import add_hermite_mesh
    from pyfieldml.model.region import Region
    import pyfieldml as fml

    # Two-element line, total length 3 (element 1 is length 1, element 2 is length 2).
    r = Region(name="hermite_line")
    nodes = np.array([[0.0], [1.0], [3.0]], dtype=np.float64)  # physical positions
    connectivity = np.array([[1, 2], [2, 3]], dtype=np.int64)
    derivatives = np.array([[1.0], [1.0], [1.0]])  # df/dxi at each node, placeholder
    scales = np.array([[1.0], [1.0, 2.0], [2.0]])  # variable per-element scaling
    # The builder signature defined in Task 5: flesh out to accept scales.
    add_hermite_mesh(
        r, name="line_mesh", nodes=nodes, elements=connectivity,
        derivatives=derivatives, scales=scales,
        topology="line", coord_name="coordinates",
    )
    doc = fml.Document.from_region(r)
    coords = doc.field("coordinates")

    # Evaluate at the shared node from both elements.
    from_left = coords.evaluate(element=1, xi=1.0)
    from_right = coords.evaluate(element=2, xi=0.0)
    np.testing.assert_allclose(from_left, from_right, atol=1e-10)
```

- [ ] **Step 2: Extend `Field` to apply scales.**

In `src/pyfieldml/eval/field.py`, add an optional `scale_factors` parameter
and multiply the derivative DOFs before summing:

```python
class Field:
    def __init__(
        self,
        *,
        evaluator: Evaluator,
        region: Region,
        name: str,
        node_coords: np.ndarray,
        connectivity: np.ndarray,
        basis: Basis,
        scale_factors: np.ndarray | None = None,  # shape (n_elements, n_dofs)
    ) -> None:
        # ... existing init ...
        self._scale = scale_factors
```

In `evaluate`, before the einsum:

```python
if self._scale is not None:
    # scale[element, dof] applies elementwise; for Hermite dofs the scale
    # factor is 1.0 on value DOFs and the per-node scale on derivative DOFs.
    phi = phi * self._scale[element - 1]
```

(Same treatment in `jacobian` for the derivative DOFs via `d_phi`.)

- [ ] **Step 3: Add `add_hermite_mesh` to builders/mesh.py**

Skeleton (simplified single-derivative case):

```python
def add_hermite_mesh(
    region: Region,
    *,
    name: str,
    nodes: np.ndarray,                       # (n_nodes, D) physical positions
    elements: np.ndarray,                    # (n_elems, n_nodes_per_elem)
    derivatives: np.ndarray,                 # (n_nodes, D) first derivatives at each node
    scales: np.ndarray | None = None,        # (n_elems, n_dofs_per_elem)
    topology: str,
    coord_name: str = "coordinates",
) -> tuple[MeshType, ParameterEvaluator]:
    """Add a Hermite mesh with per-node derivative DOFs and optional scale factors.

    Phase-3 simplification: scalar first derivatives (no cross-derivatives).
    """
    ...  # follow add_lagrange_mesh template, expand DOFs
```

- [ ] **Step 4: Run, pass**

Run: `uv run pytest tests/unit/test_hermite_scaling.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/eval/field.py src/pyfieldml/builders/mesh.py tests/unit/test_hermite_scaling.py
git commit -m "feat(eval): per-node scaling for Hermite + add_hermite_mesh builder"
```

---

## Task 5 — DOM up-converter: FieldML 0.4 → 0.5

0.4 → 0.5 is the smaller of the two legacy transforms. Known deltas:
- Root element `@version="0.4"` must be rewritten to `"0.5"` (or left as the loaded version; write path always emits 0.5).
- A few element-name renames (consult `Fieldml.xsd` vs `FieldML_0.5.xsd` diff).
- Attribute-name renames on certain DataSource children.

- [ ] **Step 1: Create fixtures**

`tests/fixtures/legacy_0_4_minimal.fieldml` — minimal 0.4 content (Region + BooleanType, per the 0.4 XSD).

Simultaneously create an equivalent `tests/fixtures/reference_0_5_minimal.fieldml` (0.5 version of the same semantic content) so tests can compare.

- [ ] **Step 2: Test `tests/unit/test_dom_upconverter.py`**

```python
"""Up-converter tests."""

from __future__ import annotations

from lxml import etree

from pyfieldml.dom.parser import parse_file
from pyfieldml.dom.upconverter import upconvert_to_0_5


def test_upconvert_0_4_matches_hand_written_0_5(fixtures_dir) -> None:
    parsed = parse_file(fixtures_dir / "legacy_0_4_minimal.fieldml")
    upconvert_to_0_5(parsed.tree)
    assert parsed.tree.getroot().get("version") == "0.5.0"

    # C14N-compare to the hand-written 0.5 fixture.
    expected = parse_file(fixtures_dir / "reference_0_5_minimal.fieldml")
    a = etree.tostring(parsed.tree, method="c14n", exclusive=True)
    b = etree.tostring(expected.tree, method="c14n", exclusive=True)
    assert a == b
```

- [ ] **Step 3: Implement `src/pyfieldml/dom/upconverter.py`**

```python
"""Legacy FieldML → 0.5 up-converter. Operates on lxml ElementTrees in place."""

from __future__ import annotations

from lxml import etree

# Element renames from 0.4 → 0.5. Keys are 0.4 tags, values are 0.5 tags.
_0_4_ELEMENT_RENAMES: dict[str, str] = {
    # Fill in from diff of the two XSDs. Known examples:
    # "SemiDenseDataSource": "DataSource",
    # "TextFileResource":    "DataResourceHref",
}

_0_3_ELEMENT_RENAMES: dict[str, str] = {
    # 0.3 → 0.5 requires more transformation than renames alone.
    # Discover during implementation — leave a TODO per XSD diff item.
}


def upconvert_to_0_5(tree: etree._ElementTree) -> None:
    root = tree.getroot()
    version = root.get("version", "")
    if version.startswith("0.4"):
        _rename_elements(tree, _0_4_ELEMENT_RENAMES)
    elif version.startswith("0.3"):
        _rename_elements(tree, _0_3_ELEMENT_RENAMES)
        _rework_0_3_structures(tree)
    root.set("version", "0.5.0")


def _rename_elements(tree: etree._ElementTree, mapping: dict[str, str]) -> None:
    for old, new in mapping.items():
        for el in tree.iter(old):
            el.tag = new


def _rework_0_3_structures(tree: etree._ElementTree) -> None:
    """0.3-specific structural differences (Phase-3 implementation TODO).

    Called out by:
      - Different top-level <Fieldml> children (RegionBlock vs Region direct).
      - Different DataResource packaging.

    Consult FieldML-API/Fieldml_0.3.xsd vs FieldML_0.5.xsd for the complete
    delta. Implement element-by-element as each is exercised by a test fixture.
    Raise ``NotImplementedError`` for any 0.3 construct not yet handled.
    """
```

- [ ] **Step 4: Implement element renames.**

Populate `_0_4_ELEMENT_RENAMES` by diffing the two XSDs. The engineer should
open both files side-by-side. Prior knowledge to expect:
- `SemiDenseDataSource` → `DataSource` (0.4 → 0.5 data model change).
- `TextFileResource` / `HDF5FileResource` → single `DataResourceHref` with `format` attribute.

- [ ] **Step 5: Run, pass**

Run: `uv run pytest tests/unit/test_dom_upconverter.py::test_upconvert_0_4_matches_hand_written_0_5 -v`

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/dom/upconverter.py tests/unit/test_dom_upconverter.py tests/fixtures/legacy_0_4_*.fieldml tests/fixtures/reference_0_5_*.fieldml
git commit -m "feat(dom): FieldML 0.4 → 0.5 up-converter"
```

---

## Task 6 — DOM up-converter: FieldML 0.3 → 0.5

0.3 is more divergent. Expected transformations:
- `<ContinuousTypeEntries>` → `<Components>` on `<ContinuousType>`.
- Old `<VariableEvaluator>` semantic → `<ArgumentEvaluator>`.
- Old `<ContinuousImport>` semantic → `<Import>` + `<ImportType>` / `<ImportEvaluator>`.
- Old `<DataObject>` / `<ImportSource>` → `<DataResource>`.

Implement iteratively — do NOT try to be exhaustive. Add an explicit
`NotImplementedError("0.3: <XYZ> not yet handled")` branch for unrecognized
elements, and extend per fixture.

- [ ] **Step 1: Create `tests/fixtures/legacy_0_3_minimal.fieldml`** — simplest possible
0.3 document (single Region with one `ContinuousType` using 0.3 syntax — use
the bundled `library_0.3.xml` in the reference repo as a syntax reference).

- [ ] **Step 2: Add tests mirroring Task 5.**

- [ ] **Step 3: Flesh out `_rework_0_3_structures`.**

- [ ] **Step 4: Pass + commit.**

```bash
git add src/pyfieldml/dom/upconverter.py tests/ ...
git commit -m "feat(dom): FieldML 0.3 → 0.5 up-converter"
```

---

## Task 7 — Parser integration

- [ ] **Step 1: Modify `parse_file` to auto-upconvert.**

In `src/pyfieldml/dom/parser.py`:

```python
from pyfieldml.dom.upconverter import upconvert_to_0_5

def parse_file(path: PathLike) -> ParsedDocument:
    # ... existing parse logic ...
    version = _validate_root(tree, source_file=str(path))
    if version.startswith(("0.3", "0.4")):
        upconvert_to_0_5(tree)
        # Internal version stays recorded as the original for tracking;
        # the tree content is now 0.5-shaped.
    return ParsedDocument(tree=tree, version=version, source_file=str(path))
```

- [ ] **Step 2: End-to-end test `tests/unit/test_legacy_read.py`**

```python
"""End-to-end test: read a 0.3 or 0.4 document and access the model."""

from __future__ import annotations

import pyfieldml as fml


def test_legacy_0_4_reads_as_0_5_semantic_model(fixtures_dir) -> None:
    doc = fml.read(fixtures_dir / "legacy_0_4_minimal.fieldml")
    assert doc.source_version == "0.4.0"  # tracked original version
    assert "boolean" in doc.booleans  # up-converted successfully


def test_legacy_0_3_reads_as_0_5_semantic_model(fixtures_dir) -> None:
    doc = fml.read(fixtures_dir / "legacy_0_3_minimal.fieldml")
    assert doc.source_version.startswith("0.3")
    # Up-converter filled in whatever the fixture declared.
```

- [ ] **Step 3: Run, pass.**

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/dom/parser.py tests/unit/test_legacy_read.py
git commit -m "feat(dom): parser auto-upconverts 0.3/0.4 documents on read"
```

---

## Task 8 — Phase-3 closeout + v0.3.0

- [ ] **Step 1: Sanity gauntlet.**
- [ ] **Step 2: CHANGELOG entry:**

```markdown
## [0.3.0] - YYYY-MM-DD

### Added
- Cubic Hermite basis functions on line, quad (bicubic), and hex
  (tricubic), with per-node scale-factor support.
- `pyfieldml.builders.add_hermite_mesh`.
- FieldML 0.3 / 0.4 → 0.5 DOM up-converter; `pyfieldml.read` now accepts
  legacy documents and exposes them through the normal model API.
- `Document.source_version` reports the original document version.
```

- [ ] **Step 3: Version bump → 0.3.0; tag; push.**

## Phase-3 done-definition

- [ ] Cubic Hermite Kronecker + cubic-reproduction tests pass on line / quad / hex.
- [ ] Two-element Hermite line C¹-continuity test passes with per-node scaling.
- [ ] 0.4-minimal and 0.3-minimal fixtures both read and produce the correct semantic model.
- [ ] `v0.3.0` on PyPI.

Next: **Phase 4 — interop bridges** (meshio, PyVista, XDMF, scikit-fem, OpenSim-adjacent export, Jupyter graph explorer).
