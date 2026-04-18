# pyfieldml Phase 5 — Builders + Model Zoo + Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.
>
> **⚠ Forward-draft notice:** Depends on Phases 1–4.
>
> **⚠ Licensing blocker:** Datasets (Tasks 8–11) depend on provenance-cleared openly-licensed assets. Start these tasks by asking Francis's professor to nominate specific contributable models; fall back to open CT/published meshes only if clearance fails. Do not ship a dataset whose license is unverified.

**Goal:** Ship the full v1.0-track public surface: biomechanics-flavored high-level builders, a curated four-asset model zoo with on-demand download, semantic linting + diff tooling, complete CLI surface, all 8 tutorial notebooks, plus the scientific reproducibility notebook. Tag `v0.5.0` (the release-candidate for v1.0).

**Architecture:** `builders/biomech.py` layers biomechanics ergonomics over the Phase-2 `add_lagrange_mesh`. `datasets/` is a small registry backed by GitHub-releases artifacts with SHA256-verified download. Linter + diff are new sub-modules that walk the model graph. Notebooks live under `docs/notebooks/` and are CI-executed via `nbmake`.

**Tech Stack:** adds `jupyter`, `nbmake` (dev), `requests` (for dataset downloads — or use `urllib` to avoid a new dep).

**Spec reference:** §2.2 Pillar A (builders), §6 (notebooks + zoo).

**Completion criteria:**
- `doc.add_fiber_field`, `add_material_field`, `add_landmark_set`, `add_scalar_field`, `add_vector_field` all work end-to-end with builder-then-read-back assertions.
- `datasets.load_unit_cube()` works offline; the other three download and SHA256-verify on first call.
- `pyfieldml lint` catches orphan evaluators, unbound arguments, and inverted elements on a handcrafted bad fixture.
- `pyfieldml diff a.fieldml b.fieldml` reports type-level and parameter-value differences.
- All 8 notebooks execute in CI via `nbmake`.
- Reproducibility notebook reproduces a classical MSK FE result within published tolerance.
- `v0.5.0` on PyPI.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `src/pyfieldml/builders/biomech.py` | `add_fiber_field`, `add_material_field`, `add_landmark_set`, `add_scalar_field`, `add_vector_field` |
| `src/pyfieldml/datasets/__init__.py` | Public `load_*` functions + `list`, `info`, `download`, `cache_dir` |
| `src/pyfieldml/datasets/registry.py` | Dataset metadata registry (name, URL, SHA256, license, citation) |
| `src/pyfieldml/datasets/_cache.py` | Download + hash-verify + cache on disk |
| `src/pyfieldml/datasets/_bundled/` | Small datasets shipped inside the wheel (unit_cube) |
| `src/pyfieldml/validation/lint.py` | Semantic linter |
| `src/pyfieldml/validation/diff.py` | Semantic graph diff |
| `src/pyfieldml/cli/inspect.py` | `pyfieldml inspect` — prints a summary tree |
| `src/pyfieldml/cli/validate.py` | `pyfieldml validate` |
| `src/pyfieldml/cli/convert.py` | `pyfieldml convert` |
| `src/pyfieldml/cli/plot.py` | `pyfieldml plot` |
| `src/pyfieldml/cli/lint.py` | `pyfieldml lint` |
| `src/pyfieldml/cli/diff.py` | `pyfieldml diff` |
| `docs/notebooks/01_quickstart.ipynb` | 5-minute read + plot |
| `docs/notebooks/02_evaluator_graph.ipynb` | Browse via `doc.explore()` |
| `docs/notebooks/03_build_femur.ipynb` | Construct a femur model from scratch |
| `docs/notebooks/04_muscle_fibers.ipynb` | Fiber-field workflow |
| `docs/notebooks/05_cardiac_hermite.ipynb` | Hermite workflow |
| `docs/notebooks/06_scikit_fem_solve.ipynb` | FieldML → scikit-fem → solve → back |
| `docs/notebooks/07_opensim_bridge.ipynb` | Export to OpenSim-compatible assets |
| `docs/notebooks/08_legacy_conversion.ipynb` | 0.3/0.4 → 0.5 upgrade walkthrough |
| `paper/benchmarks/repro_keller_1994.ipynb` | Reproduce classical femoral-stiffness result |
| `tests/unit/test_builders_biomech.py` | Biomech builder tests |
| `tests/unit/test_datasets_registry.py` | Dataset loader tests |
| `tests/unit/test_validation_lint.py` | Linter tests |
| `tests/unit/test_validation_diff.py` | Semantic diff tests |
| `tests/unit/test_cli.py` | CLI end-to-end tests |

---

## Task 1 — Biomechanics builders

- [ ] **Step 1: Test `tests/unit/test_builders_biomech.py`**

```python
"""Tests for biomechanics high-level builders."""

from __future__ import annotations

import numpy as np

import pyfieldml as fml
from pyfieldml.builders.biomech import (
    add_fiber_field,
    add_landmark_set,
    add_material_field,
    add_scalar_field,
    add_vector_field,
)
from pyfieldml.builders.mesh import add_lagrange_mesh
from pyfieldml.model.region import Region


def _unit_cube_region() -> Region:
    r = Region(name="cube")
    nodes = np.array(
        [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float64
    )
    connectivity = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    mesh, _ = add_lagrange_mesh(
        r, name="m", nodes=nodes, elements=connectivity,
        topology="hex", order=1, coord_name="coords",
    )
    return r


def test_add_scalar_field() -> None:
    r = _unit_cube_region()
    add_scalar_field(r, name="temperature", mesh_name="m", values=np.arange(8.0))
    doc = fml.Document.from_region(r)
    np.testing.assert_array_equal(doc.evaluators["temperature"].as_ndarray(), np.arange(8.0))


def test_add_vector_field() -> None:
    r = _unit_cube_region()
    vals = np.random.default_rng(0).random((8, 3))
    add_vector_field(r, name="velocity", mesh_name="m", values=vals)
    doc = fml.Document.from_region(r)
    np.testing.assert_array_equal(doc.evaluators["velocity"].as_ndarray(), vals)


def test_add_fiber_field_shape() -> None:
    r = _unit_cube_region()
    fibers = np.zeros((8, 3))
    fibers[:, 0] = 1.0  # all fibers along x
    add_fiber_field(r, mesh_name="m", vectors=fibers)
    doc = fml.Document.from_region(r)
    got = doc.evaluators["fiber_direction"].as_ndarray()
    assert got.shape == (8, 3)
    np.testing.assert_array_equal(got, fibers)


def test_add_material_field_per_element() -> None:
    r = _unit_cube_region()
    add_material_field(r, mesh_name="m", values_per_element=np.array([1.5e9]))
    doc = fml.Document.from_region(r)
    assert "material" in doc.evaluators
    np.testing.assert_array_equal(doc.evaluators["material"].as_ndarray(), [1.5e9])


def test_add_landmark_set() -> None:
    r = _unit_cube_region()
    points = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]])
    labels = ["centroid", "corner_probe"]
    add_landmark_set(r, name="anatomical_points", points=points, labels=labels)
    doc = fml.Document.from_region(r)
    got = doc.evaluators["anatomical_points"].as_ndarray()
    np.testing.assert_array_equal(got, points)
```

- [ ] **Step 2: Implement `src/pyfieldml/builders/biomech.py`**

```python
"""Biomechanics-flavored high-level builders over add_lagrange_mesh."""

from __future__ import annotations

import numpy as np

from pyfieldml.data.text import InlineTextBackend
from pyfieldml.model.evaluators import ParameterEvaluator
from pyfieldml.model.region import Region
from pyfieldml.model.types import ContinuousType


def add_scalar_field(
    region: Region, *, name: str, mesh_name: str, values: np.ndarray,
) -> ParameterEvaluator:
    vt = ContinuousType(name=f"{name}.value_type")
    region.add_type(vt)
    data = InlineTextBackend.from_ndarray(values.astype(np.float64))
    ev = ParameterEvaluator(name=name, value_type=vt, data=data)
    region.add_evaluator(ev)
    return ev


def add_vector_field(
    region: Region, *, name: str, mesh_name: str, values: np.ndarray,
) -> ParameterEvaluator:
    d = values.shape[1]
    vt = ContinuousType(
        name=f"{name}.value_type", component_name=f"{name}.component", component_count=d
    )
    region.add_type(vt)
    data = InlineTextBackend.from_ndarray(values.astype(np.float64))
    ev = ParameterEvaluator(name=name, value_type=vt, data=data)
    region.add_evaluator(ev)
    return ev


def add_fiber_field(
    region: Region, *, mesh_name: str, vectors: np.ndarray, name: str = "fiber_direction",
) -> ParameterEvaluator:
    """Vector field conventionally representing muscle/myocyte fiber directions."""
    return add_vector_field(region, name=name, mesh_name=mesh_name, values=vectors)


def add_material_field(
    region: Region, *, mesh_name: str, values_per_element: np.ndarray, name: str = "material",
) -> ParameterEvaluator:
    """Per-element scalar (e.g. Young's modulus from BMD, per-element anisotropy weight)."""
    return add_scalar_field(
        region, name=name, mesh_name=mesh_name, values=values_per_element
    )


def add_landmark_set(
    region: Region, *, name: str, points: np.ndarray, labels: list[str] | None = None,
) -> ParameterEvaluator:
    """A named set of anatomical landmarks (Nx3 points + optional labels)."""
    d = points.shape[1]
    vt = ContinuousType(
        name=f"{name}.value_type",
        component_name=f"{name}.component",
        component_count=d,
    )
    region.add_type(vt)
    data = InlineTextBackend.from_ndarray(points.astype(np.float64))
    ev = ParameterEvaluator(name=name, value_type=vt, data=data)
    region.add_evaluator(ev)
    # Labels are Phase-5 metadata — stored on the Region as an auxiliary dict.
    # Full serialization to XML arrives later.
    if labels is not None:
        if not hasattr(region, "_landmark_labels"):
            region._landmark_labels = {}
        region._landmark_labels[name] = list(labels)
    return ev
```

- [ ] **Step 3: Run, pass + commit**

```bash
git add src/pyfieldml/builders/biomech.py tests/unit/test_builders_biomech.py
git commit -m "feat(builders): biomech helpers (scalar/vector/fiber/material/landmarks)"
```

---

## Task 2 — Datasets: registry + cache + `load_unit_cube`

**Files:**
- Create: `src/pyfieldml/datasets/__init__.py`
- Create: `src/pyfieldml/datasets/registry.py`
- Create: `src/pyfieldml/datasets/_cache.py`
- Create: `src/pyfieldml/datasets/_bundled/unit_cube.fieldml` (authored; ships in the wheel)
- Test: `tests/unit/test_datasets_registry.py`

- [ ] **Step 1: Test**

```python
"""Dataset registry tests."""

from __future__ import annotations

import pyfieldml as fml
from pyfieldml import datasets


def test_list_contains_unit_cube() -> None:
    assert "unit_cube" in datasets.list()


def test_info_returns_metadata() -> None:
    info = datasets.info("unit_cube")
    assert info["license"]
    assert info["citation"]


def test_load_unit_cube_offline() -> None:
    doc = datasets.load_unit_cube()
    assert isinstance(doc, fml.Document)
    assert "coordinates" in doc.evaluators
```

- [ ] **Step 2: Author a minimal `unit_cube.fieldml`**

Create `src/pyfieldml/datasets/_bundled/unit_cube.fieldml` programmatically:

```bash
uv run python -c "
import numpy as np, pyfieldml as fml
from pyfieldml.builders.mesh import add_lagrange_mesh
from pyfieldml.model.region import Region

nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float64)
conn = np.array([[1,2,3,4,5,6,7,8]], dtype=np.int64)
r = Region(name='unit_cube')
add_lagrange_mesh(r, name='mesh', nodes=nodes, elements=conn, topology='hex', order=1, coord_name='coordinates')
fml.Document.from_region(r).write('src/pyfieldml/datasets/_bundled/unit_cube.fieldml')
"
```

- [ ] **Step 3: Implement `src/pyfieldml/datasets/registry.py`**

```python
"""Dataset metadata registry. Ship entries here as new datasets land."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetInfo:
    name: str
    files: dict[str, str] = field(default_factory=dict)   # filename → sha256 hex
    url_prefix: str | None = None                          # GH release asset base URL
    license: str = ""                                      # SPDX tag
    citation: str = ""                                     # BibTeX
    origin: str = ""                                       # upstream source URL
    bundled: bool = False                                  # true = ships in the wheel


REGISTRY: dict[str, DatasetInfo] = {
    "unit_cube": DatasetInfo(
        name="unit_cube",
        files={"unit_cube.fieldml": ""},   # bundled — hash not required
        license="CC0-1.0",
        citation="Authored for pyfieldml, CC0 dedication.",
        origin="https://github.com/kchemorion/pyfieldml",
        bundled=True,
    ),
    # "femur": DatasetInfo(...)     # populated in Task 3
    # "rectus_femoris": DatasetInfo(...)
    # "myocardium": DatasetInfo(...)
}
```

- [ ] **Step 4: Implement `src/pyfieldml/datasets/_cache.py`**

```python
"""Download + hash-verify + cache on-disk."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from pyfieldml.datasets.registry import REGISTRY, DatasetInfo


def cache_dir() -> Path:
    """Root directory under which dataset files are stored."""
    base = Path(
        os.environ.get("PYFIELDML_CACHE")
        or Path.home() / ".cache" / "pyfieldml"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def dataset_dir(name: str) -> Path:
    d = cache_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_files(name: str) -> Path:
    info = REGISTRY[name]
    if info.bundled:
        from importlib import resources

        ref = resources.files("pyfieldml.datasets._bundled").joinpath(
            f"{name}.fieldml"
        )
        with resources.as_file(ref) as path:
            return Path(path)

    d = dataset_dir(name)
    for filename, expected_sha in info.files.items():
        target = d / filename
        if not target.is_file():
            url = f"{info.url_prefix.rstrip('/')}/{filename}"
            urllib.request.urlretrieve(url, target)
        _verify_sha256(target, expected_sha)
    # Return the primary .fieldml file (first one alphabetically).
    return next(iter(sorted(d.glob("*.fieldml"))))


def _verify_sha256(path: Path, expected: str) -> None:
    if not expected:
        return
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        path.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected}, got {h.hexdigest()}"
        )
```

- [ ] **Step 5: Implement `src/pyfieldml/datasets/__init__.py`**

```python
"""Model zoo — curated FieldML assets for tutorials and research."""

from __future__ import annotations

import pyfieldml as _fml
from pyfieldml.datasets._cache import cache_dir, dataset_dir, ensure_files
from pyfieldml.datasets.registry import REGISTRY


def list() -> list[str]:  # noqa: A001
    return sorted(REGISTRY.keys())


def info(name: str) -> dict[str, str]:
    d = REGISTRY[name]
    return {
        "name": d.name,
        "license": d.license,
        "citation": d.citation,
        "origin": d.origin,
    }


def download(name: str) -> None:
    ensure_files(name)


def load(name: str):  # noqa: ANN201
    path = ensure_files(name)
    return _fml.read(path)


def load_unit_cube() -> "_fml.Document":
    return load("unit_cube")


def load_femur() -> "_fml.Document":
    return load("femur")


def load_rectus_femoris() -> "_fml.Document":
    return load("rectus_femoris")


def load_myocardium() -> "_fml.Document":
    return load("myocardium")


__all__ = [
    "cache_dir", "dataset_dir", "download", "info", "list", "load",
    "load_femur", "load_myocardium", "load_rectus_femoris", "load_unit_cube",
]
```

- [ ] **Step 6: Run, pass + commit**

```bash
git add src/pyfieldml/datasets/ tests/unit/test_datasets_registry.py
git commit -m "feat(datasets): registry + download/cache + bundled unit_cube"
```

---

## Task 3 — Datasets: femur, rectus_femoris, myocardium (provenance-gated)

**⚠ Licensing blocker.** Start by resolving provenance before writing code.

- [ ] **Step 1: Obtain provenance-cleared assets**

Ask Francis's professor to nominate contributable assets. For each asset,
record in `registry.py`:

```python
"femur": DatasetInfo(
    name="femur",
    files={
        "femur.fieldml": "<SHA256>",
        "femur.h5":      "<SHA256>",
    },
    url_prefix="https://github.com/kchemorion/pyfieldml/releases/download/dataset-assets-v1",
    license="CC-BY-4.0",
    citation="<BibTeX>",
    origin="<URL>",
),
```

Upload the asset files to a GitHub Release named `dataset-assets-v1` in the
`kchemorion/pyfieldml` repo. This keeps large files out of the source tree.

- [ ] **Step 2: Repeat for rectus_femoris and myocardium.**

- [ ] **Step 3: Author-fallback path.**

If provenance stalls, author replacements:
- **femur fallback:** Use an openly-licensed CT (e.g. SCAPOCO dataset, VHP) →
  segment bone → voxelize to tet mesh via PyVista → write FieldML via the
  Phase-2 builder. Document the full pipeline in
  `docs/datasets/AUTHORING.md`.
- **rectus_femoris fallback:** Similar — use an openly-licensed segmented
  muscle mesh. Fiber directions can be synthesized from a simple "along
  principal axis" assumption for demonstration purposes; document honestly.
- **myocardium fallback:** hand-build a small cubic Hermite mesh that
  demonstrates Hermite evaluation, even if it's not a real heart. Clearly
  label as "synthetic myocardium demo" in the docs.

- [ ] **Step 4: Test — each dataset loads and its key structure is correct.**

```python
def test_femur_dataset_loads(monkeypatch, tmp_path) -> None:
    """Smoke test — actual download is skipped in CI by pointing at a local fixture."""
    monkeypatch.setenv("PYFIELDML_CACHE", str(tmp_path))
    # If the download tests aren't online-capable, mark xfail/skip.
```

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/datasets/registry.py docs/datasets/
git commit -m "feat(datasets): add femur, rectus_femoris, myocardium (provenance cleared)"
```

---

## Task 4 — Semantic linter

**Files:**
- Create: `src/pyfieldml/validation/__init__.py`
- Create: `src/pyfieldml/validation/lint.py`
- Create: `tests/fixtures/lint_bad_orphan.fieldml` — a doc with an orphan evaluator
- Create: `tests/fixtures/lint_bad_inverted_tet.fieldml` — a tet mesh with a negative Jacobian
- Test: `tests/unit/test_validation_lint.py`

- [ ] **Step 1: Test**

```python
def test_lint_flags_orphan_evaluator(fixtures_dir):
    from pyfieldml.validation.lint import lint_document
    import pyfieldml as fml

    doc = fml.read(fixtures_dir / "lint_bad_orphan.fieldml")
    issues = lint_document(doc)
    assert any("orphan" in i.code.lower() for i in issues)


def test_lint_flags_inverted_tet(fixtures_dir):
    from pyfieldml.validation.lint import lint_document
    import pyfieldml as fml

    doc = fml.read(fixtures_dir / "lint_bad_inverted_tet.fieldml")
    issues = lint_document(doc)
    assert any("inverted" in i.code.lower() or "negative jacobian" in i.message.lower() for i in issues)
```

- [ ] **Step 2: Implement**

```python
"""Semantic linter over a loaded Document.

Reports issues that the XSD cannot catch: orphan evaluators (defined but
never referenced by anything that could evaluate them), unbound arguments,
inverted elements (negative Jacobian at the centroid), etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LintIssue:
    code: str
    severity: str      # "error" | "warning" | "info"
    message: str
    object_name: str | None = None


def lint_document(doc: Any) -> list[LintIssue]:
    issues: list[LintIssue] = []
    issues.extend(_check_orphans(doc))
    issues.extend(_check_inverted_elements(doc))
    return issues


def _check_orphans(doc: Any) -> list[LintIssue]:
    from pyfieldml.model.evaluators import ReferenceEvaluator

    # An evaluator is "orphan" if no ReferenceEvaluator points at it and it
    # is not a ParameterEvaluator that stores data (terminal).
    region = doc.region
    referenced = set()
    for name, ev in region.evaluators.items():
        if isinstance(ev, ReferenceEvaluator):
            referenced.add(ev.source.name)

    issues: list[LintIssue] = []
    terminal_kinds = {"ParameterEvaluator", "ConstantEvaluator", "ArgumentEvaluator", "ExternalEvaluator"}
    for name, ev in region.evaluators.items():
        if type(ev).__name__ in terminal_kinds:
            continue
        if name not in referenced:
            issues.append(
                LintIssue(
                    code="ORPHAN",
                    severity="warning",
                    message=f"Evaluator {name!r} is defined but never referenced",
                    object_name=name,
                )
            )
    return issues


def _check_inverted_elements(doc: Any) -> list[LintIssue]:
    from pyfieldml.eval.field import Field

    issues: list[LintIssue] = []
    for name, ev in doc.region.evaluators.items():
        try:
            f = doc.field(name)
        except Exception:
            continue
        if not isinstance(f, Field):
            continue
        # Sample Jacobian at centroid of each element. Negative det → inverted.
        centroid_xi = {
            "line": [0.5], "triangle": [1/3, 1/3], "quad": [0.5, 0.5],
            "tet": [1/4, 1/4, 1/4], "hex": [0.5, 0.5, 0.5], "wedge": [1/3, 1/3, 0.5],
        }[f._basis.topology]
        for e in range(1, f._conn.shape[0] + 1):
            J = f.jacobian(element=e, xi=centroid_xi)
            if J.shape[0] == J.shape[1]:
                d = float(np.linalg.det(J))
                if d <= 0.0:
                    issues.append(
                        LintIssue(
                            code="INVERTED_ELEMENT",
                            severity="error",
                            message=f"Element {e} of {name!r}: negative Jacobian det={d:.3g}",
                            object_name=name,
                        )
                    )
    return issues
```

- [ ] **Step 3: Run, pass + commit**

```bash
git add src/pyfieldml/validation/lint.py tests/
git commit -m "feat(validation): semantic linter (orphans + inverted elements)"
```

---

## Task 5 — Semantic diff

**Files:**
- Create: `src/pyfieldml/validation/diff.py`
- Test: `tests/unit/test_validation_diff.py`

- [ ] **Step 1: Test**

```python
def test_diff_reports_type_and_parameter_differences(tmp_path):
    import numpy as np
    import pyfieldml as fml
    from pyfieldml.builders.mesh import add_lagrange_mesh
    from pyfieldml.model.region import Region
    from pyfieldml.validation.diff import diff_documents

    def build(val):
        r = Region(name="t")
        nodes = np.array(
            [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float64
        )
        conn = np.array([[1,2,3,4,5,6,7,8]], dtype=np.int64)
        add_lagrange_mesh(r, name="m", nodes=nodes, elements=conn, topology="hex", order=1, coord_name="c")
        r.evaluators["c"].data = fml.data.InlineTextBackend.from_ndarray(nodes + val)
        return fml.Document.from_region(r)

    d = diff_documents(build(0.0), build(0.5))
    assert d.has_differences()
    assert any("parameter" in e.kind.lower() for e in d.entries)
```

- [ ] **Step 2: Implement**

```python
"""Semantic evaluator-graph diff between two FieldML documents."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DiffEntry:
    kind: str       # "missing_type" | "parameter_values_differ" | "evaluator_kind_changed" ...
    message: str
    name: str | None = None


@dataclass
class Diff:
    entries: list[DiffEntry] = field(default_factory=list)

    def has_differences(self) -> bool:
        return bool(self.entries)


def diff_documents(a, b) -> Diff:  # noqa: ANN001
    out = Diff()
    _diff_type_sets(a, b, "booleans", out)
    _diff_type_sets(a, b, "ensembles", out)
    _diff_type_sets(a, b, "continuous", out)
    _diff_type_sets(a, b, "meshes", out)
    _diff_evaluators(a, b, out)
    _diff_parameter_values(a, b, out)
    return out


def _diff_type_sets(a, b, kind: str, out: Diff) -> None:
    sa = set(getattr(a, kind).keys())
    sb = set(getattr(b, kind).keys())
    for name in sa - sb:
        out.entries.append(DiffEntry(kind=f"missing_{kind}", message=f"In A but not B: {name!r}", name=name))
    for name in sb - sa:
        out.entries.append(DiffEntry(kind=f"missing_{kind}", message=f"In B but not A: {name!r}", name=name))


def _diff_evaluators(a, b, out: Diff) -> None:
    sa = set(a.evaluators.keys())
    sb = set(b.evaluators.keys())
    for name in sa & sb:
        ka, kb = type(a.evaluators[name]).__name__, type(b.evaluators[name]).__name__
        if ka != kb:
            out.entries.append(
                DiffEntry(
                    kind="evaluator_kind_changed",
                    message=f"{name}: {ka} → {kb}",
                    name=name,
                )
            )
    for name in sa - sb:
        out.entries.append(DiffEntry(kind="missing_evaluator", message=f"In A only: {name}", name=name))
    for name in sb - sa:
        out.entries.append(DiffEntry(kind="missing_evaluator", message=f"In B only: {name}", name=name))


def _diff_parameter_values(a, b, out: Diff) -> None:
    from pyfieldml.model.evaluators import ParameterEvaluator

    for name in set(a.evaluators) & set(b.evaluators):
        ea, eb = a.evaluators[name], b.evaluators[name]
        if isinstance(ea, ParameterEvaluator) and isinstance(eb, ParameterEvaluator):
            arr_a, arr_b = ea.as_ndarray(), eb.as_ndarray()
            if arr_a.shape != arr_b.shape or not np.array_equal(arr_a, arr_b):
                out.entries.append(
                    DiffEntry(
                        kind="parameter_values_differ",
                        message=f"Parameter {name!r}: values differ",
                        name=name,
                    )
                )
```

- [ ] **Step 3: Run + commit**

```bash
git add src/pyfieldml/validation/diff.py tests/unit/test_validation_diff.py
git commit -m "feat(validation): semantic Document diff"
```

---

## Task 6 — CLI: inspect / validate / convert / plot / lint / diff

**Files:**
- Create / Modify: `src/pyfieldml/cli/main.py`, plus one file per subcommand.

- [ ] **Step 1: Flesh out the argparse dispatcher in `src/pyfieldml/cli/main.py`**

Add subparsers:

```python
from pyfieldml.cli import bench as bench_cmd, convert as convert_cmd, diff as diff_cmd, \
    inspect as inspect_cmd, lint as lint_cmd, plot as plot_cmd, validate as validate_cmd

def main(argv=None):
    p = argparse.ArgumentParser(prog="pyfieldml")
    sub = p.add_subparsers(dest="cmd", required=True)

    # inspect
    s = sub.add_parser("inspect"); s.add_argument("path")
    # validate
    s = sub.add_parser("validate"); s.add_argument("path"); s.add_argument("--strict", action="store_true")
    # convert
    s = sub.add_parser("convert"); s.add_argument("path"); s.add_argument("--to", required=True); s.add_argument("-o", required=True)
    # plot
    s = sub.add_parser("plot"); s.add_argument("path"); s.add_argument("--field")
    # lint
    s = sub.add_parser("lint"); s.add_argument("path")
    # diff
    s = sub.add_parser("diff"); s.add_argument("a"); s.add_argument("b")
    # bench
    s = sub.add_parser("bench"); s.add_argument("path"); s.add_argument("--field", required=True); s.add_argument("--n", type=int, default=10_000)

    args = p.parse_args(argv)
    dispatch = {
        "inspect":  lambda: inspect_cmd.run(path=args.path),
        "validate": lambda: validate_cmd.run(path=args.path, strict=args.strict),
        "convert":  lambda: convert_cmd.run(path=args.path, to_format=args.to, out=args.o),
        "plot":     lambda: plot_cmd.run(path=args.path, field=args.field),
        "lint":     lambda: lint_cmd.run(path=args.path),
        "diff":     lambda: diff_cmd.run(a=args.a, b=args.b),
        "bench":    lambda: bench_cmd.run(path=args.path, field=args.field, n=args.n),
    }
    return dispatch[args.cmd]()
```

- [ ] **Step 2: Implement each subcommand**

Each subcommand is ~20 lines. Example `src/pyfieldml/cli/inspect.py`:

```python
"""`pyfieldml inspect` — print a summary tree."""

from __future__ import annotations

import pyfieldml as fml


def run(*, path: str) -> int:
    doc = fml.read(path)
    print(f"FieldML document: {path}")
    print(f"  Source version: {doc.source_version}")
    print(f"  Region:         {doc.region.name}")
    print(f"  Types:")
    for name, t in {**doc.booleans, **doc.ensembles, **doc.continuous, **doc.meshes}.items():
        print(f"    - {name}: {type(t).__name__}")
    print(f"  Evaluators ({len(doc.evaluators)}):")
    for name, ev in doc.evaluators.items():
        print(f"    - {name}: {type(ev).__name__}")
    return 0
```

`convert.py` uses `meshio` interop. `plot.py` calls `doc.plot()`. `lint.py`
prints lint issues. `diff.py` prints diff entries.

- [ ] **Step 3: End-to-end CLI test**

```python
"""End-to-end CLI smoke tests using subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_inspect_prints_document_summary(fixtures_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pyfieldml.cli.main", "inspect",
         str(fixtures_dir / "minimal.fieldml")],
        capture_output=True, text=True, check=True,
    )
    assert "Region:" in result.stdout


def test_validate_exits_zero_on_valid_doc(fixtures_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pyfieldml.cli.main", "validate",
         str(fixtures_dir / "minimal.fieldml")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
```

Also add `src/pyfieldml/cli/__main__.py`:

```python
from pyfieldml.cli.main import main
import sys
sys.exit(main())
```

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/cli/ tests/unit/test_cli.py
git commit -m "feat(cli): inspect, validate, convert, plot, lint, diff subcommands"
```

---

## Task 7 — Notebooks: 01 quickstart + 02 evaluator graph + 03 build femur

Three notebooks at a time to keep progress visible.

- [ ] **Step 1: Write `docs/notebooks/01_quickstart.ipynb`**

Structure:
1. Install (code cell: `!pip install pyfieldml`)
2. Read a dataset (`doc = datasets.load_femur()`)
3. Inspect `doc.source_version`, `doc.region.name`, list evaluators.
4. Extract coordinates as NumPy, print shape.
5. Plot via `doc.plot()`.
6. Write to disk and re-read (round-trip smoke).

Runtime ≤2 minutes.

- [ ] **Step 2: Write `docs/notebooks/02_evaluator_graph.ipynb`**

Structure:
1. Load `femur`.
2. Explain types vs evaluators.
3. Walk through each evaluator subtype with `type()` introspection.
4. Open `doc.explore()` widget — screenshot in a static cell (since headless CI won't render ipywidgets).
5. Probe a point: `field.evaluate(element=1, xi=(0.3, 0.3, 0.3))`.

- [ ] **Step 3: Write `docs/notebooks/03_build_femur.ipynb`**

Structure:
1. Load node coords + connectivity from a CSV or NumPy file (small subset of the femur dataset, shipped alongside the notebook).
2. `add_lagrange_mesh(...)`.
3. `add_material_field(...)` with a synthetic BMD → E mapping.
4. `add_landmark_set(...)` for greater trochanter, femoral head, etc.
5. Validate + write.
6. Re-read and plot.

- [ ] **Step 4: Wire `nbmake` into CI**

Add to `.github/workflows/ci.yml` (extend the test job):

```yaml
      - run: uv run pytest --nbmake docs/notebooks -v
        if: matrix.os == 'ubuntu-latest' && matrix.python == '3.12'
```

- [ ] **Step 5: Commit**

```bash
git add docs/notebooks/01_quickstart.ipynb docs/notebooks/02_evaluator_graph.ipynb docs/notebooks/03_build_femur.ipynb .github/workflows/ci.yml
git commit -m "docs(notebooks): 01 quickstart + 02 eval graph + 03 build femur + nbmake CI"
```

---

## Task 8 — Notebooks: 04 muscle fibers + 05 cardiac Hermite

- [ ] **Step 1: Write `docs/notebooks/04_muscle_fibers.ipynb`**

1. Load `rectus_femoris`.
2. Visualize fibers with `field.plot()` arrows.
3. Along-fiber vs cross-fiber strain demo: at each element centroid, compute strain tensor's eigenvector closest to the fiber direction.
4. Export fiber streamlines to `.vtp` via OpenSim-adjacent export.

- [ ] **Step 2: Write `docs/notebooks/05_cardiac_hermite.ipynb`**

1. Load `myocardium`.
2. Show Hermite DOFs (value + derivatives) at one node.
3. Evaluate geometry at a grid of xi — reconstructed surface.
4. Compare reconstructed surface with the same nodes reinterpreted as linear Lagrange — visualize the fidelity gap.

- [ ] **Step 3: Commit**

```bash
git add docs/notebooks/04_muscle_fibers.ipynb docs/notebooks/05_cardiac_hermite.ipynb
git commit -m "docs(notebooks): 04 muscle fibers + 05 cardiac Hermite"
```

---

## Task 9 — Notebooks: 06 scikit-fem solve + 07 OpenSim bridge + 08 legacy conversion

- [ ] **Step 1: `docs/notebooks/06_scikit_fem_solve.ipynb`**

1. Load `femur`.
2. `to_scikit_fem(doc)`.
3. Assemble linear elasticity on the converted basis.
4. Solve with scikit-fem.
5. Write solution field back into a new FieldML document via `add_vector_field`.

- [ ] **Step 2: `docs/notebooks/07_opensim_bridge.ipynb`**

1. Load `rectus_femoris`.
2. `osim_bridge.export_geometry(...)`.
3. `osim_bridge.export_muscle_attachments(...)`.
4. `osim_bridge.export_fiber_lines(...)`.
5. Instructions in markdown: "drop these files into an OpenSim model at path X".

- [ ] **Step 3: `docs/notebooks/08_legacy_conversion.ipynb`**

1. Load a 0.3 or 0.4 asset (ship a small legacy fixture in `docs/notebooks/assets/`).
2. Show `doc.source_version == "0.3.0"` but `doc.evaluators` works.
3. Write out as 0.5, read back, `diff_documents` to confirm semantic equivalence.

- [ ] **Step 4: Commit**

```bash
git add docs/notebooks/06_scikit_fem_solve.ipynb docs/notebooks/07_opensim_bridge.ipynb docs/notebooks/08_legacy_conversion.ipynb
git commit -m "docs(notebooks): 06 scikit-fem solve + 07 OpenSim bridge + 08 legacy conversion"
```

---

## Task 10 — Reproducibility notebook

**Files:**
- Create: `paper/benchmarks/repro_keller_1994.ipynb`
- Create: `paper/benchmarks/README.md`

- [ ] **Step 1: Pick the target paper + dataset**

Candidate: *Keller TS (1994). "Predicting the compressive mechanical behavior
of bone." J. Biomech. 27(9):1159-68.* Reports femur stiffness as a function
of BMD; the mesh and boundary conditions are public. If the mesh isn't
publicly reproducible, swap to another classical paper whose full inputs
are open (there are several such femur benchmarks).

- [ ] **Step 2: Write the notebook**

Structure:
1. Problem statement + citation.
2. Load `femur` dataset.
3. Set up boundary conditions per the paper.
4. Solve via `to_scikit_fem` + linear elasticity.
5. Extract stiffness (force / displacement).
6. Compare to the paper's reported value; tabulate agreement (% error).
7. Conclude.

This notebook is the JOSS paper figure.

- [ ] **Step 3: Document in `paper/benchmarks/README.md`**

How to re-run, provenance of the target paper + dataset, any departures from the paper's setup.

- [ ] **Step 4: Commit**

```bash
git add paper/benchmarks/
git commit -m "docs(paper): reproducibility notebook — classical femoral stiffness benchmark"
```

---

## Task 11 — Phase-5 closeout + v0.5.0

- [ ] **Step 1: Full sanity gauntlet + all 9 notebooks run under nbmake.**
- [ ] **Step 2: CHANGELOG entry for 0.5.0.**
- [ ] **Step 3: Version bump → 0.5.0, tag, push.**

## Phase-5 done-definition

- [ ] All five biomech builders have unit tests passing.
- [ ] All four datasets load (bundled unit_cube + three downloaded).
- [ ] Linter flags orphans and inverted elements.
- [ ] Semantic diff reports parameter-value differences.
- [ ] All 6 CLI subcommands pass end-to-end tests (inspect, validate, convert, plot, lint, diff).
- [ ] 8 tutorial notebooks + 1 reproducibility notebook all execute green in CI.
- [ ] `v0.5.0` on PyPI.

Next: **Phase 6 — docs + v1.0 release** (jupyter-lite embedding, conformance-suite artifact, polish, tag v1.0.0, announce). Phase 7 (JOSS paper) runs in parallel.
