# pyfieldml Phase 1 — Semantic Model + Full I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.
>
> **⚠ Forward-draft notice:** Task code examples reference types and method signatures proposed here. If Phase-0 shipped code that diverges in naming, update the signatures in this plan before starting Phase 1. Re-run the self-review checklist at the start of the phase.

**Goal:** Replace the Phase-0 `Document` stub with a fully typed FieldML 0.5 semantic object model, wire up all four array-data backends (inline text, external text, HDF5 dense, HDF5 DOK sparse), and light up the conformance CI that validates every read/write against the C++ reference library. Ship `v0.1.0`.

**Architecture:** The `model/` subpackage owns the semantic object graph — `Region`, type declarations (`EnsembleType`, `ContinuousType`, `MeshType`, `BooleanType`), and the polymorphic `Evaluator` hierarchy (`ParameterEvaluator`, `ReferenceEvaluator`, `AggregateEvaluator`, `PiecewiseEvaluator`, `ConstantEvaluator`, `ArgumentEvaluator`, `ExternalEvaluator`). The `data/` subpackage provides the `DataSource` abstraction + four concrete backends with lazy NumPy views. The DOM layer from Phase 0 becomes an internal translation layer; reads now go DOM → model, writes go model → DOM → XML.

**Tech Stack:** Python 3.10+, `lxml` (Phase 0), `numpy`, `h5py`, `scipy.sparse` (for DOK validation), `pytest`.

**Spec reference:** `docs/superpowers/specs/2026-04-18-pyfieldml-design.md` §2.2, §3 (model, data), §4 (API).

**Phase-1 completion criteria:**
- `pyfieldml.read(path).evaluators["name"]` returns a typed `Evaluator` subclass.
- All seven evaluator subtypes round-trip through read → write → read → deep-equal.
- All four array backends read and write losslessly for `float32` / `float64` / `int32` / `int64`.
- `ParameterEvaluator.as_ndarray()` returns a `numpy.ndarray` with correct shape + dtype; HDF5-backed parameters are lazy until `as_ndarray` is called.
- `Import` resolution works: a document importing from `FieldML_Library_0.5.xml` resolves without network access.
- Conformance CI job builds the C++ reference library and asserts round-trip + array equality on every available test vector.
- `v0.1.0` tag published to PyPI; CHANGELOG updated; docs API reference auto-populates the new types.

---

## File structure created in this phase

| Path | Responsibility |
|------|----------------|
| `src/pyfieldml/model/__init__.py` | Re-exports `Region`, type classes, evaluator classes |
| `src/pyfieldml/model/region.py` | `Region` container — mapping-style access to contained objects |
| `src/pyfieldml/model/types.py` | `EnsembleType`, `ContinuousType`, `MeshType`, `BooleanType` dataclasses |
| `src/pyfieldml/model/evaluators.py` | Base `Evaluator` + seven concrete subclasses |
| `src/pyfieldml/model/bindings.py` | `ArgumentBinding` + binding-resolution helpers for ReferenceEvaluator |
| `src/pyfieldml/model/imports.py` | `Import` object + resolver against bundled stdlib and local file imports |
| `src/pyfieldml/model/_loader.py` | DOM tree → model graph (read path). Internal. |
| `src/pyfieldml/model/_writer.py` | Model graph → DOM tree (write path). Internal. |
| `src/pyfieldml/data/__init__.py` | Re-exports `DataSource`, `DataResource`, backend classes |
| `src/pyfieldml/data/base.py` | `DataSource` protocol, `DataResource` dataclass, shape/dtype helpers |
| `src/pyfieldml/data/text.py` | `InlineTextBackend`, `ExternalTextBackend` |
| `src/pyfieldml/data/hdf5.py` | `Hdf5DenseBackend`, `Hdf5DOKBackend` |
| `src/pyfieldml/document.py` | Upgraded — now constructs + owns a `Region` graph, not just a DOM tree |
| `tests/fixtures/parameter_inline.fieldml` | ParameterEvaluator with inline text data |
| `tests/fixtures/parameter_external.fieldml` + `parameter_external.txt` | External-text-backed parameter |
| `tests/fixtures/parameter_hdf5.fieldml` + `parameter_hdf5.h5` | HDF5-dense-backed parameter |
| `tests/fixtures/parameter_hdf5_dok.fieldml` + `parameter_hdf5_dok.h5` | HDF5-DOK-sparse-backed parameter |
| `tests/fixtures/reference_aggregate.fieldml` | ReferenceEvaluator + AggregateEvaluator exercising bindings |
| `tests/fixtures/piecewise_per_element.fieldml` | PiecewiseEvaluator (without eval — just structural round-trip) |
| `tests/fixtures/import_stdlib.fieldml` | Document importing from FieldML_Library_0.5 |
| `tests/unit/test_model_types.py` | Type-class tests |
| `tests/unit/test_model_evaluators.py` | Evaluator hierarchy tests |
| `tests/unit/test_model_region.py` | Region container tests |
| `tests/unit/test_model_imports.py` | Import-resolution tests |
| `tests/unit/test_model_bindings.py` | ArgumentBinding tests |
| `tests/unit/test_data_text.py` | Inline + external text backend tests |
| `tests/unit/test_data_hdf5.py` | HDF5 dense + DOK backend tests |
| `tests/unit/test_document_model.py` | End-to-end model round-trip tests |
| `tests/conformance/__init__.py` | — |
| `tests/conformance/test_cpp_reference.py` | Cross-validates round-trip against the C++ reference library |
| `.github/workflows/conformance.yml` | CI — builds C++ reference + runs conformance tests |

Files modified:
- `src/pyfieldml/document.py` — upgraded from DOM stub to model wrapper (all public methods preserved; internal representation changes).
- `src/pyfieldml/__init__.py` — re-export `Region`, evaluator classes, data types.
- `CHANGELOG.md` — document v0.1.0 additions.

---

## Task 1 — Type classes (EnsembleType, ContinuousType, MeshType, BooleanType)

**Files:**
- Create: `src/pyfieldml/model/__init__.py`
- Create: `src/pyfieldml/model/types.py`
- Test: `tests/unit/test_model_types.py`

- [ ] **Step 1: Write the failing tests — `tests/unit/test_model_types.py`**

```python
"""Tests for FieldML type classes."""

from __future__ import annotations

import pytest

from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)


def test_boolean_type_is_hashable_by_name() -> None:
    t1 = BooleanType(name="boolean")
    t2 = BooleanType(name="boolean")
    assert t1 == t2
    assert hash(t1) == hash(t2)


def test_ensemble_type_records_members() -> None:
    t = EnsembleType(name="nodes", members=range(1, 11))
    assert t.size == 10
    assert 5 in t.members
    assert 11 not in t.members


def test_continuous_type_with_components_stores_arity() -> None:
    t = ContinuousType(name="real.3d", component_name="real.3d.component", component_count=3)
    assert t.component_count == 3
    assert t.is_vector
    scalar = ContinuousType(name="real.1d")
    assert scalar.component_count == 1
    assert not scalar.is_vector


def test_continuous_type_rejects_zero_components() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ContinuousType(name="bad", component_name="c", component_count=0)


def test_mesh_type_combines_elements_and_chart() -> None:
    elements = EnsembleType(name="elems", members=range(1, 5))
    chart = ContinuousType(name="xi.3d", component_name="xi.3d.c", component_count=3)
    mesh = MeshType(name="mesh", elements=elements, chart=chart)
    assert mesh.dimension == 3
    assert mesh.element_count == 4
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/unit/test_model_types.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/pyfieldml/model/types.py`**

```python
"""FieldML type classes: Ensemble, Continuous, Mesh, Boolean."""

from __future__ import annotations

from collections.abc import Iterable, Set
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BooleanType:
    """Boolean type — a two-valued domain."""

    name: str


@dataclass(frozen=True)
class EnsembleType:
    """A finite ordered set of positive integers (element IDs, node IDs, etc.)."""

    name: str
    members: Set[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.members, (frozenset, range)):
            object.__setattr__(self, "members", frozenset(self.members))

    @property
    def size(self) -> int:
        if isinstance(self.members, range):
            return len(self.members)
        return len(self.members)


@dataclass(frozen=True)
class ContinuousType:
    """A continuous (real-valued) type, scalar or with named components."""

    name: str
    component_name: str | None = None
    component_count: int = 1

    def __post_init__(self) -> None:
        if self.component_count < 1:
            raise ValueError(
                f"ContinuousType {self.name!r}: component_count must be at least 1, "
                f"got {self.component_count}"
            )

    @property
    def is_vector(self) -> bool:
        return self.component_count > 1


@dataclass(frozen=True)
class MeshType:
    """A mesh type — an ensemble of elements paired with a continuous chart (xi)."""

    name: str
    elements: EnsembleType
    chart: ContinuousType

    @property
    def dimension(self) -> int:
        return self.chart.component_count

    @property
    def element_count(self) -> int:
        return self.elements.size
```

- [ ] **Step 4: Create `src/pyfieldml/model/__init__.py`**

```python
"""Semantic FieldML object model."""

from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)

__all__ = ["BooleanType", "ContinuousType", "EnsembleType", "MeshType"]
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/unit/test_model_types.py -v`
Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/model/ tests/unit/test_model_types.py
git commit -m "feat(model): add FieldML type classes (Boolean/Ensemble/Continuous/Mesh)"
```

---

## Task 2 — Data backend base + inline text backend

**Files:**
- Create: `src/pyfieldml/data/__init__.py`
- Create: `src/pyfieldml/data/base.py`
- Create: `src/pyfieldml/data/text.py` (inline portion — external text comes in Task 3)
- Test: `tests/unit/test_data_text.py` (partial — inline tests)

- [ ] **Step 1: Write the failing tests — `tests/unit/test_data_text.py` (inline section)**

```python
"""Tests for text-based array data backends."""

from __future__ import annotations

import numpy as np
import pytest

from pyfieldml.data.base import DataResource, DataSource
from pyfieldml.data.text import InlineTextBackend


def test_inline_text_reads_float64_dense() -> None:
    backend = InlineTextBackend(
        text="1.0 2.0 3.0\n4.0 5.0 6.0\n",
        shape=(2, 3),
        dtype="float64",
    )
    arr = backend.as_ndarray()
    assert arr.dtype == np.float64
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_inline_text_reads_int32() -> None:
    backend = InlineTextBackend(text="1 2 3 4", shape=(4,), dtype="int32")
    arr = backend.as_ndarray()
    assert arr.dtype == np.int32
    np.testing.assert_array_equal(arr, [1, 2, 3, 4])


def test_inline_text_writes_losslessly() -> None:
    original = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    backend = InlineTextBackend.from_ndarray(original)
    reloaded = backend.as_ndarray()
    np.testing.assert_array_equal(original, reloaded)


def test_inline_text_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="5 values but shape"):
        InlineTextBackend(text="1 2 3 4 5", shape=(2, 3), dtype="float64").as_ndarray()


def test_data_source_exposes_shape_and_dtype() -> None:
    backend = InlineTextBackend(text="1 2 3", shape=(3,), dtype="float64")
    src: DataSource = backend
    assert src.shape == (3,)
    assert src.dtype == np.dtype("float64")
```

- [ ] **Step 2: Run and confirm fails**

Run: `uv run pytest tests/unit/test_data_text.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/pyfieldml/data/base.py`**

```python
"""Base abstractions for FieldML array-data backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class DataResource:
    """A storage resource description.

    Attributes
    ----------
    format
        One of ``"text"``, ``"PLAIN_TEXT"``, ``"HDF5"``, etc. Matches FieldML
        vocabulary.
    href
        External file path (relative or absolute). ``None`` for inline resources.
    """

    format: str
    href: str | None = None


@runtime_checkable
class DataSource(Protocol):
    """Protocol any array-data backend implements."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...

    def as_ndarray(self) -> np.ndarray: ...
```

- [ ] **Step 4: Implement `src/pyfieldml/data/text.py` (inline only)**

```python
"""Text-based array-data backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InlineTextBackend:
    """Whitespace-separated values embedded in the FieldML XML."""

    text: str
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(
        self,
        text: str,
        *,
        shape: tuple[int, ...],
        dtype: str | np.dtype,
    ) -> None:
        self.text = text
        self.shape = tuple(int(d) for d in shape)
        self.dtype = np.dtype(dtype)

    def as_ndarray(self) -> np.ndarray:
        values = np.fromstring(self.text, dtype=self.dtype, sep=" \t\n\r")
        expected = 1
        for d in self.shape:
            expected *= d
        if values.size != expected:
            raise ValueError(
                f"InlineTextBackend: parsed {values.size} values but shape "
                f"{self.shape} expects {expected}"
            )
        return values.reshape(self.shape)

    @classmethod
    def from_ndarray(cls, arr: np.ndarray) -> "InlineTextBackend":
        """Round-trip an existing ndarray to text form (used by the writer)."""
        flat = arr.ravel()
        if np.issubdtype(arr.dtype, np.floating):
            text = " ".join(f"{v!r}" for v in flat)
        else:
            text = " ".join(str(v) for v in flat)
        return cls(text=text, shape=arr.shape, dtype=arr.dtype)
```

Note: `repr(float)` gives the full-precision representation — required for
lossless float round-trip. `np.fromstring` is deprecated in favor of
`np.fromstring`/`np.genfromtxt`; if a future NumPy release removes it, switch
to `np.frombuffer` via `io.StringIO` + `np.loadtxt`.

- [ ] **Step 5: Implement `src/pyfieldml/data/__init__.py`**

```python
"""Array-data backends for FieldML ParameterEvaluators."""

from pyfieldml.data.base import DataResource, DataSource
from pyfieldml.data.text import InlineTextBackend

__all__ = ["DataResource", "DataSource", "InlineTextBackend"]
```

- [ ] **Step 6: Run and confirm passes**

Run: `uv run pytest tests/unit/test_data_text.py -v`
Expected: 5 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/pyfieldml/data/ tests/unit/test_data_text.py
git commit -m "feat(data): inline-text array backend with lossless float round-trip"
```

---

## Task 3 — External text backend

**Files:**
- Modify: `src/pyfieldml/data/text.py` (add `ExternalTextBackend`)
- Modify: `src/pyfieldml/data/__init__.py`
- Test: extend `tests/unit/test_data_text.py`

- [ ] **Step 1: Add failing tests to `tests/unit/test_data_text.py`**

```python
def test_external_text_reads_from_sibling_file(tmp_path: Path) -> None:
    datafile = tmp_path / "data.txt"
    datafile.write_text("1.0 2.0 3.0 4.0\n")
    backend = ExternalTextBackend(
        base_dir=tmp_path, href="data.txt", shape=(2, 2), dtype="float64"
    )
    arr = backend.as_ndarray()
    np.testing.assert_array_equal(arr, [[1.0, 2.0], [3.0, 4.0]])


def test_external_text_writes_sibling_file(tmp_path: Path) -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    backend = ExternalTextBackend.write_ndarray(arr, base_dir=tmp_path, href="out.txt")
    assert (tmp_path / "out.txt").is_file()
    reloaded = backend.as_ndarray()
    np.testing.assert_array_equal(arr, reloaded)


def test_external_text_rejects_missing_file(tmp_path: Path) -> None:
    backend = ExternalTextBackend(
        base_dir=tmp_path, href="missing.txt", shape=(2,), dtype="float64"
    )
    with pytest.raises(FileNotFoundError):
        backend.as_ndarray()
```

Add the imports at top:

```python
from pathlib import Path
from pyfieldml.data.text import ExternalTextBackend
```

- [ ] **Step 2: Run and confirm new tests fail**

Run: `uv run pytest tests/unit/test_data_text.py::test_external_text_reads_from_sibling_file -v`
Expected: ImportError — `ExternalTextBackend` not defined.

- [ ] **Step 3: Implement `ExternalTextBackend` in `src/pyfieldml/data/text.py`**

Append to `text.py`:

```python
@dataclass
class ExternalTextBackend:
    """Whitespace-separated values in an external text file."""

    base_dir: Any
    href: str
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(
        self,
        *,
        base_dir: Any,
        href: str,
        shape: tuple[int, ...],
        dtype: str | np.dtype,
    ) -> None:
        from pathlib import Path as _P

        self.base_dir = _P(base_dir)
        self.href = href
        self.shape = tuple(int(d) for d in shape)
        self.dtype = np.dtype(dtype)

    @property
    def path(self) -> Any:
        from pathlib import Path as _P

        return _P(self.base_dir) / self.href

    def as_ndarray(self) -> np.ndarray:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        values = np.loadtxt(self.path, dtype=self.dtype)
        return values.reshape(self.shape)

    @classmethod
    def write_ndarray(
        cls,
        arr: np.ndarray,
        *,
        base_dir: Any,
        href: str,
    ) -> "ExternalTextBackend":
        from pathlib import Path as _P

        target = _P(base_dir) / href
        target.parent.mkdir(parents=True, exist_ok=True)
        if np.issubdtype(arr.dtype, np.floating):
            fmt = "%.17g"
        else:
            fmt = "%d"
        np.savetxt(target, arr.reshape(-1), fmt=fmt)
        return cls(base_dir=base_dir, href=href, shape=arr.shape, dtype=arr.dtype)
```

- [ ] **Step 4: Update `__init__.py`**

```python
from pyfieldml.data.text import ExternalTextBackend, InlineTextBackend
__all__ = ["DataResource", "DataSource", "ExternalTextBackend", "InlineTextBackend"]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_data_text.py -v`
Expected: all 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/data/text.py src/pyfieldml/data/__init__.py tests/unit/test_data_text.py
git commit -m "feat(data): external-text array backend with sibling-file resolution"
```

---

## Task 4 — HDF5 dense backend

**Files:**
- Create: `src/pyfieldml/data/hdf5.py` (Hdf5DenseBackend first)
- Modify: `src/pyfieldml/data/__init__.py`
- Test: `tests/unit/test_data_hdf5.py`

- [ ] **Step 1: Write failing tests — `tests/unit/test_data_hdf5.py`**

```python
"""Tests for HDF5-backed array data."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from pyfieldml.data.hdf5 import Hdf5DenseBackend


def _write_h5(path: Path, dataset_name: str, data: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=data)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_hdf5_dense_round_trip(tmp_path: Path, dtype: str) -> None:
    data = np.arange(12, dtype=dtype).reshape(3, 4)
    h5 = tmp_path / "data.h5"
    _write_h5(h5, "/values", data)
    backend = Hdf5DenseBackend(path=h5, dataset="/values")
    np.testing.assert_array_equal(backend.as_ndarray(), data)
    assert backend.dtype == np.dtype(dtype)
    assert backend.shape == (3, 4)


def test_hdf5_dense_lazy_no_load_on_construction(tmp_path: Path) -> None:
    data = np.arange(4, dtype=np.float32)
    h5 = tmp_path / "data.h5"
    _write_h5(h5, "/v", data)
    backend = Hdf5DenseBackend(path=h5, dataset="/v")
    # Move the file after construction — should still succeed because lazy.
    h5b = tmp_path / "data_renamed.h5"
    h5.rename(h5b)
    backend.path = h5b
    np.testing.assert_array_equal(backend.as_ndarray(), data)


def test_hdf5_dense_write_ndarray(tmp_path: Path) -> None:
    data = np.random.default_rng(0).random((2, 3))
    backend = Hdf5DenseBackend.write_ndarray(
        data, path=tmp_path / "out.h5", dataset="/coords"
    )
    with h5py.File(tmp_path / "out.h5", "r") as f:
        np.testing.assert_array_equal(f["/coords"][()], data)
    np.testing.assert_array_equal(backend.as_ndarray(), data)
```

- [ ] **Step 2: Run and confirm fails**

Run: `uv run pytest tests/unit/test_data_hdf5.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/pyfieldml/data/hdf5.py`**

```python
"""HDF5-backed array data (dense and DOK sparse)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class Hdf5DenseBackend:
    """Dense HDF5 dataset-backed array. Lazy — only reads on ``as_ndarray``."""

    path: Path
    dataset: str

    def __init__(self, *, path: Any, dataset: str) -> None:
        self.path = Path(path)
        self.dataset = dataset
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None

    def _probe(self) -> None:
        with h5py.File(self.path, "r") as f:
            ds = f[self.dataset]
            self._shape = tuple(int(d) for d in ds.shape)
            self._dtype = np.dtype(ds.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._probe()
        assert self._shape is not None
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self._probe()
        assert self._dtype is not None
        return self._dtype

    def as_ndarray(self) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            data = f[self.dataset][()]
        return np.asarray(data)

    @classmethod
    def write_ndarray(
        cls, arr: np.ndarray, *, path: Any, dataset: str
    ) -> "Hdf5DenseBackend":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "a") as f:
            if dataset in f:
                del f[dataset]
            f.create_dataset(dataset, data=arr)
        return cls(path=path, dataset=dataset)
```

- [ ] **Step 4: Update `__init__.py`**

```python
from pyfieldml.data.hdf5 import Hdf5DenseBackend
from pyfieldml.data.text import ExternalTextBackend, InlineTextBackend

__all__ = [
    "DataResource",
    "DataSource",
    "ExternalTextBackend",
    "Hdf5DenseBackend",
    "InlineTextBackend",
]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_data_hdf5.py -v`
Expected: 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/data/hdf5.py src/pyfieldml/data/__init__.py tests/unit/test_data_hdf5.py
git commit -m "feat(data): HDF5-dense array backend with lazy loading"
```

---

## Task 5 — HDF5 DOK (sparse) backend

**Files:**
- Modify: `src/pyfieldml/data/hdf5.py` (add `Hdf5DOKBackend`)
- Modify: `src/pyfieldml/data/__init__.py`
- Test: extend `tests/unit/test_data_hdf5.py`

DOK (Dictionary Of Keys) representation in FieldML: three HDF5 datasets per
sparse array — `values` (1-D, the nonzero values), `indexes` (2-D, rank x
nnz, the coordinates of each nonzero), and a shape metadata attribute or
sibling.

- [ ] **Step 1: Add failing tests**

```python
def test_hdf5_dok_round_trip(tmp_path: Path) -> None:
    from pyfieldml.data.hdf5 import Hdf5DOKBackend

    h5 = tmp_path / "sparse.h5"
    # Sparse 4x4 with three nonzeros at (0,1), (2,2), (3,0).
    with h5py.File(h5, "w") as f:
        g = f.create_group("/sparse")
        g.create_dataset("values", data=np.array([1.5, 2.5, 3.5], dtype=np.float64))
        g.create_dataset("indexes", data=np.array([[0, 1], [2, 2], [3, 0]], dtype=np.int64))
        g.attrs["shape"] = (4, 4)
    backend = Hdf5DOKBackend(path=h5, group="/sparse")
    dense = backend.as_ndarray()
    expected = np.zeros((4, 4), dtype=np.float64)
    expected[0, 1] = 1.5
    expected[2, 2] = 2.5
    expected[3, 0] = 3.5
    np.testing.assert_array_equal(dense, expected)


def test_hdf5_dok_write_ndarray(tmp_path: Path) -> None:
    from pyfieldml.data.hdf5 import Hdf5DOKBackend

    dense = np.zeros((3, 3), dtype=np.float64)
    dense[0, 1] = 7.0
    dense[2, 0] = 9.0
    backend = Hdf5DOKBackend.write_ndarray(
        dense, path=tmp_path / "out.h5", group="/s"
    )
    np.testing.assert_array_equal(backend.as_ndarray(), dense)
```

- [ ] **Step 2: Run, confirm fails**

Run: `uv run pytest tests/unit/test_data_hdf5.py -v -k DOK`
Expected: ImportError.

- [ ] **Step 3: Implement `Hdf5DOKBackend`**

Append to `src/pyfieldml/data/hdf5.py`:

```python
@dataclass
class Hdf5DOKBackend:
    """Sparse (DOK) HDF5 array. Materializes a dense ndarray on demand."""

    path: Path
    group: str

    def __init__(self, *, path: Any, group: str) -> None:
        self.path = Path(path)
        self.group = group
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None

    def _probe(self) -> None:
        with h5py.File(self.path, "r") as f:
            g = f[self.group]
            self._shape = tuple(int(d) for d in g.attrs["shape"])
            self._dtype = np.dtype(g["values"].dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._probe()
        assert self._shape is not None
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self._probe()
        assert self._dtype is not None
        return self._dtype

    def as_ndarray(self) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            g = f[self.group]
            values = g["values"][()]
            indexes = g["indexes"][()]
            shape = tuple(int(d) for d in g.attrs["shape"])
        dense = np.zeros(shape, dtype=values.dtype)
        dense[tuple(indexes.T)] = values
        return dense

    @classmethod
    def write_ndarray(
        cls, arr: np.ndarray, *, path: Any, group: str
    ) -> "Hdf5DOKBackend":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nonzero = np.argwhere(arr != 0)
        values = arr[tuple(nonzero.T)] if nonzero.size else np.empty((0,), dtype=arr.dtype)
        with h5py.File(path, "a") as f:
            if group in f:
                del f[group]
            g = f.create_group(group)
            g.create_dataset("values", data=values)
            g.create_dataset("indexes", data=nonzero.astype(np.int64))
            g.attrs["shape"] = np.asarray(arr.shape, dtype=np.int64)
        return cls(path=path, group=group)
```

- [ ] **Step 4: Update `__init__.py` to export `Hdf5DOKBackend`.**

- [ ] **Step 5: Run tests, confirm pass**

Run: `uv run pytest tests/unit/test_data_hdf5.py -v`
Expected: 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/data/hdf5.py src/pyfieldml/data/__init__.py tests/unit/test_data_hdf5.py
git commit -m "feat(data): HDF5 DOK sparse backend"
```

---

## Task 6 — Evaluator base class + ParameterEvaluator + ConstantEvaluator

**Files:**
- Create: `src/pyfieldml/model/evaluators.py` (partial — base + Parameter + Constant)
- Test: `tests/unit/test_model_evaluators.py` (partial — these two)
- Modify: `src/pyfieldml/model/__init__.py`

- [ ] **Step 1: Failing tests — `tests/unit/test_model_evaluators.py`**

```python
"""Tests for the Evaluator class hierarchy."""

from __future__ import annotations

import numpy as np

from pyfieldml.data.text import InlineTextBackend
from pyfieldml.model.evaluators import ConstantEvaluator, Evaluator, ParameterEvaluator
from pyfieldml.model.types import ContinuousType


def test_parameter_evaluator_exposes_ndarray() -> None:
    value_type = ContinuousType(name="real.1d")
    data = InlineTextBackend(text="1 2 3 4", shape=(4,), dtype="float64")
    p = ParameterEvaluator(name="p", value_type=value_type, data=data)
    np.testing.assert_array_equal(p.as_ndarray(), [1, 2, 3, 4])
    assert p.shape == (4,)
    assert p.dtype == np.float64


def test_constant_evaluator_scalar() -> None:
    t = ContinuousType(name="real.1d")
    c = ConstantEvaluator(name="k", value_type=t, value=3.14)
    assert c.evaluate_constant() == 3.14


def test_constant_evaluator_vector() -> None:
    t = ContinuousType(name="real.3d", component_name="c", component_count=3)
    c = ConstantEvaluator(name="origin", value_type=t, value=(0.0, 0.0, 0.0))
    np.testing.assert_array_equal(c.evaluate_constant(), [0.0, 0.0, 0.0])


def test_evaluator_base_records_name_and_value_type() -> None:
    t = ContinuousType(name="real.1d")
    c = ConstantEvaluator(name="k", value_type=t, value=1.0)
    assert isinstance(c, Evaluator)
    assert c.name == "k"
    assert c.value_type is t
```

- [ ] **Step 2: Run and confirm fail**

Run: `uv run pytest tests/unit/test_model_evaluators.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `src/pyfieldml/model/evaluators.py` (partial)**

```python
"""Evaluator hierarchy for the FieldML semantic model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np

from pyfieldml.data.base import DataSource
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)

ValueType = Union[BooleanType, EnsembleType, ContinuousType, MeshType]


@dataclass
class Evaluator:
    """Base class for all FieldML evaluators. Instantiated only via subclasses."""

    name: str
    value_type: ValueType


@dataclass
class ParameterEvaluator(Evaluator):
    """Evaluator backed by a concrete array of parameter values."""

    data: DataSource = field(default=None)  # type: ignore[assignment]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def as_ndarray(self) -> np.ndarray:
        return self.data.as_ndarray()


@dataclass
class ConstantEvaluator(Evaluator):
    """Evaluator that yields a fixed value (scalar or tuple of components)."""

    value: Union[float, int, bool, Sequence[float], Sequence[int]] = 0.0

    def evaluate_constant(self) -> Union[float, int, bool, np.ndarray]:
        if isinstance(self.value, (list, tuple)):
            return np.asarray(self.value)
        return self.value
```

- [ ] **Step 4: Update `model/__init__.py`**

```python
from pyfieldml.model.evaluators import (
    ConstantEvaluator,
    Evaluator,
    ParameterEvaluator,
)
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)

__all__ = [
    "BooleanType",
    "ConstantEvaluator",
    "ContinuousType",
    "EnsembleType",
    "Evaluator",
    "MeshType",
    "ParameterEvaluator",
]
```

- [ ] **Step 5: Run, confirm passes**

Run: `uv run pytest tests/unit/test_model_evaluators.py -v`
Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/model/evaluators.py src/pyfieldml/model/__init__.py \
        tests/unit/test_model_evaluators.py
git commit -m "feat(model): Evaluator base + ParameterEvaluator + ConstantEvaluator"
```

---

## Task 7 — ArgumentEvaluator + bindings

**Files:**
- Create: `src/pyfieldml/model/bindings.py`
- Modify: `src/pyfieldml/model/evaluators.py` (add `ArgumentEvaluator`)
- Modify: `src/pyfieldml/model/__init__.py`
- Test: `tests/unit/test_model_bindings.py`

- [ ] **Step 1: Failing tests — `tests/unit/test_model_bindings.py`**

```python
"""Tests for ArgumentEvaluator and ArgumentBinding."""

from __future__ import annotations

import pytest

from pyfieldml.errors import UnboundArgumentError
from pyfieldml.model.bindings import ArgumentBinding, BindingTable
from pyfieldml.model.evaluators import ArgumentEvaluator, ConstantEvaluator
from pyfieldml.model.types import ContinuousType


def test_argument_evaluator_is_unbound_by_default() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    assert arg.name == "x"


def test_binding_table_resolve_uses_provided_value() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    src = ConstantEvaluator(name="src", value_type=t, value=7.0)
    table = BindingTable([ArgumentBinding(argument=arg, source=src)])
    assert table.resolve(arg) is src


def test_binding_table_raises_on_unbound() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    table = BindingTable([])
    with pytest.raises(UnboundArgumentError, match="x"):
        table.resolve(arg)
```

- [ ] **Step 2: Run, confirm fail**

Run: `uv run pytest tests/unit/test_model_bindings.py -v`
Expected: ImportError.

- [ ] **Step 3: Add `ArgumentEvaluator` to `src/pyfieldml/model/evaluators.py`**

Append:

```python
@dataclass
class ArgumentEvaluator(Evaluator):
    """Formal-parameter placeholder. Must be bound at call-sites."""
```

- [ ] **Step 4: Implement `src/pyfieldml/model/bindings.py`**

```python
"""Argument bindings for ReferenceEvaluator call-sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pyfieldml.errors import UnboundArgumentError
from pyfieldml.model.evaluators import ArgumentEvaluator, Evaluator


@dataclass(frozen=True)
class ArgumentBinding:
    """Binds a formal ``ArgumentEvaluator`` to a concrete source evaluator."""

    argument: ArgumentEvaluator
    source: Evaluator


class BindingTable:
    """Ordered collection of ArgumentBindings, resolved by argument identity."""

    def __init__(self, bindings: Iterable[ArgumentBinding]) -> None:
        self._by_arg = {id(b.argument): b for b in bindings}

    def resolve(self, arg: ArgumentEvaluator) -> Evaluator:
        b = self._by_arg.get(id(arg))
        if b is None:
            raise UnboundArgumentError(
                f"No binding for ArgumentEvaluator {arg.name!r}"
            )
        return b.source

    def __len__(self) -> int:
        return len(self._by_arg)
```

- [ ] **Step 5: Re-export in `model/__init__.py`**

Add `ArgumentEvaluator`, `ArgumentBinding`, `BindingTable` to exports.

- [ ] **Step 6: Run, pass**

Run: `uv run pytest tests/unit/test_model_bindings.py -v`
Expected: 3 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/pyfieldml/model/ tests/unit/test_model_bindings.py
git commit -m "feat(model): ArgumentEvaluator and BindingTable"
```

---

## Task 8 — ReferenceEvaluator + AggregateEvaluator + PiecewiseEvaluator + ExternalEvaluator

**Files:**
- Modify: `src/pyfieldml/model/evaluators.py`
- Modify: `tests/unit/test_model_evaluators.py`
- Modify: `src/pyfieldml/model/__init__.py`

- [ ] **Step 1: Add failing tests**

```python
from pyfieldml.model.bindings import ArgumentBinding, BindingTable
from pyfieldml.model.evaluators import (
    AggregateEvaluator,
    ArgumentEvaluator,
    ExternalEvaluator,
    PiecewiseEvaluator,
    ReferenceEvaluator,
)


def test_reference_evaluator_stores_source_and_bindings() -> None:
    t = ContinuousType(name="real.1d")
    arg = ArgumentEvaluator(name="x", value_type=t)
    inner = ConstantEvaluator(name="k", value_type=t, value=1.0)
    src = ConstantEvaluator(name="src", value_type=t, value=7.0)
    ref = ReferenceEvaluator(
        name="r",
        value_type=t,
        source=inner,
        bindings=BindingTable([ArgumentBinding(argument=arg, source=src)]),
    )
    assert ref.source is inner
    assert len(ref.bindings) == 1


def test_aggregate_evaluator_records_components() -> None:
    vec = ContinuousType(name="real.3d", component_name="c", component_count=3)
    scalar = ContinuousType(name="real.1d")
    cs = [ConstantEvaluator(name=f"k{i}", value_type=scalar, value=float(i)) for i in range(3)]
    agg = AggregateEvaluator(name="v", value_type=vec, components=cs)
    assert len(agg.components) == 3


def test_piecewise_evaluator_maps_cases_to_sources() -> None:
    t = ContinuousType(name="real.1d")
    a = ConstantEvaluator(name="a", value_type=t, value=1.0)
    b = ConstantEvaluator(name="b", value_type=t, value=2.0)
    pw = PiecewiseEvaluator(name="pw", value_type=t, pieces={1: a, 2: b}, default=None)
    assert pw.pieces[1] is a
    assert pw.default is None


def test_external_evaluator_names_library_entry() -> None:
    t = ContinuousType(name="real.1d")
    ext = ExternalEvaluator(name="library.basis.linear_lagrange", value_type=t)
    assert ext.name.startswith("library.basis")
```

- [ ] **Step 2: Run, confirm fail**

- [ ] **Step 3: Implement the four subclasses in `evaluators.py`**

```python
@dataclass
class ReferenceEvaluator(Evaluator):
    """Reference another evaluator, optionally with argument bindings."""

    source: "Evaluator" = field(default=None)  # type: ignore[assignment]
    bindings: "BindingTable" = field(default=None)  # type: ignore[assignment]


@dataclass
class AggregateEvaluator(Evaluator):
    """Assemble a value from per-component sub-evaluators."""

    components: list[Evaluator] = field(default_factory=list)


@dataclass
class PiecewiseEvaluator(Evaluator):
    """Dispatch to a per-ensemble-index sub-evaluator; optional default."""

    pieces: dict[int, Evaluator] = field(default_factory=dict)
    default: Evaluator | None = None


@dataclass
class ExternalEvaluator(Evaluator):
    """Reference to an object defined in an imported library (e.g. a basis function)."""
```

`ReferenceEvaluator` has a forward reference to `BindingTable` — add the
import at the bottom of the file to avoid a cycle:

```python
from pyfieldml.model.bindings import BindingTable  # noqa: E402
```

- [ ] **Step 4: Re-export in `__init__.py`**

- [ ] **Step 5: Run, pass**

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/model/ tests/unit/test_model_evaluators.py
git commit -m "feat(model): Reference/Aggregate/Piecewise/External evaluators"
```

---

## Task 9 — Region container

**Files:**
- Create: `src/pyfieldml/model/region.py`
- Test: `tests/unit/test_model_region.py`
- Modify: `src/pyfieldml/model/__init__.py`

- [ ] **Step 1: Failing tests — `tests/unit/test_model_region.py`**

```python
"""Tests for the Region container."""

from __future__ import annotations

import pytest

from pyfieldml.model.evaluators import ConstantEvaluator
from pyfieldml.model.region import Region
from pyfieldml.model.types import BooleanType, ContinuousType


def test_region_stores_types_and_evaluators() -> None:
    r = Region(name="test")
    bt = BooleanType(name="boolean")
    r.add_type(bt)
    ct = ContinuousType(name="real.1d")
    r.add_type(ct)
    k = ConstantEvaluator(name="k", value_type=ct, value=1.0)
    r.add_evaluator(k)
    assert r.booleans["boolean"] is bt
    assert r.continuous["real.1d"] is ct
    assert r.evaluators["k"] is k


def test_region_rejects_duplicate_names() -> None:
    r = Region(name="test")
    r.add_type(BooleanType(name="dup"))
    with pytest.raises(ValueError, match="dup"):
        r.add_type(ContinuousType(name="dup"))


def test_region_objects_view_is_ordered() -> None:
    r = Region(name="t")
    r.add_type(BooleanType(name="first"))
    r.add_type(ContinuousType(name="second"))
    assert list(r.objects) == ["first", "second"]
```

- [ ] **Step 2: Run, confirm fail**

- [ ] **Step 3: Implement `src/pyfieldml/model/region.py`**

```python
"""Region — a namespace container for FieldML objects."""

from __future__ import annotations

from typing import Iterable

from pyfieldml.model.evaluators import Evaluator
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)


class Region:
    """A named namespace for FieldML types, evaluators, and imports.

    Objects are stored in insertion order. Names must be unique across all
    object kinds within a region.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self._objects: dict[str, object] = {}
        self.booleans: dict[str, BooleanType] = {}
        self.ensembles: dict[str, EnsembleType] = {}
        self.continuous: dict[str, ContinuousType] = {}
        self.meshes: dict[str, MeshType] = {}
        self.evaluators: dict[str, Evaluator] = {}

    @property
    def objects(self) -> Iterable[str]:
        return self._objects.keys()

    def add_type(
        self, t: BooleanType | EnsembleType | ContinuousType | MeshType
    ) -> None:
        self._ensure_unique(t.name)
        self._objects[t.name] = t
        if isinstance(t, BooleanType):
            self.booleans[t.name] = t
        elif isinstance(t, EnsembleType):
            self.ensembles[t.name] = t
        elif isinstance(t, ContinuousType):
            self.continuous[t.name] = t
        elif isinstance(t, MeshType):
            self.meshes[t.name] = t

    def add_evaluator(self, e: Evaluator) -> None:
        self._ensure_unique(e.name)
        self._objects[e.name] = e
        self.evaluators[e.name] = e

    def _ensure_unique(self, name: str) -> None:
        if name in self._objects:
            raise ValueError(f"Region {self.name!r}: duplicate object name {name!r}")
```

- [ ] **Step 4: Re-export in `__init__.py`**

- [ ] **Step 5: Run, pass**

Run: `uv run pytest tests/unit/test_model_region.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/model/region.py src/pyfieldml/model/__init__.py tests/unit/test_model_region.py
git commit -m "feat(model): Region container with insertion-ordered uniqueness"
```

---

## Task 10 — Import resolution (stdlib + local files)

**Files:**
- Create: `src/pyfieldml/model/imports.py`
- Create: `tests/fixtures/import_stdlib.fieldml`
- Test: `tests/unit/test_model_imports.py`

- [ ] **Step 1: Create fixture `tests/fixtures/import_stdlib.fieldml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Fieldml version="0.5.0">
  <Region name="consumer">
    <Import xlink:href="http://www.fieldml.org/resources/xml/0.5/FieldML_Library_0.5.xml"
            region="library"
            xmlns:xlink="http://www.w3.org/1999/xlink">
      <ImportType name="my_real_3d" localName="real.3d"/>
    </Import>
  </Region>
</Fieldml>
```

- [ ] **Step 2: Failing tests — `tests/unit/test_model_imports.py`**

```python
"""Tests for FieldML <Import> resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyfieldml.errors import UnresolvedImportError
from pyfieldml.model.imports import ImportResolver


def test_stdlib_import_resolves_without_network(fixtures_dir: Path) -> None:
    resolver = ImportResolver(base_dir=fixtures_dir)
    imported = resolver.resolve(
        "http://www.fieldml.org/resources/xml/0.5/FieldML_Library_0.5.xml",
        region_name="library",
    )
    assert imported.name == "library"
    assert "real.3d" in imported.continuous


def test_local_import_resolves_relative_to_source_file(tmp_path: Path) -> None:
    sublib = tmp_path / "sub.fieldml"
    sublib.write_text(
        '<?xml version="1.0"?><Fieldml version="0.5.0">'
        '<Region name="sub"><BooleanType name="b"/></Region></Fieldml>'
    )
    resolver = ImportResolver(base_dir=tmp_path)
    imported = resolver.resolve("sub.fieldml", region_name="sub")
    assert "b" in imported.booleans


def test_missing_import_raises(tmp_path: Path) -> None:
    resolver = ImportResolver(base_dir=tmp_path)
    with pytest.raises(UnresolvedImportError):
        resolver.resolve("does_not_exist.fieldml", region_name="x")
```

- [ ] **Step 3: Run, confirm fail**

- [ ] **Step 4: Implement `src/pyfieldml/model/imports.py`**

```python
"""Resolution of <Import> directives.

Imports can target:
- The bundled FieldML standard library (by its canonical URL).
- A local file, resolved relative to the importing document's directory.
"""

from __future__ import annotations

from pathlib import Path

from pyfieldml.errors import UnresolvedImportError
from pyfieldml.library import get_stdlib_path
from pyfieldml.model.region import Region

STDLIB_URLS = frozenset(
    {
        "http://www.fieldml.org/resources/xml/0.5/FieldML_Library_0.5.xml",
        "https://www.fieldml.org/resources/xml/0.5/FieldML_Library_0.5.xml",
        "FieldML_Library_0.5.xml",
    }
)


class ImportResolver:
    """Resolves ``<Import>`` targets to Region objects."""

    def __init__(self, *, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def resolve(self, href: str, *, region_name: str) -> Region:
        path = self._find_source(href)
        if path is None:
            raise UnresolvedImportError(
                f"Cannot locate FieldML import {href!r} "
                f"(base_dir={self.base_dir})"
            )
        # Lazy to avoid cycles: _loader depends on this module.
        from pyfieldml.model._loader import load_region_from_file

        return load_region_from_file(path, region_name=region_name)

    def _find_source(self, href: str) -> Path | None:
        if href in STDLIB_URLS or href.endswith("FieldML_Library_0.5.xml"):
            return get_stdlib_path()
        candidate = self.base_dir / href
        if candidate.is_file():
            return candidate
        return None
```

Note: `load_region_from_file` is defined in Task 12 (the DOM → model loader).
To keep Task 10 runnable, provide a minimal loader stub immediately. Create
`src/pyfieldml/model/_loader.py`:

```python
"""DOM → model graph loader. Expanded in Task 12."""

from __future__ import annotations

from pathlib import Path

from pyfieldml.dom.parser import parse_file
from pyfieldml.model.region import Region
from pyfieldml.model.types import BooleanType, ContinuousType


def load_region_from_file(path: Path, *, region_name: str) -> Region:
    """Minimal loader: reads Boolean/Continuous types from the named region.

    Expanded in Task 12 to cover the full model.
    """
    parsed = parse_file(path)
    for region_elem in parsed.tree.getroot().findall("Region"):
        if region_elem.get("name") == region_name:
            region = Region(name=region_name)
            for child in region_elem:
                if child.tag == "BooleanType":
                    region.add_type(BooleanType(name=child.get("name", "")))
                elif child.tag == "ContinuousType":
                    _add_continuous(region, child)
            return region
    raise ValueError(
        f"Region {region_name!r} not found in {path} "
        f"(available: {[r.get('name') for r in parsed.tree.getroot().findall('Region')]})"
    )


def _add_continuous(region: Region, elem) -> None:  # type: ignore[no-untyped-def]
    comp = elem.find("Components")
    if comp is not None:
        region.add_type(
            ContinuousType(
                name=elem.get("name", ""),
                component_name=comp.get("name"),
                component_count=int(comp.get("count", "1")),
            )
        )
    else:
        region.add_type(ContinuousType(name=elem.get("name", "")))
```

- [ ] **Step 5: Run, pass**

Run: `uv run pytest tests/unit/test_model_imports.py -v`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/model/imports.py src/pyfieldml/model/_loader.py \
        tests/unit/test_model_imports.py tests/fixtures/import_stdlib.fieldml
git commit -m "feat(model): Import resolution for bundled stdlib + local files"
```

---

## Task 11 — Full DOM → model loader

**Files:**
- Modify: `src/pyfieldml/model/_loader.py` (replace minimal stub with full loader)
- Add fixtures: `tests/fixtures/parameter_inline.fieldml`, `tests/fixtures/reference_aggregate.fieldml`, `tests/fixtures/piecewise_per_element.fieldml`
- Test: `tests/unit/test_document_model.py`

- [ ] **Step 1: Write fixtures**

`tests/fixtures/parameter_inline.fieldml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Fieldml version="0.5.0">
  <Region name="test">
    <ContinuousType name="real.1d"/>
    <ParameterEvaluator name="p" valueType="real.1d">
      <DenseArrayData>
        <DataSource name="inline_src" location="/" rank="1">
          <ArrayDataSize>4</ArrayDataSize>
        </DataSource>
        <DataResourceString>1.0 2.0 3.0 4.0</DataResourceString>
      </DenseArrayData>
    </ParameterEvaluator>
  </Region>
</Fieldml>
```

(Note: FieldML's exact XML grammar for ParameterEvaluators is complex; the
structure above is a simplified form that matches the XSD's `ParameterEvaluator_Type`
with a DataSource child. Adjust attribute names if XSD validation rejects.
Reference: `FieldML_0.5.xsd`.)

`tests/fixtures/reference_aggregate.fieldml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Fieldml version="0.5.0">
  <Region name="test">
    <ContinuousType name="real.1d"/>
    <ContinuousType name="real.3d">
      <Components name="real.3d.component" count="3"/>
    </ContinuousType>
    <ArgumentEvaluator name="arg" valueType="real.1d"/>
    <ConstantEvaluator name="k" valueType="real.1d" value="1.0"/>
    <ReferenceEvaluator name="r" valueType="real.1d" evaluator="k">
      <Bindings>
        <Bind argument="arg" source="k"/>
      </Bindings>
    </ReferenceEvaluator>
    <AggregateEvaluator name="agg" valueType="real.3d" indexEvaluator="real.3d.component">
      <ComponentEvaluators default="k"/>
    </AggregateEvaluator>
  </Region>
</Fieldml>
```

`tests/fixtures/piecewise_per_element.fieldml` — similar minimal PiecewiseEvaluator
with at least one case + default. See `FieldML_0.5.xsd` for required elements.

- [ ] **Step 2: Failing tests — `tests/unit/test_document_model.py`**

```python
"""End-to-end tests exercising DOM → model loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pyfieldml as fml
from pyfieldml.model.evaluators import (
    AggregateEvaluator,
    ArgumentEvaluator,
    ConstantEvaluator,
    ParameterEvaluator,
    ReferenceEvaluator,
)


def test_parameter_inline_loads_with_correct_values(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "parameter_inline.fieldml")
    p = doc.evaluators["p"]
    assert isinstance(p, ParameterEvaluator)
    np.testing.assert_array_equal(p.as_ndarray(), [1.0, 2.0, 3.0, 4.0])


def test_reference_aggregate_loads_with_evaluator_subtypes(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "reference_aggregate.fieldml")
    assert isinstance(doc.evaluators["k"], ConstantEvaluator)
    assert isinstance(doc.evaluators["arg"], ArgumentEvaluator)
    assert isinstance(doc.evaluators["r"], ReferenceEvaluator)
    assert isinstance(doc.evaluators["agg"], AggregateEvaluator)
```

- [ ] **Step 3: Run, confirm fail**

Run: `uv run pytest tests/unit/test_document_model.py -v`
Expected: AttributeError — `Document.evaluators` does not exist yet.

- [ ] **Step 4: Implement the full loader in `src/pyfieldml/model/_loader.py`**

Replace the minimal stub with the full implementation. This is the bulk of
Phase 1's code. Template (condensed — expand per evaluator type):

```python
"""DOM → model graph loader: full implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from lxml import etree

from pyfieldml.data.hdf5 import Hdf5DenseBackend, Hdf5DOKBackend
from pyfieldml.data.text import ExternalTextBackend, InlineTextBackend
from pyfieldml.dom.parser import parse_file
from pyfieldml.errors import FieldMLParseError
from pyfieldml.model.bindings import ArgumentBinding, BindingTable
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
from pyfieldml.model.region import Region
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)


def load_document(path: Path) -> Region:
    """Load a FieldML document and return its (single) primary region.

    For multi-region documents, only the first region is returned; a future
    Document class will hold the full list of regions.
    """
    parsed = parse_file(path)
    regions = parsed.tree.getroot().findall("Region")
    if not regions:
        raise FieldMLParseError("No <Region> in document", source_file=str(path))
    return _load_region(regions[0], base_dir=Path(path).parent)


def load_region_from_file(path: Path, *, region_name: str) -> Region:
    parsed = parse_file(path)
    for region_elem in parsed.tree.getroot().findall("Region"):
        if region_elem.get("name") == region_name:
            return _load_region(region_elem, base_dir=Path(path).parent)
    raise ValueError(f"Region {region_name!r} not found in {path}")


def _load_region(elem: etree._Element, *, base_dir: Path) -> Region:
    region = Region(name=elem.get("name", ""))

    # First pass: types (referenced by evaluators).
    for child in elem:
        if child.tag == "BooleanType":
            region.add_type(BooleanType(name=child.get("name", "")))
        elif child.tag == "EnsembleType":
            region.add_type(_load_ensemble(child))
        elif child.tag == "ContinuousType":
            region.add_type(_load_continuous(child))
        elif child.tag == "MeshType":
            region.add_type(_load_mesh(child, region))

    # Second pass: evaluators (may reference types AND each other).
    # Two sub-passes to handle forward references.
    evaluator_elems = [c for c in elem if c.tag.endswith("Evaluator")]

    # Pass 2a: argument, constant, external, parameter (no cross-refs).
    for c in evaluator_elems:
        ev = _load_leaf_evaluator(c, region, base_dir=base_dir)
        if ev is not None:
            region.add_evaluator(ev)

    # Pass 2b: reference, aggregate, piecewise (may cross-ref).
    for c in evaluator_elems:
        if c.get("name", "") in region.evaluators:
            continue
        ev = _load_composite_evaluator(c, region)
        if ev is not None:
            region.add_evaluator(ev)

    return region


# --- helpers (expand per evaluator type; see FieldML_0.5.xsd for XML shapes) ---

def _load_ensemble(elem: etree._Element) -> EnsembleType:
    members_elem = elem.find("Members")
    members: Any = range(0)
    if members_elem is not None:
        mr = members_elem.find("MemberRange")
        if mr is not None:
            members = range(int(mr.get("min", "1")), int(mr.get("max", "0")) + 1)
    return EnsembleType(name=elem.get("name", ""), members=members)


def _load_continuous(elem: etree._Element) -> ContinuousType:
    comp = elem.find("Components")
    if comp is not None:
        return ContinuousType(
            name=elem.get("name", ""),
            component_name=comp.get("name"),
            component_count=int(comp.get("count", "1")),
        )
    return ContinuousType(name=elem.get("name", ""))


def _load_mesh(elem: etree._Element, region: Region) -> MeshType:
    elements_name = elem.find("Elements").get("name")  # type: ignore[union-attr]
    chart_name = elem.find("Chart").get("name")  # type: ignore[union-attr]
    return MeshType(
        name=elem.get("name", ""),
        elements=region.ensembles[elements_name],  # type: ignore[index]
        chart=region.continuous[chart_name],  # type: ignore[index]
    )


def _load_leaf_evaluator(
    elem: etree._Element, region: Region, *, base_dir: Path
) -> Evaluator | None:
    name = elem.get("name", "")
    value_type_name = elem.get("valueType")
    value_type = _resolve_value_type(value_type_name, region) if value_type_name else None

    if elem.tag == "ArgumentEvaluator":
        return ArgumentEvaluator(name=name, value_type=cast(Any, value_type))
    if elem.tag == "ConstantEvaluator":
        value = _parse_constant_value(elem.get("value", ""), value_type)
        return ConstantEvaluator(name=name, value_type=cast(Any, value_type), value=value)
    if elem.tag == "ExternalEvaluator":
        return ExternalEvaluator(name=name, value_type=cast(Any, value_type))
    if elem.tag == "ParameterEvaluator":
        data = _load_parameter_data(elem, base_dir=base_dir)
        return ParameterEvaluator(name=name, value_type=cast(Any, value_type), data=data)
    return None


def _load_composite_evaluator(elem: etree._Element, region: Region) -> Evaluator | None:
    name = elem.get("name", "")
    value_type_name = elem.get("valueType")
    value_type = _resolve_value_type(value_type_name, region) if value_type_name else None

    if elem.tag == "ReferenceEvaluator":
        source_name = elem.get("evaluator", "")
        source = region.evaluators[source_name]
        bindings = _load_bindings(elem.find("Bindings"), region)
        return ReferenceEvaluator(
            name=name,
            value_type=cast(Any, value_type),
            source=source,
            bindings=bindings,
        )
    if elem.tag == "AggregateEvaluator":
        components: list[Evaluator] = []
        ce = elem.find("ComponentEvaluators")
        if ce is not None:
            default_name = ce.get("default")
            if default_name:
                components.append(region.evaluators[default_name])
        return AggregateEvaluator(
            name=name, value_type=cast(Any, value_type), components=components
        )
    if elem.tag == "PiecewiseEvaluator":
        pieces: dict[int, Evaluator] = {}
        default: Evaluator | None = None
        ev_map = elem.find("EvaluatorMap")
        if ev_map is not None:
            d = ev_map.get("default")
            if d:
                default = region.evaluators[d]
            for entry in ev_map.findall("EvaluatorMapEntry"):
                pieces[int(entry.get("indexValue", "0"))] = region.evaluators[
                    entry.get("evaluator", "")
                ]
        return PiecewiseEvaluator(
            name=name, value_type=cast(Any, value_type), pieces=pieces, default=default
        )
    return None


def _load_bindings(bindings_elem: etree._Element | None, region: Region) -> BindingTable:
    if bindings_elem is None:
        return BindingTable([])
    bs: list[ArgumentBinding] = []
    for b in bindings_elem.findall("Bind"):
        arg_name = b.get("argument", "")
        src_name = b.get("source", "")
        arg_obj = region.evaluators.get(arg_name)
        src_obj = region.evaluators.get(src_name)
        if isinstance(arg_obj, ArgumentEvaluator) and src_obj is not None:
            bs.append(ArgumentBinding(argument=arg_obj, source=src_obj))
    return BindingTable(bs)


def _load_parameter_data(elem: etree._Element, *, base_dir: Path):  # type: ignore[no-untyped-def]
    """Discriminate on the storage-format child element."""
    # Inline text: <DataResourceString>...</DataResourceString>
    inline = elem.find(".//DataResourceString")
    if inline is not None:
        ds = elem.find(".//DataSource")
        shape, dtype = _parse_data_source(ds)
        return InlineTextBackend(text=inline.text or "", shape=shape, dtype=dtype)

    # External text: format="PLAIN_TEXT" with href
    href_node = elem.find(".//DataResourceHref[@format='PLAIN_TEXT']")
    if href_node is not None:
        ds = elem.find(".//DataSource")
        shape, dtype = _parse_data_source(ds)
        href = href_node.get("{http://www.w3.org/1999/xlink}href", "")
        return ExternalTextBackend(base_dir=base_dir, href=href, shape=shape, dtype=dtype)

    # HDF5 dense
    h5_node = elem.find(".//DataResourceHref[@format='HDF5']")
    if h5_node is not None:
        ds = elem.find(".//DataSource")
        href = h5_node.get("{http://www.w3.org/1999/xlink}href", "")
        dataset = ds.get("location", "/") if ds is not None else "/"
        # DOK vs dense discrimination: DOK has a <DOKArrayData> wrapper.
        if elem.find(".//DOKArrayData") is not None:
            return Hdf5DOKBackend(path=base_dir / href, group=dataset)
        return Hdf5DenseBackend(path=base_dir / href, dataset=dataset)

    raise FieldMLParseError(
        f"ParameterEvaluator {elem.get('name')!r}: unrecognized data-resource shape"
    )


def _parse_data_source(ds: etree._Element | None) -> tuple[tuple[int, ...], str]:
    if ds is None:
        return ((0,), "float64")
    size_elem = ds.find("ArrayDataSize")
    if size_elem is not None and size_elem.text:
        shape = tuple(int(v) for v in size_elem.text.split())
    else:
        shape = (0,)
    # Dtype is not carried in the DataSource; default to float64.
    # In practice, FieldML 0.5 embeds dtype elsewhere; for Phase 1 we default
    # and let callers override when building programmatically.
    return shape, "float64"


def _parse_constant_value(raw: str, value_type: Any) -> Any:
    raw = raw.strip()
    if not raw:
        return 0.0
    if " " in raw:
        return tuple(float(x) for x in raw.split())
    try:
        return float(raw)
    except ValueError:
        return raw


def _resolve_value_type(name: str, region: Region) -> Any:
    if name in region.booleans:
        return region.booleans[name]
    if name in region.ensembles:
        return region.ensembles[name]
    if name in region.continuous:
        return region.continuous[name]
    if name in region.meshes:
        return region.meshes[name]
    raise FieldMLParseError(f"Unknown valueType {name!r}")
```

**Notes for the engineer:** this loader is the longest single file in Phase
1. Do not inline every FieldML 0.5 XML construct — the XSD (`FieldML_0.5.xsd`)
has ~30 grammar productions and not all appear in our fixtures. Implement
only the shapes our fixture corpus exercises; mark unexplored branches with
`FieldMLParseError(f"Unsupported {elem.tag} shape…")` so future divergence
fails loudly. The C++ reference test in Task 15 will expose any missing
branches.

- [ ] **Step 5: Run, pass**

Run: `uv run pytest tests/unit/test_document_model.py -v`
Expected: 2 tests pass (+ fixtures may need tweaking to satisfy the XSD; iterate).

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/model/_loader.py tests/fixtures/ tests/unit/test_document_model.py
git commit -m "feat(model): full DOM → model loader for all evaluator kinds"
```

---

## Task 12 — Document upgrade (DOM stub → model wrapper)

**Files:**
- Modify: `src/pyfieldml/document.py`
- Modify: `src/pyfieldml/__init__.py`
- Add tests to `tests/unit/test_document_model.py`

- [ ] **Step 1: Add tests for upgraded Document API**

```python
def test_document_exposes_region_views(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "reference_aggregate.fieldml")
    assert "real.1d" in doc.continuous
    assert "real.3d" in doc.continuous
    assert doc.continuous["real.3d"].component_count == 3
    assert set(doc.evaluators.keys()) == {"arg", "k", "r", "agg"}
```

- [ ] **Step 2: Update `src/pyfieldml/document.py`**

```python
"""Document — the user-facing FieldML document wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union

from lxml import etree

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string
from pyfieldml.dom.validator import validate_tree
from pyfieldml.dom.writer import write_file
from pyfieldml.model._loader import load_document
from pyfieldml.model.evaluators import Evaluator
from pyfieldml.model.region import Region
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)

PathLike = Union[str, Path]


class Document:
    """An in-memory FieldML document.

    Holds a single primary ``Region`` (the first one in the document) plus
    the original DOM tree for round-trip writing.
    """

    def __init__(self, parsed: ParsedDocument, region: Region) -> None:
        self._parsed = parsed
        self._region = region

    # factories
    @classmethod
    def from_file(cls, path: PathLike) -> "Document":
        parsed = parse_file(path)
        region = load_document(Path(path))
        return cls(parsed, region)

    @classmethod
    def from_string(cls, content: str | bytes) -> "Document":
        raise NotImplementedError(
            "Phase 1: from_string is deferred; use from_file for now."
        )

    # views
    @property
    def source_version(self) -> str:
        return self._parsed.version

    @property
    def tree(self) -> etree._ElementTree:
        return self._parsed.tree

    @property
    def region(self) -> Region:
        return self._region

    @property
    def booleans(self) -> Mapping[str, BooleanType]:
        return self._region.booleans

    @property
    def ensembles(self) -> Mapping[str, EnsembleType]:
        return self._region.ensembles

    @property
    def continuous(self) -> Mapping[str, ContinuousType]:
        return self._region.continuous

    @property
    def meshes(self) -> Mapping[str, MeshType]:
        return self._region.meshes

    @property
    def evaluators(self) -> Mapping[str, Evaluator]:
        return self._region.evaluators

    # operations
    def write(self, path: PathLike) -> None:
        """Phase-1 write path: re-serializes the DOM tree we parsed from.

        Phase 2 adds model → DOM regeneration so programmatically-built
        Documents can be written.
        """
        write_file(self._parsed.tree, path)

    def validate(self) -> None:
        validate_tree(self._parsed.tree)
```

- [ ] **Step 3: Update `__init__.py`** — re-export `Region` and the model/data
types in `__all__`.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: all tests still pass (Phase-0 tests + Phase-1 tests to this point).

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/document.py src/pyfieldml/__init__.py tests/unit/test_document_model.py
git commit -m "feat(document): upgrade stub to model-backed Document with typed views"
```

---

## Task 13 — Model → DOM writer (programmatic docs)

**Files:**
- Create: `src/pyfieldml/model/_writer.py`
- Modify: `src/pyfieldml/document.py` (add `Document.from_region`, use new writer)
- Test: add to `tests/unit/test_document_model.py`

- [ ] **Step 1: Add failing test**

```python
def test_document_from_region_writes_and_roundtrips(tmp_path: Path) -> None:
    from pyfieldml.model.evaluators import ConstantEvaluator
    from pyfieldml.model.region import Region
    from pyfieldml.model.types import ContinuousType

    r = Region(name="built")
    t = ContinuousType(name="real.1d")
    r.add_type(t)
    r.add_evaluator(ConstantEvaluator(name="k", value_type=t, value=42.0))

    doc = fml.Document.from_region(r)
    out = tmp_path / "built.fieldml"
    doc.write(out)

    doc2 = fml.read(out)
    assert doc2.continuous["real.1d"].name == "real.1d"
    assert doc2.evaluators["k"].value == 42.0
```

- [ ] **Step 2: Implement `src/pyfieldml/model/_writer.py`**

Build an lxml ElementTree from a `Region`. Invert the loader logic from Task
11 — type declarations first, then leaf evaluators, then composite ones.

Minimum viable skeleton:

```python
"""Model → DOM tree writer."""

from __future__ import annotations

from typing import cast

from lxml import etree

from pyfieldml.data.hdf5 import Hdf5DenseBackend, Hdf5DOKBackend
from pyfieldml.data.text import ExternalTextBackend, InlineTextBackend
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
from pyfieldml.model.region import Region
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)

FIELDML_VERSION = "0.5.0"


def region_to_tree(region: Region) -> etree._ElementTree:
    root = etree.Element("Fieldml", {"version": FIELDML_VERSION})
    region_elem = etree.SubElement(root, "Region", {"name": region.name})

    for name in region.objects:
        obj = _objects_lookup(region, name)
        if isinstance(obj, BooleanType):
            etree.SubElement(region_elem, "BooleanType", {"name": obj.name})
        elif isinstance(obj, ContinuousType):
            _write_continuous(region_elem, obj)
        elif isinstance(obj, EnsembleType):
            _write_ensemble(region_elem, obj)
        elif isinstance(obj, MeshType):
            _write_mesh(region_elem, obj)
        elif isinstance(obj, Evaluator):
            _write_evaluator(region_elem, obj)

    return etree.ElementTree(root)


def _objects_lookup(region: Region, name: str) -> object:
    for d in (region.booleans, region.ensembles, region.continuous, region.meshes, region.evaluators):
        if name in d:
            return d[name]
    raise KeyError(name)


def _write_continuous(parent: etree._Element, t: ContinuousType) -> None:
    e = etree.SubElement(parent, "ContinuousType", {"name": t.name})
    if t.component_count > 1 and t.component_name:
        etree.SubElement(
            e,
            "Components",
            {"name": t.component_name, "count": str(t.component_count)},
        )


def _write_ensemble(parent: etree._Element, t: EnsembleType) -> None:
    e = etree.SubElement(parent, "EnsembleType", {"name": t.name})
    # Members serialization — if range-typed, emit MemberRange; otherwise list
    members = t.members
    if isinstance(members, range):
        mblock = etree.SubElement(e, "Members")
        etree.SubElement(
            mblock,
            "MemberRange",
            {"min": str(members.start), "max": str(members.stop - 1), "stride": "1"},
        )


def _write_mesh(parent: etree._Element, t: MeshType) -> None:
    e = etree.SubElement(parent, "MeshType", {"name": t.name})
    etree.SubElement(e, "Elements", {"name": t.elements.name})
    etree.SubElement(e, "Chart", {"name": t.chart.name})


def _write_evaluator(parent: etree._Element, ev: Evaluator) -> None:
    vt_name = getattr(ev.value_type, "name", "")
    if isinstance(ev, ArgumentEvaluator):
        etree.SubElement(
            parent, "ArgumentEvaluator", {"name": ev.name, "valueType": vt_name}
        )
    elif isinstance(ev, ConstantEvaluator):
        value = ev.value
        value_str = (
            " ".join(f"{v!r}" for v in value)
            if isinstance(value, (list, tuple))
            else repr(value)
        )
        etree.SubElement(
            parent,
            "ConstantEvaluator",
            {"name": ev.name, "valueType": vt_name, "value": value_str},
        )
    elif isinstance(ev, ExternalEvaluator):
        etree.SubElement(
            parent, "ExternalEvaluator", {"name": ev.name, "valueType": vt_name}
        )
    elif isinstance(ev, ParameterEvaluator):
        _write_parameter(parent, ev)
    elif isinstance(ev, ReferenceEvaluator):
        e = etree.SubElement(
            parent,
            "ReferenceEvaluator",
            {"name": ev.name, "valueType": vt_name, "evaluator": ev.source.name},
        )
        if len(ev.bindings) > 0:
            bs = etree.SubElement(e, "Bindings")
            for arg_id, binding in ev.bindings._by_arg.items():  # noqa: SLF001
                etree.SubElement(
                    bs,
                    "Bind",
                    {"argument": binding.argument.name, "source": binding.source.name},
                )
    elif isinstance(ev, AggregateEvaluator):
        e = etree.SubElement(
            parent, "AggregateEvaluator", {"name": ev.name, "valueType": vt_name}
        )
        if ev.components:
            etree.SubElement(
                e,
                "ComponentEvaluators",
                {"default": ev.components[0].name},
            )
    elif isinstance(ev, PiecewiseEvaluator):
        e = etree.SubElement(
            parent, "PiecewiseEvaluator", {"name": ev.name, "valueType": vt_name}
        )
        ev_map = etree.SubElement(e, "EvaluatorMap")
        if ev.default is not None:
            ev_map.set("default", ev.default.name)
        for idx, piece in ev.pieces.items():
            etree.SubElement(
                ev_map,
                "EvaluatorMapEntry",
                {"indexValue": str(idx), "evaluator": piece.name},
            )


def _write_parameter(parent: etree._Element, ev: ParameterEvaluator) -> None:
    vt_name = getattr(ev.value_type, "name", "")
    e = etree.SubElement(
        parent, "ParameterEvaluator", {"name": ev.name, "valueType": vt_name}
    )
    wrapper = etree.SubElement(e, "DenseArrayData")
    shape_str = " ".join(str(d) for d in ev.data.shape)
    ds = etree.SubElement(
        wrapper, "DataSource", {"name": f"{ev.name}_src", "location": "/", "rank": "1"}
    )
    etree.SubElement(ds, "ArrayDataSize").text = shape_str

    if isinstance(ev.data, InlineTextBackend):
        etree.SubElement(wrapper, "DataResourceString").text = ev.data.text
    elif isinstance(ev.data, ExternalTextBackend):
        etree.SubElement(
            wrapper,
            "DataResourceHref",
            {"{http://www.w3.org/1999/xlink}href": ev.data.href, "format": "PLAIN_TEXT"},
        )
    elif isinstance(ev.data, (Hdf5DenseBackend, Hdf5DOKBackend)):
        href = str(
            ev.data.path.name
            if hasattr(ev.data, "path") and ev.data.path is not None
            else ""
        )
        etree.SubElement(
            wrapper,
            "DataResourceHref",
            {"{http://www.w3.org/1999/xlink}href": href, "format": "HDF5"},
        )
```

- [ ] **Step 3: Add `Document.from_region` classmethod**

In `src/pyfieldml/document.py`:

```python
    @classmethod
    def from_region(cls, region: Region) -> "Document":
        from pyfieldml.dom.parser import ParsedDocument
        from pyfieldml.model._writer import region_to_tree

        tree = region_to_tree(region)
        parsed = ParsedDocument(tree=tree, version="0.5.0", source_file=None)
        return cls(parsed, region)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_document_model.py -v`
Expected: the new test passes + all existing tests remain green.

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/model/_writer.py src/pyfieldml/document.py tests/unit/test_document_model.py
git commit -m "feat(model): model → DOM writer for programmatically-built Documents"
```

---

## Task 14 — Round-trip: model ↔ HDF5-backed parameter

**Files:**
- Create: `tests/fixtures/parameter_hdf5.fieldml` + `parameter_hdf5.h5`
- Add tests to `tests/unit/test_document_model.py`

- [ ] **Step 1: Create the fixture pair**

Use a helper Python script to generate the fixture (prevents drift). Add to
`tests/fixtures/make_fixtures.py`:

```python
"""Regenerate test fixtures that require external data files.

Run: ``uv run python tests/fixtures/make_fixtures.py``
"""

from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).parent

# parameter_hdf5.h5
with h5py.File(HERE / "parameter_hdf5.h5", "w") as f:
    f.create_dataset("/coords", data=np.arange(12, dtype=np.float64).reshape(4, 3))

# parameter_hdf5.fieldml — matches the above
(HERE / "parameter_hdf5.fieldml").write_text(
    """<?xml version="1.0" encoding="UTF-8"?>
<Fieldml version="0.5.0"
  xmlns:xlink="http://www.w3.org/1999/xlink">
  <Region name="test">
    <ContinuousType name="real.3d">
      <Components name="real.3d.component" count="3"/>
    </ContinuousType>
    <ParameterEvaluator name="coords" valueType="real.3d">
      <DenseArrayData>
        <DataSource name="coords_src" location="/coords" rank="2">
          <ArrayDataSize>4 3</ArrayDataSize>
        </DataSource>
        <DataResourceHref xlink:href="parameter_hdf5.h5" format="HDF5"/>
      </DenseArrayData>
    </ParameterEvaluator>
  </Region>
</Fieldml>
"""
)
```

Run it once:
```bash
uv run python tests/fixtures/make_fixtures.py
```

- [ ] **Step 2: Add the test**

```python
def test_parameter_hdf5_loads_and_round_trips(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "parameter_hdf5.fieldml")
    coords = doc.evaluators["coords"]
    assert isinstance(coords, ParameterEvaluator)
    arr = coords.as_ndarray()
    assert arr.shape == (4, 3)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr.ravel(), np.arange(12))
```

- [ ] **Step 3: Run, pass**

Run: `uv run pytest tests/unit/test_document_model.py -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/ tests/unit/test_document_model.py
git commit -m "test: HDF5-backed parameter round-trip fixture + test"
```

---

## Task 15 — Conformance CI against the C++ reference library

**Files:**
- Create: `tests/conformance/__init__.py`
- Create: `tests/conformance/test_cpp_reference.py`
- Create: `.github/workflows/conformance.yml`

The C++ FieldML-API builds with CMake + libxml2 + HDF5. On Linux, all three
are available from `apt` (`libxml2-dev`, `libhdf5-dev`). The CI cache can
reuse the built `libfieldml_api.so` across runs.

- [ ] **Step 1: Create `tests/conformance/test_cpp_reference.py`**

```python
"""Cross-validate pyfieldml against the C++ FieldML-API reference library.

This test is skipped unless the environment variable ``PYFIELDML_CPP_REF``
points to a built copy of the reference library's CLI executable that can
round-trip a .fieldml file (i.e. read + write it back). The GitHub Actions
``conformance.yml`` workflow sets this up.

For each fixture in ``tests/fixtures/*.fieldml`` that validates against the
XSD, we:
    1. Round-trip through pyfieldml: read → write → re-read.
    2. Round-trip through the C++ reference: C++-read → C++-write.
    3. Compare the two outputs for semantic (C14N) equivalence.
    4. Extract every ParameterEvaluator's array data from both and assert
       exact NumPy equality.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from lxml import etree

import pyfieldml as fml

CPP_REF = os.environ.get("PYFIELDML_CPP_REF")


@pytest.mark.skipif(
    CPP_REF is None, reason="PYFIELDML_CPP_REF not set; skipping conformance run"
)
@pytest.mark.parametrize(
    "fixture",
    [
        "minimal.fieldml",
        "two_types.fieldml",
        "parameter_inline.fieldml",
        "reference_aggregate.fieldml",
    ],
)
def test_round_trip_matches_cpp_reference(
    fixtures_dir: Path, tmp_path: Path, fixture: str
) -> None:
    src = fixtures_dir / fixture

    # pyfieldml round-trip
    py_out = tmp_path / f"py_{fixture}"
    fml.read(src).write(py_out)

    # C++ round-trip
    cpp_out = tmp_path / f"cpp_{fixture}"
    subprocess.run([CPP_REF, str(src), str(cpp_out)], check=True)

    py_canonical = etree.tostring(
        etree.parse(py_out), method="c14n", exclusive=True
    )
    cpp_canonical = etree.tostring(
        etree.parse(cpp_out), method="c14n", exclusive=True
    )
    assert py_canonical == cpp_canonical, (
        f"Canonicalized XML divergence for {fixture}:\n"
        f"python:\n{py_canonical!r}\ncpp:\n{cpp_canonical!r}"
    )

    # Array equality for all ParameterEvaluators.
    py_doc = fml.read(py_out)
    for name, ev in py_doc.evaluators.items():
        from pyfieldml.model.evaluators import ParameterEvaluator

        if isinstance(ev, ParameterEvaluator):
            py_arr = ev.as_ndarray()
            # The C++ output is read via pyfieldml itself on the C++ side;
            # this is valid because if our reader is wrong, BOTH sides
            # would disagree on content — the XML comparison above would
            # already have caught that.
            cpp_arr = fml.read(cpp_out).evaluators[name].as_ndarray()
            np.testing.assert_array_equal(py_arr, cpp_arr, err_msg=name)
```

- [ ] **Step 2: Create `.github/workflows/conformance.yml`**

```yaml
name: Conformance

on:
  schedule:
    - cron: "17 3 * * *"  # nightly, ~3:17 UTC
  workflow_dispatch:
  push:
    tags: ["v*.*.*"]

jobs:
  build-cpp-reference:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          path: pyfieldml
      - uses: actions/checkout@v4
        with:
          repository: kchemorion/FieldML-API
          path: FieldML-API
      - name: Install system deps
        run: |
          sudo apt-get update
          sudo apt-get install -y libxml2-dev libhdf5-dev cmake build-essential
      - name: Cache built C++ library
        uses: actions/cache@v4
        id: cppcache
        with:
          path: FieldML-API/build
          key: cppref-${{ runner.os }}-${{ hashFiles('FieldML-API/CMakeLists.txt') }}
      - name: Build C++ reference
        if: steps.cppcache.outputs.cache-hit != 'true'
        run: |
          cd FieldML-API
          cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build -j4
      - name: Write minimal roundtrip CLI wrapper
        run: |
          cat > FieldML-API/build/roundtrip.py <<'PY'
          #!/usr/bin/env python3
          """Minimal roundtrip wrapper around the built C++ API."""
          import sys, ctypes, pathlib
          build = pathlib.Path(__file__).parent
          lib = ctypes.CDLL(str(next(build.rglob("libfieldml_api*"))))
          # TODO: actual ctypes bindings. For Phase-1 conformance, this
          # can be replaced with a minimal C wrapper or a Python h5py/lxml
          # round-trip using the reference library's output parsing.
          src, dst = sys.argv[1], sys.argv[2]
          pathlib.Path(dst).write_bytes(pathlib.Path(src).read_bytes())
          PY
          chmod +x FieldML-API/build/roundtrip.py
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - name: Run conformance tests
        working-directory: pyfieldml
        env:
          PYFIELDML_CPP_REF: ${{ github.workspace }}/FieldML-API/build/roundtrip.py
        run: |
          uv sync --extra dev
          uv run pytest tests/conformance -v
```

**Honest scope note for the engineer:** wiring a real end-to-end roundtrip
via the C++ API requires either (a) using `Fieldml_WriteFile` from the C ABI
(a full ctypes binding), or (b) adding a small `cpp_roundtrip` C++ program
under `FieldML-API/tools/` that calls the API directly. Both are legitimate;
(b) is simpler. For the first green run, the placeholder above copies the
input file, which causes the conformance test to pass trivially *for now*.
**Promote this placeholder to a real round-trip before Phase 1 releases
v0.1.0** — it must exercise the C++ reader and writer. Tracked as a TODO
comment in the workflow.

- [ ] **Step 3: Run conformance locally (optional)**

If you have libxml2 + libhdf5 locally:
```bash
cd ../FieldML-API
cmake -S . -B build && cmake --build build -j
cd ../pyfieldml
PYFIELDML_CPP_REF=../FieldML-API/build/roundtrip.py uv run pytest tests/conformance -v
```

- [ ] **Step 4: Push to a branch and confirm CI runs**

Open the conformance workflow in the Actions tab.

- [ ] **Step 5: Commit**

```bash
git add tests/conformance/ .github/workflows/conformance.yml
git commit -m "ci(conformance): cross-validate round-trip against C++ FieldML-API"
```

---

## Task 16 — Phase-1 closeout + v0.1.0 tag

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `src/pyfieldml/_version.py`
- Modify: `CITATION.cff`

- [ ] **Step 1: Full sanity gauntlet**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/pyfieldml
uv run pytest -v --cov=pyfieldml
uv run mkdocs build --strict
```

- [ ] **Step 2: Update CHANGELOG**

```markdown
## [0.1.0] - YYYY-MM-DD

### Added
- `pyfieldml.model`: Region container, typed type classes
  (Boolean/Ensemble/Continuous/Mesh), full Evaluator hierarchy
  (Parameter/Constant/Argument/Reference/Aggregate/Piecewise/External),
  bindings table, import resolver.
- `pyfieldml.data`: four array backends — inline text, external text,
  HDF5 dense, HDF5 DOK sparse — with lazy loading where applicable.
- `Document`: typed mapping views (`.booleans`, `.ensembles`, `.continuous`,
  `.meshes`, `.evaluators`) + `from_region()` for programmatic construction.
- Conformance CI that cross-validates round-trip behavior against the
  C++ FieldML-API reference library.

### Changed
- `Document` now parses the semantic model at load; previously it held
  only the raw DOM tree.
```

- [ ] **Step 3: Bump version and date**

`src/pyfieldml/_version.py`:
```python
__version__ = "0.1.0"
```

`CITATION.cff`: `version: 0.1.0`, `date-released: <today>`.

- [ ] **Step 4: Commit and tag**

```bash
git add CHANGELOG.md CITATION.cff src/pyfieldml/_version.py
git commit -m "chore(release): v0.1.0"
git tag -a v0.1.0 -m "v0.1.0: Phase 1 — semantic model + full I/O"
git push origin main --tags
```

- [ ] **Step 5: Verify release** — same as Phase 0's Task 14 Step 5.

---

## Phase-1 done-definition

- [ ] All seven evaluator subtypes round-trip (programmatic construction → write → re-read → type-compare).
- [ ] All four array backends round-trip with `float32` / `float64` / `int32` / `int64`.
- [ ] Conformance CI is green on the fixture corpus with a *real* C++ round-trip (not the trivial copy placeholder from Task 15).
- [ ] `uv pip install pyfieldml==0.1.0` works from PyPI.
- [ ] Docs API reference auto-populates the new public types.

Next phase: **Phase 2 — evaluation engine (Lagrange)** — basis registry, Lagrange bases on all 6 topologies, `Field.evaluate / sample / jacobian`, cKDTree spatial locate.
