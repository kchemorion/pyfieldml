# pyfieldml Phase 0 — Scaffolding + DOM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working repo scaffold + CI + a minimal `pyfieldml.dom` layer (parse, XSD-validate, write) such that the smallest valid FieldML document round-trips end-to-end via `pyfieldml.read(...).write(...)`.

**Architecture:** `src/`-layout Python package built with hatchling. The `dom/` subpackage is a thin bijection over lxml's XML tree — it does not interpret FieldML semantics (that's Phase 1's `model/` layer). Bundled resources (XSD + standard library) ship inside the wheel via `importlib.resources`. CI runs on Linux/macOS/Windows × Python 3.10-3.13.

**Tech Stack:** Python 3.10+, hatchling, lxml, pytest, ruff, mypy, pre-commit, GitHub Actions, mkdocs-material.

**Spec reference:** `docs/superpowers/specs/2026-04-18-pyfieldml-design.md` §2.2, §3 (package layout), §4.1 (public API), §7 (tooling).

**Phase 0 completion criteria:**
- `uv run pytest` passes on all unit tests.
- `uv run pyfieldml-smoke` (or equivalent) reads `tests/fixtures/minimal.fieldml`, writes it back, and the output validates.
- `ruff check`, `ruff format --check`, `mypy --strict src/pyfieldml` all clean.
- CI matrix green on a pushed branch.
- Docs site builds locally (`mkdocs build`).
- `pyfieldml` is installable from a local checkout: `uv pip install -e .` works.

---

## File structure created in this phase

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` | Build backend (hatchling), required + extras deps, ruff/mypy/pytest config |
| `src/pyfieldml/__init__.py` | Public top-level API: `read`, `validate`, `__version__`, `Document` re-export |
| `src/pyfieldml/_version.py` | `__version__ = "0.0.1"` (hatchling-managed in later phases) |
| `src/pyfieldml/errors.py` | Error hierarchy rooted at `FieldMLError` |
| `src/pyfieldml/document.py` | Phase-0 `Document` stub — holds the parsed DOM tree, exposes `.write()` and `.source_version` |
| `src/pyfieldml/dom/__init__.py` | DOM subpackage public surface |
| `src/pyfieldml/dom/parser.py` | `parse_file`, `parse_string` → `lxml.etree._ElementTree`; detects FieldML version from root `@version` |
| `src/pyfieldml/dom/writer.py` | `write_file`, `write_string` — UTF-8, XML declaration, stable indentation |
| `src/pyfieldml/dom/validator.py` | `validate_tree(tree)` — XSD validation against bundled schema |
| `src/pyfieldml/library/__init__.py` | `get_xsd_path()`, `get_stdlib_path()` — `importlib.resources` accessors |
| `src/pyfieldml/library/FieldML_0.5.xsd` | Bundled XSD (copied from reference repo) |
| `src/pyfieldml/library/FieldML_Library_0.5.xml` | Bundled standard library (copied from reference repo) |
| `src/pyfieldml/py.typed` | PEP 561 typing marker |
| `tests/__init__.py` | Empty (tests as a package for relative imports if needed) |
| `tests/conftest.py` | `fixtures_dir` pytest fixture |
| `tests/fixtures/minimal.fieldml` | Smallest valid FieldML 0.5 doc — one Region, one BooleanType |
| `tests/fixtures/two_types.fieldml` | Region with one BooleanType + one ContinuousType |
| `tests/fixtures/invalid_no_name.fieldml` | Malformed (Region with no `name` attr) — must fail XSD validation |
| `tests/fixtures/not_fieldml.xml` | Arbitrary non-FieldML XML — must fail parse/detect |
| `tests/unit/test_errors.py` | Error hierarchy tests |
| `tests/unit/test_library.py` | Bundled-resource accessor tests |
| `tests/unit/test_dom_parser.py` | Parser tests |
| `tests/unit/test_dom_validator.py` | Validator tests |
| `tests/unit/test_dom_writer.py` | Writer tests |
| `tests/unit/test_document.py` | Document-stub tests |
| `tests/unit/test_public_api.py` | End-to-end round-trip via top-level `pyfieldml.read/validate` |
| `.github/workflows/ci.yml` | Matrix CI: ruff, mypy, pytest |
| `.github/workflows/docs.yml` | Build + deploy mkdocs on main |
| `.github/workflows/release.yml` | Skeleton: build wheel + sdist, publish to PyPI via OIDC (triggered by version tag) |
| `.github/ISSUE_TEMPLATE/bug.md` | Bug-report template |
| `.github/ISSUE_TEMPLATE/feature.md` | Feature-request template |
| `.github/pull_request_template.md` | PR template |
| `.pre-commit-config.yaml` | ruff, mypy, nbstripout hooks |
| `mkdocs.yml` | Docs site config — mkdocs-material, mkdocstrings |
| `docs/index.md` | Landing page |
| `docs/getting-started.md` | Install + first-read walkthrough |
| `docs/api/index.md` | API reference stub (mkdocstrings auto-populates later) |
| `CONTRIBUTING.md` | Dev setup, test commands, style guide |
| `CODE_OF_CONDUCT.md` | Contributor Covenant v2.1 |
| `CHANGELOG.md` | Keep-a-Changelog skeleton |
| `MAINTAINING.md` | Release checklist |

Files modified:
- `README.md` — add install + quickstart snippet once public API works.

---

## Task 1 — Package scaffolding (pyproject.toml + directory tree)

**Files:**
- Create: `pyproject.toml`
- Create: `src/pyfieldml/__init__.py`
- Create: `src/pyfieldml/_version.py`
- Create: `src/pyfieldml/py.typed`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

Write `pyproject.toml` at repo root:

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "pyfieldml"
dynamic = ["version"]
description = "A modern pure-Python implementation of FieldML 0.5 with evaluation engine and biomechanics model zoo."
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE"]
authors = [
    { name = "Francis Chemorion" },
]
keywords = ["fieldml", "finite-element", "computational-physiology", "biomechanics", "physiome"]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
]
dependencies = [
    "lxml>=5.0",
    "numpy>=1.26",
    "h5py>=3.10",
    "scipy>=1.11",
]

[project.urls]
Homepage = "https://github.com/kchemorion/pyfieldml"
Repository = "https://github.com/kchemorion/pyfieldml"
Issues = "https://github.com/kchemorion/pyfieldml/issues"

[project.optional-dependencies]
viz = [
    "pyvista>=0.43",
    "ipywidgets>=8",
    "trame-jupyter-extension",
    "ipycytoscape>=1.3",
]
meshio = ["meshio>=5.3"]
scikit-fem = ["scikit-fem>=10"]
opensim = ["meshio>=5.3"]
all = [
    "pyfieldml[viz,meshio,scikit-fem,opensim]",
]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "nbmake>=1.5",
    "ruff>=0.6",
    "mypy>=1.11",
    "pre-commit>=3.7",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.26",
    "mkdocs-jupyter>=0.25",
    "jupyter>=1",
]

[tool.hatch.version]
path = "src/pyfieldml/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["src/pyfieldml"]

[tool.hatch.build.targets.wheel.force-include]
"src/pyfieldml/library/FieldML_0.5.xsd" = "pyfieldml/library/FieldML_0.5.xsd"
"src/pyfieldml/library/FieldML_Library_0.5.xml" = "pyfieldml/library/FieldML_Library_0.5.xml"

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "F", "W",     # pycodestyle + pyflakes
    "I",               # isort
    "UP",              # pyupgrade
    "B",               # bugbear
    "SIM",             # simplify
    "RUF",             # ruff-specific
    "N",               # pep8-naming
    "D",               # pydocstyle (selective)
]
ignore = [
    "D100", "D101", "D102", "D103", "D104", "D105", "D107",  # missing docstrings — soft-enforce in later phases
    "D203", "D213",                                           # conflicting docstring conventions
]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D"]

[tool.mypy]
python_version = "3.10"
strict = true
files = ["src/pyfieldml", "tests"]
plugins = []

[[tool.mypy.overrides]]
module = ["lxml.*", "h5py.*", "scipy.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "8"
addopts = "-ra --strict-markers --strict-config"
testpaths = ["tests"]
pythonpath = ["src"]

[tool.coverage.run]
source = ["src/pyfieldml"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
]
```

- [ ] **Step 2: Create source tree and marker files**

Run:

```bash
mkdir -p src/pyfieldml/dom src/pyfieldml/library tests/unit tests/fixtures
touch src/pyfieldml/py.typed tests/__init__.py
```

- [ ] **Step 3: Write `src/pyfieldml/_version.py`**

```python
"""Single source of truth for the package version. Consumed by hatchling."""

__version__ = "0.0.1"
```

- [ ] **Step 4: Write `src/pyfieldml/__init__.py` (placeholder — tasks below fill it in)**

```python
"""pyfieldml: a pure-Python implementation of FieldML 0.5."""

from pyfieldml._version import __version__

__all__ = ["__version__"]
```

- [ ] **Step 5: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    """Absolute path to the `tests/fixtures/` directory."""
    return FIXTURES_DIR
```

- [ ] **Step 6: Install the package in editable mode and confirm import works**

Run:

```bash
uv sync --extra dev
uv run python -c "import pyfieldml; print(pyfieldml.__version__)"
```

Expected stdout: `0.0.1`

If `uv` is not available, use `python -m pip install -e ".[dev]"` then `python -c ...`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "chore(scaffold): add pyproject, src layout, and test tree"
```

---

## Task 2 — Error hierarchy

**Files:**
- Create: `src/pyfieldml/errors.py`
- Test: `tests/unit/test_errors.py`

- [ ] **Step 1: Write the failing test — `tests/unit/test_errors.py`**

```python
"""Tests for the pyfieldml error hierarchy."""

from __future__ import annotations

import pytest

from pyfieldml.errors import (
    BasisNotFoundError,
    EvaluationError,
    FieldMLError,
    FieldMLParseError,
    FieldMLValidationError,
    UnboundArgumentError,
    UnresolvedImportError,
)


@pytest.mark.parametrize(
    "cls",
    [
        FieldMLParseError,
        FieldMLValidationError,
        UnresolvedImportError,
        UnboundArgumentError,
        EvaluationError,
        BasisNotFoundError,
    ],
)
def test_all_errors_subclass_fieldml_error(cls: type[Exception]) -> None:
    assert issubclass(cls, FieldMLError)


def test_parse_error_carries_source_line() -> None:
    err = FieldMLParseError("bad tag", source_file="/tmp/x.fieldml", line=42)
    assert err.source_file == "/tmp/x.fieldml"
    assert err.line == 42
    assert "line 42" in str(err)


def test_validation_error_holds_multiple_issues() -> None:
    err = FieldMLValidationError(["issue A", "issue B"])
    assert len(err.issues) == 2
    assert "issue A" in str(err)
    assert "issue B" in str(err)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `uv run pytest tests/unit/test_errors.py -v`
Expected: ImportError — `pyfieldml.errors` does not exist.

- [ ] **Step 3: Implement `src/pyfieldml/errors.py`**

```python
"""Error hierarchy for pyfieldml. All library errors subclass ``FieldMLError``."""

from __future__ import annotations

from collections.abc import Iterable


class FieldMLError(Exception):
    """Base class for every error raised by pyfieldml."""


class FieldMLParseError(FieldMLError):
    """Raised when an XML document cannot be parsed as FieldML."""

    def __init__(
        self,
        message: str,
        *,
        source_file: str | None = None,
        line: int | None = None,
    ) -> None:
        self.source_file = source_file
        self.line = line
        suffix = ""
        if source_file is not None:
            suffix = f" (in {source_file}"
            suffix += f", line {line})" if line is not None else ")"
        elif line is not None:
            suffix = f" (line {line})"
        super().__init__(f"{message}{suffix}")


class FieldMLValidationError(FieldMLError):
    """Raised when a FieldML document fails XSD or semantic validation."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = list(issues)
        joined = "\n - ".join(self.issues) if self.issues else "(no details)"
        super().__init__(f"FieldML validation failed:\n - {joined}")


class UnresolvedImportError(FieldMLError):
    """Raised when a FieldML ``<Import>`` cannot be located."""


class UnboundArgumentError(FieldMLError):
    """Raised when an ArgumentEvaluator is evaluated without a binding."""


class EvaluationError(FieldMLError):
    """Raised when the evaluation engine cannot produce a result."""


class BasisNotFoundError(FieldMLError):
    """Raised when a referenced basis function is not registered."""
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/unit/test_errors.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/errors.py tests/unit/test_errors.py
git commit -m "feat(errors): add FieldMLError hierarchy with source-line context"
```

---

## Task 3 — Bundle library resources

**Files:**
- Copy: `FieldML-API/FieldML_0.5.xsd` → `src/pyfieldml/library/FieldML_0.5.xsd`
- Copy: `FieldML-API/FieldML_Library_0.5.xml` → `src/pyfieldml/library/FieldML_Library_0.5.xml`
- Create: `src/pyfieldml/library/__init__.py`
- Test: `tests/unit/test_library.py`

- [ ] **Step 1: Copy the XSD and standard library into the package**

Run (adjust source path if the reference clone sits elsewhere):

```bash
cp ../FieldML-API/FieldML_0.5.xsd src/pyfieldml/library/
cp ../FieldML-API/FieldML_Library_0.5.xml src/pyfieldml/library/
```

- [ ] **Step 2: Write the failing test — `tests/unit/test_library.py`**

```python
"""Tests for bundled FieldML resource accessors."""

from __future__ import annotations

from pyfieldml.library import get_stdlib_path, get_xsd_path


def test_xsd_exists_and_is_xml() -> None:
    path = get_xsd_path()
    assert path.is_file()
    with path.open("rb") as fh:
        header = fh.read(256)
    assert b"<xs:schema" in header


def test_stdlib_exists_and_is_fieldml() -> None:
    path = get_stdlib_path()
    assert path.is_file()
    with path.open("rb") as fh:
        header = fh.read(256)
    assert b'<Fieldml version="0.5' in header
```

- [ ] **Step 3: Run the test and confirm it fails**

Run: `uv run pytest tests/unit/test_library.py -v`
Expected: ImportError — `pyfieldml.library` does not expose those symbols yet.

- [ ] **Step 4: Implement `src/pyfieldml/library/__init__.py`**

```python
"""Accessors for bundled FieldML schema + standard library resources."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

_PACKAGE = "pyfieldml.library"
_XSD_FILENAME = "FieldML_0.5.xsd"
_STDLIB_FILENAME = "FieldML_Library_0.5.xml"


def get_xsd_path() -> Path:
    """Return the on-disk path to the bundled FieldML 0.5 XSD."""
    return _resource_path(_XSD_FILENAME)


def get_stdlib_path() -> Path:
    """Return the on-disk path to the bundled FieldML 0.5 standard library."""
    return _resource_path(_STDLIB_FILENAME)


def _resource_path(filename: str) -> Path:
    ref = resources.files(_PACKAGE).joinpath(filename)
    # ``as_file`` handles both loose-file and zipped-wheel installs.
    with resources.as_file(ref) as concrete:
        return Path(concrete)
```

- [ ] **Step 5: Run the test and confirm it passes**

Run: `uv run pytest tests/unit/test_library.py -v`
Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/library/ tests/unit/test_library.py
git commit -m "feat(library): bundle FieldML 0.5 XSD and standard library"
```

---

## Task 4 — DOM parser

**Files:**
- Create: `src/pyfieldml/dom/__init__.py`
- Create: `src/pyfieldml/dom/parser.py`
- Create: `tests/fixtures/minimal.fieldml`
- Create: `tests/fixtures/two_types.fieldml`
- Create: `tests/fixtures/not_fieldml.xml`
- Test: `tests/unit/test_dom_parser.py`

- [ ] **Step 1: Write fixtures**

`tests/fixtures/minimal.fieldml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Fieldml version="0.5.0">
  <Region name="test">
    <BooleanType name="boolean"/>
  </Region>
</Fieldml>
```

`tests/fixtures/two_types.fieldml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Fieldml version="0.5.0">
  <Region name="test">
    <BooleanType name="boolean"/>
    <ContinuousType name="real.1d"/>
  </Region>
</Fieldml>
```

`tests/fixtures/not_fieldml.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<SomeOtherRoot>
  <Foo bar="baz"/>
</SomeOtherRoot>
```

- [ ] **Step 2: Write the failing tests — `tests/unit/test_dom_parser.py`**

```python
"""Tests for the DOM parser layer."""

from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string
from pyfieldml.errors import FieldMLParseError


def test_parse_minimal_file_returns_parsed_document(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "minimal.fieldml")
    assert isinstance(parsed, ParsedDocument)
    assert parsed.version == "0.5.0"
    assert parsed.source_file == str(fixtures_dir / "minimal.fieldml")
    root = parsed.tree.getroot()
    assert root.tag == "Fieldml"
    assert root.find("Region").get("name") == "test"


def test_parse_two_types_preserves_child_order(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "two_types.fieldml")
    region = parsed.tree.getroot().find("Region")
    child_tags = [c.tag for c in region if isinstance(c.tag, str)]
    assert child_tags == ["BooleanType", "ContinuousType"]


def test_parse_string_matches_parse_file(fixtures_dir: Path) -> None:
    content = (fixtures_dir / "minimal.fieldml").read_bytes()
    parsed = parse_string(content)
    assert parsed.version == "0.5.0"
    assert parsed.source_file is None


def test_parse_rejects_non_fieldml_root(fixtures_dir: Path) -> None:
    with pytest.raises(FieldMLParseError, match="root element"):
        parse_file(fixtures_dir / "not_fieldml.xml")


def test_parse_reports_source_location_on_malformed_xml(tmp_path: Path) -> None:
    bad = tmp_path / "broken.fieldml"
    bad.write_text('<?xml version="1.0"?><Fieldml version="0.5.0"><Region>')
    with pytest.raises(FieldMLParseError) as exc:
        parse_file(bad)
    assert exc.value.source_file == str(bad)
    assert exc.value.line is not None


def test_parse_rejects_unknown_version(tmp_path: Path) -> None:
    bad = tmp_path / "future.fieldml"
    bad.write_text(
        '<?xml version="1.0"?><Fieldml version="9.9.9"><Region name="r"/></Fieldml>'
    )
    with pytest.raises(FieldMLParseError, match="Unsupported FieldML version"):
        parse_file(bad)


def test_parse_accepts_legacy_versions(tmp_path: Path) -> None:
    for v in ("0.3", "0.3.0", "0.4", "0.4.0"):
        legacy = tmp_path / f"legacy_{v}.fieldml"
        legacy.write_text(
            f'<?xml version="1.0"?><Fieldml version="{v}"><Region name="r"/></Fieldml>'
        )
        parsed = parse_file(legacy)
        assert parsed.version == v
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `uv run pytest tests/unit/test_dom_parser.py -v`
Expected: ImportError — `pyfieldml.dom.parser` does not exist.

- [ ] **Step 4: Implement `src/pyfieldml/dom/__init__.py`**

```python
"""DOM layer: XML ↔ lxml tree, versioning, XSD validation."""

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string
from pyfieldml.dom.writer import write_file, write_string

__all__ = [
    "ParsedDocument",
    "parse_file",
    "parse_string",
    "write_file",
    "write_string",
]
```

(Note: `writer` symbols will be added in Task 6. Temporarily omit those two names until Task 6; re-add them then.)

For now, keep the file as:

```python
"""DOM layer: XML ↔ lxml tree, versioning, XSD validation."""

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string

__all__ = ["ParsedDocument", "parse_file", "parse_string"]
```

- [ ] **Step 5: Implement `src/pyfieldml/dom/parser.py`**

```python
"""Parse FieldML XML documents into lxml ElementTrees.

This layer is a thin bijection: it validates the top-level shape (root element
is ``<Fieldml version="...">``) and records the version, but does not interpret
any FieldML semantics. Semantic parsing lives in ``pyfieldml.model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from lxml import etree

from pyfieldml.errors import FieldMLParseError

SUPPORTED_VERSIONS: frozenset[str] = frozenset(
    {"0.3", "0.3.0", "0.4", "0.4.0", "0.5", "0.5.0"}
)

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed-but-uninterpreted FieldML document.

    Attributes
    ----------
    tree
        The underlying lxml ElementTree.
    version
        The FieldML version declared on the root element (``@version``).
    source_file
        Absolute path as a string, or ``None`` when parsed from a bytes/string source.
    """

    tree: etree._ElementTree
    version: str
    source_file: str | None


def parse_file(path: PathLike) -> ParsedDocument:
    """Parse a FieldML document from an on-disk file."""
    path = Path(path)
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    try:
        tree = etree.parse(str(path), parser)
    except etree.XMLSyntaxError as exc:
        line = getattr(exc, "lineno", None)
        raise FieldMLParseError(
            f"Malformed XML: {exc.msg}", source_file=str(path), line=line
        ) from exc
    version = _validate_root(tree, source_file=str(path))
    return ParsedDocument(tree=tree, version=version, source_file=str(path))


def parse_string(content: str | bytes) -> ParsedDocument:
    """Parse a FieldML document from an in-memory string/bytes."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    try:
        root = etree.fromstring(content, parser)
    except etree.XMLSyntaxError as exc:
        line = getattr(exc, "lineno", None)
        raise FieldMLParseError(f"Malformed XML: {exc.msg}", line=line) from exc
    tree = etree.ElementTree(root)
    version = _validate_root(tree, source_file=None)
    return ParsedDocument(tree=tree, version=version, source_file=None)


def _validate_root(tree: etree._ElementTree, *, source_file: str | None) -> str:
    root = tree.getroot()
    if root.tag != "Fieldml":
        raise FieldMLParseError(
            f"Expected root element <Fieldml>, got <{root.tag}>",
            source_file=source_file,
            line=root.sourceline,
        )
    version = root.get("version")
    if version is None:
        raise FieldMLParseError(
            "Root <Fieldml> element is missing required @version attribute",
            source_file=source_file,
            line=root.sourceline,
        )
    if version not in SUPPORTED_VERSIONS:
        raise FieldMLParseError(
            f"Unsupported FieldML version {version!r}. "
            f"Supported: {sorted(SUPPORTED_VERSIONS)}",
            source_file=source_file,
            line=root.sourceline,
        )
    return version
```

- [ ] **Step 6: Run the parser tests and confirm they pass**

Run: `uv run pytest tests/unit/test_dom_parser.py -v`
Expected: 7 tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/pyfieldml/dom/ tests/unit/test_dom_parser.py tests/fixtures/
git commit -m "feat(dom): add XML parser with root-level validation and version detection"
```

---

## Task 5 — XSD validator

**Files:**
- Create: `src/pyfieldml/dom/validator.py`
- Create: `tests/fixtures/invalid_no_name.fieldml`
- Test: `tests/unit/test_dom_validator.py`
- Modify: `src/pyfieldml/dom/__init__.py` (re-export `validate_tree`)

- [ ] **Step 1: Write the failing fixture — `tests/fixtures/invalid_no_name.fieldml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Fieldml version="0.5.0">
  <Region>
    <BooleanType name="boolean"/>
  </Region>
</Fieldml>
```

(Note: `Region` has no required `name` attribute — XSD must reject.)

- [ ] **Step 2: Write the failing tests — `tests/unit/test_dom_validator.py`**

```python
"""Tests for XSD validation of FieldML documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyfieldml.dom.parser import parse_file
from pyfieldml.dom.validator import validate_tree
from pyfieldml.errors import FieldMLValidationError


def test_minimal_document_validates(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "minimal.fieldml")
    validate_tree(parsed.tree)  # must not raise


def test_bundled_stdlib_validates() -> None:
    from pyfieldml.library import get_stdlib_path

    parsed = parse_file(get_stdlib_path())
    validate_tree(parsed.tree)  # must not raise


def test_missing_required_attribute_fails(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "invalid_no_name.fieldml")
    with pytest.raises(FieldMLValidationError) as exc:
        validate_tree(parsed.tree)
    joined = "\n".join(exc.value.issues)
    assert "name" in joined.lower()
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `uv run pytest tests/unit/test_dom_validator.py -v`
Expected: ImportError — `pyfieldml.dom.validator` does not exist.

- [ ] **Step 4: Implement `src/pyfieldml/dom/validator.py`**

```python
"""XSD validation of FieldML 0.5 documents against the bundled schema."""

from __future__ import annotations

from functools import lru_cache

from lxml import etree

from pyfieldml.errors import FieldMLValidationError
from pyfieldml.library import get_xsd_path


@lru_cache(maxsize=1)
def _schema() -> etree.XMLSchema:
    """Load and cache the FieldML 0.5 XSD."""
    with get_xsd_path().open("rb") as fh:
        schema_doc = etree.parse(fh)
    return etree.XMLSchema(schema_doc)


def validate_tree(tree: etree._ElementTree) -> None:
    """Validate an lxml ElementTree against the bundled FieldML 0.5 XSD.

    Raises
    ------
    FieldMLValidationError
        If the document fails schema validation. ``.issues`` holds one entry
        per schema error, each including source line number when available.
    """
    schema = _schema()
    if schema.validate(tree):
        return
    issues = [
        f"{error.filename or '<input>'}:{error.line}: {error.message}"
        for error in schema.error_log
    ]
    raise FieldMLValidationError(issues)
```

- [ ] **Step 5: Re-export from `src/pyfieldml/dom/__init__.py`**

Replace the file contents with:

```python
"""DOM layer: XML ↔ lxml tree, versioning, XSD validation."""

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string
from pyfieldml.dom.validator import validate_tree

__all__ = ["ParsedDocument", "parse_file", "parse_string", "validate_tree"]
```

- [ ] **Step 6: Run the validator tests and confirm they pass**

Run: `uv run pytest tests/unit/test_dom_validator.py -v`
Expected: 3 tests pass.

Note: if the bundled-stdlib test fails because the XSD imports `xlink` from a remote URL, add an `xmlparser` that disables network access and pre-register the xlink XSD. In practice the FieldML XSD's `xs:import` of xlink resolves via lxml's default behavior; if CI fails on network-restricted runners, adjust by embedding a local `xlink-href.xsd` mirror under `src/pyfieldml/library/` and supplying it via `etree.XMLParser` + `lxml.etree.parse` with `load_dtd=False`. Verify in CI; fix if it breaks.

- [ ] **Step 7: Commit**

```bash
git add src/pyfieldml/dom/validator.py src/pyfieldml/dom/__init__.py \
        tests/unit/test_dom_validator.py tests/fixtures/invalid_no_name.fieldml
git commit -m "feat(dom): XSD-validate FieldML documents against bundled 0.5 schema"
```

---

## Task 6 — DOM writer (round-trip-preserving)

**Files:**
- Create: `src/pyfieldml/dom/writer.py`
- Test: `tests/unit/test_dom_writer.py`
- Modify: `src/pyfieldml/dom/__init__.py` (re-export `write_file`, `write_string`)

- [ ] **Step 1: Write the failing tests — `tests/unit/test_dom_writer.py`**

```python
"""Tests for the DOM writer."""

from __future__ import annotations

from pathlib import Path

from pyfieldml.dom.parser import parse_file, parse_string
from pyfieldml.dom.validator import validate_tree
from pyfieldml.dom.writer import write_file, write_string


def test_round_trip_minimal_produces_valid_output(fixtures_dir: Path, tmp_path: Path) -> None:
    parsed = parse_file(fixtures_dir / "minimal.fieldml")
    out = tmp_path / "out.fieldml"
    write_file(parsed.tree, out)
    reparsed = parse_file(out)
    validate_tree(reparsed.tree)
    assert reparsed.version == "0.5.0"


def test_round_trip_preserves_child_count_and_order(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    parsed = parse_file(fixtures_dir / "two_types.fieldml")
    out = tmp_path / "roundtrip.fieldml"
    write_file(parsed.tree, out)
    reparsed = parse_file(out)
    original_tags = [c.tag for c in parsed.tree.getroot().find("Region")]
    roundtrip_tags = [c.tag for c in reparsed.tree.getroot().find("Region")]
    assert original_tags == roundtrip_tags


def test_write_string_returns_bytes_with_declaration() -> None:
    parsed = parse_string(
        b'<?xml version="1.0"?><Fieldml version="0.5.0">'
        b'<Region name="r"><BooleanType name="b"/></Region></Fieldml>'
    )
    output = write_string(parsed.tree)
    assert isinstance(output, bytes)
    assert output.startswith(b"<?xml")
    assert b"<Fieldml" in output


def test_write_file_uses_utf8_and_trailing_newline(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    parsed = parse_file(fixtures_dir / "minimal.fieldml")
    out = tmp_path / "out.fieldml"
    write_file(parsed.tree, out)
    raw = out.read_bytes()
    assert raw.startswith(b"<?xml")
    assert b'encoding="UTF-8"' in raw[:80] or b"encoding='UTF-8'" in raw[:80]
    assert raw.endswith(b"\n")
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/unit/test_dom_writer.py -v`
Expected: ImportError — `pyfieldml.dom.writer` does not exist.

- [ ] **Step 3: Implement `src/pyfieldml/dom/writer.py`**

```python
"""Serialize FieldML ElementTrees back to XML.

Output policy: UTF-8, XML declaration present, two-space indentation, trailing
newline. Attribute order follows lxml's stable insertion order, which matches
the input document on round-trip. Whitespace-only text nodes are preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from lxml import etree

PathLike = Union[str, Path]


def write_file(tree: etree._ElementTree, path: PathLike) -> None:
    """Serialize ``tree`` to ``path`` as UTF-8 XML with a trailing newline."""
    payload = _serialize(tree)
    Path(path).write_bytes(payload)


def write_string(tree: etree._ElementTree) -> bytes:
    """Serialize ``tree`` to UTF-8 bytes (with XML declaration)."""
    return _serialize(tree)


def _serialize(tree: etree._ElementTree) -> bytes:
    payload = etree.tostring(
        tree,
        xml_declaration=True,
        encoding="UTF-8",
        pretty_print=True,
        standalone=None,
    )
    if not payload.endswith(b"\n"):
        payload += b"\n"
    return payload
```

- [ ] **Step 4: Re-export from `src/pyfieldml/dom/__init__.py`**

Replace contents with:

```python
"""DOM layer: XML ↔ lxml tree, versioning, XSD validation, serialization."""

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string
from pyfieldml.dom.validator import validate_tree
from pyfieldml.dom.writer import write_file, write_string

__all__ = [
    "ParsedDocument",
    "parse_file",
    "parse_string",
    "validate_tree",
    "write_file",
    "write_string",
]
```

- [ ] **Step 5: Run all DOM tests**

Run: `uv run pytest tests/unit/test_dom_parser.py tests/unit/test_dom_validator.py tests/unit/test_dom_writer.py -v`
Expected: 14 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/dom/writer.py src/pyfieldml/dom/__init__.py tests/unit/test_dom_writer.py
git commit -m "feat(dom): add XML writer with round-trip-preserving serialization"
```

---

## Task 7 — Document stub

**Files:**
- Create: `src/pyfieldml/document.py`
- Test: `tests/unit/test_document.py`

- [ ] **Step 1: Write the failing tests — `tests/unit/test_document.py`**

```python
"""Tests for the Phase-0 Document stub."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyfieldml.document import Document
from pyfieldml.errors import FieldMLError


def test_document_from_file_records_source_version(fixtures_dir: Path) -> None:
    doc = Document.from_file(fixtures_dir / "minimal.fieldml")
    assert doc.source_version == "0.5.0"


def test_document_write_produces_valid_output(fixtures_dir: Path, tmp_path: Path) -> None:
    doc = Document.from_file(fixtures_dir / "two_types.fieldml")
    out = tmp_path / "out.fieldml"
    doc.write(out)
    roundtrip = Document.from_file(out)
    assert roundtrip.source_version == "0.5.0"


def test_document_validate_succeeds_on_valid_input(fixtures_dir: Path) -> None:
    doc = Document.from_file(fixtures_dir / "minimal.fieldml")
    doc.validate()  # must not raise


def test_document_validate_raises_on_invalid_input(fixtures_dir: Path) -> None:
    doc = Document.from_file(fixtures_dir / "invalid_no_name.fieldml")
    with pytest.raises(FieldMLError):
        doc.validate()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/unit/test_document.py -v`
Expected: ImportError — `pyfieldml.document` does not exist.

- [ ] **Step 3: Implement `src/pyfieldml/document.py`**

```python
"""Phase-0 Document stub.

A thin façade over the DOM layer. Holds a parsed ``lxml`` tree and exposes
``write()`` / ``validate()`` / ``source_version``. Phase 1 replaces this
with a fully typed semantic object model; callers that only need I/O can
use this stub as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from lxml import etree

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string
from pyfieldml.dom.validator import validate_tree
from pyfieldml.dom.writer import write_file

PathLike = Union[str, Path]


class Document:
    """An in-memory FieldML document.

    Phase 0: wraps the raw DOM tree. Phase 1 replaces the internals with a
    semantic object model; the public surface here (``from_file``, ``write``,
    ``validate``, ``source_version``) is forward-compatible.
    """

    def __init__(self, parsed: ParsedDocument) -> None:
        self._parsed = parsed

    # ------------------------------------------------------------------ factories

    @classmethod
    def from_file(cls, path: PathLike) -> Document:
        """Read a FieldML document from an on-disk file."""
        return cls(parse_file(path))

    @classmethod
    def from_string(cls, content: str | bytes) -> Document:
        """Read a FieldML document from an in-memory string/bytes."""
        return cls(parse_string(content))

    # ------------------------------------------------------------------ properties

    @property
    def source_version(self) -> str:
        """FieldML version declared on the root element (``@version``)."""
        return self._parsed.version

    @property
    def tree(self) -> etree._ElementTree:
        """Underlying lxml ElementTree. Escape hatch; prefer semantic APIs."""
        return self._parsed.tree

    # ------------------------------------------------------------------ operations

    def write(self, path: PathLike) -> None:
        """Serialize the document to ``path`` as FieldML 0.5 XML."""
        write_file(self._parsed.tree, path)

    def validate(self) -> None:
        """Validate the document against the bundled FieldML 0.5 XSD."""
        validate_tree(self._parsed.tree)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/unit/test_document.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/document.py tests/unit/test_document.py
git commit -m "feat(document): add Phase-0 Document stub over the DOM layer"
```

---

## Task 8 — Public top-level API + round-trip smoke test

**Files:**
- Modify: `src/pyfieldml/__init__.py`
- Test: `tests/unit/test_public_api.py`

- [ ] **Step 1: Write the failing tests — `tests/unit/test_public_api.py`**

```python
"""End-to-end round-trip tests exercising the public top-level API."""

from __future__ import annotations

from pathlib import Path

import pytest

import pyfieldml as fml
from pyfieldml.errors import FieldMLError


def test_read_returns_document(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "minimal.fieldml")
    assert isinstance(doc, fml.Document)
    assert doc.source_version == "0.5.0"


def test_validate_passes_on_valid_document(fixtures_dir: Path) -> None:
    fml.validate(fixtures_dir / "minimal.fieldml")


def test_validate_raises_on_invalid_document(fixtures_dir: Path) -> None:
    with pytest.raises(FieldMLError):
        fml.validate(fixtures_dir / "invalid_no_name.fieldml")


def test_read_then_write_then_reread(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "two_types.fieldml"
    doc = fml.read(src)
    out = tmp_path / "roundtrip.fieldml"
    doc.write(out)
    reread = fml.read(out)
    original_tags = [c.tag for c in doc.tree.getroot().find("Region")]
    roundtrip_tags = [c.tag for c in reread.tree.getroot().find("Region")]
    assert original_tags == roundtrip_tags


def test_package_exposes_version() -> None:
    assert fml.__version__ == "0.0.1"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/unit/test_public_api.py -v`
Expected: AttributeError — `pyfieldml.read`, `pyfieldml.validate`, `pyfieldml.Document` not exposed.

- [ ] **Step 3: Update `src/pyfieldml/__init__.py`**

Replace the file contents with:

```python
"""pyfieldml: a pure-Python implementation of FieldML 0.5."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from pyfieldml._version import __version__
from pyfieldml.document import Document
from pyfieldml.errors import (
    BasisNotFoundError,
    EvaluationError,
    FieldMLError,
    FieldMLParseError,
    FieldMLValidationError,
    UnboundArgumentError,
    UnresolvedImportError,
)

PathLike = Union[str, Path]


def read(path: PathLike) -> Document:
    """Read a FieldML document from ``path``. Convenience for ``Document.from_file``."""
    return Document.from_file(path)


def validate(path: PathLike) -> None:
    """Parse and XSD-validate ``path``. Raises ``FieldMLError`` on any failure."""
    Document.from_file(path).validate()


__all__ = [
    "BasisNotFoundError",
    "Document",
    "EvaluationError",
    "FieldMLError",
    "FieldMLParseError",
    "FieldMLValidationError",
    "UnboundArgumentError",
    "UnresolvedImportError",
    "__version__",
    "read",
    "validate",
]
```

- [ ] **Step 4: Run the public-API tests and confirm they pass**

Run: `uv run pytest tests/unit/test_public_api.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `uv run pytest -v`
Expected: all tests (errors + library + parser + validator + writer + document + public API) pass — ~28 tests.

- [ ] **Step 6: Commit**

```bash
git add src/pyfieldml/__init__.py tests/unit/test_public_api.py
git commit -m "feat(api): expose read/validate/Document at the top level"
```

---

## Task 9 — Tooling configs (ruff + mypy + pre-commit)

**Files:**
- Create: `.pre-commit-config.yaml`

(ruff + mypy config already in `pyproject.toml` from Task 1.)

- [ ] **Step 1: Run ruff on the source tree and confirm it passes**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`
Expected: both commands exit 0.

If ruff flags issues, fix them in the offending files (do not disable rules). Most likely findings at this stage: import ordering, missing docstrings in tests (ignored by per-file rule), unused imports.

- [ ] **Step 2: Run mypy on the source tree and confirm it passes**

Run: `uv run mypy src/pyfieldml`
Expected: `Success: no issues found`.

If mypy flags issues, add annotations rather than `# type: ignore`. Known soft spots: lxml types occasionally need `# type: ignore[no-any-return]` — only use where truly necessary.

- [ ] **Step 3: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        files: ^src/pyfieldml/
        additional_dependencies:
          - "lxml>=5.0"
          - "numpy>=1.26"
          - "h5py>=3.10"
          - "scipy>=1.11"
          - "pytest>=8"

  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
```

- [ ] **Step 4: Install pre-commit hooks and run once over all files**

Run: `uv run pre-commit install && uv run pre-commit run --all-files`
Expected: all hooks pass (may auto-fix formatting — stage + commit any resulting changes).

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore(tooling): wire up ruff, mypy, and pre-commit hooks"
```

---

## Task 10 — CI workflow (ci.yml)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    name: Lint + typecheck
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv sync --extra dev
      - run: uv run ruff check src tests
      - run: uv run ruff format --check src tests
      - run: uv run mypy src/pyfieldml

  test:
    name: Test (${{ matrix.os }} · py${{ matrix.python }})
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.10", "3.11", "3.12", "3.13"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          python-version: ${{ matrix.python }}
      - run: uv sync --extra dev
      - run: uv run pytest --cov=pyfieldml --cov-report=xml
      - name: Upload coverage
        if: matrix.os == 'ubuntu-latest' && matrix.python == '3.12'
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
          fail_ci_if_error: false
```

- [ ] **Step 2: Validate the workflow YAML locally**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: no output (parse succeeds).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add matrix workflow (ruff, mypy, pytest on 3.10-3.13 × 3 OS)"
```

- [ ] **Step 4: Push to a branch and verify CI is green**

```bash
git push origin main
```

Open the Actions tab on GitHub (or run `gh run watch`) and confirm all 13 jobs (1 lint + 12 test-matrix) pass. Fix any platform-specific failures inline before moving on.

---

## Task 11 — Docs skeleton (mkdocs + docs.yml)

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Create: `docs/getting-started.md`
- Create: `docs/api/index.md`
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Write `mkdocs.yml`**

```yaml
site_name: pyfieldml
site_description: A pure-Python implementation of FieldML 0.5 with evaluation engine and biomechanics model zoo.
site_url: https://kchemorion.github.io/pyfieldml/
repo_url: https://github.com/kchemorion/pyfieldml
repo_name: kchemorion/pyfieldml
edit_uri: edit/main/docs/

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy
    - content.code.annotate
    - search.highlight
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: false
            show_signature_annotations: true
            separate_signature: true

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Getting started: getting-started.md
  - API reference: api/index.md
```

- [ ] **Step 2: Write `docs/index.md`**

```markdown
# pyfieldml

A modern pure-Python implementation of [FieldML](https://physiomeproject.org/software/fieldml) 0.5 with a full evaluation engine, interop bridges, and a curated biomechanics model zoo.

!!! info "Status: pre-alpha"
    pyfieldml is under active development. Phase 0 (scaffolding + DOM layer) is live. See the [design spec](https://github.com/kchemorion/pyfieldml/blob/main/docs/superpowers/specs/2026-04-18-pyfieldml-design.md) for the full v1.0 roadmap.

## Install

```bash
pip install pyfieldml
```

## Quickstart

```python
import pyfieldml as fml

doc = fml.read("model.fieldml")
print(doc.source_version)          # e.g. "0.5.0"
doc.validate()                     # XSD-check
doc.write("roundtripped.fieldml")
```
```

- [ ] **Step 3: Write `docs/getting-started.md`**

```markdown
# Getting started

## Install

```bash
pip install pyfieldml
```

For development:

```bash
git clone https://github.com/kchemorion/pyfieldml
cd pyfieldml
uv sync --extra dev
uv run pytest
```

## Your first FieldML document

```python
import pyfieldml as fml

doc = fml.read("path/to/model.fieldml")
print(doc.source_version)
doc.validate()
```

## What's here in v0.0.x

Phase 0 ships the I/O plumbing:

- `pyfieldml.read(path)` → `Document`
- `Document.validate()` — XSD validation
- `Document.write(path)` — round-trip-preserving serialization
- `Document.tree` — raw lxml escape hatch

Semantic parsing (Evaluators, Types, the evaluation engine) arrives in Phase 1+.
```

- [ ] **Step 4: Write `docs/api/index.md`**

```markdown
# API reference

::: pyfieldml
    options:
      show_submodules: false

## Errors

::: pyfieldml.errors

## Document

::: pyfieldml.document.Document

## DOM layer

::: pyfieldml.dom
```

- [ ] **Step 5: Build the docs locally**

Run: `uv run mkdocs build --strict`
Expected: `site/` directory created; no warnings.

- [ ] **Step 6: Write `.github/workflows/docs.yml`**

```yaml
name: Docs

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: docs-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv sync --extra dev
      - run: uv run mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3
        if: github.ref == 'refs/heads/main'
        with:
          path: ./site

  deploy:
    if: github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 7: Commit**

```bash
git add mkdocs.yml docs/index.md docs/getting-started.md docs/api/index.md \
        .github/workflows/docs.yml
git commit -m "docs: add mkdocs-material skeleton and gh-pages deploy workflow"
```

After merging to main, enable GitHub Pages in the repo settings → Source: "GitHub Actions". Confirm `kchemorion.github.io/pyfieldml` goes live.

---

## Task 12 — Release workflow skeleton

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Write `.github/workflows/release.yml`**

```yaml
name: Release

on:
  push:
    tags: ["v*.*.*"]
  workflow_dispatch:

permissions:
  contents: write
  id-token: write  # required for PyPI trusted publishing (OIDC)

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
      - run: uv sync --extra dev
      - run: uv run python -m build  # needs `build` — added ad-hoc
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/project/pyfieldml/
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1

  github-release:
    needs: publish
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: softprops/action-gh-release@v2
        with:
          files: dist/*
          generate_release_notes: true
```

- [ ] **Step 2: Add `build` to the dev extras**

Edit `pyproject.toml`, in `[project.optional-dependencies].dev`, add `"build>=1.2"`. Run `uv sync --extra dev` to refresh the lockfile.

- [ ] **Step 3: Configure PyPI trusted publishing**

Follow [pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/) to register `pyfieldml` with:
- Owner: `kchemorion`
- Repository: `pyfieldml`
- Workflow: `release.yml`
- Environment: `pypi`

Then in GitHub repo settings → Environments → add `pypi` environment (no protection rules needed yet).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/release.yml pyproject.toml
git commit -m "ci: add PyPI trusted-publishing release workflow"
```

(No tag pushed yet — releases happen at the end of each phase.)

---

## Task 13 — Community / governance files

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `CHANGELOG.md`
- Create: `MAINTAINING.md`
- Create: `.github/ISSUE_TEMPLATE/bug.md`
- Create: `.github/ISSUE_TEMPLATE/feature.md`
- Create: `.github/pull_request_template.md`
- Modify: `README.md` (add install + usage)

- [ ] **Step 1: Write `CONTRIBUTING.md`**

```markdown
# Contributing to pyfieldml

Thank you for your interest! pyfieldml is an open, community-built reimplementation of FieldML 0.5. We welcome issues, PRs, datasets, and example notebooks.

## Dev setup

```bash
git clone https://github.com/kchemorion/pyfieldml
cd pyfieldml
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

## Workflow

- Work on a feature branch; PR into `main`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`.
- All PRs must pass CI (ruff, mypy, pytest matrix).
- New features arrive with tests. New files arrive with docstrings.

## Style

- `ruff format` / `ruff check` are authoritative. `mypy --strict` must be clean.
- Prefer small files with a single responsibility.
- Write tests first (TDD). Tests live alongside source in `tests/unit/`.

## Adding a dataset

See `docs/contributing-datasets.md` (added in Phase 5). Short version: fork, add your `.fieldml` + a `dataset.yaml` (BibTeX citation, SPDX license, SHA256 per file), open a PR. We verify license before merging.

## Adding a basis function

See `docs/contributing-bases.md` (added in Phase 2). Short version: `@pyfieldml.basis("your_element", topology="tet")` and implement the shape-function + derivative protocol.

## Questions

Use [GitHub Discussions](https://github.com/kchemorion/pyfieldml/discussions). Use Issues only for confirmed bugs or feature proposals.
```

- [ ] **Step 2: Write `CODE_OF_CONDUCT.md`**

Download the Contributor Covenant v2.1 text and substitute the contact placeholder:

```bash
curl -fsSL https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md -o CODE_OF_CONDUCT.md
python -c "p='CODE_OF_CONDUCT.md'; import pathlib; t=pathlib.Path(p).read_text(); pathlib.Path(p).write_text(t.replace('[INSERT CONTACT METHOD]', 'francis.chemorion@upf.edu'))"
```

Verify: `grep francis.chemorion CODE_OF_CONDUCT.md` returns a single match, and `grep "INSERT CONTACT" CODE_OF_CONDUCT.md` returns nothing.

- [ ] **Step 3: Write `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to pyfieldml are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
pyfieldml follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0: project scaffolding, CI matrix, `pyfieldml.dom` layer (parser,
  XSD validator, round-trip writer), and public `read` / `validate` /
  `Document` API.

[Unreleased]: https://github.com/kchemorion/pyfieldml/compare/HEAD...HEAD
```

- [ ] **Step 4: Write `MAINTAINING.md`**

```markdown
# Maintaining pyfieldml

## Release checklist

1. Ensure CI is green on `main`.
2. Update `CHANGELOG.md` — move Unreleased entries into a dated version section.
3. Bump `src/pyfieldml/_version.py`.
4. Update `CITATION.cff` version and date.
5. Commit: `chore(release): vX.Y.Z`.
6. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`.
7. Push: `git push origin main --tags`.
8. Confirm `release.yml` workflow succeeds; verify package on PyPI.
9. Announce in GitHub Discussions + relevant mailing lists.

## Adding a phase

Phases correspond to the roadmap in the design spec. Each phase:
1. Has its own plan in `docs/superpowers/plans/`.
2. Lands as a series of Conventional-Commit PRs.
3. Concludes with a release (minor bump during pre-1.0, else as appropriate).

## Deprecations

Deprecate with `DeprecationWarning` for one minor version before removal.
Document in `CHANGELOG.md` under "Deprecated" and "Removed".
```

- [ ] **Step 5: Write `.github/ISSUE_TEMPLATE/bug.md`**

```markdown
---
name: Bug report
about: Something isn't working as expected
labels: bug
---

### What happened

(A short description of the bug.)

### Minimal repro

```python
# Smallest code that triggers the bug.
```

Attach the offending `.fieldml` file if possible (the smaller the better).

### Expected behavior

### Environment

- pyfieldml version:
- Python version:
- OS:
- How installed (pip, uv, from source):

### Stack trace

```
(Full traceback here.)
```
```

- [ ] **Step 6: Write `.github/ISSUE_TEMPLATE/feature.md`**

```markdown
---
name: Feature request
about: Suggest a new capability or improvement
labels: enhancement
---

### What problem would this solve?

### Proposed solution

### Alternatives considered

### Additional context

(Links to relevant FieldML spec sections, related tooling, papers, etc.)
```

- [ ] **Step 7: Write `.github/pull_request_template.md`**

```markdown
## Summary

<!-- What does this PR do? Focus on the "why". -->

## Checklist

- [ ] Tests added or updated
- [ ] Docs updated (if user-visible change)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] `uv run ruff check src tests` clean
- [ ] `uv run mypy src/pyfieldml` clean
- [ ] `uv run pytest` passes locally

## Related issues

Closes #
```

- [ ] **Step 8: Update `README.md`**

Append to the existing `README.md` (after the "License" section, before "Acknowledgments"):

```markdown
## Install

```bash
pip install pyfieldml
```

Development:

```bash
git clone https://github.com/kchemorion/pyfieldml
cd pyfieldml
uv sync --extra dev
uv run pytest
```

## Quickstart

```python
import pyfieldml as fml

doc = fml.read("model.fieldml")
print(doc.source_version)          # "0.5.0", "0.4", "0.3"
doc.validate()                     # XSD validation
doc.write("roundtripped.fieldml")
```

> Phase 0 ships I/O and DOM-level round-trip. The semantic object model, evaluation engine, interop bridges, and model zoo arrive in later phases — see the [design spec](docs/superpowers/specs/2026-04-18-pyfieldml-design.md) and [plans](docs/superpowers/plans/).
```

- [ ] **Step 9: Commit**

```bash
git add CONTRIBUTING.md CODE_OF_CONDUCT.md CHANGELOG.md MAINTAINING.md \
        .github/ISSUE_TEMPLATE/ .github/pull_request_template.md README.md
git commit -m "docs: add community files (CONTRIBUTING, COC, CHANGELOG, MAINTAINING, templates)"
```

---

## Task 14 — Phase-0 closeout + tag pre-release

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `src/pyfieldml/_version.py`
- Modify: `CITATION.cff`

- [ ] **Step 1: Run the full pre-release sanity gauntlet**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/pyfieldml
uv run pytest -v --cov=pyfieldml
uv run mkdocs build --strict
```

Expected: everything clean. Fix any remaining issues.

- [ ] **Step 2: Update `CHANGELOG.md`**

Replace the Unreleased section with:

```markdown
## [Unreleased]

## [0.0.1] - 2026-04-18

### Added
- Initial project scaffolding, src-layout, pyproject.toml with hatchling
  backend.
- Public top-level API: `pyfieldml.read`, `pyfieldml.validate`,
  `pyfieldml.Document`, error hierarchy.
- DOM layer (`pyfieldml.dom`): `parse_file`, `parse_string`, `validate_tree`,
  `write_file`, `write_string`. FieldML 0.3 / 0.4 / 0.5 version detection at
  the root.
- Bundled FieldML 0.5 XSD and standard library (`pyfieldml.library`).
- CI matrix (ruff + mypy + pytest) across Python 3.10-3.13 × Linux / macOS
  / Windows.
- mkdocs-material documentation site with Getting Started + API reference.
- Release workflow with PyPI trusted publishing (OIDC).
- Community files: CONTRIBUTING, CODE_OF_CONDUCT, CHANGELOG, MAINTAINING,
  issue + PR templates.

[Unreleased]: https://github.com/kchemorion/pyfieldml/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/kchemorion/pyfieldml/releases/tag/v0.0.1
```

- [ ] **Step 3: Bump version and update CITATION.cff**

`src/pyfieldml/_version.py` is already `"0.0.1"` — no change needed.

Edit `CITATION.cff`, update:
- `version: 0.0.1`
- `date-released: "2026-04-18"` (or today's date)

- [ ] **Step 4: Commit and tag**

```bash
git add CHANGELOG.md CITATION.cff
git commit -m "chore(release): v0.0.1"
git tag -a v0.0.1 -m "v0.0.1: Phase 0 — scaffolding + DOM"
git push origin main --tags
```

- [ ] **Step 5: Verify the release pipeline**

- Watch `release.yml` in the Actions tab — it should build, publish to PyPI, and create a GitHub release.
- Confirm `pip install pyfieldml==0.0.1` works from a clean venv.
- Confirm `kchemorion.github.io/pyfieldml` displays the built docs.

If trusted publishing isn't configured yet (Task 12 Step 3), the publish step will fail — do Step 3 of Task 12 first, then re-run the release workflow via `gh workflow run release.yml -r v0.0.1` or delete + re-create the tag.

---

## Phase-0 done-definition

- [ ] Full test suite (≥28 tests) green on CI across the full matrix.
- [ ] `pip install pyfieldml==0.0.1` works from PyPI.
- [ ] Docs site deployed at `kchemorion.github.io/pyfieldml`.
- [ ] `pyfieldml.read(...).write(...)` round-trips the bundled standard library unchanged (semantic equivalence, not required byte-identical).
- [ ] CHANGELOG, CITATION, version, tag all consistent.

When all boxes are ticked, Phase 0 is complete. Next up: **Phase 1 — model & I/O** (typed semantic object model + HDF5 backends + conformance CI lighting up).
