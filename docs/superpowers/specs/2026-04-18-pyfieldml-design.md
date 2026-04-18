# pyfieldml — Design Spec

**Date:** 2026-04-18
**Status:** Approved pending user review
**Author:** Francis Chemorion (Universitat Pompeu Fabra) with Claude
**Target release:** `pyfieldml 1.0.0`

---

## 1. Context

[FieldML](https://physiomeproject.org/software/fieldml) is a declarative
markup language for representing mathematical fields over discrete meshes,
developed by the [Physiome Project](https://physiomeproject.org/) and the
[Auckland Bioengineering Institute](https://www.abi.auckland.ac.nz/) as the
data-exchange layer for [OpenCMISS](https://www.opencmiss.org/). It is
designed for computational physiology: finite-element meshes with continuous
fields (coordinates, fibers, material properties) defined on them.

The reference implementation — the C++
[FieldML-API](https://github.com/kchemorion/FieldML-API) — has been
effectively abandoned since 2015. It ships no README, no Python bindings, no
notebooks, no PyPI release, no CI. A modern computational physiology
researcher cannot `import` it and try it in 30 seconds, which is the table
stakes for adoption in today's scientific-Python ecosystem. Useful tooling
has moved on: `meshio`, `PyVista`, `scikit-fem`, and the OpenSim Python
bindings are where musculoskeletal and cardiac modelers actually work.

**`pyfieldml` is an independent, pure-Python reimplementation of FieldML 0.5
with a full evaluation engine, modern interop bridges, and a curated
biomechanics model zoo.** It does not incorporate source code from the
original C++ API; it uses that library as a correctness oracle in CI.

## 2. Goals, audience, scope

### 2.1 Audience

Primary: biomechanics and computational-physiology researchers (with a
particular focus on musculoskeletal modelling — bones, muscles, fiber
fields, landmark sets) who want to read, construct, evaluate, and convert
FieldML documents from Python.

Secondary: anyone maintaining legacy Physiome data assets; anyone building
tooling that needs to interoperate with FieldML.

### 2.2 In scope for v1.0

**Pillar A — Implementation**

- Full FieldML 0.5 read + write, with round-trip validated against the C++
  reference test suite: byte-identical XML where achievable; otherwise
  C14N-canonical-equivalent XML plus exact equality of all parameter arrays.
- **Legacy read** for FieldML 0.3 and 0.4, up-converted to 0.5 in memory on
  load. Writing is 0.5-only.
- All evaluator types: Parameter, Reference, Aggregate, Piecewise, Constant,
  Argument, External.
- All array data backends: inline text, external text, HDF5 dense, HDF5 DOK
  (dictionary-of-keys sparse).
- Full resolution of `<Import>` directives, including the bundled standard
  library (`FieldML_Library_0.5.xml`).
- XSD validation.
- NumPy zero-copy views on parameter data where the backend permits.
- **Evaluation engine** over the evaluator graph with argument-binding
  resolution. Built-in bases covering the common corpus:
    - Linear and quadratic **Lagrange** on line, triangle, quad, tetrahedron,
      hexahedron, wedge.
    - Cubic **Hermite** with per-node scaling on the same topologies — this
      is required for the Auckland cardiac reference models and several
      legacy musculoskeletal models.
- Jacobians and derivatives for all built-in bases.
- `Field.evaluate(element, xi)` (scalar and vectorized) and
  `Field.sample(points)` backed by a `scipy.spatial.cKDTree` +
  inverse-xi Newton locate.
- **Plugin API for custom basis functions**:
  `@pyfieldml.basis("name", topology="tet")`.
- **High-level builders** for common construction patterns:
  `add_lagrange_mesh`, `add_hermite_mesh`, `add_vector_field`,
  `add_fiber_field`, `add_material_field`, `add_landmark_set`.

**Pillar B — Interop bridges** (each optional, one module per bridge)

- `meshio` — two-way, registered via `meshio.plugins` entry point so
  `meshio.read("*.fieldml")` routes through us automatically.
- `PyVista` — `doc.plot()` and `field.plot()` produce embedded Jupyter
  widgets via `trame`; `doc.explore()` opens an interactive evaluator-graph
  browser (ipywidgets + `ipycytoscape`) with probe-at-point.
- **XDMF** — one-way writer, no extra dependency (uses our existing `lxml`
  + `h5py`). Makes FieldML producer-compatible with ParaView / VisIt / every
  HPC viz pipeline.
- `scikit-fem` — register scikit-fem's basis catalog into our plugin
  registry; also expose a `to_scikit_fem` bridge that returns a scikit-fem
  mesh + basis for PDE assembly on FieldML-loaded geometries.
- **OpenSim-adjacent export** — write muscle geometry, attachment landmarks,
  and fiber-direction streamlines as `.vtp` assets plus a CSV/`.osim`
  fragment that can be hand-assembled into an OpenSim model. Explicitly *not*
  a full `.osim` round-trip.

**Pillar C — Scientific credibility**

- **C++ reference cross-validation in CI** (Linux-only, nightly + on
  release). Builds the original C++ library, runs every `test/input/*.fieldml`
  through both implementations, asserts byte-identical XML round-trip and
  exact array equality. A failing conformance run blocks releases.
- **Public conformance test suite**: the fixture corpus is published as a
  downloadable artifact so any FieldML implementation can validate against
  the same tests. Positions `pyfieldml` as a reference, not yet-another-impl.
- **Curated model zoo** (four openly-licensed datasets, details §6).
- **JOSS paper** submitted the week v1.0 tags — gives the project a DOI and
  a citeable reference.
- **Docs site** (`mkdocs-material`) with `jupyter-lite` in-browser runnable
  notebooks.

**Foundations**

- Type hints throughout; `mypy --strict` clean.
- Python 3.10+, Linux + macOS + Windows.
- Required deps: `lxml`, `numpy`, `h5py`, `scipy`.
- Optional extras: `[viz]`, `[meshio]`, `[scikit-fem]`, `[opensim]`, `[all]`,
  `[dev]`.

### 2.3 Out of scope for v1.0

- Differentiable evaluation backend (JAX / PyTorch) — *designed for* but not
  implemented; basis kernels written against the array API so the seam is
  there for v2.0.
- FEniCS / Firedrake direct export — XDMF already covers both.
- Desktop GUI application.
- Fortran, JNI, or C bindings.
- Extending the FieldML specification.
- Writing FieldML 0.3 / 0.4 (only reading).
- Full OpenSim `.osim` round-trip — different scope, separate project.

### 2.4 Non-goals

- `pyfieldml` does **not** reimplement OpenCMISS or Zinc. It is an I/O +
  data model + evaluation library.
- `pyfieldml` does **not** extend the FieldML specification. It targets
  FieldML 0.5 as published.

## 3. Architecture

Five layered subsystems, each independently testable. Interop adapters
depend only on the public `model/` + `eval/` APIs.

```
pyfieldml/
├── dom/           # XML ↔ raw tree. Parser, writer, XSD validator, 0.3/0.4 → 0.5 up-converter.
├── model/         # Typed semantic object model: Document, Region, Types, Evaluators, Bindings, Imports.
├── data/          # Array backends: inline text, text-ref, HDF5 dense, HDF5 DOK sparse. Lazy views.
├── eval/          # Evaluation engine.
│   ├── graph.py   #   Walks evaluator graph with binding resolution.
│   ├── bases/     #   Plugin registry + built-in Lagrange + Hermite-with-scaling.
│   ├── locate.py  #   cKDTree + inverse-xi Newton for point location.
│   └── field.py   #   User-facing Field.evaluate() / Field.sample().
├── builders/      # High-level: add_lagrange_mesh, add_hermite_mesh, add_fiber_field, etc.
├── interop/       # meshio, pyvista, xdmf, scikit_fem, opensim — each optional.
├── viz/           # doc.explore() graph widget, field.plot() PyVista renderer (Jupyter-first).
├── cli/           # `pyfieldml inspect|validate|convert|plot|lint|diff`. Thin over public API.
├── datasets/      # Model zoo registry + on-demand download/cache.
├── validation/    # Linter + semantic graph diff.
└── library/       # Bundled FieldML_Library_0.5.xml (so imports resolve offline).

tests/
├── unit/          # Per-module, fast.
├── conformance/   # Cross-validate against the C++ reference library (built once in CI).
├── models/        # End-to-end on real MSK + cardiac assets.
└── fixtures/      # .fieldml test vectors.
```

### Six architectural decisions worth calling out

1. **DOM / model split.** `dom/` is a thin XML bijection. `model/` is the
   semantic graph. Up-conversion (0.3/0.4 → 0.5) happens at the DOM layer;
   model code never sees legacy versions. Writers only emit 0.5.
2. **Evaluator graph is data, not behavior.** `model/` owns the graph;
   `eval/` consumes it. The graph can be inspected, diffed, serialized, or
   eventually re-executed by a JAX backend without touching the model layer.
3. **Basis-function registry.** Runtime plugin system. Built-ins register
   themselves on import; external users register via
   `@pyfieldml.basis("name", topology=…)`. Evaluator engine resolves by the
   External Evaluator's name from the FieldML stdlib.
4. **Interop as optional adapters.** Each bridge is a shim over the public
   `model/` + `eval/` APIs, not imported unless the corresponding extra is
   installed. Keeps `import pyfieldml` cold-start fast.
5. **Lazy array data.** HDF5-backed `ParameterEvaluator` returns a handle;
   `.as_ndarray()` materializes. Large cardiac/MSK datasets do not blow up
   memory on document load.
6. **C++ reference oracle, test-only.** `tests/conformance/` builds the
   original C++ library once in CI and cross-validates every test vector.
   Never in the runtime package.

### Dependencies

- **Required:** `lxml`, `numpy`, `h5py`, `scipy`.
- **Extras:**
  - `[viz]` → `pyvista`, `ipywidgets`, `trame-jupyter-extension`, `ipycytoscape`
  - `[meshio]` → `meshio`
  - `[scikit-fem]` → `scikit-fem`
  - `[opensim]` → `meshio` (`.vtp` export via meshio; native OpenSim Python
    lib is not required)
  - `[all]` → the above
  - `[dev]` → `pytest`, `pytest-cov`, `nbmake`, `ruff`, `mypy`,
    `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter`, `jupyter`

## 4. Public API

```python
import pyfieldml as fml

doc = fml.read("femur.fieldml")          # Document
doc = fml.Document(name="femur_study")   # new empty
fml.validate("femur.fieldml")            # XSD + semantic
doc.write("out.fieldml")
```

### 4.1 Reading & inspecting

```python
doc = fml.read("femur.fieldml")
doc.source_version                       # "0.5" | "0.4" | "0.3"

doc.ensembles["nodes"]                   # EnsembleType
doc.continuous["RC.3d"]                  # ContinuousType
doc.meshes["femur_mesh"]                 # MeshType
doc.evaluators["coordinates"]            # concrete Evaluator subclass

coords = doc.field("coordinates")        # Field — ergonomic wrapper
coords.dtype, coords.shape
coords.as_ndarray()                      # zero-copy where possible
```

The raw `Evaluator` subclasses stay inspectable for tooling / linters /
diff. The `Field` wrapper is sugar for the 90% use case: always has
`.evaluate`, `.sample`, `.as_ndarray`, `.plot`.

### 4.2 Evaluation

```python
coords = doc.field("coordinates")

x = coords.evaluate(element=5, xi=(0.25, 0.5, 0.75))     # shape (3,)

elements = np.full(10_000, 5)
xi = np.random.rand(10_000, 3)
xs = coords.evaluate(element=elements, xi=xi)             # vectorized, (10_000, 3)

probe_pts = np.random.rand(1_000, 3) * bbox
vals = coords.sample(probe_pts)                           # cKDTree + Newton

J = coords.jacobian(element=5, xi=(0.25, 0.5, 0.75))      # (3, 3)
```

### 4.3 Writing via builders

```python
doc = fml.Document(name="femur_study")

mesh = doc.add_lagrange_mesh(
    name="femur_mesh",
    nodes=xyz,
    elements=connectivity,
    topology="tet",           # line|tri|quad|tet|hex|wedge
    order=1,                  # 1 or 2
)

coords = doc.add_vector_field(
    name="coordinates", mesh=mesh, values=xyz,
    storage="hdf5", h5_file="femur_data.h5",
)

doc.add_fiber_field(mesh=mesh, vectors=fiber_xyz)
doc.add_material_field(mesh=mesh, values_per_element=E_modulus)
doc.add_landmark_set(name="anatomical_points", points=lms, labels=labels)

hmesh = doc.add_hermite_mesh(
    name="myocardium", nodes=..., elements=...,
    topology="hex", scaling="per_node",
)

doc.write("femur.fieldml")   # emits FieldML 0.5 + .h5 sidecar
```

Raw graph construction (`doc.add_parameter_evaluator`, etc.) remains
available for users who need full control.

### 4.4 IO backends

- `ParameterEvaluator.data` is a `DataSource` handle; `.as_ndarray()`
  materializes.
- Read-side backend chosen by inspection: `InlineText`, `ExternalText`,
  `Hdf5Dense`, `Hdf5DOK`.
- Write-side backend chosen by `storage=` kwarg; default heuristic is
  inline ≤ 64 KB else HDF5 sidecar.
- All backends round-trip `float32` / `float64` / `int32` / `int64`
  losslessly. Verified byte-level against the C++ reference in conformance
  CI.

### 4.5 Legacy read + meshio

```python
old = fml.read("heart_0_4.fieldml")      # auto-detects, up-converts in memory
old.source_version                       # "0.4"
old.write("heart_0_5.fieldml")           # always 0.5

m = doc.to_meshio()
new = fml.Document.from_meshio(m, name="imported")
```

### 4.6 CLI

```
pyfieldml inspect   femur.fieldml
pyfieldml validate  femur.fieldml [--strict]
pyfieldml convert   femur.fieldml --to vtu -o x.vtu
pyfieldml plot      femur.fieldml --field fiber_direction
pyfieldml lint      femur.fieldml        # orphans, inverted elements, unbounded args
pyfieldml diff      a.fieldml b.fieldml  # semantic evaluator-graph diff
pyfieldml bench     femur.fieldml        # eval/sample throughput
```

### 4.7 Errors

Single base `FieldMLError` →
`FieldMLParseError`, `FieldMLValidationError`, `UnresolvedImportError`,
`UnboundArgumentError`, `EvaluationError`, `BasisNotFoundError`. All carry
source line info from the original XML where applicable.

## 5. Interop bridges

### 5.1 meshio — two-way

Cell-type mapping:

| FieldML topology, order | meshio cell type        |
|-------------------------|-------------------------|
| tet, 1 / 2              | `tetra` / `tetra10`     |
| hex, 1 / 2              | `hexahedron` / `hexahedron27` |
| tri, 1 / 2              | `triangle` / `triangle6` |
| quad, 1 / 2             | `quad` / `quad9`        |
| wedge, 1                | `wedge`                 |
| line, 1 / 2             | `line` / `line3`        |

Hermite meshes export by sampling the reconstructed geometry at nodes and
writing a linear-Lagrange connectivity, with a warning. Data fields can still
be sampled and exported as `point_data` / `cell_data`.

Registration uses meshio's extension mechanism (entry point if supported by
the installed meshio version, otherwise an explicit `register_format` call
triggered the first time `pyfieldml.interop.meshio` is imported). No
monkey-patching of meshio internals.

### 5.2 PyVista — `doc.plot()`, `field.plot()`, `doc.explore()`

- `doc.plot()`: primary mesh as `UnstructuredGrid`, colored by a default
  scalar field, rendered via `trame` in Jupyter.
- `field.plot()`: shape-aware — scalar → surface coloring, vector → arrow
  glyphs, fiber → streamlines, per-element → element coloring.
- `doc.explore()`: interactive Jupyter widget. Left pane = evaluator tree;
  center pane = graph view (`ipycytoscape`); right pane = selected-node
  detail + probe-at-point. Built on off-the-shelf widgets; ~1 week to build.

### 5.3 XDMF — one-way, no extra dep

`pyfieldml.interop.xdmf.write(doc, "model.xdmf")` emits an XDMF XML
descriptor with heavy data in a sidecar HDF5 file. Each `MeshType` → `<Grid>`;
each field → `<Attribute>`. Instantly openable in ParaView, VisIt, and every
major HPC solver.

### 5.4 scikit-fem — basis catalog + solver bridge

- `pyfieldml.bases.register_scikit_fem()` wraps scikit-fem's basis catalog
  behind adapters that match our basis-plugin protocol (shape function +
  derivative evaluators over reference xi). Gives free access to the
  higher-order Lagrange, Raviart-Thomas, Nédélec, etc. scikit-fem already
  implements, without us reimplementing them.
- `pyfieldml.interop.scikit_fem.to_scikit_fem(mesh, order=1)` returns a
  scikit-fem `Mesh` + `Basis` ready for PDE assembly.

### 5.5 OpenSim-adjacent export

```python
from pyfieldml.interop import opensim as osim

osim.export_geometry(doc, out_dir="osim_assets/")
osim.export_muscle_attachments(doc, "landmarks.csv")
osim.export_fiber_lines(doc, mesh="muscle_mesh",
                        field="fiber_direction", out="fibers.vtp")
```

Positioned in docs as *"OpenSim-compatible assets for use in OpenSim
workflows, not a full model round-trip."*

### 5.6 Designed-for-but-deferred

- JAX / PyTorch differentiable evaluation — basis kernels written against
  the array API.
- Full OpenSim `.osim` round-trip.
- FEniCS / Firedrake direct export (covered via XDMF).

## 6. Showcase notebooks + model zoo

### 6.1 Model zoo (v1.0)

Four curated, openly-licensed assets, lazily downloaded to
`~/.cache/pyfieldml/datasets/` with SHA256 verification:

| Dataset          | Topology               | Demonstrates                                                    | Licensing path |
|------------------|------------------------|-----------------------------------------------------------------|----------------|
| `unit_cube`      | 1 hex                  | Smoke test, tutorials                                           | Authored, CC0 |
| `femur`          | ~10–50k tet            | Bone geometry + heterogeneous material field (BMD → E modulus)  | Auckland/Physiome if CC BY 4.0 available, else author from open CT data |
| `rectus_femoris` | ~5–20k tet             | Muscle mesh + fiber-direction field + attachment landmark set   | Same |
| `myocardium`     | hex, cubic Hermite     | Cardiac FE with Hermite bases + fiber/sheet fields              | Openly-licensed cardiac reference from the Physiome/Auckland corpus (specific asset and license verified during P5) |

**Provenance gate.** Every dataset must be verified openly-licensed with
attributed source before v1.0 tags. The target approach is to ask your
professor's group to nominate and contribute musculoskeletal assets under
CC BY 4.0. Where clearance fails, we author replacements from openly
available CT/published meshes rather than delay release.

### 6.2 Dataset API

```python
from pyfieldml import datasets

doc = datasets.load_femur()
datasets.list()
datasets.info("femur")          # citation, license, origin URL, size, hashes
datasets.download("femur")      # explicit pre-fetch
datasets.cache_dir()
```

Each asset ships `dataset.yaml` with BibTeX citation, SPDX license tag,
origin URL, and per-file SHA256. Convention matches scikit-learn /
HuggingFace datasets.

### 6.3 Notebook curriculum (8 notebooks + 1 reproducibility)

All CI-executed via `nbmake`. A notebook that breaks blocks merges.

| #  | Title                                | Skill taught                                                                     | Dataset          |
|----|--------------------------------------|----------------------------------------------------------------------------------|------------------|
| 01 | Quickstart                           | Install → read → plot → extract NumPy in 5 lines                                 | `femur`          |
| 02 | The evaluator graph                  | What is inside a FieldML file; `doc.explore()`; probing                          | `femur`          |
| 03 | Build a femur model from scratch     | `add_lagrange_mesh`, material field from BMD, landmarks, validate, round-trip    | CT-derived       |
| 04 | Muscle fiber fields                  | Load fibers, `field.plot()` glyphs, along-fiber strain, export streamlines       | `rectus_femoris` |
| 05 | Cardiac Hermite bases                | Cubic Hermite with scaling; sample at arbitrary xi; compare fidelity vs linear   | `myocardium`     |
| 06 | Solve a PDE on a FieldML mesh        | FieldML → scikit-fem → assemble linear elasticity → write solution back          | `femur`          |
| 07 | OpenSim bridge                       | Export muscle geometry, attachments, fiber polylines; import-into-OpenSim recipe | `rectus_femoris` |
| 08 | Legacy conversion                    | Read 0.3/0.4, up-convert, diff vs 0.5, validate                                  | Auckland legacy  |

Plus one **scientific reproducibility notebook** under `paper/benchmarks/`:
reproduce a classical MSK FE result (candidate: Keller-1994 femoral
stiffness, final target chosen during implementation based on dataset
availability) using `pyfieldml` + `scikit-fem`. Cited from the JOSS paper.

### 6.4 jupyter-lite embedding

Docs site embeds notebooks 01 and 04 as live `jupyter-lite` instances —
click "run in browser", no install. Dependencies `numpy` / `scipy` / `h5py`
/ `lxml` all have pyodide wheels. `pyvista` does not run in pyodide;
browser versions fall back to pre-rendered static figures.

### 6.5 Dataset contribution pipeline

A fifth dataset slot is left open. `CONTRIBUTING.md` includes a template
for: fork the datasets repo → add `.fieldml` + `dataset.yaml` → PR → we
clear license and add a short notebook.

## 7. Tooling, CI, release

### 7.1 Project scaffolding

- **Build:** `hatchling`. `pyproject.toml` only.
- **Dev env:** `uv sync --extra dev` is the canonical path. `hatch envs`
  supported.
- **Lint/format:** `ruff` (replaces black, isort, flake8, pyupgrade).
- **Typing:** `mypy --strict`; `py.typed` marker shipped.
- **Pre-commit:** ruff, mypy, `nbstripout`.
- **Commit style:** Conventional Commits (commit-msg hook, not CI gate).

### 7.2 CI

`ci.yml`:
- Matrix: Python **3.10 / 3.11 / 3.12 / 3.13** × **Linux / macOS / Windows**.
- Steps: ruff → mypy → pytest unit → nbmake on Linux-3.12 → coverage.

`conformance.yml` (Linux-only, nightly + on release):
- Builds the original C++ FieldML-API (cached).
- Runs every `test/input/*.fieldml` through both implementations; asserts
  byte-identical XML round-trip where achievable, else C14N-canonical
  equivalence, plus exact NumPy equality for all array backends.
- A failing conformance run blocks releases.

`docs.yml`: mkdocs build on every PR; GH Pages deploy on merge to main.

`release.yml`: on version tag, builds wheel + sdist, publishes to PyPI
via **trusted publishing (OIDC)**. Generates a GH release with auto-notes
and updates `CITATION.cff`.

### 7.3 Docs site

- `mkdocs-material` + `mkdocstrings[python]` + `mkdocs-jupyter` +
  `jupyter-lite`.
- Sections: Getting started, Guides, Notebooks, API reference, Conformance
  suite (downloadable), Contributing, Changelog, Cite this work.
- Deployed at `kchemorion.github.io/pyfieldml` (custom domain possible
  later).

### 7.4 Release model & governance

- **Semver.** 0.x during pre-release; 1.0 when §2.2 fully ships.
- **`CHANGELOG.md`** (Keep-a-Changelog format).
- **`CITATION.cff`** — kept current; drives GH "Cite this repo" button.
- **`CODE_OF_CONDUCT.md`** — Contributor Covenant v2.1.
- **`CONTRIBUTING.md`** — dev setup, tests, how to add datasets, how to
  add basis plugins.
- **Issue + PR templates.**
- **GitHub Discussions** enabled.
- Maintainer doc `MAINTAINING.md` with release checklist.

### 7.5 License & attribution

- **License:** **Apache 2.0.** Modern scientific-Python default; explicit
  patent grant; compatible with downstream GPL/BSD consumers. Fresh choice
  is permissible because `pyfieldml` is an independent reimplementation, not
  a derivative of the MPL 1.1 / GPL 2 / LGPL 2.1 C++ source.
- **`NOTICE`** credits the original FieldML authors (Caton Little, Alan Wu,
  Richard Christie, Andrew Miller, Auckland Uniservices Ltd) and the
  Physiome Project.
- README and JOSS paper both prominently cite the original implementation
  and the FieldML specification.
- Every dataset ships its own license + BibTeX; redistribution only under
  open licenses.

### 7.6 JOSS paper

- Path: `paper/paper.md`, `paper/paper.bib`, JOSS template.
- ~2–4 pages: problem statement, feature tour, one figure from the
  reproducibility notebook, statement of need, acknowledgments, references.
- Submitted the same week v1.0 tags. Typical review cycle 4–8 weeks.

### 7.7 Phased timeline

Total ~5 months full-time or 8–10 months part-time with a student.

| Phase | Weeks | Deliverable                                                                      |
|-------|-------|----------------------------------------------------------------------------------|
| P0 — Scaffolding            | 1–2 | Repo, CI matrix, docs skeleton, `dom/` parser + writer + XSD validation; simplest test vector round-trips. |
| P1 — Model & I/O            | 3   | `model/` complete; all four array backends; full round-trip on C++ test vectors; conformance CI green. |
| P2 — Eval (Lagrange)        | 4   | Basis registry; Lagrange orders 1–2 on all topologies; `Field.evaluate/sample/jacobian`; cKDTree locate. |
| P3 — Hermite + legacy       | 3   | Cubic Hermite with per-node scaling; 0.3/0.4 → 0.5 up-converter; myocardium dataset evaluable. |
| P4 — Interop                | 3   | meshio entry-point; PyVista `plot` + `explore`; XDMF; scikit-fem; OpenSim export. |
| P5 — Builders + zoo + notebooks | 4 | Builders; four datasets (provenance-cleared); all 8 notebooks; reproducibility notebook. |
| P6 — Docs + release         | 1–2 | Docs site live; jupyter-lite quickstart running; v1.0.0 tag; PyPI publish; announce. |
| P7 — JOSS (parallel with P5–P6) | — | Submit paper the week v1.0 tags.                                             |

### 7.8 Definition of "done" for v1.0

- Conformance CI green on every C++ reference test vector.
- Unit coverage ≥ 85% on `model/` / `eval/` / `data/`; ≥ 70% overall.
- `mypy --strict` clean.
- All 8 notebooks execute in CI on every PR via `nbmake`.
- Docs site deployed; in-browser quickstart runs under `jupyter-lite`.
- PyPI wheel + sdist published via trusted publishing.
- JOSS paper submitted.
- `README`, `CONTRIBUTING`, `CODE_OF_CONDUCT`, `CITATION`, `CHANGELOG`,
  `NOTICE`, `LICENSE` all in place.
- License cleared on every bundled dataset.

### 7.9 Post-release strategy (first 6 months)

- Announce: OpenCMISS + Physiome mailing lists, SciPy newsletter,
  `r/scientificcomputing`, Mastodon `#scipy`, OpenSim forum.
- Conference talks / posters: SciPy, VPH Conference, OpenSim Workshop —
  author opportunity for the professor's students.
- Monthly patch/minor releases; respond to issues; accept user datasets.
- v1.1 candidates: JAX differentiable evaluation, FieldML 0.3/0.4 write
  support, additional basis plugins, additional datasets.

## 8. Repo logistics

- `pyfieldml` lives in its own repo at
  `github.com:kchemorion/pyfieldml`.
- The existing `kchemorion/FieldML-API` fork stays on disk locally and in
  GitHub as the reference-oracle source. Its README gets a single line
  pointing at `pyfieldml`.
- License of the new repo: Apache 2.0 (see §7.5).
- The design spec this document represents is committed at
  `docs/superpowers/specs/2026-04-18-pyfieldml-design.md` in the new repo.

## 9. Open questions tracked for implementation

1. **Dataset provenance.** Each of `femur`, `rectus_femoris`, `myocardium`
   needs a specific openly-licensed source chosen and verified before v1.0.
   Action: ask Francis's professor to nominate contributable assets; fall
   back to open CT / published meshes where needed.
2. **Reproducibility benchmark target.** Classical MSK FE result whose mesh
   + boundary conditions are fully public. Candidate: Keller 1994. Final
   choice during P5.
3. **Custom `pyfieldml.org` domain.** Optional; cheap. Defer until post-v1.0
   unless the professor wants to set up an org.
4. **Future org move.** If a dedicated `openfieldml` GitHub org is created,
   `pyfieldml` can be transferred; Apache 2.0 makes this trivial.

## 10. Acknowledgments

This project builds on the foundational work of the FieldML specification
and the C++ FieldML-API. Credit to Caton Little, Alan Wu, Richard Christie,
Andrew Miller, and Auckland Uniservices Ltd / the Auckland Bioengineering
Institute, and to the Physiome Project community that maintains the FieldML
standard.
