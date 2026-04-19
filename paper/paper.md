---
title: 'pyfieldml: A modern Python implementation of FieldML 0.5 for computational biomechanics'
tags:
  - Python
  - FieldML
  - finite element
  - computational physiology
  - biomechanics
  - musculoskeletal modelling
  - Physiome Project
authors:
  - name: Francis Chemorion
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Universitat Pompeu Fabra, Barcelona, Spain
    index: 1
date: 19 April 2026
bibliography: paper.bib
---

# Summary

[FieldML](https://physiomeproject.org/software/fieldml) is a declarative
markup language, developed under the [Physiome Project](https://physiomeproject.org/)
and the Auckland Bioengineering Institute, for representing mathematical
fields — such as tissue geometry, muscle fiber directions, and material
heterogeneity — over discrete finite-element meshes
[@fieldml_christie_2009]. It has underpinned computational-physiology
research within the [OpenCMISS](https://www.opencmiss.org/) ecosystem for
over a decade, but its reference C++ implementation
[@fieldml_api] has seen little modern tooling beyond its original release.

`pyfieldml` is a pure-Python, independent reimplementation of the FieldML
0.5 specification, aimed at bringing FieldML data into today's
scientific-Python ecosystem. It provides a full evaluator-graph read/write
layer, an evaluation engine covering Lagrange and cubic-Hermite bases on
the standard element topologies, bridges to `meshio` [@meshio],
PyVista [@pyvista], `scikit-fem` [@scikit_fem], and XDMF3 [@xdmf], a
semantic linter and diff, a command-line interface, and a curated model
zoo with bundled synthetic biomechanics assets.

# Statement of Need

FieldML is a mature and carefully designed standard, but the barrier to
entry for new users has been high. The original C++ API requires a
source build, depends on libxml2 and HDF5, has no Python bindings, and
has been largely unmaintained since 2015. Meanwhile, the scientific
computational-physiology community has moved decisively toward Python —
simulation pipelines now chain together tools like `meshio`, `PyVista`,
`scikit-fem`, and OpenSim via Python interfaces.

`pyfieldml` closes that gap. A biomechanics or cardiac-modelling researcher
can now install the package with `pip install pyfieldml`, load a bundled
dataset, inspect its evaluator graph, sample fields at arbitrary parametric
coordinates, and export the geometry to VTK or XDMF for visualisation —
without compiling anything.

Legacy interoperability is a first-class concern: `pyfieldml` auto-upgrades
FieldML 0.3 and 0.4 documents to the 0.5 shape transparently on read, so
the substantial existing Physiome-Project corpus remains accessible.

# Key features

- **Full FieldML 0.5 read/write**, validated via the bundled FieldML XSD.
  Documents in FieldML 0.3 and 0.4 are transparently up-converted.
- **Typed semantic object model** (`Document` → `Region` → `Types` +
  `Evaluators`) exposed via mapping-style APIs (`doc.ensembles`,
  `doc.continuous`, `doc.evaluators`, etc.).
- **Evaluation engine** with a plugin-based basis-function registry,
  covering linear and quadratic Lagrange on six topologies (12 bases) and
  cubic Hermite on line / quad / hex (3 bases) with per-node scaling.
  `Field.evaluate(element, xi)`, `Field.sample(points)` (spatial locate +
  inverse-xi Newton), and `Field.jacobian(element, xi)` are vectorised over
  NumPy arrays.
- **Four array-data backends** — inline text, external text, HDF5 dense,
  HDF5 DOK sparse — with lazy loading and round-trip guarantees for
  `float32`/`float64`/`int32`/`int64`.
- **Interop bridges** (each an optional extra): `meshio` (two-way plugin,
  `meshio.read("*.fieldml")` works out of the box), PyVista
  (`doc.plot()` / `field.plot()`), XDMF3 + HDF5 sidecar writer, `scikit-fem`
  (returns `(Mesh, Basis)` for PDE assembly), and OpenSim-adjacent asset
  export (geometry `.vtu`, muscle attachments CSV, fiber-line polylines).
- **Validation tooling**: `pyfieldml lint` catches orphan evaluators and
  inverted elements; `pyfieldml diff` reports semantic differences between
  two documents.
- **High-level builders** (`add_lagrange_mesh`, `add_hermite_mesh`,
  `add_scalar_field`, `add_vector_field`, `add_fiber_field`,
  `add_material_field`, `add_landmark_set`) simplify programmatic
  construction of documents for testing and pipeline use.
- **Curated model zoo** with three CC0-licensed synthetic datasets
  bundled in the wheel: `unit_cube`, `femur` (cylinder-shaped bone with
  BMD-derived material), and `rectus_femoris` (spindle-shaped muscle with
  fiber field). Further datasets, including real Physiome-Project assets,
  are planned in minor releases subject to licensing.
- **CLI surface**: `pyfieldml inspect|validate|convert|plot|lint|diff|bench`.
- **Jupyter-native**: `doc.explore()` opens an `ipywidgets` +
  `ipycytoscape` evaluator-graph browser; three CI-executed tutorial
  notebooks ship in the docs site.

# Implementation and engineering

The codebase is structured in layered subpackages — `dom` (XML ↔ lxml
tree), `model` (semantic graph), `data` (array backends), `eval` (engine
and bases), `builders`, `interop`, `validation`, `viz`, `cli`, `datasets`.
This separation makes each layer independently testable and keeps
contributor-facing surface narrow.

Continuous integration runs on Linux, macOS, and Windows across Python
3.10 through 3.13. The test suite exceeds 170 unit tests plus three
CI-executed Jupyter notebooks; a separate nightly conformance workflow
builds the original C++ FieldML-API reference and cross-checks file I/O
(full cross-validation assertions are on a later-release roadmap, pending
writer-level XML-shape parity work).

Type hints are applied throughout and checked with `mypy --strict` on the
source tree. Documentation is published via `mkdocs-material` with an
auto-generated API reference and three CI-executed tutorial notebooks.

# Acknowledgements

`pyfieldml` is an independent Python reimplementation, built on the
conceptual and engineering foundations laid by the FieldML specification
and the C++ FieldML-API. We gratefully acknowledge its original authors —
Caton Little, Alan Wu, Richard Christie, and Andrew Miller, at Auckland
Uniservices Ltd / the Auckland Bioengineering Institute — and the broader
Physiome Project community.

We also thank the authors and maintainers of the scientific-Python
libraries `pyfieldml` interoperates with: `numpy` [@numpy], `scipy` [@scipy],
`lxml`, `h5py` [@h5py], `meshio`, `PyVista`, and `scikit-fem`.

# References
