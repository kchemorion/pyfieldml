# pyfieldml

[![CI](https://github.com/kchemorion/pyfieldml/actions/workflows/ci.yml/badge.svg)](https://github.com/kchemorion/pyfieldml/actions/workflows/ci.yml)
[![Docs](https://github.com/kchemorion/pyfieldml/actions/workflows/docs.yml/badge.svg)](https://kchemorion.github.io/pyfieldml/)
[![PyPI](https://img.shields.io/pypi/v/pyfieldml.svg)](https://pypi.org/project/pyfieldml/)
[![Python](https://img.shields.io/pypi/pyversions/pyfieldml.svg)](https://pypi.org/project/pyfieldml/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A modern, pure-Python implementation of [FieldML](https://physiomeproject.org/software/fieldml) 0.5 with a full evaluation engine, interop bridges (`meshio`, PyVista, XDMF, scikit-fem, OpenSim-adjacent), and a curated biomechanics model zoo.

FieldML is the [Physiome Project](https://physiomeproject.org/)'s declarative markup language for representing mathematical fields over discrete meshes — used across computational physiology (cardiac, musculoskeletal, respiratory modeling). The original C++ [FieldML-API](https://github.com/kchemorion/FieldML-API) has been effectively unmaintained since 2015. `pyfieldml` is an independent reimplementation that brings FieldML into the modern scientific-Python ecosystem.

> **Status:** v1.0 — feature-complete for Phase-1 through Phase-5 scope. See the
> [design spec](docs/superpowers/specs/2026-04-18-pyfieldml-design.md) and
> [CHANGELOG](CHANGELOG.md) for what's in and what's deferred to later minor
> releases.

## What it will do (v1.0)

- Full FieldML 0.5 read + write, round-trip validated against the C++ reference test suite
- Legacy read of FieldML 0.3 and 0.4 (auto up-conversion to 0.5)
- **Evaluation engine** with Lagrange (orders 1–2) and cubic Hermite-with-scaling bases
- `Field.evaluate(element, xi)`, vectorized `Field.sample(points)`, Jacobians
- High-level builders: `add_lagrange_mesh`, `add_fiber_field`, `add_material_field`, `add_landmark_set`, …
- Interop: `meshio` (two-way), PyVista (`doc.plot()`, `doc.explore()`), XDMF, `scikit-fem`, OpenSim-compatible asset export
- CLI: `pyfieldml inspect | validate | convert | plot | lint | diff`
- Curated model zoo: `pyfieldml.datasets.load_femur()`, `load_rectus_femoris()`, `load_myocardium()`
- Docs site with `jupyter-lite` in-browser runnable notebooks
- JOSS paper submitted alongside v1.0

## License

Apache 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

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
from pyfieldml import datasets

# Load a bundled synthetic dataset
doc = datasets.load_rectus_femoris()

# Inspect the evaluator graph
for name, ev in doc.evaluators.items():
    print(f"{name:30s}  {type(ev).__name__}")

# Evaluate the coordinate field at an element centroid
coords = doc.field("coordinates")
print("centroid of element 1:", coords.evaluate(element=1, xi=(0.25, 0.25, 0.25)))

# Export to VTK for ParaView
m = doc.to_meshio()
```

## Cite this work

If you use `pyfieldml` in academic work, please cite it — see
[`CITATION.cff`](CITATION.cff) for the canonical metadata, and
[`docs/cite.md`](docs/cite.md) for BibTeX snippets.

## Acknowledgments

`pyfieldml` is an independent Python reimplementation inspired by and validated against the C++ [FieldML-API](https://github.com/kchemorion/FieldML-API). Credit to its original authors — Caton Little, Alan Wu, Richard Christie, Andrew Miller, and Auckland Uniservices Ltd / the Auckland Bioengineering Institute — and to the Physiome Project community that maintains the FieldML specification.
