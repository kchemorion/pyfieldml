# pyfieldml

A modern, pure-Python implementation of [FieldML](https://physiomeproject.org/software/fieldml) 0.5 with a full evaluation engine, interop bridges (`meshio`, PyVista, XDMF, scikit-fem, OpenSim-adjacent), and a curated biomechanics model zoo.

FieldML is the [Physiome Project](https://physiomeproject.org/)'s declarative markup language for representing mathematical fields over discrete meshes — used across computational physiology (cardiac, musculoskeletal, respiratory modeling). The original C++ [FieldML-API](https://github.com/kchemorion/FieldML-API) has been effectively unmaintained since 2015. `pyfieldml` is an independent reimplementation that brings FieldML into the modern scientific-Python ecosystem.

> **Status: design phase.** The approved design spec lives at
> [`docs/superpowers/specs/2026-04-18-pyfieldml-design.md`](docs/superpowers/specs/2026-04-18-pyfieldml-design.md).
> Implementation plan and phased build are next.

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

doc = fml.read("model.fieldml")
print(doc.source_version)          # "0.5.0", "0.4", "0.3"
doc.validate()                     # XSD validation
doc.write("roundtripped.fieldml")
```

> Phase 0 ships I/O and DOM-level round-trip. The semantic object model, evaluation engine, interop bridges, and model zoo arrive in later phases — see the [design spec](docs/superpowers/specs/2026-04-18-pyfieldml-design.md) and [plans](docs/superpowers/plans/).

## Acknowledgments

`pyfieldml` is an independent Python reimplementation inspired by and validated against the C++ [FieldML-API](https://github.com/kchemorion/FieldML-API). Credit to its original authors — Caton Little, Alan Wu, Richard Christie, Andrew Miller, and Auckland Uniservices Ltd / the Auckland Bioengineering Institute — and to the Physiome Project community that maintains the FieldML specification.
