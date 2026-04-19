# pyfieldml

A modern pure-Python implementation of [FieldML](https://physiomeproject.org/software/fieldml) 0.5 with a full evaluation engine, interop bridges, and a curated biomechanics model zoo.

!!! info "Version 1.0.0"
    Feature-complete implementation across the core FieldML 0.5 scope:
    read/write (including 0.3/0.4 → 0.5 up-conversion), a typed semantic model,
    evaluator-graph traversal with Lagrange (orders 1–2) and cubic-Hermite bases,
    spatial sampling, interop bridges to meshio / PyVista / XDMF / scikit-fem, a
    CLI, and three bundled synthetic datasets. See the [CHANGELOG](https://github.com/kchemorion/pyfieldml/blob/main/CHANGELOG.md) for what's in scope for later releases.

## Install

```bash
pip install pyfieldml
```

For interop bridges:

```bash
pip install 'pyfieldml[all]'    # everything
pip install 'pyfieldml[meshio]' # just meshio
pip install 'pyfieldml[viz]'    # PyVista + Jupyter explorer
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

See [Getting Started](getting-started.md) for a full walkthrough.
