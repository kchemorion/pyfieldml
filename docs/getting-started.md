# Getting started

## Install

```bash
pip install pyfieldml
```

For optional interop bridges:

```bash
pip install 'pyfieldml[all]'       # everything
pip install 'pyfieldml[meshio]'    # .fieldml ↔ meshio.Mesh + plugin
pip install 'pyfieldml[viz]'       # PyVista plotting + Jupyter explorer
pip install 'pyfieldml[scikit-fem]' # scikit-fem Mesh + Basis
```

For development:

```bash
git clone https://github.com/kchemorion/pyfieldml
cd pyfieldml
uv sync --extra dev
uv run pytest
```

## Loading a document

Three ways to produce a `Document`:

### From a file

```python
import pyfieldml as fml

doc = fml.read("path/to/model.fieldml")
print(doc.source_version)  # "0.5.0" or the legacy version if up-converted
```

FieldML 0.3 and 0.4 documents are transparently up-converted to 0.5 shape
on read; the original version is still accessible via `source_version`.

### From the bundled zoo

```python
from pyfieldml import datasets

doc = datasets.load_unit_cube()          # bundled in the wheel
doc = datasets.load_femur()              # synthetic cylinder + BMD material
doc = datasets.load_rectus_femoris()     # synthetic spindle + fiber field
```

### Programmatically

```python
import numpy as np
from pyfieldml import Document
from pyfieldml.builders.mesh import add_lagrange_mesh
from pyfieldml.builders.biomech import add_fiber_field, add_material_field
from pyfieldml.model.region import Region

r = Region(name="my_mesh")
nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                  [0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=np.float64)
elems = np.array([[1,2,3,4,5,6,7,8]], dtype=np.int64)
add_lagrange_mesh(r, name="m", nodes=nodes, elements=elems,
                  topology="hex", order=1, coord_name="coords")
add_fiber_field(r, mesh_name="m", vectors=np.tile([1,0,0], (8,1)))
doc = Document.from_region(r)
doc.write("my_mesh.fieldml")
```

## Inspecting a document

```python
# Typed mapping views
doc.continuous     # Mapping[str, ContinuousType]
doc.ensembles      # Mapping[str, EnsembleType]
doc.meshes         # Mapping[str, MeshType]
doc.evaluators     # Mapping[str, Evaluator]

# Or walk the full graph
for name, ev in doc.evaluators.items():
    print(name, type(ev).__name__)
```

Interactive (in Jupyter, with `[viz]` installed):

```python
doc.explore()   # ipywidgets + ipycytoscape graph browser
```

## Evaluating fields

```python
coords = doc.field("coordinates")

# Point evaluation
v = coords.evaluate(element=5, xi=(0.25, 0.5, 0.75))

# Vectorized
xis = np.random.rand(10_000, 3)
elems = np.full(10_000, 5)
vals = coords.evaluate(element=elems, xi=xis)

# Jacobian
J = coords.jacobian(element=5, xi=(0.25, 0.5, 0.75))

# Spatial sampling (physical points → interpolated values)
pts = np.random.rand(100, 3)
values = coords.sample(pts)  # NaN for points outside the mesh
```

## Interop

```python
# meshio (two-way; meshio.read("*.fieldml") also works via plugin)
m = doc.to_meshio()
doc2 = Document.from_meshio(m, name="from_meshio")

# PyVista
grid = doc.plot()   # interactive viewer
field.plot(doc=doc)  # with glyphs for vector fields

# XDMF (one-way, for ParaView / VisIt)
from pyfieldml.interop.xdmf import write_xdmf
write_xdmf(doc, "out.xdmf")

# scikit-fem
from pyfieldml.interop.scikit_fem import to_scikit_fem
mesh, basis = to_scikit_fem(doc)

# OpenSim-adjacent export
from pyfieldml.interop.opensim import export_geometry, export_fiber_lines
export_geometry(doc, "out_dir/")
export_fiber_lines(doc, field="fiber_direction", out_path="fibers.vtu")
```

## CLI

```
pyfieldml inspect model.fieldml         # summary tree
pyfieldml validate model.fieldml        # XSD check
pyfieldml convert model.fieldml --to vtu -o m.vtu
pyfieldml plot model.fieldml            # PyVista viewer
pyfieldml lint model.fieldml            # orphans, inverted elements
pyfieldml diff a.fieldml b.fieldml      # semantic diff
pyfieldml bench model.fieldml --field coordinates
```

## Next steps

- [FAQ](faq.md)
- [Notebooks](notebooks/01_quickstart.ipynb) (quickstart, evaluator graph, muscle fibers)
- [API reference](api/index.md)
- [Cite this work](cite.md)
