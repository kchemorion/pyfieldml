# pyfieldml Phase 4 — Interop Bridges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.
>
> **⚠ Forward-draft notice:** Depends on Phases 1–3.

**Goal:** Wire `pyfieldml` into the surrounding scientific-Python ecosystem through five interop bridges plus a Jupyter graph-explorer widget. Each bridge is independently-installable via an optional extra. Tag `v0.4.0`.

**Architecture:** Each bridge lives under `pyfieldml.interop.*` as a single module with a narrow public API (1–5 functions). `Document` / `Field` methods that cross an interop boundary (`doc.plot()`, `doc.to_meshio()`, etc.) lazy-import the corresponding module and raise a crisp `ImportError` pointing to the right extra if missing.

**Tech Stack:** `meshio` (optional), `pyvista` + `trame` + `ipywidgets` + `ipycytoscape` (optional), `scikit-fem` (optional). No new required deps — all bridges are extras.

**Spec reference:** §2.2 Pillar B, §5 (per-bridge details).

**Completion criteria:**
- `meshio.read("x.fieldml")` works after `pip install pyfieldml[meshio]`.
- `doc.to_meshio()` / `Document.from_meshio()` round-trip: points, connectivity, field names, and per-point/per-cell values preserved for Lagrange meshes.
- `doc.plot()` renders an `UnstructuredGrid` in a Jupyter notebook via trame; `field.plot()` handles scalar/vector/fiber cases.
- `doc.explore()` opens the Jupyter graph-browser widget; clicking a node shows its detail.
- `xdmf.write(doc, "model.xdmf")` writes a ParaView-readable XDMF+HDF5 pair.
- `scikit_fem.to_scikit_fem(mesh)` returns a scikit-fem `Mesh` + `Basis`; round-trip a Laplace assembly end-to-end.
- OpenSim-adjacent export produces valid `.vtp` geometry files and a landmark CSV.
- `v0.4.0` tagged + published.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `src/pyfieldml/interop/__init__.py` | Namespace placeholder — nothing imported eagerly |
| `src/pyfieldml/interop/meshio.py` | meshio two-way + entry-point registration |
| `src/pyfieldml/interop/pyvista.py` | PyVista converter + `doc.plot`/`field.plot` backends |
| `src/pyfieldml/interop/xdmf.py` | XDMF writer |
| `src/pyfieldml/interop/scikit_fem.py` | scikit-fem basis adapter + mesh/basis bridge |
| `src/pyfieldml/interop/opensim.py` | OpenSim-adjacent export |
| `src/pyfieldml/viz/__init__.py` | Re-exports graph explorer |
| `src/pyfieldml/viz/explorer.py` | Jupyter graph-explorer widget |
| `src/pyfieldml/viz/plot.py` | High-level plot dispatch (calls into interop.pyvista) |
| `tests/unit/test_interop_meshio.py` | meshio round-trip tests |
| `tests/unit/test_interop_xdmf.py` | XDMF smoke tests |
| `tests/unit/test_interop_pyvista.py` | PyVista conversion tests (skipped if pyvista not installed) |
| `tests/unit/test_interop_scikit_fem.py` | scikit-fem bridge tests (skipped if scikit-fem not installed) |
| `tests/unit/test_interop_opensim.py` | OpenSim export tests |

---

## Task 1 — meshio: FieldML → meshio

**Files:**
- Create: `src/pyfieldml/interop/__init__.py` (empty docstring)
- Create: `src/pyfieldml/interop/meshio.py` (read path first)
- Test: `tests/unit/test_interop_meshio.py`

- [ ] **Step 1: Test**

```python
"""Tests for the meshio interop bridge."""

from __future__ import annotations

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")

import pyfieldml as fml


def test_to_meshio_preserves_points_and_cells() -> None:
    from tests.unit.test_field_evaluate import _make_unit_cube_doc

    doc = _make_unit_cube_doc()
    m = doc.to_meshio()
    assert m.points.shape == (8, 3)
    assert len(m.cells) == 1
    block = m.cells[0]
    assert block.type == "hexahedron"
    # meshio cells are 0-indexed; pyfieldml connectivity is 1-indexed internally.
    assert block.data.shape == (1, 8)


def test_to_meshio_carries_node_data() -> None:
    from tests.unit.test_field_evaluate import _make_unit_cube_doc

    doc = _make_unit_cube_doc()
    m = doc.to_meshio()
    assert "coordinates" in m.point_data
    np.testing.assert_array_equal(m.point_data["coordinates"], doc.field("coordinates").as_ndarray())
```

- [ ] **Step 2: Implement `src/pyfieldml/interop/meshio.py` (read path)**

```python
"""meshio interop — two-way conversion.

Call `to_meshio(doc)` to export a FieldML Document as a meshio.Mesh. Call
`from_meshio(mesh, name=...)` to import a meshio.Mesh back into a FieldML
Document. Lossy in both directions: meshio has no concept of evaluator
graphs, and FieldML's arbitrary-basis machinery does not map cleanly to
meshio's fixed cell types.

Register this module with meshio via the ``meshio.plugins`` entry point
(see pyproject.toml). Users can then call ``meshio.read("x.fieldml")`` and
``meshio.write("x.fieldml", mesh)`` directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Cell-type mapping. Topology + order → meshio cell-type string.
FIELDML_TO_MESHIO: dict[tuple[str, int], str] = {
    ("line", 1): "line",
    ("line", 2): "line3",
    ("triangle", 1): "triangle",
    ("triangle", 2): "triangle6",
    ("quad", 1): "quad",
    ("quad", 2): "quad9",
    ("tet", 1): "tetra",
    ("tet", 2): "tetra10",
    ("hex", 1): "hexahedron",
    ("hex", 2): "hexahedron27",
    ("wedge", 1): "wedge",
}

MESHIO_TO_FIELDML: dict[str, tuple[str, int]] = {v: k for k, v in FIELDML_TO_MESHIO.items()}


def to_meshio(doc: Any) -> Any:
    """Convert a pyfieldml.Document to a meshio.Mesh (primary mesh only)."""
    import meshio

    region = doc.region
    # Pick the first MeshType (Phase 4 simplification).
    if not region.meshes:
        raise ValueError("Document has no MeshType; cannot convert to meshio.")
    mesh_name, mesh_type = next(iter(region.meshes.items()))

    # Find the coordinates field and the connectivity for this mesh.
    coord_name = _find_coord_evaluator(region, mesh_name)
    if coord_name is None:
        raise ValueError(f"No coordinate evaluator found for mesh {mesh_name!r}")
    coords_arr = doc.field(coord_name).as_ndarray()

    # Connectivity is stored as `<coord>.connectivity`.
    conn_name = f"{coord_name}.connectivity"
    if conn_name not in region.evaluators:
        raise ValueError(f"No connectivity parameter {conn_name!r}")
    connectivity = region.evaluators[conn_name].as_ndarray()
    # Convert 1-indexed FieldML connectivity to 0-indexed meshio.
    connectivity = connectivity.astype(np.int64) - 1

    # Infer cell type from the basis registered in the region.
    basis_name = _find_basis_name(region)
    topology, order = _basis_topology_order(basis_name)
    cell_type = FIELDML_TO_MESHIO[(topology, order)]

    point_data = {coord_name: coords_arr}
    # Extra ParameterEvaluators that look like per-node fields.
    for ev_name, ev in region.evaluators.items():
        if ev_name in {coord_name, conn_name, basis_name}:
            continue
        if hasattr(ev, "as_ndarray"):
            arr = ev.as_ndarray()
            if arr.shape[0] == coords_arr.shape[0]:
                point_data[ev_name] = arr

    return meshio.Mesh(
        points=coords_arr,
        cells=[(cell_type, connectivity)],
        point_data=point_data,
    )


def _find_coord_evaluator(region: Any, mesh_name: str) -> str | None:
    # Heuristic: first ParameterEvaluator whose value-type has component_count == mesh.dimension.
    from pyfieldml.model.evaluators import ParameterEvaluator
    from pyfieldml.model.types import ContinuousType

    mesh = region.meshes[mesh_name]
    for name, ev in region.evaluators.items():
        if isinstance(ev, ParameterEvaluator) and isinstance(ev.value_type, ContinuousType):
            if ev.value_type.component_count == mesh.dimension:
                return name
    return None


def _find_basis_name(region: Any) -> str:
    for name in region.evaluators:
        if name.startswith("library.basis."):
            return name
    raise ValueError("No library.basis.* evaluator in region")


def _basis_topology_order(basis_name: str) -> tuple[str, int]:
    # Expected format: library.basis.<order>_<kind>.<topology>
    # e.g. library.basis.linear_lagrange.hex → (hex, 1)
    parts = basis_name.split(".")
    order_tag = parts[2].split("_")[0]
    topology = parts[3]
    order_map = {"linear": 1, "quadratic": 2, "cubic": 3}
    return topology, order_map[order_tag]
```

- [ ] **Step 3: Add `Document.to_meshio` method**

In `src/pyfieldml/document.py`:

```python
    def to_meshio(self) -> Any:
        try:
            from pyfieldml.interop.meshio import to_meshio as _to_meshio
        except ImportError as exc:
            raise ImportError(
                "to_meshio() requires the [meshio] extra: pip install pyfieldml[meshio]"
            ) from exc
        return _to_meshio(self)
```

- [ ] **Step 4: Run, pass**

Run: `uv run pytest tests/unit/test_interop_meshio.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/pyfieldml/interop/ src/pyfieldml/document.py tests/unit/test_interop_meshio.py
git commit -m "feat(interop): meshio one-way export (FieldML → meshio.Mesh)"
```

---

## Task 2 — meshio: meshio → FieldML

- [ ] **Step 1: Test**

```python
def test_from_meshio_round_trips_unit_cube() -> None:
    import meshio
    points = np.array(
        [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]],
        dtype=np.float64,
    )
    conn = np.array([[0,1,2,3,4,5,6,7]], dtype=np.int64)
    m = meshio.Mesh(points=points, cells=[("hexahedron", conn)])
    doc = fml.Document.from_meshio(m, name="imported")
    assert doc.field("coordinates").as_ndarray().shape == (8, 3)
```

- [ ] **Step 2: Implement `from_meshio` in `interop/meshio.py`**

```python
def from_meshio(m: Any, *, name: str = "imported") -> Any:
    """Convert a meshio.Mesh to a pyfieldml.Document.

    Only the first cell block is consumed. meshio meshes with multiple cell
    types are flattened to the first; this is a deliberate Phase-4 limitation.
    """
    import pyfieldml as fml
    from pyfieldml.builders.mesh import add_lagrange_mesh
    from pyfieldml.model.region import Region

    if not m.cells:
        raise ValueError("meshio mesh has no cell blocks")

    block = m.cells[0]
    topology, order = MESHIO_TO_FIELDML[block.type]
    connectivity = block.data.astype(np.int64) + 1  # 0-indexed → 1-indexed

    r = Region(name=name)
    add_lagrange_mesh(
        r, name=f"{name}_mesh",
        nodes=m.points.astype(np.float64),
        elements=connectivity,
        topology=topology, order=order,
        coord_name="coordinates",
    )
    return fml.Document.from_region(r)
```

- [ ] **Step 3: Add `Document.from_meshio` classmethod** (delegates to `interop.meshio`).

- [ ] **Step 4: Run, pass + commit**

```bash
git commit -am "feat(interop): meshio → FieldML import path"
```

---

## Task 3 — meshio entry-point registration

- [ ] **Step 1: Add to `pyproject.toml`**

```toml
[project.entry-points."meshio"]
".fieldml" = "pyfieldml.interop.meshio:_register"
```

(Note: meshio's exact entry-point group name varies by version. Consult
installed `meshio`'s `pyproject.toml`. Fallback mechanism: explicit
`pyfieldml.interop.meshio.register()` call.)

- [ ] **Step 2: Implement `_register` + reader/writer adapters**

```python
def _register() -> None:
    import meshio
    meshio.register_format("fieldml", [".fieldml"], _reader, _writer)


def _reader(path: Any) -> Any:
    import pyfieldml as fml
    return to_meshio(fml.read(path))


def _writer(path: Any, mesh: Any) -> None:
    import pyfieldml as fml
    doc = fml.Document.from_meshio(mesh, name="imported")
    doc.write(path)
```

- [ ] **Step 3: Test registration**

```python
def test_meshio_read_dispatches_through_pyfieldml(fixtures_dir, tmp_path) -> None:
    import meshio
    from pyfieldml.interop.meshio import _register

    _register()
    # Smoke: meshio.read on a .fieldml file does not raise.
    path = tmp_path / "cube.fieldml"
    from tests.unit.test_field_evaluate import _make_unit_cube_doc
    _make_unit_cube_doc().write(path)
    m = meshio.read(path)
    assert m.points.shape == (8, 3)
```

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/interop/meshio.py pyproject.toml tests/unit/test_interop_meshio.py
git commit -m "feat(interop): register pyfieldml as a meshio format plugin"
```

---

## Task 4 — PyVista converter + `doc.plot()`

**Files:**
- Create: `src/pyfieldml/interop/pyvista.py`
- Create: `src/pyfieldml/viz/plot.py`
- Modify: `src/pyfieldml/document.py` (add `plot`)
- Modify: `src/pyfieldml/eval/field.py` (add `plot` to Field)
- Test: `tests/unit/test_interop_pyvista.py`

- [ ] **Step 1: Test**

```python
"""PyVista interop tests. Skipped if pyvista is not installed."""

from __future__ import annotations

import pytest

pv = pytest.importorskip("pyvista")


def test_to_pyvista_produces_unstructured_grid() -> None:
    from tests.unit.test_field_evaluate import _make_unit_cube_doc
    from pyfieldml.interop.pyvista import to_pyvista

    doc = _make_unit_cube_doc()
    grid = to_pyvista(doc)
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_points == 8
    assert grid.n_cells == 1
    assert "coordinates" in grid.point_data
```

- [ ] **Step 2: Implement `src/pyfieldml/interop/pyvista.py`**

```python
"""PyVista interop — FieldML → VTK UnstructuredGrid."""

from __future__ import annotations

from typing import Any

import numpy as np

# VTK cell type codes for the topologies we support.
VTK_CELL_TYPE: dict[tuple[str, int], int] = {
    ("line", 1):      3,    # VTK_LINE
    ("line", 2):      21,   # VTK_QUADRATIC_EDGE
    ("triangle", 1):  5,    # VTK_TRIANGLE
    ("triangle", 2):  22,   # VTK_QUADRATIC_TRIANGLE
    ("quad", 1):      9,    # VTK_QUAD
    ("quad", 2):      28,   # VTK_BIQUADRATIC_QUAD
    ("tet", 1):       10,   # VTK_TETRA
    ("tet", 2):       24,   # VTK_QUADRATIC_TETRA
    ("hex", 1):       12,   # VTK_HEXAHEDRON
    ("hex", 2):       29,   # VTK_TRIQUADRATIC_HEXAHEDRON
    ("wedge", 1):     13,   # VTK_WEDGE
}


def to_pyvista(doc: Any) -> Any:
    """Convert a pyfieldml.Document to a PyVista UnstructuredGrid."""
    import pyvista as pv
    from pyfieldml.interop.meshio import (
        _basis_topology_order,
        _find_basis_name,
        _find_coord_evaluator,
    )

    region = doc.region
    mesh_name = next(iter(region.meshes))
    coord_name = _find_coord_evaluator(region, mesh_name)
    if coord_name is None:
        raise ValueError("No coordinate evaluator found")
    coords = doc.field(coord_name).as_ndarray()
    basis_name = _find_basis_name(region)
    topology, order = _basis_topology_order(basis_name)
    conn = region.evaluators[f"{coord_name}.connectivity"].as_ndarray().astype(np.int64) - 1

    n_per_cell = conn.shape[1]
    cells = np.hstack([np.full((conn.shape[0], 1), n_per_cell), conn]).astype(np.int64)
    celltypes = np.full(conn.shape[0], VTK_CELL_TYPE[(topology, order)])

    grid = pv.UnstructuredGrid(cells.ravel(), celltypes, coords.astype(np.float64))
    grid.point_data[coord_name] = coords

    # Include additional per-node ParameterEvaluator arrays.
    from pyfieldml.model.evaluators import ParameterEvaluator

    for name, ev in region.evaluators.items():
        if name in {coord_name, f"{coord_name}.connectivity", basis_name}:
            continue
        if isinstance(ev, ParameterEvaluator) and ev.as_ndarray().shape[0] == coords.shape[0]:
            grid.point_data[name] = ev.as_ndarray()

    return grid


def plot_doc(doc: Any, **kwargs: Any) -> Any:
    grid = to_pyvista(doc)
    return grid.plot(**kwargs)


def plot_field(field: Any, **kwargs: Any) -> Any:
    """Shape-aware field plot — scalar/vector/fiber dispatch."""
    arr = field.as_ndarray()
    grid = to_pyvista(field._region_doc)  # internal handle set by Field
    if arr.ndim == 1 or arr.shape[-1] == 1:
        grid.point_data[field.name] = arr
        return grid.plot(scalars=field.name, **kwargs)
    if arr.shape[-1] == 3:
        # Treat as vector — glyph.
        grid.point_data[field.name] = arr
        import pyvista as pv
        mesh_poly = grid.glyph(orient=field.name, scale=False, factor=0.1)
        plotter = pv.Plotter()
        plotter.add_mesh(grid, opacity=0.3)
        plotter.add_mesh(mesh_poly, color="red")
        return plotter.show(**kwargs)
    raise ValueError(f"Unsupported field shape for plot: {arr.shape}")
```

- [ ] **Step 3: Wire `doc.plot()` and `field.plot()` as thin shims.** (In `document.py` and `field.py`.)

- [ ] **Step 4: Run, pass** (skipped if `pyvista` not installed locally).

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(interop): pyvista converter + doc.plot() / field.plot()"
```

---

## Task 5 — Jupyter graph-explorer widget

**Files:**
- Create: `src/pyfieldml/viz/explorer.py`

- [ ] **Step 1: Implement `doc.explore()`**

```python
"""Jupyter widget: browse the evaluator graph of a FieldML document."""

from __future__ import annotations

from typing import Any


def explore(doc: Any) -> Any:
    """Return an ipywidgets layout displaying the evaluator graph.

    Three panes:
      - Tree of evaluators grouped by type.
      - Graph visualization (ipycytoscape).
      - Detail panel showing the selected node's bindings / shape / dtype.
    """
    try:
        import ipywidgets as widgets
        import ipycytoscape
    except ImportError as exc:
        raise ImportError(
            "doc.explore() requires pyfieldml[viz]: pip install pyfieldml[viz]"
        ) from exc

    from pyfieldml.model.evaluators import ReferenceEvaluator

    region = doc.region

    # Build ipycytoscape node/edge lists.
    nodes = [
        {"data": {"id": n, "label": n, "kind": type(ev).__name__}}
        for n, ev in region.evaluators.items()
    ]
    edges = []
    for name, ev in region.evaluators.items():
        if isinstance(ev, ReferenceEvaluator):
            edges.append({"data": {"source": name, "target": ev.source.name}})

    graph = ipycytoscape.CytoscapeWidget()
    graph.graph.add_graph_from_json({"nodes": nodes, "edges": edges})

    tree = widgets.SelectMultiple(
        options=list(region.evaluators.keys()),
        description="Evaluators:",
        rows=12,
    )
    detail = widgets.HTML("<i>Select a node to see details.</i>")

    def _on_select(change: Any) -> None:
        name = change["new"][0] if change["new"] else None
        if name is None:
            detail.value = "<i>No selection</i>"
            return
        ev = region.evaluators[name]
        detail.value = (
            f"<b>{name}</b><br>"
            f"Kind: {type(ev).__name__}<br>"
            f"Value type: {getattr(ev.value_type, 'name', '—')}<br>"
        )

    tree.observe(_on_select, names="value")
    return widgets.HBox([tree, graph, detail])
```

- [ ] **Step 2: Add `Document.explore()` shim** (delegates to `viz.explorer.explore`).

- [ ] **Step 3: Manual verification in Jupyter**

Write a quick notebook in `docs/examples/explore_widget_smoke.ipynb` that
opens the widget on the unit-cube document. (Not CI-executed — requires
ipycytoscape JS assets that don't render headless.)

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/viz/
git commit -m "feat(viz): Jupyter evaluator-graph explorer widget"
```

---

## Task 6 — XDMF writer

**Files:**
- Create: `src/pyfieldml/interop/xdmf.py`
- Test: `tests/unit/test_interop_xdmf.py`

- [ ] **Step 1: Test**

```python
"""XDMF interop tests."""

from __future__ import annotations

import os
from pathlib import Path

from lxml import etree


def test_xdmf_write_produces_well_formed_file(tmp_path: Path) -> None:
    from tests.unit.test_field_evaluate import _make_unit_cube_doc
    from pyfieldml.interop.xdmf import write_xdmf

    doc = _make_unit_cube_doc()
    xdmf = tmp_path / "cube.xdmf"
    write_xdmf(doc, xdmf)
    # File exists and is valid XML.
    etree.parse(xdmf)
    # Heavy-data companion HDF5 exists.
    h5 = tmp_path / "cube.h5"
    assert h5.is_file()
```

- [ ] **Step 2: Implement**

```python
"""Write FieldML documents as XDMF3 + HDF5 for ParaView / VisIt consumption."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from lxml import etree

# XDMF cell-type names, keyed by (topology, order).
XDMF_TOPOLOGY_TYPE = {
    ("tet", 1):  "Tetrahedron",
    ("hex", 1):  "Hexahedron",
    ("tri", 1):  "Triangle",
    ("quad", 1): "Quadrilateral",
    ("line", 1): "Polyline",
    ("wedge", 1): "Wedge",
}


def write_xdmf(doc: Any, path: Any) -> None:
    from pyfieldml.interop.meshio import (
        _basis_topology_order,
        _find_basis_name,
        _find_coord_evaluator,
    )
    from pyfieldml.model.evaluators import ParameterEvaluator

    path = Path(path)
    h5_path = path.with_suffix(".h5")
    region = doc.region
    mesh_name = next(iter(region.meshes))
    coord_name = _find_coord_evaluator(region, mesh_name)
    coords = doc.field(coord_name).as_ndarray()
    basis_name = _find_basis_name(region)
    topology, order = _basis_topology_order(basis_name)
    conn = region.evaluators[f"{coord_name}.connectivity"].as_ndarray().astype(np.int64) - 1

    # Write heavy data to HDF5.
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("/coords", data=coords)
        f.create_dataset("/conn", data=conn)
        for name, ev in region.evaluators.items():
            if name in {coord_name, f"{coord_name}.connectivity", basis_name}:
                continue
            if isinstance(ev, ParameterEvaluator):
                arr = ev.as_ndarray()
                if arr.shape[0] == coords.shape[0]:
                    f.create_dataset(f"/{name}", data=arr)

    # Write XDMF descriptor.
    xdmf = etree.Element("Xdmf", Version="3.0")
    domain = etree.SubElement(xdmf, "Domain")
    grid = etree.SubElement(domain, "Grid", Name=mesh_name, GridType="Uniform")
    topo = etree.SubElement(
        grid, "Topology",
        TopologyType=XDMF_TOPOLOGY_TYPE[(topology, order)],
        NumberOfElements=str(conn.shape[0]),
    )
    etree.SubElement(
        topo, "DataItem",
        Format="HDF", Dimensions=f"{conn.shape[0]} {conn.shape[1]}", NumberType="Int", Precision="8",
    ).text = f"{h5_path.name}:/conn"
    geom = etree.SubElement(grid, "Geometry", GeometryType="XYZ")
    etree.SubElement(
        geom, "DataItem",
        Format="HDF", Dimensions=f"{coords.shape[0]} {coords.shape[1]}", NumberType="Float", Precision="8",
    ).text = f"{h5_path.name}:/coords"
    for name, ev in region.evaluators.items():
        if name in {coord_name, f"{coord_name}.connectivity", basis_name}:
            continue
        if isinstance(ev, ParameterEvaluator):
            arr = ev.as_ndarray()
            if arr.shape[0] == coords.shape[0]:
                attr = etree.SubElement(
                    grid, "Attribute",
                    Name=name, AttributeType="Scalar" if arr.ndim == 1 else "Vector",
                    Center="Node",
                )
                etree.SubElement(
                    attr, "DataItem",
                    Format="HDF",
                    Dimensions=" ".join(str(d) for d in arr.shape),
                    NumberType="Float", Precision="8",
                ).text = f"{h5_path.name}:/{name}"

    etree.ElementTree(xdmf).write(
        str(path), pretty_print=True, xml_declaration=True, encoding="UTF-8"
    )
```

- [ ] **Step 3: Run, pass** — smoke test.

- [ ] **Step 4: Commit**

```bash
git add src/pyfieldml/interop/xdmf.py tests/unit/test_interop_xdmf.py
git commit -m "feat(interop): one-way XDMF writer (no extra deps)"
```

---

## Task 7 — scikit-fem bridge

**Files:**
- Create: `src/pyfieldml/interop/scikit_fem.py`
- Test: `tests/unit/test_interop_scikit_fem.py`

- [ ] **Step 1: Test** (skipped if scikit-fem not installed)

```python
"""scikit-fem bridge tests."""

from __future__ import annotations

import numpy as np
import pytest

skfem = pytest.importorskip("skfem")


def test_to_scikit_fem_unit_cube() -> None:
    from tests.unit.test_field_evaluate import _make_unit_cube_doc
    from pyfieldml.interop.scikit_fem import to_scikit_fem

    doc = _make_unit_cube_doc()
    mesh_name = list(doc.meshes)[0]
    m, basis = to_scikit_fem(doc, mesh_name=mesh_name)
    assert isinstance(m, skfem.MeshHex)
    assert m.nvertices == 8
```

- [ ] **Step 2: Implement**

```python
"""scikit-fem bridge — convert FieldML meshes to scikit-fem Mesh+Basis."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_scikit_fem(doc: Any, *, mesh_name: str | None = None) -> tuple[Any, Any]:
    """Return a (skfem.Mesh, skfem.Basis) pair for the named mesh (default: first)."""
    import skfem
    from pyfieldml.interop.meshio import (
        _basis_topology_order,
        _find_basis_name,
        _find_coord_evaluator,
    )

    region = doc.region
    mesh_name = mesh_name or next(iter(region.meshes))
    coord_name = _find_coord_evaluator(region, mesh_name)
    coords = doc.field(coord_name).as_ndarray()
    basis_name = _find_basis_name(region)
    topology, order = _basis_topology_order(basis_name)
    conn = region.evaluators[f"{coord_name}.connectivity"].as_ndarray().astype(np.int64) - 1

    SKFEM_MESH = {
        ("tet", 1): skfem.MeshTet,
        ("hex", 1): skfem.MeshHex,
        ("tri", 1): skfem.MeshTri,
        ("quad", 1): skfem.MeshQuad,
        ("line", 1): skfem.MeshLine,
    }
    SKFEM_ELEMENT = {
        ("tet", 1): skfem.ElementTetP1,
        ("hex", 1): skfem.ElementHex1,
        ("tri", 1): skfem.ElementTriP1,
        ("quad", 1): skfem.ElementQuad1,
        ("line", 1): skfem.ElementLineP1,
    }

    mesh_cls = SKFEM_MESH.get((topology, order))
    elem_cls = SKFEM_ELEMENT.get((topology, order))
    if mesh_cls is None or elem_cls is None:
        raise NotImplementedError(
            f"scikit-fem bridge: (topology={topology}, order={order}) not yet supported"
        )

    m = mesh_cls(coords.T, conn.T)
    basis = skfem.Basis(m, elem_cls())
    return m, basis
```

- [ ] **Step 3: Commit**

```bash
git add src/pyfieldml/interop/scikit_fem.py tests/unit/test_interop_scikit_fem.py
git commit -m "feat(interop): scikit-fem mesh+basis bridge"
```

---

## Task 8 — OpenSim-adjacent export

**Files:**
- Create: `src/pyfieldml/interop/opensim.py`
- Test: `tests/unit/test_interop_opensim.py`

- [ ] **Step 1: Implement**

```python
"""OpenSim-adjacent export: geometry (.vtp), landmarks (.csv/.osim-fragment), fiber polylines (.vtp).

Not a full .osim round-trip. These assets are intended to be hand-assembled
into an OpenSim model.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def export_geometry(doc: Any, out_dir: Any) -> list[Path]:
    """Write each mesh as a VTK PolyData (.vtp) file under out_dir. Returns paths."""
    import meshio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = doc.to_meshio()
    target = out_dir / "geometry.vtp"
    meshio.write(target, m, file_format="vtp")
    return [target]


def export_muscle_attachments(doc: Any, out_path: Any) -> Path:
    """Write landmark points as CSV: label, x, y, z."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Heuristic: find a ParameterEvaluator named "*landmark*" or "*attachment*".
    region = doc.region
    target = None
    for name, ev in region.evaluators.items():
        if "landmark" in name.lower() or "attachment" in name.lower():
            target = (name, ev.as_ndarray())
            break
    if target is None:
        raise ValueError("No landmark evaluator found in document")

    name, arr = target
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "x", "y", "z"])
        for i, row in enumerate(arr):
            w.writerow([f"{name}_{i}", *row])
    return out_path


def export_fiber_lines(
    doc: Any, *, mesh: str, field: str, out_path: Any, n_streamlines: int = 100
) -> Path:
    """Integrate streamlines through a fiber vector field and write as .vtp polylines.

    Phase-4 simplification: no true streamline integration; instead emit
    short line glyphs centered at each mesh node, aligned with the field.
    (True streamlines = PyVista `streamlines_from_source` — add in a later phase.)
    """
    import meshio

    out_path = Path(out_path)
    coords = doc.field("coordinates").as_ndarray()
    vectors = doc.evaluators[field].as_ndarray()
    # For each node, emit a line segment of length 0.01 * local bbox diagonal.
    diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    step = 0.01 * diag
    n = coords.shape[0]
    start = coords
    end = coords + vectors * step
    points = np.concatenate([start, end], axis=0)
    lines = np.arange(2 * n).reshape(2, n).T  # shape (n, 2), i.e. pairs
    m = meshio.Mesh(points=points, cells=[("line", lines)])
    meshio.write(out_path, m, file_format="vtp")
    return out_path
```

- [ ] **Step 2: Smoke test + commit**

```bash
git add src/pyfieldml/interop/opensim.py tests/unit/test_interop_opensim.py
git commit -m "feat(interop): OpenSim-adjacent geometry + landmark + fiber exports"
```

---

## Task 9 — Phase-4 closeout + v0.4.0

- [ ] **Step 1: Sanity gauntlet + docs update** (add API-reference pages for each interop module).
- [ ] **Step 2: CHANGELOG entry.**
- [ ] **Step 3: Version bump → 0.4.0, tag, push.**

## Phase-4 done-definition

- [ ] `meshio.read("x.fieldml")` works after `pip install pyfieldml[meshio]`.
- [ ] `doc.to_meshio()` / `from_meshio` round-trip unit-cube and rectus-femoris-shaped meshes.
- [ ] `doc.plot()` and `field.plot()` render in Jupyter (manually verified).
- [ ] `doc.explore()` renders the evaluator-graph widget.
- [ ] `write_xdmf(doc, "out.xdmf")` produces ParaView-readable output.
- [ ] `scikit_fem.to_scikit_fem(doc)` returns a usable Basis.
- [ ] OpenSim `.vtp` + landmarks CSV write correctly.
- [ ] `v0.4.0` on PyPI.

Next: **Phase 5 — builders + model zoo + notebooks** (biomech builders, 4 datasets, 8 tutorial notebooks + reproducibility, lint, diff, CLI).
