"""Author the Phase-5 tutorial notebooks.

Re-run to regenerate:
    uv run python docs/notebooks/build_notebooks.py

Cell sources are kept ruff-clean (double quotes, no unused imports, import
blocks separated by a blank line) so regeneration is idempotent under the
pre-commit hook set.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def notebook(cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    # Strip outputs and assign deterministic cell ids so regeneration is
    # idempotent with the nbstripout pre-commit hook.
    for i, c in enumerate(nb["cells"]):
        c["id"] = str(i)
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    return nb


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    # Ruff's notebook format drops the trailing newline from the last line
    # of a cell. Match that so regeneration is a no-op under pre-commit.
    return nbf.v4.new_code_cell(text.rstrip("\n"))


# -----------------------------------------------------------------------
# 01 Quickstart
# -----------------------------------------------------------------------
qs = notebook(
    [
        md(
            "# pyfieldml Quickstart\n\n"
            "Five-minute demo: install, load a dataset, inspect, evaluate, and export."
        ),
        md(
            "## Install\n\n"
            "This notebook assumes `pyfieldml` is already installed:\n\n"
            "```bash\n"
            "pip install pyfieldml[meshio]\n"
            "```"
        ),
        code(
            "from pyfieldml import datasets\n"
            "\n"
            "doc = datasets.load_unit_cube()\n"
            'print("Version :", doc.source_version)\n'
            'print("Region  :", doc.region.name)'
        ),
        md("## Inspect the evaluator graph"),
        code(
            'for name, ev in doc.evaluators.items():\n    print(f"{name:30s}  {type(ev).__name__}")'
        ),
        md("## Extract coordinates as NumPy"),
        code(
            'coords_ev = doc.evaluators["coordinates"]\n'
            "xyz = coords_ev.as_ndarray()\n"
            'print("coords shape:", xyz.shape)\n'
            'print("coords:")\n'
            "print(xyz)"
        ),
        md("## Evaluate the coordinate field at the element centroid"),
        code(
            'coords = doc.field("coordinates")\n'
            "centroid = coords.evaluate(element=1, xi=(0.5, 0.5, 0.5))\n"
            'print("centroid:", centroid)'
        ),
        md("## Export to meshio (requires [meshio] extra)"),
        code(
            "try:\n"
            "    m = doc.to_meshio()\n"
            '    print("meshio cell type:", m.cells[0].type, "count:", len(m.cells[0].data))\n'
            "except ImportError as exc:\n"
            '    print("meshio not installed; install with: pip install pyfieldml[meshio]")\n'
            "    print(exc)"
        ),
        md(
            "### Next steps\n\n"
            "- See `02_evaluator_graph.ipynb` for a tour of the evaluator hierarchy.\n"
            "- See `04_muscle_fibers.ipynb` for a fiber-field workflow."
        ),
    ]
)
nbf.write(qs, HERE / "01_quickstart.ipynb")


# -----------------------------------------------------------------------
# 02 Evaluator Graph
# -----------------------------------------------------------------------
eg = notebook(
    [
        md(
            "# The FieldML Evaluator Graph\n\n"
            "A Document is more than a mesh — it's a graph of named *evaluators*. "
            "This notebook walks through the evaluator subtypes and how they compose."
        ),
        code("from pyfieldml import datasets\n\ndoc = datasets.load_unit_cube()"),
        md(
            "## Types vs evaluators\n\n"
            "A Document contains:\n"
            "- **Types**: BooleanType, EnsembleType, ContinuousType, MeshType.\n"
            "- **Evaluators**: the graph that produces values over those types."
        ),
        code(
            'print("Types:")\n'
            "for name in doc.region.continuous:\n"
            '    print("  continuous :", name)\n'
            "for name in doc.region.ensembles:\n"
            '    print("  ensemble   :", name)\n'
            "for name in doc.region.meshes:\n"
            '    print("  mesh       :", name)'
        ),
        md("## Evaluator subtypes\n\nCount by kind:"),
        code(
            "from collections import Counter\n"
            "\n"
            "kinds = Counter(type(ev).__name__ for ev in doc.region.evaluators.values())\n"
            "for k, n in kinds.items():\n"
            '    print(f"  {k:25s}  x{n}")'
        ),
        md(
            "## Probing a coordinate field\n\n"
            "Even a unit cube has a full evaluator graph you can probe at any "
            "parametric location:"
        ),
        code(
            "import numpy as np\n"
            "\n"
            'coords = doc.field("coordinates")\n'
            "xi = np.array([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]])\n"
            "for point in xi:\n"
            "    val = coords.evaluate(element=1, xi=point)\n"
            '    print(f"xi={point}  ->  {val}")'
        ),
        md(
            "## Jacobians\n\n"
            "For a unit-cube mesh the identity map means the Jacobian is the "
            "identity matrix everywhere:"
        ),
        code(
            "J = coords.jacobian(element=1, xi=(0.5, 0.5, 0.5))\n"
            'print("Jacobian at centroid:")\n'
            "print(J)"
        ),
    ]
)
nbf.write(eg, HERE / "02_evaluator_graph.ipynb")


# -----------------------------------------------------------------------
# 04 Muscle Fibers
# -----------------------------------------------------------------------
mf = notebook(
    [
        md(
            "# Muscle Fiber Workflow\n\n"
            "Load the synthetic `rectus_femoris` dataset, inspect its "
            "fiber-direction vector field, and export line glyphs for "
            "visualization in ParaView/OpenSim."
        ),
        code(
            "import numpy as np\n"
            "\n"
            "from pyfieldml import datasets\n"
            "\n"
            "doc = datasets.load_rectus_femoris()\n"
            'print("Region:", doc.region.name)'
        ),
        md("## Inspect the fiber field"),
        code(
            'fibers = doc.evaluators["fiber_direction"]\n'
            "fv = fibers.as_ndarray()\n"
            'print("fiber vectors shape:", fv.shape)\n'
            'print("norm check:", np.allclose(np.linalg.norm(fv, axis=1), 1.0, atol=1e-6))\n'
            'print("dominant axis (argmax of |component|):")\n'
            'print("  axis count:", np.bincount(np.argmax(np.abs(fv), axis=1), minlength=3))'
        ),
        md(
            "## Export fiber glyphs as VTK\n\n"
            "`export_fiber_lines` writes a short line segment at each node along "
            "the fiber direction. Open the result in ParaView to see the fiber "
            "arrangement."
        ),
        code(
            "import tempfile\n"
            "from pathlib import Path\n"
            "\n"
            "from pyfieldml.interop.opensim import export_fiber_lines\n"
            "\n"
            'out_dir = Path(tempfile.mkdtemp(prefix="rectus_"))\n'
            "out = export_fiber_lines(\n"
            "    doc,\n"
            '    field="fiber_direction",\n'
            '    out_path=out_dir / "fibers.vtu",\n'
            "    length_scale=0.01,\n"
            ")\n"
            'print("wrote", out, f"({out.stat().st_size} bytes)")'
        ),
        md(
            "### Licensing note\n\n"
            "The bundled `rectus_femoris` dataset is a **synthetic** spindle shape "
            "authored for demonstration (CC0). Real Physiome-project muscle meshes "
            "require separate licensing; see the `datasets.info('rectus_femoris')` "
            "metadata."
        ),
    ]
)
nbf.write(mf, HERE / "04_muscle_fibers.ipynb")


# -----------------------------------------------------------------------
# 03 Hermite Bending
# -----------------------------------------------------------------------
hb = notebook(
    [
        md(
            "# Cubic Hermite Bending Beam\n\n"
            "Linear Lagrange elements can represent straight edges only. A single "
            "cubic-Hermite line element carries both a position *and* a tangent at "
            "each node, so it can represent a smoothly bending curve with one "
            "element — something no amount of linear elements will ever do exactly.\n\n"
            "This notebook builds a 1-D Hermite line mesh, visualizes the four cubic "
            "Hermite basis functions, and compares the reconstructed curve to the "
            "linear interpolant."
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "\n"
            "import pyfieldml as fml\n"
            "from pyfieldml.builders.mesh import add_hermite_mesh\n"
            "from pyfieldml.model.region import Region"
        ),
        md(
            "## Build a bending beam\n\n"
            "Two nodes at x=(0,0) and x=(1,0) with tangents chosen to pull the curve "
            "up in the middle. Tangent magnitude controls how aggressively the curve "
            "bends between the end-points."
        ),
        code(
            'r = Region(name="bending_beam")\n'
            "nodes = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)\n"
            "derivs = np.array([[1.0, 2.0], [1.0, -2.0]], dtype=np.float64)\n"
            "elements = np.array([[1, 2]], dtype=np.int64)\n"
            "add_hermite_mesh(\n"
            "    r,\n"
            '    name="beam",\n'
            "    nodes=nodes,\n"
            "    elements=elements,\n"
            "    derivatives=derivs,\n"
            '    topology="line",\n'
            ")\n"
            "doc = fml.Document.from_region(r)\n"
            'print("evaluators:", list(doc.evaluators))'
        ),
        md(
            "## The four cubic Hermite basis functions\n\n"
            "On the reference line xi in [0, 1] the cubic Hermite basis is:\n\n"
            "- H0(xi) = 1 - 3xi^2 + 2xi^3  (value at node 1)\n"
            "- H1(xi) = xi - 2xi^2 + xi^3  (slope at node 1)\n"
            "- H2(xi) = 3xi^2 - 2xi^3      (value at node 2)\n"
            "- H3(xi) = -xi^2 + xi^3       (slope at node 2)"
        ),
        code(
            "xi = np.linspace(0.0, 1.0, 101)\n"
            "H0 = 1 - 3 * xi**2 + 2 * xi**3\n"
            "H1 = xi - 2 * xi**2 + xi**3\n"
            "H2 = 3 * xi**2 - 2 * xi**3\n"
            "H3 = -(xi**2) + xi**3\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 3.5))\n"
            'ax.plot(xi, H0, label="H0 (value @ node 1)")\n'
            'ax.plot(xi, H1, label="H1 (slope @ node 1)")\n'
            'ax.plot(xi, H2, label="H2 (value @ node 2)")\n'
            'ax.plot(xi, H3, label="H3 (slope @ node 2)")\n'
            'ax.set_xlabel("xi")\n'
            'ax.set_ylabel("H_i(xi)")\n'
            'ax.set_title("Cubic Hermite basis functions on [0, 1]")\n'
            "ax.legend(fontsize=8)\n"
            "ax.grid(alpha=0.3)\n"
            "fig.tight_layout()"
        ),
        md(
            "## Evaluate the beam geometry\n\n"
            "`Field.evaluate` assembles the tangent-aware DOFs and returns a point "
            "for each xi. The y-coordinate is non-zero in the interior despite both "
            "end-points being at y=0 — that's the derivative DOFs at work."
        ),
        code(
            'coords = doc.field("coordinates")\n'
            "xi_samples = np.linspace(0.0, 1.0, 50).reshape(-1, 1)\n"
            "elems = np.ones(50, dtype=np.int64)\n"
            "curve = coords.evaluate(element=elems, xi=xi_samples)\n"
            'print("sample points on the beam:")\n'
            "print(curve[::10])"
        ),
        md(
            "## Compare against the linear interpolant\n\n"
            "A linear Lagrange element through the same two nodes is a straight "
            "segment; the Hermite element picks up the imposed tangents and bends."
        ),
        code(
            "linear_y = np.zeros_like(xi_samples[:, 0])  # straight line y=0\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 3.5))\n"
            'ax.plot(curve[:, 0], linear_y, "--", label="linear Lagrange (straight)")\n'
            'ax.plot(curve[:, 0], curve[:, 1], "-", lw=2, label="cubic Hermite (bending)")\n'
            'ax.scatter(nodes[:, 0], nodes[:, 1], color="k", zorder=5, label="nodes")\n'
            'ax.set_xlabel("x")\n'
            'ax.set_ylabel("y")\n'
            'ax.set_title("Single element: Hermite bends, Lagrange cannot")\n'
            "ax.legend(fontsize=8)\n"
            'ax.set_aspect("equal")\n'
            "ax.grid(alpha=0.3)\n"
            "fig.tight_layout()"
        ),
        md(
            "### Takeaway\n\n"
            "Cubic Hermite is the reason CMISS-family cardiac/musculoskeletal models "
            "can reproduce smooth anatomical surfaces with a handful of elements. "
            "`add_hermite_mesh` also supports `quad` (bicubic) and `hex` (tricubic) "
            "topologies — see the builder docstring for the DOF slot conventions."
        ),
    ]
)
nbf.write(hb, HERE / "03_hermite_bending.ipynb")


# -----------------------------------------------------------------------
# 05 meshio Round-trip
# -----------------------------------------------------------------------
mr = notebook(
    [
        md(
            "# meshio Round-trip: FieldML -> VTU -> FieldML\n\n"
            "`pyfieldml` includes a first-class two-way bridge to `meshio`, which "
            "unlocks every I/O format meshio supports (VTU, XDMF, Gmsh, Abaqus, "
            "Ansys, Nastran, ...). This notebook round-trips a dataset through "
            "`.vtu` and verifies that node coordinates survive unchanged."
        ),
        code(
            "import tempfile\n"
            "from pathlib import Path\n"
            "\n"
            "import meshio\n"
            "import numpy as np\n"
            "\n"
            "import pyfieldml as fml\n"
            "from pyfieldml import datasets"
        ),
        md("## Load a bundled dataset"),
        code(
            "doc = datasets.load_unit_cube()\n"
            'coords_before = doc.evaluators["coordinates"].as_ndarray()\n'
            'print("nodes:", coords_before.shape)'
        ),
        md("## Convert to meshio and write as VTU"),
        code(
            "m = doc.to_meshio()\n"
            'print("meshio cells:", m.cells[0].type, "x", len(m.cells[0].data))\n'
            "\n"
            'out_dir = Path(tempfile.mkdtemp(prefix="fieldml_rt_"))\n'
            'vtu_path = out_dir / "unit_cube.vtu"\n'
            "meshio.write(vtu_path, m)\n"
            'print("wrote", vtu_path, f"({vtu_path.stat().st_size} bytes)")'
        ),
        md("## Read the VTU back with meshio and re-import into pyfieldml"),
        code(
            "m2 = meshio.read(vtu_path)\n"
            'doc2 = fml.Document.from_meshio(m2, name="roundtrip")\n'
            'coords_after = doc2.evaluators["coordinates"].as_ndarray()\n'
            'print("nodes after round-trip:", coords_after.shape)'
        ),
        md("## Verify node equality"),
        code(
            "assert coords_before.shape == coords_after.shape, (\n"
            '    f"shape mismatch: {coords_before.shape} vs {coords_after.shape}"\n'
            ")\n"
            "np.testing.assert_allclose(\n"
            "    np.sort(coords_before, axis=0),\n"
            "    np.sort(coords_after, axis=0),\n"
            "    atol=1e-12,\n"
            ")\n"
            'print("OK: node coordinates survive the FieldML -> VTU -> FieldML round-trip.")'
        ),
        md(
            "### What's lost, what's kept\n\n"
            "**Kept**: node coordinates, element connectivity, per-node "
            "ParameterEvaluators (passed through `meshio.Mesh.point_data`).\n\n"
            "**Lost**: the evaluator graph (basis external references, scale fields, "
            "derivative DOFs). meshio is a mesh-I/O layer and has no representation "
            "for FieldML's evaluator hierarchy, so exporting to VTU is lossy. Keep "
            "`.fieldml` for archival; use VTU/XDMF for visualization and FEM handoff."
        ),
    ]
)
nbf.write(mr, HERE / "05_meshio_roundtrip.ipynb")


# -----------------------------------------------------------------------
# 06 scikit-fem Poisson
# -----------------------------------------------------------------------
sp = notebook(
    [
        md(
            "# Solving Poisson with scikit-fem on a FieldML Mesh\n\n"
            "This notebook takes the bundled `unit_cube` FieldML document, converts "
            "it to a `skfem.Mesh` + `skfem.Basis` pair, and solves the Poisson "
            "problem\n\n"
            "$$-\\nabla^2 u = 1 \\text{ in } \\Omega, \\qquad "
            "u = 0 \\text{ on } \\partial\\Omega.$$\n\n"
            "It demonstrates that pyfieldml slots directly into a real FEM workflow "
            "with no custom adapters."
        ),
        md("## Install scikit-fem\n\n```bash\npip install pyfieldml[scikit-fem]\n```"),
        code(
            "import numpy as np\n"
            "\n"
            "from pyfieldml import datasets\n"
            "from pyfieldml.interop.scikit_fem import to_scikit_fem"
        ),
        md("## Load a FieldML document and hand it to scikit-fem"),
        code(
            "doc = datasets.load_unit_cube()\n"
            "mesh, basis = to_scikit_fem(doc)\n"
            'print("skfem mesh:", type(mesh).__name__)\n'
            'print("skfem basis:", type(basis).__name__)\n'
            'print("dofs :", basis.N)'
        ),
        md(
            "## Assemble and solve\n\n"
            "A textbook Poisson assembly on the unit cube. Homogeneous Dirichlet BCs "
            "on every boundary face; unit source."
        ),
        code(
            "from skfem import BilinearForm, LinearForm, condense, solve\n"
            "from skfem.helpers import dot, grad\n"
            "\n"
            "\n"
            "@BilinearForm\n"
            "def laplace(u, v, _):\n"
            "    return dot(grad(u), grad(v))\n"
            "\n"
            "\n"
            "@LinearForm\n"
            "def source(v, _):\n"
            "    return 1.0 * v\n"
            "\n"
            "\n"
            "A = laplace.assemble(basis)\n"
            "b = source.assemble(basis)\n"
            "dirichlet = basis.get_dofs()\n"
            "u = solve(*condense(A, b, D=dirichlet))\n"
            'print("solved; max(u) =", float(u.max()))'
        ),
        md("## Inspect the solution at a few interior DOFs"),
        code(
            "dof_coords = basis.doflocs.T\n"
            "interior_mask = np.ones(basis.N, dtype=bool)\n"
            "interior_mask[dirichlet] = False\n"
            'print("interior DOF count:", int(interior_mask.sum()))\n'
            'print("interior solution samples:")\n'
            "for i in np.flatnonzero(interior_mask)[:5]:\n"
            '    print(f"  x={dof_coords[i]}  u={u[i]:.4f}")'
        ),
        md(
            "### Takeaway\n\n"
            "Any FieldML Lagrange mesh is now one function call away from the full "
            "scikit-fem stack — assembly, quadrature, boundary traces, adaptive "
            "refinement. See `pyfieldml.interop.scikit_fem` for the supported "
            "topology + order table."
        ),
    ]
)
nbf.write(sp, HERE / "06_scikit_fem_poisson.ipynb")


# -----------------------------------------------------------------------
# 07 Real Anatomy (model-zoo tour)
# -----------------------------------------------------------------------
ra = notebook(
    [
        md(
            "# Tour of the Model Zoo\n\n"
            "`pyfieldml.datasets` ships real anatomical meshes alongside the "
            "synthetic demo assets. This notebook walks through the "
            "`femur_bodyparts3d` and `bunny_stanford` bundles, renders them with "
            "PyVista, and prints the provenance metadata each dataset carries."
        ),
        code(
            "import tempfile\n"
            "from pathlib import Path\n"
            "\n"
            "import pyvista as pv\n"
            "\n"
            "from pyfieldml import datasets\n"
            "from pyfieldml.interop.pyvista import to_pyvista\n"
            "\n"
            "pv.set_jupyter_backend(None)\n"
            "pv.OFF_SCREEN = True\n"
            'OUT_DIR = Path(tempfile.mkdtemp(prefix="pyfieldml_tour_"))'
        ),
        md(
            "## BodyParts3D femur (CC-BY-SA 2.1 JP)\n\n"
            "Real anatomical triangle surface mesh from the DBCLS BodyParts3D "
            "project. Every copy carries the upstream license, citation, and "
            "origin URL — no hidden provenance."
        ),
        code(
            'info = datasets.info("femur_bodyparts3d")\n'
            "for k, v in info.items():\n"
            '    print(f"  {k:10s}: {v}")'
        ),
        code(
            "femur = datasets.load_femur_bodyparts3d()\n"
            "grid = to_pyvista(femur)\n"
            'print("n_points:", grid.n_points, "n_cells:", grid.n_cells)\n'
            "plotter = pv.Plotter(off_screen=True, window_size=(640, 480))\n"
            'plotter.add_mesh(grid, color="ivory", show_edges=False)\n'
            "plotter.view_isometric()\n"
            'shot = str(OUT_DIR / "femur_bodyparts3d.png")\n'
            'plotter.show(screenshot=shot, jupyter_backend="static")'
        ),
        md(
            "## Stanford Bunny (public domain)\n\n"
            "The canonical benchmark surface mesh from the Stanford 3D Scanning "
            "Repository, rewrapped as a FieldML document so it plugs into the same "
            "evaluator-graph machinery as every other dataset."
        ),
        code(
            'info = datasets.info("bunny_stanford")\n'
            "for k, v in info.items():\n"
            '    print(f"  {k:10s}: {v}")'
        ),
        code(
            "bunny = datasets.load_bunny_stanford()\n"
            "grid = to_pyvista(bunny)\n"
            'print("n_points:", grid.n_points, "n_cells:", grid.n_cells)\n'
            "plotter = pv.Plotter(off_screen=True, window_size=(640, 480))\n"
            'plotter.add_mesh(grid, color="tan", show_edges=False)\n'
            "plotter.view_isometric()\n"
            'shot = str(OUT_DIR / "bunny_stanford.png")\n'
            'plotter.show(screenshot=shot, jupyter_backend="static")'
        ),
        md(
            "### Why ship real anatomy?\n\n"
            "Synthetic primitives are great for unit tests but terrible for showing "
            "off spatial resolution, curvature handling, or visual debugging. The "
            "bundled real meshes let tutorials, benchmarks, and bug repros work "
            "offline with zero configuration while staying license-clean."
        ),
    ]
)
nbf.write(ra, HERE / "07_real_anatomy.ipynb")


# -----------------------------------------------------------------------
# 08 Self-conformance
# -----------------------------------------------------------------------
cf = notebook(
    [
        md(
            "# Dataset Self-Conformance Demo\n\n"
            "Every bundled dataset should satisfy the same basic invariants:\n\n"
            "1. Connectivity indices point to nodes that actually exist.\n"
            "2. The declared basis is one `pyfieldml` recognises.\n"
            "3. Evaluating the coordinate field at the reference-element centroid "
            "returns a finite point.\n\n"
            "This notebook runs those checks over every dataset as a "
            "fail-fast sanity screen."
        ),
        code(
            "import numpy as np\n"
            "\n"
            "from pyfieldml import datasets\n"
            "from pyfieldml.interop.meshio import _basis_topology_order, _find_basis_name"
        ),
        md(
            "## Reference-element centroids\n\n"
            "Each supported topology has its own centroid in xi-space:"
        ),
        code(
            "CENTROIDS = {\n"
            '    "line": (0.5,),\n'
            '    "triangle": (1.0 / 3.0, 1.0 / 3.0),\n'
            '    "quad": (0.5, 0.5),\n'
            '    "tet": (0.25, 0.25, 0.25),\n'
            '    "hex": (0.5, 0.5, 0.5),\n'
            '    "wedge": (1.0 / 3.0, 1.0 / 3.0, 0.5),\n'
            "}\n"
            "\n"
            "RECOGNIZED_TOPOLOGIES = set(CENTROIDS)"
        ),
        md("## Run the invariants over every bundled dataset"),
        code(
            "def check(name):\n"
            "    doc = datasets.load(name)\n"
            '    coords = doc.evaluators["coordinates"].as_ndarray()\n'
            '    conn = doc.evaluators["coordinates.connectivity"].as_ndarray().astype(np.int64)\n'
            "    n_nodes = coords.shape[0]\n"
            "\n"
            "    # (1) connectivity stays within node count (1-indexed)\n"
            '    assert conn.min() >= 1, f"{name}: connectivity has index < 1"\n'
            "    assert conn.max() <= n_nodes, (\n"
            '        f"{name}: connectivity max {conn.max()} exceeds node count {n_nodes}"\n'
            "    )\n"
            "\n"
            "    # (2) basis is recognized\n"
            "    basis = _find_basis_name(doc.region)\n"
            "    topology, order = _basis_topology_order(basis)\n"
            '    assert topology in RECOGNIZED_TOPOLOGIES, f"{name}: bad topology {topology!r}"\n'
            "\n"
            "    # (3) evaluating at the element centroid returns finite numbers\n"
            '    field = doc.field("coordinates")\n'
            "    xi = CENTROIDS[topology]\n"
            "    centroid = field.evaluate(element=1, xi=xi)\n"
            '    assert np.all(np.isfinite(centroid)), f"{name}: non-finite centroid"\n'
            "    return {\n"
            '        "dataset": name,\n'
            '        "n_nodes": n_nodes,\n'
            '        "n_elems": conn.shape[0],\n'
            '        "topology": topology,\n'
            '        "order": order,\n'
            '        "centroid": centroid,\n'
            "    }\n"
            "\n"
            "\n"
            "rows = [check(n) for n in datasets.list()]"
        ),
        md("## Summary table"),
        code(
            'fmt = "{:22s} {:>7s} {:>7s} {:10s} {:>5s}  {}"\n'
            'row_fmt = "{:22s} {:7d} {:7d} {:10s} {:5d}  {}"\n'
            'header = fmt.format("dataset", "nodes", "elems", "topology", "order", "centroid")\n'
            "print(header)\n"
            'print("-" * len(header))\n'
            "for row in rows:\n"
            '    c = np.array2string(row["centroid"], precision=4, suppress_small=True)\n'
            "    print(\n"
            "        row_fmt.format(\n"
            '            row["dataset"],\n'
            '            row["n_nodes"],\n'
            '            row["n_elems"],\n'
            '            row["topology"],\n'
            '            row["order"],\n'
            "            c,\n"
            "        )\n"
            "    )\n"
            "print()\n"
            'print(f"all {len(rows)} bundled datasets pass self-conformance.")'
        ),
        md(
            "### Why this matters\n\n"
            "Bundled assets are the foundation for every downstream tutorial, "
            "benchmark, and user demo. This notebook is a lightweight smoke test "
            "you can run after any dataset edit to confirm nothing has silently "
            "corrupted. For a strict byte-level regression canary, see "
            "`tests/benchmarks/test_reproducibility.py`."
        ),
    ]
)
nbf.write(cf, HERE / "08_conformance.ipynb")


print("wrote 8 notebooks")
