"""Author the Phase-5 tutorial notebooks.

Re-run to regenerate:
    uv run python docs/notebooks/build_notebooks.py

Cell sources are kept ruff-clean (double quotes, no unused imports, import
blocks separated by a blank line) so regeneration is idempotent under the
pre-commit hook set. After writing, the script invokes ``ruff format`` on
every notebook so that the generated output matches the pre-commit hook's
formatting exactly.
"""

from __future__ import annotations

import subprocess
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
            "Five-minute demo: install, load a dataset, inspect, **visualize**, "
            "evaluate, and export."
        ),
        md(
            "## Install\n\n"
            "This notebook assumes `pyfieldml` is already installed:\n\n"
            "```bash\n"
            "pip install pyfieldml[meshio,viz]\n"
            "```"
        ),
        code(
            "from pyfieldml import datasets\n"
            "\n"
            "doc = datasets.load_unit_cube()\n"
            'print("Version :", doc.source_version)\n'
            'print("Region  :", doc.region.name)'
        ),
        md(
            "## What you just loaded\n\n"
            "Pyvista renders the FieldML document directly through the "
            "`pyfieldml.interop.pyvista.to_pyvista` helper."
        ),
        code(
            "import pyvista as pv\n"
            "\n"
            "from pyfieldml.interop.pyvista import to_pyvista\n"
            "\n"
            "pv.OFF_SCREEN = True\n"
            'pv.set_jupyter_backend("static")\n'
            "\n"
            "grid = to_pyvista(doc)\n"
            "p = pv.Plotter(off_screen=True, window_size=(640, 480), shape=(1, 2))\n"
            "p.subplot(0, 0)\n"
            'p.add_mesh(grid, color="skyblue", show_edges=True, edge_color="black")\n'
            'p.add_text("unit_cube (solid)", font_size=9)\n'
            "p.view_isometric()\n"
            "p.show_axes()\n"
            "p.subplot(0, 1)\n"
            'p.add_mesh(grid, style="wireframe", color="steelblue", line_width=3)\n'
            'p.add_text("wireframe", font_size=9)\n'
            "p.view_isometric()\n"
            "p.show_axes()\n"
            'p.show(screenshot="/tmp/quickstart_unit_cube.png")'
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
        md(
            "## Plot the node cloud in matplotlib\n\n"
            "A pure-matplotlib 3D scatter is a minimal way to sanity-check a mesh "
            "without any OpenGL stack at all."
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "fig = plt.figure(figsize=(5, 4))\n"
            'ax = fig.add_subplot(111, projection="3d")\n'
            "ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=80, depthshade=True)\n"
            "for i, p in enumerate(xyz, start=1):\n"
            "    ax.text(p[0], p[1], p[2], str(i), fontsize=8)\n"
            'ax.set_xlabel("x")\n'
            'ax.set_ylabel("y")\n'
            'ax.set_zlabel("z")\n'
            'ax.set_title(f"unit_cube node cloud (n={xyz.shape[0]})")\n'
            "fig.tight_layout()"
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
            "- See `04_muscle_fibers.ipynb` for a fiber-field workflow.\n"
            "- See `07_real_anatomy.ipynb` for the 10-dataset model-zoo gallery."
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
            "A Document is more than a mesh - it's a graph of named *evaluators*. "
            "This notebook walks through the evaluator subtypes, **draws the graph**, "
            "and shows how they compose."
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
        md("## Evaluator subtypes - counts by kind"),
        code(
            "from collections import Counter\n"
            "\n"
            "kinds = Counter(type(ev).__name__ for ev in doc.region.evaluators.values())\n"
            "for k, n in kinds.items():\n"
            '    print(f"  {k:25s}  x{n}")'
        ),
        md(
            "## Bar chart of evaluator kinds across every bundled dataset\n\n"
            "This is a quick way to see *what kinds of evaluators* the model zoo "
            "actually uses - and where the complexity lives."
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "\n"
            "rows = []\n"
            "for name in datasets.list():\n"
            "    d = datasets.load(name)\n"
            "    c = Counter(type(ev).__name__ for ev in d.region.evaluators.values())\n"
            "    rows.append((name, c))\n"
            "\n"
            "kinds_all = sorted({k for _, c in rows for k in c})\n"
            "M = np.array([[r[1].get(k, 0) for k in kinds_all] for r in rows])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "bottom = np.zeros(len(rows))\n"
            "for j, k in enumerate(kinds_all):\n"
            "    ax.bar([r[0] for r in rows], M[:, j], bottom=bottom, label=k)\n"
            "    bottom += M[:, j]\n"
            'ax.set_ylabel("evaluator count")\n'
            'ax.set_title("Evaluator composition across the bundled model zoo")\n'
            'plt.setp(ax.get_xticklabels(), rotation=30, ha="right")\n'
            "ax.legend(fontsize=8)\n"
            "fig.tight_layout()"
        ),
        md(
            "## Draw the evaluator DAG\n\n"
            "Every ParameterEvaluator in a FieldML document is wired to a data "
            "array and (for fields that live on a mesh) to a connectivity "
            "evaluator and a basis ExternalEvaluator. We expose those edges by "
            "inspecting the evaluator-name prefixes (`coordinates` depends on "
            "`coordinates.connectivity`) and by reading the region's basis list."
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "import networkx as nx\n"
            "\n"
            "\n"
            "def build_dag(doc):\n"
            "    g = nx.DiGraph()\n"
            "    for name, ev in doc.region.evaluators.items():\n"
            "        g.add_node(name, kind=type(ev).__name__)\n"
            "    # ParameterEvaluators named 'foo' depend on 'foo.connectivity'\n"
            "    # and on any basis ExternalEvaluator in the same region.\n"
            "    names = set(doc.region.evaluators)\n"
            "    externals = [\n"
            '        n for n, ev in doc.region.evaluators.items() if type(ev).__name__ == "ExternalEvaluator"\n'  # noqa: E501
            "    ]\n"
            "    for name, ev in doc.region.evaluators.items():\n"
            '        if type(ev).__name__ != "ParameterEvaluator":\n'
            "            continue\n"
            '        conn = f"{name}.connectivity"\n'
            "        if conn in names:\n"
            "            g.add_edge(conn, name)\n"
            "            for ext in externals:\n"
            "                g.add_edge(ext, name)\n"
            "    return g\n"
            "\n"
            "\n"
            "doc_femur = datasets.load_femur()\n"
            "graph = build_dag(doc_femur)\n"
            'print("nodes:", graph.number_of_nodes(), "edges:", graph.number_of_edges())'
        ),
        code(
            "kind_colors = {\n"
            '    "ParameterEvaluator": "#4C78A8",\n'
            '    "ExternalEvaluator": "#F58518",\n'
            '    "ReferenceEvaluator": "#72B7B2",\n'
            '    "AggregateEvaluator": "#E45756",\n'
            '    "ArgumentEvaluator": "#54A24B",\n'
            "}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 5))\n"
            "try:\n"
            '    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")\n'
            "except (ImportError, Exception):\n"
            "    pos = nx.spring_layout(graph, seed=0, k=1.2)\n"
            "\n"
            'node_colors = [kind_colors.get(graph.nodes[n]["kind"], "lightgray") for n in graph.nodes]\n'  # noqa: E501
            "nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1400, ax=ax)\n"
            'nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="->", ax=ax)\n'
            "nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)\n"
            "\n"
            "legend_handles = [\n"
            "    plt.Line2D(\n"
            "        [0],\n"
            "        [0],\n"
            '        marker="o",\n'
            '        linestyle="",\n'
            "        markerfacecolor=c,\n"
            '        markeredgecolor="black",\n'
            "        markersize=10,\n"
            "        label=k,\n"
            "    )\n"
            "    for k, c in kind_colors.items()\n"
            "]\n"
            'ax.legend(handles=legend_handles, fontsize=8, loc="upper right")\n'
            'ax.set_title("Evaluator DAG - load_femur()")\n'
            'ax.axis("off")\n'
            "fig.tight_layout()"
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
            "fiber-direction vector field, visualize it in three complementary "
            "ways, and export line glyphs for ParaView/OpenSim."
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
            "## Three-panel pyvista view: wireframe, solid+glyphs, streamlines\n\n"
            "Left: bare topology. Middle: surface + fiber glyphs colored by "
            "magnitude. Right: streamlines seeded inside the muscle belly to "
            "trace the fiber field through the volume."
        ),
        code(
            "import pyvista as pv\n"
            "\n"
            "from pyfieldml.interop.pyvista import to_pyvista\n"
            "\n"
            "pv.OFF_SCREEN = True\n"
            'pv.set_jupyter_backend("static")\n'
            "\n"
            "grid = to_pyvista(doc)\n"
            'coords = doc.evaluators["coordinates"].as_ndarray()\n'
            'grid.point_data["fiber"] = fv\n'
            'grid.point_data["|fiber|"] = np.linalg.norm(fv, axis=1)\n'
            "\n"
            "# Short line glyphs at every node\n"
            "centers = coords\n"
            "bounds_arr = np.array(grid.bounds)\n"
            "bounds_diag = float(np.linalg.norm(bounds_arr[1::2] - bounds_arr[0::2]))\n"
            "scale = 0.03 * bounds_diag\n"
            "glyph_grid = pv.PolyData(centers)\n"
            'glyph_grid["fiber"] = fv\n'
            "glyphs = glyph_grid.glyph(\n"
            '    orient="fiber", scale=False, factor=scale, geom=pv.Line()\n'
            ")\n"
            "\n"
            "p = pv.Plotter(off_screen=True, window_size=(1100, 450), shape=(1, 3))\n"
            "p.subplot(0, 0)\n"
            'p.add_mesh(grid, style="wireframe", color="steelblue", line_width=1)\n'
            'p.add_text("wireframe", font_size=10)\n'
            "p.view_isometric()\n"
            "p.subplot(0, 1)\n"
            'p.add_mesh(grid, color="lightsalmon", opacity=0.35, show_edges=False)\n'
            'p.add_mesh(glyphs, color="crimson")\n'
            'p.add_text("fiber glyphs", font_size=10)\n'
            "p.view_isometric()\n"
            "p.subplot(0, 2)\n"
            "try:\n"
            "    streams = grid.streamlines_from_source(\n"
            "        pv.PolyData(centers[:: max(1, len(centers) // 30)]),\n"
            '        vectors="fiber",\n'
            "        max_time=5.0,\n"
            "        initial_step_length=0.05,\n"
            "    )\n"
            '    p.add_mesh(grid, color="lightsalmon", opacity=0.2)\n'
            '    p.add_mesh(streams.tube(radius=scale * 0.05), color="crimson")\n'
            '    p.add_text("streamlines", font_size=10)\n'
            "except Exception as exc:\n"
            '    p.add_mesh(grid, color="lightsalmon", opacity=0.35)\n'
            '    p.add_mesh(glyphs, color="crimson")\n'
            '    p.add_text(f"streamlines failed: {type(exc).__name__}", font_size=9)\n'
            "p.view_isometric()\n"
            'p.show(screenshot="/tmp/muscle_triple.png")'
        ),
        md(
            "## Alignment histogram\n\n"
            "The rectus femoris runs proximo-distally, so most fibers should "
            "have a strong z-component. Plotting the distribution confirms the "
            "synthetic field is anatomically plausible."
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)\n"
            'for ax, comp, label in zip(axes, fv.T, ["x", "y", "z"], strict=True):\n'
            '    ax.hist(comp, bins=24, color="steelblue", edgecolor="black")\n'
            '    ax.set_xlabel(f"fiber_{label}")\n'
            "    ax.set_xlim(-1.05, 1.05)\n"
            "    ax.grid(alpha=0.3)\n"
            'axes[0].set_ylabel("node count")\n'
            'fig.suptitle("rectus_femoris fiber-vector component distributions")\n'
            "fig.tight_layout()"
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
            "element - something no amount of linear elements will ever do exactly.\n\n"
            "This notebook builds a 1-D Hermite line mesh, visualizes the four cubic "
            "Hermite basis functions, compares the reconstructed curve to the "
            "linear interpolant, and simulates a tip-loaded cantilever by varying "
            "the derivative DOFs."
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
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))\n"
            'ax1.plot(xi, H0, label="H0 (value @ node 1)")\n'
            'ax1.plot(xi, H1, label="H1 (slope @ node 1)")\n'
            'ax1.plot(xi, H2, label="H2 (value @ node 2)")\n'
            'ax1.plot(xi, H3, label="H3 (slope @ node 2)")\n'
            'ax1.set_xlabel("xi")\n'
            'ax1.set_ylabel("H_i(xi)")\n'
            'ax1.set_title("Cubic Hermite basis functions on [0, 1]")\n'
            "ax1.legend(fontsize=8)\n"
            "ax1.grid(alpha=0.3)\n"
            "\n"
            "# Partition-of-unity check for the value basis:\n"
            'ax2.plot(xi, H0 + H2, label="H0 + H2")\n'
            'ax2.plot(xi, H1 + H3, label="H1 + H3 (slope)")\n'
            'ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)\n'
            'ax2.set_xlabel("xi")\n'
            'ax2.set_title("Partition-of-unity: value basis sums to 1")\n'
            "ax2.legend(fontsize=8)\n"
            "ax2.grid(alpha=0.3)\n"
            "fig.tight_layout()"
        ),
        md(
            "## Evaluate the beam geometry\n\n"
            "`Field.evaluate` assembles the tangent-aware DOFs and returns a point "
            "for each xi. The y-coordinate is non-zero in the interior despite both "
            "end-points being at y=0 - that's the derivative DOFs at work."
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
            "## Tip-loaded cantilever family\n\n"
            "Sweep the derivative DOFs at node 2 to mimic a cantilever bending "
            "under increasing tip load. With a single Hermite element we pick up a "
            "smooth curve at every load level - no refinement required."
        ),
        code(
            "fig, (ax_u, ax_d) = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)\n"
            "\n"
            'ax_u.plot([0.0, 1.0], [0.0, 0.0], "-", lw=2,\n'
            '          color="black", label="undeformed (straight beam)")\n'
            'ax_u.scatter(nodes[:, 0], nodes[:, 1], color="black", zorder=5)\n'
            'ax_u.set_title("Undeformed: linear Lagrange")\n'
            'ax_u.set_xlabel("x")\n'
            'ax_u.set_ylabel("y")\n'
            "ax_u.set_xlim(-0.05, 1.1)\n"
            "ax_u.grid(alpha=0.3)\n"
            "ax_u.legend(fontsize=8)\n"
            "\n"
            "loads = [0.5, 1.0, 1.5, 2.0, 2.5]\n"
            "for load in loads:\n"
            '    rr = Region(name="cantilever")\n'
            "    d2 = np.array([[1.0, 0.0], [1.0, -load]], dtype=np.float64)\n"
            "    add_hermite_mesh(\n"
            "        rr,\n"
            '        name="beam",\n'
            "        nodes=nodes,\n"
            "        elements=elements,\n"
            "        derivatives=d2,\n"
            '        topology="line",\n'
            "    )\n"
            "    dd = fml.Document.from_region(rr)\n"
            '    c = dd.field("coordinates").evaluate(element=elems, xi=xi_samples)\n'
            '    ax_d.plot(c[:, 0], c[:, 1], label=f"tip slope = {-load:+.1f}")\n'
            "\n"
            'ax_d.scatter(nodes[:, 0], nodes[:, 1], color="black", zorder=5)\n'
            'ax_d.set_title("Deformed family: cubic Hermite")\n'
            'ax_d.set_xlabel("x")\n'
            "ax_d.set_xlim(-0.05, 1.1)\n"
            "ax_d.grid(alpha=0.3)\n"
            "ax_d.legend(fontsize=8)\n"
            "fig.tight_layout()"
        ),
        md(
            "### Takeaway\n\n"
            "Cubic Hermite is the reason CMISS-family cardiac/musculoskeletal models "
            "can reproduce smooth anatomical surfaces with a handful of elements. "
            "`add_hermite_mesh` also supports `quad` (bicubic) and `hex` (tricubic) "
            "topologies - see the builder docstring for the DOF slot conventions."
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
            "`.vtu`, **renders before/after visually**, and verifies that node "
            "coordinates survive unchanged."
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
            "doc = datasets.load_femur_bodyparts3d()\n"
            'coords_before = doc.evaluators["coordinates"].as_ndarray()\n'
            'print("nodes:", coords_before.shape)'
        ),
        md("## Convert to meshio and write as VTU"),
        code(
            "m = doc.to_meshio()\n"
            'print("meshio cells:", m.cells[0].type, "x", len(m.cells[0].data))\n'
            "\n"
            'out_dir = Path(tempfile.mkdtemp(prefix="fieldml_rt_"))\n'
            'vtu_path = out_dir / "femur.vtu"\n'
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
        md(
            "## Side-by-side rendering\n\n"
            "Before / after should be visually indistinguishable - which is the "
            "whole point of lossless coordinate round-trip."
        ),
        code(
            "import pyvista as pv\n"
            "\n"
            "from pyfieldml.interop.pyvista import to_pyvista\n"
            "\n"
            "pv.OFF_SCREEN = True\n"
            'pv.set_jupyter_backend("static")\n'
            "\n"
            "g1 = to_pyvista(doc)\n"
            "g2 = to_pyvista(doc2)\n"
            "p = pv.Plotter(off_screen=True, window_size=(1000, 450), shape=(1, 2))\n"
            "p.subplot(0, 0)\n"
            'p.add_mesh(g1, color="ivory", show_edges=False)\n'
            'p.add_text("source FieldML", font_size=10)\n'
            "p.view_isometric()\n"
            "p.subplot(0, 1)\n"
            'p.add_mesh(g2, color="peachpuff", show_edges=False)\n'
            'p.add_text("round-tripped via VTU", font_size=10)\n'
            "p.view_isometric()\n"
            'p.show(screenshot="/tmp/roundtrip.png")'
        ),
        md(
            "## Quantitative coordinate diff\n\n"
            "Sort-and-compare because meshio/VTU do not preserve the original "
            "node ordering in general."
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "a = np.sort(coords_before, axis=0)\n"
            "b = np.sort(coords_after, axis=0)\n"
            "delta = np.abs(a - b)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            'ax.semilogy(delta[:, 0] + 1e-20, label="|dx|")\n'
            'ax.semilogy(delta[:, 1] + 1e-20, label="|dy|")\n'
            'ax.semilogy(delta[:, 2] + 1e-20, label="|dz|")\n'
            'ax.set_xlabel("sorted node index")\n'
            'ax.set_ylabel("|coord_after - coord_before|")\n'
            'ax.set_title("Round-trip coordinate residuals (should be at float noise)")\n'
            "ax.legend(fontsize=8)\n"
            "ax.grid(alpha=0.3)\n"
            "fig.tight_layout()"
        ),
        md("## Verify node equality"),
        code(
            "assert coords_before.shape == coords_after.shape, (\n"
            '    f"shape mismatch: {coords_before.shape} vs {coords_after.shape}"\n'
            ")\n"
            "np.testing.assert_allclose(a, b, atol=1e-12)\n"
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
            "it to a `skfem.Mesh` + `skfem.Basis` pair, solves the Poisson "
            "problem\n\n"
            "$$-\\nabla^2 u = 1 \\text{ in } \\Omega, \\qquad "
            "u = 0 \\text{ on } \\partial\\Omega,$$\n\n"
            "and visualizes the field, its gradient magnitude, and contour slices "
            "in a four-panel pyvista figure. It demonstrates that pyfieldml slots "
            "directly into a real FEM workflow with no custom adapters."
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
            "## Assemble and solve on a refined mesh\n\n"
            "A textbook Poisson assembly on the unit cube. Homogeneous Dirichlet BCs "
            "on every boundary face; unit source. We refine the FieldML unit cube a "
            "few times so there is a visible interior field to plot."
        ),
        code(
            "from skfem import Basis, BilinearForm, LinearForm, condense, solve\n"
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
            "mesh_r = mesh.refined(3)\n"
            "basis_r = Basis(mesh_r, basis.elem)\n"
            "A = laplace.assemble(basis_r)\n"
            "b = source.assemble(basis_r)\n"
            "dirichlet = basis_r.get_dofs()\n"
            "u = solve(*condense(A, b, D=dirichlet))\n"
            'print("dofs after refinement:", basis_r.N)\n'
            'print("solved; max(u) =", float(u.max()))'
        ),
        md(
            "## Convergence study\n\n"
            "Sweep refinement levels and track the peak solution value. For the "
            "unit-cube Poisson problem this converges to the analytic solution "
            "max ~ 0.0559 (Dirichlet-Laplacian eigen-expansion)."
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "levels = range(1, 5)\n"
            "peaks = []\n"
            "dofs = []\n"
            "for lv in levels:\n"
            "    m_lv = mesh.refined(lv)\n"
            "    b_lv = Basis(m_lv, basis.elem)\n"
            "    A_lv = laplace.assemble(b_lv)\n"
            "    f_lv = source.assemble(b_lv)\n"
            "    u_lv = solve(*condense(A_lv, f_lv, D=b_lv.get_dofs()))\n"
            "    peaks.append(float(u_lv.max()))\n"
            "    dofs.append(b_lv.N)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 3.5))\n"
            'ax.semilogx(dofs, peaks, marker="o")\n'
            'ax.axhline(0.0559, color="gray", linestyle=":", label="analytic ~ 0.0559")\n'
            'ax.set_xlabel("degrees of freedom")\n'
            'ax.set_ylabel("max u")\n'
            'ax.set_title("Poisson on unit cube: peak solution vs refinement")\n'
            "ax.legend(fontsize=8)\n"
            "ax.grid(alpha=0.3)\n"
            "fig.tight_layout()"
        ),
        md(
            "## Four-panel solution visualization\n\n"
            "1. wireframe mesh; "
            "2. scalar u colored on the volume; "
            "3. gradient magnitude on a clipped surface; "
            "4. iso-contours."
        ),
        code(
            "import pyvista as pv\n"
            "\n"
            "pv.OFF_SCREEN = True\n"
            'pv.set_jupyter_backend("static")\n'
            "\n"
            "pts = mesh_r.p.T\n"
            "cells_t = mesh_r.t.T\n"
            "n_v = cells_t.shape[1]\n"
            "VTK_CELL = {4: 10, 8: 12, 3: 5, 6: 13}  # tet=10, hex=12, triangle=5, wedge=13\n"
            "cells = np.hstack(\n"
            "    [np.full((cells_t.shape[0], 1), n_v, dtype=np.int64), cells_t]\n"
            ").astype(np.int64)\n"
            "grid = pv.UnstructuredGrid(\n"
            "    cells,\n"
            "    np.full(cells_t.shape[0], VTK_CELL[n_v]),\n"
            "    pts,\n"
            ")\n"
            'grid.point_data["u"] = u\n'
            "\n"
            "# Approximate gradient magnitude at nodes via pyvista filter\n"
            'grad_grid = grid.compute_derivative(scalars="u", gradient=True)\n'
            'grad_grid["|grad u|"] = np.linalg.norm(grad_grid["gradient"], axis=1)\n'
            "\n"
            "p = pv.Plotter(off_screen=True, window_size=(1000, 850), shape=(2, 2))\n"
            "p.subplot(0, 0)\n"
            'p.add_mesh(grid, style="wireframe", color="steelblue", line_width=1)\n'
            'p.add_text("refined mesh", font_size=10)\n'
            "p.view_isometric()\n"
            "p.subplot(0, 1)\n"
            'p.add_mesh(grid, scalars="u", cmap="viridis", show_edges=False,\n'
            '           scalar_bar_args={"title": "u"})\n'
            'p.add_text("scalar field u", font_size=10)\n'
            "p.view_isometric()\n"
            "p.subplot(1, 0)\n"
            "clipped = grad_grid.clip(\n"
            '    normal="x", origin=grad_grid.center, invert=False\n'
            ")\n"
            'p.add_mesh(clipped, scalars="|grad u|", cmap="magma", show_edges=False,\n'
            '           scalar_bar_args={"title": "|grad u|"})\n'
            'p.add_text("|grad u| (clipped)", font_size=10)\n'
            "p.view_isometric()\n"
            "p.subplot(1, 1)\n"
            'contours = grid.contour(isosurfaces=6, scalars="u")\n'
            'p.add_mesh(grid, color="lightgray", opacity=0.15)\n'
            'p.add_mesh(contours, cmap="viridis", scalars="u", show_edges=False)\n'
            'p.add_text("iso-contours of u", font_size=10)\n'
            "p.view_isometric()\n"
            'p.show(screenshot="/tmp/poisson4panel.png")'
        ),
        md("## Inspect the solution at a few interior DOFs"),
        code(
            "dof_coords = basis_r.doflocs.T\n"
            "interior_mask = np.ones(basis_r.N, dtype=bool)\n"
            "interior_mask[basis_r.get_dofs()] = False\n"
            'print("interior DOF count:", int(interior_mask.sum()))\n'
            'print("interior solution samples:")\n'
            "for i in np.flatnonzero(interior_mask)[:5]:\n"
            '    print(f"  x={dof_coords[i]}  u={u[i]:.4f}")'
        ),
        md(
            "### Takeaway\n\n"
            "Any FieldML Lagrange mesh is now one function call away from the full "
            "scikit-fem stack - assembly, quadrature, boundary traces, adaptive "
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
            "`pyfieldml.datasets` ships ten FieldML documents: five synthetic/"
            "reference assets and five real BodyParts3D (BP3D) anatomical meshes. "
            "This notebook renders the whole gallery, assembles a multi-part MSK "
            "scene, and summarizes the dataset sizes.\n\n"
            "Each dataset carries its upstream license, citation, and origin URL; "
            "nothing here has hidden provenance."
        ),
        code(
            "import numpy as np\n"
            "import pyvista as pv\n"
            "\n"
            "from pyfieldml import datasets\n"
            "from pyfieldml.interop.pyvista import to_pyvista\n"
            "\n"
            "pv.OFF_SCREEN = True\n"
            'pv.set_jupyter_backend("static")'
        ),
        md(
            "## The gallery\n\n"
            "Ten datasets, ten panels. Color encodes dataset family:\n"
            "- **blue** = synthetic / reference (`unit_cube`, `femur` CSG, "
            "`rectus_femoris`, `bunny_stanford`)\n"
            "- **orange** = BodyParts3D real anatomy "
            "(`femur_bodyparts3d`, `vertebra_l3`, `scapula`, `tibia_left`, "
            "`hip_bone_left`, `skull`)"
        ),
        code(
            "GALLERY = [\n"
            '    ("unit_cube",         "synthetic",  "smoke-test hex primitive"),\n'
            '    ("femur",             "synthetic",  "CSG synthetic + BMD field"),\n'
            '    ("rectus_femoris",    "synthetic",  "fiber-direction vector field"),\n'
            '    ("bunny_stanford",    "reference",  "public-domain benchmark"),\n'
            '    ("femur_bodyparts3d", "BP3D",       "atlas-real anatomy"),\n'
            '    ("vertebra_l3",       "BP3D",       "foramen topology (spine SSM)"),\n'
            '    ("scapula",           "BP3D",       "thin curved shell"),\n'
            '    ("tibia_left",        "BP3D",       "bilateral companion to femur"),\n'
            '    ("hip_bone_left",     "BP3D",       "pelvic MSK pivot"),\n'
            '    ("skull",             "BP3D",       "74-component messy real data"),\n'
            "]\n"
            "COLORS = {\n"
            '    "synthetic": "skyblue",\n'
            '    "reference": "wheat",\n'
            '    "BP3D": "peachpuff",\n'
            "}\n"
            "\n"
            "docs = {n: datasets.load(n) for n, _, _ in GALLERY}\n"
            "grids = {n: to_pyvista(d) for n, d in docs.items()}\n"
            "for name, _, _ in GALLERY:\n"
            "    g = grids[name]\n"
            '    print(f"  {name:22s}  n_points={g.n_points:6d}  n_cells={g.n_cells:6d}")'
        ),
        code(
            "p = pv.Plotter(off_screen=True, window_size=(1500, 700), shape=(2, 5))\n"
            "for i, (name, family, note) in enumerate(GALLERY):\n"
            "    p.subplot(i // 5, i % 5)\n"
            "    g = grids[name]\n"
            "    p.add_mesh(g, color=COLORS[family], show_edges=False)\n"
            '    p.add_text(f"{name}\\n{g.n_points} nodes / {g.n_cells} cells",\n'
            '               font_size=8, color="black", position="upper_edge")\n'
            '    p.add_text(note, font_size=7, color="dimgray", position="lower_edge")\n'
            "    p.view_isometric()\n"
            'p.show(screenshot="/tmp/zoo_gallery.png")'
        ),
        md(
            "## Vertebra-L3 close-up: wireframe + edge-colored foramen\n\n"
            "The lumbar-3 vertebra is the gateway dataset for spine SSM work: it "
            "has a **foramen** (the central hole through which the spinal cord "
            "runs), which is non-trivial topology for statistical shape models. "
            "The wireframe panel reveals triangulation density; the edge-colored "
            "panel makes the foramen boundary visually obvious."
        ),
        code(
            'vert = grids["vertebra_l3"]\n'
            "p = pv.Plotter(off_screen=True, window_size=(900, 450), shape=(1, 2))\n"
            "p.subplot(0, 0)\n"
            'p.add_mesh(vert, style="wireframe", color="steelblue", line_width=1)\n'
            'p.add_text("L3 vertebra - wireframe", font_size=10)\n'
            "p.view_vector((1, 0, 0.3))\n"
            "p.subplot(0, 1)\n"
            "p.add_mesh(\n"
            "    vert,\n"
            '    color="lightcoral",\n'
            "    show_edges=True,\n"
            '    edge_color="black",\n'
            "    line_width=1,\n"
            "    opacity=0.92,\n"
            ")\n"
            'p.add_text("L3 vertebra - edge-highlighted foramen", font_size=10)\n'
            "p.view_vector((1, 0, 0.3))\n"
            'p.show(screenshot="/tmp/vertebra_closeup.png")'
        ),
        md(
            "## Clinical MSK assembly\n\n"
            "Loading four BP3D parts (left femur, left tibia, left hip bone, L3 "
            "vertebra) into the same pyvista scene demonstrates how pyfieldml "
            "handles multi-mesh assemblies. The pieces are not surgically "
            "co-registered (they come from different BP3D donors), so we roughly "
            "stack them along the z-axis using each part's own bounding box - "
            "enough to make the MSK layout visually coherent."
        ),
        code(
            'PARTS = ["vertebra_l3", "hip_bone_left", "femur_bodyparts3d", "tibia_left"]\n'
            "COLOR_MAP = {\n"
            '    "vertebra_l3": "indianred",\n'
            '    "hip_bone_left": "mediumseagreen",\n'
            '    "femur_bodyparts3d": "cornflowerblue",\n'
            '    "tibia_left": "goldenrod",\n'
            "}\n"
            "\n"
            "\n"
            "def stacked_copy(g, z_offset):\n"
            "    gc = g.copy()\n"
            "    bounds = np.array(gc.bounds).reshape(3, 2)\n"
            "    center_xy = bounds[:2].mean(axis=1)\n"
            "    z_min = bounds[2, 0]\n"
            "    gc.translate((-center_xy[0], -center_xy[1], -z_min + z_offset),\n"
            "                 inplace=True)\n"
            "    return gc\n"
            "\n"
            "\n"
            "# Rough anatomical stack: vertebra on top, hip below, femur lower, tibia bottom\n"
            "OFFSETS = {\n"
            '    "vertebra_l3": 240.0,\n'
            '    "hip_bone_left": 120.0,\n'
            '    "femur_bodyparts3d": -180.0,\n'
            '    "tibia_left": -440.0,\n'
            "}\n"
            "\n"
            "p = pv.Plotter(off_screen=True, window_size=(700, 900))\n"
            "for name in PARTS:\n"
            "    gc = stacked_copy(grids[name], OFFSETS[name])\n"
            "    p.add_mesh(gc, color=COLOR_MAP[name], show_edges=False,\n"
            "               label=name)\n"
            "p.add_legend(bcolor=(1, 1, 1), size=(0.2, 0.2))\n"
            'p.add_text("Left-side MSK assembly (roughly stacked)", font_size=10)\n'
            "p.view_vector((1, 0.2, 0.3))\n"
            "p.show_axes()\n"
            'p.show(screenshot="/tmp/msk_assembly.png")'
        ),
        md(
            "## Provenance card for every dataset\n\n"
            "License, citation, origin - printed inline so this notebook is a "
            "stand-alone reference without needing to crack open the registry."
        ),
        code(
            "for name, _, _ in GALLERY:\n"
            "    info = datasets.info(name)\n"
            '    print(f"== {name} ==")\n'
            "    for k, v in info.items():\n"
            '        print(f"  {k:10s}: {v}")\n'
            "    print()"
        ),
        md("## Dataset-size summary\n\nNode count per dataset, on a log scale."),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "names = [n for n, _, _ in GALLERY]\n"
            "nps = [grids[n].n_points for n in names]\n"
            "ncs = [grids[n].n_cells for n in names]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 3.5))\n"
            "x = np.arange(len(names))\n"
            'ax.bar(x - 0.2, nps, width=0.4, label="n_points", color="steelblue")\n'
            'ax.bar(x + 0.2, ncs, width=0.4, label="n_cells", color="darkorange")\n'
            "ax.set_xticks(x)\n"
            'ax.set_xticklabels(names, rotation=30, ha="right")\n'
            'ax.set_yscale("log")\n'
            'ax.set_ylabel("count (log)")\n'
            'ax.set_title("Model zoo: nodes and cells per dataset")\n'
            "ax.legend(fontsize=8)\n"
            "ax.grid(alpha=0.3)\n"
            "fig.tight_layout()"
        ),
        md(
            "### Why ship real anatomy?\n\n"
            "Synthetic primitives are great for unit tests but terrible for "
            "showing off spatial resolution, curvature handling, or visual "
            "debugging. The bundled real meshes let tutorials, benchmarks, and "
            "bug repros work offline with zero configuration while staying "
            "license-clean.\n\n"
            "The BP3D meshes are distributed under **CC-BY-SA 2.1 JP** (DBCLS); "
            "the Stanford Bunny is **public domain**; the synthetic assets are "
            "CC0 within the pyfieldml source tree."
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
            "This notebook runs those checks over every dataset as a fail-fast "
            "sanity screen, then **visualizes** the structural invariants across "
            "the zoo so regressions jump out at a glance."
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
            '        "coords": coords,\n'
            '        "conn": conn,\n'
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
            "## Visual invariant 1: bounding-box extents\n\n"
            "BP3D meshes are stored in millimeters; synthetic primitives live in "
            "the unit interval. Plotting the per-axis extent on a log scale makes "
            "the unit conventions obvious."
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            'names = [r["dataset"] for r in rows]\n'
            "extents = np.array(\n"
            '    [np.ptp(r["coords"], axis=0) for r in rows],\n'
            "    dtype=np.float64,\n"
            ")\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 3.8))\n"
            "x = np.arange(len(names))\n"
            "w = 0.25\n"
            'axis_colors = ["#4C78A8", "#F58518", "#54A24B"]\n'
            'for i, (ax_label, color) in enumerate(zip("xyz", axis_colors, strict=True)):\n'
            "    ax.bar(x + (i - 1) * w, extents[:, i], width=w, color=color,\n"
            '           label=f"extent_{ax_label}")\n'
            'ax.set_yscale("log")\n'
            "ax.set_xticks(x)\n"
            'ax.set_xticklabels(names, rotation=30, ha="right")\n'
            'ax.set_ylabel("axis extent (log)")\n'
            'ax.set_title("Bounding-box extents across the model zoo")\n'
            "ax.legend(fontsize=8)\n"
            "ax.grid(alpha=0.3)\n"
            "fig.tight_layout()"
        ),
        md(
            "## Visual invariant 2: connectivity-degree histograms\n\n"
            "For each dataset, count how many elements touch each node. Manifold "
            "surfaces peak around 6; pathological or heavily joined meshes (e.g., "
            "the 74-component skull) skew the distribution."
        ),
        code(
            "n = len(rows)\n"
            "cols = 4\n"
            "rows_ = (n + cols - 1) // cols\n"
            "fig, axes = plt.subplots(rows_, cols, figsize=(11, 2.4 * rows_))\n"
            "axes = axes.flatten()\n"
            "for ax, r in zip(axes, rows, strict=False):\n"
            '    flat = r["conn"].ravel()\n'
            '    counts = np.bincount(flat, minlength=r["n_nodes"] + 1)[1:]\n'
            "    ax.hist(counts, bins=range(0, int(counts.max()) + 2),\n"
            '            color="steelblue", edgecolor="black")\n'
            "    ax.set_title(f\"{r['dataset']}\\n(max deg {int(counts.max())})\", fontsize=9)\n"
            '    ax.set_xlabel("element-incidence per node")\n'
            "    ax.tick_params(labelsize=7)\n"
            "for ax in axes[len(rows):]:\n"
            "    ax.set_visible(False)\n"
            "fig.tight_layout()"
        ),
        md(
            "## Visual invariant 3: one element per dataset, colored by topology\n\n"
            "Extract the *first* element from each dataset and render it solo in a "
            "grid. Topology differences (hex vs tet vs triangle) are visually "
            "unambiguous."
        ),
        code(
            "import pyvista as pv\n"
            "\n"
            "pv.OFF_SCREEN = True\n"
            'pv.set_jupyter_backend("static")\n'
            "\n"
            "TOPO_COLORS = {\n"
            '    "hex": "#4C78A8",\n'
            '    "tet": "#F58518",\n'
            '    "triangle": "#54A24B",\n'
            '    "quad": "#E45756",\n'
            '    "wedge": "#B279A2",\n'
            '    "line": "#9D755D",\n'
            "}\n"
            "\n"
            "cols = 5\n"
            "nrows = (len(rows) + cols - 1) // cols\n"
            "p = pv.Plotter(\n"
            "    off_screen=True,\n"
            "    window_size=(1500, 320 * nrows),\n"
            "    shape=(nrows, cols),\n"
            ")\n"
            "for i, r in enumerate(rows):\n"
            "    p.subplot(i // cols, i % cols)\n"
            '    first_conn = r["conn"][0] - 1  # 1-indexed -> 0-indexed\n'
            '    first_pts = r["coords"][first_conn]\n'
            "    cloud = pv.PolyData(first_pts)\n"
            "    p.add_mesh(\n"
            "        cloud,\n"
            '        color=TOPO_COLORS.get(r["topology"], "gray"),\n'
            "        render_points_as_spheres=True,\n"
            "        point_size=18,\n"
            "    )\n"
            "    # draw edges of the first element\n"
            "    n_v = first_pts.shape[0]\n"
            "    lines = []\n"
            "    for a in range(n_v):\n"
            "        for b in range(a + 1, n_v):\n"
            "            lines += [2, a, b]\n"
            "    edges = pv.PolyData(first_pts, lines=np.array(lines, dtype=np.int64))\n"
            '    p.add_mesh(edges, color="black", line_width=2)\n'
            "    p.add_text(\n"
            "        f\"{r['dataset']}\\n{r['topology']}/order {r['order']}\",\n"
            "        font_size=8,\n"
            '        position="upper_edge",\n'
            "    )\n"
            "    p.view_isometric()\n"
            'p.show(screenshot="/tmp/conformance_first_elem.png")'
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

# Run ruff format on every notebook so regenerated output matches the
# pre-commit hook's formatting exactly (the hook would otherwise rewrite
# multi-line cells that ruff-format considers one-liner material).
_fmt = subprocess.run(
    ["ruff", "format", "--quiet", *[str(p) for p in sorted(HERE.glob("*.ipynb"))]],
    check=False,
)
if _fmt.returncode != 0:
    print(f"ruff format exited with code {_fmt.returncode}")
