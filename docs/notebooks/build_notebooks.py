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
            '    out_path=out_dir / "fibers.vtp",\n'
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


print("wrote 3 notebooks")
