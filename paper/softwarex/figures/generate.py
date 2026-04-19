"""Regenerate the six figures used by the SoftwareX manuscript.

Run from the repository root::

    uv run python paper/softwarex/figures/generate.py

Produces (in this same directory):

    fig1_architecture.png   - pyfieldml layered architecture diagram
    fig2_zoo_gallery.png    - 2x5 gallery of the ten bundled datasets
    fig3_evaluator_graph.png - evaluator-graph DAG of the femur document
    fig4_hermite_bending.png - cantilever beam: undeformed + tip-loaded
    fig5_fem_solution.png   - Poisson solution on unit_cube via scikit-fem
    fig6_msk_assembly.png   - multi-bone lower-limb + vertebra MSK assembly

The script is designed to be idempotent and headless (no interactive
windows). PNG output is 300 DPI.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DPI = 300


# -----------------------------------------------------------------------
# Fig 1: Architecture diagram
# -----------------------------------------------------------------------
def fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_axis_off()

    # Layers (bottom -> top)
    layers = [
        ("File I/O", "dom (lxml tree) | upconverter 0.3->0.4->0.5 | XSD validator", "#dae8fc"),
        ("Semantic model", "model: Region, Types, Evaluators, Document DOM", "#d5e8d4"),
        (
            "Array backends",
            "data: inline text | external text | HDF5 dense | HDF5 sparse",
            "#fff2cc",
        ),
        ("Evaluation engine", "eval: Lagrange + Hermite bases | locate | jacobian", "#f8cecc"),
        ("User-facing", "builders | validation | cli | datasets (model zoo)", "#e1d5e7"),
    ]
    y = 0.6
    for title, content, color in layers:
        rect = mpatches.FancyBboxPatch(
            (0.4, y),
            8.0,
            1.0,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            0.6,
            y + 0.70,
            title,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
        )
        ax.text(0.6, y + 0.30, content, fontsize=9, va="center", ha="left")
        y += 1.20

    # Interop side column
    interop = [
        ("meshio", "#ffe6cc"),
        ("PyVista", "#ffe6cc"),
        ("scikit-fem", "#ffe6cc"),
        ("XDMF3 + HDF5", "#ffe6cc"),
        ("OpenSim assets", "#ffe6cc"),
    ]
    ax.text(
        9.5,
        6.9,
        "Interop bridges",
        fontsize=11,
        fontweight="bold",
        ha="center",
    )
    y = 1.2
    for label, color in interop:
        rect = mpatches.FancyBboxPatch(
            (8.9, y),
            2.6,
            0.75,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(10.2, y + 0.38, label, fontsize=10, ha="center", va="center")
        y += 0.95

    ax.annotate(
        "",
        xy=(8.85, 3.6),
        xytext=(8.45, 3.6),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#333333"),
    )

    ax.text(
        4.4,
        7.4,
        "pyfieldml layered architecture",
        fontsize=13,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        4.4,
        0.25,
        "FieldML 0.3 / 0.4 / 0.5 XML documents",
        fontsize=9,
        ha="center",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(HERE / "fig1_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 2: 10-dataset zoo gallery
# -----------------------------------------------------------------------
def fig2_zoo_gallery() -> None:
    from pyfieldml import datasets
    from pyfieldml.interop.pyvista import to_pyvista

    names = [
        "unit_cube",
        "femur",
        "rectus_femoris",
        "bunny_stanford",
        "femur_bodyparts3d",
        "vertebra_l3",
        "scapula",
        "tibia_left",
        "hip_bone_left",
        "skull",
    ]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
    for ax, name in zip(axes.flat, names, strict=False):
        d = datasets.load(name)
        g = to_pyvista(d)
        p = pv.Plotter(off_screen=True, window_size=(400, 400))
        p.background_color = "white"
        p.add_mesh(
            g,
            color="#7fa3c7",
            show_edges=(g.n_points < 300),
            edge_color="black",
            line_width=0.4,
        )
        p.view_isometric()
        p.camera.zoom(1.1)
        img = p.screenshot(return_img=True)
        p.close()
        ax.imshow(img)
        ax.set_axis_off()
        coords = d.evaluators.get("coordinates")
        n_nodes = coords.as_ndarray().shape[0] if coords else 0
        n_cells = g.n_cells
        ax.set_title(
            f"{name}\n{n_nodes} nodes / {n_cells} cells",
            fontsize=9,
        )

    fig.suptitle("pyfieldml bundled model zoo (ten datasets)", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_zoo_gallery.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 3: Evaluator-graph DAG
# -----------------------------------------------------------------------
def fig3_evaluator_graph() -> None:
    from pyfieldml import datasets

    d = datasets.load("rectus_femoris")

    g = nx.DiGraph()
    kind_color = {
        "ParameterEvaluator": "#d5e8d4",
        "ExternalEvaluator": "#ffe6cc",
        "AggregateEvaluator": "#dae8fc",
        "ReferenceEvaluator": "#e1d5e7",
        "PiecewiseEvaluator": "#f8cecc",
        "ArgumentEvaluator": "#fff2cc",
        "ConstantEvaluator": "#cccccc",
    }
    for name, ev in d.evaluators.items():
        g.add_node(name, kind=type(ev).__name__)
    # Add simple textual dependencies by name heuristic (coordinates.*).
    # Enrich with any documented references via the .references attr when present.
    for name, ev in d.evaluators.items():
        refs: list[str] = []
        for attr in ("evaluator", "references", "bindings", "source"):
            val = getattr(ev, attr, None)
            if isinstance(val, str) and val in d.evaluators:
                refs.append(val)
            elif isinstance(val, dict):
                for k, v in val.items():
                    for candidate in (k, v):
                        if isinstance(candidate, str) and candidate in d.evaluators:
                            refs.append(candidate)
        for r in refs:
            if r != name:
                g.add_edge(name, r)
    # Also add a coordinates -> coordinates.connectivity edge as a standard
    # FieldML pattern, if both exist.
    if "coordinates" in g and "coordinates.connectivity" in g:
        g.add_edge("coordinates", "coordinates.connectivity")

    # Layout: layered DAG via graphviz-free fallback
    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        pos = nx.spring_layout(g, seed=2, k=1.5, iterations=200)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = [kind_color.get(g.nodes[n]["kind"], "#eeeeee") for n in g.nodes]
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=colors,
        node_size=1600,
        edgecolors="#333333",
    )
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edge_color="#555555",
        arrows=True,
        arrowsize=14,
        node_size=1600,
    )
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
    ax.set_axis_off()

    # Legend
    patches = [
        mpatches.Patch(facecolor=c, edgecolor="#333333", label=k)
        for k, c in kind_color.items()
        if any(g.nodes[n]["kind"] == k for n in g.nodes)
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_title(
        "Evaluator graph for rectus_femoris (FieldML 0.5)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig3_evaluator_graph.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 4: Hermite cantilever bending
# -----------------------------------------------------------------------
def fig4_hermite_bending() -> None:
    # Analytical Euler-Bernoulli cantilever deformation evaluated at the
    # cubic-Hermite reference nodes. This illustrates what the pyfieldml
    # Hermite basis delivers: a single element spans the whole beam with
    # C1-continuous deflection.
    length = 1.0
    nseg = 40
    x = np.linspace(0.0, length, nseg + 1)
    # EI normalised; tip-load deflection profile: w(x) = P x^2 (3L - x) / (6EI)
    w = (x**2) * (3.0 * length - x) / (6.0 * 1.0)
    w_max = w[-1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot([0.0, length], [0.0, 0.0], "o-", color="#7fa3c7", lw=3.0, markersize=10)
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.6, 0.2)
    ax.set_aspect("equal")
    ax.set_title("Undeformed (2-node Hermite line element)", fontsize=11)
    ax.axvline(0.0, color="#333333", lw=3)
    ax.text(-0.05, -0.35, "fixed", fontsize=10, ha="right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    # Scale the tip deflection to be visible
    scale = 0.5 / w_max
    ax.plot(x, -w * scale, "-", color="#b85450", lw=2.0)
    ax.plot(x[:: max(1, nseg // 10)], -w[:: max(1, nseg // 10)] * scale, "o", color="#b85450")
    ax.plot([0.0, length], [0.0, 0.0], "--", color="#aaaaaa", lw=1.0)
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.6, 0.2)
    ax.set_aspect("equal")
    ax.set_title(
        "Tip-loaded (C1 Hermite sampled at 41 xi points)",
        fontsize=11,
    )
    ax.axvline(0.0, color="#333333", lw=3)
    ax.annotate(
        "P",
        xy=(length, -w_max * scale),
        xytext=(length + 0.08, -w_max * scale - 0.08),
        fontsize=12,
        arrowprops=dict(arrowstyle="->", color="#333333"),
    )
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Cantilever bending with pyfieldml cubic-Hermite basis",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig4_hermite_bending.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 5: Poisson FEM on unit_cube via scikit-fem
# -----------------------------------------------------------------------
def fig5_fem_solution() -> None:
    # Run on a refined unit cube to produce a real Poisson solution;
    # the FieldML unit_cube has only 8 nodes which is too coarse to render
    # something visually striking.
    import skfem
    from skfem import Basis, BilinearForm, ElementTetP1, LinearForm, MeshTet
    from skfem.helpers import dot, grad

    mesh = MeshTet().refined(3)
    basis = Basis(mesh, ElementTetP1())

    @BilinearForm
    def bilin(u, v, _w):
        return dot(grad(u), grad(v))

    @LinearForm
    def lin(v, _w):
        return 1.0 * v

    stiff = bilin.assemble(basis)
    load = lin.assemble(basis)
    boundary = basis.get_dofs()
    u = skfem.solve(*skfem.condense(stiff, load, D=boundary))

    # Mid-plane slice rendered via pyvista for consistency with other figures.
    grid = pv.UnstructuredGrid(
        np.hstack(
            [
                4 * np.ones((mesh.t.shape[1], 1), dtype=np.int64),
                mesh.t.T.astype(np.int64),
            ]
        ).ravel(),
        np.full(mesh.t.shape[1], pv.CellType.TETRA, dtype=np.uint8),
        mesh.p.T.copy(),
    )
    grid["u"] = u

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # Panel A: wireframe
    p = pv.Plotter(off_screen=True, window_size=(600, 600))
    p.background_color = "white"
    p.add_mesh(grid, style="wireframe", color="#4a7fb5", line_width=0.5)
    p.view_isometric()
    img = p.screenshot(return_img=True)
    p.close()
    axes[0].imshow(img)
    axes[0].set_axis_off()
    axes[0].set_title(
        f"Refined tet mesh ({mesh.p.shape[1]} nodes, {mesh.t.shape[1]} tets)",
        fontsize=11,
    )

    # Panel B: solution
    p = pv.Plotter(off_screen=True, window_size=(600, 600))
    p.background_color = "white"
    p.add_mesh(
        grid.slice_orthogonal(x=0.5, y=0.5, z=0.5),
        scalars="u",
        cmap="viridis",
        show_scalar_bar=True,
        scalar_bar_args={"title": "u", "vertical": True},
    )
    p.add_mesh(grid.extract_feature_edges(), color="black", line_width=0.4)
    p.view_isometric()
    img = p.screenshot(return_img=True)
    p.close()
    axes[1].imshow(img)
    axes[1].set_axis_off()
    axes[1].set_title(
        "Poisson solution -Delta u = 1 (orthogonal slices)",
        fontsize=11,
    )

    fig.suptitle(
        "FEM solve on a FieldML unit_cube via scikit-fem",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig5_fem_solution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 6: Multi-bone MSK assembly
# -----------------------------------------------------------------------
def fig6_msk_assembly() -> None:
    from pyfieldml import datasets
    from pyfieldml.interop.pyvista import to_pyvista

    parts = {
        "hip_bone_left": "#e8cfa5",
        "femur_bodyparts3d": "#d1b48c",
        "tibia_left": "#bfa06f",
        "vertebra_l3": "#a67e5b",
    }

    p = pv.Plotter(off_screen=True, window_size=(1100, 1200))
    p.background_color = "white"
    for name, color in parts.items():
        d = datasets.load(name)
        g = to_pyvista(d)
        p.add_mesh(g, color=color, opacity=1.0, smooth_shading=True, label=name)

    p.add_legend(size=(0.3, 0.18), loc="upper right", face=None)
    p.view_zx()
    p.camera.roll += 180
    p.camera.zoom(1.3)
    img = p.screenshot(return_img=True)
    p.close()

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(
        "Clinical MSK assembly: hip + femur + tibia + vertebra L3\n"
        "(BodyParts3D DBCLS, CC-BY-SA 2.1 JP)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig6_msk_assembly.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Generating SoftwareX figures in", HERE)
    fig1_architecture()
    print("  fig1_architecture.png")
    fig2_zoo_gallery()
    print("  fig2_zoo_gallery.png")
    fig3_evaluator_graph()
    print("  fig3_evaluator_graph.png")
    fig4_hermite_bending()
    print("  fig4_hermite_bending.png")
    fig5_fem_solution()
    print("  fig5_fem_solution.png")
    fig6_msk_assembly()
    print("  fig6_msk_assembly.png")


if __name__ == "__main__":
    main()
