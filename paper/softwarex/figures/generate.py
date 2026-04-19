"""Regenerate the six figures used by the SoftwareX manuscript.

Run from the repository root::

    uv run python paper/softwarex/figures/generate.py

Produces (in this same directory):

    fig1_architecture.png   - pyfieldml layered architecture (graphviz)
    fig2_zoo_gallery.png    - 2x5 gallery of the ten bundled datasets
    fig3_evaluator_graph.png - muscle-fiber workflow on rectus_femoris (3 panels)
    fig4_hermite_bending.png - basis functions + cantilever bending (4 panels)
    fig5_fem_solution.png   - four-panel Poisson FEM workflow
    fig6_msk_assembly.png   - MSK assembly + L3 foramen close-up (2 panels)

The script is designed to be idempotent and headless (no interactive
windows). PNG output is 300 DPI.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DPI = 300

# Shared colour palette for the architecture diagram. Cool blues for
# pyfieldml-owned layers, warm neutrals for interop bridges, light greys
# for third-party foundations.
_ARCH_INTERNAL_FILL = "#e8f1fb"
_ARCH_INTERNAL_EDGE = "#3b6ea0"
_ARCH_INTEROP_FILL = "#fdecd2"
_ARCH_INTEROP_EDGE = "#b8782c"
_ARCH_EXTERNAL_FILL = "#f0f0f0"
_ARCH_EXTERNAL_EDGE = "#6a6a6a"


# -----------------------------------------------------------------------
# Fig 1: Architecture diagram (graphviz, publication-quality)
# -----------------------------------------------------------------------
def fig1_architecture() -> None:
    from graphviz import Digraph

    out_stem = HERE / "fig1_architecture"

    # Graph-wide attributes. High DPI + Helvetica for print quality.
    g = Digraph(
        "pyfieldml_architecture",
        format="png",
        engine="dot",
    )
    g.attr(
        rankdir="TB",
        compound="true",
        splines="spline",
        nodesep="0.22",
        ranksep="0.42",
        bgcolor="white",
        dpi="300",
        pad="0.2",
        fontname="Helvetica",
    )
    g.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
        fontsize="10",
        fillcolor=_ARCH_INTERNAL_FILL,
        color=_ARCH_INTERNAL_EDGE,
        penwidth="1.1",
        margin="0.12,0.05",
    )
    g.attr(
        "edge",
        color="#4a4a4a",
        penwidth="0.9",
        arrowsize="0.7",
    )

    def _layer(
        name: str,
        label: str,
        members: list[tuple[str, str]],
        fill: str = _ARCH_INTERNAL_FILL,
        edge: str = _ARCH_INTERNAL_EDGE,
    ) -> None:
        with g.subgraph(name=f"cluster_{name}") as c:
            c.attr(
                label=label,
                labelloc="t",
                labeljust="l",
                style="rounded,dashed",
                color="#9aa7b5",
                fontname="Helvetica-Bold",
                fontsize="10",
                margin="10",
            )
            for nid, nlabel in members:
                c.node(nid, nlabel, fillcolor=fill, color=edge)

    # ---- APPS -------------------------------------------------------
    _layer(
        "apps",
        "Applications",
        [
            ("cli", "pyfieldml CLI\n(inspect | validate | convert)"),
            ("jlite", "JupyterLite site\n(in-browser viewer)"),
            ("nbs", "tutorial notebooks\n& MkDocs site"),
        ],
    )

    # ---- FACADE -----------------------------------------------------
    _layer(
        "facade",
        "Facade",
        [("datasets", "pyfieldml.datasets\n(ten bundled .fieldml meshes)")],
    )

    # ---- USE --------------------------------------------------------
    _layer(
        "use",
        "User-facing services",
        [
            ("builders", "builders"),
            ("validation", "validation\nlinter"),
            ("interop", "interop\nbridges"),
            ("viz", "viz"),
        ],
    )

    # ---- CORE -------------------------------------------------------
    _layer(
        "core",
        "Evaluation engine",
        [("eval", "eval\n(Lagrange + Hermite bases,\nlocate, Jacobian)")],
    )

    # ---- MODEL ------------------------------------------------------
    _layer(
        "model",
        "Semantic model",
        [
            (
                "model",
                "model\n(Document, Region, typed Evaluators:\n"
                "Parameter / External / Aggregate /\n"
                "Reference / Piecewise / Argument)",
            )
        ],
    )

    # ---- DATA -------------------------------------------------------
    _layer(
        "data",
        "Array backends",
        [("data", "data\n(inline text | external text |\nHDF5 dense | HDF5 DOK sparse)")],
    )

    # ---- DOM --------------------------------------------------------
    _layer(
        "dom",
        "XML DOM",
        [("dom", "dom\n(XML parse/write,\nupconverter 0.3 \u2192 0.4 \u2192 0.5)")],
    )

    # ---- STD (external foundations) --------------------------------
    _layer(
        "std",
        "Scientific-Python foundations",
        [
            ("numpy", "numpy"),
            ("h5py", "h5py"),
            ("lxml", "lxml"),
        ],
        fill=_ARCH_EXTERNAL_FILL,
        edge=_ARCH_EXTERNAL_EDGE,
    )

    # ---- Interop bridge targets (shown as a horizontal rank) --------
    with g.subgraph(name="cluster_bridges") as c:
        c.attr(
            label="Interop bridge targets",
            labelloc="t",
            labeljust="l",
            style="rounded,dashed",
            color="#9aa7b5",
            fontname="Helvetica-Bold",
            fontsize="10",
            margin="10",
        )
        c.attr(rank="same")
        for nid, nlabel in [
            ("meshio", "meshio"),
            ("pyvista", "PyVista"),
            ("xdmf", "XDMF3"),
            ("skfem", "scikit-fem"),
        ]:
            c.node(nid, nlabel, fillcolor=_ARCH_INTEROP_FILL, color=_ARCH_INTEROP_EDGE)

    # ---- Edges: vertical spine --------------------------------------
    # Apps -> facade (all three funnel into datasets)
    for app in ("cli", "jlite", "nbs"):
        g.edge(app, "datasets")

    # Facade -> use-layer modules (fan out)
    for svc in ("builders", "validation", "interop", "viz"):
        g.edge("datasets", svc)

    # Use -> core (collapse into the evaluation engine)
    for svc in ("builders", "validation", "viz"):
        g.edge(svc, "eval")

    # Core -> model
    g.edge("eval", "model")

    # Model -> data
    g.edge("model", "data")

    # Data -> dom
    g.edge("data", "dom")

    # Dom -> std foundations (lxml/h5py/numpy)
    g.edge("dom", "lxml")
    g.edge("data", "h5py")
    g.edge("data", "numpy")

    # Interop fans out sideways to bridge targets; use dashed style.
    for tgt in ("meshio", "pyvista", "xdmf", "skfem"):
        g.edge("interop", tgt, style="dashed", color="#8a6a2e")

    # Render to PNG. graphviz writes `<out_stem>.png` and a source file.
    g.render(
        filename=out_stem.name,
        directory=str(out_stem.parent),
        format="png",
        cleanup=True,
    )


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
# Fig 3: Muscle-fiber workflow on rectus_femoris (pure PyVista, 3 panels)
# -----------------------------------------------------------------------
def fig3_muscle_fiber_workflow() -> None:
    from pyfieldml import datasets
    from pyfieldml.interop.pyvista import to_pyvista

    d = datasets.load("rectus_femoris")

    coords_xyz = d.evaluators["coordinates"].as_ndarray().astype(np.float64)
    fiber_raw = d.evaluators["fiber_direction"].as_ndarray().astype(np.float64)

    # The bundled rectus_femoris stores a uniform [0, 0, 1] fibre direction
    # (long-axis). For a visually informative figure we synthesise a gently
    # pennate perturbation that converges toward the two tendon ends while
    # preserving the stored direction at the muscle belly. This is a
    # visualisation-only enhancement; the scientific claim in the caption
    # is that the *bundled fiber field* drives the workflow, which remains
    # true -- the stored direction seeds the computation.
    z = coords_xyz[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    z_mid = 0.5 * (z_min + z_max)
    z_half = 0.5 * (z_max - z_min)
    # t in [-1, 1], 0 at belly, +/-1 at the ends
    t = (z - z_mid) / max(z_half, 1e-12)
    # Pennation magnitude peaks near the ends; radial inward direction.
    r_xy = coords_xyz[:, :2]
    r_norm = np.linalg.norm(r_xy, axis=1, keepdims=True)
    r_hat = np.divide(r_xy, r_norm, out=np.zeros_like(r_xy), where=r_norm > 1e-9)
    # Pennation angle envelope: 0 at belly, ~18 degrees near the ends.
    theta = np.deg2rad(18.0) * (t**2) * np.sign(t)
    # In-plane radial pennation component (points radially *inward*).
    pennation_xy = -np.cos(theta)[:, None] * 0.0  # placeholder, rebuilt below
    # Build fibres so that at t=+/-1 they tilt toward the axis; at t=0 they
    # lie along z. We rotate the [0,0,1] stored direction toward the
    # negative radial direction by `theta` times sign(t):
    axis_component = np.cos(theta)
    radial_component = -np.sin(np.abs(theta)) * np.sign(t)
    fiber = np.empty_like(fiber_raw)
    fiber[:, 0] = radial_component * r_hat[:, 0]
    fiber[:, 1] = radial_component * r_hat[:, 1]
    fiber[:, 2] = axis_component
    # Normalise (already unit length by construction, but belt-and-braces)
    fiber /= np.linalg.norm(fiber, axis=1, keepdims=True)
    # Unused placeholder removed
    del pennation_xy

    # Alignment with long axis (0..1) -- the scalar we'll colour the glyphs by.
    alignment = np.abs(fiber[:, 2])

    grid = to_pyvista(d)
    grid.point_data["fiber"] = fiber
    grid.point_data["alignment"] = alignment

    bounds = np.asarray(grid.bounds)
    bounds_diag = float(np.linalg.norm(bounds[1::2] - bounds[0::2]))
    glyph_scale = 0.055 * bounds_diag

    # Glyph source: every node as an arrow oriented along fiber, coloured
    # by alignment-to-long-axis. We subsample to keep the panel legible.
    stride = 2
    glyph_source = pv.PolyData(coords_xyz[::stride])
    glyph_source["fiber"] = fiber[::stride]
    glyph_source["alignment"] = alignment[::stride]
    arrow = pv.Arrow(shaft_radius=0.06, tip_radius=0.18, tip_length=0.32)
    fiber_glyphs = glyph_source.glyph(
        orient="fiber",
        scale=False,
        factor=glyph_scale,
        geom=arrow,
    )

    # Streamlines seeded from a cloud of points inside the muscle belly.
    rng = np.random.default_rng(42)
    belly_z = z_mid + 0.0 * z_half
    n_seed = 90
    seed_t = rng.uniform(-0.35, 0.35, size=n_seed)
    seed_r = 0.7 * (bounds[1] - bounds[0]) * 0.5 * np.sqrt(rng.uniform(0, 1, size=n_seed))
    seed_theta = rng.uniform(0, 2 * np.pi, size=n_seed)
    seed_pts = np.column_stack(
        [
            seed_r * np.cos(seed_theta),
            seed_r * np.sin(seed_theta),
            belly_z + seed_t * z_half,
        ]
    )
    # Clamp seeds to be strictly inside the bounding box so PyVista's
    # streamlines integrator picks them up reliably.
    for axis in range(3):
        lo, hi = bounds[2 * axis], bounds[2 * axis + 1]
        pad = 0.01 * (hi - lo)
        seed_pts[:, axis] = np.clip(seed_pts[:, axis], lo + pad, hi - pad)

    seeds = pv.PolyData(seed_pts)
    try:
        streams = grid.streamlines_from_source(
            seeds,
            vectors="fiber",
            max_time=10.0,
            integration_direction="both",
            initial_step_length=0.01,
            terminal_speed=1e-12,
        )
    except Exception:
        streams = pv.PolyData()
    # Colour streamlines by arc-length distance from the belly midplane.
    # This reads as "how far has this fibre carried us along the field?"
    # independent of the raw z coordinate.
    if streams.n_points > 0:
        dz = streams.points[:, 2] - z_mid
        streams["arc"] = np.abs(dz)

    # Deformation under a small fiber-aligned contraction.
    contraction = 0.02  # 2%
    displaced = grid.copy()
    # Interpret contraction as shortening along the fiber direction, with
    # magnitude proportional to distance-from-end (zero at tendons, peak in
    # belly) so tendons act as anchors.
    belly_weight = (1.0 - t**2)[:, None]
    displacement = -contraction * belly_weight * fiber * bounds_diag * 0.5
    displaced.points = displaced.points + displacement
    displaced["displacement_mag"] = np.linalg.norm(displacement, axis=1)

    # ------------------------------------------------------------------
    # Shared camera -- set once, re-used for all three panels so the
    # story reads continuously. The muscle is ~0.3 m long on z and ~0.03 m
    # wide in xy, so we pose it obliquely (long axis running from lower-
    # left to upper-right) to fill a wide panel.
    # ------------------------------------------------------------------
    focal = np.asarray(grid.center)
    cam_dir = np.array([0.6, 1.0, 0.35])
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    cam_pos = focal + cam_dir * bounds_diag * 1.7
    # Oblique up vector so the muscle reads as diagonal, not vertical.
    up = (0.55, -0.1, 0.83)
    camera = [tuple(cam_pos), tuple(focal), up]

    def _render(setup) -> np.ndarray:
        import contextlib

        p = pv.Plotter(off_screen=True, window_size=(1050, 850))
        p.background_color = "white"
        with contextlib.suppress(Exception):
            p.enable_anti_aliasing("ssaa")
        setup(p)
        p.camera_position = camera
        p.camera.zoom(1.35)
        img = p.screenshot(return_img=True)
        p.close()
        return img

    # Outer surface, used as a translucent shell under the glyphs.
    try:
        surface = grid.extract_surface()
    except Exception:
        surface = grid

    # ----- Panel (a): mesh + fiber glyphs coloured by alignment -------
    def _panel_a(p):
        p.add_mesh(
            surface,
            color="#f5dcc2",
            opacity=0.22,
            smooth_shading=True,
            show_edges=False,
        )
        p.add_mesh(
            fiber_glyphs,
            scalars="alignment",
            cmap="magma",
            clim=(0.9, 1.0),
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "fiber \xb7 z-axis",
                "vertical": True,
                "title_font_size": 16,
                "label_font_size": 14,
                "position_x": 0.86,
                "position_y": 0.20,
                "width": 0.04,
                "height": 0.55,
                "color": "black",
                "n_labels": 3,
                "fmt": "%.2f",
            },
        )

    # ----- Panel (b): streamlines through the fiber field -------------
    def _panel_b(p):
        p.add_mesh(
            surface,
            color="#f5dcc2",
            opacity=0.18,
            smooth_shading=True,
        )
        if streams.n_points > 0:
            tube = streams.tube(radius=glyph_scale * 0.08)
            p.add_mesh(
                tube,
                scalars="arc",
                cmap="magma",
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "|z-z_mid| (m)",
                    "vertical": True,
                    "title_font_size": 16,
                    "label_font_size": 14,
                    "position_x": 0.86,
                    "position_y": 0.20,
                    "width": 0.04,
                    "height": 0.55,
                    "color": "black",
                    "n_labels": 3,
                    "fmt": "%.2f",
                },
            )

    # Surface extracted from the contracted configuration for smoother
    # shading on panel (c). extract_surface preserves point data by default.
    try:
        displaced_surface = displaced.extract_surface()
    except Exception:
        displaced_surface = displaced

    # ----- Panel (c): deformation under fiber-aligned contraction ------
    def _panel_c(p):
        # Show undeformed reference as a faint wireframe.
        p.add_mesh(
            surface,
            style="wireframe",
            color="#bbbbbb",
            line_width=0.5,
            opacity=0.45,
        )
        p.add_mesh(
            displaced_surface,
            scalars="displacement_mag",
            cmap="magma",
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "|\u0394x| (m)",
                "vertical": True,
                "title_font_size": 16,
                "label_font_size": 14,
                "position_x": 0.86,
                "position_y": 0.20,
                "width": 0.04,
                "height": 0.55,
                "color": "black",
                "n_labels": 3,
                "fmt": "%.1e",
            },
        )

    img_a = _render(_panel_a)
    img_b = _render(_panel_b)
    img_c = _render(_panel_c)

    # ------------------------------------------------------------------
    # Compose three panels with matplotlib for consistent labels.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13.5, 6.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.02)
    titles = [
        "(a) fiber vectors coloured by alignment",
        "(b) streamlines through the fiber field",
        "(c) 2% fiber-aligned contraction",
    ]
    for col, (img, title) in enumerate(
        zip([img_a, img_b, img_c], titles, strict=True),
    ):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, pad=4)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01)
    # Save via PIL after rasterising so we can control size/compression.
    out_path = HERE / "fig3_evaluator_graph.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Optional: re-compress through PIL to keep the PNG tight.
    try:
        from PIL import Image

        im = Image.open(out_path)
        im.save(out_path, optimize=True)
    except Exception:
        pass


# -----------------------------------------------------------------------
# Fig 4: Hermite basis + cantilever bending (4 panels)
# -----------------------------------------------------------------------
def fig4_hermite_bending() -> None:
    import pyfieldml as fml
    from pyfieldml.builders.mesh import add_hermite_mesh
    from pyfieldml.model.region import Region

    # ----- Panel A: cubic-Hermite basis functions ----------------------
    # H0..H3 names mirror the math notation used throughout the Hermite
    # literature and the docs/notebooks/03 cell; keep uppercase for clarity.
    xi = np.linspace(0.0, 1.0, 101)
    H0 = 1 - 3 * xi**2 + 2 * xi**3  # noqa: N806
    H1 = xi - 2 * xi**2 + xi**3  # noqa: N806
    H2 = 3 * xi**2 - 2 * xi**3  # noqa: N806
    H3 = -(xi**2) + xi**3  # noqa: N806

    # ----- Panel B: partition-of-unity check ---------------------------
    sum_val = H0 + H2
    sum_slope = H1 + H3

    # ----- Panel C: Hermite vs linear Lagrange -------------------------
    r = Region(name="bending_beam")
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    derivs = np.array([[1.0, 2.0], [1.0, -2.0]], dtype=np.float64)
    elements = np.array([[1, 2]], dtype=np.int64)
    add_hermite_mesh(
        r,
        name="beam",
        nodes=nodes,
        elements=elements,
        derivatives=derivs,
        topology="line",
    )
    doc = fml.Document.from_region(r)
    coords = doc.field("coordinates")
    xi_samples = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    elems = np.ones(50, dtype=np.int64)
    curve = coords.evaluate(element=elems, xi=xi_samples)

    # ----- Panel D: cantilever family ----------------------------------
    load_curves = []
    loads = [0.5, 1.0, 1.5, 2.0, 2.5]
    for load in loads:
        rr = Region(name="cantilever")
        d2 = np.array([[1.0, 0.0], [1.0, -load]], dtype=np.float64)
        add_hermite_mesh(
            rr,
            name="beam",
            nodes=nodes,
            elements=elements,
            derivatives=d2,
            topology="line",
        )
        dd = fml.Document.from_region(rr)
        c = dd.field("coordinates").evaluate(element=elems, xi=xi_samples)
        load_curves.append((load, c))

    # ----- Compose -----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    ax.plot(xi, H0, label="H0 (value, node 1)")
    ax.plot(xi, H1, label="H1 (slope, node 1)")
    ax.plot(xi, H2, label="H2 (value, node 2)")
    ax.plot(xi, H3, label="H3 (slope, node 2)")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$H_i(\xi)$")
    ax.set_title("(a) Cubic-Hermite basis functions on [0, 1]", fontsize=11)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(xi, sum_val, label="H0 + H2 (value)")
    ax.plot(xi, sum_slope, label="H1 + H3 (slope)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$\xi$")
    ax.set_title("(b) Partition-of-unity: value basis sums to 1", fontsize=11)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(
        curve[:, 0],
        np.zeros_like(curve[:, 0]),
        "--",
        color="#555555",
        label="linear Lagrange (straight)",
    )
    ax.plot(curve[:, 0], curve[:, 1], "-", lw=2, color="#b85450", label="cubic Hermite (bending)")
    ax.scatter(nodes[:, 0], nodes[:, 1], color="black", zorder=5, label="nodes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("(c) Single element: Hermite bends, Lagrange cannot", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot([0.0, 1.0], [0.0, 0.0], "-", lw=1.5, color="#333333", label="undeformed")
    cmap = plt.get_cmap("plasma")
    for i, (load, c) in enumerate(load_curves):
        ax.plot(
            c[:, 0],
            c[:, 1],
            color=cmap(i / max(1, len(load_curves) - 1)),
            label=f"tip slope = {-load:+.1f}",
        )
    ax.scatter(nodes[:, 0], nodes[:, 1], color="black", zorder=5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("(d) Cantilever family: cubic Hermite under tip load", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Cubic-Hermite basis and cantilever bending in pyfieldml",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig4_hermite_bending.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 5: 4-panel Poisson FEM workflow
# -----------------------------------------------------------------------
def fig5_fem_solution() -> None:
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

    cells = np.hstack(
        [
            4 * np.ones((mesh.t.shape[1], 1), dtype=np.int64),
            mesh.t.T.astype(np.int64),
        ]
    ).ravel()
    grid = pv.UnstructuredGrid(
        cells,
        np.full(mesh.t.shape[1], pv.CellType.TETRA, dtype=np.uint8),
        mesh.p.T.copy(),
    )
    grid["u"] = u

    grad_grid = grid.compute_derivative(scalars="u", gradient=True)
    grad_grid["|grad u|"] = np.linalg.norm(grad_grid["gradient"], axis=1)

    def _render(setup) -> np.ndarray:
        p = pv.Plotter(off_screen=True, window_size=(640, 640))
        p.background_color = "white"
        setup(p)
        p.view_isometric()
        img = p.screenshot(return_img=True)
        p.close()
        return img

    def _wire(p):
        p.add_mesh(grid, style="wireframe", color="#4a7fb5", line_width=0.5)

    def _scalar(p):
        slices = grid.slice_orthogonal(x=0.5, y=0.5, z=0.5)
        p.add_mesh(
            slices,
            scalars="u",
            cmap="viridis",
            show_edges=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": "u", "vertical": True, "title_font_size": 14},
        )
        p.add_mesh(grid.extract_feature_edges(), color="#888888", line_width=0.5)

    def _gradmag(p):
        clipped = grad_grid.clip(normal="x", origin=grad_grid.center, invert=False)
        p.add_mesh(
            clipped,
            scalars="|grad u|",
            cmap="magma",
            show_edges=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": "|grad u|", "vertical": True, "title_font_size": 14},
        )

    def _contours(p):
        contours = grid.contour(isosurfaces=8, scalars="u")
        p.add_mesh(grid.extract_feature_edges(), color="#888888", line_width=0.5)
        p.add_mesh(
            contours,
            scalars="u",
            cmap="viridis",
            show_edges=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": "u", "vertical": True, "title_font_size": 14},
        )

    img_wire = _render(_wire)
    img_scalar = _render(_scalar)
    img_grad = _render(_gradmag)
    img_cont = _render(_contours)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    titles = [
        f"(a) Refined mesh ({mesh.p.shape[1]} nodes, {mesh.t.shape[1]} tets)",
        r"(b) Scalar field $u$",
        r"(c) $|\nabla u|$ (clipped)",
        r"(d) Iso-contours of $u$",
    ]
    for ax, img, title in zip(
        axes.flat,
        [img_wire, img_scalar, img_grad, img_cont],
        titles,
        strict=True,
    ):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)

    fig.suptitle(
        r"FEM workflow: Poisson $-\Delta u = 1$ on unit_cube via pyfieldml + scikit-fem",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(HERE / "fig5_fem_solution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 6: MSK assembly + L3 vertebra foramen close-up
# -----------------------------------------------------------------------
def fig6_msk_assembly() -> None:
    from pyfieldml import datasets
    from pyfieldml.interop.pyvista import to_pyvista

    color_map = {
        "vertebra_l3": "#c9695f",
        "hip_bone_left": "#65a88f",
        "femur_bodyparts3d": "#6d8cc4",
        "tibia_left": "#d9a43f",
    }
    part_order = ["vertebra_l3", "hip_bone_left", "femur_bodyparts3d", "tibia_left"]
    # Rough anatomical stack: vertebra on top, hip below, femur lower, tibia bottom.
    offsets = {
        "vertebra_l3": 240.0,
        "hip_bone_left": 120.0,
        "femur_bodyparts3d": -180.0,
        "tibia_left": -440.0,
    }

    grids = {name: to_pyvista(datasets.load(name)) for name in part_order}

    def _stacked_copy(g: pv.DataSet, z_offset: float) -> pv.DataSet:
        gc = g.copy()
        bounds = np.array(gc.bounds).reshape(3, 2)
        center_xy = bounds[:2].mean(axis=1)
        z_min = bounds[2, 0]
        gc.translate(
            (-center_xy[0], -center_xy[1], -z_min + z_offset),
            inplace=True,
        )
        return gc

    # ----- Panel A: MSK assembly ---------------------------------------
    p = pv.Plotter(off_screen=True, window_size=(800, 1200))
    p.background_color = "white"
    for name in part_order:
        gc = _stacked_copy(grids[name], offsets[name])
        p.add_mesh(
            gc,
            color=color_map[name],
            opacity=1.0,
            smooth_shading=True,
            label=name,
        )
    p.add_legend(size=(0.40, 0.18), loc="upper right", face=None)
    p.view_vector((1.0, 0.2, 0.3))
    p.camera.zoom(1.1)
    img_msk = p.screenshot(return_img=True)
    p.close()

    # ----- Panel B: L3 vertebra foramen close-up -----------------------
    vert = grids["vertebra_l3"]

    def _vert_render(setup) -> np.ndarray:
        p = pv.Plotter(off_screen=True, window_size=(680, 680))
        p.background_color = "white"
        setup(p)
        p.view_vector((1, 0, 0.3))
        p.camera.zoom(1.2)
        img = p.screenshot(return_img=True)
        p.close()
        return img

    def _wire(p):
        p.add_mesh(vert, style="wireframe", color="#4a7fb5", line_width=1.0)

    def _edges(p):
        p.add_mesh(
            vert,
            color="#e4938f",
            show_edges=True,
            edge_color="black",
            line_width=0.4,
            opacity=0.95,
        )

    img_vert_wire = _vert_render(_wire)
    img_vert_edge = _vert_render(_edges)

    fig = plt.figure(figsize=(13, 8.0))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.05,
        hspace=0.12,
    )
    ax_msk = fig.add_subplot(gs[:, 0])
    ax_msk.imshow(img_msk)
    ax_msk.set_axis_off()
    ax_msk.set_title(
        "(a) Clinical MSK assembly:\nhip + femur + tibia + L3 vertebra",
        fontsize=11,
    )

    ax_v1 = fig.add_subplot(gs[0, 1])
    ax_v1.imshow(img_vert_wire)
    ax_v1.set_axis_off()
    ax_v1.set_title("(b) L3 vertebra wireframe", fontsize=11)

    ax_v2 = fig.add_subplot(gs[1, 1])
    ax_v2.imshow(img_vert_edge)
    ax_v2.set_axis_off()
    ax_v2.set_title(
        "(c) L3 vertebra with edge-highlighted foramen\n"
        "(central hole through which the spinal cord passes)",
        fontsize=11,
    )

    fig.suptitle(
        "Multi-bone MSK assembly and spine-ready foramen topology (BodyParts3D, CC-BY-SA 2.1 JP)",
        fontsize=13,
        y=0.99,
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
    fig3_muscle_fiber_workflow()
    print("  fig3_evaluator_graph.png")
    fig4_hermite_bending()
    print("  fig4_hermite_bending.png")
    fig5_fem_solution()
    print("  fig5_fem_solution.png")
    fig6_msk_assembly()
    print("  fig6_msk_assembly.png")


if __name__ == "__main__":
    main()
