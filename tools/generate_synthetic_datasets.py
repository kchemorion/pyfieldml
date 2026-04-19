"""Regenerate the synthetic bundled datasets for pyfieldml.

Run from repo root:
    uv run python tools/generate_synthetic_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

import pyfieldml as fml
from pyfieldml.builders.biomech import add_fiber_field, add_scalar_field
from pyfieldml.builders.mesh import add_lagrange_mesh
from pyfieldml.model.region import Region

OUT = Path(__file__).parents[1] / "src/pyfieldml/datasets/_bundled"
OUT.mkdir(parents=True, exist_ok=True)


def _femur_envelope(pt: np.ndarray) -> bool:
    """Test whether ``pt = (x, y, z)`` lies inside the synthetic-femur CSG envelope.

    The femur is assembled from primitives in the local frame:

    * a shaft along +z on ``z in [0, 0.4]``, whose radius swells at the
      proximal trochanter and distal condyles;
    * a neck cylinder angling medially from the proximal shaft to a head;
    * a spherical femoral head at the end of the neck;
    * two distal condyle lobes offset in ``+x / -x`` near ``z = 0.41``.
    """
    x, y, z = pt[0], pt[1], pt[2]

    # Shaft: z in [0, 0.4], radius modulated axially.
    if 0.0 <= z <= 0.4:
        # Swell at proximal greater trochanter (wide) and distal condyle region.
        r_shaft = (
            0.012
            + 0.005 * np.exp(-((z - 0.05) ** 2) / 0.001)
            + 0.012 * np.exp(-((z - 0.38) ** 2) / 0.0005)
        )
        # Greater trochanter: extra bulge on +x side near z~0.02
        # (modelled as a radial boost when x > 0 and z in trochanter range).
        trochanter_boost = 0.0
        if x > 0 and 0.0 <= z <= 0.06:
            trochanter_boost = 0.008 * np.exp(-(((z - 0.02) ** 2) / 0.0008 + (y**2) / 0.0006))
        # Lesser trochanter: small bump on -x side, z~0.04
        lesser_boost = 0.0
        if x < 0 and 0.02 <= z <= 0.06:
            lesser_boost = 0.004 * np.exp(-(((z - 0.04) ** 2) / 0.0003 + (y**2) / 0.0004))
        rr = np.sqrt(x * x + y * y)
        if rr <= r_shaft + trochanter_boost + lesser_boost:
            return True

    # Femoral neck: a tapered cylinder from (0, 0, 0.01) to head centre.
    # Head centre ~ (0.03, 0, -0.02); direction roughly (+x, 0, -z).
    head_centre = np.array([0.03, 0.0, -0.02])
    neck_start = np.array([0.005, 0.0, 0.015])
    neck_vec = head_centre - neck_start
    neck_len = float(np.linalg.norm(neck_vec))
    neck_dir = neck_vec / neck_len
    rel = pt - neck_start
    t = float(np.dot(rel, neck_dir))
    if 0.0 <= t <= neck_len:
        perp = rel - t * neck_dir
        if float(np.linalg.norm(perp)) <= 0.011:
            return True

    # Femoral head: sphere.
    if float(np.linalg.norm(pt - head_centre)) <= 0.016:
        return True

    # Distal condyles: two spheres at +/-x near z=0.41.
    for cx in (0.018, -0.018):
        centre = np.array([cx, 0.0, 0.41])
        if float(np.linalg.norm(pt - centre)) <= 0.017:
            return True

    return False


def _femur_bmd(pt: np.ndarray) -> float:
    """Young's modulus proxy (Pa): denser at cortical surface, softer in marrow."""
    x, y = pt[0], pt[1]
    # Shaft radial distance to local axis. Assume axis near origin for shaft.
    rr = float(np.sqrt(x * x + y * y))
    # Cortical-like high modulus at outer shaft radius, trabecular/cancellous
    # interior. Below is a smooth radial ramp with a boost in head/condyle.
    base = 1.2e9 * min(rr / 0.02, 1.0) + 2.0e8
    # Dense cortical shell around femoral head and condyles (proxy).
    head_centre = np.array([0.03, 0.0, -0.02])
    if float(np.linalg.norm(pt - head_centre)) <= 0.018:
        base += 3.0e8
    for cx in (0.018, -0.018):
        centre = np.array([cx, 0.0, 0.41])
        if float(np.linalg.norm(pt - centre)) <= 0.019:
            base += 3.0e8
    return base


def generate_synthetic_femur(seed: int = 42) -> None:
    """Anatomical-synthetic femur: CSG-assembled shaft + head + neck + trochanters + condyles.

    The mesh is produced by rejection-sampling points inside the CSG envelope
    defined by :func:`_femur_envelope`, plus a thin layer of boundary seed
    points so that the Delaunay tetrahedralisation traces the external shape
    instead of the convex hull of the interior samples alone.
    """
    rng = np.random.default_rng(seed)

    # Bounding box for rejection sampling.
    # x: [-0.04, 0.05], y: [-0.025, 0.025], z: [-0.04, 0.43]
    bbox_lo = np.array([-0.04, -0.025, -0.04])
    bbox_hi = np.array([0.05, 0.025, 0.43])
    bbox_diff = bbox_hi - bbox_lo

    n_target_interior = 180
    interior: list[np.ndarray] = []
    attempts = 0
    max_attempts = 60_000
    while len(interior) < n_target_interior and attempts < max_attempts:
        batch = rng.random((512, 3)) * bbox_diff + bbox_lo
        for p in batch:
            attempts += 1
            if _femur_envelope(p):
                interior.append(p)
                if len(interior) >= n_target_interior:
                    break
    interior_arr = np.array(interior)

    # Boundary seed rings so the tet mesh traces the shape.
    boundary: list[np.ndarray] = []

    # Shaft rings.
    for z_val in np.linspace(0.01, 0.39, 12):
        r_shaft = (
            0.012
            + 0.005 * np.exp(-((z_val - 0.05) ** 2) / 0.001)
            + 0.012 * np.exp(-((z_val - 0.38) ** 2) / 0.0005)
        )
        for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            boundary.append(np.array([r_shaft * np.cos(theta), r_shaft * np.sin(theta), z_val]))

    # Femoral-head sphere samples (poles + equatorial ring).
    head_centre = np.array([0.03, 0.0, -0.02])
    for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        for phi in (np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            boundary.append(
                head_centre
                + 0.016
                * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
            )

    # Distal condyle caps.
    for cx in (0.018, -0.018):
        centre = np.array([cx, 0.0, 0.41])
        for theta in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            for phi in (np.pi / 3, 2 * np.pi / 3):
                boundary.append(
                    centre
                    + 0.017
                    * np.array(
                        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
                    )
                )

    boundary_arr = np.array(boundary)

    points = np.vstack([interior_arr, boundary_arr])

    tri = Delaunay(points)
    # Filter out tets whose centroids fall outside the CSG envelope, so the
    # mesh traces the anatomical shape instead of the bounding convex hull.
    centroids = points[tri.simplices].mean(axis=1)
    keep = np.array([_femur_envelope(c) for c in centroids])
    tets = tri.simplices[keep]

    # Drop any nodes that ended up unused after the tet filter (keeps the
    # node count tight and the output file small).
    used = np.unique(tets)
    remap = np.full(points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    points = points[used]
    tets = remap[tets]

    # BMD / Young's-modulus field per node.
    bmd = np.array([_femur_bmd(p) for p in points], dtype=np.float64)

    r = Region(name="femur")
    add_lagrange_mesh(
        r,
        name="femur_mesh",
        nodes=points.astype(np.float64),
        elements=(tets + 1).astype(np.int64),  # 1-indexed for FieldML
        topology="tet",
        order=1,
        coord_name="coordinates",
    )
    add_scalar_field(
        r,
        name="youngs_modulus",
        mesh_name="femur_mesh",
        values=bmd.astype(np.float64),
    )

    doc = fml.Document.from_region(r)
    out = OUT / "femur.fieldml"
    doc.write(out)
    print(f"wrote {out} ({points.shape[0]} nodes, {tets.shape[0]} tets)")


def generate_synthetic_rectus_femoris(seed: int = 43) -> None:
    """Spindle-shaped muscle tet mesh with a fiber-direction vector field."""
    rng = np.random.default_rng(seed)
    length = 0.3

    def envelope_radius(z: np.ndarray) -> np.ndarray:
        return 0.015 * np.sin(np.pi * z / length)

    # Rejection sampling inside the spindle envelope.
    n_target = 80
    samples: list[np.ndarray] = []
    while len(samples) < n_target:
        z = rng.random(n_target * 3) * length
        r_env = envelope_radius(z)
        r_try = rng.random(n_target * 3) * 0.015
        th = 2 * np.pi * rng.random(n_target * 3)
        accept = r_try <= r_env
        for i in np.flatnonzero(accept):
            xv = r_try[i] * np.cos(th[i])
            yv = r_try[i] * np.sin(th[i])
            samples.append(np.array([xv, yv, z[i]]))
            if len(samples) >= n_target:
                break
    interior = np.array(samples)

    # Envelope surface points to keep the convex hull meaningful.
    surface = []
    for z_val in np.linspace(0.02, length - 0.02, 10):
        rr = envelope_radius(np.array([z_val]))[0]
        for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            surface.append([rr * np.cos(theta), rr * np.sin(theta), z_val])
    # Tips
    surface.append([0.0, 0.0, 0.0])
    surface.append([0.0, 0.0, length])
    points = np.vstack([interior, np.array(surface)])

    tri = Delaunay(points)
    tets = tri.simplices

    # Fiber direction: aligned with +z at every node.
    fibers = np.tile([0.0, 0.0, 1.0], (points.shape[0], 1))

    r = Region(name="rectus_femoris")
    add_lagrange_mesh(
        r,
        name="muscle_mesh",
        nodes=points.astype(np.float64),
        elements=(tets + 1).astype(np.int64),
        topology="tet",
        order=1,
        coord_name="coordinates",
    )
    add_fiber_field(r, mesh_name="muscle_mesh", vectors=fibers.astype(np.float64))

    doc = fml.Document.from_region(r)
    out = OUT / "rectus_femoris.fieldml"
    doc.write(out)
    print(f"wrote {out} ({points.shape[0]} nodes, {tets.shape[0]} tets)")


if __name__ == "__main__":
    generate_synthetic_femur()
    generate_synthetic_rectus_femoris()
