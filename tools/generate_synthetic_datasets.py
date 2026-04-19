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


# ---------------- Femur CSG primitives ----------------
#
# All coordinates in metres, adult-femur-scale:
#   - total length ~ 0.40 m
#   - head radius ~ 0.022 m
# The femur runs along +z, with the proximal (head) end at high z and the
# distal (condyles) end at low z.

_SHAFT_R = 0.013
_SHAFT_Z_LO = 0.05
_SHAFT_Z_HI = 0.35

_HEAD_C = np.array([0.035, 0.0, 0.41])
_HEAD_R = 0.022

_NECK_C = np.array([0.020, 0.0, 0.385])
_NECK_AXES = np.array([0.020, 0.010, 0.022])

_GT_C = np.array([-0.020, 0.0, 0.40])
_GT_AXES = np.array([0.014, 0.012, 0.020])

_LT_C = np.array([0.010, -0.008, 0.39])
_LT_AXES = np.array([0.007, 0.006, 0.010])

_MC_C = np.array([0.015, 0.0, 0.03])
_MC_AXES = np.array([0.018, 0.020, 0.015])

_LC_C = np.array([-0.015, 0.0, 0.03])
_LC_AXES = np.array([0.018, 0.020, 0.015])


def _inside_union(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask: True for points inside ANY of the anatomical parts.

    Vectorised over an ``(N, 3)`` array of query points so the rejection
    sampler and the tet-centroid filter can both call it on big batches.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inside = np.zeros(len(points), dtype=bool)

    # Shaft: truncated cylinder along +z.
    inside |= (np.sqrt(x**2 + y**2) < _SHAFT_R) & (z > _SHAFT_Z_LO) & (z < _SHAFT_Z_HI)

    # Femoral head: sphere at proximal end, offset medially.
    inside |= np.linalg.norm(points - _HEAD_C, axis=1) < _HEAD_R

    # Femoral neck: ellipsoid bridging shaft and head.
    inside |= (((points - _NECK_C) / _NECK_AXES) ** 2).sum(axis=1) < 1.0

    # Greater trochanter: prolate ellipsoid lateral to the neck-shaft junction.
    inside |= (((points - _GT_C) / _GT_AXES) ** 2).sum(axis=1) < 1.0

    # Lesser trochanter: smaller ellipsoid medial side, slightly below head.
    inside |= (((points - _LT_C) / _LT_AXES) ** 2).sum(axis=1) < 1.0

    # Medial condyle (distal).
    inside |= (((points - _MC_C) / _MC_AXES) ** 2).sum(axis=1) < 1.0

    # Lateral condyle (distal).
    inside |= (((points - _LC_C) / _LC_AXES) ** 2).sum(axis=1) < 1.0

    return inside


def _ellipsoid_surface_samples(
    centre: np.ndarray,
    axes: np.ndarray,
    n_phi: int,
    n_theta: int,
    phi_inset: float = 0.3,
) -> list[list[float]]:
    """Sample points on the surface of an axis-aligned ellipsoid.

    ``phi_inset`` trims the polar caps so we don't double-stack samples on
    the z-axis; Delaunay dislikes near-coincident points.
    """
    out: list[list[float]] = []
    for phi in np.linspace(phi_inset, np.pi - phi_inset, n_phi):
        for theta in np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False):
            p = centre + axes * np.array(
                [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
            )
            out.append(p.tolist())
    return out


def _femur_boundary_seeds() -> np.ndarray:
    """Return boundary-surface seed points so Delaunay traces the CSG shape.

    Without these, Delaunay tets would only span the convex hull of the
    rejection-sampled interior cloud and the greater trochanter / condyles
    / head wouldn't be outlined properly.
    """
    boundary: list[list[float]] = []

    # Shaft: 7 rings of 7 points along the mid-shaft.
    for z_val in np.linspace(0.06, 0.34, 7):
        for theta in np.linspace(0.0, 2 * np.pi, 7, endpoint=False):
            boundary.append([_SHAFT_R * np.cos(theta), _SHAFT_R * np.sin(theta), float(z_val)])

    # Femoral head surface.
    boundary.extend(_ellipsoid_surface_samples(_HEAD_C, np.full(3, _HEAD_R), n_phi=4, n_theta=7))
    # One extra pole point to anchor the very top of the head.
    boundary.append((_HEAD_C + np.array([0.0, 0.0, _HEAD_R])).tolist())

    # Greater trochanter surface.
    boundary.extend(_ellipsoid_surface_samples(_GT_C, _GT_AXES, n_phi=3, n_theta=6))

    # Lesser trochanter surface (smaller — fewer samples).
    boundary.extend(_ellipsoid_surface_samples(_LT_C, _LT_AXES, n_phi=3, n_theta=5, phi_inset=0.4))

    # Condyle surfaces.
    boundary.extend(_ellipsoid_surface_samples(_MC_C, _MC_AXES, n_phi=4, n_theta=6))
    boundary.extend(_ellipsoid_surface_samples(_LC_C, _LC_AXES, n_phi=4, n_theta=6))

    return np.asarray(boundary, dtype=np.float64)


def _femur_youngs_modulus(points: np.ndarray) -> np.ndarray:
    """Vectorised Young's modulus proxy (Pa): dense cortical, soft trabecular.

    A smooth radial ramp — stiffer near the cortical surface of the shaft,
    softer toward the marrow cavity — with small extra boosts inside the
    femoral head and condyles where cortical shell is thicker.
    """
    x, y = points[:, 0], points[:, 1]
    rr = np.sqrt(x * x + y * y)
    base = 1.2e9 * np.minimum(rr / 0.02, 1.0) + 2.0e8

    head_boost = np.where(np.linalg.norm(points - _HEAD_C, axis=1) <= _HEAD_R + 0.002, 3.0e8, 0.0)
    mc_boost = np.where(
        (((points - _MC_C) / (_MC_AXES + 0.002)) ** 2).sum(axis=1) <= 1.0, 3.0e8, 0.0
    )
    lc_boost = np.where(
        (((points - _LC_C) / (_LC_AXES + 0.002)) ** 2).sum(axis=1) <= 1.0, 3.0e8, 0.0
    )
    return np.asarray(base + head_boost + mc_boost + lc_boost, dtype=np.float64)


def generate_synthetic_femur(seed: int = 42) -> None:
    """Anatomically-plausible synthetic femur (CSG union of parts).

    Parts:
      - Shaft: truncated cylinder along z axis, z in [0.05, 0.35] m, radius 0.013.
      - Femoral head: sphere at proximal end, offset medially.
          center: (0.035, 0, 0.41), radius: 0.022
      - Femoral neck: cylinder connecting head center to shaft proximal end.
      - Greater trochanter: prolate ellipsoid lateral to neck-shaft junction.
          center: (-0.020, 0, 0.40), axes: (0.014, 0.012, 0.020)
      - Lesser trochanter: smaller ellipsoid medial side, slightly below head.
          center: (0.010, -0.008, 0.39), axes: (0.007, 0.006, 0.010)
      - Medial condyle (distal): ellipsoid at knee end.
          center: (0.015, 0, 0.03), axes: (0.018, 0.020, 0.015)
      - Lateral condyle (distal): ellipsoid at knee end.
          center: (-0.015, 0, 0.03), axes: (0.018, 0.020, 0.015)

    Orient so the femur runs along +z, with proximal (head) at high z and
    distal (condyles) at low z. All dimensions in meters, roughly
    adult-scale (~40-cm length, ~3-cm head).

    Implementation: rejection-sample inside the CSG union, add surface
    seeds on each primitive, Delaunay-tetrahedralise, then drop tets whose
    centroid falls outside the union so the mesh traces the anatomical
    shape instead of the bounding convex hull.
    """
    rng = np.random.default_rng(seed)

    # Bounding box enclosing every primitive, padded slightly.
    bbox_lo = np.array([-0.040, -0.025, 0.005])
    bbox_hi = np.array([0.060, 0.025, 0.435])
    bbox_diff = bbox_hi - bbox_lo

    # Rejection-sample ~250 interior points inside the CSG union.
    n_target_interior = 250
    interior: list[np.ndarray] = []
    while len(interior) < n_target_interior:
        batch = rng.random((2000, 3)) * bbox_diff + bbox_lo
        accepted = batch[_inside_union(batch)]
        interior.extend(accepted)
        if len(interior) >= n_target_interior:
            interior = interior[:n_target_interior]
            break
    interior_arr = np.asarray(interior, dtype=np.float64)

    # Surface seeds on each primitive anchor the Delaunay hull to the shape.
    boundary_arr = _femur_boundary_seeds()

    # Stack, then dedupe to 1 micron so Delaunay doesn't see coincident nodes.
    points = np.vstack([interior_arr, boundary_arr])
    points = np.unique(points.round(6), axis=0)

    tri = Delaunay(points)
    centroids = points[tri.simplices].mean(axis=1)
    tets = tri.simplices[_inside_union(centroids)]

    # Drop any nodes unused after tet filtering (keeps file size tight).
    used = np.unique(tets)
    remap = np.full(points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    points = points[used]
    tets = remap[tets]

    youngs_modulus = _femur_youngs_modulus(points).astype(np.float64)

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
        values=youngs_modulus,
    )

    doc = fml.Document.from_region(r)
    out = OUT / "femur.fieldml"
    doc.write(out)
    size = out.stat().st_size
    print(f"wrote {out} ({points.shape[0]} nodes, {tets.shape[0]} tets, {size:,} bytes)")


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

    # Fiber direction: a deterministic bipennate pattern converging on the
    # central z-axis (rectus femoris is bipennate around a central
    # aponeurosis). Fibers at the axis run along +z; fibers near the
    # surface angle inward by up to `alpha_max`, tapering to zero at the
    # tendons so the field is continuous at the tips.
    alpha_max = np.deg2rad(18.0)
    fibers = np.zeros_like(points)
    for i, (x, y, z) in enumerate(points):
        r = float(np.hypot(x, y))
        if r < 1e-9:
            fibers[i] = [0.0, 0.0, 1.0]
            continue
        r_local = float(envelope_radius(np.array([z]))[0]) or 1e-9
        rho = min(r / r_local, 1.0)
        taper = float(np.sin(np.pi * z / length))
        angle = alpha_max * rho * taper
        fibers[i] = [
            -np.sin(angle) * x / r,
            -np.sin(angle) * y / r,
            np.cos(angle),
        ]

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
