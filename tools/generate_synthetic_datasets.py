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


def generate_synthetic_femur(seed: int = 42) -> None:
    """Cylinder-like tet mesh with a BMD scalar field."""
    rng = np.random.default_rng(seed)
    radius = 0.02
    length = 0.4

    # Interior: uniform sample inside a cylinder.
    n_interior = 100
    u = rng.random(n_interior)
    v = rng.random(n_interior)
    w = rng.random(n_interior)
    rr = radius * np.sqrt(u)
    th = 2 * np.pi * v
    z = length * w
    x = rr * np.cos(th)
    y = rr * np.sin(th)
    interior = np.column_stack([x, y, z])

    # Cap + mid surface rings for a reasonable convex hull.
    cap_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    caps = np.concatenate(
        [
            np.column_stack(
                [
                    radius * np.cos(cap_angles),
                    radius * np.sin(cap_angles),
                    np.zeros_like(cap_angles),
                ]
            ),
            np.column_stack(
                [
                    radius * np.cos(cap_angles),
                    radius * np.sin(cap_angles),
                    np.full_like(cap_angles, length),
                ]
            ),
        ]
    )
    mid = np.column_stack(
        [
            radius * np.cos(cap_angles),
            radius * np.sin(cap_angles),
            np.full_like(cap_angles, length / 2),
        ]
    )
    points = np.vstack([interior, caps, mid])

    tri = Delaunay(points)
    tets = tri.simplices  # 0-indexed

    # BMD field per node: high near outer radius, low in interior.
    rr_nodes = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    bmd = 1.2e9 * (rr_nodes / radius) + 2.0e8  # Pa-equivalent Young's modulus

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
