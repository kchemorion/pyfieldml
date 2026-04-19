"""OpenSim-adjacent export — produce OpenSim-compatible assets from a Document.

Not a full .osim round-trip. Exports mesh geometry (.vtp), landmark sets
(CSV), and fiber vector fields as short line glyphs (.vtp polylines) for
hand-assembly into OpenSim models.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def export_geometry(doc: Any, out_dir: Any, *, filename: str = "geometry.vtp") -> Path:
    """Write the document's primary mesh as a VTK PolyData (.vtp) file.

    Requires pyfieldml[opensim] (which installs meshio).
    """
    import meshio

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    target = out_dir_path / filename
    # meshio's .vtp PolyData writer only supports surface cells, and meshio's
    # read-by-extension does not handle .vtp at all. The unit hex fixture is a
    # volume, so emit .vtu content regardless of the caller's extension and
    # adjust the path accordingly so downstream tools can dispatch on it.
    if target.suffix.lower() != ".vtu":
        target = target.with_suffix(".vtu")

    m = doc.to_meshio()
    meshio.write(target, m, file_format="vtu")
    return target


def export_muscle_attachments(doc: Any, out_path: Any, *, landmark_name: str | None = None) -> Path:
    """Write a landmark set as CSV: label, x, y, z.

    Looks up a ParameterEvaluator whose name contains "landmark" or
    "attachment" — or uses ``landmark_name`` if provided.
    """
    from pyfieldml.model.evaluators import ParameterEvaluator

    target_path = Path(out_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    region = doc.region
    target: tuple[str, np.ndarray] | None = None
    if landmark_name is not None:
        ev = region.evaluators.get(landmark_name)
        if not isinstance(ev, ParameterEvaluator):
            raise ValueError(f"No ParameterEvaluator named {landmark_name!r} in the region")
        target = (landmark_name, ev.as_ndarray())
    else:
        for name, ev in region.evaluators.items():
            if isinstance(ev, ParameterEvaluator) and (
                "landmark" in name.lower() or "attachment" in name.lower()
            ):
                target = (name, ev.as_ndarray())
                break
    if target is None:
        raise ValueError(
            "No landmark/attachment evaluator found in document. "
            "Pass landmark_name=... to pick one explicitly."
        )

    name, arr = target
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Landmark evaluator {name!r} has shape {arr.shape}; expected (N, 3).")

    with target_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "x", "y", "z"])
        for i, row in enumerate(arr):
            w.writerow([f"{name}_{i}", float(row[0]), float(row[1]), float(row[2])])
    return target_path


def export_fiber_lines(
    doc: Any,
    *,
    mesh: str | None = None,
    field: str,
    out_path: Any,
    length_scale: float = 0.01,
) -> Path:
    """Export a vector field as short line glyphs (.vtp polylines).

    For each mesh node, emit a line segment from the node position along the
    local fiber direction. Length = ``length_scale * bbox_diagonal``.
    """
    import meshio

    from pyfieldml.interop.meshio import _find_coord_evaluator
    from pyfieldml.model.evaluators import ParameterEvaluator

    target_path = Path(out_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # See note in ``export_geometry``: emit .vtu content regardless of the
    # caller's extension so meshio (which does not dispatch on .vtp) can read
    # it back cleanly.
    if target_path.suffix.lower() != ".vtu":
        target_path = target_path.with_suffix(".vtu")

    region = doc.region
    coord_name = _find_coord_evaluator(region)
    if coord_name is None:
        raise ValueError("No coordinate evaluator found in document.")
    coord_ev = region.evaluators[coord_name]
    if not isinstance(coord_ev, ParameterEvaluator):
        raise ValueError(f"{coord_name!r} is not a ParameterEvaluator")
    coords = coord_ev.as_ndarray().astype(np.float64)

    field_ev = region.evaluators.get(field)
    if not isinstance(field_ev, ParameterEvaluator):
        raise ValueError(f"Field evaluator {field!r} must be a per-node ParameterEvaluator.")
    vectors = field_ev.as_ndarray().astype(np.float64)
    if vectors.shape != coords.shape:
        raise ValueError(f"Field {field!r} has shape {vectors.shape}; expected {coords.shape}.")

    bbox_diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    step = length_scale * max(bbox_diag, 1.0)

    n = coords.shape[0]
    start = coords
    end = coords + vectors * step
    points = np.concatenate([start, end], axis=0)
    # Line cells: for each node i, segment from point i to point i+n.
    lines = np.column_stack([np.arange(n), np.arange(n, 2 * n)]).astype(np.int64)
    m = meshio.Mesh(points=points, cells=[("line", lines)])
    meshio.write(target_path, m, file_format="vtu")
    return target_path
