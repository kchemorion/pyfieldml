#!/usr/bin/env python
"""Fetch and convert real-mesh datasets for pyfieldml.

Run from repo root:
    uv run python tools/fetch_real_datasets.py

This fetches public-domain and CC-BY-SA anatomical meshes, converts them
to FieldML format via meshio, and writes them to
src/pyfieldml/datasets/_bundled/. Network required.
"""

from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

import pyfieldml as fml

OUT = Path(__file__).parents[1] / "src/pyfieldml/datasets/_bundled"
OUT.mkdir(parents=True, exist_ok=True)

BUNNY_URL = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
# Ordered from smallest to largest - try smallest first to keep the bundle tight.
BUNNY_CANDIDATES = (
    "bunny/reconstruction/bun_zipper_res4.ply",  # ~948 triangles
    "bunny/reconstruction/bun_zipper_res3.ply",  # ~3k triangles
)

MAX_BUNDLE_BYTES = 150_000


def _download(url: str, *, timeout: float = 60.0) -> bytes:
    """Download ``url`` and return its bytes. Raises on HTTP errors."""
    print(f"  GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pyfieldml-fetch/1.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return bytes(resp.read())


def fetch_stanford_bunny() -> Path | None:
    """Fetch the Stanford Bunny, convert to FieldML, write to _bundled/.

    Returns the output path on success, or ``None`` on failure.
    """
    print("Stanford Bunny")
    print("  source: https://graphics.stanford.edu/data/3Dscanrep/")
    try:
        blob = _download(BUNNY_URL)
    except Exception as exc:
        print(f"  FAILED: could not download bunny.tar.gz: {exc}", file=sys.stderr)
        return None

    # Extract the smallest candidate that exists.
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        members = {m.name: m for m in tar.getmembers()}
        chosen = None
        for name in BUNNY_CANDIDATES:
            if name in members:
                chosen = name
                break
        if chosen is None:
            print(
                f"  FAILED: none of {BUNNY_CANDIDATES!r} present in the archive",
                file=sys.stderr,
            )
            return None
        print(f"  using {chosen}")
        f = tar.extractfile(members[chosen])
        if f is None:
            print(f"  FAILED: could not extract {chosen}", file=sys.stderr)
            return None
        ply_bytes = f.read()

    # meshio reads PLY directly.
    import meshio

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as tmp:
        tmp.write(ply_bytes)
        tmp.flush()
        mesh = meshio.read(tmp.name)

    # Keep only the first triangle cell block (the reconstructed surface).
    tri_blocks = [b for b in mesh.cells if b.type == "triangle"]
    if not tri_blocks:
        print("  FAILED: no triangle cells in bunny mesh", file=sys.stderr)
        return None
    tri = tri_blocks[0]
    mesh_tri = meshio.Mesh(
        points=np.asarray(mesh.points, dtype=np.float64),
        cells=[("triangle", np.asarray(tri.data, dtype=np.int64))],
    )

    doc = fml.Document.from_meshio(mesh_tri, name="bunny_stanford")
    out = OUT / "bunny_stanford.fieldml"
    doc.write(out)
    size = out.stat().st_size
    n_points = mesh_tri.points.shape[0]
    n_tri = mesh_tri.cells[0].data.shape[0]
    print(f"  wrote {out} ({n_points} points, {n_tri} triangles, {size:,} bytes)")
    if size > MAX_BUNDLE_BYTES:
        print(
            f"  WARNING: file exceeds {MAX_BUNDLE_BYTES:,}-byte budget; "
            f"consider falling back to a coarser resolution.",
            file=sys.stderr,
        )
    return out


# ---------------- BodyParts3D femur ----------------

# Per-FMA OBJ endpoints on lifesciencedb.jp are defunct as of 2024-2025;
# the DBCLS archive at biosciencedbc.jp only publishes bulk zips. We fetch
# the smaller part-of archive (~62 MB, ~1258 entries) whose members are
# named by BodyParts3D element-file ids (``FJ####.obj``). The separate
# parts-index text file maps FMA concept ids to those element ids.
BP3D_ARCHIVE_URL = (
    "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/partof_BP3D_4.0_obj_99.zip"
)
BP3D_ARCHIVE_URL_FALLBACK = (
    "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/isa_BP3D_4.0_obj_99.zip"
)
# FMA24475 == "left femur" (FMA Ontology). The part-of index maps it to
# element file FJ3259; right femur (FMA24474) would be FJ3365. We hard-code
# the element id so the tool doesn't need a second round-trip for the index.
BP3D_FEMUR_ELEMENT = "FJ3259"
BP3D_FEMUR_LABEL = "left femur (FMA24475)"


def _decimate_triangles(
    points: np.ndarray, triangles: np.ndarray, target: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return a ``target``-triangle random subsample plus renumbered nodes.

    Random subsampling (no edge-collapse) is coarse but good enough for a
    zoo asset - we just want a sub-150-KB FieldML surface mesh that
    recognisably shows a femur.
    """
    if triangles.shape[0] <= target:
        return points, triangles
    rng = np.random.default_rng(seed)
    idx = rng.choice(triangles.shape[0], size=target, replace=False)
    tri_sub = triangles[np.sort(idx)]
    used = np.unique(tri_sub)
    remap = np.full(points.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    return points[used], remap[tri_sub]


def _extract_bp3d_femur_obj(zip_blob: bytes) -> tuple[str, bytes] | None:
    """Return ``(member_name, obj_bytes)`` for the femur OBJ inside the BP3D archive.

    Members are named ``<tree>/<FJid>.obj``. We look up our hard-coded
    element id (derived from the DBCLS ``element_parts`` index) and
    return the first matching member.
    """
    import zipfile

    target = f"{BP3D_FEMUR_ELEMENT}.obj"
    with zipfile.ZipFile(io.BytesIO(zip_blob)) as zf:
        for name in zf.namelist():
            if name.endswith(target):
                return name, zf.read(name)
    return None


def fetch_bodyparts3d_femur() -> Path | None:
    """Fetch the BodyParts3D femur, decimate, convert to FieldML.

    Returns the output path on success, or ``None`` on failure. Never
    raises - the caller treats failure as a skippable non-fatal event.
    """
    print("BodyParts3D femur")
    print("  source: https://dbarchive.biosciencedbc.jp/en/bodyparts3d/")
    import meshio

    zip_blob: bytes | None = None
    for url in (BP3D_ARCHIVE_URL, BP3D_ARCHIVE_URL_FALLBACK):
        try:
            print(f"  fetching {url} (may be 60+ MB)")
            zip_blob = _download(url, timeout=300.0)
            print(f"  downloaded {len(zip_blob):,} bytes")
            break
        except Exception as exc:
            print(f"  FAILED: {url} -> {exc}", file=sys.stderr)
            zip_blob = None

    if zip_blob is None:
        print("  SKIPPED: BodyParts3D archives unreachable", file=sys.stderr)
        return None

    try:
        extracted = _extract_bp3d_femur_obj(zip_blob)
    except Exception as exc:
        print(f"  SKIPPED: archive extraction failed: {exc}", file=sys.stderr)
        return None
    if extracted is None:
        print("  SKIPPED: no femur OBJ found in archive", file=sys.stderr)
        return None
    member_name, obj_bytes = extracted
    print(f"  using {member_name} = {BP3D_FEMUR_LABEL} ({len(obj_bytes):,} bytes OBJ)")

    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
        tmp.write(obj_bytes)
        tmp_path = Path(tmp.name)
    try:
        try:
            mesh = meshio.read(str(tmp_path))
        except Exception as exc:
            print(f"  SKIPPED: meshio could not parse OBJ: {exc}", file=sys.stderr)
            return None
    finally:
        tmp_path.unlink(missing_ok=True)

    tri_blocks = [b for b in mesh.cells if b.type == "triangle"]
    if not tri_blocks:
        print("  SKIPPED: no triangle cells in BodyParts3D mesh", file=sys.stderr)
        return None
    pts = np.asarray(mesh.points, dtype=np.float64)
    tris = np.asarray(tri_blocks[0].data, dtype=np.int64)

    # Decimate to keep file < 150 KB. Empirically, ~1500 triangles with
    # InlineTextBackend fits comfortably; we pick 1500 to be safe.
    target_tris = 1500
    pts_d, tris_d = _decimate_triangles(pts, tris, target=target_tris, seed=0)

    mesh_tri = meshio.Mesh(points=pts_d, cells=[("triangle", tris_d)])
    doc = fml.Document.from_meshio(mesh_tri, name="femur_bodyparts3d")
    out = OUT / "femur_bodyparts3d.fieldml"
    doc.write(out)
    size = out.stat().st_size
    print(f"  wrote {out} ({pts_d.shape[0]} points, {tris_d.shape[0]} triangles, {size:,} bytes)")
    if size > MAX_BUNDLE_BYTES:
        print(
            f"  WARNING: file exceeds {MAX_BUNDLE_BYTES:,}-byte budget; "
            "retrying with a coarser decimation.",
            file=sys.stderr,
        )
        # Try again with half the triangles.
        pts_d, tris_d = _decimate_triangles(pts, tris, target=target_tris // 2, seed=0)
        mesh_tri = meshio.Mesh(points=pts_d, cells=[("triangle", tris_d)])
        doc = fml.Document.from_meshio(mesh_tri, name="femur_bodyparts3d")
        doc.write(out)
        size = out.stat().st_size
        print(
            f"  rewrote {out} ({pts_d.shape[0]} points, {tris_d.shape[0]} "
            f"triangles, {size:,} bytes)"
        )
    return out


def main() -> int:
    print("=== pyfieldml real-mesh fetcher ===\n")
    bunny = fetch_stanford_bunny()
    print()
    bp3d = fetch_bodyparts3d_femur()
    print()
    print("Summary:")
    print(f"  bunny_stanford:     {'ok' if bunny else 'FAILED'}")
    print(f"  femur_bodyparts3d:  {'ok' if bp3d else 'skipped/failed'}")
    # Fatal only if bunny (public-domain anchor) failed: BP3D is best-effort.
    return 0 if bunny is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
