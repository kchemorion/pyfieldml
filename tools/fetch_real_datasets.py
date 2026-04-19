#!/usr/bin/env python
"""Fetch and convert real-mesh datasets for pyfieldml.

Run from repo root:
    uv run python tools/fetch_real_datasets.py

This fetches public-domain and CC-BY-SA anatomical meshes, converts them
to FieldML format via meshio, and writes them to
src/pyfieldml/datasets/_bundled/. Network required (unless the BodyParts3D
archive is pre-cached; see ``BP3D_ZIP_CACHE`` below).

Env vars
--------
BP3D_ZIP_CACHE
    Path to a pre-downloaded ``partof_BP3D_4.0_obj_99.zip``. When set, the
    tool reads the zip from disk instead of fetching ~62 MB over the
    network. Lets developers iterate on the decimation / conversion step
    without hitting DBCLS repeatedly.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
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
# Budget for the BodyParts3D femur: larger than the bunny because
# watertight edge-collapse decimation needs more triangles than random-drop.
MAX_BP3D_BYTES = 200_000
# v1.2+ additions (vertebra, scapula, tibia, hip, skull). Slightly larger
# than the femur budget because some of these shapes need more triangles
# to stay watertight (scapula shell, multi-part skull).
MAX_BP3D_EXT_BYTES = 300_000


def _download(url: str, *, timeout: float = 60.0) -> bytes:
    """Download ``url`` and return its bytes. Raises on HTTP errors."""
    print(f"  GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pyfieldml-fetch/1.2"})
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


# ---------------- BodyParts3D ----------------

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

# Hard-coded BP3D element ids for the v1.2+ bundle. See partof_parts_list_e.txt
# in the DBCLS distribution for the full FMA -> element-file-id mapping.
BP3D_FEMUR_ELEMENT = "FJ3259"  # FMA24475 left femur
BP3D_FEMUR_LABEL = "left femur (FMA24475)"

BP3D_VERTEBRA_L3_ELEMENT = "FJ3159"  # FMA9921 lumbar vertebra (L-series member)
BP3D_VERTEBRA_L3_LABEL = "lumbar vertebra (BodyParts3D FJ3159)"

BP3D_SCAPULA_ELEMENT = "FJ3279"  # FMA13396 left scapula
BP3D_SCAPULA_LABEL = "left scapula (FMA13396)"

BP3D_TIBIA_LEFT_ELEMENT = "FJ3282"  # FMA24478 left tibia
BP3D_TIBIA_LEFT_LABEL = "left tibia (FMA24478)"

BP3D_HIP_BONE_LEFT_ELEMENT = "FJ3152"  # FMA16586 hip bone
BP3D_HIP_BONE_LEFT_LABEL = "hip bone (FMA16586, BodyParts3D FJ3152)"

# FMA46565 "skull" is a compound of ~43 cranial / facial sub-parts in the
# partof index. Hard-coded here so a fetch does not need a second network
# round-trip to re-derive the list.
BP3D_SKULL_ELEMENTS: tuple[str, ...] = (
    "FJ1282",
    "FJ1285",
    "FJ1286",
    "FJ1289",
    "FJ1297",
    "FJ1299",
    "FJ1305",
    "FJ1317",
    "FJ1320",
    "FJ1331",
    "FJ1336",
    "FJ1337",
    "FJ1340",
    "FJ1348",
    "FJ1350",
    "FJ1356",
    "FJ1368",
    "FJ1371",
    "FJ1382",
    "FJ2772",
    "FJ3199",
    "FJ3200",
    "FJ3201",
    "FJ3263",
    "FJ3265",
    "FJ3269",
    "FJ3272",
    "FJ3273",
    "FJ3274",
    "FJ3281",
    "FJ3287",
    "FJ3289",
    "FJ3309",
    "FJ3369",
    "FJ3371",
    "FJ3375",
    "FJ3378",
    "FJ3379",
    "FJ3380",
    "FJ3386",
    "FJ3392",
    "FJ3394",
    "FJ3395",
)
BP3D_SKULL_LABEL = "skull (FMA46565, BodyParts3D compound of 43 sub-parts)"


def _decimate_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    target: int,
    *,
    multi_component: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a topology-preserving edge-collapse decimation to ~``target`` triangles.

    Uses pyvista / VTK quadric decimation, which collapses edges rather
    than dropping random triangles, so the output stays watertight. If the
    mesh is already at or below ``target``, returns the input unchanged.
    Falls back silently to the input mesh if pyvista can't be imported
    (the caller still gets a full, non-swiss-cheese mesh, just a little
    bigger).

    Parameters
    ----------
    points
        ``(N, 3)`` vertex array.
    triangles
        ``(M, 3)`` integer connectivity array into ``points``.
    target
        Desired triangle count after decimation. Acts as a lower bound
        the algorithms may not exactly hit — the output typically
        bounces within ~20%% of this value.
    multi_component
        When ``True`` (for compound meshes like the skull) we use VTK's
        ``decimate_pro`` with boundary-vertex deletion and a permissive
        feature angle, which handles non-manifold / multi-component input
        that ``decimate`` refuses. The tradeoff is that ``decimate_pro``
        is less aggressive at preserving local curvature; for monolithic
        anatomical meshes ``decimate`` (the default) gives cleaner
        results.

    """
    if triangles.shape[0] <= target:
        return points, triangles
    try:
        import pyvista as pv
    except ImportError:
        # Without pyvista we can't do edge-collapse decimation. Random
        # subsampling destroys watertightness (see v1.1 regression), so
        # prefer the full mesh to a holey one.
        return points, triangles

    # Build a PolyData: flat [3, i, j, k, 3, ...] cells array.
    faces = np.hstack([np.full((triangles.shape[0], 1), 3, dtype=np.int64), triangles]).flatten()
    poly = pv.PolyData(points, faces)
    target_reduction = 1.0 - (target / triangles.shape[0])
    if multi_component:
        # ``decimate`` raises on non-manifold input. ``decimate_pro`` with
        # boundary_vertex_deletion + feature_angle=180 tolerates the
        # many-disconnected-parts skull mesh.
        dec = poly.decimate_pro(
            target_reduction,
            preserve_topology=False,
            splitting=False,
            boundary_vertex_deletion=True,
            feature_angle=180.0,
        )
    else:
        dec = poly.decimate(target_reduction, volume_preservation=True)

    pts_out = np.asarray(dec.points, dtype=np.float64)
    faces_flat = np.asarray(dec.faces, dtype=np.int64)
    tris_out = faces_flat.reshape(-1, 4)[:, 1:]
    return pts_out, tris_out


def _count_components(points: np.ndarray, triangles: np.ndarray) -> int:
    """Return the number of connected components of the triangle mesh's vertex graph.

    Tries scipy; falls back to a pure-numpy union-find if scipy is absent.
    """
    n = int(points.shape[0])
    if triangles.size == 0:
        return 0
    try:
        import scipy.sparse
        import scipy.sparse.csgraph

        row = np.concatenate(
            [
                triangles[:, 0],
                triangles[:, 1],
                triangles[:, 0],
                triangles[:, 1],
                triangles[:, 2],
                triangles[:, 2],
            ]
        )
        col = np.concatenate(
            [
                triangles[:, 1],
                triangles[:, 0],
                triangles[:, 2],
                triangles[:, 2],
                triangles[:, 0],
                triangles[:, 1],
            ]
        )
        data = np.ones(row.shape[0], dtype=np.float64)
        adj = scipy.sparse.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
        n_components, _ = scipy.sparse.csgraph.connected_components(adj)
        return int(n_components)
    except ImportError:
        parent = np.arange(n, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        for tri in triangles.astype(np.int64):
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for i, j in ((a, b), (b, c), (a, c)):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
        return len({find(i) for i in range(n)})


def _load_bp3d_zip() -> bytes | None:
    """Return the bytes of the BodyParts3D part-of zip.

    Reads from the ``BP3D_ZIP_CACHE`` env var if set, otherwise downloads
    from DBCLS (with a fallback URL). Returns ``None`` if every source
    fails, so the caller can skip gracefully.
    """
    cache_env = os.environ.get("BP3D_ZIP_CACHE")
    if cache_env:
        cache_path = Path(cache_env)
        if cache_path.is_file():
            print(f"  using cached archive {cache_path}")
            return cache_path.read_bytes()
        print(
            f"  WARNING: BP3D_ZIP_CACHE={cache_env} does not exist; falling back to network",
            file=sys.stderr,
        )

    for url in (BP3D_ARCHIVE_URL, BP3D_ARCHIVE_URL_FALLBACK):
        try:
            print(f"  fetching {url} (may be 60+ MB)")
            blob = _download(url, timeout=300.0)
            print(f"  downloaded {len(blob):,} bytes")
            return blob
        except Exception as exc:
            print(f"  FAILED: {url} -> {exc}", file=sys.stderr)
    return None


def _extract_obj_from_bp3d_zip(zip_path_or_blob: Path | bytes, fj_id: str) -> bytes | None:
    """Return the OBJ bytes for ``fj_id`` inside the BP3D archive, or ``None``.

    Members are named ``<tree>/<FJid>.obj``; we match suffix so the helper
    is independent of how the archive happens to be nested.
    """
    target = f"{fj_id}.obj"
    if isinstance(zip_path_or_blob, Path):
        zf_ctx = zipfile.ZipFile(zip_path_or_blob)
    else:
        zf_ctx = zipfile.ZipFile(io.BytesIO(zip_path_or_blob))
    with zf_ctx as zf:
        for name in zf.namelist():
            if name.endswith(f"/{target}") or name.endswith(target):
                return zf.read(name)
    return None


def _read_obj_bytes(obj_bytes: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse OBJ ``obj_bytes`` with meshio and return ``(points, triangles)``.

    Returns ``None`` if meshio can't parse it or the file contains no
    triangles.
    """
    import meshio

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
        return None
    pts = np.asarray(mesh.points, dtype=np.float64)
    tris = np.asarray(tri_blocks[0].data, dtype=np.int64)
    return pts, tris


def _write_bp3d_dataset(
    pts: np.ndarray,
    tris: np.ndarray,
    *,
    out_name: str,
    target_tris: int,
    max_bytes: int,
    label: str,
    multi_component: bool = False,
) -> Path:
    """Decimate to ``target_tris``, write ``out_name.fieldml``, retry-halve if oversized.

    Emits a progress log line with point / triangle / byte count and the
    number of connected components so a reviewer can verify the mesh is
    one piece (or see exactly how many pieces it decomposed into).
    Returns the output path.
    """
    import meshio

    print(f"  input: {pts.shape[0]} points, {tris.shape[0]} triangles")
    pts_d, tris_d = _decimate_triangles(
        pts, tris, target=target_tris, multi_component=multi_component
    )
    mesh_tri = meshio.Mesh(points=pts_d, cells=[("triangle", tris_d)])
    doc = fml.Document.from_meshio(mesh_tri, name=out_name)
    out = OUT / f"{out_name}.fieldml"
    doc.write(out)
    size = out.stat().st_size
    n_comp = _count_components(pts_d, tris_d)
    print(
        f"  wrote {out} ({pts_d.shape[0]} points, {tris_d.shape[0]} "
        f"triangles, {size:,} bytes, {n_comp} connected component(s)) — {label}"
    )

    if size > max_bytes:
        print(
            f"  WARNING: file exceeds {max_bytes:,}-byte budget; "
            "retrying with a coarser decimation.",
            file=sys.stderr,
        )
        pts_d, tris_d = _decimate_triangles(
            pts, tris, target=max(target_tris // 2, 500), multi_component=multi_component
        )
        mesh_tri = meshio.Mesh(points=pts_d, cells=[("triangle", tris_d)])
        doc = fml.Document.from_meshio(mesh_tri, name=out_name)
        doc.write(out)
        size = out.stat().st_size
        n_comp = _count_components(pts_d, tris_d)
        print(
            f"  rewrote {out} ({pts_d.shape[0]} points, {tris_d.shape[0]} "
            f"triangles, {size:,} bytes, {n_comp} connected component(s))"
        )
    return out


def fetch_bp3d_single(
    fj_id: str, out_name: str, label: str, *, target_tris: int = 2500
) -> Path | None:
    """Fetch a single-FJ BodyParts3D mesh, decimate, convert to FieldML.

    Returns the output path on success, or ``None`` on failure. Never
    raises - the caller treats failure as a skippable non-fatal event.
    """
    print(f"BodyParts3D {out_name}")
    print("  source: https://dbarchive.biosciencedbc.jp/en/bodyparts3d/")
    zip_blob = _load_bp3d_zip()
    if zip_blob is None:
        print("  SKIPPED: BodyParts3D archives unreachable", file=sys.stderr)
        return None

    try:
        obj_bytes = _extract_obj_from_bp3d_zip(zip_blob, fj_id)
    except Exception as exc:
        print(f"  SKIPPED: archive extraction failed: {exc}", file=sys.stderr)
        return None
    if obj_bytes is None:
        print(f"  SKIPPED: no {fj_id}.obj in archive", file=sys.stderr)
        return None
    print(f"  using {fj_id}.obj = {label} ({len(obj_bytes):,} bytes OBJ)")

    parsed = _read_obj_bytes(obj_bytes)
    if parsed is None:
        print("  SKIPPED: no triangles in OBJ", file=sys.stderr)
        return None
    pts, tris = parsed

    # Preserve the existing femur file byte-for-byte: it was built against
    # the legacy MAX_BP3D_BYTES (200 KB) budget, and the reproducibility
    # fingerprint is pinned to that output. New datasets use the looser
    # MAX_BP3D_EXT_BYTES (300 KB) budget.
    max_bytes = MAX_BP3D_BYTES if out_name == "femur_bodyparts3d" else MAX_BP3D_EXT_BYTES
    return _write_bp3d_dataset(
        pts, tris, out_name=out_name, target_tris=target_tris, max_bytes=max_bytes, label=label
    )


def fetch_bp3d_compound(
    fj_ids: list[str] | tuple[str, ...],
    out_name: str,
    label: str,
    *,
    target_tris: int = 3000,
) -> Path | None:
    """Fetch many BP3D sub-parts, union into one mesh, decimate, convert.

    Used for the skull (FMA46565) which is an aggregate of ~43 cranial
    and facial bones. Points are concatenated and triangle indices are
    offset per-part so the merged mesh has a single vertex array.
    """
    print(f"BodyParts3D {out_name} (compound, {len(fj_ids)} parts)")
    print("  source: https://dbarchive.biosciencedbc.jp/en/bodyparts3d/")
    zip_blob = _load_bp3d_zip()
    if zip_blob is None:
        print("  SKIPPED: BodyParts3D archives unreachable", file=sys.stderr)
        return None

    all_pts: list[np.ndarray] = []
    all_tris: list[np.ndarray] = []
    offset = 0
    missing: list[str] = []
    for fj_id in fj_ids:
        try:
            obj_bytes = _extract_obj_from_bp3d_zip(zip_blob, fj_id)
        except Exception as exc:
            print(f"  WARN: extraction failed for {fj_id}: {exc}", file=sys.stderr)
            missing.append(fj_id)
            continue
        if obj_bytes is None:
            missing.append(fj_id)
            continue
        parsed = _read_obj_bytes(obj_bytes)
        if parsed is None:
            missing.append(fj_id)
            continue
        pts, tris = parsed
        all_pts.append(pts)
        all_tris.append(tris + offset)
        offset += pts.shape[0]

    if not all_pts:
        print("  SKIPPED: no sub-parts extracted", file=sys.stderr)
        return None
    if missing:
        tail = "..." if len(missing) > 5 else ""
        print(f"  note: {len(missing)} sub-part(s) missing: {missing[:5]}{tail}")

    pts_all = np.concatenate(all_pts, axis=0)
    tris_all = np.concatenate(all_tris, axis=0).astype(np.int64)
    print(
        f"  merged: {pts_all.shape[0]} points, {tris_all.shape[0]} triangles "
        f"from {len(all_pts)} parts"
    )

    return _write_bp3d_dataset(
        pts_all,
        tris_all,
        out_name=out_name,
        target_tris=target_tris,
        max_bytes=MAX_BP3D_EXT_BYTES,
        label=label,
        multi_component=True,
    )


def fetch_bodyparts3d_femur() -> Path | None:
    """Fetch the BodyParts3D femur (v1.1-compatible wrapper for backward use)."""
    return fetch_bp3d_single(
        BP3D_FEMUR_ELEMENT,
        "femur_bodyparts3d",
        BP3D_FEMUR_LABEL,
        target_tris=2500,
    )


def fetch_bp3d_vertebra_l3() -> Path | None:
    """Fetch the BodyParts3D lumbar-vertebra mesh (FJ3159)."""
    return fetch_bp3d_single(
        BP3D_VERTEBRA_L3_ELEMENT,
        "vertebra_l3",
        BP3D_VERTEBRA_L3_LABEL,
        target_tris=2500,
    )


def fetch_bp3d_scapula() -> Path | None:
    """Fetch the BodyParts3D left-scapula mesh (FJ3279)."""
    return fetch_bp3d_single(
        BP3D_SCAPULA_ELEMENT,
        "scapula",
        BP3D_SCAPULA_LABEL,
        target_tris=3000,
    )


def fetch_bp3d_tibia_left() -> Path | None:
    """Fetch the BodyParts3D left-tibia mesh (FJ3282)."""
    return fetch_bp3d_single(
        BP3D_TIBIA_LEFT_ELEMENT,
        "tibia_left",
        BP3D_TIBIA_LEFT_LABEL,
        target_tris=2500,
    )


def fetch_bp3d_hip_bone_left() -> Path | None:
    """Fetch the BodyParts3D left hip-bone mesh (FJ3152)."""
    return fetch_bp3d_single(
        BP3D_HIP_BONE_LEFT_ELEMENT,
        "hip_bone_left",
        BP3D_HIP_BONE_LEFT_LABEL,
        target_tris=2500,
    )


def fetch_bp3d_skull() -> Path | None:
    """Fetch + merge the BodyParts3D skull compound (FMA46565, 43 sub-parts)."""
    return fetch_bp3d_compound(
        BP3D_SKULL_ELEMENTS,
        "skull",
        BP3D_SKULL_LABEL,
        target_tris=3000,
    )


def main() -> int:
    print("=== pyfieldml real-mesh fetcher ===\n")
    bunny = fetch_stanford_bunny()
    print()
    bp3d_femur = fetch_bodyparts3d_femur()
    print()
    bp3d_vert = fetch_bp3d_vertebra_l3()
    print()
    bp3d_scap = fetch_bp3d_scapula()
    print()
    bp3d_tib = fetch_bp3d_tibia_left()
    print()
    bp3d_hip = fetch_bp3d_hip_bone_left()
    print()
    bp3d_skull = fetch_bp3d_skull()
    print()
    print("Summary:")
    print(f"  bunny_stanford:     {'ok' if bunny else 'FAILED'}")
    print(f"  femur_bodyparts3d:  {'ok' if bp3d_femur else 'skipped/failed'}")
    print(f"  vertebra_l3:        {'ok' if bp3d_vert else 'skipped/failed'}")
    print(f"  scapula:            {'ok' if bp3d_scap else 'skipped/failed'}")
    print(f"  tibia_left:         {'ok' if bp3d_tib else 'skipped/failed'}")
    print(f"  hip_bone_left:      {'ok' if bp3d_hip else 'skipped/failed'}")
    print(f"  skull:              {'ok' if bp3d_skull else 'skipped/failed'}")
    # Fatal only if bunny (public-domain anchor) failed: BP3D is best-effort.
    return 0 if bunny is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
