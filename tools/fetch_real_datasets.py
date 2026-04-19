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


def main() -> int:
    print("=== pyfieldml real-mesh fetcher ===\n")
    bunny = fetch_stanford_bunny()
    print()
    print("Summary:")
    print(f"  bunny_stanford:     {'ok' if bunny else 'FAILED'}")
    return 0 if bunny is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
