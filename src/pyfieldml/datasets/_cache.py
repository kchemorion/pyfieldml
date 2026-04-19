"""Download + hash-verify + cache dataset files on disk.

Bundled datasets resolve via ``importlib.resources``. Non-bundled datasets
are downloaded from the ``url_prefix`` on first access and cached under
``~/.cache/pyfieldml/datasets/<name>/``.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from importlib import resources
from pathlib import Path

from pyfieldml.datasets.registry import REGISTRY


def cache_dir() -> Path:
    """Root directory under which dataset files are cached."""
    base = Path(os.environ.get("PYFIELDML_CACHE") or Path.home() / ".cache" / "pyfieldml")
    base.mkdir(parents=True, exist_ok=True)
    return base


def dataset_dir(name: str) -> Path:
    """Per-dataset cache subdirectory."""
    d = cache_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_files(name: str) -> Path:
    """Materialize the dataset's primary .fieldml file and return its path.

    For bundled datasets, resolves via ``importlib.resources``. For non-bundled,
    downloads any missing files from ``url_prefix`` and verifies SHA256s.
    """
    info = REGISTRY[name]
    if info.bundled:
        ref = resources.files("pyfieldml.datasets._bundled").joinpath(f"{name}.fieldml")
        # Loose-file install: use the resource directly to avoid copying.
        direct = Path(str(ref))
        if direct.is_file():
            return direct
        # Zipped install: ``as_file`` returns a context-managed temp path that
        # would be cleaned up on context exit. Extract to a stable cache
        # location instead so the returned path survives.
        cached_dir = cache_dir() / "bundled"
        cached_dir.mkdir(parents=True, exist_ok=True)
        target = cached_dir / f"{name}.fieldml"
        if not target.is_file():
            with resources.as_file(ref) as concrete:
                target.write_bytes(Path(concrete).read_bytes())
        return target

    d = dataset_dir(name)
    for filename, expected_sha in info.files.items():
        target = d / filename
        if not target.is_file():
            if info.url_prefix is None:
                raise RuntimeError(f"Dataset {name!r} is not bundled and has no url_prefix")
            url = f"{info.url_prefix.rstrip('/')}/{filename}"
            urllib.request.urlretrieve(url, target)
        if expected_sha:
            _verify_sha256(target, expected_sha)

    # Primary .fieldml file - first alphabetical.
    fieldml_files = sorted(d.glob("*.fieldml"))
    if not fieldml_files:
        raise RuntimeError(f"Dataset {name!r}: no .fieldml file found in {d}")
    return fieldml_files[0]


def _verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        path.unlink()
        raise RuntimeError(f"SHA256 mismatch for {path.name}: expected {expected}, got {actual}")
