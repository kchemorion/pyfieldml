"""Accessors for bundled FieldML schema + standard library resources."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

_PACKAGE = "pyfieldml.library"
_XSD_FILENAME = "FieldML_0.5.xsd"
_STDLIB_FILENAME = "FieldML_Library_0.5.xml"


def get_xsd_path() -> Path:
    """Return the on-disk path to the bundled FieldML 0.5 XSD."""
    return _resource_path(_XSD_FILENAME)


def get_stdlib_path() -> Path:
    """Return the on-disk path to the bundled FieldML 0.5 standard library."""
    return _resource_path(_STDLIB_FILENAME)


def _resource_path(filename: str) -> Path:
    """Return a stable on-disk path to a bundled package resource.

    ``importlib.resources.as_file`` returns a context-managed path that is only
    guaranteed to exist while the context is open — for zipapp/zipimport
    installs the file is extracted to a temp location that gets cleaned up on
    ``__exit__``. Returning the raw path from inside the ``with`` leaves the
    caller with a stale path.

    Instead, materialize the resource into a stable cache directory on first
    access and return that path. The files are small (an XSD and an XML
    library), bundled once per install, and safe to re-extract when missing.
    """
    ref = resources.files(_PACKAGE).joinpath(filename)
    # Loose-file installs: ``ref`` is already a real on-disk path. Prefer it
    # directly to avoid a pointless cache copy.
    direct = Path(str(ref))
    if direct.is_file():
        return direct

    # Zipped installs: extract to the pyfieldml cache. Imported lazily to
    # avoid pulling the datasets module at package-import time.
    from pyfieldml.datasets._cache import cache_dir

    cached_dir = cache_dir() / "bundled"
    cached_dir.mkdir(parents=True, exist_ok=True)
    target = cached_dir / filename
    if not target.is_file():
        with resources.as_file(ref) as concrete:
            target.write_bytes(Path(concrete).read_bytes())
    return target
