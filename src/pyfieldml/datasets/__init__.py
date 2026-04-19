"""Model zoo - curated FieldML assets for tutorials and research.

Public API:
    datasets.list()                  -> list of available dataset names
    datasets.info(name)              -> metadata dict for a dataset
    datasets.download(name)          -> explicit pre-fetch
    datasets.cache_dir()             -> path to the on-disk cache root
    datasets.load(name)              -> pyfieldml.Document
    datasets.load_unit_cube()        -> shortcut for the bundled unit_cube
    datasets.load_femur()            -> (added in Task 3)
    datasets.load_rectus_femoris()   -> (added in Task 3)
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

from pyfieldml.datasets._cache import cache_dir, dataset_dir, ensure_files
from pyfieldml.datasets.registry import REGISTRY

if TYPE_CHECKING:
    from pyfieldml.document import Document


def list() -> builtins.list[str]:
    """Return the sorted list of registered dataset names."""
    return sorted(REGISTRY.keys())


def info(name: str) -> dict[str, str]:
    """Return metadata (name, license, citation, origin) for ``name``."""
    d = REGISTRY[name]
    return {
        "name": d.name,
        "license": d.license,
        "citation": d.citation,
        "origin": d.origin,
    }


def download(name: str) -> None:
    """Pre-fetch ``name`` into the cache without loading it."""
    ensure_files(name)


def load(name: str) -> Document:
    """Load ``name`` as a pyfieldml.Document."""
    import pyfieldml as _fml  # local import to avoid cycles

    path = ensure_files(name)
    return _fml.read(path)


def load_unit_cube() -> Document:
    """Return the bundled ``unit_cube`` Document (shortcut for ``load('unit_cube')``)."""
    return load("unit_cube")


__all__ = [
    "cache_dir",
    "dataset_dir",
    "download",
    "info",
    "list",
    "load",
    "load_unit_cube",
]
