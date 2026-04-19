"""Model zoo - curated FieldML assets for tutorials and research.

Public API:
    datasets.list()                      -> list of available dataset names
    datasets.info(name)                  -> metadata dict for a dataset
    datasets.download(name)              -> explicit pre-fetch
    datasets.cache_dir()                 -> path to the on-disk cache root
    datasets.load(name)                  -> pyfieldml.Document
    datasets.load_unit_cube()            -> shortcut for the bundled unit_cube
    datasets.load_femur()                -> anatomical-synthetic femur
    datasets.load_rectus_femoris()       -> synthetic spindle muscle
    datasets.load_bunny_stanford()       -> Stanford Bunny (public domain)
    datasets.load_femur_bodyparts3d()    -> BodyParts3D femur (CC-BY-SA 2.1 JP)
    datasets.load_vertebra_l3()          -> BodyParts3D lumbar vertebra
    datasets.load_scapula()              -> BodyParts3D left scapula
    datasets.load_tibia_left()           -> BodyParts3D left tibia
    datasets.load_hip_bone_left()        -> BodyParts3D hip bone
    datasets.load_skull()                -> BodyParts3D compound skull
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


def load_femur() -> Document:
    """Return the bundled anatomical-synthetic ``femur`` Document.

    Produced by a CSG envelope (shaft + femoral head + neck + trochanters +
    distal condyles) with a BMD-proxy Young's modulus field.
    """
    return load("femur")


def load_rectus_femoris() -> Document:
    """Return the bundled synthetic ``rectus_femoris`` Document (spindle + fiber field)."""
    return load("rectus_femoris")


def load_bunny_stanford() -> Document:
    """Return the bundled Stanford Bunny Document (public domain triangle mesh)."""
    return load("bunny_stanford")


def load_femur_bodyparts3d() -> Document:
    """Return the bundled BodyParts3D femur Document (CC-BY-SA 2.1 JP triangle mesh)."""
    return load("femur_bodyparts3d")


def load_vertebra_l3() -> Document:
    """Return the bundled BodyParts3D lumbar-vertebra Document (CC-BY-SA 2.1 JP)."""
    return load("vertebra_l3")


def load_scapula() -> Document:
    """Return the bundled BodyParts3D left-scapula Document (CC-BY-SA 2.1 JP)."""
    return load("scapula")


def load_tibia_left() -> Document:
    """Return the bundled BodyParts3D left-tibia Document (CC-BY-SA 2.1 JP)."""
    return load("tibia_left")


def load_hip_bone_left() -> Document:
    """Return the bundled BodyParts3D left hip-bone Document (CC-BY-SA 2.1 JP)."""
    return load("hip_bone_left")


def load_skull() -> Document:
    """Return the bundled BodyParts3D compound-skull Document (CC-BY-SA 2.1 JP).

    Assembled from 43 cranial and facial sub-part meshes (FMA46565) unioned
    into a single vertex array, then decimated. The union is non-manifold;
    callers that need a watertight mesh should use one of the single-bone
    loaders (``load_femur_bodyparts3d``, ``load_tibia_left``, etc.) instead.
    """
    return load("skull")


__all__ = [
    "cache_dir",
    "dataset_dir",
    "download",
    "info",
    "list",
    "load",
    "load_bunny_stanford",
    "load_femur",
    "load_femur_bodyparts3d",
    "load_hip_bone_left",
    "load_rectus_femoris",
    "load_scapula",
    "load_skull",
    "load_tibia_left",
    "load_unit_cube",
    "load_vertebra_l3",
]
