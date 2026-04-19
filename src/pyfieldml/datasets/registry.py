"""Dataset metadata registry. Extend this as new datasets land."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetInfo:
    """Metadata for one curated dataset.

    Attributes
    ----------
    name
        Unique lookup key (also the subdir under the cache directory).
    files
        Mapping of filename -> SHA256 hex digest (empty string permitted for
        bundled datasets whose hash is not required).
    url_prefix
        GitHub release asset base URL. ``None`` for bundled datasets.
    license
        SPDX license tag, e.g. ``"CC0-1.0"``, ``"CC-BY-4.0"``.
    citation
        Citation text (BibTeX or free-form).
    origin
        Upstream source URL (for provenance).
    bundled
        ``True`` if the asset ships inside the wheel under
        ``pyfieldml/datasets/_bundled/``.

    """

    name: str
    files: dict[str, str] = field(default_factory=dict)
    url_prefix: str | None = None
    license: str = ""
    citation: str = ""
    origin: str = ""
    bundled: bool = False


REGISTRY: dict[str, DatasetInfo] = {
    "unit_cube": DatasetInfo(
        name="unit_cube",
        files={"unit_cube.fieldml": ""},
        license="CC0-1.0",
        citation="Authored for pyfieldml, CC0 dedication.",
        origin="https://github.com/kchemorion/pyfieldml",
        bundled=True,
    ),
    "femur": DatasetInfo(
        name="femur",
        files={"femur.fieldml": ""},
        license="CC0-1.0",
        citation=(
            "Anatomical-synthetic femur (CSG-assembled shaft + femoral head + "
            "neck + trochanters + distal condyles, with a radial BMD gradient) "
            "authored for pyfieldml. CC0 dedication."
        ),
        origin="https://github.com/kchemorion/pyfieldml",
        bundled=True,
    ),
    "rectus_femoris": DatasetInfo(
        name="rectus_femoris",
        files={"rectus_femoris.fieldml": ""},
        license="CC0-1.0",
        citation=("Synthetic muscle spindle shape authored for pyfieldml. CC0 dedication."),
        origin="https://github.com/kchemorion/pyfieldml",
        bundled=True,
    ),
    "bunny_stanford": DatasetInfo(
        name="bunny_stanford",
        files={"bunny_stanford.fieldml": ""},
        license="Public domain (Stanford 3D Scanning Repository)",
        citation=(
            "Turk, G. and Levoy, M. (1994). Zippered polygon meshes from range "
            "images. SIGGRAPH '94. Stanford 3D Scanning Repository."
        ),
        origin="https://graphics.stanford.edu/data/3Dscanrep/",
        bundled=True,
    ),
    "femur_bodyparts3d": DatasetInfo(
        name="femur_bodyparts3d",
        files={"femur_bodyparts3d.fieldml": ""},
        license="CC-BY-SA-2.1-JP",
        citation=(
            "BodyParts3D, (c) The Database Center for Life Science (DBCLS), "
            "licensed under CC-BY-SA 2.1 JP. "
            "https://lifesciencedb.jp/bp3d/"
        ),
        origin="https://lifesciencedb.jp/bp3d/",
        bundled=True,
    ),
    "vertebra_l3": DatasetInfo(
        name="vertebra_l3",
        files={"vertebra_l3.fieldml": ""},
        license="CC-BY-SA-2.1-JP",
        citation=(
            "BodyParts3D lumbar vertebra (element FJ3159), "
            "(c) The Database Center for Life Science (DBCLS), "
            "licensed under CC-BY-SA 2.1 JP. "
            "https://lifesciencedb.jp/bp3d/"
        ),
        origin="https://lifesciencedb.jp/bp3d/",
        bundled=True,
    ),
    "scapula": DatasetInfo(
        name="scapula",
        files={"scapula.fieldml": ""},
        license="CC-BY-SA-2.1-JP",
        citation=(
            "BodyParts3D left scapula (element FJ3279, FMA13396), "
            "(c) The Database Center for Life Science (DBCLS), "
            "licensed under CC-BY-SA 2.1 JP. "
            "https://lifesciencedb.jp/bp3d/"
        ),
        origin="https://lifesciencedb.jp/bp3d/",
        bundled=True,
    ),
    "tibia_left": DatasetInfo(
        name="tibia_left",
        files={"tibia_left.fieldml": ""},
        license="CC-BY-SA-2.1-JP",
        citation=(
            "BodyParts3D left tibia (element FJ3282, FMA24478), "
            "(c) The Database Center for Life Science (DBCLS), "
            "licensed under CC-BY-SA 2.1 JP. "
            "https://lifesciencedb.jp/bp3d/"
        ),
        origin="https://lifesciencedb.jp/bp3d/",
        bundled=True,
    ),
    "hip_bone_left": DatasetInfo(
        name="hip_bone_left",
        files={"hip_bone_left.fieldml": ""},
        license="CC-BY-SA-2.1-JP",
        citation=(
            "BodyParts3D hip bone (element FJ3152, FMA16586), "
            "(c) The Database Center for Life Science (DBCLS), "
            "licensed under CC-BY-SA 2.1 JP. "
            "https://lifesciencedb.jp/bp3d/"
        ),
        origin="https://lifesciencedb.jp/bp3d/",
        bundled=True,
    ),
    "skull": DatasetInfo(
        name="skull",
        files={"skull.fieldml": ""},
        license="CC-BY-SA-2.1-JP",
        citation=(
            "BodyParts3D skull (FMA46565, compound of 43 cranial and facial "
            "sub-part elements), (c) The Database Center for Life Science "
            "(DBCLS), licensed under CC-BY-SA 2.1 JP. "
            "https://lifesciencedb.jp/bp3d/"
        ),
        origin="https://lifesciencedb.jp/bp3d/",
        bundled=True,
    ),
    # myocardium added in a later task as a synthetic build.
}
