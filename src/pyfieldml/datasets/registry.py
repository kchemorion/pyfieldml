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
    # femur, rectus_femoris, myocardium added in T3 as synthetic builds.
}
