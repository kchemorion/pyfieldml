"""Resolution of <Import> directives.

Imports can target:
- The bundled FieldML standard library (by its canonical URL).
- A local file, resolved relative to the importing document's directory.
"""

from __future__ import annotations

from pathlib import Path

from pyfieldml.errors import FieldMLParseError, UnresolvedImportError
from pyfieldml.library import get_stdlib_path
from pyfieldml.model.region import Region

STDLIB_URLS = frozenset(
    {
        "http://www.fieldml.org/resources/xml/0.5/FieldML_Library_0.5.xml",
        "https://www.fieldml.org/resources/xml/0.5/FieldML_Library_0.5.xml",
        "FieldML_Library_0.5.xml",
    }
)


class ImportResolver:
    """Resolves ``<Import>`` targets to Region objects."""

    def __init__(self, *, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def resolve(self, href: str, *, region_name: str) -> Region:
        path = self._find_source(href)
        if path is None:
            raise UnresolvedImportError(
                f"Cannot locate FieldML import {href!r} (base_dir={self.base_dir})"
            )
        # Lazy to avoid cycles: _loader depends on this module.
        from pyfieldml.model._loader import load_region_from_file

        return load_region_from_file(path, region_name=region_name)

    def _find_source(self, href: str) -> Path | None:
        if href in STDLIB_URLS or href.endswith("FieldML_Library_0.5.xml"):
            return get_stdlib_path()
        # Guard against path-traversal: reject hrefs that escape base_dir.
        candidate = (self.base_dir / href).resolve()
        base = self.base_dir.resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            raise FieldMLParseError(
                f"Refusing to resolve import href {href!r}: escapes base_dir {base}"
            ) from None
        if candidate.is_file():
            return candidate
        return None
