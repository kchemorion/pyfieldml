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
    ref = resources.files(_PACKAGE).joinpath(filename)
    # ``as_file`` handles both loose-file and zipped-wheel installs.
    with resources.as_file(ref) as concrete:
        return Path(concrete)
