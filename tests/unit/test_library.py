"""Tests for bundled FieldML resource accessors."""

from __future__ import annotations

from pyfieldml.library import get_stdlib_path, get_xsd_path


def test_xsd_exists_and_is_xml() -> None:
    path = get_xsd_path()
    assert path.is_file()
    with path.open("rb") as fh:
        header = fh.read(256)
    assert b"<xs:schema" in header


def test_stdlib_exists_and_is_fieldml() -> None:
    path = get_stdlib_path()
    assert path.is_file()
    with path.open("rb") as fh:
        header = fh.read(256)
    assert b'<Fieldml version="0.5' in header
