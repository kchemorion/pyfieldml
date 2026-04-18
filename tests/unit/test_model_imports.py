"""Tests for FieldML <Import> resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyfieldml.errors import UnresolvedImportError
from pyfieldml.model.imports import ImportResolver


def test_stdlib_import_resolves_without_network(fixtures_dir: Path) -> None:
    resolver = ImportResolver(base_dir=fixtures_dir)
    imported = resolver.resolve(
        "http://www.fieldml.org/resources/xml/0.5/FieldML_Library_0.5.xml",
        region_name="library",
    )
    assert imported.name == "library"
    assert "real.3d" in imported.continuous


def test_local_import_resolves_relative_to_source_file(tmp_path: Path) -> None:
    sublib = tmp_path / "sub.fieldml"
    sublib.write_text(
        '<?xml version="1.0"?><Fieldml version="0.5.0">'
        '<Region name="sub"><BooleanType name="b"/></Region></Fieldml>'
    )
    resolver = ImportResolver(base_dir=tmp_path)
    imported = resolver.resolve("sub.fieldml", region_name="sub")
    assert "b" in imported.booleans


def test_missing_import_raises(tmp_path: Path) -> None:
    resolver = ImportResolver(base_dir=tmp_path)
    with pytest.raises(UnresolvedImportError):
        resolver.resolve("does_not_exist.fieldml", region_name="x")
