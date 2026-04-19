"""Tests for the Phase-0 Document stub."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyfieldml.document import Document
from pyfieldml.errors import FieldMLError


def test_document_from_file_records_source_version(fixtures_dir: Path) -> None:
    doc = Document.from_file(fixtures_dir / "minimal.fieldml")
    assert doc.source_version == "0.5.0"


def test_document_write_produces_valid_output(fixtures_dir: Path, tmp_path: Path) -> None:
    doc = Document.from_file(fixtures_dir / "two_types.fieldml")
    out = tmp_path / "out.fieldml"
    doc.write(out)
    roundtrip = Document.from_file(out)
    assert roundtrip.source_version == "0.5.0"


def test_document_validate_succeeds_on_valid_input(fixtures_dir: Path) -> None:
    doc = Document.from_file(fixtures_dir / "minimal.fieldml")
    doc.validate()  # must not raise


def test_document_validate_raises_on_invalid_input(fixtures_dir: Path) -> None:
    doc = Document.from_file(fixtures_dir / "invalid_no_name.fieldml")
    with pytest.raises(FieldMLError):
        doc.validate()


def test_from_string_parses_minimal_document() -> None:
    doc = Document.from_string(
        b'<?xml version="1.0"?><Fieldml version="0.5.0">'
        b'<Region name="t"><BooleanType name="b"/></Region></Fieldml>'
    )
    assert doc.source_version == "0.5.0"
    assert "b" in doc.booleans


def test_from_string_accepts_str_input() -> None:
    doc = Document.from_string(
        '<?xml version="1.0"?><Fieldml version="0.5.0">'
        '<Region name="r"><BooleanType name="flag"/></Region></Fieldml>'
    )
    assert doc.region.name == "r"
    assert "flag" in doc.booleans


def test_from_string_raises_on_missing_region() -> None:
    with pytest.raises(FieldMLError):
        Document.from_string(b'<?xml version="1.0"?><Fieldml version="0.5.0"/>')
