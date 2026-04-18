"""Tests for the DOM parser layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string
from pyfieldml.errors import FieldMLParseError


def test_parse_minimal_file_returns_parsed_document(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "minimal.fieldml")
    assert isinstance(parsed, ParsedDocument)
    assert parsed.version == "0.5.0"
    assert parsed.source_file == str(fixtures_dir / "minimal.fieldml")
    root = parsed.tree.getroot()
    assert root.tag == "Fieldml"
    assert root.find("Region").get("name") == "test"


def test_parse_two_types_preserves_child_order(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "two_types.fieldml")
    region = parsed.tree.getroot().find("Region")
    child_tags = [c.tag for c in region if isinstance(c.tag, str)]
    assert child_tags == ["BooleanType", "ContinuousType"]


def test_parse_string_matches_parse_file(fixtures_dir: Path) -> None:
    content = (fixtures_dir / "minimal.fieldml").read_bytes()
    parsed = parse_string(content)
    assert parsed.version == "0.5.0"
    assert parsed.source_file is None


def test_parse_rejects_non_fieldml_root(fixtures_dir: Path) -> None:
    with pytest.raises(FieldMLParseError, match="root element"):
        parse_file(fixtures_dir / "not_fieldml.xml")


def test_parse_reports_source_location_on_malformed_xml(tmp_path: Path) -> None:
    bad = tmp_path / "broken.fieldml"
    bad.write_text('<?xml version="1.0"?><Fieldml version="0.5.0"><Region>')
    with pytest.raises(FieldMLParseError) as exc:
        parse_file(bad)
    assert exc.value.source_file == str(bad)
    assert exc.value.line is not None


def test_parse_rejects_unknown_version(tmp_path: Path) -> None:
    bad = tmp_path / "future.fieldml"
    bad.write_text('<?xml version="1.0"?><Fieldml version="9.9.9"><Region name="r"/></Fieldml>')
    with pytest.raises(FieldMLParseError, match="Unsupported FieldML version"):
        parse_file(bad)


def test_parse_accepts_legacy_versions(tmp_path: Path) -> None:
    for v in ("0.3", "0.3.0", "0.4", "0.4.0"):
        legacy = tmp_path / f"legacy_{v}.fieldml"
        legacy.write_text(
            f'<?xml version="1.0"?><Fieldml version="{v}"><Region name="r"/></Fieldml>'
        )
        parsed = parse_file(legacy)
        assert parsed.version == v
