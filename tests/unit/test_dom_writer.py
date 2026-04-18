"""Tests for the DOM writer."""

from __future__ import annotations

from pathlib import Path

from pyfieldml.dom.parser import parse_file, parse_string
from pyfieldml.dom.validator import validate_tree
from pyfieldml.dom.writer import write_file, write_string


def test_round_trip_minimal_produces_valid_output(fixtures_dir: Path, tmp_path: Path) -> None:
    parsed = parse_file(fixtures_dir / "minimal.fieldml")
    out = tmp_path / "out.fieldml"
    write_file(parsed.tree, out)
    reparsed = parse_file(out)
    validate_tree(reparsed.tree)
    assert reparsed.version == "0.5.0"


def test_round_trip_preserves_child_count_and_order(fixtures_dir: Path, tmp_path: Path) -> None:
    parsed = parse_file(fixtures_dir / "two_types.fieldml")
    out = tmp_path / "roundtrip.fieldml"
    write_file(parsed.tree, out)
    reparsed = parse_file(out)
    original_tags = [c.tag for c in parsed.tree.getroot().find("Region")]
    roundtrip_tags = [c.tag for c in reparsed.tree.getroot().find("Region")]
    assert original_tags == roundtrip_tags


def test_write_string_returns_bytes_with_declaration() -> None:
    parsed = parse_string(
        b'<?xml version="1.0"?><Fieldml version="0.5.0">'
        b'<Region name="r"><BooleanType name="b"/></Region></Fieldml>'
    )
    output = write_string(parsed.tree)
    assert isinstance(output, bytes)
    assert output.startswith(b"<?xml")
    assert b"<Fieldml" in output


def test_write_file_uses_utf8_and_trailing_newline(fixtures_dir: Path, tmp_path: Path) -> None:
    parsed = parse_file(fixtures_dir / "minimal.fieldml")
    out = tmp_path / "out.fieldml"
    write_file(parsed.tree, out)
    raw = out.read_bytes()
    assert raw.startswith(b"<?xml")
    assert b'encoding="UTF-8"' in raw[:80] or b"encoding='UTF-8'" in raw[:80]
    assert raw.endswith(b"\n")
