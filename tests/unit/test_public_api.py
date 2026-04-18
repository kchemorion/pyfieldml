"""End-to-end round-trip tests exercising the public top-level API."""

from __future__ import annotations

from pathlib import Path

import pytest

import pyfieldml as fml
from pyfieldml.errors import FieldMLError


def test_read_returns_document(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "minimal.fieldml")
    assert isinstance(doc, fml.Document)
    assert doc.source_version == "0.5.0"


def test_validate_passes_on_valid_document(fixtures_dir: Path) -> None:
    fml.validate(fixtures_dir / "minimal.fieldml")


def test_validate_raises_on_invalid_document(fixtures_dir: Path) -> None:
    with pytest.raises(FieldMLError):
        fml.validate(fixtures_dir / "invalid_no_name.fieldml")


def test_read_then_write_then_reread(fixtures_dir: Path, tmp_path: Path) -> None:
    src = fixtures_dir / "two_types.fieldml"
    doc = fml.read(src)
    out = tmp_path / "roundtrip.fieldml"
    doc.write(out)
    reread = fml.read(out)
    original_tags = [c.tag for c in doc.tree.getroot().find("Region")]
    roundtrip_tags = [c.tag for c in reread.tree.getroot().find("Region")]
    assert original_tags == roundtrip_tags


def test_package_exposes_version() -> None:
    import re

    assert re.match(r"^\d+\.\d+\.\d+([a-zA-Z0-9.+-]*)$", fml.__version__)
