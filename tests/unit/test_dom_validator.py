"""Tests for XSD validation of FieldML documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyfieldml.dom.parser import parse_file
from pyfieldml.dom.validator import validate_tree
from pyfieldml.errors import FieldMLValidationError


def test_minimal_document_validates(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "minimal.fieldml")
    validate_tree(parsed.tree)  # must not raise


def test_bundled_stdlib_validates() -> None:
    from pyfieldml.library import get_stdlib_path

    parsed = parse_file(get_stdlib_path())
    validate_tree(parsed.tree)  # must not raise


def test_missing_required_attribute_fails(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "invalid_no_name.fieldml")
    with pytest.raises(FieldMLValidationError) as exc:
        validate_tree(parsed.tree)
    joined = "\n".join(exc.value.issues)
    assert "name" in joined.lower()
