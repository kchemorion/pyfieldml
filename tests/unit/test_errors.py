"""Tests for the pyfieldml error hierarchy."""

from __future__ import annotations

import pytest

from pyfieldml.errors import (
    BasisNotFoundError,
    EvaluationError,
    FieldMLError,
    FieldMLParseError,
    FieldMLValidationError,
    UnboundArgumentError,
    UnresolvedImportError,
)


@pytest.mark.parametrize(
    "cls",
    [
        FieldMLParseError,
        FieldMLValidationError,
        UnresolvedImportError,
        UnboundArgumentError,
        EvaluationError,
        BasisNotFoundError,
    ],
)
def test_all_errors_subclass_fieldml_error(cls: type[Exception]) -> None:
    assert issubclass(cls, FieldMLError)


def test_parse_error_carries_source_line() -> None:
    err = FieldMLParseError("bad tag", source_file="/tmp/x.fieldml", line=42)
    assert err.source_file == "/tmp/x.fieldml"
    assert err.line == 42
    assert "line 42" in str(err)


def test_validation_error_holds_multiple_issues() -> None:
    err = FieldMLValidationError(["issue A", "issue B"])
    assert len(err.issues) == 2
    assert "issue A" in str(err)
    assert "issue B" in str(err)
