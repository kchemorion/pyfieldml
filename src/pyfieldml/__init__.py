"""pyfieldml: a pure-Python implementation of FieldML 0.5."""

from __future__ import annotations

from pathlib import Path

from pyfieldml._version import __version__
from pyfieldml.document import Document
from pyfieldml.errors import (
    BasisNotFoundError,
    EvaluationError,
    FieldMLError,
    FieldMLParseError,
    FieldMLValidationError,
    UnboundArgumentError,
    UnresolvedImportError,
)
from pyfieldml.model.region import Region

PathLike = str | Path


def read(path: PathLike) -> Document:
    """Read a FieldML document from ``path``. Convenience for ``Document.from_file``."""
    return Document.from_file(path)


def validate(path: PathLike) -> None:
    """Parse and XSD-validate ``path``. Raises ``FieldMLError`` on any failure."""
    Document.from_file(path).validate()


__all__ = [
    "BasisNotFoundError",
    "Document",
    "EvaluationError",
    "FieldMLError",
    "FieldMLParseError",
    "FieldMLValidationError",
    "Region",
    "UnboundArgumentError",
    "UnresolvedImportError",
    "__version__",
    "read",
    "validate",
]
