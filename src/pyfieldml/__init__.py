"""pyfieldml: a pure-Python implementation of FieldML 0.5."""

from __future__ import annotations

import contextlib
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


# Auto-register with meshio (if installed) so that ``meshio.read("x.fieldml")``
# works out of the box. meshio does not expose an entry-point hook for format
# registration, so the cleanest place to register is on ``import pyfieldml``.
# Silent no-op when meshio is not installed — the [meshio] extra is optional.
with contextlib.suppress(ImportError):
    import pyfieldml.interop.meshio as _interop_meshio  # noqa: F401
