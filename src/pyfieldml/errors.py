"""Error hierarchy for pyfieldml. All library errors subclass ``FieldMLError``."""

from __future__ import annotations

from collections.abc import Iterable


class FieldMLError(Exception):
    """Base class for every error raised by pyfieldml."""


class FieldMLParseError(FieldMLError):
    """Raised when an XML document cannot be parsed as FieldML."""

    def __init__(
        self,
        message: str,
        *,
        source_file: str | None = None,
        line: int | None = None,
    ) -> None:
        self.source_file = source_file
        self.line = line
        suffix = ""
        if source_file is not None:
            suffix = f" (in {source_file}"
            suffix += f", line {line})" if line is not None else ")"
        elif line is not None:
            suffix = f" (line {line})"
        super().__init__(f"{message}{suffix}")


class FieldMLValidationError(FieldMLError):
    """Raised when a FieldML document fails XSD or semantic validation."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = list(issues)
        joined = "\n - ".join(self.issues) if self.issues else "(no details)"
        super().__init__(f"FieldML validation failed:\n - {joined}")


class UnresolvedImportError(FieldMLError):
    """Raised when a FieldML ``<Import>`` cannot be located."""


class UnboundArgumentError(FieldMLError):
    """Raised when an ArgumentEvaluator is evaluated without a binding."""


class EvaluationError(FieldMLError):
    """Raised when the evaluation engine cannot produce a result."""


class BasisNotFoundError(FieldMLError):
    """Raised when a referenced basis function is not registered."""
