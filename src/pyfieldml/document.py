"""Phase-0 Document stub.

A thin façade over the DOM layer. Holds a parsed ``lxml`` tree and exposes
``write()`` / ``validate()`` / ``source_version``. Phase 1 replaces this
with a fully typed semantic object model; callers that only need I/O can
use this stub as-is.
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string
from pyfieldml.dom.validator import validate_tree
from pyfieldml.dom.writer import write_file

PathLike = str | Path


class Document:
    """An in-memory FieldML document.

    Phase 0: wraps the raw DOM tree. Phase 1 replaces the internals with a
    semantic object model; the public surface here (``from_file``, ``write``,
    ``validate``, ``source_version``) is forward-compatible.
    """

    def __init__(self, parsed: ParsedDocument) -> None:
        self._parsed = parsed

    @classmethod
    def from_file(cls, path: PathLike) -> Document:
        """Read a FieldML document from an on-disk file."""
        return cls(parse_file(path))

    @classmethod
    def from_string(cls, content: str | bytes) -> Document:
        """Read a FieldML document from an in-memory string/bytes."""
        return cls(parse_string(content))

    @property
    def source_version(self) -> str:
        """FieldML version declared on the root element (``@version``)."""
        return self._parsed.version

    @property
    def tree(self) -> etree._ElementTree:
        """Underlying lxml ElementTree. Escape hatch; prefer semantic APIs."""
        return self._parsed.tree

    def write(self, path: PathLike) -> None:
        """Serialize the document to ``path`` as FieldML 0.5 XML."""
        write_file(self._parsed.tree, path)

    def validate(self) -> None:
        """Validate the document against the bundled FieldML 0.5 XSD."""
        validate_tree(self._parsed.tree)
