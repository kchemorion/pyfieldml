"""Document — the user-facing FieldML document wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from lxml import etree

from pyfieldml.dom.parser import ParsedDocument, parse_file
from pyfieldml.dom.validator import validate_tree
from pyfieldml.dom.writer import write_file
from pyfieldml.model._loader import load_document
from pyfieldml.model.evaluators import Evaluator
from pyfieldml.model.region import Region
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)

PathLike = str | Path


class Document:
    """An in-memory FieldML document.

    Holds a single primary ``Region`` (the first one in the document) plus
    the original DOM tree for round-trip writing.
    """

    def __init__(self, parsed: ParsedDocument, region: Region) -> None:
        self._parsed = parsed
        self._region = region

    @classmethod
    def from_file(cls, path: PathLike) -> Document:
        """Read a FieldML document from an on-disk file."""
        parsed = parse_file(path)
        region = load_document(Path(path))
        return cls(parsed, region)

    @classmethod
    def from_string(cls, content: str | bytes) -> Document:
        """Read a FieldML document from an in-memory string/bytes."""
        raise NotImplementedError("Phase 1: from_string is deferred; use from_file for now.")

    @classmethod
    def from_region(cls, region: Region) -> Document:
        """Build a Document from a programmatically-constructed Region."""
        from pyfieldml.dom.parser import ParsedDocument
        from pyfieldml.model._writer import region_to_tree

        tree = region_to_tree(region)
        parsed = ParsedDocument(tree=tree, version="0.5.0", source_file=None)
        return cls(parsed, region)

    @property
    def source_version(self) -> str:
        """FieldML version declared on the root element (``@version``)."""
        return self._parsed.version

    @property
    def tree(self) -> etree._ElementTree:
        """Underlying lxml ElementTree. Escape hatch; prefer semantic APIs."""
        return self._parsed.tree

    @property
    def region(self) -> Region:
        """The document's primary (first) region."""
        return self._region

    @property
    def booleans(self) -> Mapping[str, BooleanType]:
        return self._region.booleans

    @property
    def ensembles(self) -> Mapping[str, EnsembleType]:
        return self._region.ensembles

    @property
    def continuous(self) -> Mapping[str, ContinuousType]:
        return self._region.continuous

    @property
    def meshes(self) -> Mapping[str, MeshType]:
        return self._region.meshes

    @property
    def evaluators(self) -> Mapping[str, Evaluator]:
        return self._region.evaluators

    def write(self, path: PathLike) -> None:
        """Phase-1 write path: re-serializes the DOM tree we parsed from."""
        write_file(self._parsed.tree, path)

    def validate(self) -> None:
        """Validate the document against the bundled FieldML 0.5 XSD."""
        validate_tree(self._parsed.tree)
