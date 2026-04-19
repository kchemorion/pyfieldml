"""Parse FieldML XML documents into lxml ElementTrees.

This layer is a thin bijection: it validates the top-level shape (root element
is ``<Fieldml version="...">``) and records the version, but does not interpret
any FieldML semantics. Semantic parsing lives in ``pyfieldml.model``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from pyfieldml.dom.upconverter import upconvert_to_0_5
from pyfieldml.errors import FieldMLParseError

SUPPORTED_VERSIONS: frozenset[str] = frozenset(
    {
        "0.3",
        "0.3.0",
        "0.3_alpha",
        "0.4",
        "0.4.0",
        "0.5",
        "0.5.0",
    }
)

PathLike = str | Path


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed-but-uninterpreted FieldML document.

    Attributes
    ----------
    tree
        The underlying lxml ElementTree.
    version
        The FieldML version declared on the root element (``@version``).
    source_file
        Absolute path as a string, or ``None`` when parsed from a bytes/string source.

    """

    tree: etree._ElementTree
    version: str
    source_file: str | None


def parse_file(path: PathLike) -> ParsedDocument:
    """Parse a FieldML document from an on-disk file."""
    path = Path(path)
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    try:
        tree = etree.parse(str(path), parser)
    except etree.XMLSyntaxError as exc:
        line = getattr(exc, "lineno", None)
        raise FieldMLParseError(
            f"Malformed XML: {exc.msg}", source_file=str(path), line=line
        ) from exc
    version = _validate_root(tree, source_file=str(path))
    _maybe_upconvert(tree, version)
    return ParsedDocument(tree=tree, version=version, source_file=str(path))


def parse_string(content: str | bytes) -> ParsedDocument:
    """Parse a FieldML document from an in-memory string/bytes."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    try:
        root = etree.fromstring(content, parser)
    except etree.XMLSyntaxError as exc:
        line = getattr(exc, "lineno", None)
        raise FieldMLParseError(f"Malformed XML: {exc.msg}", line=line) from exc
    tree = etree.ElementTree(root)
    version = _validate_root(tree, source_file=None)
    _maybe_upconvert(tree, version)
    return ParsedDocument(tree=tree, version=version, source_file=None)


def _validate_root(tree: etree._ElementTree, *, source_file: str | None) -> str:
    root = tree.getroot()
    if root.tag != "Fieldml":
        raise FieldMLParseError(
            f"Expected root element <Fieldml>, got <{root.tag}>",
            source_file=source_file,
            line=root.sourceline,
        )
    version = root.get("version")
    if version is None:
        raise FieldMLParseError(
            "Root <Fieldml> element is missing required @version attribute",
            source_file=source_file,
            line=root.sourceline,
        )
    if version not in SUPPORTED_VERSIONS:
        raise FieldMLParseError(
            f"Unsupported FieldML version {version!r}. Supported: {sorted(SUPPORTED_VERSIONS)}",
            source_file=source_file,
            line=root.sourceline,
        )
    return str(version)


def _maybe_upconvert(tree: etree._ElementTree, version: str) -> None:
    """If ``version`` is 0.3.x or 0.4.x, upconvert ``tree`` to 0.5 in place.

    ``ParsedDocument.version`` still records the original version so callers
    (notably ``Document.source_version``) can introspect the pre-upconvert
    version number. The tree itself is 0.5-shaped after this call.
    """
    if version.startswith(("0.3", "0.4")):
        upconvert_to_0_5(tree)
