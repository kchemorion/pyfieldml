"""Serialize FieldML ElementTrees back to XML.

Output policy: UTF-8, XML declaration present, two-space indentation, trailing
newline. Attribute order follows lxml's stable insertion order, which matches
the input document on round-trip. Whitespace-only text nodes are preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from lxml import etree

PathLike = str | Path


def write_file(tree: etree._ElementTree, path: PathLike) -> None:
    """Serialize ``tree`` to ``path`` as UTF-8 XML with a trailing newline."""
    payload = _serialize(tree)
    Path(path).write_bytes(payload)


def write_string(tree: etree._ElementTree) -> bytes:
    """Serialize ``tree`` to UTF-8 bytes (with XML declaration)."""
    return _serialize(tree)


def _serialize(tree: etree._ElementTree) -> bytes:
    payload = cast(
        bytes,
        etree.tostring(
            tree,
            xml_declaration=True,
            encoding="UTF-8",
            pretty_print=True,
            standalone=None,
        ),
    )
    if not payload.endswith(b"\n"):
        payload += b"\n"
    return payload
