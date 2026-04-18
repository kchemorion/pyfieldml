"""DOM layer: XML ↔ lxml tree, versioning, XSD validation, serialization."""

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string
from pyfieldml.dom.validator import validate_tree
from pyfieldml.dom.writer import write_file, write_string

__all__ = [
    "ParsedDocument",
    "parse_file",
    "parse_string",
    "validate_tree",
    "write_file",
    "write_string",
]
