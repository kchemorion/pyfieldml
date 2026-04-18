"""DOM layer: XML ↔ lxml tree, versioning, XSD validation."""

from pyfieldml.dom.parser import ParsedDocument, parse_file, parse_string

__all__ = ["ParsedDocument", "parse_file", "parse_string"]
