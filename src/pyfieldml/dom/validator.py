"""XSD validation of FieldML 0.5 documents against the bundled schema."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from lxml import etree

from pyfieldml.errors import FieldMLValidationError
from pyfieldml.library import get_xsd_path

# URLs the FieldML 0.5 XSD references via ``<xs:import>`` that we mirror locally
# to avoid any network dependency during schema parsing.
_REMOTE_XLINK_URLS: frozenset[str] = frozenset(
    {
        "http://www.cellml.org/tools/cellml_1_1_schema/common/xlink-href.xsd",
        "https://www.cellml.org/tools/cellml_1_1_schema/common/xlink-href.xsd",
    }
)


class _LocalXlinkResolver(etree.Resolver):  # type: ignore[misc]
    """Redirect the FieldML XSD's remote xlink import to a bundled local copy."""

    def __init__(self, xlink_path: Path) -> None:
        super().__init__()
        self._xlink_path = xlink_path

    def resolve(self, system_url, public_id, context):  # type: ignore[no-untyped-def]
        if system_url in _REMOTE_XLINK_URLS:
            return self.resolve_filename(str(self._xlink_path), context)
        return None  # fall through to default resolution


@lru_cache(maxsize=1)
def _schema() -> etree.XMLSchema:
    """Load and cache the FieldML 0.5 XSD, with the xlink import mirrored locally."""
    xsd_path = get_xsd_path()
    xlink_path = xsd_path.parent / "xlink-href.xsd"

    parser = etree.XMLParser(load_dtd=False, no_network=True)
    parser.resolvers.add(_LocalXlinkResolver(xlink_path))

    with xsd_path.open("rb") as fh:
        schema_doc = etree.parse(fh, parser)
    return etree.XMLSchema(schema_doc)


def validate_tree(tree: etree._ElementTree) -> None:
    """Validate an lxml ElementTree against the bundled FieldML 0.5 XSD.

    Raises
    ------
    FieldMLValidationError
        If the document fails schema validation. ``.issues`` holds one entry
        per schema error, each including source line number when available.

    """
    schema = _schema()
    if schema.validate(tree):
        return
    issues = [
        f"{error.filename or '<input>'}:{error.line}: {error.message}" for error in schema.error_log
    ]
    raise FieldMLValidationError(issues)
