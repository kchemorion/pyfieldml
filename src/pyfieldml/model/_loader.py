"""DOM → model graph loader. Expanded in Task 11."""

from __future__ import annotations

from pathlib import Path

from pyfieldml.dom.parser import parse_file
from pyfieldml.model.region import Region
from pyfieldml.model.types import BooleanType, ContinuousType


def load_region_from_file(path: Path, *, region_name: str) -> Region:
    """Minimal loader: reads Boolean/Continuous types from the named region.

    Expanded in Task 11 to cover the full model.
    """
    parsed = parse_file(path)
    for region_elem in parsed.tree.getroot().findall("Region"):
        if region_elem.get("name") == region_name:
            region = Region(name=region_name)
            for child in region_elem:
                if child.tag == "BooleanType":
                    region.add_type(BooleanType(name=child.get("name", "")))
                elif child.tag == "ContinuousType":
                    _add_continuous(region, child)
            return region
    raise ValueError(
        f"Region {region_name!r} not found in {path} "
        f"(available: {[r.get('name') for r in parsed.tree.getroot().findall('Region')]})"
    )


def _add_continuous(region: Region, elem) -> None:  # type: ignore[no-untyped-def]
    comp = elem.find("Components")
    if comp is not None:
        region.add_type(
            ContinuousType(
                name=elem.get("name", ""),
                component_name=comp.get("name"),
                component_count=int(comp.get("count", "1")),
            )
        )
    else:
        region.add_type(ContinuousType(name=elem.get("name", "")))
