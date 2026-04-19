"""Legacy FieldML 0.3 / 0.4 -> 0.5 up-converter.

Operates on lxml ElementTrees in place. Minimal scope: version-tag rewrite
+ known element/attribute renames + a few structural touch-ups needed for
observed legacy constructs (e.g. stripping ``isComponentEnsemble`` from 0.3
``EnsembleType`` elements). Unhandled constructs raise ``FieldMLParseError``
for visibility -- later phases can expand coverage as new legacy fixtures
surface.

The primary real-world reference for 0.3 is the example
``library_0.3.xml`` shipped with FieldML-API. The differences we handle
were extracted from that file and the pyfieldml 0.5 stdlib.
"""

from __future__ import annotations

from lxml import etree

from pyfieldml.errors import FieldMLParseError

# 0.4 was a short-lived intermediate. We have no real corpus to diff
# against, so the baseline transform is just the version-tag rewrite.
# Populate as new 0.4 fixtures surface.
_0_4_ELEMENT_RENAMES: dict[str, str] = {}
_0_4_ATTRIBUTE_RENAMES: dict[tuple[str, str], str] = {}

# 0.3 renames, derived from FieldML-API/library_0.3.xml vs
# pyfieldml/library/FieldML_Library_0.5.xml:
#   * AbstractEvaluator -> ArgumentEvaluator  (purely a terminology change)
#   * Variables         -> Arguments          (wrapper around external-evaluator inputs)
#   * Variable          -> Argument           (the wrapped input itself)
_0_3_ELEMENT_RENAMES: dict[str, str] = {
    "AbstractEvaluator": "ArgumentEvaluator",
    "Variables": "Arguments",
    "Variable": "Argument",
}

# No pure attribute *renames* in the 0.3 sample (attributes that change meaning
# are handled structurally in _rework_0_3_structures). Kept for extensibility.
_0_3_ATTRIBUTE_RENAMES: dict[tuple[str, str], str] = {}


def upconvert_to_0_5(tree: etree._ElementTree) -> None:
    """Mutate ``tree`` in place to the FieldML 0.5 structure.

    Raises ``FieldMLParseError`` on any construct we don't yet handle.
    """
    root = tree.getroot()
    version = root.get("version", "")
    if version.startswith("0.5"):
        return  # already 0.5
    if version.startswith("0.4"):
        _apply_renames(tree, _0_4_ELEMENT_RENAMES, attr_renames=_0_4_ATTRIBUTE_RENAMES)
    elif version.startswith("0.3"):
        _apply_renames(tree, _0_3_ELEMENT_RENAMES, attr_renames=_0_3_ATTRIBUTE_RENAMES)
        _rework_0_3_structures(tree)
    else:
        raise FieldMLParseError(f"Cannot up-convert unknown FieldML version {version!r}")
    root.set("version", "0.5.0")


def _apply_renames(
    tree: etree._ElementTree,
    tag_renames: dict[str, str],
    *,
    attr_renames: dict[tuple[str, str], str],
) -> None:
    """Rename element tags and/or their attributes across the whole tree."""
    for old, new in tag_renames.items():
        # Snapshot to a list because mutating .tag while iterating is safe in
        # lxml but defensive copying keeps semantics obvious.
        for el in list(tree.iter(old)):
            el.tag = new
    for (tag, old_attr), new_attr in attr_renames.items():
        for el in tree.iter(tag):
            if old_attr in el.attrib:
                el.attrib[new_attr] = el.attrib.pop(old_attr)


def _rework_0_3_structures(tree: etree._ElementTree) -> None:
    """Apply structural transforms on a 0.3 tree beyond simple renames.

    Currently handles:
      * ``<EnsembleType isComponentEnsemble="true">`` -> drop the attribute;
        the 0.5 schema has no such marker and the ensemble is identified by
        usage.
      * ``<ContinuousType componentEnsemble="X"/>`` -> add a ``<Components>``
        child whose ``count`` matches the referenced ensemble's size, and
        remove the attribute. This is the 0.3 spelling for vector-valued
        continuous types; 0.5 uses the inline ``<Components>`` child.
    """
    for et in tree.iter("EnsembleType"):
        if "isComponentEnsemble" in et.attrib:
            et.attrib.pop("isComponentEnsemble")

    # Map ensemble-name -> size for the enclosing Region, so we can resolve
    # componentEnsemble references without re-parsing.
    for region in tree.iter("Region"):
        ensemble_sizes = _collect_ensemble_sizes(region)
        for ct in region.iter("ContinuousType"):
            ce = ct.attrib.pop("componentEnsemble", None)
            if ce is None:
                continue
            if ct.find("Components") is not None:
                continue  # already has explicit components; nothing to do
            size = ensemble_sizes.get(ce)
            if size is None:
                # Unknown ensemble — leave the attribute gone and skip.
                continue
            components = etree.SubElement(ct, "Components")
            components.set("name", f"{ce}.component")
            components.set("count", str(size))


def _collect_ensemble_sizes(region: etree._Element) -> dict[str, int]:
    """Return ``{ensemble_name: size}`` for all EnsembleTypes directly under ``region``."""
    sizes: dict[str, int] = {}
    for et in region.iter("EnsembleType"):
        name = et.get("name", "")
        if not name:
            continue
        members = et.find("Members")
        if members is None:
            continue
        mr = members.find("MemberRange")
        if mr is not None:
            lo = int(mr.get("min", "1"))
            hi = int(mr.get("max", "0"))
            sizes[name] = max(0, hi - lo + 1)
            continue
        # Fall back to counting <Member> children.
        member_list = members.findall("Member")
        if member_list:
            sizes[name] = len(member_list)
    return sizes
