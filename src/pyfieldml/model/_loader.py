"""DOM → model graph loader: full implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from lxml import etree

from pyfieldml.data.hdf5 import Hdf5DenseBackend, Hdf5DOKBackend
from pyfieldml.data.text import ExternalTextBackend, InlineTextBackend
from pyfieldml.dom.parser import parse_file
from pyfieldml.errors import FieldMLParseError
from pyfieldml.model.bindings import ArgumentBinding, BindingTable
from pyfieldml.model.evaluators import (
    AggregateEvaluator,
    ArgumentEvaluator,
    ConstantEvaluator,
    Evaluator,
    ExternalEvaluator,
    ParameterEvaluator,
    PiecewiseEvaluator,
    ReferenceEvaluator,
)
from pyfieldml.model.region import Region
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)


def load_document(path: Path) -> Region:
    """Load a FieldML document and return its (single) primary region.

    For multi-region documents, only the first region is returned; a future
    Document class will hold the full list of regions.
    """
    parsed = parse_file(path)
    regions = parsed.tree.getroot().findall("Region")
    if not regions:
        raise FieldMLParseError("No <Region> in document", source_file=str(path))
    return _load_region(regions[0], base_dir=Path(path).parent)


def load_region_from_file(path: Path, *, region_name: str) -> Region:
    parsed = parse_file(path)
    for region_elem in parsed.tree.getroot().findall("Region"):
        if region_elem.get("name") == region_name:
            return _load_region(region_elem, base_dir=Path(path).parent)
    raise ValueError(f"Region {region_name!r} not found in {path}")


def _load_region(elem: etree._Element, *, base_dir: Path) -> Region:
    region = Region(name=elem.get("name", ""))

    # First pass: types (referenced by evaluators).
    for child in elem:
        if child.tag == "BooleanType":
            region.add_type(BooleanType(name=child.get("name", "")))
        elif child.tag == "EnsembleType":
            region.add_type(_load_ensemble(child))
        elif child.tag == "ContinuousType":
            region.add_type(_load_continuous(child))
        elif child.tag == "MeshType":
            region.add_type(_load_mesh(child, region))

    # Second pass: evaluators (may reference types AND each other).
    evaluator_elems = [c for c in elem if isinstance(c.tag, str) and c.tag.endswith("Evaluator")]

    # Pass 2a: argument, constant, external, parameter (no cross-refs).
    for c in evaluator_elems:
        ev = _load_leaf_evaluator(c, region, base_dir=base_dir)
        if ev is not None:
            region.add_evaluator(ev)

    # Pass 2b: reference, aggregate, piecewise (may cross-ref).
    for c in evaluator_elems:
        if c.get("name", "") in region.evaluators:
            continue
        ev = _load_composite_evaluator(c, region)
        if ev is not None:
            region.add_evaluator(ev)

    return region


# --- helpers ---


def _load_ensemble(elem: etree._Element) -> EnsembleType:
    members_elem = elem.find("Members")
    members: Any = range(0)
    if members_elem is not None:
        mr = members_elem.find("MemberRange")
        if mr is not None:
            members = range(int(mr.get("min", "1")), int(mr.get("max", "0")) + 1)
    return EnsembleType(name=elem.get("name", ""), members=members)


def _load_continuous(elem: etree._Element) -> ContinuousType:
    comp = elem.find("Components")
    if comp is not None:
        return ContinuousType(
            name=elem.get("name", ""),
            component_name=comp.get("name"),
            component_count=int(comp.get("count", "1")),
        )
    return ContinuousType(name=elem.get("name", ""))


def _load_mesh(elem: etree._Element, region: Region) -> MeshType:
    elements_elem = elem.find("Elements")
    chart_elem = elem.find("Chart")
    if elements_elem is None or chart_elem is None:
        raise FieldMLParseError(f"MeshType {elem.get('name')!r} missing Elements or Chart child")
    elements_name = elements_elem.get("name", "")
    chart_name = chart_elem.get("name", "")
    return MeshType(
        name=elem.get("name", ""),
        elements=region.ensembles[elements_name],
        chart=region.continuous[chart_name],
    )


def _load_leaf_evaluator(
    elem: etree._Element, region: Region, *, base_dir: Path
) -> Evaluator | None:
    name = elem.get("name", "")
    value_type_name = elem.get("valueType")
    value_type = _resolve_value_type(value_type_name, region) if value_type_name else None

    if elem.tag == "ArgumentEvaluator":
        return ArgumentEvaluator(name=name, value_type=cast(Any, value_type))
    if elem.tag == "ConstantEvaluator":
        value = _parse_constant_value(elem.get("value", ""), value_type)
        return ConstantEvaluator(name=name, value_type=cast(Any, value_type), value=value)
    if elem.tag == "ExternalEvaluator":
        return ExternalEvaluator(name=name, value_type=cast(Any, value_type))
    if elem.tag == "ParameterEvaluator":
        data = _load_parameter_data(elem, base_dir=base_dir)
        return ParameterEvaluator(name=name, value_type=cast(Any, value_type), data=data)
    return None


def _load_composite_evaluator(elem: etree._Element, region: Region) -> Evaluator | None:
    name = elem.get("name", "")
    value_type_name = elem.get("valueType")
    value_type = _resolve_value_type(value_type_name, region) if value_type_name else None

    if elem.tag == "ReferenceEvaluator":
        source_name = elem.get("evaluator", "")
        source = region.evaluators.get(source_name)
        if source is None:
            # Source not present in this region (e.g. it lives in an import
            # we have not yet resolved). Skip — a stricter pass can diagnose.
            return None
        bindings = _load_bindings(elem.find("Bindings"), region)
        return ReferenceEvaluator(
            name=name,
            value_type=cast(Any, value_type),
            source=source,
            bindings=bindings,
        )
    if elem.tag == "AggregateEvaluator":
        components: list[Evaluator] = []
        ce = elem.find("ComponentEvaluators")
        if ce is not None:
            default_name = ce.get("default")
            if default_name and default_name in region.evaluators:
                components.append(region.evaluators[default_name])
        return AggregateEvaluator(
            name=name, value_type=cast(Any, value_type), components=components
        )
    if elem.tag == "PiecewiseEvaluator":
        pieces: dict[int, Evaluator] = {}
        default: Evaluator | None = None
        ev_map = elem.find("EvaluatorMap")
        if ev_map is not None:
            d = ev_map.get("default")
            if d and d in region.evaluators:
                default = region.evaluators[d]
            for entry in ev_map.findall("EvaluatorMapEntry"):
                ev_name = entry.get("evaluator", "")
                if ev_name in region.evaluators:
                    pieces[int(entry.get("indexValue", "0"))] = region.evaluators[ev_name]
        return PiecewiseEvaluator(
            name=name, value_type=cast(Any, value_type), pieces=pieces, default=default
        )
    return None


def _load_bindings(bindings_elem: etree._Element | None, region: Region) -> BindingTable:
    if bindings_elem is None:
        return BindingTable([])
    bs: list[ArgumentBinding] = []
    for b in bindings_elem.findall("Bind"):
        arg_name = b.get("argument", "")
        src_name = b.get("source", "")
        arg_obj = region.evaluators.get(arg_name)
        src_obj = region.evaluators.get(src_name)
        if isinstance(arg_obj, ArgumentEvaluator) and src_obj is not None:
            bs.append(ArgumentBinding(argument=arg_obj, source=src_obj))
    return BindingTable(bs)


def _safe_href_path(base_dir: Path, href: str) -> Path:
    """Resolve href relative to base_dir. Reject traversal outside base_dir.

    Guards against malicious FieldML XML references like
    ``<DataResourceHref xlink:href="../../etc/passwd" .../>`` which would
    otherwise let a document read arbitrary files off the host filesystem
    when loaded with ``pyfieldml.read(...)``.
    """
    candidate = (base_dir / href).resolve()
    base = base_dir.resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise FieldMLParseError(
            f"Refusing to resolve href {href!r}: escapes base_dir {base}"
        ) from None
    return candidate


def _load_parameter_data(elem: etree._Element, *, base_dir: Path) -> Any:
    """Discriminate on the storage-format child element."""
    inline = elem.find(".//DataResourceString")
    if inline is not None:
        ds = elem.find(".//DataSource")
        shape, dtype = _parse_data_source(ds)
        return InlineTextBackend(text=inline.text or "", shape=shape, dtype=dtype)

    href_node = elem.find(".//DataResourceHref[@format='PLAIN_TEXT']")
    if href_node is not None:
        ds = elem.find(".//DataSource")
        shape, dtype = _parse_data_source(ds)
        href = href_node.get("{http://www.w3.org/1999/xlink}href", "")
        # Validate the resolved path stays inside base_dir, but keep the
        # backend's (base_dir, href) pair intact so round-trip writing of
        # relative hrefs still works.
        _safe_href_path(base_dir, href)
        return ExternalTextBackend(base_dir=base_dir, href=href, shape=shape, dtype=dtype)

    h5_node = elem.find(".//DataResourceHref[@format='HDF5']")
    if h5_node is not None:
        ds = elem.find(".//DataSource")
        href = h5_node.get("{http://www.w3.org/1999/xlink}href", "")
        dataset = ds.get("location", "/") if ds is not None else "/"
        safe_path = _safe_href_path(base_dir, href)
        if elem.find(".//DOKArrayData") is not None:
            return Hdf5DOKBackend(path=safe_path, group=dataset)
        return Hdf5DenseBackend(path=safe_path, dataset=dataset)

    raise FieldMLParseError(
        f"ParameterEvaluator {elem.get('name')!r}: unrecognized data-resource shape"
    )


def _parse_data_source(ds: etree._Element | None) -> tuple[tuple[int, ...], str]:
    if ds is None:
        return ((0,), "float64")
    size_elem = ds.find("ArrayDataSize")
    if size_elem is not None and size_elem.text:
        shape = tuple(int(v) for v in size_elem.text.split())
    else:
        shape = (0,)
    return shape, "float64"


def _parse_constant_value(raw: str, value_type: Any) -> Any:
    raw = raw.strip()
    if not raw:
        return 0.0
    if " " in raw:
        return tuple(float(x) for x in raw.split())
    try:
        return float(raw)
    except ValueError:
        return raw


def _resolve_value_type(name: str, region: Region) -> Any:
    if name in region.booleans:
        return region.booleans[name]
    if name in region.ensembles:
        return region.ensembles[name]
    if name in region.continuous:
        return region.continuous[name]
    if name in region.meshes:
        return region.meshes[name]
    # Tolerate component-name references: a ``Components`` child of a
    # ContinuousType defines a companion type used by component-argument
    # evaluators in the stdlib. Resolve those to the parent ContinuousType.
    for ct in region.continuous.values():
        if ct.component_name == name:
            return ct
    # Unknown valueType (e.g. from an unresolved import): return ``None``
    # rather than failing the whole load — callers treat this as an opaque
    # placeholder. A stricter pass can validate later.
    return None
