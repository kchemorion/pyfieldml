"""Model → DOM tree writer.

Conforming to the FieldML 0.5 library convention, the root ``<Fieldml>``
element carries the ``xmlns:xsi`` namespace declaration plus the
``xsi:noNamespaceSchemaLocation`` hint pointing at the published XSD. This
matches the output of the C++ FieldML-API reference writer and keeps
cross-implementation round-trips shape-equivalent at the XML level.
"""

from __future__ import annotations

from lxml import etree

from pyfieldml.data.hdf5 import Hdf5DenseBackend, Hdf5DOKBackend
from pyfieldml.data.text import ExternalTextBackend, InlineTextBackend
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

FIELDML_VERSION = "0.5.0"

# XSD schemaLocation + xsi namespace, matching the FieldML 0.5 reference
# library's writer output. See module docstring for the rationale.
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_FIELDML_XSD_URL = "http://www.fieldml.org/resources/xml/0.5/FieldML_0.5.xsd"


def region_to_tree(region: Region) -> etree._ElementTree:
    nsmap = {"xsi": _XSI_NS}
    root = etree.Element(
        "Fieldml",
        attrib={
            "version": FIELDML_VERSION,
            f"{{{_XSI_NS}}}noNamespaceSchemaLocation": _FIELDML_XSD_URL,
        },
        nsmap=nsmap,
    )
    region_elem = etree.SubElement(root, "Region", {"name": region.name})

    for name in region.objects:
        obj = _objects_lookup(region, name)
        if isinstance(obj, BooleanType):
            etree.SubElement(region_elem, "BooleanType", {"name": obj.name})
        elif isinstance(obj, ContinuousType):
            _write_continuous(region_elem, obj)
        elif isinstance(obj, EnsembleType):
            _write_ensemble(region_elem, obj)
        elif isinstance(obj, MeshType):
            _write_mesh(region_elem, obj)
        elif isinstance(obj, Evaluator):
            _write_evaluator(region_elem, obj)

    return etree.ElementTree(root)


def _objects_lookup(region: Region, name: str) -> object:
    for d in (
        region.booleans,
        region.ensembles,
        region.continuous,
        region.meshes,
        region.evaluators,
    ):
        if name in d:
            return d[name]
    raise KeyError(name)


def _write_continuous(parent: etree._Element, t: ContinuousType) -> None:
    e = etree.SubElement(parent, "ContinuousType", {"name": t.name})
    if t.component_count > 1 and t.component_name:
        etree.SubElement(
            e,
            "Components",
            {"name": t.component_name, "count": str(t.component_count)},
        )


def _write_ensemble(parent: etree._Element, t: EnsembleType) -> None:
    e = etree.SubElement(parent, "EnsembleType", {"name": t.name})
    members = t.members
    if isinstance(members, range):
        mblock = etree.SubElement(e, "Members")
        etree.SubElement(
            mblock,
            "MemberRange",
            {"min": str(members.start), "max": str(members.stop - 1), "stride": "1"},
        )


def _write_mesh(parent: etree._Element, t: MeshType) -> None:
    e = etree.SubElement(parent, "MeshType", {"name": t.name})
    etree.SubElement(e, "Elements", {"name": t.elements.name})
    etree.SubElement(e, "Chart", {"name": t.chart.name})


def _write_evaluator(parent: etree._Element, ev: Evaluator) -> None:
    vt_name = getattr(ev.value_type, "name", "")
    if isinstance(ev, ArgumentEvaluator):
        etree.SubElement(
            parent,
            "ArgumentEvaluator",
            {"name": ev.name, "valueType": vt_name},
        )
    elif isinstance(ev, ConstantEvaluator):
        value = ev.value
        if isinstance(value, list | tuple):
            # Use repr(float(v)) to round-trip float via Python scalar
            # (avoids numpy scalar repr issues, same pattern as InlineTextBackend).
            value_str = " ".join(
                repr(float(v)) if isinstance(v, int | float) else str(v) for v in value
            )
        else:
            value_str = repr(value)
        etree.SubElement(
            parent,
            "ConstantEvaluator",
            {"name": ev.name, "valueType": vt_name, "value": value_str},
        )
    elif isinstance(ev, ExternalEvaluator):
        etree.SubElement(
            parent,
            "ExternalEvaluator",
            {"name": ev.name, "valueType": vt_name},
        )
    elif isinstance(ev, ParameterEvaluator):
        _write_parameter(parent, ev)
    elif isinstance(ev, ReferenceEvaluator):
        attrs = {"name": ev.name, "valueType": vt_name}
        if ev.source is not None:
            attrs["evaluator"] = ev.source.name
        e = etree.SubElement(parent, "ReferenceEvaluator", attrs)
        if ev.bindings is not None and len(ev.bindings) > 0:
            bs = etree.SubElement(e, "Bindings")
            for binding in ev.bindings._by_arg.values():
                etree.SubElement(
                    bs,
                    "Bind",
                    {"argument": binding.argument.name, "source": binding.source.name},
                )
    elif isinstance(ev, AggregateEvaluator):
        e = etree.SubElement(
            parent,
            "AggregateEvaluator",
            {"name": ev.name, "valueType": vt_name},
        )
        if ev.components:
            etree.SubElement(
                e,
                "ComponentEvaluators",
                {"default": ev.components[0].name},
            )
    elif isinstance(ev, PiecewiseEvaluator):
        e = etree.SubElement(
            parent,
            "PiecewiseEvaluator",
            {"name": ev.name, "valueType": vt_name},
        )
        ev_map = etree.SubElement(e, "EvaluatorMap")
        if ev.default is not None:
            ev_map.set("default", ev.default.name)
        for idx, piece in ev.pieces.items():
            etree.SubElement(
                ev_map,
                "EvaluatorMapEntry",
                {"indexValue": str(idx), "evaluator": piece.name},
            )


def _write_parameter(parent: etree._Element, ev: ParameterEvaluator) -> None:
    vt_name = getattr(ev.value_type, "name", "")
    e = etree.SubElement(
        parent,
        "ParameterEvaluator",
        {"name": ev.name, "valueType": vt_name},
    )
    wrapper = etree.SubElement(e, "DenseArrayData")
    shape_str = " ".join(str(d) for d in ev.data.shape)
    ds = etree.SubElement(
        wrapper,
        "DataSource",
        {
            "name": f"{ev.name}_src",
            "location": "/",
            "rank": str(max(1, len(ev.data.shape))),
        },
    )
    etree.SubElement(ds, "ArrayDataSize").text = shape_str

    if isinstance(ev.data, InlineTextBackend):
        etree.SubElement(wrapper, "DataResourceString").text = ev.data.text
    elif isinstance(ev.data, ExternalTextBackend):
        etree.SubElement(
            wrapper,
            "DataResourceHref",
            {
                "{http://www.w3.org/1999/xlink}href": ev.data.href,
                "format": "PLAIN_TEXT",
            },
        )
    elif isinstance(ev.data, Hdf5DenseBackend | Hdf5DOKBackend):
        href = (
            str(ev.data.path.name) if hasattr(ev.data, "path") and ev.data.path is not None else ""
        )
        etree.SubElement(
            wrapper,
            "DataResourceHref",
            {
                "{http://www.w3.org/1999/xlink}href": href,
                "format": "HDF5",
            },
        )
