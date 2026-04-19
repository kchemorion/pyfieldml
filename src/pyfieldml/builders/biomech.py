"""Biomechanics-flavored high-level builders over add_lagrange_mesh."""

from __future__ import annotations

import numpy as np

from pyfieldml.data.text import InlineTextBackend
from pyfieldml.model.evaluators import ParameterEvaluator
from pyfieldml.model.region import Region
from pyfieldml.model.types import ContinuousType


def add_scalar_field(
    region: Region,
    *,
    name: str,
    mesh_name: str,
    values: np.ndarray,
) -> ParameterEvaluator:
    """Add a per-node scalar field.

    ``values`` has shape ``(n_nodes,)``. ``mesh_name`` is the MeshType name
    (currently used only to disambiguate; no validation in Phase 5).
    """
    vt = ContinuousType(name=f"{name}.value_type")
    region.add_type(vt)
    data = InlineTextBackend.from_ndarray(np.asarray(values, dtype=np.float64))
    ev = ParameterEvaluator(name=name, value_type=vt, data=data)
    region.add_evaluator(ev)
    return ev


def add_vector_field(
    region: Region,
    *,
    name: str,
    mesh_name: str,
    values: np.ndarray,
) -> ParameterEvaluator:
    """Add a per-node vector field, shape ``(n_nodes, D)``."""
    values = np.asarray(values, dtype=np.float64)
    d = values.shape[1]
    vt = ContinuousType(
        name=f"{name}.value_type",
        component_name=f"{name}.component",
        component_count=d,
    )
    region.add_type(vt)
    data = InlineTextBackend.from_ndarray(values)
    ev = ParameterEvaluator(name=name, value_type=vt, data=data)
    region.add_evaluator(ev)
    return ev


def add_fiber_field(
    region: Region,
    *,
    mesh_name: str,
    vectors: np.ndarray,
    name: str = "fiber_direction",
) -> ParameterEvaluator:
    """Vector field conventionally representing muscle/myocyte fiber directions."""
    return add_vector_field(region, name=name, mesh_name=mesh_name, values=vectors)


def add_material_field(
    region: Region,
    *,
    mesh_name: str,
    values_per_element: np.ndarray,
    name: str = "material",
) -> ParameterEvaluator:
    """Per-element scalar field (e.g. Young's modulus, density)."""
    return add_scalar_field(region, name=name, mesh_name=mesh_name, values=values_per_element)


def add_landmark_set(
    region: Region,
    *,
    name: str,
    points: np.ndarray,
    labels: list[str] | None = None,
) -> ParameterEvaluator:
    """Add a named set of anatomical landmarks: (N, 3) points + optional labels.

    Labels are stored on the Region as auxiliary metadata (``_landmark_labels``)
    until Phase-6 extends the XML writer to emit them as a FieldML extension.
    """
    points = np.asarray(points, dtype=np.float64)
    d = points.shape[1]
    vt = ContinuousType(
        name=f"{name}.value_type",
        component_name=f"{name}.component",
        component_count=d,
    )
    region.add_type(vt)
    data = InlineTextBackend.from_ndarray(points)
    ev = ParameterEvaluator(name=name, value_type=vt, data=data)
    region.add_evaluator(ev)
    if labels is not None:
        if not hasattr(region, "_landmark_labels"):
            region._landmark_labels = {}  # type: ignore[attr-defined]
        region._landmark_labels[name] = list(labels)  # type: ignore[attr-defined]
    return ev
