"""Tests for FieldML type classes."""

from __future__ import annotations

import pytest

from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)


def test_boolean_type_is_hashable_by_name() -> None:
    t1 = BooleanType(name="boolean")
    t2 = BooleanType(name="boolean")
    assert t1 == t2
    assert hash(t1) == hash(t2)


def test_ensemble_type_records_members() -> None:
    t = EnsembleType(name="nodes", members=range(1, 11))  # type: ignore[arg-type]
    assert t.size == 10
    assert 5 in t.members
    assert 11 not in t.members


def test_continuous_type_with_components_stores_arity() -> None:
    t = ContinuousType(name="real.3d", component_name="real.3d.component", component_count=3)
    assert t.component_count == 3
    assert t.is_vector
    scalar = ContinuousType(name="real.1d")
    assert scalar.component_count == 1
    assert not scalar.is_vector


def test_continuous_type_rejects_zero_components() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ContinuousType(name="bad", component_name="c", component_count=0)


def test_mesh_type_combines_elements_and_chart() -> None:
    elements = EnsembleType(name="elems", members=range(1, 5))  # type: ignore[arg-type]
    chart = ContinuousType(name="xi.3d", component_name="xi.3d.c", component_count=3)
    mesh = MeshType(name="mesh", elements=elements, chart=chart)
    assert mesh.dimension == 3
    assert mesh.element_count == 4
