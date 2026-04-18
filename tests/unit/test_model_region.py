"""Tests for the Region container."""

from __future__ import annotations

import pytest

from pyfieldml.model.evaluators import ConstantEvaluator
from pyfieldml.model.region import Region
from pyfieldml.model.types import BooleanType, ContinuousType


def test_region_stores_types_and_evaluators() -> None:
    r = Region(name="test")
    bt = BooleanType(name="boolean")
    r.add_type(bt)
    ct = ContinuousType(name="real.1d")
    r.add_type(ct)
    k = ConstantEvaluator(name="k", value_type=ct, value=1.0)
    r.add_evaluator(k)
    assert r.booleans["boolean"] is bt
    assert r.continuous["real.1d"] is ct
    assert r.evaluators["k"] is k


def test_region_rejects_duplicate_names() -> None:
    r = Region(name="test")
    r.add_type(BooleanType(name="dup"))
    with pytest.raises(ValueError, match="dup"):
        r.add_type(ContinuousType(name="dup"))


def test_region_objects_view_is_ordered() -> None:
    r = Region(name="t")
    r.add_type(BooleanType(name="first"))
    r.add_type(ContinuousType(name="second"))
    assert list(r.objects) == ["first", "second"]
