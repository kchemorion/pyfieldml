"""Tests for biomechanics high-level builders."""

from __future__ import annotations

import numpy as np

import pyfieldml as fml
from pyfieldml.builders.biomech import (
    add_fiber_field,
    add_landmark_set,
    add_material_field,
    add_scalar_field,
    add_vector_field,
)
from pyfieldml.builders.mesh import add_lagrange_mesh
from pyfieldml.model.evaluators import ParameterEvaluator
from pyfieldml.model.region import Region


def _unit_cube_region() -> Region:
    r = Region(name="cube")
    nodes = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    connectivity = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    add_lagrange_mesh(
        r,
        name="m",
        nodes=nodes,
        elements=connectivity,
        topology="hex",
        order=1,
        coord_name="coords",
    )
    return r


def test_add_scalar_field_round_trips() -> None:
    r = _unit_cube_region()
    add_scalar_field(r, name="temperature", mesh_name="m", values=np.arange(8.0))
    doc = fml.Document.from_region(r)
    ev = doc.evaluators["temperature"]
    assert isinstance(ev, ParameterEvaluator)
    np.testing.assert_array_equal(ev.as_ndarray(), np.arange(8.0))


def test_add_vector_field_preserves_shape() -> None:
    r = _unit_cube_region()
    vals = np.random.default_rng(0).random((8, 3))
    add_vector_field(r, name="velocity", mesh_name="m", values=vals)
    doc = fml.Document.from_region(r)
    ev = doc.evaluators["velocity"]
    assert isinstance(ev, ParameterEvaluator)
    arr = ev.as_ndarray()
    assert arr.shape == (8, 3)
    np.testing.assert_array_equal(arr, vals)


def test_add_fiber_field_alias() -> None:
    """add_fiber_field uses default name 'fiber_direction' and is a vector field."""
    r = _unit_cube_region()
    fibers = np.zeros((8, 3))
    fibers[:, 0] = 1.0
    add_fiber_field(r, mesh_name="m", vectors=fibers)
    doc = fml.Document.from_region(r)
    ev = doc.evaluators["fiber_direction"]
    assert isinstance(ev, ParameterEvaluator)
    np.testing.assert_array_equal(ev.as_ndarray(), fibers)


def test_add_material_field_per_element() -> None:
    r = _unit_cube_region()
    add_material_field(r, mesh_name="m", values_per_element=np.array([1.5e9]))
    doc = fml.Document.from_region(r)
    ev = doc.evaluators["material"]
    assert isinstance(ev, ParameterEvaluator)
    np.testing.assert_array_equal(ev.as_ndarray(), [1.5e9])


def test_add_landmark_set_stores_points_and_labels() -> None:
    r = _unit_cube_region()
    points = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]])
    labels = ["centroid", "corner_probe"]
    add_landmark_set(r, name="anatomical_points", points=points, labels=labels)
    doc = fml.Document.from_region(r)
    ev = doc.evaluators["anatomical_points"]
    assert isinstance(ev, ParameterEvaluator)
    arr = ev.as_ndarray()
    np.testing.assert_array_equal(arr, points)
    # Labels are stored on the Region as auxiliary metadata.
    assert doc.region._landmark_labels["anatomical_points"] == labels  # type: ignore[attr-defined]
