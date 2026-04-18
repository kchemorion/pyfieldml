"""Tests for text-based array data backends."""

from __future__ import annotations

import numpy as np
import pytest

from pyfieldml.data.base import DataSource
from pyfieldml.data.text import InlineTextBackend


def test_inline_text_reads_float64_dense() -> None:
    backend = InlineTextBackend(
        text="1.0 2.0 3.0\n4.0 5.0 6.0\n",
        shape=(2, 3),
        dtype="float64",
    )
    arr = backend.as_ndarray()
    assert arr.dtype == np.float64
    assert arr.shape == (2, 3)
    np.testing.assert_array_equal(arr, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_inline_text_reads_int32() -> None:
    backend = InlineTextBackend(text="1 2 3 4", shape=(4,), dtype="int32")
    arr = backend.as_ndarray()
    assert arr.dtype == np.int32
    np.testing.assert_array_equal(arr, [1, 2, 3, 4])


def test_inline_text_writes_losslessly() -> None:
    original = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    backend = InlineTextBackend.from_ndarray(original)
    reloaded = backend.as_ndarray()
    np.testing.assert_array_equal(original, reloaded)


def test_inline_text_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="5 values but shape"):
        InlineTextBackend(text="1 2 3 4 5", shape=(2, 3), dtype="float64").as_ndarray()


def test_data_source_exposes_shape_and_dtype() -> None:
    backend = InlineTextBackend(text="1 2 3", shape=(3,), dtype="float64")
    src: DataSource = backend
    assert src.shape == (3,)
    assert src.dtype == np.dtype("float64")
