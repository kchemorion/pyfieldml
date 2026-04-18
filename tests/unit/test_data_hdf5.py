"""Tests for HDF5-backed array data."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from pyfieldml.data.hdf5 import Hdf5DenseBackend


def _write_h5(path: Path, dataset_name: str, data: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=data)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_hdf5_dense_round_trip(tmp_path: Path, dtype: str) -> None:
    data = np.arange(12, dtype=dtype).reshape(3, 4)
    h5 = tmp_path / "data.h5"
    _write_h5(h5, "/values", data)
    backend = Hdf5DenseBackend(path=h5, dataset="/values")
    np.testing.assert_array_equal(backend.as_ndarray(), data)
    assert backend.dtype == np.dtype(dtype)
    assert backend.shape == (3, 4)


def test_hdf5_dense_lazy_no_load_on_construction(tmp_path: Path) -> None:
    data = np.arange(4, dtype=np.float32)
    h5 = tmp_path / "data.h5"
    _write_h5(h5, "/v", data)
    backend = Hdf5DenseBackend(path=h5, dataset="/v")
    h5b = tmp_path / "data_renamed.h5"
    h5.rename(h5b)
    backend.path = h5b
    np.testing.assert_array_equal(backend.as_ndarray(), data)


def test_hdf5_dense_write_ndarray(tmp_path: Path) -> None:
    data = np.random.default_rng(0).random((2, 3))
    backend = Hdf5DenseBackend.write_ndarray(data, path=tmp_path / "out.h5", dataset="/coords")
    with h5py.File(tmp_path / "out.h5", "r") as f:
        np.testing.assert_array_equal(f["/coords"][()], data)
    np.testing.assert_array_equal(backend.as_ndarray(), data)
