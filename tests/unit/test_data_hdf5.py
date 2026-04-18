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


def test_hdf5_dok_round_trip(tmp_path: Path) -> None:
    from pyfieldml.data.hdf5 import Hdf5DOKBackend

    h5 = tmp_path / "sparse.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("/sparse")
        g.create_dataset("values", data=np.array([1.5, 2.5, 3.5], dtype=np.float64))
        g.create_dataset("indexes", data=np.array([[0, 1], [2, 2], [3, 0]], dtype=np.int64))
        g.attrs["shape"] = (4, 4)
    backend = Hdf5DOKBackend(path=h5, group="/sparse")
    dense = backend.as_ndarray()
    expected = np.zeros((4, 4), dtype=np.float64)
    expected[0, 1] = 1.5
    expected[2, 2] = 2.5
    expected[3, 0] = 3.5
    np.testing.assert_array_equal(dense, expected)


def test_hdf5_dok_write_ndarray(tmp_path: Path) -> None:
    from pyfieldml.data.hdf5 import Hdf5DOKBackend

    dense = np.zeros((3, 3), dtype=np.float64)
    dense[0, 1] = 7.0
    dense[2, 0] = 9.0
    backend = Hdf5DOKBackend.write_ndarray(dense, path=tmp_path / "out.h5", group="/s")
    np.testing.assert_array_equal(backend.as_ndarray(), dense)
