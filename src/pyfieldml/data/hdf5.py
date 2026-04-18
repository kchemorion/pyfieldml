"""HDF5-backed array data (dense and DOK sparse)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class Hdf5DenseBackend:
    """Dense HDF5 dataset-backed array. Lazy — only reads on ``as_ndarray``."""

    def __init__(self, *, path: str | Path, dataset: str) -> None:
        self.path = Path(path)
        self.dataset = dataset
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None

    def _probe(self) -> None:
        with h5py.File(self.path, "r") as f:
            ds = f[self.dataset]
            self._shape = tuple(int(d) for d in ds.shape)
            self._dtype = np.dtype(ds.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._probe()
        assert self._shape is not None
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self._probe()
        assert self._dtype is not None
        return self._dtype

    def as_ndarray(self) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            data = f[self.dataset][()]
        return np.asarray(data)

    @classmethod
    def write_ndarray(cls, arr: np.ndarray, *, path: str | Path, dataset: str) -> Hdf5DenseBackend:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "a") as f:
            if dataset in f:
                del f[dataset]
            f.create_dataset(dataset, data=arr)
        return cls(path=path, dataset=dataset)


class Hdf5DOKBackend:
    """Sparse (DOK) HDF5 array. Materializes a dense ndarray on demand."""

    def __init__(self, *, path: str | Path, group: str) -> None:
        self.path = Path(path)
        self.group = group
        self._shape: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None

    def _probe(self) -> None:
        with h5py.File(self.path, "r") as f:
            g = f[self.group]
            self._shape = tuple(int(d) for d in g.attrs["shape"])
            self._dtype = np.dtype(g["values"].dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._probe()
        assert self._shape is not None
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self._probe()
        assert self._dtype is not None
        return self._dtype

    def as_ndarray(self) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            g = f[self.group]
            values = g["values"][()]
            indexes = g["indexes"][()]
            shape = tuple(int(d) for d in g.attrs["shape"])
        dense = np.zeros(shape, dtype=values.dtype)
        dense[tuple(indexes.T)] = values
        return dense

    @classmethod
    def write_ndarray(cls, arr: np.ndarray, *, path: str | Path, group: str) -> Hdf5DOKBackend:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nonzero = np.argwhere(arr != 0)
        values = arr[tuple(nonzero.T)] if nonzero.size else np.empty((0,), dtype=arr.dtype)
        with h5py.File(path, "a") as f:
            if group in f:
                del f[group]
            g = f.create_group(group)
            g.create_dataset("values", data=values)
            g.create_dataset("indexes", data=nonzero.astype(np.int64))
            g.attrs["shape"] = np.asarray(arr.shape, dtype=np.int64)
        return cls(path=path, group=group)
