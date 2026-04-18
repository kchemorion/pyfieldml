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
