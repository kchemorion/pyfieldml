"""Base abstractions for FieldML array-data backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class DataResource:
    """A storage resource description.

    Attributes
    ----------
    format
        One of ``"text"``, ``"PLAIN_TEXT"``, ``"HDF5"``, etc. Matches FieldML
        vocabulary.
    href
        External file path (relative or absolute). ``None`` for inline resources.

    """

    format: str
    href: str | None = None


@runtime_checkable
class DataSource(Protocol):
    """Protocol any array-data backend implements."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...

    def as_ndarray(self) -> np.ndarray: ...
