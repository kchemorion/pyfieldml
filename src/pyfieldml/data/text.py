"""Text-based array-data backends."""

from __future__ import annotations

import numpy as np


class InlineTextBackend:
    """Whitespace-separated values embedded in the FieldML XML."""

    text: str
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(
        self,
        text: str,
        *,
        shape: tuple[int, ...],
        dtype: str | np.dtype,
    ) -> None:
        self.text = text
        self.shape = tuple(int(d) for d in shape)
        self.dtype = np.dtype(dtype)

    def as_ndarray(self) -> np.ndarray:
        values = np.array(self.text.split(), dtype=self.dtype)
        expected = 1
        for d in self.shape:
            expected *= d
        if values.size != expected:
            raise ValueError(
                f"InlineTextBackend: parsed {values.size} values but shape "
                f"{self.shape} expects {expected}"
            )
        return values.reshape(self.shape)

    @classmethod
    def from_ndarray(cls, arr: np.ndarray) -> InlineTextBackend:
        """Round-trip an existing ndarray to text form (used by the writer)."""
        flat = arr.ravel()
        if np.issubdtype(arr.dtype, np.floating):
            text = " ".join(repr(float(v)) for v in flat)
        else:
            text = " ".join(str(v.item()) for v in flat)
        return cls(text=text, shape=arr.shape, dtype=arr.dtype)
