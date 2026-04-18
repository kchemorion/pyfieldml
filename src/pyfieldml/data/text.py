"""Text-based array-data backends."""

from __future__ import annotations

from pathlib import Path

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


class ExternalTextBackend:
    """Whitespace-separated values in an external text file."""

    base_dir: Path
    href: str
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(
        self,
        *,
        base_dir: str | Path,
        href: str,
        shape: tuple[int, ...],
        dtype: str | np.dtype,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.href = href
        self.shape = tuple(int(d) for d in shape)
        self.dtype = np.dtype(dtype)

    @property
    def path(self) -> Path:
        return self.base_dir / self.href

    def as_ndarray(self) -> np.ndarray:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        values = np.loadtxt(self.path, dtype=self.dtype)
        return values.reshape(self.shape)

    @classmethod
    def write_ndarray(
        cls,
        arr: np.ndarray,
        *,
        base_dir: str | Path,
        href: str,
    ) -> ExternalTextBackend:
        target = Path(base_dir) / href
        target.parent.mkdir(parents=True, exist_ok=True)
        fmt = "%.17g" if np.issubdtype(arr.dtype, np.floating) else "%d"
        np.savetxt(target, arr.reshape(-1), fmt=fmt)
        return cls(base_dir=base_dir, href=href, shape=arr.shape, dtype=arr.dtype)
