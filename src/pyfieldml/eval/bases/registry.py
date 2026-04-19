"""Runtime registry for basis functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pyfieldml.errors import BasisNotFoundError
from pyfieldml.eval.bases.base import Basis

_registry: dict[str, Basis] = {}

T = TypeVar("T")


def basis(name: str, *, topology: str, order: int) -> Callable[[type[T]], type[T]]:
    """Class decorator: register a basis-function implementation under ``name``."""

    def wrap(cls: type[T]) -> type[T]:
        instance = cls()
        instance.topology = topology  # type: ignore[attr-defined]
        instance.order = order  # type: ignore[attr-defined]
        _registry[name] = instance  # type: ignore[assignment]
        return cls

    return wrap


def get_basis(name: str) -> Basis:
    if name not in _registry:
        raise BasisNotFoundError(f"No basis registered under {name!r}")
    return _registry[name]


def list_bases() -> list[str]:
    return sorted(_registry.keys())


def _clear_for_tests() -> None:
    """Reset the registry — tests only."""
    _registry.clear()
