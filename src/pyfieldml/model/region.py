"""Region — a namespace container for FieldML objects."""

from __future__ import annotations

from collections.abc import Iterable

from pyfieldml.model.evaluators import Evaluator
from pyfieldml.model.types import (
    BooleanType,
    ContinuousType,
    EnsembleType,
    MeshType,
)


class Region:
    """A named namespace for FieldML types, evaluators, and imports.

    Objects are stored in insertion order. Names must be unique across all
    object kinds within a region.
    """

    def __init__(self, *, name: str) -> None:
        self.name = name
        self._objects: dict[str, object] = {}
        self.booleans: dict[str, BooleanType] = {}
        self.ensembles: dict[str, EnsembleType] = {}
        self.continuous: dict[str, ContinuousType] = {}
        self.meshes: dict[str, MeshType] = {}
        self.evaluators: dict[str, Evaluator] = {}

    @property
    def objects(self) -> Iterable[str]:
        return self._objects.keys()

    def add_type(self, t: BooleanType | EnsembleType | ContinuousType | MeshType) -> None:
        self._ensure_unique(t.name)
        self._objects[t.name] = t
        if isinstance(t, BooleanType):
            self.booleans[t.name] = t
        elif isinstance(t, EnsembleType):
            self.ensembles[t.name] = t
        elif isinstance(t, ContinuousType):
            self.continuous[t.name] = t
        elif isinstance(t, MeshType):
            self.meshes[t.name] = t

    def add_evaluator(self, e: Evaluator) -> None:
        self._ensure_unique(e.name)
        self._objects[e.name] = e
        self.evaluators[e.name] = e

    def _ensure_unique(self, name: str) -> None:
        if name in self._objects:
            raise ValueError(f"Region {self.name!r}: duplicate object name {name!r}")
