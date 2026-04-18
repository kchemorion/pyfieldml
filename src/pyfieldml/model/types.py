"""FieldML type classes: Ensemble, Continuous, Mesh, Boolean."""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BooleanType:
    """Boolean type — a two-valued domain."""

    name: str


@dataclass(frozen=True)
class EnsembleType:
    """A finite ordered set of positive integers (element IDs, node IDs, etc.)."""

    name: str
    members: Set[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.members, frozenset | range):
            object.__setattr__(self, "members", frozenset(self.members))

    @property
    def size(self) -> int:
        if isinstance(self.members, range):
            return len(self.members)
        return len(self.members)


@dataclass(frozen=True)
class ContinuousType:
    """A continuous (real-valued) type, scalar or with named components."""

    name: str
    component_name: str | None = None
    component_count: int = 1

    def __post_init__(self) -> None:
        if self.component_count < 1:
            raise ValueError(
                f"ContinuousType {self.name!r}: component_count must be at least 1, "
                f"got {self.component_count}"
            )

    @property
    def is_vector(self) -> bool:
        return self.component_count > 1


@dataclass(frozen=True)
class MeshType:
    """A mesh type — an ensemble of elements paired with a continuous chart (xi)."""

    name: str
    elements: EnsembleType
    chart: ContinuousType

    @property
    def dimension(self) -> int:
        return self.chart.component_count

    @property
    def element_count(self) -> int:
        return self.elements.size
