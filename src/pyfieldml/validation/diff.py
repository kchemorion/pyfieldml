"""Semantic diff between two FieldML documents.

Not an XML diff — compares the evaluator-graph structure and parameter
array values. Useful for answering "did the mesh change?" or "did the
material field differ between these two models?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiffEntry:
    """One difference between two Documents.

    Attributes
    ----------
    kind
        Short identifier, e.g. ``"missing_types"``, ``"parameter_values_differ"``.
    message
        Human-readable description.
    name
        Offending object name (if any).

    """

    kind: str
    message: str
    name: str | None = None


@dataclass
class Diff:
    """Collection of differences between two Documents."""

    entries: list[DiffEntry] = field(default_factory=list)

    def has_differences(self) -> bool:
        """Return True if any entries are recorded."""
        return bool(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def diff_documents(a: Any, b: Any) -> Diff:
    """Return a :class:`Diff` describing how ``a`` differs from ``b``."""
    out = Diff()
    _diff_type_sets(a, b, "booleans", out)
    _diff_type_sets(a, b, "ensembles", out)
    _diff_type_sets(a, b, "continuous", out)
    _diff_type_sets(a, b, "meshes", out)
    _diff_evaluators(a, b, out)
    _diff_parameter_values(a, b, out)
    return out


def _diff_type_sets(a: Any, b: Any, kind: str, out: Diff) -> None:
    sa = set(getattr(a, kind).keys())
    sb = set(getattr(b, kind).keys())
    for name in sorted(sa - sb):
        out.entries.append(
            DiffEntry(
                kind=f"missing_{kind}",
                message=f"In A but not B: {name!r}",
                name=name,
            )
        )
    for name in sorted(sb - sa):
        out.entries.append(
            DiffEntry(
                kind=f"missing_{kind}",
                message=f"In B but not A: {name!r}",
                name=name,
            )
        )


def _diff_evaluators(a: Any, b: Any, out: Diff) -> None:
    sa = set(a.evaluators.keys())
    sb = set(b.evaluators.keys())
    for name in sorted(sa & sb):
        ka, kb = type(a.evaluators[name]).__name__, type(b.evaluators[name]).__name__
        if ka != kb:
            out.entries.append(
                DiffEntry(
                    kind="evaluator_kind_changed",
                    message=f"Evaluator {name!r}: {ka} → {kb}",
                    name=name,
                )
            )
    for name in sorted(sa - sb):
        out.entries.append(
            DiffEntry(kind="missing_evaluator", message=f"In A only: {name!r}", name=name)
        )
    for name in sorted(sb - sa):
        out.entries.append(
            DiffEntry(kind="missing_evaluator", message=f"In B only: {name!r}", name=name)
        )


def _diff_parameter_values(a: Any, b: Any, out: Diff) -> None:
    from pyfieldml.model.evaluators import ParameterEvaluator

    for name in sorted(set(a.evaluators) & set(b.evaluators)):
        ea, eb = a.evaluators[name], b.evaluators[name]
        if isinstance(ea, ParameterEvaluator) and isinstance(eb, ParameterEvaluator):
            arr_a, arr_b = ea.as_ndarray(), eb.as_ndarray()
            if arr_a.shape != arr_b.shape:
                out.entries.append(
                    DiffEntry(
                        kind="parameter_shape_differs",
                        message=(f"Parameter {name!r}: shape {arr_a.shape} vs {arr_b.shape}"),
                        name=name,
                    )
                )
            elif not np.array_equal(arr_a, arr_b):
                out.entries.append(
                    DiffEntry(
                        kind="parameter_values_differ",
                        message=f"Parameter {name!r}: values differ",
                        name=name,
                    )
                )
