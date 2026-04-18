"""Evaluation-time scope: binding tables, element/xi, free-variable map."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyfieldml.model.bindings import BindingTable


@dataclass
class EvaluationContext:
    """Scope carried through a single evaluation call."""

    bindings: list[BindingTable] = field(default_factory=list)
    element: int | None = None
    xi: np.ndarray | None = None
