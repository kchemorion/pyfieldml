"""Semantic linter over a loaded Document.

Reports issues the XSD cannot catch:
- Orphan evaluators (defined but never referenced).
- Inverted elements (negative Jacobian at centroid).
- Future: unbound arguments, cyclic references, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LintIssue:
    """A single lint issue surfaced by ``lint_document``.

    Attributes
    ----------
    code
        Short identifier, e.g. ``"ORPHAN"``, ``"INVERTED_ELEMENT"``.
    severity
        ``"error"`` | ``"warning"`` | ``"info"``.
    message
        Human-readable description.
    object_name
        Name of the offending FieldML object (if any).

    """

    code: str
    severity: str
    message: str
    object_name: str | None = None


def lint_document(doc: Any) -> list[LintIssue]:
    """Return all lint issues found in ``doc``."""
    issues: list[LintIssue] = []
    issues.extend(_check_orphans(doc))
    issues.extend(_check_inverted_elements(doc))
    return issues


def _check_orphans(doc: Any) -> list[LintIssue]:
    """Find evaluators defined but not referenced by any ReferenceEvaluator or similar."""
    from pyfieldml.model.evaluators import ReferenceEvaluator

    region = doc.region
    referenced: set[str] = set()
    for _name, ev in region.evaluators.items():
        if isinstance(ev, ReferenceEvaluator) and ev.source is not None:
            referenced.add(ev.source.name)

    terminal_kinds = {
        "ParameterEvaluator",
        "ConstantEvaluator",
        "ArgumentEvaluator",
        "ExternalEvaluator",
    }
    issues: list[LintIssue] = []
    for name, ev in region.evaluators.items():
        kind = type(ev).__name__
        if kind in terminal_kinds:
            continue
        if name not in referenced:
            issues.append(
                LintIssue(
                    code="ORPHAN",
                    severity="warning",
                    message=f"Evaluator {name!r} is defined but never referenced",
                    object_name=name,
                )
            )
    return issues


def _check_inverted_elements(doc: Any) -> list[LintIssue]:
    """Coord-field evaluators whose Jacobian has negative determinant at centroid."""
    from pyfieldml.eval.field import Field

    region = doc.region
    issues: list[LintIssue] = []

    centroid_xi_per_topology = {
        "line": np.array([0.5]),
        "triangle": np.array([1 / 3, 1 / 3]),
        "quad": np.array([0.5, 0.5]),
        "tet": np.array([1 / 4, 1 / 4, 1 / 4]),
        "hex": np.array([0.5, 0.5, 0.5]),
        "wedge": np.array([1 / 3, 1 / 3, 0.5]),
    }

    for name, _ev in region.evaluators.items():
        try:
            f = doc.field(name)
        except Exception:
            continue
        if not isinstance(f, Field):
            continue
        xi = centroid_xi_per_topology.get(f._basis.topology)
        if xi is None:
            continue
        n_elems = f._conn.shape[0]
        for e in range(1, n_elems + 1):
            try:
                j_mat = f.jacobian(element=e, xi=xi)
            except Exception:
                continue
            # Need square to compute determinant; skip non-square.
            if j_mat.ndim != 2 or j_mat.shape[0] != j_mat.shape[1]:
                continue
            det = float(np.linalg.det(j_mat))
            if det <= 0.0:
                issues.append(
                    LintIssue(
                        code="INVERTED_ELEMENT",
                        severity="error",
                        message=(
                            f"Element {e} of field {name!r} has non-positive Jacobian det={det:.4g}"
                        ),
                        object_name=name,
                    )
                )
    return issues
