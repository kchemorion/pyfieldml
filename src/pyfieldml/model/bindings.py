"""Argument bindings for ReferenceEvaluator call-sites."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pyfieldml.errors import UnboundArgumentError
from pyfieldml.model.evaluators import ArgumentEvaluator, Evaluator


@dataclass(frozen=True)
class ArgumentBinding:
    """Binds a formal ``ArgumentEvaluator`` to a concrete source evaluator."""

    argument: ArgumentEvaluator
    source: Evaluator


class BindingTable:
    """Ordered collection of ArgumentBindings, resolved by argument identity."""

    def __init__(self, bindings: Iterable[ArgumentBinding]) -> None:
        self._by_arg = {id(b.argument): b for b in bindings}

    def resolve(self, arg: ArgumentEvaluator) -> Evaluator:
        b = self._by_arg.get(id(arg))
        if b is None:
            raise UnboundArgumentError(f"No binding for ArgumentEvaluator {arg.name!r}")
        return b.source

    def __len__(self) -> int:
        return len(self._by_arg)
