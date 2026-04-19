"""Basis-function registry and protocol."""

# Importing ``lagrange`` triggers built-in basis registration
# (Task 2+ will populate it).
from pyfieldml.eval.bases import (
    hermite,  # noqa: F401
    lagrange,  # noqa: F401
)
from pyfieldml.eval.bases.base import Basis
from pyfieldml.eval.bases.registry import basis, get_basis, list_bases

__all__ = ["Basis", "basis", "get_basis", "list_bases"]
