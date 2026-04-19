"""``pyfieldml diff`` — semantic diff between two FieldML documents."""

from __future__ import annotations

import pyfieldml as fml
from pyfieldml.validation.diff import diff_documents


def run(*, a: str, b: str) -> int:
    """Print differences. Exit 0 if identical, 1 otherwise."""
    doc_a = fml.read(a)
    doc_b = fml.read(b)
    d = diff_documents(doc_a, doc_b)
    if not d.has_differences():
        print(f"identical: {a} vs {b}")
        return 0
    print(f"{len(d)} difference(s) between {a} and {b}:")
    for entry in d.entries:
        line = f"  [{entry.kind}] "
        if entry.name:
            line += f"{entry.name}: "
        line += entry.message
        print(line)
    return 1
