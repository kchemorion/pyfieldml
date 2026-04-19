"""``pyfieldml inspect`` — print a summary tree of a FieldML document."""

from __future__ import annotations

import pyfieldml as fml


def run(*, path: str) -> int:
    """Print the Document's structure: region, types, evaluators."""
    doc = fml.read(path)
    print(f"FieldML document: {path}")
    print(f"  Source version: {doc.source_version}")
    print(f"  Region:         {doc.region.name}")
    for kind, mapping in (
        ("Booleans", doc.booleans),
        ("Ensembles", doc.ensembles),
        ("Continuous", doc.continuous),
        ("Meshes", doc.meshes),
    ):
        if mapping:
            print(f"  {kind}:")
            for name in mapping:
                print(f"    - {name}")
    print(f"  Evaluators ({len(doc.evaluators)}):")
    for name, ev in doc.evaluators.items():
        print(f"    - {name}: {type(ev).__name__}")
    return 0
