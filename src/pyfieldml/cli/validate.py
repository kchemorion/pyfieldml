"""``pyfieldml validate`` — XSD-validate a FieldML document."""

from __future__ import annotations

import pyfieldml as fml
from pyfieldml.errors import FieldMLError


def run(*, path: str, strict: bool = False) -> int:
    """Validate ``path``. Prints ``OK`` on success, errors on failure."""
    try:
        fml.validate(path)
    except FieldMLError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"OK: {path} passes XSD validation")
    return 0
