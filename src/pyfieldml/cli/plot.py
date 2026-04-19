"""``pyfieldml plot`` — render a FieldML document via PyVista."""

from __future__ import annotations

import pyfieldml as fml


def run(*, path: str, field: str | None = None) -> int:
    """Render the document (or a specific field) in an interactive viewer."""
    try:
        import pyvista  # noqa: F401
    except ImportError:
        print("ERROR: `pyfieldml plot` requires the [viz] extra: pip install pyfieldml[viz]")
        return 1
    doc = fml.read(path)
    # If a specific field is requested, try to plot it; else plot the document.
    if field is None:
        doc.plot()
    else:
        f = doc.field(field)
        f.plot(doc=doc)
    return 0
