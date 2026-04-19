"""``pyfieldml convert`` — convert a FieldML document to another format (via meshio)."""

from __future__ import annotations

import pyfieldml as fml


def run(*, path: str, to_format: str, out: str) -> int:
    """Convert ``path`` to ``to_format`` and write to ``out``."""
    try:
        import meshio
    except ImportError:
        print(
            "ERROR: `pyfieldml convert` requires the [meshio] extra: pip install pyfieldml[meshio]"
        )
        return 1

    doc = fml.read(path)
    mesh = doc.to_meshio()
    meshio.write(out, mesh, file_format=to_format)
    print(f"wrote {out} (format={to_format})")
    return 0
