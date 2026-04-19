"""Standalone runner for the pyfieldml conformance test suite.

Usage:
    python runner.py --fixtures ./fixtures --reader-writer <executable>

where <executable> takes (input_path, output_path) and round-trips a FieldML file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lxml import etree


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument(
        "--reader-writer",
        required=True,
        help="executable that takes (src, dst) and round-trips",
    )
    args = p.parse_args()

    failures = []
    fixtures = sorted(args.fixtures.glob("*.fieldml"))
    for src in fixtures:
        out = src.with_suffix(".roundtrip.fieldml")
        subprocess.run([args.reader_writer, str(src), str(out)], check=True)
        if not _canonical_equivalent(src, out):
            failures.append(src.name)
        out.unlink(missing_ok=True)
    if failures:
        print(f"FAIL: {len(failures)} fixtures round-tripped unequally:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK: {len(fixtures)} fixtures round-tripped.")
    return 0


def _canonical_equivalent(a: Path, b: Path) -> bool:
    ca = etree.tostring(etree.parse(a), method="c14n", exclusive=True)
    cb = etree.tostring(etree.parse(b), method="c14n", exclusive=True)
    return ca == cb


if __name__ == "__main__":
    sys.exit(main())
