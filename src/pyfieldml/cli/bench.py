"""`pyfieldml bench` — measure field evaluation throughput."""

from __future__ import annotations

import time

import numpy as np

import pyfieldml as fml


def run(*, path: str, field: str, n: int) -> int:
    """Benchmark ``field.evaluate`` throughput on the named field."""
    doc = fml.read(path)
    f = doc.field(field)
    rng = np.random.default_rng(0)
    xi = rng.random((n, 3))
    elements = np.ones(n, dtype=np.int64)

    t0 = time.perf_counter()
    f.evaluate(element=elements, xi=xi)
    dt = time.perf_counter() - t0
    print(f"evaluate: {n:,} points in {dt * 1000:.2f} ms ({n / dt:.0f} pts/s)")
    return 0
