"""Reproducibility canary — fingerprint every bundled dataset byte-for-byte.

Loads each bundled ``.fieldml`` dataset, computes a SHA256 over the concatenation
of every ParameterEvaluator's raw bytes (coordinates, connectivity, derived
fields), and asserts the result against a hard-coded known-good hash.

The test doubles as a regression canary: if someone changes a bundled dataset
(re-generates the femur, re-decimates the bunny, etc.) without updating the
expected hash here, this test fails and forces a deliberate acknowledgment of
the change.

Updating the hashes
-------------------
After intentionally modifying a dataset, rerun this test; the failure message
prints the actual SHA256. Copy it into ``EXPECTED_FINGERPRINTS`` below.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pyfieldml import datasets
from pyfieldml.model.evaluators import ParameterEvaluator

# Hard-coded known-good SHA256 digests. Update deliberately when a bundled
# dataset is intentionally regenerated; never auto-update.
EXPECTED_FINGERPRINTS: dict[str, str] = {
    "bunny_stanford": "afea52247d575ef83d7ffdd20d57995101bb6f38a4a9aac2ac466bae17df6331",
    "femur": "47a2ca89aceabe335678858db061c6edad27c0f79e09a6976bfaaece608d6af1",
    "femur_bodyparts3d": "729dc198259083e6a90c3d794697c96feda7efd559a88dbb16af5a30e983db2d",
    "rectus_femoris": "a0109855616f3210e4aa1eeaef4186fa10d4752de832030a93b316e69bc2f306",
    "unit_cube": "3bb7710ffbdacbd7162f260a19a425e496970a5d3cc40d143ebdbbf79078ae15",
}


def _fingerprint(name: str) -> str:
    """Return a deterministic SHA256 over every ParameterEvaluator in the dataset.

    Evaluators are visited in sorted-name order so the hash is stable across
    insertion-order quirks. Each evaluator contributes its name, shape, dtype,
    and raw bytes (after a canonical C-contiguous float64 / int64 cast).
    """
    doc = datasets.load(name)
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))

    for ev_name in sorted(doc.region.evaluators):
        ev = doc.region.evaluators[ev_name]
        if not isinstance(ev, ParameterEvaluator):
            continue
        arr = ev.as_ndarray()
        # Normalize to a canonical representation so the fingerprint is
        # independent of loader-level dtype quirks (connectivity sometimes
        # comes back as float64).
        if np.issubdtype(arr.dtype, np.integer):
            canon = np.ascontiguousarray(arr, dtype=np.int64)
        else:
            canon = np.ascontiguousarray(arr, dtype=np.float64)
        h.update(ev_name.encode("utf-8"))
        h.update(str(canon.shape).encode("utf-8"))
        h.update(str(canon.dtype).encode("utf-8"))
        h.update(canon.tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("name", sorted(EXPECTED_FINGERPRINTS))
def test_dataset_fingerprint(name: str) -> None:
    """Dataset byte-level fingerprint must match the hard-coded value."""
    actual = _fingerprint(name)
    expected = EXPECTED_FINGERPRINTS[name]
    assert actual == expected, (
        f"Fingerprint changed for {name!r}.\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"If this change is intentional, update EXPECTED_FINGERPRINTS."
    )
