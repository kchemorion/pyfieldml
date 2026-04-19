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
    "hip_bone_left": "f9dbec3dca67fb918b59351dc74a49ae62c57c611817bc701218338f1f26cbbe",
    "rectus_femoris": "56e367a9501905e48126ab9ad429c8ce231f9fcd3362bf65faff6c69b6e1c557",
    "scapula": "e98878650dbd1ae03c79df0126b1c78dad9742792cf049b790c0892f9ecdfede",
    "skull": "b37d3ee6464fa041d2efabeaf50690c2385501f7b3ae387d4cd7b9ab4c30b249",
    "tibia_left": "32a4bfe9f8039678dc19ab2e67b6a3ebf1de6265ebc25822e17c796ca529fea8",
    "unit_cube": "3bb7710ffbdacbd7162f260a19a425e496970a5d3cc40d143ebdbbf79078ae15",
    "vertebra_l3": "fc301286bffe9f33db65c6e51670fd3f2023061a08f39354d645561736f7d662",
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
