"""Cross-validate pyfieldml against the C++ FieldML-API reference library.

Skipped unless PYFIELDML_CPP_REF points to a `cpp_roundtrip` executable that
takes (input.fieldml, output.fieldml) and exits 0 on success.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from lxml import etree

import pyfieldml as fml

CPP_REF = os.environ.get("PYFIELDML_CPP_REF")


@pytest.mark.skipif(
    CPP_REF is None,
    reason="PYFIELDML_CPP_REF not set; skipping conformance run",
)
@pytest.mark.parametrize(
    "fixture",
    [
        "minimal.fieldml",
        "two_types.fieldml",
    ],
)
def test_round_trip_matches_cpp_reference(fixtures_dir: Path, tmp_path: Path, fixture: str) -> None:
    assert CPP_REF is not None
    src = fixtures_dir / fixture

    # pyfieldml round-trip
    py_out = tmp_path / f"py_{fixture}"
    fml.read(src).write(py_out)

    # C++ round-trip
    cpp_out = tmp_path / f"cpp_{fixture}"
    subprocess.run([CPP_REF, str(src), str(cpp_out)], check=True)

    # Semantic equivalence: C14N canonicalization makes the comparison
    # whitespace/ordering-insensitive where possible.
    py_canonical = etree.tostring(
        etree.parse(py_out),
        method="c14n",
        exclusive=True,
    )
    cpp_canonical = etree.tostring(
        etree.parse(cpp_out),
        method="c14n",
        exclusive=True,
    )
    assert py_canonical == cpp_canonical, (
        f"Canonicalized XML divergence for {fixture}:\n"
        f"python:\n{py_canonical!r}\ncpp:\n{cpp_canonical!r}"
    )


@pytest.mark.skipif(
    CPP_REF is None,
    reason="PYFIELDML_CPP_REF not set; skipping conformance run",
)
@pytest.mark.parametrize(
    "fixture",
    ["parameter_inline.fieldml"],
)
def test_parameter_array_equality_with_cpp_reference(
    fixtures_dir: Path, tmp_path: Path, fixture: str
) -> None:
    from pyfieldml.model.evaluators import ParameterEvaluator

    assert CPP_REF is not None
    src = fixtures_dir / fixture
    cpp_out = tmp_path / f"cpp_{fixture}"
    subprocess.run([CPP_REF, str(src), str(cpp_out)], check=True)

    py_doc = fml.read(src)
    cpp_doc = fml.read(cpp_out)

    for name, ev in py_doc.evaluators.items():
        if isinstance(ev, ParameterEvaluator):
            py_arr = ev.as_ndarray()
            cpp_ev = cpp_doc.evaluators.get(name)
            assert cpp_ev is not None, f"{name!r} missing from C++ round-trip output"
            assert isinstance(cpp_ev, ParameterEvaluator)
            np.testing.assert_array_equal(py_arr, cpp_ev.as_ndarray(), err_msg=name)
