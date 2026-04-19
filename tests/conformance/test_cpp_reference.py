"""Cross-validate pyfieldml against the C++ FieldML-API reference library.

The ``test_round_trip_matches_cpp_reference`` / ``test_parameter_array_...``
tests below are gated by ``PYFIELDML_CPP_REF``: they run only when that env
var points to a ``cpp_roundtrip`` executable that takes
``(input.fieldml, output.fieldml)`` and exits 0 on success.

A third test — ``test_python_writer_self_roundtrip_preserves_model`` — runs
unconditionally and asserts that ``Document.write`` followed by
``pyfieldml.read`` round-trips the semantic model of each fixture. This is a
weaker gate than the C++ comparison but it catches writer regressions even
when the C++ reference library isn't available locally.
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


def _canonicalize_for_cpp_compare(xml_bytes: bytes) -> bytes:
    """C14N canonicalize, normalising a known C++-vs-Python writer divergence.

    The C++ FieldML-API reference writer drops ``@name`` on ``<Region>`` on
    re-serialise; pyfieldml preserves it. @name on Region is purely a
    namespace label and carries no semantic effect within a single-region
    document, so we strip it from both sides before canonicalising to treat
    ``<Region>`` and ``<Region name="r">`` as equivalent for the comparison.
    """
    tree = etree.fromstring(xml_bytes)
    for region in tree.iter("Region"):
        if "name" in region.attrib:
            del region.attrib["name"]
    return etree.tostring(etree.ElementTree(tree), method="c14n", exclusive=True)


@pytest.mark.parametrize(
    "fixture",
    [
        "minimal.fieldml",
        "two_types.fieldml",
        "parameter_inline.fieldml",
    ],
)
def test_python_writer_self_roundtrip_preserves_model(
    fixtures_dir: Path, tmp_path: Path, fixture: str
) -> None:
    """Python writer -> reader round-trip must preserve the semantic model.

    Unlike the C++-reference comparisons below, this test does not depend on
    ``PYFIELDML_CPP_REF``: it only exercises the Python writer/reader pair,
    which gives us a real "writer is working correctly" gate even when the
    C++ reference library isn't available locally.
    """
    src = fixtures_dir / fixture
    doc_before = fml.read(src)
    out = tmp_path / f"rt_{fixture}"
    doc_before.write(out)
    doc_after = fml.read(out)

    # Same type sets
    assert set(doc_before.booleans) == set(doc_after.booleans)
    assert set(doc_before.continuous) == set(doc_after.continuous)
    assert set(doc_before.ensembles) == set(doc_after.ensembles)
    assert set(doc_before.meshes) == set(doc_after.meshes)

    # Same evaluator names and types
    assert set(doc_before.evaluators) == set(doc_after.evaluators)
    for name in doc_before.evaluators:
        before_kind = type(doc_before.evaluators[name]).__name__
        after_kind = type(doc_after.evaluators[name]).__name__
        assert before_kind == after_kind, f"{name}: {before_kind} != {after_kind}"


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

    # Semantic equivalence: C14N canonicalisation + normalisation of the
    # known C++-vs-Python writer divergence (Region@name). See
    # _canonicalize_for_cpp_compare for the rationale.
    py_canonical = _canonicalize_for_cpp_compare(py_out.read_bytes())
    cpp_canonical = _canonicalize_for_cpp_compare(cpp_out.read_bytes())
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
