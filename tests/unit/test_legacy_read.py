"""End-to-end test: read a legacy document through the public API."""

from __future__ import annotations

from pathlib import Path

import pyfieldml as fml


def test_legacy_0_4_reads_as_0_5_semantic_model(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "legacy_0_4_minimal.fieldml")
    # source_version tracks the original (pre-upconvert) version string.
    assert doc.source_version == "0.4"
    # The model is populated correctly after up-conversion.
    assert "boolean" in doc.booleans


def test_legacy_0_3_reads_as_0_5_semantic_model(fixtures_dir: Path) -> None:
    doc = fml.read(fixtures_dir / "legacy_0_3_minimal.fieldml")
    assert doc.source_version == "0.3"
    # ContinuousType survives unchanged across 0.3 -> 0.5.
    assert "real.1d" in doc.continuous
    # AbstractEvaluator has been renamed to ArgumentEvaluator, so the loader
    # picks it up as an ArgumentEvaluator in the evaluators map.
    assert "real.1d.variable" in doc.evaluators
