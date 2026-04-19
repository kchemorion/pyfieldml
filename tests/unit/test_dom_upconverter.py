"""Up-converter unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyfieldml.dom.parser import parse_file, parse_string
from pyfieldml.dom.upconverter import upconvert_to_0_5
from pyfieldml.errors import FieldMLParseError


def test_upconvert_0_4_rewrites_version(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "legacy_0_4_minimal.fieldml")
    # parse_file already upconverts on read, but re-running must be idempotent.
    upconvert_to_0_5(parsed.tree)
    assert parsed.tree.getroot().get("version") == "0.5.0"


def test_upconvert_0_3_rewrites_version(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "legacy_0_3_minimal.fieldml")
    upconvert_to_0_5(parsed.tree)
    assert parsed.tree.getroot().get("version") == "0.5.0"


def test_upconvert_no_op_on_0_5(fixtures_dir: Path) -> None:
    parsed = parse_file(fixtures_dir / "minimal.fieldml")
    upconvert_to_0_5(parsed.tree)
    assert parsed.tree.getroot().get("version") == "0.5.0"


def test_upconvert_0_3_renames_abstract_evaluator() -> None:
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Fieldml version="0.3">'
        '<Region name="r">'
        '<ContinuousType name="real.1d"/>'
        '<AbstractEvaluator name="real.1d.variable" valueType="real.1d"/>'
        "</Region>"
        "</Fieldml>"
    )
    parsed = parse_string(content)
    root = parsed.tree.getroot()
    # AbstractEvaluator must be rewritten to ArgumentEvaluator.
    assert root.find(".//AbstractEvaluator") is None
    assert root.find(".//ArgumentEvaluator") is not None
    # ParsedDocument.version tracks the ORIGINAL pre-upconvert version.
    assert parsed.version == "0.3"


def test_upconvert_0_3_strips_is_component_ensemble() -> None:
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Fieldml version="0.3_alpha">'
        '<Region name="r">'
        '<EnsembleType name="e" isComponentEnsemble="true">'
        '<Members><MemberRange min="1" max="2"/></Members>'
        "</EnsembleType>"
        "</Region>"
        "</Fieldml>"
    )
    parsed = parse_string(content)
    et = parsed.tree.getroot().find(".//EnsembleType")
    assert et is not None
    assert "isComponentEnsemble" not in et.attrib


def test_upconvert_0_3_renames_variables_wrapper() -> None:
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Fieldml version="0.3">'
        '<Region name="r">'
        '<ContinuousType name="real.1d"/>'
        '<AbstractEvaluator name="x" valueType="real.1d"/>'
        '<ExternalEvaluator name="ext" valueType="real.1d">'
        '<Variables><Variable name="x"/></Variables>'
        "</ExternalEvaluator>"
        "</Region>"
        "</Fieldml>"
    )
    parsed = parse_string(content)
    root = parsed.tree.getroot()
    assert root.find(".//Variables") is None
    assert root.find(".//Variable") is None
    assert root.find(".//Arguments") is not None
    assert root.find(".//Arguments/Argument") is not None


def test_upconvert_unknown_version_raises() -> None:
    # Build a tree manually; parse_string would reject an unknown version
    # in _validate_root before it could reach the upconverter.
    from lxml import etree

    root = etree.fromstring(
        b'<?xml version="1.0"?><Fieldml version="0.2"><Region name="r"/></Fieldml>'
    )
    tree = etree.ElementTree(root)
    with pytest.raises(FieldMLParseError, match="Cannot up-convert unknown"):
        upconvert_to_0_5(tree)
