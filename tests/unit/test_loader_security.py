"""Security-focused tests for FieldML document loading.

Covers path-traversal guards on external data resources (HDF5 / plain-text
hrefs) and on <Import> directives. A malicious FieldML document must not
be able to dereference files outside the document's base directory.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import pyfieldml as fml
from pyfieldml.errors import FieldMLParseError
from pyfieldml.model.imports import ImportResolver


def _write_h5(path: Path, dataset_name: str, data: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=data)


def test_hdf5_href_traversal_is_rejected(tmp_path: Path) -> None:
    """A FieldML doc with ``href="../outside.h5"`` must refuse to resolve."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_h5 = outside_dir / "secret.h5"
    _write_h5(outside_h5, "/coords", np.arange(12, dtype=np.float64).reshape(4, 3))

    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    doc_path = doc_dir / "malicious.fieldml"
    doc_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Fieldml version="0.5.0" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        '  <Region name="evil">\n'
        '    <ContinuousType name="real.3d">\n'
        '      <Components name="real.3d.component" count="3"/>\n'
        "    </ContinuousType>\n"
        '    <ParameterEvaluator name="coords" valueType="real.3d">\n'
        "      <DenseArrayData>\n"
        '        <DataSource name="coords_src" location="/coords" rank="2">\n'
        "          <ArrayDataSize>4 3</ArrayDataSize>\n"
        "        </DataSource>\n"
        '        <DataResourceHref xlink:href="../outside/secret.h5" format="HDF5"/>\n'
        "      </DenseArrayData>\n"
        "    </ParameterEvaluator>\n"
        "  </Region>\n"
        "</Fieldml>\n",
    )

    with pytest.raises(FieldMLParseError, match="escapes base_dir"):
        fml.read(doc_path)


def test_plain_text_href_traversal_is_rejected(tmp_path: Path) -> None:
    """A FieldML doc with ``href="../outside.txt"`` must refuse to resolve."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("1.0 2.0 3.0\n")

    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    doc_path = doc_dir / "malicious.fieldml"
    doc_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Fieldml version="0.5.0" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        '  <Region name="evil">\n'
        '    <ContinuousType name="real.1d"/>\n'
        '    <ParameterEvaluator name="values" valueType="real.1d">\n'
        "      <DenseArrayData>\n"
        '        <DataSource name="values_src" location="/values" rank="1">\n'
        "          <ArrayDataSize>3</ArrayDataSize>\n"
        "        </DataSource>\n"
        '        <DataResourceHref xlink:href="../outside/secret.txt" format="PLAIN_TEXT"/>\n'
        "      </DenseArrayData>\n"
        "    </ParameterEvaluator>\n"
        "  </Region>\n"
        "</Fieldml>\n",
    )

    with pytest.raises(FieldMLParseError, match="escapes base_dir"):
        fml.read(doc_path)


def test_hdf5_href_within_base_dir_still_works(tmp_path: Path) -> None:
    """Sanity: legitimate relative hrefs (including nested sibling dirs) continue to load."""
    doc_dir = tmp_path / "doc"
    (doc_dir / "sub").mkdir(parents=True)
    h5_path = doc_dir / "sub" / "ok.h5"
    _write_h5(h5_path, "/coords", np.arange(12, dtype=np.float64).reshape(4, 3))

    doc_path = doc_dir / "good.fieldml"
    doc_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Fieldml version="0.5.0" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        '  <Region name="ok">\n'
        '    <ContinuousType name="real.3d">\n'
        '      <Components name="real.3d.component" count="3"/>\n'
        "    </ContinuousType>\n"
        '    <ParameterEvaluator name="coords" valueType="real.3d">\n'
        "      <DenseArrayData>\n"
        '        <DataSource name="coords_src" location="/coords" rank="2">\n'
        "          <ArrayDataSize>4 3</ArrayDataSize>\n"
        "        </DataSource>\n"
        '        <DataResourceHref xlink:href="sub/ok.h5" format="HDF5"/>\n'
        "      </DenseArrayData>\n"
        "    </ParameterEvaluator>\n"
        "  </Region>\n"
        "</Fieldml>\n",
    )

    doc = fml.read(doc_path)
    arr = doc.region.evaluators["coords"].as_ndarray()  # type: ignore[attr-defined]
    assert arr.shape == (4, 3)


def test_import_resolver_rejects_traversal(tmp_path: Path) -> None:
    """ImportResolver._find_source must reject ``../`` hrefs symmetrically."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "sibling.xml").write_text(
        '<?xml version="1.0"?><Fieldml version="0.5.0">'
        '<Region name="imp"><BooleanType name="b"/></Region></Fieldml>'
    )

    base_dir = tmp_path / "doc"
    base_dir.mkdir()
    resolver = ImportResolver(base_dir=base_dir)

    with pytest.raises(FieldMLParseError, match="escapes base_dir"):
        resolver.resolve("../outside/sibling.xml", region_name="imp")
