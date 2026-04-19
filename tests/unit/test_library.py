"""Tests for bundled FieldML resource accessors."""

from __future__ import annotations

from pyfieldml.library import get_stdlib_path, get_xsd_path


def test_xsd_exists_and_is_xml() -> None:
    path = get_xsd_path()
    assert path.is_file()
    with path.open("rb") as fh:
        header = fh.read(256)
    assert b"<xs:schema" in header


def test_stdlib_exists_and_is_fieldml() -> None:
    path = get_stdlib_path()
    assert path.is_file()
    with path.open("rb") as fh:
        header = fh.read(256)
    assert b'<Fieldml version="0.5' in header


def test_get_xsd_path_returns_stable_path_across_calls() -> None:
    """The returned path must remain valid after the ``as_file`` context exits.

    Prior to v1.0.1 the accessor returned a context-managed temp path which,
    on zipapp/zipimport installs, became stale the moment the ``with`` block
    closed. Even on loose-file installs we want idempotent return values.
    """
    p1 = get_xsd_path()
    p2 = get_xsd_path()
    assert p1 == p2
    # Path is still readable even after the context manager inside ``_resource_path`` exits.
    assert p1.is_file()
    assert p1.read_bytes()[:64]


def test_get_stdlib_path_returns_stable_path_across_calls() -> None:
    p1 = get_stdlib_path()
    p2 = get_stdlib_path()
    assert p1 == p2
    assert p1.is_file()
