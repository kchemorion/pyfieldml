# Changelog

All notable changes to pyfieldml are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
pyfieldml follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-04-18

### Added
- Initial project scaffolding, src-layout, pyproject.toml with hatchling
  backend.
- Public top-level API: `pyfieldml.read`, `pyfieldml.validate`,
  `pyfieldml.Document`, error hierarchy.
- DOM layer (`pyfieldml.dom`): `parse_file`, `parse_string`, `validate_tree`,
  `write_file`, `write_string`. FieldML 0.3 / 0.4 / 0.5 version detection at
  the root.
- Bundled FieldML 0.5 XSD and standard library (`pyfieldml.library`).
- CI matrix (ruff + mypy + pytest) across Python 3.10-3.13 × Linux / macOS
  / Windows.
- mkdocs-material documentation site with Getting Started + API reference.
- Release workflow with PyPI trusted publishing (OIDC).
- Community files: CONTRIBUTING, CODE_OF_CONDUCT, CHANGELOG, MAINTAINING,
  issue + PR templates.

[Unreleased]: https://github.com/kchemorion/pyfieldml/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/kchemorion/pyfieldml/releases/tag/v0.0.1
