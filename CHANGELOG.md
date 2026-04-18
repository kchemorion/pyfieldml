# Changelog

All notable changes to pyfieldml are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
pyfieldml follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-18

### Added
- FieldML type classes: `BooleanType`, `EnsembleType`, `ContinuousType`,
  `MeshType` (`pyfieldml.model.types`).
- Data backends (`pyfieldml.data`): `DataBackend` base, `InlineTextBackend`
  with lossless float round-trip, `ExternalTextBackend` with sibling-file
  resolution, `Hdf5DenseBackend` lazy-loading, `Hdf5DOKBackend` sparse.
- Evaluator model (`pyfieldml.model`): `Evaluator` base, `ParameterEvaluator`,
  `ConstantEvaluator`, `ArgumentEvaluator` with `BindingTable`,
  `ReferenceEvaluator`, `AggregateEvaluator`, `PiecewiseEvaluator`,
  `ExternalEvaluator`.
- `Region` container with insertion-ordered uniqueness and name lookup.
- Import resolution (`ImportResolver`) for bundled stdlib and local files.
- Full DOM to model loader for all evaluator kinds.
- Model to DOM writer for programmatically-built `Document`s.
- Upgraded `Document` with typed model-backed views.
- HDF5-backed parameter round-trip fixture and regression test.
- Conformance CI scaffold cross-validating round-trip against the C++
  FieldML-API reference oracle (assertions skipped until writer parity).

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

[Unreleased]: https://github.com/kchemorion/pyfieldml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kchemorion/pyfieldml/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/kchemorion/pyfieldml/releases/tag/v0.0.1
