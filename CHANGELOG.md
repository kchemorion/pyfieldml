# Changelog

All notable changes to pyfieldml are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
pyfieldml follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-19

### Added
- Cubic Hermite basis functions: line (4 DOFs), bicubic quad (16 DOFs),
  tricubic hex (64 DOFs). Registered as `library.basis.cubic_hermite.line`,
  `.bicubic_hermite.quad`, `.tricubic_hermite.hex`.
- `pyfieldml.builders.add_hermite_mesh` — one-call Hermite line mesh builder
  with per-node scale-factor support (Phase-3 simplification: scalar-per-node
  scaling; quad/hex builders raise NotImplementedError for now).
- `Field.evaluate` and `Field.jacobian` apply per-node scale factors when
  the field is a Hermite-line mesh.
- FieldML 0.3 and 0.4 → 0.5 DOM up-converter (`pyfieldml.dom.upconverter`).
  Handles the version tag, plus 0.3 element renames observed in the
  reference library (`AbstractEvaluator` → `ArgumentEvaluator`,
  `Variables`/`Variable` → `Arguments`/`Argument`) and strips the 0.3
  `isComponentEnsemble` marker attribute.
- `parse_file` / `parse_string` auto-invoke the up-converter on legacy
  documents; `Document.source_version` continues to track the original
  pre-upconversion version.
- Supported-version set now includes `"0.3_alpha"` (the literal used by
  the Physiome reference 0.3 library).

### Known limitations
- The 0.3 `ContinuousType componentEnsemble="X"` structural transform is
  not yet implemented — ensemble-backed continuous types load as scalar
  types, silently losing the component link. Full 0.3 library ingestion
  is a future-phase task.
- No 0.4 test corpus is available, so 0.4 upconverter is version-rewrite
  only; element renames will be added as real fixtures surface.
- Hermite quad/hex builders raise NotImplementedError; only line meshes
  are constructible via `add_hermite_mesh`.

## [0.2.0] - 2026-04-19

### Added
- `pyfieldml.eval` subpackage: evaluator-graph walker (`evaluate_evaluator`),
  `EvaluationContext` with binding stack, `Field` user-facing wrapper.
- `pyfieldml.eval.bases`: runtime basis-function plugin registry (`@basis`
  decorator, `get_basis`, `list_bases`); built-in linear + quadratic Lagrange
  on line / triangle / quad / tet / hex / wedge (12 bases total).
- `Field.evaluate(element, xi)` — scalar and vectorized parametric evaluation
  on Lagrange meshes.
- `Field.sample(points)` — physical-point sampling backed by `scipy.spatial.cKDTree`
  candidate selection + inverse-xi Newton root-finding per element; points
  outside the mesh return NaN.
- `Field.jacobian(element, xi)` — returns the ∂(field)/∂(xi) Jacobian.
- `pyfieldml.builders.add_lagrange_mesh` — one-call builder that adds a
  complete Lagrange mesh graph (types, coords, connectivity, basis external).
- `pyfieldml bench <path> --field <name>` CLI subcommand for throughput
  benchmarks. Entry point exposed via `[project.scripts]`.
- Benchmark regression test: vectorized `Field.evaluate` is asserted to be
  at least 50× a scalar Python loop.

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

[Unreleased]: https://github.com/kchemorion/pyfieldml/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/kchemorion/pyfieldml/releases/tag/v0.3.0
[0.2.0]: https://github.com/kchemorion/pyfieldml/releases/tag/v0.2.0
[0.1.0]: https://github.com/kchemorion/pyfieldml/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/kchemorion/pyfieldml/releases/tag/v0.0.1
