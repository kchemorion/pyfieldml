# Contributing to pyfieldml

Thank you for your interest! pyfieldml is an open, community-built reimplementation of FieldML 0.5. We welcome issues, PRs, datasets, and example notebooks.

## Dev setup

```bash
git clone https://github.com/kchemorion/pyfieldml
cd pyfieldml
uv sync --extra dev
uv run pre-commit install
uv run pytest
```

## Workflow

- Work on a feature branch; PR into `main`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`.
- All PRs must pass CI (ruff, mypy, pytest matrix).
- New features arrive with tests. New files arrive with docstrings.

## Style

- `ruff format` / `ruff check` are authoritative. `mypy --strict` must be clean.
- Prefer small files with a single responsibility.
- Write tests first (TDD). Tests live alongside source in `tests/unit/`.

## Adding a dataset

See `docs/contributing-datasets.md` (added in Phase 5). Short version: fork, add your `.fieldml` + a `dataset.yaml` (BibTeX citation, SPDX license, SHA256 per file), open a PR. We verify license before merging.

## Adding a basis function

See `docs/contributing-bases.md` (added in Phase 2). Short version: `@pyfieldml.basis("your_element", topology="tet")` and implement the shape-function + derivative protocol.

## Questions

Use [GitHub Discussions](https://github.com/kchemorion/pyfieldml/discussions). Use Issues only for confirmed bugs or feature proposals.
