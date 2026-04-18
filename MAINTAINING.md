# Maintaining pyfieldml

## Release checklist

1. Ensure CI is green on `main`.
2. Update `CHANGELOG.md` — move Unreleased entries into a dated version section.
3. Bump `src/pyfieldml/_version.py`.
4. Update `CITATION.cff` version and date.
5. Commit: `chore(release): vX.Y.Z`.
6. Tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`.
7. Push: `git push origin main --tags`.
8. Confirm `release.yml` workflow succeeds; verify package on PyPI.
9. Announce in GitHub Discussions + relevant mailing lists.

## Adding a phase

Phases correspond to the roadmap in the design spec. Each phase:
1. Has its own plan in `docs/superpowers/plans/`.
2. Lands as a series of Conventional-Commit PRs.
3. Concludes with a release (minor bump during pre-1.0, else as appropriate).

## Deprecations

Deprecate with `DeprecationWarning` for one minor version before removal.
Document in `CHANGELOG.md` under "Deprecated" and "Removed".
