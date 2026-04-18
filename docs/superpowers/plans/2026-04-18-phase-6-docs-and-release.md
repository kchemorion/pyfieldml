# pyfieldml Phase 6 — Docs + v1.0 Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.
>
> **⚠ Forward-draft notice:** Depends on Phases 1–5 completing.

**Goal:** Finish what needs finishing for v1.0.0: embed runnable notebooks in the docs site via `jupyter-lite`, publish the conformance test suite as a downloadable artifact for other FieldML implementations, polish docs prose, cut the `v1.0.0` tag, and announce. Phase 7 (the JOSS paper) runs in parallel.

**Architecture:** `jupyter-lite` builds a pyodide-powered browser Jupyter environment from a config that lists our quickstart + fiber notebooks + a `pyfieldml` wheel. The conformance suite is a GitHub-release artifact bundling `tests/fixtures/` + a `runner.py` that third-party implementations can execute against their own reader/writer.

**Tech Stack:** adds `jupyterlite-core`, `jupyterlite-pyodide-kernel` (dev). Everything else is already in place.

**Spec reference:** §6.4 (jupyter-lite), §7.8 (done-definition), §7.9 (post-release).

**Completion criteria:**
- Visit `kchemorion.github.io/pyfieldml` → click the "run quickstart in browser" button → cell 1 (`import pyfieldml`) succeeds, cells through `doc.plot()` execute (with static-image fallback).
- A `pyfieldml-conformance-v1.0.zip` asset exists on the `v1.0.0` GH release, containing the fixture corpus and a stand-alone runner script.
- All docs prose has been read and copy-edited — no TBDs, no dead links, all code blocks tested.
- `v1.0.0` tag pushed, PyPI release published, announce posts drafted.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `docs/jupyterlite/jupyter-lite.json` | jupyter-lite config |
| `docs/jupyterlite/jupyter_lite_config.json` | build-time config |
| `docs/jupyterlite/overrides.json` | UI tweaks |
| `docs/jupyterlite/contents/quickstart.ipynb` | Slimmer browser-compatible version of Notebook 01 |
| `docs/jupyterlite/contents/muscle_fibers.ipynb` | Slimmer browser-compatible version of Notebook 04 |
| `docs/jupyterlite/README.md` | Build instructions |
| `conformance/runner.py` | Stand-alone runner that consumes a FieldML implementation + runs all fixtures |
| `conformance/README.md` | "How to run against your own impl" |
| `conformance/fixtures/` | Copied from `tests/fixtures/` at build time |
| `.github/workflows/docs.yml` | Extended to also build jupyter-lite |
| `.github/workflows/release.yml` | Extended to attach conformance-suite asset |

---

## Task 1 — Browser-compatible notebooks

`pyvista` does not run under pyodide. Adjust notebooks 01 + 04 for the
browser context: replace `pyvista` plots with static `matplotlib` renderings
of mesh + field overlays.

- [ ] **Step 1: Copy + adapt**

```bash
cp docs/notebooks/01_quickstart.ipynb docs/jupyterlite/contents/quickstart.ipynb
cp docs/notebooks/04_muscle_fibers.ipynb docs/jupyterlite/contents/muscle_fibers.ipynb
```

- [ ] **Step 2: Edit both to remove `pyvista` usage**

Replace each `doc.plot()` / `field.plot()` call with a matplotlib fallback
that draws a 3D scatter of node positions, colored by the selected field. A
small utility `pyfieldml.viz.plot_matplotlib_fallback(doc, field)` can
centralize this.

- [ ] **Step 3: Verify the notebooks execute**

Run locally:

```bash
uv run jupyter nbconvert --to notebook --execute docs/jupyterlite/contents/quickstart.ipynb --output quickstart_exec.ipynb
```

- [ ] **Step 4: Commit**

```bash
git add docs/jupyterlite/contents/ src/pyfieldml/viz/
git commit -m "docs(browser): jupyter-lite-compatible quickstart + fiber notebooks"
```

---

## Task 2 — jupyter-lite build

- [ ] **Step 1: Install jupyter-lite**

```bash
uv add --optional dev "jupyterlite-core>=0.4" "jupyterlite-pyodide-kernel>=0.4"
uv sync --extra dev
```

- [ ] **Step 2: Write `docs/jupyterlite/jupyter_lite_config.json`**

```json
{
  "LiteBuildConfig": {
    "contents": ["./contents"],
    "output_dir": "../../site/lite",
    "federated_extensions": [],
    "no_unused_shared_packages": true
  }
}
```

- [ ] **Step 3: Build locally**

```bash
cd docs/jupyterlite
uv run jupyter lite build
```

Confirm `site/lite/index.html` exists after the build.

- [ ] **Step 4: Host `pyfieldml` wheel so pyodide can install it**

```bash
uv run hatchling build
cp dist/pyfieldml-*.whl docs/jupyterlite/contents/
```

Update notebooks to begin with:

```python
%pip install -q pyfieldml
```

- [ ] **Step 5: Link from the main docs site**

In `docs/index.md`, add a banner:

```markdown
!!! tip "Try it in your browser"
    [Run the quickstart in a live Jupyter session →](./lite/lab/?path=quickstart.ipynb)
    (Powered by `jupyter-lite` + pyodide. No install needed.)
```

- [ ] **Step 6: Extend `.github/workflows/docs.yml`**

```yaml
      - run: uv sync --extra dev
      - run: uv run hatchling build
      - run: cp dist/pyfieldml-*.whl docs/jupyterlite/contents/
      - run: |
          cd docs/jupyterlite
          uv run jupyter lite build
      - run: uv run mkdocs build --strict
      - run: cp -r docs/jupyterlite/../../site/lite site/lite
```

- [ ] **Step 7: Commit + push; verify the deployed site**

```bash
git add docs/jupyterlite/ docs/index.md .github/workflows/docs.yml pyproject.toml
git commit -m "docs(lite): jupyter-lite in-browser runnable quickstart + fibers"
git push origin main
```

Visit the deployed URL — browser quickstart must load within 60 s.

---

## Task 3 — Public conformance test suite

Bundle `tests/fixtures/*` + a runner so any third-party FieldML library can
cross-check against us.

- [ ] **Step 1: Write `conformance/runner.py`**

```python
"""Stand-alone runner for the pyfieldml conformance test suite.

Usage:
    python runner.py --fixtures ./fixtures --reader <executable> --writer <executable>

where <executable> takes (input_path, output_path) arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from lxml import etree


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--reader-writer", required=True,
                   help="executable that takes (src, dst) and round-trips")
    args = p.parse_args()

    failures = []
    for src in sorted(args.fixtures.glob("*.fieldml")):
        out = src.with_suffix(".roundtrip.fieldml")
        subprocess.run([args.reader_writer, str(src), str(out)], check=True)
        if not _canonical_equivalent(src, out):
            failures.append(src.name)
        out.unlink()
    if failures:
        print(f"FAIL: {len(failures)} fixtures round-tripped unequally:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK: {len(list(args.fixtures.glob('*.fieldml')))} fixtures round-tripped.")
    return 0


def _canonical_equivalent(a: Path, b: Path) -> bool:
    ca = etree.tostring(etree.parse(a), method="c14n", exclusive=True)
    cb = etree.tostring(etree.parse(b), method="c14n", exclusive=True)
    return ca == cb


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write `conformance/README.md`**

Instructions for adopting the conformance suite, plus a link to the latest
release asset.

- [ ] **Step 3: Package step in `.github/workflows/release.yml`**

Add to the build job:

```yaml
      - name: Package conformance suite
        run: |
          mkdir -p conformance-pkg/fixtures
          cp tests/fixtures/*.fieldml conformance-pkg/fixtures/
          cp conformance/runner.py conformance-pkg/
          cp conformance/README.md conformance-pkg/
          cd conformance-pkg
          zip -r ../pyfieldml-conformance.zip .
      - uses: actions/upload-artifact@v4
        with:
          name: conformance
          path: pyfieldml-conformance.zip
```

And in the `github-release` job, attach the zip:

```yaml
      - uses: softprops/action-gh-release@v2
        with:
          files: |
            dist/*
            pyfieldml-conformance.zip
          generate_release_notes: true
```

- [ ] **Step 4: Commit**

```bash
git add conformance/ .github/workflows/release.yml
git commit -m "release: public conformance test suite packaged as GH release asset"
```

---

## Task 4 — Docs prose pass

- [ ] **Step 1: Read every `.md` in `docs/`. Fix typos, dead links, outdated code.**
- [ ] **Step 2: Update `README.md` for v1.0** — flagship badges, install instructions, cite-this, contributors list.
- [ ] **Step 3: Add `docs/faq.md`** — seed with 5-10 real questions likely to come up.
- [ ] **Step 4: Add `docs/cite.md`** — BibTeX + CITATION.cff pointer + JOSS DOI placeholder.
- [ ] **Step 5: Verify mkdocs builds with `--strict` (no warnings).**

```bash
uv run mkdocs build --strict
```

- [ ] **Step 6: Commit**

```bash
git commit -am "docs: v1.0 prose polish pass"
```

---

## Task 5 — v1.0.0 cut

- [ ] **Step 1: Gate checklist**

Only proceed when ALL of the following are true:

- [ ] Conformance CI green on every fixture with a real (not trivial-copy) C++ round-trip.
- [ ] Unit coverage ≥ 85% on `model/`, `eval/`, `data/`; ≥ 70% overall.
- [ ] `mypy --strict` clean.
- [ ] All 9 notebooks execute in nbmake CI.
- [ ] jupyter-lite loads the quickstart in a browser.
- [ ] All four datasets license-cleared.
- [ ] `pyfieldml lint` + `pyfieldml diff` working end-to-end.
- [ ] Docs site deployed; no `--strict` warnings.
- [ ] `CHANGELOG`, `CITATION`, `NOTICE`, `CODE_OF_CONDUCT`, `MAINTAINING`, `CONTRIBUTING` all current.

- [ ] **Step 2: Bump version to 1.0.0**

`src/pyfieldml/_version.py` → `"1.0.0"`.
`CITATION.cff` → `version: 1.0.0`, today's date.

- [ ] **Step 3: Write `CHANGELOG.md` 1.0.0 entry**

Cover highlights across all six implementation phases; cite the JOSS paper
DOI once assigned (or mark TBD and update in a patch release).

- [ ] **Step 4: Tag and push**

```bash
git commit -am "chore(release): v1.0.0"
git tag -a v1.0.0 -m "v1.0.0: initial public release"
git push origin main --tags
```

- [ ] **Step 5: Watch the release workflow**

- PyPI trusted-publishing upload succeeds.
- GH release created with wheel, sdist, and the conformance zip.
- Docs site redeploys.

- [ ] **Step 6: Smoke-test the install**

```bash
uv venv /tmp/pyfieldml-smoke
source /tmp/pyfieldml-smoke/bin/activate
pip install pyfieldml==1.0.0
python -c "
import pyfieldml, pyfieldml.datasets
print(pyfieldml.__version__)
print(pyfieldml.datasets.list())
"
```

---

## Task 6 — Announce

Draft (do not yet send) an announcement for each channel.

- [ ] **OpenCMISS mailing list:** `docs/release_notes/announce_opencmiss.md`
- [ ] **Physiome mailing list:** `docs/release_notes/announce_physiome.md`
- [ ] **r/scientificcomputing:** `docs/release_notes/announce_reddit.md`
- [ ] **Mastodon (`#scipy`, `#fediphys`):** 3-toot thread draft
- [ ] **OpenSim forum:** `docs/release_notes/announce_opensim.md`

Hold posts until Francis + his professor have reviewed.

- [ ] **Commit**

```bash
git add docs/release_notes/
git commit -m "docs: v1.0 announcement drafts (awaiting review)"
```

---

## Phase-6 done-definition

- [ ] jupyter-lite quickstart runs in-browser in ≤ 60 s on a cold load.
- [ ] `pyfieldml-conformance-v1.0.zip` is attached to the `v1.0.0` release.
- [ ] `pip install pyfieldml==1.0.0` succeeds.
- [ ] `kchemorion.github.io/pyfieldml` shows the 1.0 site.
- [ ] Announcement drafts ready for review.

Parallel: **Phase 7 — JOSS paper submission** (see the Phase 7 plan).
