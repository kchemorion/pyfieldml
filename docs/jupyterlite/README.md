# pyfieldml JupyterLite site

This directory contains the build configuration for the in-browser
[JupyterLite](https://jupyterlite.readthedocs.io/) deployment of the
`pyfieldml` tutorial notebooks. The resulting static site lets visitors run
the tutorials directly in their browser with no local installation — the
Pyodide kernel fetches the `pyfieldml` wheel from PyPI at runtime.

## Layout

- `build.py` — driver script that copies the canonical tutorial notebooks
  from `docs/notebooks/` into `content/` and invokes `jupyter lite build`.
- `jupyter_lite_config.json` — JupyterLite configuration (piplite wheel
  index, litePluginSettings).
- `requirements.txt` — packages piplite pre-installs into the Pyodide kernel
  so notebooks run without an explicit `await piplite.install(...)` cell.
- `content/` — build-time staging area for the notebooks (gitignored).
- `_output/` — the generated static site, served as the site root
  (gitignored).

## Build locally

From the repo root:

```bash
uv pip install --extra jupyterlite
python docs/jupyterlite/build.py
# open docs/jupyterlite/_output/index.html in a browser, or:
python -m http.server --directory docs/jupyterlite/_output 8000
```

The first build takes ~1 min while JupyterLite downloads the Pyodide
runtime and bakes it into `_output/`.

## Deployment

The site is built on every push to `main` by
`.github/workflows/jupyterlite.yml` and uploaded as a `github-pages`
artifact. The workflow does **not** currently publish the artifact — enable
the Pages publish job (see the commented stub at the bottom of the
workflow) once the site is vetted in the GitHub Pages UI.

## Caveats

- The Pyodide kernel runs Python in WebAssembly; any pyfieldml feature that
  shells out to native binaries (e.g. `pyvista` 3D rendering) will fall
  back to non-interactive behavior or fail. The bundled notebooks are
  deliberately chosen to work in-browser.
- `h5py` and `meshio` have pre-built Pyodide wheels; `scipy` and `lxml` are
  also available. If a new dependency lacks a Pyodide wheel, the build
  will surface the error at runtime when the first notebook cell runs.
