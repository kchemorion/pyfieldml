# FAQ

Common questions about pyfieldml. If your question isn't here, open a
[GitHub Discussion](https://github.com/kchemorion/pyfieldml/discussions).

## What is FieldML?

FieldML is a declarative markup language for representing mathematical
fields over discrete meshes — developed by the
[Physiome Project](https://physiomeproject.org/) and Auckland Bioengineering
Institute for computational physiology. A FieldML document describes a mesh
topology, its geometric embedding, and any number of scalar or vector fields
defined on top, using a small graph of typed *evaluators*.

`pyfieldml` is a modern Python reimplementation of the FieldML 0.5
specification — independent of, but validated against, the C++
[FieldML-API](https://github.com/kchemorion/FieldML-API) reference library.

## Is pyfieldml compatible with existing FieldML documents?

Yes — `pyfieldml.read(path)` accepts FieldML 0.3, 0.4, and 0.5 documents.
Legacy 0.3 / 0.4 documents are transparently up-converted to the 0.5 shape
on read. The original version is tracked via `Document.source_version`.
Writers always emit 0.5.

## What bases are supported?

As of v1.0:
- Lagrange (linear + quadratic) on line, triangle, quad, tet, hex, wedge — 12 bases.
- Cubic Hermite on line, bicubic on quad, tricubic on hex — 3 bases.

External bases can be registered via `@pyfieldml.basis(...)` — see the
[scientific-computing contribution guide](https://github.com/kchemorion/pyfieldml/blob/main/CONTRIBUTING.md).

## Can I use pyfieldml with OpenSim, ParaView, scikit-fem, etc.?

Yes — `pyfieldml.interop` ships bridges to:
- `meshio` (two-way, with plugin registration — `meshio.read("*.fieldml")` works)
- `PyVista` (VTK UnstructuredGrid, `doc.plot()`, `field.plot()`)
- `XDMF3` (one-way writer for ParaView / VisIt)
- `scikit-fem` (returns `(Mesh, Basis)` ready for PDE assembly)
- OpenSim-adjacent export (geometry `.vtu`, muscle attachments CSV,
  fiber-line polylines)

Each bridge is an optional extra — `pip install pyfieldml[meshio]`,
`[viz]`, `[scikit-fem]`, `[opensim]`, or `[all]`.

## Why are some bundled datasets synthetic?

The bundled model zoo in `pyfieldml.datasets` is a mix of authored
CC0 geometries and permissively-licensed real meshes:

- `unit_cube`, `rectus_femoris` — synthetic, CC0 dedication. Tiny
  authored geometries useful for tests and tutorials.
- `femur` — **anatomical-synthetic**, CC0 dedication. As of v1.1 this
  is no longer a plain cylinder: it is assembled from CSG primitives
  (shaft with trochanter / condyle swellings, femoral neck, spherical
  femoral head, and two distal condyle lobes) with a BMD-proxy Young's
  modulus field.
- `bunny_stanford` — **real 3D-scanned mesh**, effectively public
  domain. The Stanford Bunny (Turk & Levoy 1994) from the
  [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/).
  Decimated to ~948 triangles to fit the wheel size budget.
- `femur_bodyparts3d` — **real anatomical atlas sample**,
  CC-BY-SA 2.1 JP. Derived from
  [BodyParts3D](https://lifesciencedb.jp/bp3d/) (DBCLS, Japan), the
  left-femur element (FMA24475). Decimated for bundling; downstream
  redistributors must comply with CC-BY-SA 2.1 JP for this file
  specifically (attribution + share-alike). See `NOTICE` and
  `LICENSES/CC-BY-SA-2.1-JP.txt` in the repository.

We still lack real Physiome-sourced cardiac / MSK meshes pending
licensing with contributing research groups (e.g. the Auckland heart
atlas), but as of v1.1 we ship one public-domain real-scanned mesh
(bunny) and one CC-BY-SA real anatomical atlas sample (femur). See
[CONTRIBUTING.md](https://github.com/kchemorion/pyfieldml/blob/main/CONTRIBUTING.md)
for how to contribute your own dataset.

## How do I report a bug?

Open an issue at <https://github.com/kchemorion/pyfieldml/issues>. A
minimal `.fieldml` reproduction file is enormously helpful.

## Is there a paper?

A JOSS submission is in preparation. Once accepted, a DOI badge will
appear on the repo's README and in `CITATION.cff`.
