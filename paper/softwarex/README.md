# SoftwareX submission: pyfieldml

This directory holds the SoftwareX (Elsevier, `elsarticle` class) manuscript
for `pyfieldml`, alongside its bibliography, figures and a convenience
Makefile. The JOSS draft lives next door in `paper/paper.md` and is kept as
a short companion.

## Files

- `pyfieldml-softwarex.tex` — the manuscript (5 mandatory numbered sections,
  2 mandatory metadata tables).
- `paper.bib` — BibTeX references (~25 entries).
- `figures/` — the six publication-quality PNG figures at 300 DPI.
  Regenerated from the source notebooks / bundled datasets by
  `figures/generate.py`.
- `Makefile` — `make` builds the PDF; `make figures` regenerates figures;
  `make clean` removes LaTeX aux files.

## Build the PDF

With a standard TeX Live (pdflatex + bibtex) install:

```bash
cd paper/softwarex
make                   # or, equivalently:
pdflatex pyfieldml-softwarex
bibtex   pyfieldml-softwarex
pdflatex pyfieldml-softwarex
pdflatex pyfieldml-softwarex
```

The `elsarticle` class and the `hyperref`, `graphicx`, `listings`,
`xcolor`, `amssymb` packages are required.

## Regenerate the figures

Figures are deterministic and regenerated from the bundled datasets and
a handful of direct matplotlib/pyvista calls — no notebook execution is
required. From the repository root:

```bash
uv run python paper/softwarex/figures/generate.py
```

### Figure provenance

| File | Source |
| --- | --- |
| `fig1_architecture.png` | Hand-composed matplotlib diagram in `generate.py` (function `fig1_architecture`). |
| `fig2_zoo_gallery.png` | 2×5 grid of PyVista static renders of the ten bundled `pyfieldml.datasets`, pattern adapted from `docs/notebooks/07_real_anatomy.ipynb`. |
| `fig3_evaluator_graph.png` | NetworkX DAG of the `rectus_femoris` evaluator graph, pattern adapted from `docs/notebooks/02_evaluator_graph.ipynb`. |
| `fig4_hermite_bending.png` | Euler–Bernoulli tip-loaded cantilever sampled at the Hermite xi nodes; pattern adapted from `docs/notebooks/03_hermite_bending.ipynb`. |
| `fig5_fem_solution.png` | Poisson solve on a refined `unit_cube` via `scikit-fem`; pattern adapted from `docs/notebooks/06_scikit_fem_poisson.ipynb`. |
| `fig6_msk_assembly.png` | Multi-mesh PyVista scene composed from four BodyParts3D bundled datasets. |

## Submission notes

- GitHub repo: <https://github.com/kchemorion/pyfieldml> (required by SoftwareX).
- Repository root carries both `LICENSE` and `LICENSE.txt` (Apache 2.0);
  SoftwareX specifically requires the `.txt` extension.
- Corresponding author: Francis Chemorion,
  Universitat Pompeu Fabra — `francis.chemorion@upf.edu`.
- Before submission: fill in ORCID in the `\author[upf]{...}` block.
