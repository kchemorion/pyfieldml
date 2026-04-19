# Cite `pyfieldml`

If you use `pyfieldml` in academic work, please cite the software release.

## Software citation

The canonical citation metadata lives in [`CITATION.cff`](https://github.com/kchemorion/pyfieldml/blob/main/CITATION.cff).
BibTeX equivalent:

```bibtex
@software{pyfieldml,
  author  = {Chemorion, Francis},
  title   = {pyfieldml: A modern pure-Python implementation of FieldML 0.5 for computational biomechanics},
  year    = {2026},
  url     = {https://github.com/kchemorion/pyfieldml},
  version = {1.0.0},
}
```

## Paper (pending)

A Journal of Open Source Software (JOSS) paper is in preparation. Once
accepted, the DOI and full citation will be published here.

## Citing the underlying FieldML specification

`pyfieldml` is an implementation of the FieldML 0.5 standard, authored by
the Physiome Project / Auckland Bioengineering Institute. Please cite the
original FieldML work as well:

```bibtex
@article{fieldml_christie_2009,
  author  = {Christie, G. Richard and Nielsen, Poul M. F. and Blackett, Sean A. and Bradley, Chris P. and Hunter, Peter J.},
  title   = {FieldML: concepts and implementation},
  journal = {Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences},
  volume  = {367},
  number  = {1895},
  pages   = {1869--1884},
  year    = {2009},
  doi     = {10.1098/rsta.2009.0025},
}
```

## Original C++ FieldML-API

The C++ reference implementation we cross-validate against:

```bibtex
@misc{fieldml_api,
  author       = {Little, Caton and Wu, Alan and Christie, Richard and Miller, Andrew},
  title        = {{FieldML-API}: C++ reference implementation},
  howpublished = {\url{https://github.com/kchemorion/FieldML-API}},
  note         = {Auckland Uniservices Ltd},
}
```
