# pyfieldml

A modern pure-Python implementation of [FieldML](https://physiomeproject.org/software/fieldml) 0.5 with a full evaluation engine, interop bridges, and a curated biomechanics model zoo.

!!! info "Status: pre-alpha"
    pyfieldml is under active development. Phase 0 (scaffolding + DOM layer) is live. See the [design spec](https://github.com/kchemorion/pyfieldml/blob/main/docs/superpowers/specs/2026-04-18-pyfieldml-design.md) for the full v1.0 roadmap.

## Install

```bash
pip install pyfieldml
```

## Quickstart

```python
import pyfieldml as fml

doc = fml.read("model.fieldml")
print(doc.source_version)          # e.g. "0.5.0"
doc.validate()                     # XSD-check
doc.write("roundtripped.fieldml")
```
