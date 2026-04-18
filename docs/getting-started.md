# Getting started

## Install

```bash
pip install pyfieldml
```

For development:

```bash
git clone https://github.com/kchemorion/pyfieldml
cd pyfieldml
uv sync --extra dev
uv run pytest
```

## Your first FieldML document

```python
import pyfieldml as fml

doc = fml.read("path/to/model.fieldml")
print(doc.source_version)
doc.validate()
```

## What's here in v0.0.x

Phase 0 ships the I/O plumbing:

- `pyfieldml.read(path)` -> `Document`
- `Document.validate()` - XSD validation
- `Document.write(path)` - round-trip-preserving serialization
- `Document.tree` - raw lxml escape hatch

Semantic parsing (Evaluators, Types, the evaluation engine) arrives in Phase 1+.
