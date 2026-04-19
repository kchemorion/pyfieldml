# pyfieldml conformance test suite

Standalone runner that any FieldML implementation can use to cross-check
round-trip fidelity against the pyfieldml reference fixtures.

## Usage

1. Download the latest conformance zip from the GitHub release page.
2. Extract.
3. Point `runner.py` at your implementation's CLI:

```bash
python runner.py --fixtures ./fixtures --reader-writer /path/to/your_roundtrip_tool
```

where `your_roundtrip_tool input.fieldml output.fieldml` reads then writes
the file through your library.

The runner compares C14N-canonicalized XML output against the expected
shapes. Non-zero exit code indicates a divergence.

## Contents

- `fixtures/` — `.fieldml` test documents covering the core FieldML 0.5 shapes
- `runner.py` — the harness described above
- `README.md` — this file

## Licensing

Same as pyfieldml itself: Apache 2.0. See the pyfieldml repository for the
full LICENSE + NOTICE.
