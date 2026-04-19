"""``pyfieldml lint`` — run the semantic linter on a FieldML document."""

from __future__ import annotations

import pyfieldml as fml
from pyfieldml.validation.lint import lint_document


def run(*, path: str) -> int:
    """Lint and print issues. Exit 0 if no errors, 1 if any error severity."""
    doc = fml.read(path)
    issues = lint_document(doc)
    if not issues:
        print(f"OK: no lint issues in {path}")
        return 0
    n_err = 0
    for issue in issues:
        tag = issue.severity.upper()
        line = f"{tag} [{issue.code}] "
        if issue.object_name:
            line += f"{issue.object_name}: "
        line += issue.message
        print(line)
        if issue.severity == "error":
            n_err += 1
    if n_err:
        print(f"\n{n_err} error(s); {len(issues) - n_err} warning(s)/info(s)")
        return 1
    print(f"\n{len(issues)} warning(s)/info(s), no errors")
    return 0
