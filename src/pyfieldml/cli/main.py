"""Top-level ``pyfieldml`` CLI."""

from __future__ import annotations

import argparse

from pyfieldml.cli import (
    bench as bench_cmd,
)
from pyfieldml.cli import (
    convert as convert_cmd,
)
from pyfieldml.cli import (
    diff as diff_cmd,
)
from pyfieldml.cli import (
    inspect as inspect_cmd,
)
from pyfieldml.cli import (
    lint as lint_cmd,
)
from pyfieldml.cli import (
    plot as plot_cmd,
)
from pyfieldml.cli import (
    validate as validate_cmd,
)


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``pyfieldml <subcommand>``."""
    p = argparse.ArgumentParser(prog="pyfieldml")
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("bench", help="Benchmark field evaluation throughput.")
    bp.add_argument("path", help="FieldML document to benchmark.")
    bp.add_argument("--field", required=True, help="Evaluator name to use.")
    bp.add_argument("--n", type=int, default=10_000, help="Point count.")

    ip = sub.add_parser("inspect", help="Print a summary tree of a FieldML document.")
    ip.add_argument("path", help="FieldML document to inspect.")

    vp = sub.add_parser("validate", help="XSD-validate a FieldML document.")
    vp.add_argument("path", help="FieldML document to validate.")
    vp.add_argument("--strict", action="store_true", help="Reserved for future semantic checks.")

    cp = sub.add_parser("convert", help="Convert a FieldML document via meshio.")
    cp.add_argument("path", help="Input FieldML document.")
    cp.add_argument(
        "--to",
        dest="to_format",
        required=True,
        help="Target meshio format (e.g. 'vtu', 'obj').",
    )
    cp.add_argument("-o", "--output", dest="out", required=True, help="Output path.")

    pp = sub.add_parser("plot", help="Render a FieldML document via PyVista.")
    pp.add_argument("path", help="FieldML document to plot.")
    pp.add_argument("--field", help="Specific field to render (defaults to doc.plot()).")

    lp = sub.add_parser("lint", help="Run the semantic linter.")
    lp.add_argument("path", help="FieldML document to lint.")

    dp = sub.add_parser("diff", help="Semantic diff between two FieldML documents.")
    dp.add_argument("a", help="First document.")
    dp.add_argument("b", help="Second document.")

    args = p.parse_args(argv)
    if args.cmd == "bench":
        return bench_cmd.run(path=args.path, field=args.field, n=args.n)
    if args.cmd == "inspect":
        return inspect_cmd.run(path=args.path)
    if args.cmd == "validate":
        return validate_cmd.run(path=args.path, strict=args.strict)
    if args.cmd == "convert":
        return convert_cmd.run(path=args.path, to_format=args.to_format, out=args.out)
    if args.cmd == "plot":
        return plot_cmd.run(path=args.path, field=args.field)
    if args.cmd == "lint":
        return lint_cmd.run(path=args.path)
    if args.cmd == "diff":
        return diff_cmd.run(a=args.a, b=args.b)
    return 2
