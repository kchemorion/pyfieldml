"""Top-level `pyfieldml` CLI."""

from __future__ import annotations

import argparse

from pyfieldml.cli import bench as bench_cmd


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``pyfieldml <subcommand>``."""
    p = argparse.ArgumentParser(prog="pyfieldml")
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("bench", help="Benchmark field evaluation throughput.")
    bp.add_argument("path", help="FieldML document to benchmark.")
    bp.add_argument("--field", required=True, help="Evaluator name to use.")
    bp.add_argument("--n", type=int, default=10_000, help="Point count.")

    args = p.parse_args(argv)
    if args.cmd == "bench":
        return bench_cmd.run(path=args.path, field=args.field, n=args.n)
    return 2
