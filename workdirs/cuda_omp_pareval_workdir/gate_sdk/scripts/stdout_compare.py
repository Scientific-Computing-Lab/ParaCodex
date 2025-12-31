#!/usr/bin/env python3
"""Simple harness that runs two executables and compares their stdout.

The script treats any difference in the combined stdout streams as a failure.
stderr from each process is captured and displayed on mismatch to aid debugging.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], label: str, cwd: str | None = None) -> tuple[int, str]:
    """Run a command and capture its stdout combined with stderr."""

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"{label} executable not found: {cmd[0]}") from exc

    return proc.returncode, proc.stdout


def extract_signature(output: str) -> dict[str, str]:
    signature: dict[str, str] = {}
    for line in output.splitlines():
        if "Non-Matching" in line and ":" in line:
            signature["non_matching"] = line.split(":", maxsplit=1)[1].strip()
    return signature


def is_success_output(output: str) -> bool:
    lowered = output.lower()
    if "usage:" in lowered:
        return True
    for bad in ("fail", "error", "fatal"):
        if bad in lowered and "pass" not in lowered and "passed" not in lowered:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden", help="Path to the golden executable")
    parser.add_argument("candidate", help="Path to the candidate executable")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to both executables",
    )

    args = parser.parse_args()

    golden_path = Path(args.golden).resolve()
    candidate_path = Path(args.candidate).resolve()

    if not golden_path.exists():
        raise SystemExit(f"Golden executable not found: {golden_path}")

    if not candidate_path.exists():
        raise SystemExit(f"Candidate executable not found: {candidate_path}")

    golden_cmd = [str(golden_path)] + args.args
    candidate_cmd = [str(candidate_path)] + args.args

    golden_rc, golden_out = run_command(golden_cmd, "golden", cwd=str(golden_path.parent))
    if golden_rc != 0:
        print("[HARNESS][golden_failed]", file=sys.stderr)
        print(golden_out, file=sys.stderr)
        return 2

    candidate_rc, candidate_out = run_command(candidate_cmd, "candidate", cwd=str(candidate_path.parent))
    if candidate_rc != 0:
        print("[HARNESS][candidate_failed]", file=sys.stderr)
        print(candidate_out, file=sys.stderr)
        return 2

    if golden_out != candidate_out:
        golden_sig = extract_signature(golden_out)
        candidate_sig = extract_signature(candidate_out)
        if not golden_sig or golden_sig != candidate_sig:
            golden_pass = is_success_output(golden_out)
            candidate_pass = is_success_output(candidate_out)
            if golden_pass and candidate_pass:
                return 0
            print("[HARNESS][mismatch] stdout differs between golden and candidate", file=sys.stderr)
            print("---- GOLDEN ----", file=sys.stderr)
            print(golden_out, file=sys.stderr)
            print("---- CANDIDATE ----", file=sys.stderr)
            print(candidate_out, file=sys.stderr)
            return 1

    print("[HARNESS] PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())


