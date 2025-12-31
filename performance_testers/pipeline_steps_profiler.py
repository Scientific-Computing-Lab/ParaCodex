#!/usr/bin/env python3
import argparse
import json
import os
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Directories and build configuration
DATA_SRC_ROOT = Path("/root/codex_baseline/cuda_omp_workdir/data/src")
MAKEFILE_NAME = "Makefile.nvc"
RUN_TARGET = "run"


def find_benchmarks_with_steps(base_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Scan base_dir for benchmark subdirectories containing step files.

    Returns a mapping: bench_name -> { step_key -> cpp_path }
    where step_key is like "step1", "step2_supervised".
    """
    benchmarks: Dict[str, Dict[str, Path]] = {}
    if not base_dir.exists():
        return benchmarks

    step_re = re.compile(r"^main_step(\d+)(?:_supervised)?\.cpp$")
    supervised_re = re.compile(r"^main_step(\d+)_supervised\.cpp$")
    unsupervised_re = re.compile(r"^main_step(\d+)\.cpp$")

    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        step_map: Dict[str, Path] = {}
        try:
            for f in entry.iterdir():
                if not f.is_file():
                    continue
                name = f.name
                if step_re.match(name) is None:
                    continue
                m_sup = supervised_re.match(name)
                if m_sup:
                    step_num = m_sup.group(1)
                    key = f"step{step_num}_supervised"
                    step_map[key] = f
                    continue
                m_unsup = unsupervised_re.match(name)
                if m_unsup:
                    step_num = m_unsup.group(1)
                    key = f"step{step_num}"
                    step_map[key] = f
        except Exception:
            # Skip unreadable directories
            continue

        if step_map:
            benchmarks[entry.name] = step_map

    return benchmarks


def copy_cpp_to_datasrc_main(bench_name: str, cpp_file: Path) -> Path:
    """Copy the specified cpp_file into data/src/<bench_name>/main.cpp."""
    target_project_dir = DATA_SRC_ROOT / bench_name
    if not target_project_dir.exists():
        raise FileNotFoundError(f"Target benchmark directory not found: {target_project_dir}")
    dst = target_project_dir / "main.cpp"
    shutil.copy2(cpp_file, dst)
    return target_project_dir


def run_ncu_in_dir(work_dir: Path) -> Tuple[int, str]:
    """Run Nsight Compute over the build and execution in the given directory."""
    mk = work_dir / MAKEFILE_NAME
    if not mk.exists():
        return 2, f"Missing {MAKEFILE_NAME} in {work_dir}"

    # Clean first to avoid stale builds
    clean_cmd = ["make", "-f", MAKEFILE_NAME, "clean"]

    base_cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--section",
        "SpeedOfLight",
        "--launch-count",
        "4",
    ]

    cmd = base_cmd + [
        "make",
        "-f",
        MAKEFILE_NAME,
        RUN_TARGET,
    ]

    subprocess.run(clean_cmd, cwd=str(work_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def parse_duration_ms(ncu_output: str) -> Optional[float]:
    """
    Parse Nsight Compute output to extract kernel Duration in milliseconds.
    Collect all Duration entries and take the max to capture main kernel.
    """
    table_re = re.compile(r"^\s*Duration\s+(ms|us)\s+([0-9]+(?:\.[0-9]+)?)\s*$")
    durations_ms: List[float] = []
    for line in ncu_output.splitlines():
        m = table_re.search(line)
        if not m:
            continue
        unit = m.group(1)
        val = float(m.group(2))
        durations_ms.append(val if unit == "ms" else val / 1000.0)
    if durations_ms:
        return max(durations_ms)

    loose_re = re.compile(r"Duration\s+(ms|us)\s+([0-9]+(?:\.[0-9]+)?)")
    durations_ms = []
    for m in loose_re.finditer(ncu_output):
        unit = m.group(1)
        val = float(m.group(2))
        durations_ms.append(val if unit == "ms" else val / 1000.0)
    if durations_ms:
        return max(durations_ms)
    return None


def profile_mean_ms(work_dir: Path, runs: int = 3) -> Optional[float]:
    values: List[float] = []
    for _ in range(runs):
        rc, out = run_ncu_in_dir(work_dir)
        if rc != 0:
            continue
        ms = parse_duration_ms(out)
        if ms is not None:
            values.append(ms)
    if values:
        return sum(values) / len(values)
    return None


def write_benchmark_results_json(bench_name: str, step_to_ms: Dict[str, Optional[float]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = out_dir / bench_name
    bench_dir.mkdir(parents=True, exist_ok=True)
    json_path = bench_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(step_to_ms, f, indent=2)
    return json_path


def plot_benchmark_steps(bench_name: str, step_to_ms: Dict[str, Optional[float]], out_dir: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # Sort steps numerically, with unsupervised first then supervised for same step
    def sort_key(k: str) -> Tuple[int, int]:
        m = re.match(r"step(\d+)(?:_supervised)?$", k)
        step_num = int(m.group(1)) if m else 0
        supervised = 1 if k.endswith("_supervised") else 0
        return (step_num, supervised)

    items = sorted(step_to_ms.items(), key=lambda kv: sort_key(kv[0]))
    if not items:
        return None

    labels = [k for k, _ in items]
    raw_values = [v if v is not None else float("nan") for _, v in items]

    # Only plot if we have at least one positive, finite value; map non-positive to NaN for log scale
    has_positive = any((isinstance(v, float) and math.isfinite(v) and v > 0.0) for v in raw_values)
    if not has_positive:
        return None
    values = [v if (isinstance(v, float) and math.isfinite(v) and v > 0.0) else float("nan") for v in raw_values]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    ax.plot(range(len(labels)), values, marker="o")
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Kernel Duration (ms) [log]")
    ax.set_yscale("log")
    ax.set_title(f"{bench_name}: Kernel Duration Across Steps")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    bench_dir = out_dir / bench_name
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_path = bench_dir / "steps.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Profile kernel durations for pipeline steps using Nsight Compute")
    parser.add_argument(
        "--input",
        type=str,
        default="/root/codex_baseline/pipeline/translated_codes",
        help="Directory containing translated benchmark subdirectories with main_step*.cpp files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/codex_baseline/cuda_omp_workdir/results_steps",
        help="Output directory where per-benchmark results will be written",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of profiling runs per step to average",
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()

    # Preflight: ensure required tools exist in PATH
    required_tools = ["make"]
    if args.runs and args.runs > 0:
        required_tools.append("ncu")
    for tool in required_tools:
        if shutil.which(tool) is None:
            print(f"Required tool not found in PATH: {tool}", file=sys.stderr)
            return 2

    # Discover benchmarks and their step files
    bench_to_steps = find_benchmarks_with_steps(input_dir)
    if not bench_to_steps:
        print("No step files found in input directory", file=sys.stderr)
        return 1

    # Process each benchmark
    for bench_name, step_map in sorted(bench_to_steps.items()):
        print(f"Processing benchmark: {bench_name}")
        step_to_ms: Dict[str, Optional[float]] = {}

        # Sort by step then supervised
        def sort_key(k: str) -> Tuple[int, int]:
            m = re.match(r"step(\d+)(?:_supervised)?$", k)
            step_num = int(m.group(1)) if m else 0
            supervised = 1 if k.endswith("_supervised") else 0
            return (step_num, supervised)

        for step_key in sorted(step_map.keys(), key=sort_key):
            cpp_file = step_map[step_key]
            print(f"  Profiling {step_key}: {cpp_file}")
            try:
                project_dir = copy_cpp_to_datasrc_main(bench_name, cpp_file)
            except Exception as e:
                print(f"  [WARN] copy failed for {bench_name}/{step_key}: {e}")
                step_to_ms[step_key] = None
                continue

            ms = profile_mean_ms(project_dir, runs=args.runs)
            step_to_ms[step_key] = ms
            print(f"  -> mean duration: {ms} ms")

        # Write outputs for this benchmark
        json_path = write_benchmark_results_json(bench_name, step_to_ms, out_dir)
        plot_path = plot_benchmark_steps(bench_name, step_to_ms, out_dir)
        print(f"Wrote {json_path}")
        if plot_path:
            print(f"Wrote {plot_path}")
        else:
            print("matplotlib not available, skipping plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


