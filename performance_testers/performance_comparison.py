#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DATA_SRC_ROOT = Path("/root/codex_baseline/cuda_omp_workdir/data/src")
MAKEFILE_NAME = "Makefile.nvc"
RUN_TARGET = "run"


@dataclass
class BenchmarkResult:
    name: str
    reference_ms: Optional[float]
    candidate_ms: Optional[float]
    reference_log: Optional[str]
    candidate_log: Optional[str]

def clean_nsys_artifacts(work_dir: Path):
    """
    Remove Nsight Systems artifacts (nsys_profile.*) from the work directory.
    """
    for pattern in ["nsys_profile*", "*.qdstrm*", "*.nsys-rep*", "*.sqlite*"]:
        for file in work_dir.glob(pattern):
            try:
                file.unlink()
            except Exception:
                pass


def find_subdirs_with_files(base_dir: Path, required_filename: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not base_dir.exists():
        return mapping
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        candidate = entry / required_filename
        if candidate.exists():
            mapping[entry.name] = entry
    return mapping


def copy_candidate_to_datasrc(bench_name: str, candidate_dir: Path) -> Path:
    target_project_dir = DATA_SRC_ROOT / bench_name
    if not target_project_dir.exists():
        raise FileNotFoundError(f"Target benchmark directory not found: {target_project_dir}")
    src = candidate_dir / "main_optimized.cpp"
    if not src.exists():
        raise FileNotFoundError(f"Missing main_optimized.cpp in {candidate_dir}")
    dst = target_project_dir / "main.cpp"
    shutil.copy2(src, dst)
    return target_project_dir


def run_nsys_in_dir(work_dir: Path) -> Tuple[int, str]:
    """
    Run Nsight Systems on `make -f Makefile.nvc run` in the given directory,
    capturing stdout which includes the cuda_gpu_kern_sum stats.
    """
    mk = work_dir / MAKEFILE_NAME
    print(f"Making clean in {work_dir}")
    if not mk.exists():
        return 2, f"Missing {MAKEFILE_NAME} in {work_dir}"

    # Clean first
    clean_cmd = ["make", "-f", MAKEFILE_NAME, "clean"]
    subprocess.run(
        clean_cmd,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Nsight Systems profile command
    # We rely on --stats=true to print cuda_gpu_kern_sum to stdout,
    # and we force-overwrite the report to avoid accumulation between runs.
    cmd = [
        "nsys",
        "profile",
        "--stats=true",
        "--trace=cuda,osrt",
        "--force-overwrite=true",
        "-o",
        "nsys_profile",
        "make",
        "-f",
        MAKEFILE_NAME,
        RUN_TARGET,
    ]

    # Set environment variables for OpenMP GPU offloading
    env = os.environ.copy()
    env["FORCE_OMP_GPU"] = "1"
    env["OMP_TARGET_OFFLOAD"] = "MANDATORY"

    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    clean_nsys_artifacts(work_dir)

    return proc.returncode, proc.stdout


def parse_total_gpu_time_ms(nsys_output: str) -> Optional[float]:
    """
    Parse Nsight Systems stdout (with cuda_gpu_kern_sum report) and compute
    the total GPU kernel time by summing 'Total Time (ns)' across all kernels.
    Returns total time in milliseconds, or None if parsing fails.
    """
    lines = nsys_output.splitlines()
    in_kernel_table = False
    total_ns = 0

    # We enter the table after seeing the cuda_gpu_kern_sum header
    # or the "CUDA GPU Kernel Summary" title.
    for line in lines:
        if "cuda_gpu_kern_sum" in line or "CUDA GPU Kernel Summary" in line:
            in_kernel_table = True
            continue

        if not in_kernel_table:
            continue

        # End of table: blank line or start of another report
        if not line.strip():
            # allow a blank line to terminate the table
            if total_ns > 0:
                break
            else:
                continue

        # Match data lines: they start with numeric Time (%)
        # Example:
        #  94.5       4086451525       1900 ...
        m = re.match(
            r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+)\s+([0-9]+)\s+",
            line,
        )
        if not m:
            # If we hit a non-data line after we've already parsed some rows,
            # assume table ended.
            if total_ns > 0:
                break
            continue

        time_percent_str, total_time_ns_str, instances_str = m.groups()
        try:
            total_ns += int(total_time_ns_str)
        except ValueError:
            continue

    if total_ns > 0:
        return total_ns / 1e6  # ns -> ms
    return None


def profile_mean_ms(work_dir: Path, runs: int = 3) -> Optional[float]:
    """
    Run Nsight Systems multiple times and return the mean total GPU kernel time (ms).
    """
    values: List[float] = []
    for _ in range(runs):
        rc, out = run_nsys_in_dir(work_dir)
        if rc != 0:
            continue
        ms = parse_total_gpu_time_ms(out)
        if ms is not None:
            values.append(ms)
    if values:
        return sum(values) / len(values)
    return None


def write_results_json_txt(results: List[BenchmarkResult], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    txt_path = out_dir / "results.txt"

    serializable = []
    for r in results:
        serializable.append(
            {
                "name": r.name,
                "reference_ms": r.reference_ms,
                "candidate_ms": r.candidate_ms,
            }
        )
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)

    with open(txt_path, "w") as f:
        for r in results:
            f.write(
                f"{r.name}: reference={r.reference_ms} ms, candidate={r.candidate_ms} ms\n"
            )

    return json_path, txt_path


def plot_results(results: List[BenchmarkResult], out_dir: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    names = [r.name for r in results]
    ref_vals = [r.reference_ms if r.reference_ms is not None else float("nan") for r in results]
    cand_vals = [r.candidate_ms if r.candidate_ms is not None else float("nan") for r in results]

    x = list(range(len(names)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
    ax.bar([i - width / 2 for i in x], ref_vals, width, label="reference")
    ax.bar([i + width / 2 for i in x], cand_vals, width, label="candidate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Total GPU Kernel Time (ms) [log]")
    ax.set_yscale("log")
    ax.set_title("Reference vs Candidate Total GPU Kernel Time")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile candidate vs reference benchmarks using Nsight Systems (total GPU kernel time)"
    )
    parser.add_argument("--reference_dir", type=str, help="Directory containing subdirs with main.cpp")
    parser.add_argument("--candidate_dir", type=str, help="Directory containing subdirs with main_optimized.cpp")
    parser.add_argument(
        "--output",
        type=str,
        default="/root/codex_baseline/cuda_omp_workdir/results_perf_nsys",
        help="Output directory for results",
    )
    args = parser.parse_args(argv)

    reference_dir = Path(args.reference_dir).resolve()
    candidate_dir = Path(args.candidate_dir).resolve()
    out_dir = Path(args.output).resolve()

    # Preflight: ensure required tools exist in PATH
    for tool in ("nsys", "make"):
        if shutil.which(tool) is None:
            print(f"Required tool not found in PATH: {tool}", file=sys.stderr)
            return 2

    ref_map = find_subdirs_with_files(reference_dir, "main.cpp")
    cand_map = find_subdirs_with_files(candidate_dir, "main_optimized.cpp")

    common = sorted(set(ref_map.keys()) & set(cand_map.keys()))
    if not common:
        print("No common benchmarks found between reference and candidate", file=sys.stderr)
        return 1

    results: List[BenchmarkResult] = []

    for bench in common:
        print(f"=== Benchmark: {bench} ===")
        # Copy candidate into data/src/<bench>/main.cpp
        try:
            project_dir = copy_candidate_to_datasrc(bench, cand_map[bench])
        except Exception as e:
            results.append(BenchmarkResult(bench, None, None, None, None))
            print(f"[WARN] {bench}: copy failed: {e}")
            continue

        # Run Nsight Systems for candidate (after copy/build), 3 runs mean
        cand_ms = profile_mean_ms(project_dir, runs=3)

        # Restore reference by copying its main.cpp over (to measure reference in the exact same env)
        ref_src = ref_map[bench] / "main.cpp"
        try:
            shutil.copy2(ref_src, project_dir / "main.cpp")
        except Exception as e:
            print(f"[WARN] {bench}: restoring reference failed: {e}")

        ref_ms = profile_mean_ms(project_dir, runs=3)

        results.append(
            BenchmarkResult(
                name=bench,
                reference_ms=ref_ms,
                candidate_ms=cand_ms,
                reference_log=None,
                candidate_log=None,
            )
        )

    json_path, txt_path = write_results_json_txt(results, out_dir)
    plot_path = plot_results(results, out_dir)

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    if plot_path:
        print(f"Wrote {plot_path}")
    else:
        print("matplotlib not available, skipping plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
