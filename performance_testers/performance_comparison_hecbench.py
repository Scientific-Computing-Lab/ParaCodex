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


DATA_SRC_ROOT = Path("/path/to/workdir/cuda_omp_workdir/data/src")
# Reference files are the original OpenMP implementations from Hecbench repository
# They are located in: /path/to/workdir/hecbench_reference/src/<bench>-omp/main.cpp
REFERENCE_ROOT = Path("/path/to/workdir/hecbench_reference/src")
BASELINE_ROOT = Path("/path/to/workdir/pipeline/translated_codes_baseline_hecbench")
PARACODEX_ROOT = Path("/path/to/workdir/pipeline/translated_codes_hecbench")
BUILD_CMD = "make -f Makefile.nvc run"
CLEAN_CMD = "make -f Makefile.nvc clean"


@dataclass
class BenchmarkResult:
    name: str
    reference_ms: Optional[float]
    baseline_ms: Optional[float]
    paracodex_ms: Optional[float]


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


def find_benchmarks() -> List[str]:
    """
    Find all benchmarks from the paracodex directory.
    """
    benchmarks = []
    if not PARACODEX_ROOT.exists():
        return benchmarks
    
    for entry in PARACODEX_ROOT.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            # Check if it has main_optimized.cpp
            if (entry / "main_optimized.cpp").exists():
                benchmarks.append(entry.name)
    
    return sorted(benchmarks)


def run_bench_in_dir(work_dir: Path, sm: str = "cc89") -> Tuple[int, str]:
    """
    Build and run the benchmark using NV_ACC_TIME=1 (nvc++ built-in GPU profiler).

    NV_ACC_TIME=1 outputs per-region device timing to stderr without needing CUPTI,
    making it compatible with Modal's gVisor-based A100 containers where nsys
    CUPTI injection fails with UUID errors or segfaults.

    Combined stdout+stderr is returned for parsing.
    """
    print(f"Building in {work_dir}")

    # Check for Makefile.nvc
    mk = work_dir / "Makefile.nvc"
    if not mk.exists():
        return 2, f"Missing Makefile.nvc in {work_dir}"

    # Clean first
    subprocess.run(
        ["make", "-f", "Makefile.nvc", "clean", f"SM={sm}", "EXTRA_CFLAGS=-gpu=nomanaged"],
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Build with -gpu=nomanaged to avoid UVM segfaults inside Modal gVisor
    build_proc = subprocess.run(
        ["make", "-f", "Makefile.nvc", f"SM={sm}", "EXTRA_CFLAGS=-gpu=nomanaged"],
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if build_proc.returncode != 0:
        return build_proc.returncode, build_proc.stdout

    # Dry-run to get the actual run command (e.g. "./main 100")
    dry_run = subprocess.run(
        ["make", "-n", "-f", "Makefile.nvc", "run", f"SM={sm}", "EXTRA_CFLAGS=-gpu=nomanaged"],
        cwd=str(work_dir), stdout=subprocess.PIPE, text=True,
    )
    run_cmds = [ln.strip() for ln in dry_run.stdout.splitlines()
                if ln.strip() and not ln.startswith("make")]
    actual_run_cmd = run_cmds[-1] if run_cmds else "./main"

    # Set environment for GPU offloading + nvc++ built-in profiler
    env = os.environ.copy()
    env["FORCE_OMP_GPU"] = "1"
    env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
    # NV_ACC_TIME=1 activates nvc++ built-in timing; output goes to stderr
    env["NV_ACC_TIME"] = "1"
    env["NVCOMPILER_ACC_TIME"] = "1"

    proc = subprocess.run(
        ["sh", "-c", actual_run_cmd],
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Combine stdout + stderr so the parser sees both program output and timing
    combined = proc.stdout + "\n" + proc.stderr
    return proc.returncode, combined


def parse_total_gpu_time_ms(acc_output: str) -> Optional[float]:
    """
    Parse NV_ACC_TIME=1 output (nvc++ built-in profiler) and sum all device time.

    The output looks like:
        Accelerator Kernel Timing data
        /path/to/main.cpp
          main  NVIDIA  devicenum=0
            time(us): 1,523,310
            369: data region reached 2 times
                369: data copyin transfers: 2
                     device time(us): total=79,163 max=...
                402: data copyout transfers: 2
                     device time(us): total=1,444,147 max=...

    We sum ALL 'device time(us): total=N' values (kernels + transfers).
    Returns time in milliseconds.
    """
    if not acc_output:
        return None

    total_us = 0
    found = False
    # Match lines like:  device time(us): total=79,163 ...
    for m in re.finditer(r'device time\(us\):\s*total=([0-9,]+)', acc_output):
        try:
            total_us += int(m.group(1).replace(",", ""))
            found = True
        except ValueError:
            continue

    if found and total_us > 0:
        return total_us / 1e3  # us -> ms
    return None


def profile_mean_ms(work_dir: Path, runs: int = 2, temp_dir: Optional[Path] = None, sm: str = "cc89") -> Optional[float]:
    """
    Run the benchmark multiple times with NV_ACC_TIME=1 and return the mean GPU time (ms).
    """
    values: List[float] = []
    for run_idx in range(runs):
        rc, out = run_bench_in_dir(work_dir, sm=sm)
        ms = parse_total_gpu_time_ms(out)
        if ms is None and rc != 0:
            print(f"[WARN] run {run_idx + 1} failed with return code {rc} and no valid GPU time found")
            continue
        elif rc != 0:
            print(f"[WARN] run {run_idx + 1} had non-zero return code {rc} but GPU time was parsed")

        if ms is not None:
            values.append(ms)

    if values:
        return sum(values) / len(values)
    return None


def write_results_json(results: List[BenchmarkResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"

    serializable = []
    for r in results:
        result_dict = {
            "name": r.name,
            "reference_ms": r.reference_ms,
            "baseline_ms": r.baseline_ms,
            "paracodex_ms": r.paracodex_ms,
        }
        serializable.append(result_dict)
    
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)

    return json_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile baseline vs reference vs paracodex hecbench benchmarks using Nsight Systems (GPU time)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/path/to/workdir/cuda_omp_workdir/results_perf_nsys_baseline_hecbench",
        help="Output directory for results",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of profiling runs per benchmark (default: 2)",
    )
    parser.add_argument(
        "--sm",
        type=str,
        default="cc89",
        help="CUDA architecture target (default: cc89)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output).resolve()

    # Preflight: ensure required tools exist in PATH
    for tool in ("make", "nvc++"):
        if shutil.which(tool) is None:
            print(f"Required tool not found in PATH: {tool}", file=sys.stderr)
            return 2

    # Find all benchmarks
    benchmarks = find_benchmarks()
    if not benchmarks:
        print("No benchmarks found in paracodex directory", file=sys.stderr)
        return 1

    print(f"Found {len(benchmarks)} benchmarks: {', '.join(benchmarks)}")

    # Create temp directory for intermediate storage
    temp_dir = out_dir / "temp_nsys_outputs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    results: List[BenchmarkResult] = []

    for bench in benchmarks:
        print(f"\n=== Benchmark: {bench} ===")
        
        bench_dir = DATA_SRC_ROOT / bench
        if not bench_dir.exists():
            print(f"[WARN] {bench}: Benchmark directory not found: {bench_dir}, skipping")
            result = BenchmarkResult(bench, None, None, None)
            results.append(result)
            continue

        # Check for reference file (original OpenMP implementation from Hecbench repository)
        # Located in: REFERENCE_ROOT/<bench>-omp/main.cpp
        ref_source_file = REFERENCE_ROOT / bench / "main.cpp"
        if not ref_source_file.exists():
            print(f"[WARN] {bench}: Reference file (Hecbench OpenMP) not found: {ref_source_file}, skipping")
            result = BenchmarkResult(bench, None, None, None)
            results.append(result)
            continue
        
        # Copy reference file to work directory
        ref_file = bench_dir / "main.cpp"
        try:
            shutil.copy2(ref_source_file, ref_file)
            print(f"[INFO] {bench}: Copied reference file from {ref_source_file}")
        except Exception as e:
            print(f"[ERROR] {bench}: Failed to copy reference file: {e}")
            result = BenchmarkResult(bench, None, None, None)
            results.append(result)
            continue

        # Check for baseline file
        baseline_file = BASELINE_ROOT / bench / "main_optimized.cpp"
        if not baseline_file.exists():
            print(f"[WARN] {bench}: Baseline file not found: {baseline_file}, skipping baseline measurement")
            baseline_file = None

        # Check for paracodex file
        paracodex_file = PARACODEX_ROOT / bench / "main_optimized.cpp"
        if not paracodex_file.exists():
            print(f"[WARN] {bench}: ParaCodex file not found: {paracodex_file}, skipping paracodex measurement")
            paracodex_file = None

        # Measure reference (copied from Hecbench repository)
        print(f"[INFO] {bench}: Measuring reference (Hecbench OpenMP)...")
        ref_ms = profile_mean_ms(bench_dir, runs=args.runs, temp_dir=temp_dir, sm=args.sm)
        print(f"[INFO] {bench}: Reference GPU time: {ref_ms} ms" if ref_ms else f"[WARN] {bench}: Reference measurement failed")

        # Measure baseline
        baseline_ms = None
        if baseline_file:
            print(f"[INFO] {bench}: Measuring baseline...")
            # Backup original main.cpp
            backup_file = bench_dir / "main.cpp.backup"
            try:
                shutil.copy2(ref_file, backup_file)
                # Copy baseline file
                shutil.copy2(baseline_file, ref_file)
                baseline_ms = profile_mean_ms(bench_dir, runs=args.runs, temp_dir=temp_dir, sm=args.sm)
                print(f"[INFO] {bench}: Baseline GPU time: {baseline_ms} ms" if baseline_ms else f"[WARN] {bench}: Baseline measurement failed")
                # Restore original
                shutil.copy2(backup_file, ref_file)
                backup_file.unlink()
            except Exception as e:
                print(f"[ERROR] {bench}: Baseline measurement error: {e}")
                # Try to restore original
                if backup_file.exists():
                    shutil.copy2(backup_file, ref_file)
                    backup_file.unlink()

        # Measure paracodex
        paracodex_ms = None
        if paracodex_file:
            print(f"[INFO] {bench}: Measuring paracodex...")
            # Backup original main.cpp
            backup_file = bench_dir / "main.cpp.backup"
            try:
                shutil.copy2(ref_file, backup_file)
                # Copy paracodex file
                shutil.copy2(paracodex_file, ref_file)
                paracodex_ms = profile_mean_ms(bench_dir, runs=args.runs, temp_dir=temp_dir, sm=args.sm)
                print(f"[INFO] {bench}: ParaCodex GPU time: {paracodex_ms} ms" if paracodex_ms else f"[WARN] {bench}: ParaCodex measurement failed")
                # Restore original
                shutil.copy2(backup_file, ref_file)
                backup_file.unlink()
            except Exception as e:
                print(f"[ERROR] {bench}: ParaCodex measurement error: {e}")
                # Try to restore original
                if backup_file.exists():
                    shutil.copy2(backup_file, ref_file)
                    backup_file.unlink()

        # Restore reference file at the end (in case it was modified)
        try:
            shutil.copy2(ref_source_file, ref_file)
        except Exception as e:
            print(f"[WARN] {bench}: Failed to restore reference file: {e}")

        result = BenchmarkResult(
            name=bench,
            reference_ms=ref_ms,
            baseline_ms=baseline_ms,
            paracodex_ms=paracodex_ms,
        )
        results.append(result)
        print(f"[INFO] {bench}: reference={ref_ms} ms, baseline={baseline_ms} ms, paracodex={paracodex_ms} ms")

    # Write results
    json_path = write_results_json(results, out_dir)
    
    # Clean up temp directory
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n[INFO] Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"[WARN] Failed to clean up temp directory: {e}")

    print(f"\nWrote {json_path}")
    print(f"Processed {len(results)} benchmarks")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

