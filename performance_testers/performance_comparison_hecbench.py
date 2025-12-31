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
# Reference files are the original OpenMP implementations from Hecbench repository
# They are located in: /root/codex_baseline/hecbench_reference/src/<bench>-omp/main.cpp
REFERENCE_ROOT = Path("/root/codex_baseline/hecbench_reference/src")
BASELINE_ROOT = Path("/root/codex_baseline/pipeline/translated_codes_baseline_hecbench")
PARACODEX_ROOT = Path("/root/codex_baseline/pipeline/translated_codes_hecbench")
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


def run_nsys_in_dir(work_dir: Path, save_output_to: Optional[Path] = None) -> Tuple[int, str]:
    """
    Run Nsight Systems on `make -f Makefile.nvc run` in the given directory,
    capturing stdout which includes the cuda_gpu_kern_sum stats.
    
    If save_output_to is provided, the nsys output will be saved to that file
    to reduce memory usage, and only a summary will be kept in memory.
    """
    print(f"Making clean in {work_dir}")
    
    # Check for Makefile.nvc
    mk = work_dir / "Makefile.nvc"
    if not mk.exists():
        return 2, f"Missing Makefile.nvc in {work_dir}"

    # Clean first
    clean_cmd = ["make", "-f", "Makefile.nvc", "clean"]
    subprocess.run(
        clean_cmd,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Nsight Systems profile command
    # Format: FORCE_OMP_GPU=1 OMP_TARGET_OFFLOAD=MANDATORY nsys profile ... make -f Makefile.nvc run
    cmd = [
        "sh",
        "-c",
        f"FORCE_OMP_GPU=1 OMP_TARGET_OFFLOAD=MANDATORY nsys profile --stats=true --trace=cuda,osrt "
        f"--force-overwrite=true -o nsys_profile {BUILD_CMD}",
    ]

    # Set environment variables for OpenMP GPU offloading
    env = os.environ.copy()
    env["FORCE_OMP_GPU"] = "1"
    env["OMP_TARGET_OFFLOAD"] = "MANDATORY"

    # If saving to file, write directly to file to avoid keeping large output in memory
    if save_output_to:
        save_output_to.parent.mkdir(parents=True, exist_ok=True)
        with open(save_output_to, "w") as f:
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        # Read back only what we need for parsing
        with open(save_output_to, "r") as f:
            output = f.read()
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(work_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = proc.stdout
    
    clean_nsys_artifacts(work_dir)

    # Kill GPU processes after nsys run
    kill_script = Path("/root/codex_baseline/kill_gpu_processes.py")
    if kill_script.exists():
        print(f"Running {kill_script} to clean up GPU processes...")
        subprocess.run(
            [sys.executable, str(kill_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return proc.returncode, output


def parse_total_gpu_time_ms(nsys_output: str) -> Optional[float]:
    """
    Parse Nsight Systems stdout and compute GPU time (kernel + memory).
    Returns time in milliseconds.
    """
    if not nsys_output:
        return None

    lines = nsys_output.splitlines()
    kernel_ns = 0
    memory_ns = 0

    # Parse GPU kernel times from cuda_gpu_kern_sum
    in_kernel_table = False
    kernel_parsed = False
    for line in lines:
        if "cuda_gpu_kern_sum" in line or "CUDA GPU Kernel Summary" in line:
            in_kernel_table = True
            continue

        if not in_kernel_table:
            continue

        if not line.strip():
            if kernel_parsed:
                break
            continue

        m = re.match(
            r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9,]+)\s+([0-9,]+)\s+",
            line,
        )
        if not m:
            if kernel_parsed:
                break
            continue

        _, total_time_ns_str, _ = m.groups()
        try:
            kernel_ns += int(total_time_ns_str.replace(",", ""))
            kernel_parsed = True
        except ValueError:
            continue

    # Parse GPU memory transfer times from cuda_gpu_mem_time_sum
    in_mem_table = False
    mem_parsed = False
    for line in lines:
        if "cuda_gpu_mem_time_sum" in line or "CUDA GPU Memory Time Summary" in line:
            in_mem_table = True
            continue

        if not in_mem_table:
            continue

        if not line.strip():
            if mem_parsed:
                break
            continue

        m = re.match(
            r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9,]+)\s+([0-9,]+)\s+",
            line,
        )
        if not m:
            if "Time (%)" in line or "--------" in line:
                continue
            if mem_parsed:
                break
            continue

        _, total_time_ns_str, _ = m.groups()
        try:
            mem_time = int(total_time_ns_str.replace(",", ""))
            memory_ns += mem_time
            mem_parsed = True
        except ValueError:
            continue

    # Sum kernel + memory for GPU time
    total_ns = kernel_ns + memory_ns

    if total_ns > 0:
        return total_ns / 1e6  # ns -> ms
    return None


def profile_mean_ms(work_dir: Path, runs: int = 2, temp_dir: Optional[Path] = None) -> Optional[float]:
    """
    Run Nsight Systems multiple times and return the mean GPU time (ms).
    
    If temp_dir is provided, nsys output will be saved to disk to reduce memory usage.
    """
    values: List[float] = []
    for run_idx in range(runs):
        # Save output to temp file if temp_dir is provided
        save_path = None
        if temp_dir:
            save_path = temp_dir / f"nsys_output_{work_dir.name}_run{run_idx}.txt"
        
        rc, out = run_nsys_in_dir(work_dir, save_output_to=save_path)
        # Try to parse even if return code is non-zero, as nsys might have valid stats
        # even if the program crashes during cleanup (e.g., meanshift-omp)
        ms = parse_total_gpu_time_ms(out)
        if ms is None and rc != 0:
            print(f"[WARN] nsys run {run_idx + 1} failed with return code {rc} and no valid stats found")
            continue
        elif rc != 0:
            print(f"[WARN] nsys run {run_idx + 1} had non-zero return code {rc} but stats were parsed successfully")
        
        # Clear output from memory after parsing
        del out
        
        if ms is not None:
            values.append(ms)
        
        # Clean up temp file if it exists
        if save_path and save_path.exists():
            try:
                save_path.unlink()
            except Exception:
                pass
    
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
        default="/root/codex_baseline/cuda_omp_workdir/results_perf_nsys_baseline_hecbench",
        help="Output directory for results",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of profiling runs per benchmark (default: 2)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output).resolve()

    # Preflight: ensure required tools exist in PATH
    for tool in ("nsys", "make", "nvc++"):
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
        ref_ms = profile_mean_ms(bench_dir, runs=args.runs, temp_dir=temp_dir)
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
                baseline_ms = profile_mean_ms(bench_dir, runs=args.runs, temp_dir=temp_dir)
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
                paracodex_ms = profile_mean_ms(bench_dir, runs=args.runs, temp_dir=temp_dir)
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

