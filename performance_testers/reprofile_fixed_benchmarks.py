#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DATA_SRC_ROOT = Path("/root/codex_baseline/cuda_omp_workdir/data/src")
REFERENCE_ROOT = Path("/root/codex_baseline/hecbench_reference/src")
RESULTS_JSON = Path("/root/codex_baseline/cuda_omp_workdir/results_perf_nsys_baseline_hecbench/results.json")

# Benchmarks to reprofile
BENCHMARKS_TO_FIX = ["atomicIntrinsics-omp", "jacobi-omp", "meanshift-omp", "stencil1d-omp"]


def clean_nsys_artifacts(work_dir: Path):
    """Remove Nsight Systems artifacts."""
    for pattern in ["nsys_profile*", "*.qdstrm*", "*.nsys-rep*", "*.sqlite*"]:
        for file in work_dir.glob(pattern):
            try:
                file.unlink()
            except Exception:
                pass


def parse_total_gpu_time_ms(nsys_output: str) -> float:
    """Parse GPU time (kernel + memory) from nsys output."""
    if not nsys_output:
        return None

    lines = nsys_output.splitlines()
    kernel_ns = 0
    memory_ns = 0

    # Parse GPU kernel times
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

        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9,]+)\s+([0-9,]+)\s+", line)
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

    # Parse GPU memory transfer times
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

        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9,]+)\s+([0-9,]+)\s+", line)
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


def find_reference_file(bench_name: str) -> Path:
    """Find the reference source file, handling different file names."""
    ref_dir = REFERENCE_ROOT / bench_name
    if not ref_dir.exists():
        return None
    
    # Try common file names
    for filename in ["main.cpp", "heat.cpp", "stencil_1d.cpp"]:
        ref_file = ref_dir / filename
        if ref_file.exists():
            return ref_file
    return None


def profile_benchmark(bench_name: str, runs: int = 2) -> float:
    """Profile a benchmark and return mean GPU time in ms."""
    bench_dir = DATA_SRC_ROOT / bench_name
    ref_source_file = find_reference_file(bench_name)
    ref_file = bench_dir / "main.cpp"

    if not bench_dir.exists():
        print(f"[ERROR] {bench_name}: Benchmark directory not found: {bench_dir}")
        return None

    if not ref_source_file:
        print(f"[ERROR] {bench_name}: Reference file not found in {REFERENCE_ROOT / bench_name}")
        return None

    # Copy reference file (handle different source file names)
    try:
        if ref_source_file.name != "main.cpp":
            shutil.copy2(ref_source_file, ref_file)
            print(f"[INFO] {bench_name}: Copied {ref_source_file.name} to main.cpp")
        else:
            shutil.copy2(ref_source_file, ref_file)
            print(f"[INFO] {bench_name}: Copied reference file")
    except Exception as e:
        print(f"[ERROR] {bench_name}: Failed to copy reference file: {e}")
        return None

    # Clean first
    clean_cmd = ["make", "-f", "Makefile.nvc", "clean"]
    subprocess.run(
        clean_cmd,
        cwd=str(bench_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    values = []
    for run_idx in range(runs):
        print(f"[INFO] {bench_name}: Running nsys profile (run {run_idx + 1}/{runs})...")
        
        # Run nsys
        cmd = [
            "sh",
            "-c",
            f"FORCE_OMP_GPU=1 OMP_TARGET_OFFLOAD=MANDATORY nsys profile --stats=true --trace=cuda,osrt "
            f"--force-overwrite=true -o nsys_profile make -f Makefile.nvc run",
        ]

        env = os.environ.copy()
        env["FORCE_OMP_GPU"] = "1"
        env["OMP_TARGET_OFFLOAD"] = "MANDATORY"

        proc = subprocess.run(
            cmd,
            cwd=str(bench_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Check if there are GPU kernels
        no_gpu_kernels = "does not contain CUDA trace data" in proc.stdout or "does not contain CUDA kernel data" in proc.stdout
        
        # Parse output even if return code is non-zero (cleanup errors are OK)
        ms = parse_total_gpu_time_ms(proc.stdout)
        if ms is not None:
            values.append(ms)
            print(f"[INFO] {bench_name}: Run {run_idx + 1} GPU time: {ms:.6f} ms")
        elif no_gpu_kernels:
            # No GPU kernels - GPU time is 0
            values.append(0.0)
            print(f"[INFO] {bench_name}: Run {run_idx + 1} - No GPU kernels detected, GPU time: 0.0 ms")
        else:
            print(f"[WARN] {bench_name}: Run {run_idx + 1} failed to parse stats (return code: {proc.returncode})")

        clean_nsys_artifacts(bench_dir)

        # Kill GPU processes
        kill_script = Path("/root/codex_baseline/kill_gpu_processes.py")
        if kill_script.exists():
            subprocess.run(
                [sys.executable, str(kill_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

    if values:
        mean_ms = sum(values) / len(values)
        print(f"[INFO] {bench_name}: Mean GPU time: {mean_ms:.6f} ms")
        return mean_ms
    else:
        print(f"[ERROR] {bench_name}: No valid measurements obtained")
        return None


def main():
    # Load results
    with open(RESULTS_JSON, "r") as f:
        results = json.load(f)

    # Find benchmarks to fix
    benchmarks_to_fix = [r for r in results if r["name"] in BENCHMARKS_TO_FIX]
    
    if not benchmarks_to_fix:
        print("No benchmarks to fix found.")
        return 0

    print(f"Reprofiling {len(benchmarks_to_fix)} benchmarks:")
    for r in benchmarks_to_fix:
        print(f"  - {r['name']} (current: {r.get('reference_ms')})")

    # Profile each benchmark
    for result in benchmarks_to_fix:
        bench_name = result["name"]
        print(f"\n{'='*60}")
        print(f"Profiling {bench_name}")
        print(f"{'='*60}")
        
        ref_ms = profile_benchmark(bench_name, runs=2)
        if ref_ms is not None:
            result["reference_ms"] = ref_ms
            print(f"[SUCCESS] {bench_name}: Updated reference_ms = {ref_ms:.6f} ms")
        else:
            print(f"[FAILED] {bench_name}: Could not obtain reference_ms")

    # Save updated results
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Updated results saved to:", RESULTS_JSON)
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



