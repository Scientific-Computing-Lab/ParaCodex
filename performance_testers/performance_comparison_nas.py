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


DATA_SRC_ROOT = Path("/root/codex_baseline/serial_omp_nas_workdir/data/src")
BUILD_CMD = "make CC=nvc++ CLASS=EREL run"
CLEAN_CMD = "make clean"


@dataclass
class BenchmarkResult:
    name: str
    reference_ms: Optional[float]
    candidate_ms: Optional[float]
    reference_log: Optional[str]
    candidate_log: Optional[str]
    # Additional measurements when --measure_all is used
    reference_kernel_ms: Optional[float] = None
    reference_gpu_ms: Optional[float] = None
    reference_full_ms: Optional[float] = None
    candidate_kernel_ms: Optional[float] = None
    candidate_gpu_ms: Optional[float] = None
    candidate_full_ms: Optional[float] = None
    # Baseline measurements (optional)
    baseline_ms: Optional[float] = None
    baseline_kernel_ms: Optional[float] = None
    baseline_gpu_ms: Optional[float] = None
    baseline_full_ms: Optional[float] = None
    # Step comparison mode (step1 vs step2 vs reference vs baseline)
    step1_ms: Optional[float] = None
    step2_ms: Optional[float] = None
    # Additional measurements for step comparison when --measure_all is used
    reference_kernel_ms_step: Optional[float] = None
    reference_gpu_ms_step: Optional[float] = None
    reference_full_ms_step: Optional[float] = None
    step1_kernel_ms: Optional[float] = None
    step1_gpu_ms: Optional[float] = None
    step1_full_ms: Optional[float] = None
    step2_kernel_ms: Optional[float] = None
    step2_gpu_ms: Optional[float] = None
    step2_full_ms: Optional[float] = None
    # Baseline measurements for step comparison mode
    baseline_kernel_ms_step: Optional[float] = None
    baseline_gpu_ms_step: Optional[float] = None
    baseline_full_ms_step: Optional[float] = None


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


def find_benchmark_source_file(bench_dir: Path) -> Optional[Path]:
    """
    Find the main source file in a NAS benchmark directory.
    Looks for files matching <benchmark_name>.c pattern.
    """
    if not bench_dir.exists() or not bench_dir.is_dir():
        return None
    
    # Extract benchmark name from directory (e.g., "cg-omp" -> "cg")
    bench_name = bench_dir.name.split("-")[0]
    
    # Look for the main source file
    candidate = bench_dir / f"{bench_name}.c"
    if candidate.exists():
        return candidate
    
    return None


def find_reference_files(reference_dir: Path) -> Dict[str, Path]:
    """
    Find reference source files.
    Looks directly in <reference_dir>/<bench_name>/<kernel_name>.c
    Returns mapping: bench_name -> source_file_path
    """
    mapping: Dict[str, Path] = {}
    if not reference_dir.exists():
        return mapping
    
    for entry in reference_dir.iterdir():
        if not entry.is_dir():
            continue
        
        # Extract kernel name from bench name (e.g., "ft-omp" -> "ft")
        bench_name = entry.name.split("-")[0]
        kernel_file = entry / f"{bench_name}.c"
        
        if kernel_file.exists():
            mapping[entry.name] = kernel_file
    
    return mapping


def find_baseline_files(baseline_dir: Path) -> Dict[str, Path]:
    """
    Find baseline source files from optimized/ subdirectory.
    Looks in <baseline_dir>/<bench_name>/optimized/<kernel_name>.c
    Returns mapping: bench_name -> source_file_path
    """
    mapping: Dict[str, Path] = {}
    if not baseline_dir.exists():
        return mapping
    
    for entry in baseline_dir.iterdir():
        if not entry.is_dir():
            continue
        
        # Extract kernel name from bench name (e.g., "ft-omp" -> "ft")
        bench_name = entry.name.split("-")[0]
        optimized_file = entry / "optimized" / f"{bench_name}.c"
        
        if optimized_file.exists():
            mapping[entry.name] = optimized_file
    
    return mapping


def find_subdirs_with_source_files(base_dir: Path) -> Dict[str, Path]:
    """
    Find subdirectories containing source files (any .c file).
    Looks in the directory itself and in subdirectories (e.g., optimized/).
    Returns mapping: bench_name -> source_file_path
    """
    mapping: Dict[str, Path] = {}
    if not base_dir.exists():
        return mapping
    
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        
        # Look for .c files directly in the directory
        c_files_direct = list(entry.glob("*.c"))
        
        # Also look in subdirectories (e.g., optimized/, step1/, etc.)
        # Use rglob to find .c files in subdirectories, but exclude the direct ones
        c_files_subdirs = [f for f in entry.rglob("*.c") if f.parent != entry]
        
        # Combine and remove duplicates
        all_c_files = list(set(c_files_direct + c_files_subdirs))
        
        if all_c_files:
            # For candidate files, prefer files in "optimized/" subdirectory
            optimized_dir = [f for f in all_c_files if "optimized" in str(f)]
            if optimized_dir:
                # Prefer files named like "cg.c" or "ep.c" (base name) in optimized/
                bench_name = entry.name.split("-")[0]
                preferred = [f for f in optimized_dir if f.name == f"{bench_name}.c"]
                if preferred:
                    mapping[entry.name] = preferred[0]
                else:
                    mapping[entry.name] = optimized_dir[0]
            else:
                # Use the first .c file found, preferring files with base name
                bench_name = entry.name.split("-")[0]
                preferred = [f for f in all_c_files if f.name == f"{bench_name}.c"]
                if preferred:
                    mapping[entry.name] = preferred[0]
                else:
                    mapping[entry.name] = all_c_files[0]
    
    return mapping


def find_steps_in_pipeline_dir(pipeline_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Find step1 and step2 files in pipeline directory structure.
    Expected structure: pipeline_dir/<bench_name>/step1/<bench>.c and step2/<bench>.c
    
    Returns mapping: bench_name -> {"step1": path, "step2": path}
    """
    mapping: Dict[str, Dict[str, Path]] = {}
    if not pipeline_dir.exists():
        return mapping
    
    for entry in pipeline_dir.iterdir():
        if not entry.is_dir():
            continue
        
        bench_name = entry.name.split("-")[0]
        step1_file = entry / "step1" / f"{bench_name}.c"
        step2_file = entry / "step2" / f"{bench_name}.c"
        
        steps = {}
        if step1_file.exists():
            steps["step1"] = step1_file
        if step2_file.exists():
            steps["step2"] = step2_file
        
        if steps:
            mapping[entry.name] = steps
    
    return mapping


def get_target_source_filename(bench_name: str) -> str:
    """
    Get the target source filename for a benchmark.
    e.g., "cg-omp" -> "cg.c", "ep-omp" -> "ep.c"
    """
    base_name = bench_name.split("-")[0]
    return f"{base_name}.c"


def copy_candidate_to_datasrc(bench_name: str, candidate_file: Path) -> Path:
    """
    Copy candidate source file to the NAS benchmark directory.
    Returns the target project directory.
    """
    target_project_dir = DATA_SRC_ROOT / bench_name
    if not target_project_dir.exists():
        raise FileNotFoundError(f"Target benchmark directory not found: {target_project_dir}")
    
    if not candidate_file.exists():
        raise FileNotFoundError(f"Missing candidate file: {candidate_file}")
    
    # Determine target filename (e.g., cg_optimized.c -> cg.c)
    target_filename = get_target_source_filename(bench_name)
    dst = target_project_dir / target_filename
    
    shutil.copy2(candidate_file, dst)
    return target_project_dir


def run_nsys_in_dir(work_dir: Path, save_output_to: Optional[Path] = None) -> Tuple[int, str]:
    """
    Run Nsight Systems on `make CC=nvc++ CLASS=C run` in the given directory,
    capturing stdout which includes the cuda_gpu_kern_sum stats.
    
    If save_output_to is provided, the nsys output will be saved to that file
    to reduce memory usage, and only a summary will be kept in memory.
    """
    print(f"Making clean in {work_dir}")
    
    # Check for Makefile
    mk = work_dir / "Makefile"
    if not mk.exists():
        return 2, f"Missing Makefile in {work_dir}"

    # Clean first
    clean_cmd = ["make", "clean"]
    subprocess.run(
        clean_cmd,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Nsight Systems profile command
    # Format: FORCE_OMP_GPU=1 OMP_TARGET_OFFLOAD=MANDATORY nsys profile ... make CC=nvc++ CLASS=C run
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
        # Read back only what we need for parsing (or return a marker)
        # For now, we'll still need to read it for parsing, but at least it's on disk
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


def parse_total_gpu_time_ms(nsys_output: str, gpu_time: bool = False, full_time: bool = False) -> Optional[float]:
    """
    Parse Nsight Systems stdout and compute total time.

    When both gpu_time and full_time are False, only sums kernel time (cuda_gpu_kern_sum).
    When gpu_time is True, adds GPU memory transfer time (cuda_gpu_mem_time_sum).
    When full_time is True, includes everything: kernels + memory + CUDA API + OS runtime.
    """
    if not nsys_output:
        return None

    lines = nsys_output.splitlines()
    kernel_ns = 0
    memory_ns = 0
    api_ns = 0
    osrt_ns = 0

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
    if gpu_time or full_time:
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

    # Parse CUDA API times from cuda_api_sum
    if full_time:
        in_api_table = False
        api_parsed = False
        for line in lines:
            if "cuda_api_sum" in line or "CUDA API Summary" in line:
                in_api_table = True
                continue

            if not in_api_table:
                continue

            if not line.strip():
                if api_parsed:
                    break
                continue

            m = re.match(
                r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9,]+)\s+([0-9,]+)\s+",
                line,
            )
            if not m:
                if "Time (%)" in line or "--------" in line:
                    continue
                if api_parsed:
                    break
                continue

            _, total_time_ns_str, _ = m.groups()
            try:
                api_time = int(total_time_ns_str.replace(",", ""))
                api_ns += api_time
                api_parsed = True
            except ValueError:
                continue

    # Parse OS runtime times from osrt_sum
    if full_time:
        in_osrt_table = False
        osrt_parsed = False
        for line in lines:
            if "osrt_sum" in line or "OS Runtime Summary" in line:
                in_osrt_table = True
                continue

            if not in_osrt_table:
                continue

            if not line.strip():
                if osrt_parsed:
                    break
                continue

            m = re.match(
                r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9,]+)\s+([0-9,]+)\s+",
                line,
            )
            if not m:
                if "Time (%)" in line or "--------" in line:
                    continue
                if osrt_parsed:
                    break
                continue

            _, total_time_ns_str, _ = m.groups()
            try:
                osrt_time = int(total_time_ns_str.replace(",", ""))
                osrt_ns += osrt_time
                osrt_parsed = True
            except ValueError:
                continue

    # Sum up the appropriate components
    total_ns = kernel_ns
    if gpu_time or full_time:
        total_ns += memory_ns
    if full_time:
        total_ns += api_ns
        total_ns += osrt_ns

    if total_ns > 0:
        return total_ns / 1e6  # ns -> ms
    return None


def profile_mean_ms(work_dir: Path, runs: int = 3, gpu_time: bool = False, full_time: bool = False, temp_dir: Optional[Path] = None) -> Optional[float]:
    """
    Run Nsight Systems multiple times and return the mean total time (ms).
    
    If temp_dir is provided, nsys output will be saved to disk to reduce memory usage.
    """
    values: List[float] = []
    for run_idx in range(runs):
        # Save output to temp file if temp_dir is provided
        save_path = None
        if temp_dir:
            save_path = temp_dir / f"nsys_output_{work_dir.name}_run{run_idx}.txt"
        
        rc, out = run_nsys_in_dir(work_dir, save_output_to=save_path)
        if rc != 0:
            continue
        ms = parse_total_gpu_time_ms(out, gpu_time=gpu_time, full_time=full_time)
        
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


def profile_all_measurements(work_dir: Path, runs: int = 2, temp_dir: Optional[Path] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Run Nsight Systems multiple times and return all three measurement types:
    (kernel_ms, gpu_ms, full_ms)
    This is more efficient than running nsys separately for each measurement type.
    
    If temp_dir is provided, nsys output will be saved to disk to reduce memory usage.
    """
    kernel_values: List[float] = []
    gpu_values: List[float] = []
    full_values: List[float] = []
    
    for run_idx in range(runs):
        # Save output to temp file if temp_dir is provided
        save_path = None
        if temp_dir:
            save_path = temp_dir / f"nsys_output_{work_dir.name}_run{run_idx}.txt"
        
        rc, out = run_nsys_in_dir(work_dir, save_output_to=save_path)
        if rc != 0:
            continue
        
        # Parse all three measurements from the same nsys output
        kernel_ms = parse_total_gpu_time_ms(out, gpu_time=False, full_time=False)
        gpu_ms = parse_total_gpu_time_ms(out, gpu_time=True, full_time=False)
        full_ms = parse_total_gpu_time_ms(out, gpu_time=False, full_time=True)
        
        # Debug output to verify parsing
        if kernel_ms is not None and gpu_ms is not None and full_ms is not None:
            if abs(kernel_ms - gpu_ms) < 0.01 and abs(gpu_ms - full_ms) < 0.01:
                print(f"[WARN] All measurements are identical (likely parsing issue): kernel={kernel_ms:.2f}, gpu={gpu_ms:.2f}, full={full_ms:.2f}")
        
        # Clear output from memory after parsing
        del out
        
        if kernel_ms is not None:
            kernel_values.append(kernel_ms)
        if gpu_ms is not None:
            gpu_values.append(gpu_ms)
        if full_ms is not None:
            full_values.append(full_ms)
        
        # Clean up temp file if it exists
        if save_path and save_path.exists():
            try:
                save_path.unlink()
            except Exception:
                pass
    
    kernel_mean = sum(kernel_values) / len(kernel_values) if kernel_values else None
    gpu_mean = sum(gpu_values) / len(gpu_values) if gpu_values else None
    full_mean = sum(full_values) / len(full_values) if full_values else None
    
    return kernel_mean, gpu_mean, full_mean


def write_results_json_txt(results: List[BenchmarkResult], out_dir: Path, measure_all: bool = False, has_baseline: bool = False, step_comparison: bool = False) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    txt_path = out_dir / "results.txt"

    serializable = []
    for r in results:
        if step_comparison:
            result_dict = {
                "name": r.name,
                "reference_ms": r.reference_ms,
                "step1_ms": r.step1_ms,
                "step2_ms": r.step2_ms,
            }
            if has_baseline:
                result_dict["baseline_ms"] = r.baseline_ms
            if measure_all:
                result_dict.update({
                    "reference_kernel_ms": r.reference_kernel_ms_step,
                    "reference_gpu_ms": r.reference_gpu_ms_step,
                    "reference_full_ms": r.reference_full_ms_step,
                    "step1_kernel_ms": r.step1_kernel_ms,
                    "step1_gpu_ms": r.step1_gpu_ms,
                    "step1_full_ms": r.step1_full_ms,
                    "step2_kernel_ms": r.step2_kernel_ms,
                    "step2_gpu_ms": r.step2_gpu_ms,
                    "step2_full_ms": r.step2_full_ms,
                })
                if has_baseline:
                    result_dict.update({
                        "baseline_kernel_ms": r.baseline_kernel_ms_step,
                        "baseline_gpu_ms": r.baseline_gpu_ms_step,
                        "baseline_full_ms": r.baseline_full_ms_step,
                    })
        else:
            result_dict = {
                "name": r.name,
                "reference_ms": r.reference_ms,
                "candidate_ms": r.candidate_ms,
            }
            if has_baseline:
                result_dict["baseline_ms"] = r.baseline_ms
        if measure_all and not step_comparison:
            result_dict.update({
                "reference_kernel_ms": r.reference_kernel_ms,
                "reference_gpu_ms": r.reference_gpu_ms,
                "reference_full_ms": r.reference_full_ms,
                "candidate_kernel_ms": r.candidate_kernel_ms,
                "candidate_gpu_ms": r.candidate_gpu_ms,
                "candidate_full_ms": r.candidate_full_ms,
            })
            if has_baseline:
                result_dict.update({
                    "baseline_kernel_ms": r.baseline_kernel_ms,
                    "baseline_gpu_ms": r.baseline_gpu_ms,
                    "baseline_full_ms": r.baseline_full_ms,
                })
        serializable.append(result_dict)
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)

    with open(txt_path, "w") as f:
        for r in results:
            if step_comparison:
                if measure_all:
                    ref_kernel_str = f"{r.reference_kernel_ms_step / 1000.0:.4f} s" if r.reference_kernel_ms_step is not None else "N/A"
                    step1_kernel_str = f"{r.step1_kernel_ms / 1000.0:.4f} s" if r.step1_kernel_ms is not None else "N/A"
                    step2_kernel_str = f"{r.step2_kernel_ms / 1000.0:.4f} s" if r.step2_kernel_ms is not None else "N/A"
                    baseline_kernel_str = f"{r.baseline_kernel_ms_step / 1000.0:.4f} s" if r.baseline_kernel_ms_step is not None else "N/A"
                    ref_gpu_str = f"{r.reference_gpu_ms_step / 1000.0:.4f} s" if r.reference_gpu_ms_step is not None else "N/A"
                    step1_gpu_str = f"{r.step1_gpu_ms / 1000.0:.4f} s" if r.step1_gpu_ms is not None else "N/A"
                    step2_gpu_str = f"{r.step2_gpu_ms / 1000.0:.4f} s" if r.step2_gpu_ms is not None else "N/A"
                    baseline_gpu_str = f"{r.baseline_gpu_ms_step / 1000.0:.4f} s" if r.baseline_gpu_ms_step is not None else "N/A"
                    ref_full_str = f"{r.reference_full_ms_step / 1000.0:.4f} s" if r.reference_full_ms_step is not None else "N/A"
                    step1_full_str = f"{r.step1_full_ms / 1000.0:.4f} s" if r.step1_full_ms is not None else "N/A"
                    step2_full_str = f"{r.step2_full_ms / 1000.0:.4f} s" if r.step2_full_ms is not None else "N/A"
                    baseline_full_str = f"{r.baseline_full_ms_step / 1000.0:.4f} s" if r.baseline_full_ms_step is not None else "N/A"
                    f.write(f"{r.name}:\n")
                    baseline_kernel_part = f", baseline={baseline_kernel_str}" if has_baseline else ""
                    baseline_gpu_part = f", baseline={baseline_gpu_str}" if has_baseline else ""
                    baseline_full_part = f", baseline={baseline_full_str}" if has_baseline else ""
                    f.write(f"  Kernel: reference={ref_kernel_str}, step1={step1_kernel_str}, step2={step2_kernel_str}{baseline_kernel_part}\n")
                    f.write(f"  GPU: reference={ref_gpu_str}, step1={step1_gpu_str}, step2={step2_gpu_str}{baseline_gpu_part}\n")
                    f.write(f"  Full: reference={ref_full_str}, step1={step1_full_str}, step2={step2_full_str}{baseline_full_part}\n")
                else:
                    ref_str = f"{r.reference_ms / 1000.0:.4f} s" if r.reference_ms is not None else "N/A"
                    step1_str = f"{r.step1_ms / 1000.0:.4f} s" if r.step1_ms is not None else "N/A"
                    step2_str = f"{r.step2_ms / 1000.0:.4f} s" if r.step2_ms is not None else "N/A"
                    baseline_str = f", baseline={r.baseline_ms / 1000.0:.4f} s" if has_baseline and r.baseline_ms is not None else ", baseline=N/A" if has_baseline else ""
                    f.write(f"{r.name}: reference={ref_str}, step1={step1_str}, step2={step2_str}{baseline_str}\n")
            elif measure_all:
                baseline_kernel_str = f"{r.baseline_kernel_ms} ms" if r.baseline_kernel_ms is not None else "N/A"
                baseline_gpu_str = f"{r.baseline_gpu_ms} ms" if r.baseline_gpu_ms is not None else "N/A"
                baseline_full_str = f"{r.baseline_full_ms} ms" if r.baseline_full_ms is not None else "N/A"
                f.write(f"{r.name}:\n")
                f.write(f"  Kernel: reference={r.reference_kernel_ms} ms, ParaCodex={r.candidate_kernel_ms} ms")
                if has_baseline:
                    f.write(f", codex={baseline_kernel_str}")
                f.write("\n")
                f.write(f"  GPU: reference={r.reference_gpu_ms} ms, ParaCodex={r.candidate_gpu_ms} ms")
                if has_baseline:
                    f.write(f", codex={baseline_gpu_str}")
                f.write("\n")
                f.write(f"  Full: reference={r.reference_full_ms} ms, ParaCodex={r.candidate_full_ms} ms")
                if has_baseline:
                    f.write(f", codex={baseline_full_str}")
                f.write("\n")
            else:
                baseline_str = f", codex={r.baseline_ms} ms" if r.baseline_ms is not None else ", codex=N/A" if has_baseline else ""
                f.write(
                    f"{r.name}: reference={r.reference_ms} ms, ParaCodex={r.candidate_ms} ms{baseline_str}\n"
                )

    return json_path, txt_path


def append_result_to_file(result: BenchmarkResult, out_dir: Path, measure_all: bool = False, has_baseline: bool = False, step_comparison: bool = False) -> None:
    """
    Append a single result to the intermediate results file.
    This allows incremental writing to reduce memory usage.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results_intermediate.json"
    txt_path = out_dir / "results_intermediate.txt"
    
    # Append to JSON (read existing, append, write back)
    existing_results = []
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_results = []
    
    if step_comparison:
        result_dict = {
            "name": result.name,
            "reference_ms": result.reference_ms,
            "step1_ms": result.step1_ms,
            "step2_ms": result.step2_ms,
        }
        if has_baseline:
            result_dict["baseline_ms"] = result.baseline_ms
        if measure_all:
            result_dict.update({
                "reference_kernel_ms": result.reference_kernel_ms_step,
                "reference_gpu_ms": result.reference_gpu_ms_step,
                "reference_full_ms": result.reference_full_ms_step,
                "step1_kernel_ms": result.step1_kernel_ms,
                "step1_gpu_ms": result.step1_gpu_ms,
                "step1_full_ms": result.step1_full_ms,
                "step2_kernel_ms": result.step2_kernel_ms,
                "step2_gpu_ms": result.step2_gpu_ms,
                "step2_full_ms": result.step2_full_ms,
            })
            if has_baseline:
                result_dict.update({
                    "baseline_kernel_ms": result.baseline_kernel_ms_step,
                    "baseline_gpu_ms": result.baseline_gpu_ms_step,
                    "baseline_full_ms": result.baseline_full_ms_step,
                })
    else:
        result_dict = {
            "name": result.name,
            "reference_ms": result.reference_ms,
            "candidate_ms": result.candidate_ms,
        }
        if has_baseline:
            result_dict["baseline_ms"] = result.baseline_ms
    if measure_all and not step_comparison:
        result_dict.update({
            "reference_kernel_ms": result.reference_kernel_ms,
            "reference_gpu_ms": result.reference_gpu_ms,
            "reference_full_ms": result.reference_full_ms,
            "candidate_kernel_ms": result.candidate_kernel_ms,
            "candidate_gpu_ms": result.candidate_gpu_ms,
            "candidate_full_ms": result.candidate_full_ms,
        })
        if has_baseline:
            result_dict.update({
                "baseline_kernel_ms": result.baseline_kernel_ms,
                "baseline_gpu_ms": result.baseline_gpu_ms,
                "baseline_full_ms": result.baseline_full_ms,
            })
    existing_results.append(result_dict)
    
    with open(json_path, "w") as f:
        json.dump(existing_results, f, indent=2)
    
    # Append to text file
    with open(txt_path, "a") as f:
        if step_comparison:
            if measure_all:
                ref_kernel_str = f"{result.reference_kernel_ms_step / 1000.0:.4f} s" if result.reference_kernel_ms_step is not None else "N/A"
                step1_kernel_str = f"{result.step1_kernel_ms / 1000.0:.4f} s" if result.step1_kernel_ms is not None else "N/A"
                step2_kernel_str = f"{result.step2_kernel_ms / 1000.0:.4f} s" if result.step2_kernel_ms is not None else "N/A"
                baseline_kernel_str = f"{result.baseline_kernel_ms_step / 1000.0:.4f} s" if result.baseline_kernel_ms_step is not None else "N/A"
                ref_gpu_str = f"{result.reference_gpu_ms_step / 1000.0:.4f} s" if result.reference_gpu_ms_step is not None else "N/A"
                step1_gpu_str = f"{result.step1_gpu_ms / 1000.0:.4f} s" if result.step1_gpu_ms is not None else "N/A"
                step2_gpu_str = f"{result.step2_gpu_ms / 1000.0:.4f} s" if result.step2_gpu_ms is not None else "N/A"
                baseline_gpu_str = f"{result.baseline_gpu_ms_step / 1000.0:.4f} s" if result.baseline_gpu_ms_step is not None else "N/A"
                ref_full_str = f"{result.reference_full_ms_step / 1000.0:.4f} s" if result.reference_full_ms_step is not None else "N/A"
                step1_full_str = f"{result.step1_full_ms / 1000.0:.4f} s" if result.step1_full_ms is not None else "N/A"
                step2_full_str = f"{result.step2_full_ms / 1000.0:.4f} s" if result.step2_full_ms is not None else "N/A"
                baseline_full_str = f"{result.baseline_full_ms_step / 1000.0:.4f} s" if result.baseline_full_ms_step is not None else "N/A"
                f.write(f"{result.name}:\n")
                baseline_kernel_part = f", baseline={baseline_kernel_str}" if has_baseline else ""
                baseline_gpu_part = f", baseline={baseline_gpu_str}" if has_baseline else ""
                baseline_full_part = f", baseline={baseline_full_str}" if has_baseline else ""
                f.write(f"  Kernel: reference={ref_kernel_str}, step1={step1_kernel_str}, step2={step2_kernel_str}{baseline_kernel_part}\n")
                f.write(f"  GPU: reference={ref_gpu_str}, step1={step1_gpu_str}, step2={step2_gpu_str}{baseline_gpu_part}\n")
                f.write(f"  Full: reference={ref_full_str}, step1={step1_full_str}, step2={step2_full_str}{baseline_full_part}\n")
            else:
                ref_str = f"{result.reference_ms / 1000.0:.4f} s" if result.reference_ms is not None else "N/A"
                step1_str = f"{result.step1_ms / 1000.0:.4f} s" if result.step1_ms is not None else "N/A"
                step2_str = f"{result.step2_ms / 1000.0:.4f} s" if result.step2_ms is not None else "N/A"
                baseline_str = f", baseline={result.baseline_ms / 1000.0:.4f} s" if has_baseline and result.baseline_ms is not None else ", baseline=N/A" if has_baseline else ""
                f.write(f"{result.name}: reference={ref_str}, step1={step1_str}, step2={step2_str}{baseline_str}\n")
        elif measure_all:
            baseline_kernel_str = f"{result.baseline_kernel_ms} ms" if result.baseline_kernel_ms is not None else "N/A"
            baseline_gpu_str = f"{result.baseline_gpu_ms} ms" if result.baseline_gpu_ms is not None else "N/A"
            baseline_full_str = f"{result.baseline_full_ms} ms" if result.baseline_full_ms is not None else "N/A"
            f.write(f"{result.name}:\n")
            f.write(f"  Kernel: reference={result.reference_kernel_ms} ms, ParaCodex={result.candidate_kernel_ms} ms")
            if has_baseline:
                f.write(f", codex={baseline_kernel_str}")
            f.write("\n")
            f.write(f"  GPU: reference={result.reference_gpu_ms} ms, ParaCodex={result.candidate_gpu_ms} ms")
            if has_baseline:
                f.write(f", codex={baseline_gpu_str}")
            f.write("\n")
            f.write(f"  Full: reference={result.reference_full_ms} ms, ParaCodex={result.candidate_full_ms} ms")
            if has_baseline:
                f.write(f", codex={baseline_full_str}")
            f.write("\n")
        else:
            baseline_str = f", codex={result.baseline_ms} ms" if result.baseline_ms is not None else ", codex=N/A" if has_baseline else ""
            f.write(
                f"{result.name}: reference={result.reference_ms} ms, ParaCodex={result.candidate_ms} ms{baseline_str}\n"
            )


def plot_results(results: List[BenchmarkResult], out_dir: Path, gpu_time: bool = False, full_time: bool = False, measure_all: bool = False, has_baseline: bool = False, step_comparison: bool = False) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    names = [r.name for r in results]
    x = list(range(len(names)))

    if step_comparison:
        if measure_all:
            # Step comparison with measure_all: show all three measurement types
            # Convert ms to seconds and use linear scale
            ref_kernel = [(r.reference_kernel_ms_step / 1000.0) if r.reference_kernel_ms_step is not None else float("nan") for r in results]
            step1_kernel = [(r.step1_kernel_ms / 1000.0) if r.step1_kernel_ms is not None else float("nan") for r in results]
            step2_kernel = [(r.step2_kernel_ms / 1000.0) if r.step2_kernel_ms is not None else float("nan") for r in results]
            baseline_kernel = [(r.baseline_kernel_ms_step / 1000.0) if r.baseline_kernel_ms_step is not None else float("nan") for r in results]
            ref_gpu = [(r.reference_gpu_ms_step / 1000.0) if r.reference_gpu_ms_step is not None else float("nan") for r in results]
            step1_gpu = [(r.step1_gpu_ms / 1000.0) if r.step1_gpu_ms is not None else float("nan") for r in results]
            step2_gpu = [(r.step2_gpu_ms / 1000.0) if r.step2_gpu_ms is not None else float("nan") for r in results]
            baseline_gpu = [(r.baseline_gpu_ms_step / 1000.0) if r.baseline_gpu_ms_step is not None else float("nan") for r in results]
            ref_full = [(r.reference_full_ms_step / 1000.0) if r.reference_full_ms_step is not None else float("nan") for r in results]
            step1_full = [(r.step1_full_ms / 1000.0) if r.step1_full_ms is not None else float("nan") for r in results]
            step2_full = [(r.step2_full_ms / 1000.0) if r.step2_full_ms is not None else float("nan") for r in results]
            baseline_full = [(r.baseline_full_ms_step / 1000.0) if r.baseline_full_ms_step is not None else float("nan") for r in results]

            # Create figure with 3 subplots (one for each measurement type)
            title = "Performance Comparison: Reference vs Step 1 vs Step 2" + (" vs Baseline" if has_baseline else "")
            fig, axes = plt.subplots(1, 3, figsize=(18, 7))
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
            
            # Colors
            ref_color = '#2E86AB'  # Blue for reference
            step1_color = '#A23B72'  # Purple/Pink for step1
            step2_color = '#F18F01'  # Orange for step2
            baseline_color = '#06A77D'  # Green for baseline
            width = 0.2 if has_baseline else 0.25
            
            # Plot 1: Kernel-only
            ax1 = axes[0]
            if has_baseline:
                ax1.bar([i - 1.5*width for i in x], ref_kernel, width, label='Reference', color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.bar([i - 0.5*width for i in x], step1_kernel, width, label='Step 1', color=step1_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.bar([i + 0.5*width for i in x], step2_kernel, width, label='Step 2', color=step2_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.bar([i + 1.5*width for i in x], baseline_kernel, width, label='Baseline', color=baseline_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            else:
                ax1.bar([i - width for i in x], ref_kernel, width, label='Reference', color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.bar([i for i in x], step1_kernel, width, label='Step 1', color=step1_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax1.bar([i + width for i in x], step2_kernel, width, label='Step 2', color=step2_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.set_xticks(x)
            ax1.set_xticklabels(names, rotation=45, ha="right")
            ax1.set_ylabel("Time (seconds)", fontsize=10)
            ax1.set_title("Kernel-only\n(GPU Kernel Execution)", fontsize=11, fontweight='bold')
            ax1.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
            ax1.legend(fontsize=9)
            
            # Plot 2: GPU (Kernel + Memory)
            ax2 = axes[1]
            if has_baseline:
                ax2.bar([i - 1.5*width for i in x], ref_gpu, width, label='Reference', color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax2.bar([i - 0.5*width for i in x], step1_gpu, width, label='Step 1', color=step1_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax2.bar([i + 0.5*width for i in x], step2_gpu, width, label='Step 2', color=step2_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax2.bar([i + 1.5*width for i in x], baseline_gpu, width, label='Baseline', color=baseline_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            else:
                ax2.bar([i - width for i in x], ref_gpu, width, label='Reference', color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax2.bar([i for i in x], step1_gpu, width, label='Step 1', color=step1_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax2.bar([i + width for i in x], step2_gpu, width, label='Step 2', color=step2_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=45, ha="right")
            ax2.set_ylabel("Time (seconds)", fontsize=10)
            ax2.set_title("GPU Time\n(Kernel + Memory Transfers)", fontsize=11, fontweight='bold')
            ax2.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
            ax2.legend(fontsize=9)
            
            # Plot 3: Full (All components)
            ax3 = axes[2]
            if has_baseline:
                ax3.bar([i - 1.5*width for i in x], ref_full, width, label='Reference', color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax3.bar([i - 0.5*width for i in x], step1_full, width, label='Step 1', color=step1_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax3.bar([i + 0.5*width for i in x], step2_full, width, label='Step 2', color=step2_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax3.bar([i + 1.5*width for i in x], baseline_full, width, label='Baseline', color=baseline_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            else:
                ax3.bar([i - width for i in x], ref_full, width, label='Reference', color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax3.bar([i for i in x], step1_full, width, label='Step 1', color=step1_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax3.bar([i + width for i in x], step2_full, width, label='Step 2', color=step2_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax3.set_xticks(x)
            ax3.set_xticklabels(names, rotation=45, ha="right")
            ax3.set_ylabel("Time (seconds)", fontsize=10)
            ax3.set_title("Full Execution Time\n(Kernel + Memory + API + OS)", fontsize=11, fontweight='bold')
            ax3.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
            ax3.legend(fontsize=9)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        else:
            # Step comparison mode: step1 vs step2 vs reference vs baseline (single measurement)
            # Convert ms to seconds and use linear scale
            ref_vals = [(r.reference_ms / 1000.0) if r.reference_ms is not None else float("nan") for r in results]
            step1_vals = [(r.step1_ms / 1000.0) if r.step1_ms is not None else float("nan") for r in results]
            step2_vals = [(r.step2_ms / 1000.0) if r.step2_ms is not None else float("nan") for r in results]
            baseline_vals = [(r.baseline_ms / 1000.0) if r.baseline_ms is not None else float("nan") for r in results]

            width = 0.2 if has_baseline else 0.25
            ref_color = '#2E86AB'  # Blue for reference
            step1_color = '#A23B72'  # Purple/Pink for step1
            step2_color = '#F18F01'  # Orange for step2
            baseline_color = '#06A77D'  # Green for baseline

            fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
            if has_baseline:
                ax.bar([i - 1.5*width for i in x], ref_vals, width, label="Reference", color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.bar([i - 0.5*width for i in x], step1_vals, width, label="Step 1", color=step1_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.bar([i + 0.5*width for i in x], step2_vals, width, label="Step 2", color=step2_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.bar([i + 1.5*width for i in x], baseline_vals, width, label="Baseline", color=baseline_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            else:
                ax.bar([i - width for i in x], ref_vals, width, label="Reference", color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.bar([i for i in x], step1_vals, width, label="Step 1", color=step1_color, alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.bar([i + width for i in x], step2_vals, width, label="Step 2", color=step2_color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_ylabel("Time (seconds)", fontsize=10)
            title = "Performance Comparison: Reference vs Step 1 vs Step 2" + (" vs Baseline" if has_baseline else "")
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
            ax.legend(fontsize=9)
            fig.tight_layout()
    elif measure_all:
        # Use subplots for each measurement type for better clarity
        ref_kernel = [r.reference_kernel_ms if r.reference_kernel_ms is not None else float("nan") for r in results]
        cand_kernel = [r.candidate_kernel_ms if r.candidate_kernel_ms is not None else float("nan") for r in results]
        baseline_kernel = [r.baseline_kernel_ms if r.baseline_kernel_ms is not None else float("nan") for r in results]
        ref_gpu = [r.reference_gpu_ms if r.reference_gpu_ms is not None else float("nan") for r in results]
        cand_gpu = [r.candidate_gpu_ms if r.candidate_gpu_ms is not None else float("nan") for r in results]
        baseline_gpu = [r.baseline_gpu_ms if r.baseline_gpu_ms is not None else float("nan") for r in results]
        ref_full = [r.reference_full_ms if r.reference_full_ms is not None else float("nan") for r in results]
        cand_full = [r.candidate_full_ms if r.candidate_full_ms is not None else float("nan") for r in results]
        baseline_full = [r.baseline_full_ms if r.baseline_full_ms is not None else float("nan") for r in results]

        # Create figure with 3 subplots (one for each measurement type)
        title = "Performance Comparison: Reference vs ParaCodex" + (" vs codex" if has_baseline else "")
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        
        # Colors
        ref_color = '#2E86AB'  # Blue for reference
        cand_color = '#A23B72'  # Purple/Pink for ParaCodex
        baseline_color = '#F18F01'  # Orange for codex
        width = 0.25 if has_baseline else 0.35
        
        # Plot 1: Kernel-only
        ax1 = axes[0]
        ax1.bar([i - width if has_baseline else i - width/2 for i in x], ref_kernel, width, label='Reference', color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.bar([i for i in x], cand_kernel, width, label='ParaCodex', color=cand_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        if has_baseline:
            ax1.bar([i + width for i in x], baseline_kernel, width, label='codex', color=baseline_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("Time (ms) [log scale]", fontsize=10)
        ax1.set_yscale("log")
        ax1.set_title("Kernel-only\n(GPU Kernel Execution)", fontsize=11, fontweight='bold')
        ax1.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax1.legend(fontsize=9)
        
        # Plot 2: GPU (Kernel + Memory)
        ax2 = axes[1]
        ax2.bar([i - width if has_baseline else i - width/2 for i in x], ref_gpu, width, label='Reference', color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.bar([i for i in x], cand_gpu, width, label='ParaCodex', color=cand_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        if has_baseline:
            ax2.bar([i + width for i in x], baseline_gpu, width, label='codex', color=baseline_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylabel("Time (ms) [log scale]", fontsize=10)
        ax2.set_yscale("log")
        ax2.set_title("GPU Time\n(Kernel + Memory Transfers)", fontsize=11, fontweight='bold')
        ax2.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax2.legend(fontsize=9)
        
        # Plot 3: Full (All components)
        ax3 = axes[2]
        ax3.bar([i - width if has_baseline else i - width/2 for i in x], ref_full, width, label='Reference', color=ref_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.bar([i for i in x], cand_full, width, label='ParaCodex', color=cand_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        if has_baseline:
            ax3.bar([i + width for i in x], baseline_full, width, label='codex', color=baseline_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha="right")
        ax3.set_ylabel("Time (ms) [log scale]", fontsize=10)
        ax3.set_yscale("log")
        ax3.set_title("Full Execution Time\n(Kernel + Memory + API + OS)", fontsize=11, fontweight='bold')
        ax3.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
        ax3.legend(fontsize=9)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    else:
        # Single measurement mode
        ref_vals = [r.reference_ms if r.reference_ms is not None else float("nan") for r in results]
        cand_vals = [r.candidate_ms if r.candidate_ms is not None else float("nan") for r in results]
        baseline_vals = [r.baseline_ms if r.baseline_ms is not None else float("nan") for r in results]

        width = 0.25 if has_baseline else 0.35

        # Determine label based on measurement mode
        if full_time:
            ylabel = "Total Execution Time (ms) [log]"
            title = "Reference vs ParaCodex" + (" vs codex" if has_baseline else "") + " Total Execution Time (GPU + Memory + API + OS)"
        elif gpu_time:
            ylabel = "Total GPU Time (ms) [log]"
            title = "Reference vs ParaCodex" + (" vs codex" if has_baseline else "") + " Total GPU Time (Kernels + Memory)"
        else:
            ylabel = "Total GPU Kernel Time (ms) [log]"
            title = "Reference vs ParaCodex" + (" vs codex" if has_baseline else "") + " Total GPU Kernel Time"

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
        ax.bar([i - width if has_baseline else i - width/2 for i in x], ref_vals, width, label="Reference", color='#2E86AB', alpha=0.8)
        ax.bar([i for i in x], cand_vals, width, label="ParaCodex", color='#A23B72', alpha=0.8)
        if has_baseline:
            ax.bar([i + width for i in x], baseline_vals, width, label="codex", color='#F18F01', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend()

    if not measure_all:
        fig.tight_layout()
    # For measure_all, tight_layout is already called above with rect parameter

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile candidate vs reference NAS benchmarks using Nsight Systems (total GPU kernel time)"
    )
    parser.add_argument("--reference_dir", type=str, help="Directory containing subdirs with source .c files")
    parser.add_argument("--candidate_dir", type=str, help="Directory containing subdirs with optimized .c files")
    parser.add_argument("--baseline_dir", type=str, help="Directory containing subdirs with baseline .c files (optional)")
    parser.add_argument(
        "--output",
        type=str,
        default="/root/codex_baseline/serial_omp_nas_workdir/results_perf_nas_nsys",
        help="Output directory for results",
    )
    parser.add_argument(
        "--gpu_time",
        action="store_true",
        help="Include GPU memory transfer time (cuda_gpu_mem_time_sum) in addition to kernel time",
    )
    parser.add_argument(
        "--full_time",
        action="store_true",
        help="Include all time: GPU kernels + memory transfers + CUDA API calls + OS runtime",
    )
    parser.add_argument(
        "--measure_all",
        action="store_true",
        help="Measure all three ways (kernels, GPU time, full time) and plot all on the graph",
    )
    parser.add_argument(
        "--step-comparison",
        "--step_comparison",
        dest="step_comparison",
        action="store_true",
        help="Compare step1 vs step2 vs reference from pipeline directory structure",
    )
    parser.add_argument(
        "--pipeline_dir",
        type=str,
        help="Directory containing pipeline results with step1/step2 subdirectories (required for --step_comparison)",
    )
    args = parser.parse_args(argv)

    reference_dir = Path(args.reference_dir).resolve() if args.reference_dir else None
    candidate_dir = Path(args.candidate_dir).resolve() if args.candidate_dir else None
    baseline_dir = Path(args.baseline_dir).resolve() if args.baseline_dir else None
    pipeline_dir = Path(args.pipeline_dir).resolve() if args.pipeline_dir else None
    out_dir = Path(args.output).resolve()

    if args.step_comparison:
        if not args.pipeline_dir:
            print("--pipeline_dir is required when using --step_comparison", file=sys.stderr)
            return 2
        if not args.reference_dir:
            print("--reference_dir is required when using --step_comparison", file=sys.stderr)
            return 2

    # Preflight: ensure required tools exist in PATH
    for tool in ("nsys", "make", "nvc++"):
        if shutil.which(tool) is None:
            print(f"Required tool not found in PATH: {tool}", file=sys.stderr)
            return 2

    if args.step_comparison:
        # Step comparison mode: step1 vs step2 vs reference vs baseline (if provided)
        # Use specific functions for reference and baseline
        ref_map = find_reference_files(reference_dir)
        steps_map = find_steps_in_pipeline_dir(pipeline_dir)
        
        # Baseline is optional in step comparison mode
        if baseline_dir:
            baseline_map = find_baseline_files(baseline_dir)
        else:
            baseline_map = {}
        
        # Find common benchmarks (reference and steps; baseline is optional per benchmark)
        common = sorted(set(ref_map.keys()) & set(steps_map.keys()))
        if not common:
            print("No common benchmarks found between reference and pipeline steps", file=sys.stderr)
            return 1
        
        cand_map = {}  # Not used in step comparison mode
        has_baseline = bool(baseline_dir and baseline_map)
    else:
        # Normal mode: candidate vs reference
        ref_map = find_subdirs_with_source_files(reference_dir)
        cand_map = find_subdirs_with_source_files(candidate_dir)
        baseline_map = find_subdirs_with_source_files(baseline_dir) if baseline_dir else {}

        common = sorted(set(ref_map.keys()) & set(cand_map.keys()))
        if not common:
            print("No common benchmarks found between reference and candidate", file=sys.stderr)
            return 1
        
        has_baseline = bool(baseline_dir and baseline_map)

    # Create temp directory for intermediate storage
    temp_dir = out_dir / "temp_nsys_outputs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    results: List[BenchmarkResult] = []

    for bench in common:
        if args.step_comparison:
            # Step comparison mode
            if bench not in steps_map:
                print(f"[WARN] {bench}: No step files found, skipping")
                result = BenchmarkResult(bench, None, None, None, None, step1_ms=None, step2_ms=None)
                results.append(result)
                append_result_to_file(result, out_dir, step_comparison=True, measure_all=args.measure_all)
                continue
            
            steps = steps_map[bench]
            if "step1" not in steps or "step2" not in steps:
                print(f"[WARN] {bench}: Missing step1 or step2 file, skipping")
                result = BenchmarkResult(bench, None, None, None, None, step1_ms=None, step2_ms=None)
                results.append(result)
                append_result_to_file(result, out_dir, step_comparison=True, measure_all=args.measure_all)
                continue
            
            # Measure reference
            ref_src = ref_map[bench]
            target_filename = get_target_source_filename(bench)
            project_dir = copy_candidate_to_datasrc(bench, ref_src)
            
            if args.measure_all:
                # Measure all three types for reference
                ref_kernel, ref_gpu, ref_full = profile_all_measurements(project_dir, runs=2, temp_dir=temp_dir)
                ref_ms = ref_kernel  # Use kernel as default for backward compatibility
                
                # Measure all three types for step1
                step1_kernel, step1_gpu, step1_full = None, None, None
                try:
                    shutil.copy2(steps["step1"], project_dir / target_filename)
                    step1_kernel, step1_gpu, step1_full = profile_all_measurements(project_dir, runs=2, temp_dir=temp_dir)
                except Exception as e:
                    print(f"[WARN] {bench}: step1 measurement failed: {e}")
                
                # Measure all three types for step2
                step2_kernel, step2_gpu, step2_full = None, None, None
                try:
                    shutil.copy2(steps["step2"], project_dir / target_filename)
                    step2_kernel, step2_gpu, step2_full = profile_all_measurements(project_dir, runs=2, temp_dir=temp_dir)
                except Exception as e:
                    print(f"[WARN] {bench}: step2 measurement failed: {e}")
                
                # Measure baseline if available
                baseline_kernel_step, baseline_gpu_step, baseline_full_step = None, None, None
                if bench in baseline_map:
                    baseline_src = baseline_map[bench]
                    try:
                        shutil.copy2(baseline_src, project_dir / target_filename)
                        baseline_kernel_step, baseline_gpu_step, baseline_full_step = profile_all_measurements(project_dir, runs=2, temp_dir=temp_dir)
                        print(f"[INFO] {bench}: Measured baseline")
                    except Exception as e:
                        print(f"[WARN] {bench}: baseline measurement failed: {e}")
                else:
                    print(f"[INFO] {bench}: No baseline file found, skipping baseline measurement")
                
                result = BenchmarkResult(
                    name=bench,
                    reference_ms=ref_ms,
                    candidate_ms=None,  # Not used in step comparison mode
                    reference_log=None,
                    candidate_log=None,
                    step1_ms=step1_kernel,  # Use kernel as default
                    step2_ms=step2_kernel,  # Use kernel as default
                    reference_kernel_ms_step=ref_kernel,
                    reference_gpu_ms_step=ref_gpu,
                    reference_full_ms_step=ref_full,
                    step1_kernel_ms=step1_kernel,
                    step1_gpu_ms=step1_gpu,
                    step1_full_ms=step1_full,
                    step2_kernel_ms=step2_kernel,
                    step2_gpu_ms=step2_gpu,
                    step2_full_ms=step2_full,
                    baseline_kernel_ms_step=baseline_kernel_step,
                    baseline_gpu_ms_step=baseline_gpu_step,
                    baseline_full_ms_step=baseline_full_step,
                )
                baseline_info = f", baseline (kernel={baseline_kernel_step}, gpu={baseline_gpu_step}, full={baseline_full_step}) ms" if baseline_kernel_step is not None else ""
                print(f"[INFO] {bench}: reference (kernel={ref_kernel}, gpu={ref_gpu}, full={ref_full}) ms, step1 (kernel={step1_kernel}, gpu={step1_gpu}, full={step1_full}) ms, step2 (kernel={step2_kernel}, gpu={step2_gpu}, full={step2_full}) ms{baseline_info}")
            else:
                # Single measurement mode
                ref_ms = profile_mean_ms(project_dir, runs=2, gpu_time=args.gpu_time, full_time=args.full_time, temp_dir=temp_dir)
                
                # Measure step1
                try:
                    shutil.copy2(steps["step1"], project_dir / target_filename)
                    step1_ms = profile_mean_ms(project_dir, runs=2, gpu_time=args.gpu_time, full_time=args.full_time, temp_dir=temp_dir)
                except Exception as e:
                    print(f"[WARN] {bench}: step1 measurement failed: {e}")
                    step1_ms = None
                
                # Measure step2
                try:
                    shutil.copy2(steps["step2"], project_dir / target_filename)
                    step2_ms = profile_mean_ms(project_dir, runs=2, gpu_time=args.gpu_time, full_time=args.full_time, temp_dir=temp_dir)
                except Exception as e:
                    print(f"[WARN] {bench}: step2 measurement failed: {e}")
                    step2_ms = None
                
                # Measure baseline if available
                baseline_ms_step = None
                if bench in baseline_map:
                    baseline_src = baseline_map[bench]
                    try:
                        shutil.copy2(baseline_src, project_dir / target_filename)
                        baseline_ms_step = profile_mean_ms(project_dir, runs=2, gpu_time=args.gpu_time, full_time=args.full_time, temp_dir=temp_dir)
                        print(f"[INFO] {bench}: Measured baseline")
                    except Exception as e:
                        print(f"[WARN] {bench}: baseline measurement failed: {e}")
                else:
                    print(f"[INFO] {bench}: No baseline file found, skipping baseline measurement")
                
                result = BenchmarkResult(
                    name=bench,
                    reference_ms=ref_ms,
                    candidate_ms=None,  # Not used in step comparison mode
                    reference_log=None,
                    candidate_log=None,
                    step1_ms=step1_ms,
                    step2_ms=step2_ms,
                    baseline_ms=baseline_ms_step,  # Store baseline in step comparison mode
                )
                baseline_str = f", baseline={baseline_ms_step} ms" if baseline_ms_step is not None else ""
                print(f"[INFO] {bench}: reference={ref_ms} ms, step1={step1_ms} ms, step2={step2_ms} ms{baseline_str}")
            
            results.append(result)
            append_result_to_file(result, out_dir, step_comparison=True, measure_all=args.measure_all, has_baseline=has_baseline)
            continue
        
        # Normal mode (existing logic)
        print(f"=== Benchmark: {bench} ===")
        # Check if candidate file exists
        if bench not in cand_map:
            print(f"[WARN] {bench}: No candidate file found (optimized/ may be empty), skipping")
            # Create result with N/A values
            if args.measure_all:
                result = BenchmarkResult(
                    name=bench,
                    reference_ms=None,
                    candidate_ms=None,
                    reference_log=None,
                    candidate_log=None,
                    reference_kernel_ms=None,
                    reference_gpu_ms=None,
                    reference_full_ms=None,
                    candidate_kernel_ms=None,
                    candidate_gpu_ms=None,
                    candidate_full_ms=None,
                    baseline_kernel_ms=None,
                    baseline_gpu_ms=None,
                    baseline_full_ms=None,
                )
            else:
                result = BenchmarkResult(bench, None, None, None, None)
            results.append(result)
            append_result_to_file(result, out_dir, measure_all=args.measure_all, has_baseline=has_baseline)
            continue
        
        # Normal mode: Copy candidate into data/src/<bench>/<bench>.c
        try:
            project_dir = copy_candidate_to_datasrc(bench, cand_map[bench])
        except Exception as e:
            result = BenchmarkResult(bench, None, None, None, None) if not args.measure_all else BenchmarkResult(bench, None, None, None, None, None, None, None, None, None, None)
            results.append(result)
            append_result_to_file(result, out_dir, measure_all=args.measure_all, has_baseline=has_baseline)
            print(f"[WARN] {bench}: copy failed: {e}")
            continue

        if args.measure_all:
            # Measure all three types for candidate
            cand_kernel, cand_gpu, cand_full = profile_all_measurements(project_dir, runs=2, temp_dir=temp_dir)
            
            # Restore reference by copying its source file over (to measure reference in the exact same env)
            ref_src = ref_map[bench]
            target_filename = get_target_source_filename(bench)
            try:
                shutil.copy2(ref_src, project_dir / target_filename)
            except Exception as e:
                print(f"[WARN] {bench}: restoring reference failed: {e}")
            
            # Measure all three types for reference
            ref_kernel, ref_gpu, ref_full = profile_all_measurements(project_dir, runs=2, temp_dir=temp_dir)
            
            # Measure baseline if available
            baseline_kernel, baseline_gpu, baseline_full = None, None, None
            if bench in baseline_map:
                baseline_src = baseline_map[bench]
                try:
                    shutil.copy2(baseline_src, project_dir / target_filename)
                    baseline_kernel, baseline_gpu, baseline_full = profile_all_measurements(project_dir, runs=2, temp_dir=temp_dir)
                    print(f"[INFO] {bench}: Measured baseline")
                except Exception as e:
                    print(f"[WARN] {bench}: baseline measurement failed: {e}")
            else:
                print(f"[INFO] {bench}: No baseline file found, skipping baseline measurement")
            
            result = BenchmarkResult(
                name=bench,
                reference_ms=ref_kernel,  # Use kernel as default for backward compatibility
                candidate_ms=cand_kernel,
                reference_log=None,
                candidate_log=None,
                reference_kernel_ms=ref_kernel,
                reference_gpu_ms=ref_gpu,
                reference_full_ms=ref_full,
                candidate_kernel_ms=cand_kernel,
                candidate_gpu_ms=cand_gpu,
                candidate_full_ms=cand_full,
                baseline_kernel_ms=baseline_kernel,
                baseline_gpu_ms=baseline_gpu,
                baseline_full_ms=baseline_full,
            )
            results.append(result)
            # Write intermediate result to disk immediately
            append_result_to_file(result, out_dir, measure_all=args.measure_all, has_baseline=has_baseline)
            print(f"[INFO] {bench}: Saved intermediate results")
        else:
            # Run Nsight Systems for candidate (after copy/build), 3 runs mean
            cand_ms = profile_mean_ms(project_dir, runs=2, gpu_time=args.gpu_time, full_time=args.full_time, temp_dir=temp_dir)

            # Restore reference by copying its source file over (to measure reference in the exact same env)
            ref_src = ref_map[bench]
            target_filename = get_target_source_filename(bench)
            try:
                shutil.copy2(ref_src, project_dir / target_filename)
            except Exception as e:
                print(f"[WARN] {bench}: restoring reference failed: {e}")

            ref_ms = profile_mean_ms(project_dir, runs=2, gpu_time=args.gpu_time, full_time=args.full_time, temp_dir=temp_dir)

            # Measure baseline if available
            baseline_ms = None
            if bench in baseline_map:
                baseline_src = baseline_map[bench]
                try:
                    shutil.copy2(baseline_src, project_dir / target_filename)
                    baseline_ms = profile_mean_ms(project_dir, runs=2, gpu_time=args.gpu_time, full_time=args.full_time, temp_dir=temp_dir)
                    print(f"[INFO] {bench}: Measured baseline")
                except Exception as e:
                    print(f"[WARN] {bench}: baseline measurement failed: {e}")
            else:
                print(f"[INFO] {bench}: No baseline file found, skipping baseline measurement")

            result = BenchmarkResult(
                name=bench,
                reference_ms=ref_ms,
                candidate_ms=cand_ms,
                reference_log=None,
                candidate_log=None,
                baseline_ms=baseline_ms,
            )
            results.append(result)
            # Write intermediate result to disk immediately
            append_result_to_file(result, out_dir, measure_all=args.measure_all, has_baseline=has_baseline)
            print(f"[INFO] {bench}: Saved intermediate results")

    # Final results file (copy from intermediate)
    json_path, txt_path = write_results_json_txt(results, out_dir, measure_all=args.measure_all, has_baseline=has_baseline, step_comparison=args.step_comparison)
    
    # Clean up temp directory
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"[INFO] Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"[WARN] Failed to clean up temp directory: {e}")
    plot_path = plot_results(results, out_dir, gpu_time=args.gpu_time, full_time=args.full_time, measure_all=args.measure_all, has_baseline=has_baseline, step_comparison=args.step_comparison)

    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    if plot_path:
        print(f"Wrote {plot_path}")
    else:
        print("matplotlib not available, skipping plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

