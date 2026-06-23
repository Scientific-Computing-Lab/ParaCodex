
import os
import re
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Tuple

def clean_nsys_artifacts(work_dir: Path):
    """
    Remove Nsight Systems artifacts from the work directory.
    """
    for pattern in ["nsys_profile*", "*.qdstrm*", "*.nsys-rep*", "*.sqlite*"]:
        for file in work_dir.glob(pattern):
            try:
                file.unlink()
            except Exception:
                pass

def parse_total_gpu_time_ms(nsys_output: str, gpu_time: bool = True) -> Optional[float]:
    """
    Parse Nsight Systems stdout and compute total time (ms).
    Based on reference implementation in performance_comparison_cuda_ocl.py
    """
    if not nsys_output:
        return None

    lines = nsys_output.splitlines()
    kernel_ns = 0
    memory_ns = 0
    
    # Parse GPU kernel times (CUDA)
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

    # Parse GPU memory transfer times (CUDA)
    if gpu_time:
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

    total_ns = kernel_ns + (memory_ns if gpu_time else 0)

    if total_ns > 0:
        return total_ns / 1e6  # ns -> ms
    return None

def find_benchmark_dir(root_dir: Path) -> Optional[Path]:
    """
    Find the directory containing the Makefile.
    Searches for Makefile.nvc, Makefile.sycl, Makefile.hip or Makefile.
    If prefer_data_src is True, it will prioritize directories under data/src/.
    """
    # 1. If prefer_data_src, search data/src first
    if (root_dir / "data" / "src").exists():
        data_src = root_dir / "data" / "src"
        for makefile in ["Makefile.nvc", "Makefile.sycl", "Makefile.hip", "Makefile"]:
            for path in data_src.rglob(makefile):
                if "vendor" not in str(path):
                    return path.parent

    # 2. Check root
    for makefile in ["Makefile.nvc", "Makefile.sycl", "Makefile.hip", "Makefile"]:
        if (root_dir / makefile).exists():
            return root_dir
        
    # 3. Search recursively (up to depth 4 to avoid scanning too much)
    # matching the pattern data/src/BENCHMARK
    for makefile in ["Makefile.nvc", "Makefile.sycl", "Makefile.hip", "Makefile"]:
        for path in root_dir.rglob(makefile):
            # Avoid finding makefiles in vendor or other aux dirs if possible
            if "vendor" not in str(path) and "golden_labels" not in str(path).lower():
                return path.parent
            
    return None

def detect_run_command(work_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[Path]]:
    """
    Detect the build/run command, clean command, and the actual execution directory.
    Returns (run_cmd, clean_cmd, exec_dir)
    """
    exec_dir = find_benchmark_dir(work_dir)
    if not exec_dir:
        return None, None, None
        
    # Priority: nvc -> Makefile
    if (exec_dir / "Makefile.nvc").exists():
        return "make -f Makefile.nvc run", "make -f Makefile.nvc clean", exec_dir
    elif (exec_dir / "Makefile").exists():
        return "make run", "make clean", exec_dir
    
    # Check for legacy names just in case
    for m in ["Makefile.sycl", "Makefile.hip"]:
        if (exec_dir / m).exists():
            return f"make -f {m} run", f"make -f {m} clean", exec_dir
            
    return None, None, None

def find_golden_reference_dir(work_dir: Path) -> Optional[Path]:
    """
    Find the golden reference directory for the original code.
    Pattern: workdir/golden_labels/src/<benchmark-name>
    """
    golden_base = work_dir / "golden_labels" / "src"
    if not golden_base.exists():
        return None
    
    # Find subdirectory with Makefile
    for subdir in golden_base.iterdir():
        if subdir.is_dir():
            for makefile in ["Makefile.nvc", "Makefile.sycl", "Makefile.hip", "Makefile"]:
                if (subdir / makefile).exists():
                    return subdir
    
    return None

def find_optimized_code_dir(output_dir: Path) -> Optional[Path]:
    """
    Find the optimized translated code directory.
    Pattern: output_dir/<benchmark-name>/optimized/
    """
    # Look for optimized directory, with fallbacks for early stages
    for stage_name in ["optimized", "step2_supervised", "step2", "step1", "initial"]:
        for stage_dir in output_dir.rglob(stage_name):
            if stage_dir.is_dir():
                # Check for common HPC source file extensions
                extensions = ['*.c', '*.cl', '*.cpp', '*.cu', '*.h', '*.hpp']
                if any(list(stage_dir.glob(ext)) for ext in extensions):
                    print(f"Found translated code in: {stage_dir}")
                    return stage_dir
    
    return None
    
    return None

def copy_optimized_to_workdir(optimized_dir: Path, work_dir: Path) -> Optional[Path]:
    """
    Copy optimized files from output_dir to workdir build directory.
    Returns the target directory where files were copied.
    """
    import shutil
    
    # Find the target directory in workdir (data/src/<benchmark-name>)
    # Specifically look in data/src to avoid overwriting golden_labels
    target = None
    if (work_dir / "data" / "src").exists():
        data_src = work_dir / "data" / "src"
        for makefile in ["Makefile.nvc", "Makefile.sycl", "Makefile.hip", "Makefile"]:
            for path in data_src.rglob(makefile):
                if "vendor" not in str(path):
                    target = path.parent
                    break
            if target: break
            
    if not target:
        # Fallback to general search but exclude golden_labels
        target = find_benchmark_dir(work_dir)
        if target and "golden_labels" in str(target).lower():
            target = None
            
    if not target:
        return None
    
    print(f"Copying optimized files from {optimized_dir} to {target}")
    
    # Backup original files (optional, but safer)
    backup_dir = target.parent / f"{target.name}_backup_{int(time.time())}"
    
    # Copy all source files from optimized
    for src_file in optimized_dir.glob("*"):
        if src_file.is_file() and src_file.suffix in ['.c', '.cl', '.h', '.cpp', '.cu', '.hpp']:
            dest_file = target / src_file.name
            print(f"  Copying {src_file.name}")
            shutil.copy2(src_file, dest_file)
    
    return target

# Self-reported timing patterns emitted by HeCBench/pipeline kernels
_SELF_REPORTED_PATTERNS = [
    # e.g. "Average kernel execution time 0.01539 (s)"
    (re.compile(r'Average kernel execution time\s+([0-9.eE+-]+)\s*\(s\)', re.IGNORECASE), 's'),
    # e.g. "Kernel time: 15.39 ms"
    (re.compile(r'Kernel time[:\s]+([0-9.eE+-]+)\s*ms', re.IGNORECASE), 'ms'),
    # e.g. "GPU time: 15.39 ms"
    (re.compile(r'GPU time[:\s]+([0-9.eE+-]+)\s*ms', re.IGNORECASE), 'ms'),
    # e.g. "Total time: 15.39 ms"
    (re.compile(r'Total time[:\s]+([0-9.eE+-]+)\s*ms', re.IGNORECASE), 'ms'),
]

def parse_self_reported_time_ms(stdout: str) -> Optional[float]:
    """Parse application self-reported kernel/GPU time from stdout."""
    for pattern, unit in _SELF_REPORTED_PATTERNS:
        m = pattern.search(stdout)
        if m:
            try:
                val = float(m.group(1))
                return val * 1000.0 if unit == 's' else val
            except ValueError:
                continue
    return None


def detect_api_from_dir(exec_dir: Path) -> str:
    """Guess the parallel API from source files present in the directory."""
    if list(exec_dir.glob('*.cl')):
        return 'opencl'
    if list(exec_dir.glob('*.cu')) or list(exec_dir.glob('*.cuh')):
        return 'cuda'
    if list(exec_dir.glob('*.hip')):
        return 'hip'
    return 'unknown'


def _find_executable(exec_dir: Path) -> Optional[Path]:
    """Find the built executable in exec_dir."""
    for makefile in ['Makefile.nvc', 'Makefile.sycl', 'Makefile.hip', 'Makefile']:
        mf = exec_dir / makefile
        if not mf.exists():
            continue
        m = re.search(r'^program\s*[=:?]+\s*(\S+)', mf.read_text(errors='replace'), re.MULTILINE)
        if m:
            candidate = exec_dir / m.group(1).strip()
            if candidate.exists():
                return candidate
    # Fallback: find any executable file
    for f in exec_dir.iterdir():
        if f.is_file() and os.access(f, os.X_OK) and f.suffix == '':
            return f
    return None


def run_nsys_profile(work_dir: Path, is_original: bool = True, output_dir: Path = None,
                     run_args: Optional[list] = None) -> Dict:
    """
    Build and run the benchmark, measuring GPU time.

    Strategy (API-agnostic):
      1. Build with make clean + make.
      2. Run the executable directly with `run_args` if provided, else via `make run`.
         Using explicit args ensures fair apples-to-apples comparison between APIs.
      3. Parse self-reported "Average kernel execution time" from stdout.
      4. For CUDA code with no self-reported timing, fall back to nsys.

    `run_args`: if provided (e.g. from parbench spec performance config), the executable
    is invoked as `./binary <run_args>` instead of `make run`, ensuring the same workload
    for both the original and translated versions.
    """
    try:
        if is_original:
            exec_dir = find_golden_reference_dir(work_dir)
            if not exec_dir:
                return {"error": "Golden reference directory not found in workdir/golden_labels/src/"}
            print(f"Benchmarking original (golden reference) in {exec_dir}")
        else:
            if not output_dir:
                return {"error": "output_dir required for translated code profiling"}
            optimized_dir = find_optimized_code_dir(output_dir)
            if not optimized_dir:
                return {"error": "No optimized directory found in output_dir"}
            exec_dir = copy_optimized_to_workdir(optimized_dir, work_dir)
            if not exec_dir:
                return {"error": "Failed to copy optimized files to workdir"}
            print(f"Benchmarking translated (optimized) in {exec_dir}")

        run_cmd, clean_cmd, _ = detect_run_command(exec_dir)
        if not run_cmd:
            return {"error": f"No Makefile found in {exec_dir}"}

        # --- Clean + Build ---
        subprocess.run(clean_cmd.split(), cwd=str(exec_dir), capture_output=True, text=True)
        build_cmd = run_cmd.replace(' run', '').strip()
        build_proc = subprocess.run(build_cmd.split(), cwd=str(exec_dir),
                                    capture_output=True, text=True, timeout=120)
        if build_proc.returncode != 0:
            return {
                "error": f"Build failed (exit {build_proc.returncode})",
                "stderr": build_proc.stderr[:1000],
            }

        # --- Determine run command ---
        if run_args:
            # Run the executable directly with specified args (fair cross-API comparison)
            exe = _find_executable(exec_dir)
            if exe:
                run_only_cmd = [str(exe)] + [str(a) for a in run_args]
                print(f"Running with parbench args: {' '.join(run_only_cmd)}")
            else:
                # Fall back to make run
                run_only_cmd = run_cmd.split()
                print(f"Executable not found; falling back to: {' '.join(run_only_cmd)}")
        else:
            run_only_cmd = run_cmd.split()
            print(f"Running: {' '.join(run_only_cmd)}")

        proc = subprocess.run(
            run_only_cmd,
            cwd=str(exec_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )

        stdout = proc.stdout
        raw_output = stdout[:3000] if len(stdout) > 3000 else stdout

        if proc.returncode != 0:
            return {
                "error": f"Run failed (exit {proc.returncode})",
                "stderr": proc.stderr[:500],
                "stdout": raw_output,
            }

        # --- Parse self-reported timing (works for CUDA, OCL, HIP, SYCL) ---
        gpu_time = parse_self_reported_time_ms(stdout)
        method = "self_reported"

        if gpu_time is not None:
            clean_nsys_artifacts(exec_dir)
            return {
                "success": True,
                "gpu_time_ms": gpu_time,
                "method": method,
                "raw_output": raw_output,
                "exec_dir": str(exec_dir),
            }

        # --- Fallback: nsys (CUDA only) ---
        api = detect_api_from_dir(exec_dir)
        if api == 'cuda':
            subprocess.run(clean_cmd.split(), cwd=str(exec_dir), capture_output=True, text=True)
            nsys_cmd = [
                "nsys", "profile",
                "--stats=true", "--trace=cuda,osrt",
                "--force-overwrite=true", "-o", "nsys_profile",
                *run_only_cmd,
            ]
            print(f"Self-reported time not found; falling back to nsys: {' '.join(nsys_cmd)}")
            nsys_proc = subprocess.run(
                nsys_cmd, cwd=str(exec_dir), capture_output=True, text=True, timeout=180
            )
            if nsys_proc.returncode == 0:
                gpu_time = parse_total_gpu_time_ms(nsys_proc.stdout, gpu_time=True)
                if gpu_time is not None:
                    clean_nsys_artifacts(exec_dir)
                    return {
                        "success": True,
                        "gpu_time_ms": gpu_time,
                        "method": "nsys_cuda",
                        "raw_output": nsys_proc.stdout[:3000],
                        "exec_dir": str(exec_dir),
                    }

        # --- Nothing worked ---
        clean_nsys_artifacts(exec_dir)
        return {
            "success": True,
            "gpu_time_ms": None,
            "method": "none",
            "raw_output": raw_output,
            "exec_dir": str(exec_dir),
            "warning": "Could not extract GPU time from output. Check raw_output.",
        }

    except subprocess.TimeoutExpired:
        return {"error": "Profiling timed out"}
    except Exception as e:
        return {"error": str(e)}

def parse_existing_nsys_results(work_dir: Path) -> Dict:
    """
    Parse existing nsys results from step directories when no Makefile is found.
    This is common in output directories where the pipeline already ran nsys.
    """
    try:
        # Look for nsys_relevant.txt or nsys_output.txt files
        nsys_files = []
        for pattern in ["nsys_relevant.txt", "nsys_output.txt"]:
            nsys_files.extend(work_dir.rglob(pattern))
        
        if not nsys_files:
            return {"error": "No Makefile and no existing nsys results found"}
        
        # Use the first (usually most recent) nsys file found
        nsys_file = nsys_files[0]
        print(f"Found existing nsys results: {nsys_file}")
        
        content = nsys_file.read_text()
        
        # Look for "Total GPU kernel time (nsys): XX.XXX ms" pattern
        m = re.search(r"Total GPU kernel time \(nsys\):\s*([0-9.]+)\s*ms", content)
        if m:
            gpu_time = float(m.group(1))
            return {
                "success": True,
                "gpu_time_ms": gpu_time,
                "method": "existing_nsys_results",
                "raw_output": content[:2000] if len(content) > 2000 else content,
                "exec_dir": str(nsys_file.parent)
            }
        
        # Fallback: try parsing nsys stats format
        gpu_time = parse_total_gpu_time_ms(content, gpu_time=True)
        if gpu_time:
            return {
                "success": True,
                "gpu_time_ms": gpu_time,
                "method": "existing_nsys_parsed",
                "raw_output": content[:2000] if len(content) > 2000 else content,
                "exec_dir": str(nsys_file.parent)
            }
        
        return {
            "error": "Found nsys file but couldn't parse GPU time",
            "raw_output": content[:1000]
        }
        
    except Exception as e:
        return {"error": f"Error parsing existing results: {str(e)}"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run performance analysis on pipeline workdir")
    parser.add_argument("--workdir", required=True, help="Path to pipeline workdir")
    parser.add_argument("--output_dir", required=True, help="Path to pipeline output_dir")
    args = parser.parse_args()
    
    work_dir = Path(args.workdir)
    output_dir = Path(args.output_dir)
    
    print(f"=== Performance Analysis for {work_dir.name} ===")
    
    print("\n1. Profiling Original Code...")
    orig_res = run_nsys_profile(work_dir, is_original=True)
    if orig_res.get("success"):
        print(f"  ✓ Original GPU Time: {orig_res['gpu_time_ms']:.3f} ms ({orig_res['method']})")
    else:
        print(f"  ✗ Original Failed: {orig_res.get('error')}")
        if "stderr" in orig_res:
            print(f"    Stderr: {orig_res['stderr'][:200]}...")
            
    print("\n2. Profiling Translated Code...")
    trans_res = run_nsys_profile(work_dir, is_original=False, output_dir=output_dir)
    if trans_res.get("success"):
        print(f"  ✓ Translated GPU Time: {trans_res['gpu_time_ms']:.3f} ms ({trans_res['method']})")
    else:
        print(f"  ✗ Translated Failed: {trans_res.get('error')}")
        if "stderr" in trans_res:
             print(f"    Stderr: {trans_res['stderr'][:200]}...")
             
    if orig_res.get("success") and trans_res.get("success"):
        speedup = orig_res['gpu_time_ms'] / trans_res['gpu_time_ms']
        print(f"\nSpeedup: {speedup:.2f}x")

