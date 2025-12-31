#!/usr/bin/env python3
"""
Script to optimize translated code using Codex CLI after successful compilation.
"""

import subprocess
import tempfile
import shutil
import os
import time
from pathlib import Path
import re
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import sys

from path_config import (
    build_codex_cli_cmd,
    get_codex_workdir,
    get_make_cmd,
    get_make_cmd_str,
    get_nsys_run_make_cmd,
    get_nsys_run_make_cmd_str,
    get_correctness_run_cmd_str,
    get_correctness_fallback_cmd_str,
    get_profile_run_cmd_str,
    get_profile_fallback_cmd_str,
    get_nsys_profile_cmd_str,
    get_nsys_profile_fallback_cmd_str,
    run_with_gpu_timeout,
    kill_gpu_processes,
    get_gpu_processes,
    GPU_TIMEOUT_SECONDS,
)

NSYS_ARTIFACT_PATTERNS = ["nsys_profile*", "*.qdstrm*", "*.nsys-rep*", "*.sqlite*"]
PERFORMANCE_GATE_FACTOR = 1.1


def _build_nsys_profile_cmd(target_api: str, kernel_name: Optional[str] = None) -> List[str]:
    return [
        "nsys",
        "profile",
        "--stats=true",
        "--trace=cuda,osrt",
        "--force-overwrite=true",
        "-o",
        "nsys_profile",
        *get_nsys_run_make_cmd(target_api, kernel_name),
    ]


def _nsys_profile_cmd_str(target_api: str, kernel_name: Optional[str] = None) -> str:
    run_str = get_nsys_run_make_cmd_str(target_api, kernel_name)
    return (
        "FORCE_OMP_GPU=1 OMP_TARGET_OFFLOAD=MANDATORY nsys profile --stats=true --trace=cuda,osrt "
        f"--force-overwrite=true -o nsys_profile {run_str}"
    )


def resolve_kernel_file_name(file_name: str, target_api: str) -> str:
    """Normalize the kernel filename for the desired target API extension."""
    path = Path(file_name)
    suffix = path.suffix.lower()

    if target_api == 'cuda':
        if suffix in {'.cpp', '.c'}:
            return str(path.with_suffix('.cu'))
        return str(path)

    if target_api == 'omp':
        if suffix == '.cu':
            return str(path.with_suffix('.cpp'))
        return str(path)

    return str(path)


def _cleanup_nsys_artifacts(work_dir: Path) -> None:
    """Remove Nsight Systems output artifacts to avoid accumulation between runs."""
    for pattern in NSYS_ARTIFACT_PATTERNS:
        for path in work_dir.glob(pattern):
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass


def copy_translated_file(kernel_name, file_name, target_api, hecbench_src_dir, output_dir, suffix):
    """
    Copy the translated file(s) from kernel directory to results directory.
    Files are organized in subdirectories by phase/step (e.g., initial/, step1/, step2_supervised/).
    
    Args:
        kernel_name: Name of the kernel
        file_name: Original file name (str) or list of file names (List[str])
        target_api: Target API (omp, cuda, etc.)
        hecbench_src_dir: Directory containing kernel source code
        output_dir: Output directory for results
        suffix: Suffix/phase name ('initial', 'step1', 'step2_supervised', 'optimized', etc.)
    
    Returns:
        str or List[str]: Path(s) to copied file(s) or None if failed. Returns list if multiple files, str if single.
    """
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_name, str):
        file_names = [file_name]
        return_single = True
    else:
        file_names = file_name
        return_single = False
    
    # Determine source file path in kernel directory
    kernel_dir = Path(hecbench_src_dir) / f"{kernel_name}-{target_api}"
    
    # Use kernel name and target API for directory structure
    # Create destination path - standardized folder structure
    output_dir = Path(output_dir)
    kernel_output_dir = output_dir / f"{kernel_name}-{target_api}"
    
    # Create subdirectory based on suffix/phase (e.g., initial/, step1/, step2_supervised/)
    phase_dir = kernel_output_dir / suffix
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    copied_paths = []
    
    # Copy all files
    for fn in file_names:
        # Determine the actual file name in kernel directory
        source_file_name = resolve_kernel_file_name(fn, target_api)
        source_file_path = kernel_dir / source_file_name
        
        # If resolved file doesn't exist, try to find it with different extensions
        if not source_file_path.exists():
            base_name = Path(source_file_name).stem
            # Try common extensions for the target API
            if target_api == 'omp':
                # Try .c, .cpp in that order
                for ext in ['.c', '.cpp']:
                    candidate = kernel_dir / f"{base_name}{ext}"
                    if candidate.exists():
                        source_file_name = candidate.name
                        source_file_path = candidate
                        break
            elif target_api == 'cuda':
                # Try .cu
                candidate = kernel_dir / f"{base_name}.cu"
                if candidate.exists():
                    source_file_name = candidate.name
                    source_file_path = candidate
        
        # Use original filename in the subdirectory (no suffix in filename, since it's in the dir name)
        dest_file_path = phase_dir / source_file_name
        
        try:
            if source_file_path.exists():
                shutil.copy2(source_file_path, dest_file_path)
                print(f"✓ Copied {suffix} file: {source_file_path} -> {dest_file_path}")
                copied_paths.append(str(dest_file_path))
            else:
                print(f"⚠ Source file not found: {source_file_path}")
        except Exception as e:
            print(f"✗ Error copying {suffix} file {fn}: {e}")
    
    if not copied_paths:
        return None
    
    # Return single path for backward compatibility, or list if multiple files
    if return_single:
        return copied_paths[0]
    return copied_paths if len(copied_paths) > 1 else copied_paths[0]


def _record_gate_failure(
    kernel_output_dir: Path,
    step: int,
    runtime_ms: Optional[float],
    threshold_ms: Optional[float],
    transcript_summary: Optional[str],
    combined_transcript: Optional[str],
    run_output: Optional[str],
) -> None:
    """Log a performance gate rejection with contextual data."""
    report_path = kernel_output_dir / "performance_gate_reports.md"
    timestamp = datetime.utcnow().isoformat()
    lines = [
        f"=== {timestamp} - Stage {step} gate rejection ===",
        f"Runtime: {runtime_ms if runtime_ms is not None else 'unknown'} ms",
        f"Threshold: {threshold_ms if threshold_ms is not None else 'unknown'} ms",
        f"Model summary:\n{transcript_summary or 'N/A'}",
        f"Model transcript:\n{combined_transcript or 'N/A'}",
        "Nsight output (truncated):",
        (run_output or "N/A")[:2000],
        "",
    ]
    try:
        with open(report_path, 'a') as f:
            f.write("\n".join(lines) + "\n")
        print(f"⚠ Performance gate log written to {report_path}")
    except Exception as e:
        print(f"⚠ Failed to write performance gate report: {e}")


def _restore_stage_snapshot(
    kernel_dir: Path,
    output_dir: Path,
    kernel_name: str,
    primary_file_name: str,
    target_api: str,
    file_names: List[str],
    stage_suffix: str,
) -> None:
    """Restore source files from a previous snapshot when a stage is rejected."""
    snapshot_dir = output_dir / f"{kernel_name}-{target_api}" / stage_suffix
    if not snapshot_dir.exists():
        print(f"⚠ Snapshot {snapshot_dir} not found for restoration.")
        return
    for fn in file_names:
        resolved = resolve_kernel_file_name(fn, target_api)
        src_path = snapshot_dir / resolved
        dest_path = kernel_dir / resolved
        if src_path.exists():
            try:
                shutil.copy2(src_path, dest_path)
                print(f"→ Restored {dest_path} from snapshot {snapshot_dir}")
            except Exception as e:
                print(f"⚠ Failed to restore {dest_path}: {e}")
        else:
            print(f"⚠ Snapshot file missing: {src_path}")


def test_optimized_compilation(kernel_dir, target_api, kernel_name: Optional[str] = None, gpu_timeout: int = None):
    """Test compilation of optimized code with GPU timeout protection.
    
    Args:
        kernel_dir: Directory containing the kernel code
        target_api: Target API (omp, cuda, etc.)
        kernel_name: Optional kernel name for profiling
        gpu_timeout: Timeout in seconds for GPU execution (default: GPU_TIMEOUT_SECONDS env var or 300)
    """
    
    kernel_dir = Path(kernel_dir)
    clean_cmd = get_make_cmd(target_api, 'clean')
    timeout = gpu_timeout or GPU_TIMEOUT_SECONDS

    # Record GPU processes before we start
    pre_gpu_pids = get_gpu_processes()

    try:
        # Set environment for OpenMP builds to ensure nvc++ is used
        env = os.environ.copy()
        if target_api == 'omp':
            env['CC'] = 'nvc++'
        
        subprocess.run(clean_cmd, capture_output=True, text=True, timeout=30, cwd=kernel_dir, env=env)

        if target_api == 'omp':
            compile_result = subprocess.run(
                get_make_cmd(target_api, 'build'),
                capture_output=True,
                text=True,
                timeout=300,
                cwd=kernel_dir,
                env=env,
            )
            if compile_result.returncode != 0:
                print("Optimized code compilation failed.")
                return False, compile_result.stderr

        # Set environment variables for OpenMP GPU offloading
        # Preserve CC=nvc++ if it was set above
        if target_api == 'omp' and 'CC' not in env:
            env['CC'] = 'nvc++'
        env["FORCE_OMP_GPU"] = "1"
        env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
        
        # Use safe GPU execution with timeout (CLASS=C)
        nsys_cmd = _build_nsys_profile_cmd(target_api, kernel_name)
        print(f"Running GPU command with {timeout}s timeout: {' '.join(nsys_cmd)}")
        
        success, stdout, stderr = run_with_gpu_timeout(
            nsys_cmd,
            cwd=kernel_dir,
            timeout_seconds=timeout,
            kill_gpu_on_timeout=True,
            env=env,
        )
        _cleanup_nsys_artifacts(kernel_dir)

        # Check if timeout occurred - if so, retry with CLASS=B
        if not success and stderr and "[TIMEOUT" in stderr:
            print(f"CLASS=C timed out, retrying with CLASS=B...")
            # Run kill GPU script after CLASS C failure
            _run_kill_gpu_script()
            # Get fallback command (CLASS=B) - this is a shell command string
            fallback_cmd_str = get_nsys_profile_fallback_cmd_str(target_api, kernel_name, "B")
            print(f"Running fallback GPU command with {timeout}s timeout: {fallback_cmd_str}")
            
            # Run fallback command with shell=True (it includes env VAR=value prefix)
            # Use the same timeout and GPU cleanup logic
            try:
                result = subprocess.run(
                    fallback_cmd_str,
                    shell=True,
                    cwd=kernel_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                success = result.returncode == 0
                stdout = result.stdout
                stderr = result.stderr
            except subprocess.TimeoutExpired:
                print(f"Fallback CLASS=B also timed out after {timeout}s")
                _cleanup_gpu_after_failure(pre_gpu_pids)
                # Run kill GPU script after CLASS B failure
                _run_kill_gpu_script()
                success = False
                stdout = ""
                stderr = f"Both CLASS=C and CLASS=B timed out after {timeout}s"
            except Exception as e:
                print(f"Error running fallback command: {e}")
                success = False
                stdout = ""
                stderr = str(e)
            
            _cleanup_nsys_artifacts(kernel_dir)
            if success:
                print("CLASS=B run succeeded!")

        if stdout and stdout.strip():
            output = stdout.strip()
        elif stderr and stderr.strip():
            output = stderr.strip()
        else:
            output = ""

        if not success:
            print("Optimized code run failed.")
            if stderr:
                print(stderr[:500])  # Print first 500 chars of error
            # Clean up any hung GPU processes from our run
            _cleanup_gpu_after_failure(pre_gpu_pids)

        return success, output

    except subprocess.TimeoutExpired:
        print(f"Optimized code execution timeout after {timeout}s")
        _cleanup_gpu_after_failure(pre_gpu_pids)
        return False, f"Optimized code execution timeout after {timeout}s"
    except Exception as e:
        print(f"Error during optimized code execution: {e}")
        _cleanup_gpu_after_failure(pre_gpu_pids)
        return False, str(e)
    finally:
        # Always run kill GPU script at the end to ensure cleanup
        _run_kill_gpu_script()


def _cleanup_gpu_after_failure(pre_gpu_pids: List[int]):
    """Clean up GPU processes that were started after pre_gpu_pids snapshot."""
    import time
    time.sleep(1)
    current_pids = get_gpu_processes()
    new_pids = [p for p in current_pids if p not in pre_gpu_pids]
    if new_pids:
        print(f"Cleaning up hung GPU processes: {new_pids}")
        for pid in new_pids:
            try:
                os.kill(pid, 9)  # SIGKILL
                print(f"  Killed PID {pid}")
            except Exception as e:
                print(f"  Failed to kill PID {pid}: {e}")


def _run_kill_gpu_script():
    """Run the kill GPU processes script to clean up all WSL GPU processes."""
    script_path = Path(__file__).parent.parent / "kill_gpu_processes.py"
    if script_path.exists():
        try:
            print("Running kill_gpu_processes.py to clean up all WSL GPU processes...")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode != 0:
                print(f"Warning: kill_gpu_processes.py had errors: {result.stderr}")
        except Exception as e:
            print(f"Warning: Failed to run kill_gpu_processes.py: {e}")
    else:
        print(f"Warning: kill_gpu_processes.py not found at {script_path}")


## UNUSED: legacy artifact saver (commented out)
# def save_optimized_code(optimized_code, kernel_name, file_name, output_dir, target_api, compilation_result=None):
#     pass


def _run_compare_and_optimize_steps(kernel_name: str, file_name: str, output_dir: str, target_api: str) -> bool:
    """Run the compare_and_optimize_steps script after supervisor completes.
    
    Args:
        kernel_name: Name of the kernel
        file_name: Name of the file being processed
        output_dir: Output directory where step supervised files are saved
        target_api: Target API (omp, cuda, etc.)
        
    Returns:
        True if script ran successfully, False otherwise
    """
    try:
        script_path = Path(__file__).parent.parent / "scripts" / "compare_and_optimize_steps.py"
        if not script_path.exists():
            print(f"[WARN] compare_and_optimize_steps.py not found at {script_path}")
            return False
        
        # The benchmark name in output_dir is "{kernel_name}-{target_api}"
        benchmark_name = f"{kernel_name}-{target_api}"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--base_dir", str(output_dir),
            "--benchmark", benchmark_name,
        ]
        
        print(f"Running compare_and_optimize_steps for {benchmark_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"✓ compare_and_optimize_steps completed successfully for {benchmark_name}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"⚠ compare_and_optimize_steps returned non-zero exit code for {benchmark_name}")
            if result.stderr:
                print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⚠ compare_and_optimize_steps timed out for {benchmark_name}")
        return False
    except Exception as e:
        print(f"⚠ Error running compare_and_optimize_steps for {benchmark_name}: {e}")
        return False


def _run_supervisor(
    kernel_name: str, target_api: str, hecbench_src: str, output_dir: str, file_name: Optional[str] | List[str] = None
) -> Tuple[bool, str]:
    """Invoke the supervisor agent for a specific kernel. Returns (success, output).

    Kept local to avoid circular imports with initial_translation_codex.
    
    Args:
        file_name: Can be a single file name (str), list of file names, or None
    """
    try:
        supervisor_path = Path(__file__).parent / 'supervisor_codex.py'
        cmd = [
            'python3', str(supervisor_path),
            '--hecbench-src', str(hecbench_src),
            '--target-api', target_api,
            '--kernels', kernel_name,
            '--results-dir', output_dir,
            '--codex-workdir', str(get_codex_workdir()),
        ]
        if file_name:
            # Handle both single file (str) and list of files
            if isinstance(file_name, list):
                # For multiple files, pass the first one as hint (supervisor will use the list internally)
                if file_name:
                    cmd.extend(['--file-name', file_name[0]])
            else:
                cmd.extend(['--file-name', file_name])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        success = result.returncode == 0
        output = (result.stdout or '') + '\n' + (result.stderr or '')
        return success, output
    except subprocess.TimeoutExpired:
        return False, 'Supervisor timed out'
    except Exception as e:
        return False, f'Supervisor error: {e}'


## UNUSED helper (commented out)
# def _read_kernel_file_path(kernel_dir: Path, file_name: str, target_api: str) -> Path:
#     pass


def _parse_nsys_runtime_ms(nsys_output: str) -> Optional[float]:
    """Parse total GPU execution time from nsys profiler output.
    
    Calculates total GPU time by summing:
    1. All GPU kernel execution times from cuda_gpu_kern_sum
    2. All GPU memory transfer times from cuda_gpu_mem_time_sum
    
    Returns total time in milliseconds.
    """
    if not nsys_output:
        return None

    lines = nsys_output.splitlines()
    total_ns = 0
    
    # Parse GPU kernel times from cuda_gpu_kern_sum
    in_kernel_table = False
    for i, line in enumerate(lines):
        if "cuda_gpu_kern_sum" in line or "CUDA GPU Kernel Summary" in line:
            in_kernel_table = True
            continue

        if not in_kernel_table:
            continue

        if not line.strip():
            if total_ns > 0:
                # End of kernel table, move to memory table
                in_kernel_table = False
                break
            continue

        # Match: Time (%) (float), Total Time (ns) (comma-separated int), Instances (comma-separated int)
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9,]+)\s+([0-9,]+)\s+", line)
        if not m:
            if total_ns > 0:
                in_kernel_table = False
                break
            continue

        _, total_time_ns_str, _ = m.groups()
        try:
            # Remove commas before converting to int
            total_ns += int(total_time_ns_str.replace(',', ''))
        except ValueError:
            continue
    
    # Parse GPU memory transfer times from cuda_gpu_mem_time_sum
    # Track memory time separately to know if we've parsed any memory data
    memory_ns = 0
    in_mem_table = False
    for i, line in enumerate(lines):
        if "cuda_gpu_mem_time_sum" in line or "CUDA GPU Memory Time Summary" in line:
            in_mem_table = True
            continue

        if not in_mem_table:
            continue

        if not line.strip():
            if memory_ns > 0:
                # End of memory table after we've parsed data
                break
            continue

        # Match: Time (%) (float), Total Time (ns) (comma-separated int), Count (int), ...
        # Format: "    96.9    6,337,774,057   65,538  96,703.8  ..."
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9,]+)\s+([0-9,]+)\s+", line)
        if not m:
            # Check if this is a header line or separator
            if "Time (%)" in line or "--------" in line:
                continue
            if memory_ns > 0:
                # End of memory table after we've parsed data
                break
            continue

        _, total_time_ns_str, _ = m.groups()
        try:
            # Remove commas before converting to int
            mem_time = int(total_time_ns_str.replace(',', ''))
            memory_ns += mem_time
            total_ns += mem_time
        except ValueError:
            continue

    if total_ns > 0:
        return total_ns / 1e6  # ns -> ms
    return None


def _measure_performance_with_nsys(kernel_dir: Path, target_api: str, kernel_name: Optional[str]) -> Tuple[bool, Optional[float], str]:
    """Run the nsys + make run command and return (success, runtime_ms, raw_output)."""
    success, output = test_optimized_compilation(kernel_dir, target_api, kernel_name=kernel_name)
    runtime_ms = _parse_nsys_runtime_ms(output) if success else None

        
    # Save nsys output to profile.log for the model to read
    profile_log_path = kernel_dir / "profile.log"
    try:
        with open(profile_log_path, 'w') as f:
            f.write(output)
    except Exception as e:
        print(f"Warning: Could not write profile.log: {e}")
    

    return success, runtime_ms, output


def _extract_relevant_nsys(output: str) -> str:
    """Extract all relevant nsys summary sections from nsys output for passing between stages.
    
    Extracts:
    - OS Runtime Summary (osrt_sum)
    - CUDA API Summary (cuda_api_sum)
    - CUDA GPU Kernel Summary (cuda_gpu_kern_sum)
    - CUDA GPU Memory Time Summary (cuda_gpu_mem_time_sum)
    - CUDA GPU Memory Size Summary (cuda_gpu_mem_size_sum)
    """
    if not output:
        return ""

    lines = output.splitlines()
    summary_lines: List[str] = []
    
    # Add total GPU kernel time if available
    runtime_ms = _parse_nsys_runtime_ms(output)
    if runtime_ms is not None:
        summary_lines.append(f"Total GPU kernel time (nsys): {runtime_ms:.3f} ms")
        summary_lines.append("")

    # Sections to extract (in order)
    sections = [
        ("osrt_sum", "OS Runtime Summary"),
        ("cuda_api_sum", "CUDA API Summary"),
        ("cuda_gpu_kern_sum", "CUDA GPU Kernel Summary"),
        ("cuda_gpu_mem_time_sum", "CUDA GPU Memory Time Summary"),
        ("cuda_gpu_mem_size_sum", "CUDA GPU Memory Size Summary"),
    ]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        section_found = False
        
        # Check if this line starts a new section
        for section_key, section_title in sections:
            if f"Executing '{section_key}'" in line:
                # Found a section start
                section_lines: List[str] = [line]  # Include the header line
                i += 1
                
                # Skip blank line after header
                if i < len(lines) and not lines[i].strip():
                    section_lines.append(lines[i])
                    i += 1
                
                # Collect all lines until we hit the next section or end
                while i < len(lines):
                    current_line = lines[i]
                    # Stop if we hit the next section
                    if any(f"Executing '{sk}'" in current_line for sk, _ in sections if sk != section_key):
                        break
                    # Stop if we hit a blank line followed by a section header (double blank)
                    if not current_line.strip() and i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if any(f"Executing '{sk}'" in next_line for sk, _ in sections):
                            break
                    
                    section_lines.append(current_line)
                    i += 1
                
                # Add section to summary
                if len(section_lines) > 1:  # More than just the header
                    summary_lines.append(f"[{section_title}]")
                    summary_lines.extend(section_lines)
                    summary_lines.append("")  # Blank line between sections
                
                section_found = True
                break
        
        if not section_found:
            i += 1
    
    # If no sections found, provide truncated output
    if len(summary_lines) == (1 if runtime_ms is not None else 0):
        summary_lines.append("Nsight Systems Output (truncated):")
        summary_lines.extend(lines[:100])

    return "\n".join(summary_lines)


def _build_step_prompt(
    target_api: str,
    kernel_dir: Path,
    file_name: str | List[str],
    step: int,
    custom_prompt: Optional[str],
    kernel_name: str,
    prev_transcript_summary: Optional[str] = None,
    source_api: str = 'serial',
) -> str:
    """Construct the per-step optimization prompt. Always include global constraints; prepend previous summaries when available.
    
    Note: nsys profiling results are read from profile.log in kernel_dir, not passed as text.
    """
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_name, str):
        file_names = [file_name]
    else:
        file_names = file_name
    
    file_list_str = ', '.join(file_names)
    file_listing = '\n'.join(f'- {name}' for name in file_names)
    
    context_sections: List[str] = []
    if prev_transcript_summary and step != 1:
        context_sections.append(
            "Previous step transcript summary (for context):\n"
            f"{prev_transcript_summary.strip()}"
        )
    
    # Tell the model to read from profile.log instead of passing nsys summary as text
    profile_log_path = kernel_dir / "profile.log"
    nsys_instructions = ""
    if step > 1:
        nsys_instructions = (
            f"\n**IMPORTANT: Profiling Information**\n"
            f"- Read the previous profiling results from `{profile_log_path}` to understand the current performance characteristics.\n"
            f"- The file contains the full nsys profiling output from the previous step.\n"
            f"- Pay special attention to sections like `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`, and `cuda_api_sum`.\n"
        )
    
    context_header = ""
    if context_sections:
        context_header = "\n\n".join(context_sections) + "\n\n"
    if custom_prompt:
        return context_header + nsys_instructions + custom_prompt

    # Default prompts per-step
    # Use CUDA prompts for parallel-to-parallel translations (source_api != 'serial')
    # Use OMP prompts for serial-to-omp translations (source_api == 'serial' and target_api == 'omp')
    if target_api == 'omp' and source_api == 'serial':
        # Dynamic command generation
        build_cmd_str = get_make_cmd_str(target_api, 'build')
        clean_cmd_str = get_make_cmd_str(target_api, 'clean')
        # Correctness testing: CLASS=A (fallback S)
        correctness_run_cmd = get_correctness_run_cmd_str(target_api, "S", "S")
        correctness_fallback_cmd = get_correctness_fallback_cmd_str(target_api, "S")
        # Profiling: CLASS=C (fallback B)
        profile_run_cmd = get_profile_run_cmd_str(target_api, kernel_name, "C", "B")
        profile_fallback_cmd = get_profile_fallback_cmd_str(target_api, kernel_name, "B")
        nsys_profile_cmd = get_nsys_profile_cmd_str(target_api, kernel_name, use_class="C")
        nsys_profile_fallback_cmd = get_nsys_profile_fallback_cmd_str(target_api, kernel_name, "B")
        
        macros_info = '''the code might contain macros like GATE_CHECKSUM_* or GATE_STATS_*, you should not change them.'''
        hadware_info = '''you need to check what hardware you are running on in `system_info.txt` and use the information to optimize your code.'''
        comment_info = '''you might want to leave comments in the code to explain your changes.'''
        cwd = str(get_codex_workdir())
        if step == 1:
            body = f'''# GPU Offload with OpenMP

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- Use `OMP_TARGET_OFFLOAD=MANDATORY` for all runs
- DO NOT use `distribute parallel for`

## Workflow

### 0. Backup
Save backup of {file_listing}.

### 1. Get Baseline (CLASS A/S)
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > baseline_output.txt 2>&1
# Fallback: timeout 60 {correctness_fallback_cmd} > baseline_output.txt 2>&1
grep -E "Verification|SUCCESSFUL|FAILED" baseline_output.txt

DO NOT SKIP THIS STEP.
```

### 2. Choose Data Strategy
Walk through IN ORDER, stop at first match:

```
RULE 1: Type B (Sparse/CSR)?              → STRATEGY A/C
RULE 2: Type C1 (Iterative Solvers/Butterfly)?→ STRATEGY C
RULE 3: Type C2 (Multigrid)?              → STRATEGY A
RULE 4: Outer A + inner E (per-thread RNG)?→ STRATEGY A
RULE 5: Multiple independent kernels?     → STRATEGY B
RULE 6: Otherwise                         → STRATEGY A
```

### 2.5. Create Data Management Plan
MANDATORY: Create data_plan.md in {kernel_dir} before implementation

**FIRST: Check if original algorithm can be simplified for GPU:**
- Large scratch arrays for intermediate results → Can per-thread locals replace them?
- Block-based iteration (for cache) → REMOVE blocking, use single parallel loop over ALL work items
- Multi-stage with host sync → Can everything run in one kernel?

**Rule:** If scratch arrays exist ONLY to avoid atomics on small data (<1KB), 
DELETE them and use per-thread locals + atomic merge instead.

**Block elimination:** If code has `for (blk = 0; blk < numblks; blk++)` with scratch arrays,
this is a CPU cache optimization. For GPU: remove blocking, parallelize over all N items directly.

Analyze ALL arrays and functions in timed region:

```markdown

# Data Management Plan

## Arrays Inventory
List ALL arrays used in timed region:

| Array Name | Size | Type | Init | Access |
|------------|------|------|------|--------|
| [name] | [bytes] | working/scratch/const/index | host/device | R/W/RO |

**Types:** working (main data), scratch (temp), const (read-only), index (maps)

## Functions in Timed Region
| Function | Arrays Accessed | Frequency | Must Run On |
|----------|----------------|-----------|-------------|
| [name] | [list] | per-iteration/once | device/host |

## Data Movement Strategy

**Chosen Strategy:** [A/B/C]

**Device Allocations (once):**
```
Strategy C: d_[array]: [size] via omp_target_alloc
Strategy A: [arrays] in target data region
```

**Host→Device Transfers:**
- When: [before iterations/once at start]
- Arrays: [array1]→d_[array1] ([size] MB)
- Total H→D: ~[X] MB

**Device→Host Transfers:**
- When: [after iterations/once at end]
- Arrays: d_[array1]→[array1] ([size] MB)
- Total D→H: ~[Y] MB

**Transfers During Iterations:** [YES/NO]
- If YES: [which arrays and why]
- If NO: All data stays on device

## Critical Checks (for chosen strategy)

**Strategy A:**
- [ ] Functions inside target data use `present,alloc` wrapper?
- [ ] Scratch arrays use enter/exit data OR omp_target_alloc?

**Strategy C:**
- [ ] ALL functions in iteration loop use is_device_ptr?
- [ ] Scratch arrays allocated on device (not host)?
- [ ] No map() clauses (only is_device_ptr)?

**Common Mistakes:**
-  Some functions on device, others on host (causes copying)
-  Scratch as host arrays in Strategy C
-  Forgetting to offload ALL functions in loop

## Expected Transfer Volume
- Total: ~[X+Y] MB for entire execution
- **Red flag:** If actual >2x expected → data management wrong

## Additional Parallelization Notes
- **RNG Replicable?** [YES/NO] - If YES, use `#pragma omp declare target` on RNG function
- **Outer Saturation?** [outer iters]
- **Sparse Matrix NONZER?** [value]
- **Histogram Strategy?** For small bin (≤ 100) counts: use per-thread local array + atomic merge (NO scratch arrays needed!)

**Summary:** [num] arrays ([num] scratch, [num] working), [num] functions, Strategy [A/B/C]. Expected: ~[X] MB H→D, ~[Y] MB D→H.
```

### 2.6. Implement Data Plan

**Use data_plan.md as implementation guide**

### Step 1: Setup Data Structures
From "Arrays Inventory" and "Data Movement Strategy":
- Declare device arrays/pointers as needed for chosen strategy
- Create allocation/initialization functions based on strategy:
  - **Strategy A:** Setup target data regions with map clauses from plan
  - **Strategy B:** Prepare depend clauses for async operations
  - **Strategy C:** Create omp_target_alloc calls using sizes from plan

### Step 2: Implement Transfers
From "H→D Transfers" and "D→H Transfers" sections:
- Implement each transfer listed with timing specified in plan
- Use method appropriate for strategy (map clauses, omp_target_memcpy, update, etc.)

### Step 3: Offload Functions
Use "Functions in Timed Region" table:
- For each function where "Must Run On" = device:
  - Add appropriate pragma for strategy
  - Include arrays from "Arrays Accessed" column
  - Follow strategy-specific patterns from Step 2

### Step 4: Main Program Flow
Follow "Data Movement Strategy" timing:
```
[setup from plan]
[H→D transfers at specified time]
[timed computation - call functions]
[D→H transfers at specified time]
[cleanup]
```

### Step 5: Verify Implementation
Check ALL items in "Critical Checks" section for YOUR strategy:
- [ ] Verify each checkpoint matches implementation
- [ ] Cross-reference "Functions in Timed Region" table
- [ ] Confirm transfer timing matches plan

**Common errors:** Mismatched array names, missing functions from table, wrong transfer timing

**Ready when:** All strategy-specific checks ✓ and compiles
---

## Strategy Details

### STRATEGY A: target data Region

**Map Clause Selection:**
| Scenario | Map Clause | Why |
|----------|------------|-----|
| Device-init arrays (zero(), fill()) | `alloc` | Avoid copying garbage |
| Host RNG init then sync | `alloc` + `update to` | Explicit sync after host init |
| Read + modify + write | `tofrom` | Bidirectional |
| Read-only | `to` | One-way |

**Functions Called Inside target data:**
Wrap with `present,alloc`/'to,tofrom', then use bare `target teams loop`:
```c
void compute(double *u, double *v, int n) {{
  #pragma omp target data map(present,alloc:u[0:n],v[0:n])
  {{
    #pragma omp target teams loop
    for (int i = 0; i < n; i++) {{ ... }}
  }}
}}
```

**RNG replicable:**
```c
#pragma omp target teams loop reduction(+:sum1, sum2) firstprivate(seed_base, params)
for (int sample = 0; sample < N; ++sample) {{
  double rng_state = compute_seed_for_sample(sample);  // Per-thread seed
  double local_hist[BINS] = {{0}};  // Per-thread histogram
  
  // Type E (RNG) is sequential WITHIN this thread
  for (int i = 0; i < work_per_sample; ++i) {{
    double r = my_rng(&rng_state, A);
    int bin = compute_bin(r);
    local_hist[bin] += 1.0;
    sum1 += ...; sum2 += ...;  // Reduction handles these
  }}
  
  // Atomic merge histogram at end
  for (int b = 0; b < BINS; ++b) {{
    if (local_hist[b] != 0.0) {{
      #pragma omp atomic update
      global_hist[b] += local_hist[b];
    }}
  }}
}}
```

**Scratch Arrays (two options):**

- **Option 1: enter/exit data**
```c
double scratch[N];
#pragma omp target enter data map(alloc:scratch[0:n])
#pragma omp target data map(present,alloc:in[0:n])
{{
  #pragma omp target teams loop
  for (...) {{ /* use scratch */ }}
}}
#pragma omp target exit data map(delete:scratch[0:n])
```

- **Option 2: omp_target_alloc**
```c
double *scratch = (double*)omp_target_alloc(n*sizeof(double), 0);
#pragma omp target data map(present,alloc:in[0:n])
{{
  #pragma omp target teams loop is_device_ptr(scratch)
  for (...) {{ ... }}
}}
omp_target_free(scratch, 0);
```

**Mid-computation sync:**
```c
#pragma omp target update from(result)
host_compute(result);
#pragma omp target update to(indices)
```

### STRATEGY B: Asynchronous Offload
Use when: Overlapping compute/transfer possible
```c
#pragma omp target teams loop nowait depend(out:x[0])
for (i = 0; i < N; i++) {{ x[i] = init(i); }}

#pragma omp target teams loop nowait depend(in:x[0]) depend(out:y[0])
for (i = 0; i < N; i++) {{ y[i] = compute(x[i]); }}

#pragma omp taskwait
```

STRATEGY C: Global Device State (Iterative Solvers)
Use omp_target_alloc + is_device_ptr for all device arrays.

**Pattern:**
```c
// Device pointers: static double *d_arr
allocate_device_arrays();  // omp_target_alloc once
copy_to_device();          // omp_target_memcpy once

for (iter ...) {{
  #pragma omp target teams is_device_ptr(d_arr1, d_arr2, ...)
  {{ ... }}
}}

free_device_arrays();
```

**Key Rules:**
- Use `is_device_ptr` everywhere (no map clauses in hot path)
- Reduction helpers (dot, norm) OK - they return scalars
- stage loops: parallelize outer k,j; keep stage loop L serial
- Iterative solvers: inline SpMV, updates in main loop
---

### 3. Map Globals & Functions
```c
#pragma omp declare target
double helper_func() {{ ... }};
#pragma omp end declare target

#pragma omp declare target(global_var)
```
---

## 4. Parallelize loops

**Parallelization patterns:**

**Type A (Dense):**
```c
#pragma omp target teams loop collapse(2)
for (i = 0; i < N; i++)
  for (j = 0; j < M; j++) ...
```

**Type B (Sparse/CSR) - Nested Parallelism:**
```c
int tmp1, tmp2, tmp3;  // Function scope
#pragma omp target teams loop is_device_ptr(...)
for (int row = 0; row < nrows; row++) {{
  tmp1 = rowptr[row];
  tmp2 = rowptr[row+1];
  double sum = 0.0;
  ***#pragma omp loop reduction(+:sum)***  // Parallelize inner *based on GPU saturation* 
  for (int k = tmp1; k < tmp2; k++) {{
    tmp3 = colidx[k];
    sum += A[k] * x[tmp3];
  }}
  y[row] = sum;
}}
```

**Type C1 (Iterative Solvers) - Serial Inner:**
```c
#pragma omp target teams is_device_ptr(...)
{{
#pragma omp loop collapse(2)
  for (k = 0; k < K; k++) {{
    for (j = 0; j < J; j++) {{
      for (stage = 0; stage < S; stage++) {{ ... }}  // No pragma - keep inner serial!
    }}
  }}
}}
**Rationale:** K×J teams already saturate GPU. Inner serial = better register reuse, no barriers.
```

**Type C2 (Multigrid):** Wrap with `present,alloc`; each stencil call gets `target teams loop`.

**Type C special rule:** Stage-dependent algorithms (multigrid, iterative stages) 
should NEVER have inner parallelism, regardless of GPU. The barrier overhead between 
stages exceeds any benefit from inner thread parallelism.

**Type D (Histogram):** Add `#pragma omp atomic` on indirect writes.

**Type F (Reduction):** `reduction(+:sum)`

**Type G (Stencil):** `collapse(2)` on spatial dimensions.

**Type A+E (Outer parallel, inner RNG):** 
**When analysis says "RNG replicable: YES":**
- Add `declare target` on RNG function - GPU callable.
- Parallelize over samples, each thread has private RNG + histogram
- Atomic merge histogram at the end

## Histogram Optimization 
If histogram bins ≤ 100:
```c
// GOOD: Per-thread local array (80 bytes for 10 bins)
#pragma omp target teams loop reduction(+:sx, sy)
for (int k = 0; k < N; ++k) {{
  double q_local[BINS] = {{0}};  // Thread-private
  // ... accumulate into q_local ...
  for (int b = 0; b < BINS; ++b) {{
    if (q_local[b] != 0.0) {{
      #pragma omp atomic update
      q[b] += q_local[b];
    }}
  }}
}}
```
**DO NOT** create large scratch arrays for small histograms - the atomic overhead is negligible compared to memory transfer costs.
**Key:** Each thread replicates the RNG state for its sample. Type E becomes parallelizable at the OUTER level.

## 5. Compile and Test (CLASS A/S)
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {correctness_run_cmd} > gpu_output.txt 2>&1
# Fallback: timeout 60 {correctness_fallback_cmd} > gpu_output.txt 2>&1
```

If timeout/segfault: Remove `#pragma omp loop` from Type C inner loops.

## 6. Verify Correctness
```bash
diff baseline_output.txt gpu_output.txt
```

## 8. Profile (CLASS B/C)
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
# Check for kernel information (OpenMP kernels may appear in cuda_gpu_kern_sum or with different names)
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10 || echo "No kernel information found - check if code is offloading to GPU"
```

#**RULES** BRAKING A RULE = FAILURE.
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- DO NOT EDIT MAKEFILES.
- ALWAYS CLEAN BEFORE BUILD.
- DO NOT CHANGE/EDIT FILES OTHER THEN {file_listing}
'''
        else:
            transcript_context_text = (
                prev_transcript_summary.strip()
                if prev_transcript_summary
                else "Not available; ensure the previous step completed and saved its transcript summary."
            )
            body = f'''
# Performance Tuning

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Do not change data strategy from used in the code**

## EARLY EXIT CHECK
If current runtime is within 5% of expected optimal (based on nsys kernel times):
- Document current metrics in optimization_plan.md
- Skip optimization - code is already well-tuned
- Focus only on micro-optimizations (const, restrict, cache locals)

## Workflow

### 1. Verify Baseline (CLASS A/S)
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
# Fallback: timeout 60 {correctness_fallback_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt | grep -E "Verification|SUCCESSFUL|FAILED"
```

If results differ, fix Step 2 first.
If there are any errors, fix them before continuing.

### 2. Analyze Profile and Create Plan
 1.1. Read profile data:
 ```bash
# Try to find kernel information (OpenMP kernels may not appear in standard sections)
cat {profile_log_path} | grep -A20 "cuda_gpu_kern_sum" || echo "No cuda_gpu_kern_sum found - kernels may not be offloading to GPU"
cat {profile_log_path} | grep -A10 "cuda_api_sum"
cat {profile_log_path} | grep -A10 "cuda_gpu_mem_time_sum"
# Also check for any GPU activity
cat {profile_log_path} | grep -i "gpu\|kernel\|target" | head -20
```
 1.2. Run 
 ```bush
 nvidia-smi --query-gpu=name,compute_cap --format=csv
 ```
 roughly estimate the GPU saturation threshold
---

2. Create optimization_plan.md in {kernel_dir}:
```markdown
# Performance Analysis

## Current Metrics
- Runtime: [X]s
- Main kernel: [name], [Y]% GPU, [Z] instances
- Memory transfer: [%] time, [MB] total
- Kernel launches: [count]

## Fusion Opportunities:

### Identified Fusions:
- Lines X-Y: init → FUSE (same bounds)
- Lines A-B: compute+reduce → FUSE (register value)

## Iteration Loop (if present):
- Main: lines [X-Y], [N] iters
- SpMV line Z: [N] times
- Update line W: [N] times
- Total: [N×M] ops

## SpMV Inner Loop Decision
- Avg nonzeros per row (NONZER): [value from code/headers]
- If NONZER < 50: Keep inner loop SERIAL
- If NONZER > 100: Add `#pragma omp loop reduction`

## Bottleneck Checklist (priority order)
| Issue | Symptom | Fix |
|-------|---------|-----|
| Data transfers | >30% transfer time | Move to Strategy C, use is_device_ptr |
| Launch overhead | instances >> iterations | Inline helper functions |
| Over-parallelization | Type C slow, outer saturated | Remove inner pragmas |
| Hot kernel | One kernel >50% time | collapse, simd, cache locals |
| Stage parallelization | FAIL verification | Remove pragma from stage loops |


## Strategy (priority)
1. [ACTION]: [what] - [why] - expect [gain]
2. [ACTION]: [what] - [why] - expect [gain]

## Micro-opts
[ ] const, restrict, firstprivate, cache locals

## Target
- Runtime: [X]s
- Kernels: ~[N] for [M] iters
- Memory: <[X]%
```
### Fusion rules

**Fuse when:**
- Adjacent independent, same bounds
- Producer-consumer
- Multi-vector ops

**Don't fuse:**
- Different bounds
- Intermediate sync required

### 3. Execute Optimization Plan
- Apply changes and document in optimization_plan.md

### 4. Optimization Actions

### 4A. Fix Data Movement

- Hoist target data outside loops
- omp_target_alloc + is_device_ptr for scratch
- Remove map inside target data
- Wrap functions: present,alloc
- Host init: target update to after

### 4B. Optimize Hot Kernel

- Use combined target teams loop
- Type B: Add inner #pragma omp loop reduction(+:sum)
- collapse(N) on nested dense loops
- Add #pragma omp simd to innermost
- Cache array accesses (SpMV/CSR):

```c
int tmp1, tmp2, tmp3;  // Function scope
#pragma omp target teams loop is_device_ptr(...)
for (int i = 0; i < nrows; i++) {{
  tmp1 = d_rowptr[i];
  tmp2 = d_rowptr[i+1];
  double sum = 0.0;
  #pragma omp loop reduction(+:sum)
  for (int k = tmp1; k < tmp2; k++) {{
    tmp3 = d_col[k];
    sum += d_val[k] * d_x[tmp3];
  }}
  d_y[i] = sum;
}}
```

### 4C. Launch Overhead

**Rule:** If kernel instances >> iteration count, inline helper functions in the main loop.
- Keep reduction helpers (dot, norm) - they return scalars
- Inline SpMV, vector updates, scaling operations
- Fuse adjacent loops with same bounds

### 4D. Fix Type C1 (Multi-Stage)

Outer loops: collapse(2) on spatial dimensions
Inner stage loops: Remove all pragmas (must be serial)

### 4E. Increase Parallelism

- Increase collapse depth
-  Use tile sizes(32, 32)
- Remove manual num_teams/thread_limit

### 5. Final Summary
Update optimization_plan.md:
```markdown
# Final Performance Summary

### Baseline (Step 2)
- Runtime: [X]s
- Main kernel: [Y] instances, [Z]ms total

### Final (Step 3)
- Runtime: [X]s
- Speedup: [X]x
- Main kernel: [Y] instances, [Z]ms total

### Optimizations Applied
1. [] [ACTION]: [description] → [±X%]
2. [] [ACTION]: REVERTED (slower)

### Micro-optimizations Applied
1. [] [MICRO-OPT]: [description] → [±X%]
2. [] [MICRO-OPT]: REVERTED (slower)

### Key Insights
- [Most impactful optimization]
- [Remaining bottlenecks]
```

**Reference: Available Opts**
## Bottlenecks (mark applicable)
### [ ] 1. Data Management Issue (CRITICAL - fix first!)
- Transfer ratio: [actual/expected] = [X]x
- If >2.5x: Data management wrong
- Root cause: [from data_plan.md verification]
- Fix: [specific action - e.g., offload missing functions, move scratch to device]
- Expected gain: [X]x speedup

### [ ] 2. Kernel Launch Overhead
- Kernel instances: [count]
- Expected: ~[N] for [N] iterations
- If instances >> N: Helper functions called in loop
- Root cause: [which functions - e.g., device_spmv, device_axpy]
- Fix: Inline operations in loop (ACTION 4C)
- Expected gain: [X]x (reduce [Y] launches to [Z])

### [ ] 3. Memory Transfer Bottleneck
- Transfer time: [X]% of total time
- If >50% AND ratio <2x: Transfers correct but dominant
- Fix: Optimize data movement (ACTION 4A)
- Expected gain: [X]%

### [ ] 4. Hot Kernel Performance
- Kernel: [name] takes [X]% GPU time, [Y]ms avg
- Root cause: [inefficient algorithm/missing optimization]
- Fix: [collapse/simd/cache/etc.] (ACTION 4B)
- Expected gain: [X]% faster kernel

### [ ] 5. Type C Parallelization Error
- Verification: [PASS/FAIL]
- If FAIL: Wrong stage loop parallelization
- Fix: Remove inner pragmas (ACTION 4D)

[ ] 6. Over-Parallelization (saturated outer loops)
- Outer parallelized iterations: [K × J = ?]
- Saturation threshold: [Saturation threshold]
- IF saturated AND inner has pragma → REMOVE inner pragmas
- Symptoms: Type C kernel slower after (or before) "optimization", GPU over-saturated
- Fix: Remove collapse/omp loop from inner/stage/writeback loops
- Expected gain: [X]%

## Profiling (CLASS B/C)
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
# Check for kernel information (OpenMP kernels may appear in cuda_gpu_kern_sum or with different names)
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10 || echo "No kernel information found - check if code is offloading to GPU"
```

### Deliverables
- optimization_plan.md - Complete analysis and results
- Optimized source code
- Final profile: {profile_log_path}

#**RULES** BRAKING A RULE = FAILURE.
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- DO NOT EDIT MAKEFILES.
- ALWAYS CLEAN BEFORE BUILD.
- DO NOT CHANGE FILES OTHER THEN {file_listing}
'''
        return (
            f"Directory: {kernel_dir}\n\n"
            f"{body}\n\n"
            f"{comment_info}\n\n"
            f"{macros_info}\n\n"
            f"{hadware_info}\n\n"
        )
    else:
        nsys_cmd = _nsys_profile_cmd_str(target_api, kernel_name)
        build_cmd_str = get_make_cmd_str(target_api, 'build')
        clean_cmd_str = get_make_cmd_str(target_api, 'clean')
        run_cmd_str = get_make_cmd_str(target_api, 'run')
        # Correctness testing: CLASS=A (fallback S)
        correctness_run_cmd = get_correctness_run_cmd_str(target_api, "S", "S")
        correctness_fallback_cmd = get_correctness_fallback_cmd_str(target_api, "S")
        # Profiling: CLASS=C (fallback B)
        nsys_profile_cmd = get_nsys_profile_cmd_str(target_api, kernel_name, use_class="C")
        nsys_profile_fallback_cmd = get_nsys_profile_fallback_cmd_str(target_api, kernel_name, "B")
        macros_info = '''the code might contain macros like GATE_CHECKSUM_* or GATE_STATS_*, you should not change them.'''
        hadware_info = '''you need to check what hardware you are running on in `system_info.txt` and use the information to optimize your code.'''
        comment_info = '''you might want to leave comments in the code to explain your changes.'''
        cwd = str(get_codex_workdir())
        if step == 1:
            body = f'''# CUDA to OpenMP Migration

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Reference:** `{kernel_dir}/analysis.md`

**Required:** 
- Use `OMP_TARGET_OFFLOAD=MANDATORY` for all runs
- DO NOT use `distribute parallel for`

** IMPORTANT ** YOU MAY MODIFY THE MAKEFILE TO ADD ANYTHING YOU NEED TO RUN THE CODE.

## Workflow

### 0. Backup
Save backup of {file_listing}.

### 1. Get Baseline
```bash
Baseline cuda outpuut is in baseline_output.txt in {kernel_dir}/
```

### 2. Choose Data Strategy
Walk through IN ORDER, stop at first match:

```
RULE 1: Type B (Sparse/CSR)?              → STRATEGY A/C
RULE 2: Type C1 (Iterative Solvers/Butterfly)?→ STRATEGY C
RULE 3: Type C2 (Multigrid)?              → STRATEGY A
RULE 4: Multiple independent kernels?     → STRATEGY B
RULE 5: Otherwise                         → STRATEGY A
```

### 2.5. Create Data Management Plan
MANDATORY: Create data_plan.md in {kernel_dir} before implementation

**FIRST: Understand CUDA memory model and map to OMP:**
- cudaMalloc + device pointers → omp_target_alloc OR target data map(alloc)
- cudaMemcpy H→D → map(to) OR omp_target_memcpy OR update to
- cudaMemcpy D→H → map(from) OR omp_target_memcpy OR update from
- Kernel launches in loops → target teams loop with is_device_ptr

**CUDA Pattern Recognition:**
```
Pattern 1: cudaMalloc once → kernel loop → cudaFree
  → Strategy C: omp_target_alloc + is_device_ptr

Pattern 2: Single kernel launch with data transfer
  → Strategy A: target data region

Pattern 3: Multiple kernels with dependencies
  → Strategy B: nowait + depend clauses
```

Analyze ALL arrays and kernels in timed region:

```markdown
# Data Management Plan

## CUDA Memory Analysis
List ALL device allocations and transfers:

| Array/Pointer | CUDA Allocation | Size | Transfer Pattern |
|---------------|-----------------|------|------------------|
| d_[name] | cudaMalloc | [bytes] | H→D once/D→H once/both |
| [name] | host array | [bytes] | source/destination |

**CUDA Operations:**
- cudaMalloc calls: [list with sizes]
- cudaMemcpy H→D: [list with timing]
- cudaMemcpy D→H: [list with timing]
- Kernel launches: [list with frequency]

## Kernel Inventory
| Kernel Name | Launch Config | Frequency | Arrays Used |
|-------------|---------------|-----------|-------------|
| kernel_name<<<G,B>>> | grid=[X], block=[Y] | per-iteration/once | [list] |

**Kernel Launch Patterns:**
- In outer loop? → Multiple target teams loop
- Sequential kernels? → Multiple target regions OR nowait+depend
- Conditional launch? → target if clause

## OMP Data Movement Strategy

**Chosen Strategy:** [A/B/C]

**Rationale:** [Map CUDA pattern to strategy]

**Device Allocations (OMP equivalent):**
```
CUDA: cudaMalloc(&d_arr, size)
OMP Strategy C: d_arr = omp_target_alloc(size, 0)
OMP Strategy A: #pragma omp target data map(alloc:arr[0:n])
```

**Host→Device Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice)
OMP Strategy C: omp_target_memcpy(d_arr, h_arr, size, 0, 0, 0, omp_get_initial_device())
OMP Strategy A: map(to:arr[0:n]) OR #pragma omp target update to(arr[0:n])
```
- When: [before iterations/once at start]
- Arrays: [list with sizes]
- Total H→D: ~[X] MB

**Device→Host Transfers (OMP equivalent):**
```
CUDA: cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost)
OMP Strategy C: omp_target_memcpy(h_arr, d_arr, size, 0, 0, omp_get_initial_device(), 0)
OMP Strategy A: map(from:arr[0:n]) OR #pragma omp target update from(arr[0:n])
```
- When: [after iterations/once at end]
- Arrays: [list with sizes]
- Total D→H: ~[Y] MB

**Transfers During Iterations:** [YES/NO]
- If YES: [which arrays and why - may indicate wrong strategy]

## Kernel to OMP Mapping (short)
- Replace each CUDA kernel launch with a `#pragma omp target teams loop` over the same *logical* work domain.
- Replace `blockIdx/threadIdx` indexing with the loop induction variable.
- Keep bounds checks; keep inner device loops as normal C loops inside the offloaded loop body.

## Critical Migration Issues

**From analysis.md "OMP Migration Issues":**
- [ ] __syncthreads() usage: [locations and resolution strategy]
- [ ] Shared memory: [convert to private/firstprivate]
- [ ] Atomics: [verify OMP atomic equivalents]
- [ ] Dynamic indexing: [verify OMP handles correctly]

**__syncthreads() Resolution:**
- Within single kernel → May need to split into multiple target regions
- At kernel boundaries → Natural OMP barrier between target regions
- Strategy: [describe approach]

**Shared memory / barriers:**
- No direct equivalent for CUDA `__shared__` + `__syncthreads()`; refactor and document your approach.

## Expected Performance
- CUDA kernel time: [X] ms (from profiling if available)
- OMP expected: [Y] ms (may be slower due to __syncthreads elimination)
- Red flag: If >3x slower → wrong strategy or missing parallelism

**Summary:** [num] kernels, [num] device arrays, Strategy [A/B/C]. 
CUDA pattern: [describe]. OMP approach: [describe].
Expected: ~[X] MB H→D, ~[Y] MB D→H.
```

### 2.6. Implement Data Plan

**Use data_plan.md as implementation guide**

### Step 1: Remove CUDA API Calls
From "CUDA Memory Analysis":
- Remove all cudaMalloc/cudaFree calls
- Remove all cudaMemcpy calls
- Remove kernel launch syntax <<<grid, block>>>
- Keep all kernel BODY code (will convert to functions)

### Step 2: Convert Kernels to Functions
From "Kernel Inventory":
```
CUDA:
  __global__ void kernel_name(double *arr, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = ...;
  }}

OMP:
  void kernel_name(double *arr, int n) {{
    #pragma omp target teams loop is_device_ptr(arr)
    for (int idx = 0; idx < n; idx++) {{  
      arr[idx] = ...;
    }}
  }}
```

### Step 3: Setup Data Structures
From "OMP Data Movement Strategy":
- Create OMP allocations based on chosen strategy
- For Strategy C: Add omp_target_alloc calls
- For Strategy A: Setup target data regions

### Step 4: Implement Transfers
From "Host→Device" and "Device→Host" sections:
- Implement transfers using method for chosen strategy
- Match timing from original CUDA code

### Step 5: Convert Thread Indexing
From "Thread Indexing Conversion":
- Replace blockIdx/threadIdx with loop iterator
- Remove if (idx < N) guards (loop bounds handle this)
- Convert grid-stride loops to simple loops

### Step 6: Handle Special CUDA Constructs
From "Critical Migration Issues":
- **atomicAdd** → `#pragma omp atomic update`
- **__syncthreads()** → Split kernel OR remove if not critical
- **Shared memory** → Per-thread private OR elimination
- **Reduction in kernel** → `reduction(op:var)` clause

### Step 7: Verify Implementation
Check ALL items in "Critical Migration Issues":
- [ ] All kernels converted to OMP functions
- [ ] Thread indexing removed
- [ ] Memory management matches strategy
- [ ] Special constructs handled

**Common errors:** 
- Forgot to remove <<<>>> syntax
- Left blockIdx/threadIdx in code
- Missed cudaMemcpy conversions
- Wrong is_device_ptr usage

**CRITICAL: OpenMP Clause Syntax Limitation**
OpenMP pragma clauses (`is_device_ptr`, `use_device_addr`, `map`) do NOT support struct member access.
You MUST extract struct members to local pointer variables first.

WRONG (will not compile):
```c
#pragma omp target teams loop is_device_ptr(data.arr1, data.arr2)
```

CORRECT:
```c
double *d_arr1 = data.arr1;
double *d_arr2 = data.arr2;
#pragma omp target teams loop is_device_ptr(d_arr1, d_arr2)
for (int i = 0; i < n; i++) {{
    // use d_arr1[i], d_arr2[i] inside the loop
}}
```

When converting CUDA code that passes structs to kernels, extract ALL device pointer members
to local variables BEFORE the pragma, then use those local variables in the clause AND loop body.

**Ready when:** Compiles and runs with OMP flags, no CUDA API calls remain

---

## Strategy / Pattern Notes (short)
- Strategy A: `target data map(...)` for simpler flows (few kernels).
- Strategy C: `omp_target_alloc` + `omp_target_memcpy` + `is_device_ptr` for persistent device pointers (CUDA-like).
- Device helpers: former `__device__` helpers typically need `#pragma omp declare target`.

## 5. Compile and Test
```bash
{clean_cmd_str}
{build_cmd_str}
timeout 300 {run_cmd_str} > gpu_output.txt 2>&1
```

If timeout/segfault: Check for unconverted CUDA constructs.
If core dumped/Aborted: run compute sanitizer.

## 6. Verify Correctness
```bash
diff baseline_output.txt gpu_output.txt
```

## 8. Profile
```bash
{clean_cmd_str}
{nsys_profile_cmd} > {profile_log_path} 2>&1
# Fallback: {nsys_profile_fallback_cmd} > {profile_log_path} 2>&1
# Check for kernel information (OpenMP kernels may appear in cuda_gpu_kern_sum or with different names)
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10 || echo "No kernel information found - check if code is offloading to GPU"
```

## RULES - BREAKING A RULE = FAILURE
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- DO NOT EDIT MAKEFILES.
- ALWAYS CLEAN BEFORE BUILD.
- YOU MAY MODIFY THE MAKEFILE TO ADD ANYTHING YOU NEED TO RUN THE CODE.
- REMOVE ALL CUDA API CALLS (cudaMalloc, cudaMemcpy, cudaFree, kernel<<<>>>)
- CONVERT ALL __global__ FUNCTIONS TO REGULAR FUNCTIONS
- REMOVE ALL CUDA-SPECIFIC SYNTAX (blockIdx, threadIdx, __syncthreads, __shared__)
'''

        else:
            transcript_context_text = (
                prev_transcript_summary.strip()
                if prev_transcript_summary
                else "Not available; ensure the previous step completed and saved its transcript summary."
            )
            body = f'''
# Performance Tuning - CUDA to OMP Migration

**Directory:** `{kernel_dir}/`  
**Files:** {file_listing}  
**Profile:** `{profile_log_path}`
**Do not change data strategy from used in the code**

## EARLY EXIT CHECK
If current runtime is within 5% of expected optimal (based on nsys kernel times):
- Document current metrics in optimization_plan.md
- Skip optimization - code is already well-tuned
- Focus only on micro-optimizations (const, restrict, cache locals)

## Context: CUDA to OMP Migration
The code was migrated from CUDA to OMP. Key differences affect optimization:
- CUDA kernels → OMP target teams loop
- cudaMemcpy → OMP map clauses or omp_target_memcpy
- __syncthreads() → May have been split into multiple target regions
- Shared memory → Converted to private or eliminated
- atomicAdd → OMP atomic

**Common migration bottlenecks:**
1. Excessive data transfers (lost explicit CUDA control)
2. Over-decomposed kernels (from __syncthreads() elimination)
3. Missing collapse on nested loops (CUDA had 2D/3D grids)
4. Suboptimal thread mapping (CUDA grid-stride → OMP loop)

## Workflow

### 1. Verify Baseline
```bash
cd {kernel_dir}
{clean_cmd_str}
timeout 300 {correctness_run_cmd} > current_output.txt 2>&1
diff baseline_output.txt current_output.txt | grep -E "Verification|SUCCESSFUL|FAILED"
```

If results differ, fix Step 2 first.
If there are any errors, fix them before continuing.

### 2. Analyze Profile and Create Plan

2.1. Read profile data:
```bash
# Try to find kernel information (OpenMP kernels may not appear in standard sections)
cat {profile_log_path} | grep -A20 "cuda_gpu_kern_sum" || echo "No cuda_gpu_kern_sum found - kernels may not be offloading to GPU"
cat {profile_log_path} | grep -A10 "cuda_api_sum"
cat {profile_log_path} | grep -A10 "cuda_gpu_mem_time_sum"
# Also check for any GPU activity
cat {profile_log_path} | grep -i "gpu\|kernel\|target" | head -20
```

2.2. Check GPU capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```
Roughly estimate the GPU saturation threshold

2.3. Compare with original CUDA performance (if available):
- CUDA kernel time: [X]ms
- OMP target teams loop time: [Y]ms
- Ratio: [Y/X]
- If >2x slower: Major optimization opportunity

---

3. Create optimization_plan.md in {kernel_dir}:
```markdown
# Performance Analysis - CUDA to OMP Migration

## Current Metrics
- Runtime: [X]s
- Main kernel: [name], [Y]% GPU, [Z] instances
- Memory transfer: [%] time, [MB] total
- Kernel launches: [count]

## Bottleneck Hypothesis (pick 1–2)
- [ ] Transfers too high (CUDA avoided transfers in loop)
- [ ] Too many kernels / target regions (launch overhead)
- [ ] Missing collapse vs CUDA grid dimensionality
- [ ] Hot kernel needs micro-opts

## Actions (1–3 max)
1. [ACTION]: [what] - [why] - expected [gain]
2. [ACTION]: ...
```

### Fusion Rules

**Fuse when:**
- CUDA had single kernel for operations
- Adjacent independent, same bounds
- Producer-consumer in CUDA
- Multi-vector ops in one CUDA kernel

**Don't fuse:**
- Different bounds
- CUDA had separate kernels with cudaDeviceSynchronize()
- __syncthreads() required synchronization

### 3. Execute Optimization Plan
- Apply changes and document in optimization_plan.md

### 4. Optimization Actions (short)
- **Transfers high**: hoist data; use `omp_target_alloc` + `is_device_ptr` for persistent arrays; avoid per-iteration mapping
- **Too many target regions**: fuse adjacent target loops; inline helper kernels when safe
- **Grid shape mismatch**: add `collapse(N)` to mirror CUDA grid dimensionality
- **Kernel micro-opts**: `const`, `restrict`, cache locals, reduce recomputation

### 5. Final Summary
Update optimization_plan.md:
```markdown
# Final Performance Summary - CUDA to OMP Migration

### Baseline (from CUDA)
- CUDA Runtime: [X]s (if available)
- CUDA Main kernel: [Y] launches, [Z]ms total

### OMP Before Optimization
- Runtime: [X]s
- Slowdown vs CUDA: [X]x
- Main kernel: [Y] instances, [Z]ms total

### OMP After Optimization
- Runtime: [X]s
- Slowdown vs CUDA: [X]x (target <1.5x)
- Speedup vs initial OMP: [X]x
- Main kernel: [Y] instances, [Z]ms total

### Optimizations Applied
1. [X] [ACTION]: [description] → [±X%] [recovered CUDA pattern Y]
2. [X] [ACTION]: REVERTED (slower)

### CUDA→OMP Recovery Status
- [X] Restored 2D/3D grid mapping with collapse
- [X] Matched CUDA kernel fusion structure
- [X] Eliminated excessive transfers (matched CUDA pattern)
- [ ] Still missing: [any CUDA optimizations that couldn't be recovered]

### Micro-optimizations Applied
1. [X] [MICRO-OPT]: [description] → [±X%]
2. [X] [MICRO-OPT]: REVERTED (slower)

### Key Insights
- [Most impactful optimization - relate to CUDA pattern]
- [Remaining bottlenecks vs CUDA]
- [OMP limitations compared to CUDA]
```

## Optimization Checklist (short)
- [ ] Transfers dominate: hoist data; `omp_target_alloc` + `is_device_ptr`; avoid per-iter mapping
- [ ] Too many kernels/regions: fuse adjacent target loops; inline helper kernels when safe
- [ ] Missing CUDA grid shape: add `collapse(N)`
- [ ] Hot kernel: `const`, `restrict`, cache locals, reduce recomputation (and `simd` where safe)

## Profiling
```bash
{clean_cmd_str}
# Fallback: {build_cmd_str} run > {profile_log_path} 2>&1
# Check for kernel information (OpenMP kernels may appear in cuda_gpu_kern_sum or with different names)
grep -E "cuda_gpu_kern|CUDA GPU Kernel|GPU activities" {profile_log_path} | head -10 || echo "No kernel information found - check if code is offloading to GPU"
```

### Deliverables
- optimization_plan.md - Complete analysis including CUDA comparison
- Optimized source code
- Final profile: {profile_log_path}

**REMINDER: OpenMP Clause Syntax**
OpenMP clauses (`is_device_ptr`, `use_device_addr`, `map`) require bare pointer variables.
Extract struct members to local variables before the pragma:
```c
double *d_arr = data.arr;  // Extract first
#pragma omp target teams loop is_device_ptr(d_arr)  // Use local var
```

## RULES - BREAKING A RULE = FAILURE
- NO GIT COMMANDS.
- DO NOT READ/WRITE OUTSIDE THE WORKING DIRECTORY.
- DO NOT EDIT MAKEFILES.
- ALWAYS CLEAN BEFORE BUILD.
- PRESERVE CORRECTNESS - diff against baseline after each change
- YOU MAY MODIFY THE MAKEFILE TO ADD ANYTHING YOU NEED TO RUN THE CODE.
'''

        return (
            f"Directory: {kernel_dir}\n\n"
            f"{body}\n\n"
            f"{comment_info}\n\n"
            f"{macros_info}\n\n"
            f"{hadware_info}\n\n"
        )
        
def _run_codex_step(kernel_dir: Path, kernel_name: str, file_name: str, target_api: str, step: int, prompt_text: str) -> Optional[dict]:
    """Run a single Codex optimization step with the provided prompt.

    Returns dict on success: { 'combined': stdout+stderr, 'summary': stdout }
    Returns None on failure.
    """
    try:
        result = subprocess.run(
            build_codex_cli_cmd() + [prompt_text],
            capture_output=True,
            text=True,
            timeout=18000,
        )
        if result.returncode == 0:
            combined = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
            return { 'combined': combined.strip(), 'summary': (result.stdout or "").strip() }
        print(f"Codex step {step} error: {result.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print(f"Codex step {step} timeout for kernel {kernel_name}")
        return None
    except Exception as e:
        print(f"Error running Codex step {step}: {e}")
        return None


def _save_step_artifacts(output_dir: Path, kernel_name: str, file_name: str, target_api: str, step: str, transcript: Optional[str], run_output: Optional[str], transcript_summary: Optional[str] = None) -> None:
    """Save transcript (full and summary) and nsys output per step to the step subdirectory.
    
    Args:
        step: Step identifier (can be int like 1, or string like "1_attempt2")
    """
    kernel_output_dir = output_dir / f"{kernel_name}-{target_api}"
    # Create step subdirectory (e.g., step1/, step2/, step1_attempt2/)
    step_dir = kernel_output_dir / f"step{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    
    if transcript is not None:
        with open(step_dir / "transcript.txt", 'w') as f:
            f.write(transcript)
        if transcript_summary is not None:
            with open(step_dir / "transcript_summary.txt", 'w') as f:
                f.write(transcript_summary)
    if run_output is not None:
        with open(step_dir / "nsys_output.txt", 'w') as f:
            f.write(run_output)


def _load_step_transcript_summary(output_dir: Path, kernel_name: str, file_name: str, step: int, target_api: str) -> Optional[str]:
    """Load the most recent transcript summary for a given step from disk."""
    if step <= 0:
        return None
    kernel_output_dir = Path(output_dir) / f"{kernel_name}-{target_api}"
    if not kernel_output_dir.exists():
        return None
    # Try to load from step subdirectory first
    step_dir = kernel_output_dir / f"step{step}"
    transcript_summary_path = step_dir / "transcript_summary.txt"
    if transcript_summary_path.exists():
        try:
            content = transcript_summary_path.read_text().strip()
            if content:
                return content
        except Exception:
            pass
    # Fallback: look for old format files in root (for backward compatibility)
    candidates = sorted(
        kernel_output_dir.glob(f"step{step}*_transcript_summary.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            content = path.read_text().strip()
            if content:
                return content
        except Exception:
            continue
    return None


def _run_step_with_retry(
    kernel_dir: Path,
    kernel_name: str,
    file_name: str | List[str],
    target_api: str,
    step: int,
    prompt_text: str,
    output_dir: Path,
    hecbench_src_dir: str,
    max_attempts: int,
) -> Tuple[bool, Optional[str], Optional[str], Optional[float], Optional[str]]:
    """
    Run a single optimization step with retry logic.
    
    Returns:
        Tuple of (success, transcript, runtime_ms, run_output)
    """
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_name, str):
        file_names = [file_name]
    else:
        file_names = file_name
    
    for attempt in range(1, max_attempts + 1):
        print(f"  Attempt {attempt}/{max_attempts} for step {step}...")
        
        # Run the codex step
        transcripts = _run_codex_step(kernel_dir, kernel_name, file_name, target_api, step, prompt_text)
        if transcripts is None:
            print(f"  ✗ Step {step} attempt {attempt} failed to produce a transcript.")
            if attempt < max_attempts:
                print(f"  Retrying step {step}...")
                continue
            else:
                return False, None, None, None, None
        
        # Test compile and run under nsys
        ok, runtime_ms, run_output = _measure_performance_with_nsys(kernel_dir, target_api, kernel_name)
        
        # Save artifacts for this attempt (including source code files)
        attempt_suffix = f"_attempt{attempt}" if attempt > 1 else ""
        step_attempt_suffix = f"{step}{attempt_suffix}"  # Just the step number and attempt (e.g., "1" or "1_attempt2")
        _save_step_artifacts(
            output_dir, kernel_name, file_name, target_api, step_attempt_suffix,
            transcripts.get('combined') if transcripts else None,
            run_output,
            transcript_summary=(transcripts.get('summary') if transcripts else None)
        )
        
        # Save source code files for this attempt (both successful and failed attempts)
        # _save_step_artifacts adds "step" prefix, but copy_translated_file uses suffix directly, so add "step" here
        copy_translated_file(
            kernel_name, file_names, target_api, hecbench_src_dir, output_dir, f"step{step_attempt_suffix}"
        )
        
        if ok:
            print(f"  ✓ Step {step} attempt {attempt} succeeded (runtime: {runtime_ms if runtime_ms is not None else 'unknown'} ms)")
            return True, transcripts.get('combined'), transcripts.get('summary'), runtime_ms, run_output
        else:
            print(f"  ✗ Step {step} attempt {attempt} failed (nsys/make run failed).")
            if attempt < max_attempts:
                print(f"  Retrying step {step}...")
                # Optionally, we could modify the prompt for retry attempts
                # For now, we'll use the same prompt
                continue
            else:
                print(f"  ✗ Step {step} failed after {max_attempts} attempts.")
                return False, transcripts.get('combined'), transcripts.get('summary'), runtime_ms, run_output
    
    return False, None, None, None, None


def optimize_translated_code_four_stage(
    kernel_name: str,
    file_name: str | List[str],
    target_api: str,
    output_dir: str,
    hecbench_src_dir: str,
    steps: Optional[List[int]] = None,
    custom_prompts: Optional[Dict[int, str]] = None,
    cyclic: bool = False,
    target_speedup: Optional[float] = None,
    target_runtime_ms: Optional[float] = None,
    max_cycles: int = 2,
    supervisor_steps: Optional[List[int]] = None,
    max_attempts: int = 3,
    supervise_max_attempts: int = 1,
    source_api: str = 'serial',
) -> Dict[str, Any]:
    """Run a multi-stage optimization pipeline (default: 2 stages) with optional cycling until target criteria.

    Args:
        kernel_name: Name of the kernel to optimize
        file_name: Name of the file(s) to optimize (can be a single string or list of strings)
        target_api: Target API ('omp' or 'cuda')
        output_dir: Directory to save results
        hecbench_src_dir: Directory containing source code
        steps: List of step numbers to run (default: [1, 2])
        custom_prompts: Custom prompts for specific steps
        cyclic: Whether to cycle through steps multiple times
        target_speedup: Target speedup ratio to achieve
        target_runtime_ms: Target runtime in milliseconds
        max_cycles: Maximum number of cycles to run
        supervisor_steps: Steps after which to run supervisor
        max_attempts: Maximum number of retry attempts per step (default: 3)

    Returns a dict with success flag, performance metrics, and artifact paths.
    """
    # Handle both single file (string) and multiple files (list) for backward compatibility
    if isinstance(file_name, str):
        file_names = [file_name]
        primary_file_name = file_name
    else:
        file_names = file_name
        primary_file_name = file_names[0] if file_names else 'unknown'
    
    file_list_str = ', '.join(file_names) if len(file_names) > 1 else file_names[0]
    print(f"Starting optimization (steps: {steps}) for {kernel_name}/{file_list_str}...")

    steps = steps or [1, 2]
    supervisor_steps = supervisor_steps or []
    custom_prompts = custom_prompts or {}

    output_dir_p = Path(output_dir)
    kernel_dir = Path(hecbench_src_dir) / f"{kernel_name}-{target_api}"
    if not kernel_dir.exists():
        return {
            'success': False,
            'error_msg': f'Kernel directory not found: {kernel_dir}',
        }

    # Copy initial state using a distinct suffix to avoid overwriting pre-supervisor snapshot
    # Copy all files for this kernel
    initial_file_path = copy_translated_file(
        kernel_name, file_names, target_api, hecbench_src_dir, output_dir, 'initial_correct'
    )
    
    # Baseline performance
    # Use kernel name and target API for directory structure
    kernel_output_dir = output_dir_p / f"{kernel_name}-{target_api}"

    best_runtime_ms = None
    best_cycle = 0
    cycle_index = 0
    all_step_outputs: Dict[int, Dict[str, Any]] = {}
    transcript_summary_map: Dict[int, str] = {}
    last_accepted_stage_runtime_ms = None
    last_accepted_stage_suffix = 'initial_correct'

    while True:
        cycle_index += 1
        print(f"\n--- Cycle {cycle_index} ---")

        last_run_output = None
        last_successful_runtime_ms = None
        for step in steps:
            print(f"Running step {step}...")
            prev_step_summary = None
            previous_step_number = step - 1
            if previous_step_number >= 1:
                prev_step_summary = transcript_summary_map.get(previous_step_number)
                if not prev_step_summary:
                    prev_step_summary = _load_step_transcript_summary(
                        output_dir_p, kernel_name, primary_file_name, previous_step_number, target_api
                    )
                    if prev_step_summary:
                        transcript_summary_map[previous_step_number] = prev_step_summary

            prompt_text = _build_step_prompt(
                target_api,
                kernel_dir,
                file_names,  # Pass list of files
                step,
                custom_prompts.get(step),
                kernel_name,
                prev_step_summary,
                source_api,
            )
            
            # Use retry logic for this step
            success, transcript, transcript_summary, runtime_ms, run_output = _run_step_with_retry(
                kernel_dir, kernel_name, file_names, target_api, step, prompt_text, 
                output_dir_p, hecbench_src_dir, max_attempts
            )

            step_dir = kernel_output_dir / f"step{step}"
            step_dir.mkdir(parents=True, exist_ok=True)
            if run_output:
                step_relevant = _extract_relevant_nsys(run_output)
                if step_relevant:
                    with open(step_dir / "nsys_relevant.txt", 'w') as f:
                        f.write(step_relevant)

            # Save source files even if rejected (so we can see what was attempted)
            # Note: Artifacts (transcript, nsys output) are already saved by _run_step_with_retry
            if success:
                copy_translated_file(kernel_name, file_names, target_api, hecbench_src_dir, output_dir, f'step{step}')

            gate_rejected = False
            gate_threshold_ms = None
            if success and runtime_ms is not None and last_accepted_stage_runtime_ms is not None:
                gate_threshold_ms = last_accepted_stage_runtime_ms * PERFORMANCE_GATE_FACTOR
                if runtime_ms >= gate_threshold_ms:
                    gate_rejected = True
                    print(f"⚠ Stage {step} rejected by performance gate "
                          f"(runtime {runtime_ms:.3f} ms >= threshold {gate_threshold_ms:.3f} ms).")
                    _record_gate_failure(
                        kernel_output_dir,
                        step,
                        runtime_ms,
                        gate_threshold_ms,
                        transcript_summary,
                        transcript,
                        run_output,
                    )
                    _restore_stage_snapshot(
                        kernel_dir,
                        output_dir_p,
                        kernel_name,
                        primary_file_name,
                        target_api,
                        file_names,
                        last_accepted_stage_suffix,
                    )
                    print(f"  ⇒ Reverted to {last_accepted_stage_suffix} snapshot for the next stage.")

            if gate_rejected:
                print(f"  Continuing to next step using previously accepted stage output.")
                continue

            if not success:
                print(f"⚠ Step {step} failed after {max_attempts} attempts (nsys/make run failed).")
                print(f"  Continuing to next step despite failure...")
                if run_output:
                    last_run_output = run_output
                _restore_stage_snapshot(
                    kernel_dir,
                    output_dir_p,
                    kernel_name,
                    primary_file_name,
                    target_api,
                    file_names,
                    last_accepted_stage_suffix,
                )
                continue

            if transcript_summary:
                transcript_summary_map[step] = transcript_summary.strip()
            else:
                loaded_summary = _load_step_transcript_summary(
                    output_dir_p, kernel_name, primary_file_name, step, target_api
                )
                if loaded_summary:
                    transcript_summary_map[step] = loaded_summary

            last_run_output = run_output
            if runtime_ms is not None:
                last_successful_runtime_ms = runtime_ms
                last_accepted_stage_runtime_ms = runtime_ms
            last_accepted_stage_suffix = f"step{step}"

            print(f"Step {step} runtime: {runtime_ms if runtime_ms is not None else 'unknown'} ms")

            # Optional supervisor after this step
            if step in supervisor_steps:
                print(f"Running supervisor after step {step}...")
                sup_ok = False
                sup_out = ''
                for sup_attempt in range(1, supervise_max_attempts + 1):
                    print(f"  Supervisor attempt {sup_attempt}/{supervise_max_attempts}...")
                    sup_ok, sup_out = _run_supervisor(
                        kernel_name, target_api, hecbench_src_dir, output_dir, file_names
                    )
                    if sup_ok:
                        break
                kernel_output_dir = output_dir_p / f"{kernel_name}-{target_api}"
                # Save supervisor artifacts in step{step}_supervised/ subdirectory
                supervised_dir = kernel_output_dir / f"step{step}_supervised"
                supervised_dir.mkdir(parents=True, exist_ok=True)
                with open(supervised_dir / "supervised_output.txt", 'w') as f:
                    f.write(sup_out)

                # Copy supervisor transcript and correctness output into step-scoped names
                try:
                    sup_trans_src = kernel_output_dir / "supervisor_transcript.txt"
                    if sup_trans_src.exists():
                        (supervised_dir / "supervised_transcript.txt").write_text(sup_trans_src.read_text())
                    sup_result_src = kernel_output_dir / "supervisor_result.txt"
                    if sup_result_src.exists():
                        (supervised_dir / "supervised_compilation.txt").write_text(sup_result_src.read_text())
                except Exception:
                    pass

                # Save the supervised file with requested naming: main_step{step}_supervised.cpp
                if sup_ok:
                    supervised_file_copy = copy_translated_file(
                        kernel_name, file_names, target_api, hecbench_src_dir, output_dir, f'step{step}_supervised'
                    )
                    if supervised_file_copy:
                        print(f"✓ Step {step} - Supervised version saved: {supervised_file_copy}")

                # Run NSYS after supervisor and save in step{step}_supervised/ subdirectory
                post_sup_ok, post_sup_ms, post_sup_out = _measure_performance_with_nsys(kernel_dir, target_api, kernel_name)
                try:
                    (supervised_dir / "supervised_nsys_output.txt").write_text(post_sup_out or "")
                    rel_sup = _extract_relevant_nsys(post_sup_out or "")
                    (supervised_dir / "supervised_nsys_relevant.txt").write_text(rel_sup)
                except Exception:
                    pass
                if sup_ok:
                    print("Supervisor PASS")
                    # Run compare_and_optimize_steps after successful supervisor
                    _run_compare_and_optimize_steps(kernel_name, primary_file_name, str(output_dir), target_api)

        # Evaluate post-cycle performance
        post_ok, post_ms, post_out = True, None, last_run_output
        if post_out is None or post_ms is None:
            # Measure explicitly if we didn't get a runtime
            post_ok, post_ms, post_out = _measure_performance_with_nsys(kernel_dir, target_api, kernel_name)
        if not post_ok:
            print("⚠ Post-cycle nsys run failed. Continuing with available data...")
            # Use previous successful runtime_ms if available, otherwise keep None
            if post_ms is None:
                post_ms = last_successful_runtime_ms  # Use the last successful step's runtime if available

        print(f"Cycle {cycle_index} runtime: {post_ms if post_ms is not None else 'unknown'} ms")

        # Track best
        current_ms = post_ms if post_ms is not None else best_runtime_ms
        if current_ms is not None and (best_runtime_ms is None or current_ms < best_runtime_ms):
            best_runtime_ms = current_ms
            best_cycle = cycle_index

        # Check termination criteria
        meets_runtime = False
        if target_runtime_ms is not None and current_ms is not None:
            meets_runtime = current_ms <= target_runtime_ms
            print(f"Runtime target: {current_ms:.3f} ms (target <= {target_runtime_ms} ms)")

        # Stop if: (no targets specified) OR (runtime target is specified and met)
        # Note: speedup target cannot be checked without baseline_ms, so it's ignored
        if target_speedup is not None and target_runtime_ms is None:
            # Speedup-only target: cannot check without baseline, so continue until max cycles
            print("Warning: target_speedup specified but cannot be checked without baseline measurement. Continuing until max cycles.")
        elif (target_speedup is None and target_runtime_ms is None) or (target_runtime_ms is not None and meets_runtime):
            print("Target achieved or no target specified. Stopping.")
            break

        if not cyclic or cycle_index >= max_cycles:
            print("Cyclic disabled or max cycles reached. Stopping.")
            break

    # Final copies and summary
    optimized_file_path = copy_translated_file(
        kernel_name, file_names, target_api, hecbench_src_dir, output_dir, 'optimized'
    )
    
    summary = {
        'success': True,
        'error_msg': '',
        'best_runtime_ms': best_runtime_ms,
        'best_cycle': best_cycle,
        'optimized_file_copy': optimized_file_path,
        'initial_file_copy': initial_file_path,
        'cycles': cycle_index,
    }

    # Use kernel name and target API for directory structure
    kernel_output_dir = output_dir_p / f"{kernel_name}-{target_api}"
    kernel_output_dir.mkdir(parents=True, exist_ok=True)
    with open(kernel_output_dir / '4_stages_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)

    print(f"Optimization complete. Best runtime: {best_runtime_ms if best_runtime_ms is not None else 'unknown'} ms")
    return summary


if __name__ == "__main__":
    """
    Example usage of the optimization script with retry mechanism.
    """
    import sys
    
    if len(sys.argv) < 6:
        print("Usage: python optimize_codex.py <kernel_name> <file_name> <target_api> <output_dir> <hecbench_src_dir> [max_attempts]")
        print("Example: python optimize_codex.py matrix-rotate main.cpp omp ./results ./src 5")
        sys.exit(1)
    
    kernel_name = sys.argv[1]
    file_name = sys.argv[2]
    target_api = sys.argv[3]
    output_dir = sys.argv[4]
    hecbench_src_dir = sys.argv[5]
    max_attempts = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    
    print(f"Running optimization with retry mechanism:")
    print(f"  Kernel: {kernel_name}")
    print(f"  File: {file_name}")
    print(f"  Target API: {target_api}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Source Dir: {hecbench_src_dir}")
    print(f"  Max Attempts: {max_attempts}")
    
    result = optimize_translated_code_four_stage(
        kernel_name=kernel_name,
        file_name=file_name,
        target_api=target_api,
        output_dir=output_dir,
        hecbench_src_dir=hecbench_src_dir,
        max_attempts=max_attempts
    )
    
    if result['success']:
        print(f"✓ Optimization completed successfully!")
        print(f"  Best runtime: {result.get('best_runtime_ms', 'unknown')} ms")
        print(f"  Cycles completed: {result.get('cycles', 0)}")
    else:
        print(f"✗ Optimization failed: {result.get('error_msg', 'Unknown error')}")
        sys.exit(1)

