#!/usr/bin/env python3
"""
Script to optimize translated code using Codex CLI after successful compilation.

Refactored to use centralized logging, configuration, and shared utilities.
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

# Add pipeline_refactored directory to path to allow imports when running as script
_script_dir = Path(__file__).parent
_pipeline_refactored_dir = _script_dir.parent
if str(_pipeline_refactored_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_refactored_dir))

from utils.path_config import (
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
)
from utils.prompt_loader import build_skill_trigger_prompt, normalize_api_name
from utils.logger import get_logger
from utils.config import get_config
from agents.common import (
    copy_translated_file,
    normalize_file_list,
    run_compare_and_optimize_steps,
    launch_makefile_fix_recovery,
    test_compilation,
    run_codex_command,
)

logger = get_logger(__name__)
config = get_config()

# Configuration constants (can be moved to config if needed)
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
    if target_api == 'omp':
        omp_flags = "FORCE_OMP_GPU=1 OMP_TARGET_OFFLOAD=MANDATORY "
    else:
        omp_flags = ""
    return (
        f"{omp_flags}nsys profile --stats=true --trace=cuda,osrt "
        f"--force-overwrite=true -o nsys_profile {run_str}"
    )


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
        logger.warning(f"Performance gate log written to {report_path}")
    except Exception as e:
        logger.warning(f"Failed to write performance gate report: {e}")


def _restore_stage_snapshot(
    kernel_dir: Path,
    output_dir: Path,
    kernel_name: str,
    primary_file_name: str,
    target_api: str,
    file_names: List[str],
    stage_suffix: str,
) -> None:
    """Restore source files from a previous snapshot when a stage is rejected.
    
    Restores ALL source files found in the snapshot directory, not just the ones
    in file_names (which may have outdated names from the original input).
    """
    snapshot_dir = output_dir / f"{kernel_name}-{target_api}" / stage_suffix
    if not snapshot_dir.exists():
        logger.warning(f"Snapshot {snapshot_dir} not found for restoration.")
        return
    
    # Find all code files in the snapshot directory (only actual code, not headers)
    code_extensions = {'.c', '.cpp', '.cu', '.cl'}
    restored_count = 0
    
    for src_path in snapshot_dir.iterdir():
        if src_path.is_file() and src_path.suffix in code_extensions:
            # Skip backup files
            if src_path.name.startswith('.') or '.bak' in src_path.name:
                continue
            
            dest_path = kernel_dir / src_path.name
            try:
                shutil.copy2(src_path, dest_path)
                logger.info(f"→ Restored {src_path.name} from snapshot {stage_suffix}")
                restored_count += 1
            except Exception as e:
                logger.warning(f"Failed to restore {src_path.name}: {e}")
    
    if restored_count == 0:
        logger.warning(f"No source files found in snapshot {snapshot_dir}")


def test_optimized_compilation(kernel_dir, target_api, kernel_name: Optional[str] = None, gpu_timeout: int = None):
    """Test compilation of optimized code with GPU timeout protection.
    
    Args:
        kernel_dir: Directory containing the kernel code
        target_api: Target API (omp, cuda, etc.)
        kernel_name: Optional kernel name for profiling
        gpu_timeout: Timeout in seconds for GPU execution (default: GPU_TIMEOUT_SECONDS env var or config default)
    """
    
    kernel_dir = Path(kernel_dir)
    clean_cmd = get_make_cmd(target_api, 'clean')
    timeout = gpu_timeout or config.gpu_timeout_seconds

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
                logger.info("Optimized code compilation failed.")
                return False, compile_result.stderr

        # Set environment variables for OpenMP GPU offloading ONLY if OpenMP
        if target_api == 'omp':
            if 'CC' not in env:
                env['CC'] = 'nvc++'
            env["FORCE_OMP_GPU"] = "1"
            env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
        
        # Use safe GPU execution with timeout (CLASS=C)
        nsys_cmd = _build_nsys_profile_cmd(target_api, kernel_name)
        logger.info(f"Running GPU command with {timeout}s timeout: {' '.join(nsys_cmd)}")
        
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
            logger.info(f"CLASS=C timed out, retrying with CLASS=B...")
            # Run kill GPU script after CLASS C failure
            _run_kill_gpu_script()
            # Get fallback command (CLASS=B) - this is a shell command string
            fallback_cmd_str = get_nsys_profile_fallback_cmd_str(target_api, kernel_name, "B")
            logger.info(f"Running fallback GPU command with {timeout}s timeout: {fallback_cmd_str}")
            
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
                logger.info(f"Fallback CLASS=B also timed out after {timeout}s")
                _cleanup_gpu_after_failure(pre_gpu_pids)
                # Run kill GPU script after CLASS B failure
                _run_kill_gpu_script()
                success = False
                stdout = ""
                stderr = f"Both CLASS=C and CLASS=B timed out after {timeout}s"
            except Exception as e:
                logger.info(f"Error running fallback command: {e}")
                success = False
                stdout = ""
                stderr = str(e)
            
            _cleanup_nsys_artifacts(kernel_dir)
            if success:
                logger.info("CLASS=B run succeeded!")

        if stdout and stdout.strip():
            output = stdout.strip()
        elif stderr and stderr.strip():
            output = stderr.strip()
        else:
            output = ""

        if not success:
            logger.info("Optimized code run failed.")
            if stderr:
                print(stderr[:500])  # Print first 500 chars of error
            # Clean up any hung GPU processes from our run
            _cleanup_gpu_after_failure(pre_gpu_pids)

        return success, output

    except subprocess.TimeoutExpired:
        logger.info(f"Optimized code execution timeout after {timeout}s")
        _cleanup_gpu_after_failure(pre_gpu_pids)
        return False, f"Optimized code execution timeout after {timeout}s"
    except Exception as e:
        logger.info(f"Error during optimized code execution: {e}")
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
        logger.info(f"Cleaning up hung GPU processes: {new_pids}")
        for pid in new_pids:
            try:
                os.kill(pid, 9)  # SIGKILL
                logger.info(f"  Killed PID {pid}")
            except Exception as e:
                logger.info(f"  Failed to kill PID {pid}: {e}")


def _run_kill_gpu_script():
    """Run the kill GPU processes script to clean up all WSL GPU processes."""
    script_path = config.pipeline_root / "kill_gpu_processes.py"
    if script_path.exists():
        try:
            logger.info("Running kill_gpu_processes.py to clean up all WSL GPU processes...")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.stdout:
                logger.debug(result.stdout)
            if result.stderr and result.returncode != 0:
                logger.info(f"Warning: kill_gpu_processes.py had errors: {result.stderr}")
        except Exception as e:
            logger.info(f"Warning: Failed to run kill_gpu_processes.py: {e}")
    else:
        logger.info(f"Warning: kill_gpu_processes.py not found at {script_path}")


def _run_supervisor(
    kernel_name: str, target_api: str, output_dir: str, file_name: Optional[str] | List[str] = None,
    step: Optional[int] = None, attempt: Optional[int] = None
) -> Tuple[bool, str]:
    """Invoke the supervisor agent for a specific kernel. Returns (success, output).

    Kept local to avoid circular imports with initial_translation_codex.

    Args:
        file_name: Can be a single file name (str), list of file names, or None
        step: Optimization step number (1, 2, etc.) for proper naming of output directories
        attempt: Attempt number within this supervisor call (1-based), used to name output directories
    """
    try:
        supervisor_path = config.pipeline_root / "agents" / "supervisor_codex.py"
        cmd = [
            'python3', str(supervisor_path),
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
        if step is not None:
            cmd.extend(['--step', str(step)])
        if attempt is not None:
            cmd.extend(['--attempt', str(attempt)])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        success = result.returncode == 0
        output = (result.stdout or '') + '\n' + (result.stderr or '')
        return success, output
    except subprocess.TimeoutExpired:
        return False, 'Supervisor timed out'
    except Exception as e:
        return False, f'Supervisor error: {e}'


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


def _sanitize_nsys_output(output: str) -> str:
    """Remove progress bars and junk from nsys output to save tokens."""
    if not output:
        return ""
    
    lines = output.splitlines()
    cleaned_lines = []
    for line in lines:
        # Filter progress bars like [1/7] [==30% ... ]
        if re.search(r"\[\d+/\d+\]\s+\[=*\d*%.*\]", line):
            continue
        # Filter "Collecting data..." type patterns if they are just noise
        if "Collecting data..." in line or "Generating '/tmp/nsys" in line:
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)


def _measure_performance_with_nsys(kernel_dir: Path, target_api: str, kernel_name: Optional[str], stage: str = "unknown") -> Tuple[bool, Optional[float], str]:
    """Run the nsys + make run command and return (success, runtime_ms, raw_output).
    
    Args:
        kernel_dir: Directory containing the kernel
        target_api: Target API (omp, cuda, ocl)
        kernel_name: Optional kernel name for profiling
        stage: Stage name (e.g., "step1", "step2") for recovery logging
    """
    success, output = test_optimized_compilation(kernel_dir, target_api, kernel_name=kernel_name)
    
    # Check if failure is due to compilation error (not runtime error)
    # Compilation errors typically contain: "error:", "undefined reference", "cannot find", "make:", etc.
    is_compilation_error = False
    if not success and output:
        compilation_indicators = [
            "error:",
            "undefined reference",
            "cannot find",
            "make:",
            "compilation terminated",
            "no such file",
            "multiple definition",
            "redefinition",
            "expected",
            "syntax error",
        ]
        output_lower = output.lower()
        is_compilation_error = any(indicator in output_lower for indicator in compilation_indicators)
    
    # If it's a compilation error, try recovery
    if not success and is_compilation_error:
        logger.warning(f"Compilation failure detected at {stage} stage. Attempting Makefile fix recovery...")
        recovery_success, recovery_error = launch_makefile_fix_recovery(
            kernel_dir=kernel_dir,
            target_api=target_api,
            compilation_error=output,
            stage=stage,
            max_attempts=2,
        )
        
        if recovery_success:
            logger.success(f"Recovery succeeded - retrying compilation after Makefile fix...")
            # Retry compilation after recovery
            success, output = test_optimized_compilation(kernel_dir, target_api, kernel_name=kernel_name)
            if success:
                logger.success(f"Compilation succeeded after recovery!")
            else:
                logger.warning(f"Compilation still fails after recovery: {output[:200]}")
        else:
            logger.warning(f"Recovery failed: {recovery_error}")
    
    runtime_ms = _parse_nsys_runtime_ms(output) if success else None

        
    # Save nsys output to profile.log for the model to read
    profile_log_path = kernel_dir / "profile.log"
    try:
        sanitized_output = _sanitize_nsys_output(output)
        with open(profile_log_path, 'w') as f:
            f.write(sanitized_output)
    except Exception as e:
        logger.info(f"Warning: Could not write profile.log: {e}")
    

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
    file_names = normalize_file_list(file_name)
    
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

    # Dynamic command generation
    build_cmd_str = get_make_cmd_str(target_api, "build")
    clean_cmd_str = get_make_cmd_str(target_api, "clean")

    # Correctness testing: CLASS=S (for initial testing)
    correctness_run_cmd = get_correctness_run_cmd_str(target_api, "S", "S")
    correctness_fallback_cmd = get_correctness_fallback_cmd_str(target_api, "S")

    # Profiling: CLASS=C (fallback B)
    profile_run_cmd = get_profile_run_cmd_str(target_api, kernel_name, "C", "B")
    profile_fallback_cmd = get_profile_fallback_cmd_str(target_api, kernel_name, "B")
    nsys_profile_cmd = get_nsys_profile_cmd_str(target_api, kernel_name, use_class="C")
    nsys_profile_fallback_cmd = get_nsys_profile_fallback_cmd_str(target_api, kernel_name, "B")

    workdir = get_codex_workdir()
    skill_name = f"paracodex-{normalize_api_name(source_api)}-{normalize_api_name(target_api)}-step{step}"

    variables = {
        "kernel_dir": str(kernel_dir),
        "file_listing": file_listing,
        "profile_log_path": str(profile_log_path),
        "clean_cmd_str": clean_cmd_str,
        "build_cmd_str": build_cmd_str,
        "correctness_run_cmd": correctness_run_cmd,
        "correctness_fallback_cmd": correctness_fallback_cmd,
        "profile_run_cmd": profile_run_cmd,
        "profile_fallback_cmd": profile_fallback_cmd,
        "nsys_profile_cmd": nsys_profile_cmd,
        "nsys_profile_fallback_cmd": nsys_profile_fallback_cmd,
        "kernel_name": kernel_name,
        "cwd": str(workdir),
    }

    body = build_skill_trigger_prompt(
        skill_name=skill_name,
        task=f"{source_api} -> {target_api} optimization step {step} for kernel {kernel_name}.",
        workdir=workdir,
        source_dir=None,
        target_dir=kernel_dir,
        file_listing=file_listing,
        variables=variables,
        notes=[
            f"Use CODEX_WORKDIR={workdir}.",
            f"Read `{workdir}/system_info_summary.txt` to understand the target system hardware configuration before starting.",
            "For shell commands: Prefer redirecting large output to a temporary file, then read that file using the `read_file` tool (BEST) or `cat` WITHOUT redirection.",
            "Do not run git commands.",
            "Use Variables to resolve placeholders in the skill instructions.",
        ],
    )

    return context_header + nsys_instructions + body

def _run_codex_step(kernel_dir: Path, kernel_name: str, file_name: str, target_api: str, step: int, prompt_text: str, model: Optional[str] = None) -> Optional[dict]:
    """Run a single Codex optimization step with the provided prompt.

    Returns dict on success: { 'combined': stdout+stderr, 'summary': stdout }
    Returns None on failure.
    """
    try:
        result = run_codex_command(prompt_text, timeout=18000, model=model)
        if result:
            return result
        logger.info(f"Codex step {step} failed (SDK error/timeout)")
        return None
    except Exception as e:
        logger.info(f"Error running Codex step {step}: {e}")
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
    data_src_dir: str,
    max_attempts: int,
    model: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str], Optional[float], Optional[str]]:
    """
    Run a single optimization step with retry logic.
    
    Returns:
        Tuple of (success, transcript, runtime_ms, run_output)
    """
    file_names = normalize_file_list(file_name)
    
    for attempt in range(1, max_attempts + 1):
        logger.info(f"  Attempt {attempt}/{max_attempts} for step {step}...")
        
        # Run the codex step
        transcripts = _run_codex_step(kernel_dir, kernel_name, file_name, target_api, step, prompt_text, model=model)
        if transcripts is None:
            logger.info(f"  ✗ Step {step} attempt {attempt} failed to produce a transcript.")
            if attempt < max_attempts:
                logger.info(f"  Retrying step {step}...")
                continue
            else:
                return False, None, None, None, None
        
        # Test compile and run under nsys
        ok, runtime_ms, run_output = _measure_performance_with_nsys(kernel_dir, target_api, kernel_name, stage=f"step{step}")
        
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
            kernel_name, file_names, target_api, data_src_dir, output_dir, f"step{step_attempt_suffix}"
        )

        # Copy profile.log from kernel_dir into the attempt-specific step directory so it
        # is not overwritten when a subsequent attempt runs _measure_performance_with_nsys.
        src_profile = kernel_dir / "profile.log"
        if src_profile.exists():
            dest_step_dir = output_dir / f"{kernel_name}-{target_api}" / f"step{step_attempt_suffix}"
            dest_step_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_profile, dest_step_dir / "profile.log")
        
        if ok:
            logger.info(f"  ✓ Step {step} attempt {attempt} succeeded (runtime: {runtime_ms if runtime_ms is not None else 'unknown'} ms)")
            return True, transcripts.get('combined'), transcripts.get('summary'), runtime_ms, run_output
        else:
            logger.info(f"  ✗ Step {step} attempt {attempt} failed (nsys/make run failed).")
            if attempt < max_attempts:
                logger.info(f"  Retrying step {step}...")
                # Optionally, we could modify the prompt for retry attempts
                # For now, we'll use the same prompt
                continue
            else:
                logger.info(f"  ✗ Step {step} failed after {max_attempts} attempts.")
                return False, transcripts.get('combined'), transcripts.get('summary'), runtime_ms, run_output
    
    return False, None, None, None, None


def optimize_translated_code_two_stage(
    kernel_name: str,
    file_name: str | List[str],
    target_api: str,
    output_dir: str,
    data_src_dir: str,
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
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a multi-stage optimization pipeline (default: 2 stages) with optional cycling until target criteria.

    Args:
        kernel_name: Name of the kernel to optimize
        file_name: Name of the file(s) to optimize (can be a single string or list of strings)
        target_api: Target API ('omp' or 'cuda')
        output_dir: Directory to save results
        data_src_dir: Directory containing translated code (data/src)
        steps: List of step numbers to run (default: [1, 2])
        custom_prompts: Custom prompts for specific steps
        cyclic: Whether to cycle through steps multiple times
        target_speedup: Target speedup ratio to achieve
        target_runtime_ms: Target runtime in milliseconds
        max_cycles: Maximum number of cycles to run
        supervisor_steps: Steps after which to run supervisor
        max_attempts: Maximum number of retry attempts per step (default: 3)
        model: Codex model to use

    Returns a dict with success flag, performance metrics, and artifact paths.
    """
    file_names = normalize_file_list(file_name)
    primary_file_name = file_names[0] if file_names else 'unknown'
    
    file_list_str = ', '.join(file_names) if file_names else 'unknown'
    logger.info(f"Starting optimization (steps: {steps}) for {kernel_name}/{file_list_str}...")

    steps = steps or [1, 2]
    supervisor_steps = supervisor_steps or []
    custom_prompts = custom_prompts or {}

    output_dir_p = Path(output_dir)
    kernel_dir = Path(data_src_dir) / f"{kernel_name}-{target_api}"
    if not kernel_dir.exists():
        return {
            'success': False,
            'error_msg': f'Kernel directory not found: {kernel_dir}',
        }

    # Copy initial state using a distinct suffix to avoid overwriting pre-supervisor snapshot
    # Copy all files for this kernel
    initial_file_path = copy_translated_file(
        kernel_name, file_names, target_api, data_src_dir, output_dir, 'initial_correct'
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
    last_supervised_step = None  # Track the last step where supervisor passed

    while True:
        cycle_index += 1
        logger.info(f"\n--- Cycle {cycle_index} ---")

        last_run_output = None
        last_successful_runtime_ms = None
        for step in steps:
            logger.info(f"Running step {step}...")
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
                output_dir_p, data_src_dir, max_attempts, model=model
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
                copy_translated_file(kernel_name, file_names, target_api, data_src_dir, output_dir, f'step{step}')

            gate_rejected = False
            gate_threshold_ms = None
            if success and runtime_ms is not None and last_accepted_stage_runtime_ms is not None:
                gate_threshold_ms = last_accepted_stage_runtime_ms * PERFORMANCE_GATE_FACTOR
                if runtime_ms >= gate_threshold_ms:
                    gate_rejected = True
                    logger.warning(f"Stage {step} rejected by performance gate "
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
                    logger.info(f"  ⇒ Reverted to {last_accepted_stage_suffix} snapshot for the next stage.")

            # Optional supervisor after this step (run even if gate rejected or step failed)
            if step in supervisor_steps:
                logger.info(f"Running supervisor after step {step}...")
                sup_ok = False
                sup_out = ''
                for sup_attempt in range(1, supervise_max_attempts + 1):
                    logger.info(f"  Supervisor attempt {sup_attempt}/{supervise_max_attempts}...")
                    sup_ok, sup_out = _run_supervisor(
                        kernel_name, target_api, output_dir, file_names, step=step, attempt=sup_attempt
                    )
                    if sup_ok:
                        break
                    else:
                        logger.warning(f"  Supervisor attempt {sup_attempt} failed. Output:\n{sup_out.strip()}")

                # Supervisor already saved all artifacts to step{step}_supervised/ directory
                kernel_output_dir = output_dir_p / f"{kernel_name}-{target_api}"
                supervised_dir = kernel_output_dir / f"step{step}_supervised"

                # Just log the result
                if sup_ok:
                    logger.success(f"Step {step} - Supervisor correctness check passed")
                    last_supervised_step = step  # Track the last successful supervisor run
                else:
                    logger.failure(f"Step {step} - Supervisor correctness check failed (all {supervise_max_attempts} attempts)")

                # Run NSYS after supervisor and save in step{step}_supervised/ subdirectory
                post_sup_ok, post_sup_ms, post_sup_out = _measure_performance_with_nsys(kernel_dir, target_api, kernel_name, stage=f"step{step}_supervised")
                try:
                    supervised_dir.mkdir(parents=True, exist_ok=True)
                    (supervised_dir / "nsys_output.txt").write_text(post_sup_out or "")
                    rel_sup = _extract_relevant_nsys(post_sup_out or "")
                    (supervised_dir / "nsys_relevant.txt").write_text(rel_sup)
                except Exception:
                    pass
                if sup_ok:
                    logger.info("Supervisor PASS")
                    # Run compare_and_optimize_steps after successful supervisor
                    run_compare_and_optimize_steps(kernel_name, primary_file_name, str(output_dir), target_api)

            if gate_rejected:
                logger.info(f"  Continuing to next step using previously accepted stage output.")
                continue

            if not success:
                logger.warning(f"Step {step} failed after {max_attempts} attempts (nsys/make run failed).")
                logger.info(f"  Continuing to next step despite failure...")
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

            logger.info(f"Step {step} runtime: {runtime_ms if runtime_ms is not None else 'unknown'} ms")

        # Evaluate post-cycle performance
        post_ok, post_ms, post_out = True, None, last_run_output
        if post_out is None or post_ms is None:
            # Measure explicitly if we didn't get a runtime
            post_ok, post_ms, post_out = _measure_performance_with_nsys(kernel_dir, target_api, kernel_name, stage=f"cycle{cycle_index}")
        if not post_ok:
            logger.info("⚠ Post-cycle nsys run failed. Continuing with available data...")
            # Use previous successful runtime_ms if available, otherwise keep None
            if post_ms is None:
                post_ms = last_successful_runtime_ms  # Use the last successful step's runtime if available

        logger.info(f"Cycle {cycle_index} runtime: {post_ms if post_ms is not None else 'unknown'} ms")

        # Track best
        current_ms = post_ms if post_ms is not None else best_runtime_ms
        if current_ms is not None and (best_runtime_ms is None or current_ms < best_runtime_ms):
            best_runtime_ms = current_ms
            best_cycle = cycle_index

        # Check termination criteria
        meets_runtime = False
        if target_runtime_ms is not None and current_ms is not None:
            meets_runtime = current_ms <= target_runtime_ms
            logger.info(f"Runtime target: {current_ms:.3f} ms (target <= {target_runtime_ms} ms)")

        # Stop if: (no targets specified) OR (runtime target is specified and met)
        # Note: speedup target cannot be checked without baseline_ms, so it's ignored
        if target_speedup is not None and target_runtime_ms is None:
            # Speedup-only target: cannot check without baseline, so continue until max cycles
            logger.info("Warning: target_speedup specified but cannot be checked without baseline measurement. Continuing until max cycles.")
        elif (target_speedup is None and target_runtime_ms is None) or (target_runtime_ms is not None and meets_runtime):
            logger.info("Target achieved or no target specified. Stopping.")
            break

        if not cyclic or cycle_index >= max_cycles:
            logger.info("Cyclic disabled or max cycles reached. Stopping.")
            break

    # Final copies and summary
    # If supervisor ran and passed after any step, use the supervised code as final optimized code
    # Otherwise, use whatever is currently in the kernel directory (could be step2 or reverted code)
    if last_supervised_step is not None:
        logger.info(f"Using supervised code from step{last_supervised_step}_supervised as final optimized code")
        supervised_source_dir = output_dir_p / f"{kernel_name}-{target_api}" / f"step{last_supervised_step}_supervised"
        optimized_dest_dir = output_dir_p / f"{kernel_name}-{target_api}" / "optimized"
        try:
            if supervised_source_dir.exists():
                if optimized_dest_dir.exists():
                    shutil.rmtree(optimized_dest_dir)
                shutil.copytree(supervised_source_dir, optimized_dest_dir,
                                ignore=shutil.ignore_patterns("*.o", "main", ".*", "*.bak"))
                logger.success(f"Snapshotted optimized directory from supervised: {supervised_source_dir} -> {optimized_dest_dir}")
                optimized_file_path = str(optimized_dest_dir)
            else:
                logger.warning(f"Supervised source dir not found: {supervised_source_dir}, falling back to kernel dir")
                optimized_file_path = copy_translated_file(
                    kernel_name, file_names, target_api, data_src_dir, output_dir, 'optimized'
                )
        except Exception as e:
            logger.warning(f"Error copying supervised dir to optimized: {e}")
            optimized_file_path = None
    else:
        # No supervisor run, use current kernel directory code
        optimized_file_path = copy_translated_file(
            kernel_name, file_names, target_api, data_src_dir, output_dir, 'optimized'
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

    logger.info(f"Optimization complete. Best runtime: {best_runtime_ms if best_runtime_ms is not None else 'unknown'} ms")
    return summary


if __name__ == "__main__":
    """
    Example usage of the optimization script with retry mechanism.
    """
    import sys
    
    if len(sys.argv) < 6:
        logger.info("Usage: python optimize_codex.py <kernel_name> <file_name> <target_api> <output_dir> <data_src_dir> [max_attempts]")
        logger.info("Example: python optimize_codex.py matrix-rotate main.cpp omp ./results ./src 5")
        sys.exit(1)
    
    kernel_name = sys.argv[1]
    file_name = sys.argv[2]
    target_api = sys.argv[3]
    output_dir = sys.argv[4]
    data_src_dir = sys.argv[5]
    max_attempts = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    
    logger.info(f"Running optimization with retry mechanism:")
    logger.info(f"  Kernel: {kernel_name}")
    logger.info(f"  File: {file_name}")
    logger.info(f"  Target API: {target_api}")
    logger.info(f"  Output Dir: {output_dir}")
    logger.info(f"  Source Dir: {data_src_dir}")
    logger.info(f"  Max Attempts: {max_attempts}")
    
    result = optimize_translated_code_two_stage(
        kernel_name=kernel_name,
        file_name=file_name,
        target_api=target_api,
        output_dir=output_dir,
        data_src_dir=data_src_dir,
        max_attempts=max_attempts
    )
    
    if result['success']:
        logger.success(f"Optimization completed successfully!")
        logger.info(f"  Best runtime: {result.get('best_runtime_ms', 'unknown')} ms")
        logger.info(f"  Cycles completed: {result.get('cycles', 0)}")
    else:
        logger.failure(f"Optimization failed: {result.get('error_msg', 'Unknown error')}")
        sys.exit(1)
