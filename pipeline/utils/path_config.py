#!/usr/bin/env python3
"""
Shared helpers for configuring Codex pipeline paths.

Refactored to use centralized configuration and logging.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import get_config
from .logger import get_logger

logger = get_logger(__name__)


def _expand_path(path: str | os.PathLike[str]) -> Path:
    """Expand and resolve a path."""
    return Path(path).expanduser().resolve()


def get_codex_workdir() -> Path:
    """Return the current Codex working directory."""
    return get_config().codex_workdir


def set_codex_workdir(path: str | os.PathLike[str]) -> Path:
    """Persist a new Codex workdir override in the current environment."""
    config = get_config()
    resolved = _expand_path(path)
    config.codex_workdir = resolved
    logger.info(f"Set CODEX_WORKDIR to: {resolved}")
    return resolved


def get_gate_sdk_dir() -> Path:
    """Return the absolute gate SDK path (overridable via GATE_SDK_DIR)."""
    return get_config().gate_sdk_dir


def default_data_src() -> Path:
    """Default data source directory where translated code goes (data/src)."""
    return get_config().data_src()


def default_golden_root() -> Path:
    """Default golden labels root under the current workdir."""
    return get_config().golden_root()


def default_jsonl(filename: Optional[str] = None) -> Path:
    """Default JSONL helper—useful for supervisor CLI defaults."""
    return get_config().default_jsonl(filename)


def _override_class_arg(cmd: List[str], new_class: str) -> List[str]:
    result: List[str] = []
    replaced = False
    for part in cmd:
        if part.startswith("CLASS="):
            result.append(f"CLASS={new_class}")
            replaced = True
        else:
            result.append(part)
    if not replaced:
        result.insert(1, f"CLASS={new_class}")
    return result


def _is_nas_workdir(workdir: Path) -> bool:
    """Check if workdir is a NAS workdir."""
    config = get_config()
    return config.nas_identifier in str(workdir)


def _nas_make_base() -> List[str]:
    """Get base NAS make command."""
    config = get_config()
    return ["make", f"CC={config.nas_cc}", f"CLASS={config.nas_class}"]


def _default_make_cmd(target_api: str, goal: str) -> List[str]:
    """
    Return the default make command (used for DEFAULT_CODEX_WORKDIR and all non-NAS workdirs).
    
    goal ∈ {"clean", "build", "check", "run"}.
    """
    makefile_name = "Makefile.nvc"
    if target_api == 'hip':
        makefile_name = "Makefile.hip"
    elif target_api == 'sycl':
        makefile_name = "Makefile.sycl"

    if goal == "clean":
        return ["make", "-f", makefile_name, "clean"]
    if goal == "build":
        return ["make", "-f", makefile_name]
    if goal == "check":
        return ["make", "-f", makefile_name, "check-correctness"]
    if goal == "run":
        return ["make", "-f", makefile_name, "run"]
    raise ValueError(f"Unsupported goal: {goal}")


def get_make_cmd(target_api: str, goal: str) -> List[str]:
    """
    Return the appropriate make command for the current workdir/goal.
    
    For NAS workdirs, returns NAS-specific commands.
    For all other workdirs (including DEFAULT_CODEX_WORKDIR), returns the default commands.

    goal ∈ {"clean", "build", "check", "run"}.
    """
    workdir = get_codex_workdir()
    if _is_nas_workdir(workdir):
        if goal == "clean":
            return ["make", "clean"]
        cmd = list(_nas_make_base())
        if goal == "check":
            cmd.append("check-correctness")
        elif goal == "run":
            cmd.append("run")
        elif goal == "build":
            pass
        else:
            raise ValueError(f"Unsupported goal: {goal}")
        return cmd

    # All non-NAS workdirs use the same commands as DEFAULT_CODEX_WORKDIR
    return _default_make_cmd(target_api, goal)


def get_make_cmd_str(target_api: str, goal: str) -> str:
    return " ".join(get_make_cmd(target_api, goal))


def get_nsys_run_make_cmd(target_api: str, kernel_name: Optional[str]) -> List[str]:
    """
    Return the nsys run make command for the current workdir.
    
    For NAS workdirs, may override CLASS argument (except for bt kernels).
    For all other workdirs, returns the default run command.
    """
    cmd = list(get_make_cmd(target_api, "run"))
    workdir = get_codex_workdir()
    if not _is_nas_workdir(workdir):
        return cmd
    return _override_class_arg(cmd, "C")


def get_nsys_run_make_cmd_str(target_api: str, kernel_name: Optional[str]) -> str:
    return " ".join(get_nsys_run_make_cmd(target_api, kernel_name))


# =============================================================================
# CLASS-specific Command Generators (for prompts)
# =============================================================================

def get_correctness_run_cmd_str(target_api: str, primary_class: str = "A", fallback_class: str = "S") -> str:
    """Generate run command string for correctness testing.
    
    Uses primary_class (default A) with fallback_class (default S) on timeout/segfault.
    Uses 'env' prefix so it works with 'timeout' command.
    """
    env_prefix = "env OMP_TARGET_OFFLOAD=MANDATORY " if target_api == 'omp' else ""
    
    workdir = get_codex_workdir()
    if _is_nas_workdir(workdir):
        config = get_config()
        return f"{env_prefix}make CC={config.nas_cc} CLASS={primary_class} run"
    
    makefile_name = "Makefile.nvc"
    if target_api == 'hip':
        makefile_name = "Makefile.hip"
    elif target_api == 'sycl':
        makefile_name = "Makefile.sycl"

    return f"{env_prefix}make -f {makefile_name} run"


def get_correctness_fallback_cmd_str(target_api: str, fallback_class: str = "S") -> str:
    """Generate fallback run command string for correctness testing when primary times out.
    Uses 'env' prefix so it works with 'timeout' command."""
    env_prefix = "env OMP_TARGET_OFFLOAD=MANDATORY " if target_api == 'omp' else ""
    
    workdir = get_codex_workdir()
    if _is_nas_workdir(workdir):
        config = get_config()
        return f"{env_prefix}make CC={config.nas_cc} CLASS={fallback_class} run"
    
    makefile_name = "Makefile.nvc"
    if target_api == 'hip':
        makefile_name = "Makefile.hip"
    elif target_api == 'sycl':
        makefile_name = "Makefile.sycl"

    return f"{env_prefix}make -f {makefile_name} run"


def get_profile_run_cmd_str(target_api: str, kernel_name: Optional[str] = None, primary_class: str = "B", fallback_class: str = "B") -> str:
    """Generate run command string for profiling (larger workload).
    
    Uses primary_class (default C) with fallback_class (default B) on timeout/segfault.
    Uses 'env' prefix so it works with 'timeout' command.
    """
    env_prefix = "env OMP_TARGET_OFFLOAD=MANDATORY " if target_api == 'omp' else ""
    
    workdir = get_codex_workdir()
    if _is_nas_workdir(workdir):
        config = get_config()
        # bt kernels are especially heavy, use smaller class
        if kernel_name and kernel_name.lower().startswith("bt"):
            return f"{env_prefix}make CC={config.nas_cc} CLASS=A run"
        return f"{env_prefix}make CC={config.nas_cc} CLASS={primary_class} run"
    
    makefile_name = "Makefile.nvc"
    if target_api == 'hip':
        makefile_name = "Makefile.hip"
    elif target_api == 'sycl':
        makefile_name = "Makefile.sycl"

    return f"{env_prefix}make -f {makefile_name} run"


def get_profile_fallback_cmd_str(target_api: str, kernel_name: Optional[str] = None, fallback_class: str = "B") -> str:
    """Generate fallback run command string for profiling when primary times out.
    Uses 'env' prefix so it works with 'timeout' command."""
    env_prefix = "env OMP_TARGET_OFFLOAD=MANDATORY " if target_api == 'omp' else ""
    
    workdir = get_codex_workdir()
    if _is_nas_workdir(workdir):
        config = get_config()
        return f"{env_prefix}make CC={config.nas_cc} CLASS={fallback_class} run"
    
    makefile_name = "Makefile.nvc"
    if target_api == 'hip':
        makefile_name = "Makefile.hip"
    elif target_api == 'sycl':
        makefile_name = "Makefile.sycl"

    return f"{env_prefix}make -f {makefile_name} run"


def get_nsys_profile_cmd_str(target_api: str, kernel_name: Optional[str] = None, use_class: str = "C") -> str:
    """Generate nsys profile command string with specified CLASS."""
    env_prefix = "env FORCE_OMP_GPU=1 OMP_TARGET_OFFLOAD=MANDATORY " if target_api == 'omp' else ""
    
    workdir = get_codex_workdir()
    if _is_nas_workdir(workdir):
        config = get_config()
        if kernel_name and kernel_name.lower().startswith("bt"):
            use_class = "A"  # bt is heavy
        return (
            f"{env_prefix}nsys profile --stats=true --trace=cuda,osrt "
            f"--force-overwrite=true -o nsys_profile make CC={config.nas_cc} CLASS={use_class} run"
        )
    
    makefile_name = "Makefile.nvc"
    if target_api == 'hip':
        makefile_name = "Makefile.hip"
    elif target_api == 'sycl':
        makefile_name = "Makefile.sycl"

    return (
        f"{env_prefix}nsys profile --stats=true --trace=cuda,osrt "
        f"--force-overwrite=true -o nsys_profile make -f {makefile_name} run"
    )


def get_nsys_profile_fallback_cmd_str(target_api: str, kernel_name: Optional[str] = None, fallback_class: str = "B") -> str:
    """Generate nsys profile fallback command string when primary CLASS times out."""
    env_prefix = "env FORCE_OMP_GPU=1 OMP_TARGET_OFFLOAD=MANDATORY " if target_api == 'omp' else ""
    
    workdir = get_codex_workdir()
    if _is_nas_workdir(workdir):
        config = get_config()
        return (
            f"{env_prefix}nsys profile --stats=true --trace=cuda,osrt "
            f"--force-overwrite=true -o nsys_profile make CC={config.nas_cc} CLASS={fallback_class} run"
        )
    
    makefile_name = "Makefile.nvc"
    if target_api == 'hip':
        makefile_name = "Makefile.hip"
    elif target_api == 'sycl':
        makefile_name = "Makefile.sycl"

    return (
        f"{env_prefix}nsys profile --stats=true --trace=cuda,osrt "
        f"--force-overwrite=true -o nsys_profile make -f {makefile_name} run"
    )


# =============================================================================
# GPU Protection Utilities
# =============================================================================

def get_gpu_processes() -> List[int]:
    """Get list of PIDs using the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            pids = []
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line:
                    try:
                        pids.append(int(line))
                    except ValueError:
                        pass
            return pids
    except Exception:
        pass
    return []


def kill_gpu_processes(exclude_pids: Optional[List[int]] = None) -> int:
    """Kill all GPU processes, optionally excluding some PIDs.
    
    Returns the number of processes killed.
    """
    exclude_pids = exclude_pids or []
    killed = 0
    gpu_pids = get_gpu_processes()
    
    for pid in gpu_pids:
        if pid in exclude_pids:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
            logger.info(f"Killed GPU process {pid}")
        except ProcessLookupError:
            pass  # Already dead
        except PermissionError:
            logger.warning(f"Cannot kill GPU process {pid} (permission denied)")
    
    return killed


def run_with_gpu_timeout(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout_seconds: int = 300,
    kill_gpu_on_timeout: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str, str]:
    """Run a command with timeout and GPU cleanup on failure.
    
    Args:
        cmd: Command to run
        cwd: Working directory
        timeout_seconds: Maximum execution time (default 5 minutes)
        kill_gpu_on_timeout: Kill GPU processes if command times out
        env: Optional environment variables dict (defaults to os.environ.copy())
    
    Returns:
        (success, stdout, stderr)
    """
    # Record GPU processes before running
    pre_gpu_pids = get_gpu_processes() if kill_gpu_on_timeout else []
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
        return result.returncode == 0, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired as e:
        logger.warning(f"Command timed out after {timeout_seconds}s: {' '.join(cmd)}")
        
        if kill_gpu_on_timeout:
            # Wait a moment for cleanup
            time.sleep(1)
            # Kill any NEW GPU processes (that weren't running before)
            post_gpu_pids = get_gpu_processes()
            new_pids = [p for p in post_gpu_pids if p not in pre_gpu_pids]
            if new_pids:
                logger.warning(f"Killing hung GPU processes: {new_pids}")
                for pid in new_pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
            # If still have processes, kill all
            time.sleep(1)
            remaining = get_gpu_processes()
            if remaining:
                logger.warning(f"Force killing remaining GPU processes: {remaining}")
                kill_gpu_processes(exclude_pids=pre_gpu_pids)
        
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
        return False, stdout, stderr + f"\n[TIMEOUT after {timeout_seconds}s]"
        
    except Exception as e:
        return False, "", str(e)


def safe_run_gpu_command(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout_seconds: int = 300,
) -> Tuple[bool, str]:
    """Convenience wrapper for running GPU commands safely.
    
    Returns:
        (success, combined_output)
    """
    success, stdout, stderr = run_with_gpu_timeout(
        cmd, cwd=cwd, timeout_seconds=timeout_seconds, kill_gpu_on_timeout=True
    )
    combined = stdout + ("\n" + stderr if stderr else "")
    return success, combined.strip()


# Default timeout for GPU operations
def get_gpu_timeout_seconds() -> int:
    """Get GPU timeout in seconds from config."""
    return get_config().gpu_timeout_seconds


# For backward compatibility
GPU_TIMEOUT_SECONDS = get_gpu_timeout_seconds()


# Export NAS_IDENTIFIER for backward compatibility
def get_nas_identifier() -> str:
    """Get NAS identifier from config."""
    return get_config().nas_identifier


NAS_IDENTIFIER = get_nas_identifier()

