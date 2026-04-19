"""
Common utilities and shared functions for agent modules.

Consolidates duplicate code and provides reusable functionality.
"""

import json
import shutil
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add pipeline_refactored directory to path to allow imports when running as script
_script_dir = Path(__file__).parent
_pipeline_refactored_dir = _script_dir.parent
if str(_pipeline_refactored_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_refactored_dir))

from utils.logger import get_logger
from utils.config import get_config
from utils.path_config import (
    get_codex_workdir,
    get_make_cmd,
    get_make_cmd_str,
    default_data_src,
    default_golden_root,
)
from utils.prompt_loader import build_skill_trigger_prompt
from utils.run_opencode import run_opencode
from utils.run_codex_ts import run_codex_ts

logger = get_logger(__name__)
config = get_config()


def parse_jsonl(file_path: Path, source_api: str) -> List[Dict[str, Any]]:
    """Parse the JSONL file and extract source codes.
    
    Args:
        file_path: Path to JSONL file
        source_api: Source API to filter by (e.g., 'serial', 'cuda')
        
    Returns:
        List of dictionaries containing kernel information
    """
    source_codes = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if data.get('parallel_api') == source_api:
                        source_codes.append({
                            'line_num': line_num,
                            'kernel_name': data.get('kernel_name', 'unknown'),
                            'code': data.get('code', {}),
                            'original_data': data
                        })
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"JSONL file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading JSONL file {file_path}: {e}")
    
    return source_codes


def test_compilation(kernel_dir: Path, target_api: str) -> Tuple[bool, str]:
    """Test compilation by running make clean and make.
    
    Args:
        kernel_dir: Directory containing the kernel
        target_api: Target API (omp, cuda, ocl)
        
    Returns:
        Tuple of (success, error_message)
    """
    kernel_dir = Path(kernel_dir)
    clean_cmd = get_make_cmd(target_api, 'clean')
    build_cmd = get_make_cmd(target_api, 'build')

    try:
        subprocess.run(
            clean_cmd, 
            capture_output=True, 
            text=True, 
            timeout=30, 
            cwd=kernel_dir
        )
        make_result = subprocess.run(
            build_cmd, 
            capture_output=True, 
            text=True, 
            timeout=120, 
            cwd=kernel_dir
        )

        success = make_result.returncode == 0
        if not success:
            logger.warning("Make compilation failed")
        return success, "" if success else (make_result.stderr or make_result.stdout or "Unknown compilation error")

    except subprocess.TimeoutExpired:
        logger.warning("Make compilation timeout")
        return False, "Make compilation timeout"
    except Exception as e:
        logger.error(f"Error during make compilation: {e}")
        return False, str(e)


def run_codex_command(prompt: str, timeout: int = 600000, max_tokens: Optional[int] = None, allowed_tools: Optional[List[str]] = None, model: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Run the coding agent (opencode) non-interactively with the given prompt.

    Args:
        prompt: Prompt text to send to the agent
        timeout: Timeout in seconds (default: 6000)
        max_tokens: Unused — kept for API compatibility (opencode manages context internally)
        allowed_tools: Unused — opencode uses the 'build' agent which enables all dev tools
        model: Model in opencode 'provider/model' format (e.g., 'anthropic/claude-sonnet-4-5').
               Falls back to OPENCODE_MODEL env var, then CODEX_MODEL env var.

    Returns:
        Dict with 'combined' and 'summary' keys on success, None on failure
    """
    workdir = get_codex_workdir()
    
    # Choose engine: PIPELINE_ENGINE env var (set by setup_pipeline_workdir.py) or default
    engine = os.environ.get("PIPELINE_ENGINE", "codex").lower()

    traces_dir = _pipeline_refactored_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    runner_kwargs = dict(
        prompt=prompt,
        workdir=str(workdir),
        model=model,
        timeout=timeout,
        allowed_tools=allowed_tools,
        traces_dir=str(traces_dir),
    )

    try:
        if engine == "codex":
            logger.info("Engine: OpenAI Codex (TypeScript SDK)")
            return run_codex_ts(**runner_kwargs)
        else:
            logger.info("Engine: opencode")
            return run_opencode(**runner_kwargs)
    except RuntimeError as e:
        logger.error(str(e))
        return None
    except Exception as e:
        logger.error(f"Error running {engine}: {e}")
        return None


def get_default_input_jsonl() -> Path:
    """Get default input JSONL file path.
    
    Returns:
        Path to default JSONL file
    """
    config = get_config()
    # Try data directory first, then fall back to pipeline directory
    data_dir = config.pipeline_root / "data"
    if data_dir.exists():
        # Look for any combined_*_filenames.jsonl file
        for jsonl_file in data_dir.glob("combined_*_filenames.jsonl"):
            return jsonl_file
    
    # Fallback to config default
    return config.default_jsonl()


def get_default_output_dir() -> Path:
    """Get default output directory.
    
    Returns:
        Path to default output directory
    """
    config = get_config()
    return config.codex_workdir / "results"


def get_clean_kernel_script_path() -> Path:
    """Get path to clean kernel directories script.
    
    Returns:
        Path to clean_kernel_dirs.py script
    """
    config = get_config()
    # Try utils directory in base_dir first
    utils_script = config.base_dir / "utils" / "clean_kernel_dirs.py"
    if utils_script.exists():
        return utils_script
    
    # Try pipeline_refactored/utils
    refactored_utils = config.pipeline_root / "utils" / "clean_kernel_dirs.py"
    if refactored_utils.exists():
        return refactored_utils
    
    # Return the most likely path even if it doesn't exist
    return utils_script


def normalize_file_list(file_name: str | List[str]) -> List[str]:
    """Normalize file name(s) to a list.
    
    Args:
        file_name: Single file name (str) or list of file names
        
    Returns:
        List of file names
    """
    if isinstance(file_name, str):
        return [file_name]
    return file_name


def format_file_list(file_names: List[str]) -> str:
    """Format list of file names for display.
    
    Args:
        file_names: List of file names
        
    Returns:
        Formatted string
    """
    if len(file_names) == 1:
        return file_names[0]
    return ', '.join(file_names)


def resolve_kernel_file_name(file_name: str, target_api: str) -> str:
    """Normalize the kernel filename for the desired target API extension.
    
    Args:
        file_name: Original file name
        target_api: Target API (omp, cuda, ocl)
        
    Returns:
        Normalized file name with appropriate extension
    """
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

    if target_api == 'ocl':
        # For OpenCL, keep .cl files as is, but .cu host files might become .c or .cpp
        if suffix == '.cl':
            return str(path)
        # Don't force conversion - let the translation decide
        return str(path)

    return str(path)


def copy_translated_file(kernel_name, file_name, target_api, data_src_dir, output_dir, suffix):
    """
    Copy the entire translated directory to the results directory.
    Files are organized in subdirectories by phase/step (e.g., initial/, step1/, step2_supervised/).
    
    Args:
        kernel_name: Name of the kernel
        file_name: Original file name (str) or list of file names (List[str]) (used only to return a mapped file path)
        target_api: Target API (omp, cuda, etc.)
        data_src_dir: Directory containing translated code (data/src)
        output_dir: Output directory for results
        suffix: Suffix/phase name ('initial', 'step1', 'step2_supervised', 'optimized', etc.)
    
    Returns:
        str or List[str]: Path(s) to the primary source file(s) in the snapshot, or None if failed.
    """
    file_names = normalize_file_list(file_name)
    return_single = isinstance(file_name, str)
    
    # Determine source directory
    kernel_dir = Path(data_src_dir) / f"{kernel_name}-{target_api}"
    
    # Create destination paths
    output_dir = Path(output_dir)
    kernel_output_dir = output_dir / f"{kernel_name}-{target_api}"
    phase_dir = kernel_output_dir / suffix
    
    try:
        if kernel_dir.exists():
            # Copy everything, ignoring .o files and executable binaries (e.g., "main")
            # Note: ignore_patterns behaves like glob on filename matched against list
            shutil.copytree(kernel_dir, phase_dir, dirs_exist_ok=True, 
                            ignore=shutil.ignore_patterns("*.o", "main", ".*", "*.bak"))
            logger.success(f"Snapshotted {suffix} directory: {kernel_dir} -> {phase_dir}")
        else:
            logger.warning(f"Source directory not found for snapshotting: {kernel_dir}")
            return None
    except Exception as e:
        logger.failure(f"Error copying {suffix} directory {kernel_dir}: {e}")
        return None

    # Resolve paths to the requested main files inside the snapshot directory
    copied_paths = []
    if target_api == 'ocl':
        # For OpenCL, return all .cl files as we don't know the exact names
        for cl_file in phase_dir.glob("*.cl"):
            copied_paths.append(str(cl_file))
    else:
        for fn in file_names:
            source_file_name = resolve_kernel_file_name(fn, target_api)
            dest_file_path = phase_dir / source_file_name
            
            if not dest_file_path.exists():
                base_name = Path(source_file_name).stem
                if target_api == 'omp':
                    for ext in ['.c', '.cpp']:
                        if (phase_dir / f"{base_name}{ext}").exists():
                            dest_file_path = phase_dir / f"{base_name}{ext}"
                            break
                elif target_api == 'cuda':
                    for ext in ['.c', '.cpp', '.cu']:
                        if (phase_dir / f"{base_name}{ext}").exists():
                            dest_file_path = phase_dir / f"{base_name}{ext}"
                            break
            
            if dest_file_path.exists():
                copied_paths.append(str(dest_file_path))

    # Also copy all .md files to the root kernel output directory
    for md_file in kernel_dir.glob("*.md"):
        try:
            dest_md = kernel_output_dir / md_file.name
            shutil.copy2(md_file, dest_md)
        except Exception as e:
            pass

    if not copied_paths:
        return None
    
    if return_single:
        return copied_paths[0]
    return copied_paths


def launch_makefile_fix_recovery(
    kernel_dir: Path,
    target_api: str,
    compilation_error: str,
    stage: str = "unknown",
    max_attempts: int = 2,
) -> Tuple[bool, Optional[str]]:
    """Launch a Codex recovery session to fix Makefile and dependencies after compilation failure.
    
    Args:
        kernel_dir: Directory containing the kernel code and Makefile
        target_api: Target API (omp, cuda, ocl)
        compilation_error: The compilation error message/output
        stage: Stage where failure occurred (e.g., "initial", "step1", "step2")
        max_attempts: Maximum number of recovery attempts (default: 2)
        
    Returns:
        Tuple of (success, error_message)
        - success: True if compilation succeeds after recovery, False otherwise
        - error_message: Error message if recovery failed, None if successful
    """
    kernel_dir = Path(kernel_dir)
    workdir = get_codex_workdir()
    
    # Save compilation error to a log file
    error_log_path = kernel_dir / f"compilation_error_{stage}.log"
    try:
        with open(error_log_path, 'w') as f:
            f.write(f"Compilation failure at stage: {stage}\n")
            f.write("="*70 + "\n")
            f.write(compilation_error)
    except Exception as e:
        logger.warning(f"Failed to save compilation error log: {e}")
        error_log_path = None
    
    # Build the recovery prompt
    skill_name = "paracodex-makefile-fix"
    clean_cmd_str = get_make_cmd_str(target_api, "clean")
    build_cmd_str = get_make_cmd_str(target_api, "build")
    
    variables = {
        "kernel_dir": str(kernel_dir),
        "target_api": target_api,
        "compilation_error_log": str(error_log_path) if error_log_path else "N/A",
        "clean_cmd": clean_cmd_str,
        "build_cmd": build_cmd_str,
    }
    
    prompt = build_skill_trigger_prompt(
        skill_name=skill_name,
        task=f"Fix Makefile and dependencies for compilation failure at {stage} stage in {kernel_dir}",
        workdir=workdir,
        source_dir=None,
        target_dir=kernel_dir,
        file_listing="Makefile.nvc and source files",
        variables=variables,
        notes=[
            f"Use CODEX_WORKDIR={workdir}.",
            "For shell commands: Prefer redirecting large output to a temporary file, then read that file using the `read_file` tool (BEST) or `cat` WITHOUT redirection. For short output, run the command regularly.",
            "  - Step 1: Run commands with output redirection: `<command> > /tmp/command_output.txt 2>&1`",
            "  - Step 2: Read the temp file using `read_file` tool to read `/tmp/command_output.txt` directly, OR use `cat /tmp/command_output.txt` (WITHOUT `> /tmp/...` redirection).",
            "  - For reading existing files: Use the `read_file` tool directly (e.g., read `system_info_summary.txt` or skill files) - do NOT copy them to temp files first.",
            "  - CRITICAL - DO NOT: Copy temp files to other temp files (e.g., `cat /tmp/file1.txt > /tmp/file2.txt` is wasteful - just read `/tmp/file1.txt` directly using `read_file` tool).",
            "Do not run git commands.",
            f"Compilation failed at stage: {stage}",
            f"Error log available at: {error_log_path}" if error_log_path else "Error details in prompt",
        ],
    )
    
    logger.info(f"Launching Makefile fix recovery for {kernel_dir} (stage: {stage})...")
    
    for attempt in range(1, max_attempts + 1):
        logger.info(f"  Recovery attempt {attempt}/{max_attempts}...")
        
        # Run the recovery Codex session
        result = run_codex_command(prompt, timeout=3000)
        
        if result is None:
            logger.warning(f"  Recovery attempt {attempt} failed: Codex CLI returned no result")
            if attempt < max_attempts:
                continue
            else:
                return False, "Recovery failed: Codex CLI returned no result"
        
        # Test if compilation now succeeds
        success, error_msg = test_compilation(kernel_dir, target_api)
        
        if success:
            logger.success(f"  ✓ Recovery attempt {attempt} succeeded - compilation now passes")
            return True, None
        else:
            logger.warning(f"  ✗ Recovery attempt {attempt} failed - compilation still fails")
            # Update the error log with the new error
            if error_log_path:
                try:
                    with open(error_log_path, 'a') as f:
                        f.write(f"\n\n--- Recovery attempt {attempt} error ---\n")
                        f.write(error_msg)
                except Exception:
                    pass
            # Update prompt with new error for next attempt
            if attempt < max_attempts:
                variables["compilation_error_log"] = str(error_log_path) if error_log_path else error_msg
                prompt = build_skill_trigger_prompt(
                    skill_name=skill_name,
                    task=f"Fix Makefile and dependencies for compilation failure at {stage} stage in {kernel_dir} (attempt {attempt + 1})",
                    workdir=workdir,
                    source_dir=None,
                    target_dir=kernel_dir,
                    file_listing="Makefile.nvc and source files",
                    variables=variables,
                    notes=[
                        f"Use CODEX_WORKDIR={workdir}.",
                        "For shell commands: Prefer redirecting large output to a temporary file, then read that file using the `read_file` tool (BEST) or `cat` WITHOUT redirection.",
                        "  - Step 1: Run commands with output redirection: `<command> > /tmp/command_output.txt 2>&1`",
                        "  - Step 2: Read the temp file using `read_file` tool to read `/tmp/command_output.txt` directly, OR use `cat /tmp/command_output.txt` (WITHOUT `> /tmp/...` redirection).",
                        "  - For reading existing files: Use the `read_file` tool directly (e.g., read `system_info_summary.txt` or skill files) - do NOT copy them to temp files first.",
                        "  - CRITICAL - DO NOT: Copy temp files to other temp files (e.g., `cat /tmp/file1.txt > /tmp/file2.txt` is wasteful - just read `/tmp/file1.txt` directly using `read_file` tool).",
                        "Do not run git commands.",
                        f"Compilation failed at stage: {stage}",
                        f"Previous recovery attempt {attempt} still failed",
                        f"Latest error: {error_msg[:500]}",
                    ],
                )
    
    return False, f"Recovery failed after {max_attempts} attempts. Last error: {error_msg}"


def run_compare_and_optimize_steps(kernel_name: str, file_name: str, output_dir: str, target_api: str) -> bool:
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
        script_path = config.pipeline_root / "scripts" / "compare_and_optimize_steps.py"
        if not script_path.exists():
            logger.info(f"[WARN] compare_and_optimize_steps.py not found at {script_path}")
            return False
        
        # The benchmark name in output_dir is "{kernel_name}-{target_api}"
        benchmark_name = f"{kernel_name}-{target_api}"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--base_dir", str(output_dir),
            "--benchmark", benchmark_name,
        ]
        
        logger.info(f"Running compare_and_optimize_steps for {benchmark_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            logger.success(f"compare_and_optimize_steps completed successfully for {benchmark_name}")
            if result.stdout:
                logger.debug(result.stdout)
            return True
        else:
            logger.warning(f"compare_and_optimize_steps returned non-zero exit code for {benchmark_name}")
            if result.stderr:
                logger.debug(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"compare_and_optimize_steps timed out for {benchmark_name}")
        return False
    except Exception as e:
        logger.warning(f"Error running compare_and_optimize_steps for {benchmark_name}: {e}")
        return False
